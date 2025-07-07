import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools 


# Singleagent Q network
class AgentNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q_out = nn.Linear(hidden_dim, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        fc1_out = F.relu(self.fc1(x))
        fc2_out = F.relu(self.fc2(fc1_out))
        return self.q_out(fc2_out)


# QMIX mixing network
class MixingNetwork(nn.Module):
    def __init__(self, n_agents: int, state_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.hyper_w1 = nn.Linear(state_dim, n_agents * hidden_dim)
        self.hyper_b1 = nn.Linear(state_dim, hidden_dim)
        self.hyper_w2 = nn.Linear(state_dim, hidden_dim)
        self.hyper_b2 = nn.Linear(state_dim, 1)
        self.elu = nn.ELU()

    def forward(self, agent_qs: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        B = agent_qs.size(0)
        w1 = torch.abs(self.hyper_w1(states)).view(B, self.n_agents, self.hidden_dim)
        b1 = self.hyper_b1(states)
        hidden = torch.bmm(agent_qs.unsqueeze(1), w1).squeeze(1) + b1
        hidden = self.elu(hidden)
        w2 = torch.abs(self.hyper_w2(states)).view(B, self.hidden_dim, 1)
        b2 = self.hyper_b2(states)
        q_tot = torch.bmm(hidden.unsqueeze(1), w2).squeeze(1) + b2
        return q_tot.squeeze(1)  # (B,)


# QMIX wrapper 
class QMIXAgent:
   
    def __init__(self, action_space, n_agents: int, state_dim: int):
        self.n_agents = n_agents
        self.n_actions = int(action_space.nvec[0])
        self.state_dim = state_dim

        # Per Agent networksszx
        self.agent_nets = nn.ModuleList(
            [AgentNetwork(state_dim, self.n_actions) for _ in range(n_agents)]
        )
        self.target_agent_nets = nn.ModuleList(
            [AgentNetwork(state_dim, self.n_actions) for _ in range(n_agents)]
        )

        # Mixing networks
        self.mixing_net = MixingNetwork(n_agents, state_dim)
        self.target_mixing = MixingNetwork(n_agents, state_dim)

        # HArdware to use
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for net in (
            *self.agent_nets,
            *self.target_agent_nets,
            self.mixing_net,
            self.target_mixing,
        ):
            net.to(self.device)

        # Optimizer
        params = list(self.mixing_net.parameters())
        for net in self.agent_nets:
            params.extend(net.parameters())
        self.optim = torch.optim.RMSprop(params, lr=1e-4)

        # Replay
        self.buffer = deque(maxlen=16000)
        self.batch_size = 64
        self.gamma = 0.99

        self.update_target()

    def update_target(self):
        for tgt, src in zip(self.target_agent_nets, self.agent_nets):
            tgt.load_state_dict(src.state_dict())
        self.target_mixing.load_state_dict(self.mixing_net.state_dict())

    def select_action(self, obs: np.ndarray, epsilon: float = 0.05) -> list[int]:
      
        if random.random() < epsilon:
            return [random.randrange(self.n_actions) for _ in range(self.n_agents)]

        obs_t = torch.FloatTensor(obs).to(self.device)
        actions = []
        with torch.no_grad():
            for net in self.agent_nets:
                q_vals = net(obs_t)
                actions.append(int(q_vals.argmax().item()))
        return actions

    # Replay helpers
    def store_transition(self, o, a, r, o2, d):
        self.buffer.append((o, a, r, o2, d))

    # Training step function
    # def train_step(self):
    #     if len(self.buffer) < self.batch_size:
    #         return None

    #     batch = random.sample(self.buffer, self.batch_size)
    #     obs_np = np.stack([b[0] for b in batch]).astype(np.float32)   
    #     acts_np = np.stack([b[1] for b in batch]).astype(np.int64)   
    #     r_np = np.array([b[2] for b in batch], dtype=np.float32)      
    #     obs2_np = np.stack([b[3] for b in batch]).astype(np.float32)  
    #     done_np = np.array([b[4] for b in batch], dtype=np.float32)   
    #     obs_b = torch.from_numpy(obs_np).to(self.device)
    #     obs2_b = torch.from_numpy(obs2_np).to(self.device)
    #     acts_b = torch.from_numpy(acts_np).to(self.device)
    #     r_b = torch.from_numpy(r_np).to(self.device)
    #     done_b = torch.from_numpy(done_np).to(self.device)

    #     # Each agent Qs
    #     qs, qs_next = [], []
    #     for i, net in enumerate(self.agent_nets):
    #         q = net(obs_b).gather(1, acts_b[:, i].unsqueeze(1)).squeeze(1)
    #         q_next = self.target_agent_nets[i](obs2_b).max(dim=1)[0]
    #         qs.append(q)
    #         qs_next.append(q_next)
    #     q_stack = torch.stack(qs, dim=1)       
    #     q_next_stack = torch.stack(qs_next, dim=1)

    #     # Mixing log
    #     q_tot = self.mixing_net(q_stack, obs_b)
    #     with torch.no_grad():
    #         target_q_tot = self.target_mixing(q_next_stack, obs2_b)
    #         y = r_b + (1 - done_b) * self.gamma * target_q_tot


    #     # print(f"q_tot: {q_tot}, y: {y}, r_b: {r_b}")
    #     loss = F.mse_loss(q_tot, y)

    #     self.optim.zero_grad()
    #     loss.backward()
    #     self.optim.step()

    #     return loss.item()
    
    # Training step function
    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return None

        batch = random.sample(self.buffer, self.batch_size)
        obs = torch.tensor(np.stack([b[0] for b in batch]),dtype=torch.float32, device=self.device)    
        acts = torch.tensor(np.stack([b[1] for b in batch]),dtype=torch.int64,  device=self.device)      
        rew = torch.tensor([b[2] for b in batch],dtype=torch.float32, device=self.device)     
        obs2 = torch.tensor(np.stack([b[3] for b in batch]),dtype=torch.float32, device=self.device)
        done = torch.tensor([b[4] for b in batch],dtype=torch.float32, device=self.device)     

        rew *= 10.0                             
        eps = 1e-5
        state = (obs - obs.mean(dim=0, keepdim=True)) / (obs .std(dim=0, keepdim=True) + eps)
        state2 = (obs2 - obs2.mean(dim=0, keepdim=True)) / (obs2.std(dim=0, keepdim=True) + eps)

        qs, qs_next = [], []
        for i, net in enumerate(self.agent_nets):
            q_all = net(obs)                       
            act_i = acts[:, i].unsqueeze(1)        
            qs.append(q_all.gather(1, act_i).squeeze(1)) 

            with torch.no_grad():
                greedy_next = q_all.argmax(1, keepdim=True)
                tgt_all = self.target_agent_nets[i](obs2)
                qs_next.append(tgt_all.gather(1, greedy_next).squeeze(1))

        q_tot = self.mixing_net(torch.stack(qs,1), state)
        with torch.no_grad():
            tgt_tot = self.target_mixing(torch.stack(qs_next, 1), state2)
            y = rew + (1 - done) * self.gamma * tgt_tot          

        loss = F.mse_loss(q_tot, y)
        self.optim.zero_grad(set_to_none=True)
        loss.backward()

        # clip all parameters of agent nets + mixer to a max-norm of 10
        torch.nn.utils.clip_grad_norm_(itertools.chain(*(net.parameters() for net in self.agent_nets),self.mixing_net.parameters()),10.0)
        self.optim.step()
        return loss.item()
    

    # To keep persistence (Look at logic)
    def save(self, path: str):
        torch.save(
            {
                "agents": [net.state_dict() for net in self.agent_nets],
                "mixing": self.mixing_net.state_dict(),
            },
            path,
        )

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        for net, sd in zip(self.agent_nets, ckpt["agents"]):
            net.load_state_dict(sd)
        self.mixing_net.load_state_dict(ckpt["mixing"])
        self.update_target()
