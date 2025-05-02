import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class AgentNetwork(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class MixingNetwork(nn.Module):
    def __init__(self, n_agents, state_dim, hidden_dim=32):
        super().__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

        # hypernets for first layer
        self.hyper_w1 = nn.Linear(state_dim, n_agents * hidden_dim)
        self.hyper_b1 = nn.Linear(state_dim, hidden_dim)

        # hypernets for second layer
        self.hyper_w2 = nn.Linear(state_dim, hidden_dim)
        self.hyper_b2 = nn.Linear(state_dim, 1)

        self.elu = nn.ELU()

    def forward(self, agent_qs, states):
        """
        agent_qs: (batch, n_agents)
        states:   (batch, state_dim)
        returns:  (batch,)
        """
        bs = agent_qs.size(0)

        # first layer
        w1 = torch.abs(self.hyper_w1(states))                 # (bs, n_agents*hidden_dim)
        b1 = self.hyper_b1(states)                            # (bs, hidden_dim)
        w1 = w1.view(bs, self.n_agents, self.hidden_dim)      # (bs, n_agents, hidden_dim)
        hidden = torch.bmm(agent_qs.unsqueeze(1), w1).squeeze(1) + b1
        hidden = self.elu(hidden)

        # second layer
        w2 = torch.abs(self.hyper_w2(states)).view(bs, self.hidden_dim, 1)
        b2 = self.hyper_b2(states).view(bs, 1)
        q_tot = torch.bmm(hidden.unsqueeze(1), w2).squeeze(1) + b2

        return q_tot.squeeze(1)


class QMIXAgent:
    def __init__(self, action_space, n_agents, state_dim):
        self.n_agents   = n_agents
        self.n_actions  = action_space.nvec[0]
        self.state_dim  = state_dim
        self.obs_dim    = state_dim

        # networks
        self.agent_net       = AgentNetwork(self.obs_dim, self.n_actions)
        self.target_agent    = AgentNetwork(self.obs_dim, self.n_actions)
        self.mixing_net      = MixingNetwork(n_agents, state_dim)
        self.target_mixing   = MixingNetwork(n_agents, state_dim)

        # device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for net in (self.agent_net, self.target_agent, self.mixing_net, self.target_mixing):
            net.to(self.device)

        # optimizer
        params = list(self.agent_net.parameters()) + list(self.mixing_net.parameters())
        self.optim = torch.optim.RMSprop(params, lr=5e-4)

        # replay
        self.buffer     = deque(maxlen=5000)
        self.batch_size = 32
        self.gamma      = 0.99

        self.update_target()

    def update_target(self):
        self.target_agent.load_state_dict(self.agent_net.state_dict())
        self.target_mixing.load_state_dict(self.mixing_net.state_dict())

    def select_action(self, obs, epsilon=0.05):
        # epsilon-greedy
        if random.random() < epsilon:
            return [random.randrange(self.n_actions) for _ in range(self.n_agents)]

        obs_t = torch.FloatTensor(obs).to(self.device)
        # duplicate per agent
        qs = []
        for _ in range(self.n_agents):
            qs.append(self.agent_net(obs_t).unsqueeze(0))
        qs = torch.cat(qs, dim=0)          # (n_agents, n_actions)
        acts = qs.argmax(dim=1).cpu().tolist()
        return acts

    def store_transition(self, o, a, r, o2, d):
        self.buffer.append((o, a, r, o2, d))

    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return None

        batch = random.sample(self.buffer, self.batch_size)

        # ----- Efficient stacking via NumPy avoids the warning -----
        import numpy as np
        obs_np   = np.stack([b[0] for b in batch], axis=0).astype(np.float32)
        acts_np  = np.stack([b[1] for b in batch], axis=0).astype(np.int64)
        r_np     = np.array([b[2] for b in batch], dtype=np.float32)
        obs2_np  = np.stack([b[3] for b in batch], axis=0).astype(np.float32)
        done_np  = np.array([b[4] for b in batch], dtype=np.float32)

        obs_b  = torch.from_numpy(obs_np).to(self.device)
        acts_b = torch.from_numpy(acts_np).to(self.device)
        r_b    = torch.from_numpy(r_np).to(self.device)
        obs2_b = torch.from_numpy(obs2_np).to(self.device)
        done_b = torch.from_numpy(done_np).to(self.device)
        # ------------------------------------------------------------

        # compute each agent’s Q and target
        qs, qs_next = [], []
        for i in range(self.n_agents):
            q      = self.agent_net(obs_b).gather(1, acts_b[:, i].unsqueeze(1)).squeeze(1)
            q_next = self.target_agent(obs2_b).max(dim=1)[0]
            qs.append(q)
            qs_next.append(q_next)

        q_stack      = torch.stack(qs, dim=1)       # (B, n_agents)
        q_next_stack = torch.stack(qs_next, dim=1)  # (B, n_agents)

        # mixing
        q_tot      = self.mixing_net(q_stack, obs_b)
        with torch.no_grad():
            q_tot_next = self.target_mixing(q_next_stack, obs2_b)
            target     = r_b + (1 - done_b) * self.gamma * q_tot_next

        loss = F.mse_loss(q_tot, target)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        return loss.item()

    def save(self, path):
        torch.save({
            'agent':  self.agent_net.state_dict(),
            'mixing': self.mixing_net.state_dict()
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.agent_net.load_state_dict(ckpt['agent'])
        self.mixing_net.load_state_dict(ckpt['mixing'])
        self.update_target()