import os
import re
import logging
import inspect
from typing import Optional, Dict, Any, Tuple, List
import numpy as np
import random, torch  
from environment.env import InsideBoxPushingEnv

# Setting up the logger
LOG_FILE = "training.log"
LOGGER = logging.getLogger("inside_box")
LOGGER.setLevel(logging.INFO)
fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
fh.setFormatter(fmt)
LOGGER.addHandler(fh)
# Setting up the console
ch = logging.StreamHandler()
ch.setFormatter(fmt)
# Fomating the logger
LOGGER.addHandler(ch)

# Determine run number by scanning existing log
_run_id = 1
if os.path.exists(LOG_FILE):
    with open(LOG_FILE, "r", encoding="utf-8", errors="ignore") as _f:
        matches = re.findall(r"=== Run (\d+) ===", _f.read())
        if matches:
            _run_id = max(int(m) for m in matches) + 1

LOGGER.info("")
LOGGER.info(f"=== Run {_run_id} ===")
LOGGER.info("")

def run_episode(agent, env, train=False, render=False):
    obs = env.reset()
    done = False
    total_reward = 0.0

    # Buffer for the professor requested episode trace
    # Reference for buffer return type lol
    episode_buffer: List[Tuple[np.ndarray, Any, float, np.ndarray, bool]] = []


    # Does the agent’s signature support an `explore` kw-arg?
    accepts_explore = "explore" in inspect.signature(agent.select_action).parameters

    while not done:
        action = ""
        if accepts_explore:
            action=agent.select_action(obs, explore=train)
        else:
            action=agent.select_action(obs)

        next_obs, reward, done, info = env.step(action)
     
        # Professor Zinovi’s request #1:
        # Every transition is now funnelled into the replay buffer via the agent’s store_transition hook.
        if train and hasattr(agent, "store_transition"):
            agent.store_transition(obs, action, reward, next_obs, done)

        # Appending the data to the buffer
        episode_buffer.append((obs, action, reward, next_obs, done))

        obs = next_obs
        total_reward += reward
        if render:
            env.render()

    # Lets also attach the buffer so train_qmix can read ittt
    
    agent.episode_buffer = episode_buffer

    success = 0
    if info.get("success", False):
        success=1
    elimination = int(env.agents_alive[env.faulty_agent] == 0)
    return total_reward, success, elimination

def train_qmix(agent, num_episodes, env_params=None, rng_seed=None):
    env = InsideBoxPushingEnv(**env_params) if env_params else InsideBoxPushingEnv()
    # Professor Zinovi’s request #2
    if rng_seed is not None:  
        env.reset(seed=rng_seed)        
        np.random.seed(rng_seed)        
        random.seed(rng_seed)           
        torch.manual_seed(rng_seed)     
        if torch.cuda.is_available():           
            torch.cuda.manual_seed_all(rng_seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    rewards= []
    successes= []
    eliminations = []
    LOGGER.info(f"Starting training for {num_episodes} episode(s)…")

    for ep in range(1, num_episodes + 1):
        r, s, e = run_episode(agent, env, train=True, render=False)
        if hasattr(agent, "train_step"):
            agent.train_step()

        rewards.append(r)
        successes.append(s)
        eliminations.append(e)



        # Professor Zinovi’s request #3 — per-episode trace

        # Extra details - lets discuss
        # for step_idx, (ob, ac, rw, nxt, dn) in enumerate(agent.episode_buffer):
        #     LOGGER.info("step={} | act={} | r={:.3f} | done={}".format(step_idx, ac, rw, dn))
        succ_result = "N"
        if s:
            succ_result = "Y"
        elim_result = "N"
        if e:
            elim_result = "Y"
        LOGGER.info("Model {} | Run-ID {} | Seed {} | Episode {}/{} | Reward {:>7.2f} | Success {} | Elim {}".format(agent.__class__.__name__, _run_id, rng_seed, ep, num_episodes, r, succ_result, elim_result))

    LOGGER.info("Training complete — overall metrics:")
    LOGGER.info("Avg Reward: {:>7.2f} | Success Rate: {:>5.1f}% | Elimination Rate: {:>5.1f}%".format(np.mean(rewards),np.mean(successes) * 100.0,np.mean(eliminations) * 100.0,))
    env.close()


def evaluate_model(agent, env_params = None, render = True, rng_seed=None):
    env = InsideBoxPushingEnv(**env_params) if env_params else InsideBoxPushingEnv()
    if rng_seed is not None:  
        env.reset(seed=rng_seed)        
        np.random.seed(rng_seed)        
        random.seed(rng_seed)           
        torch.manual_seed(rng_seed)     
        if torch.cuda.is_available():           
            torch.cuda.manual_seed_all(rng_seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    LOGGER.info("Evaluating trained QMIXAgent…")
    rewards, successes, eliminations = [], [], []
    for i in range(10):
        r, s, e = run_episode(agent, env, train=False, render=render)
        rewards.append(r)
        successes.append(s)
        eliminations.append(e)
    LOGGER.info("Eval 10-ep mean reward {:>.2f} | Success {:>4.1f}% | Elim {:>4.1f}%".format(np.mean(rewards),np.mean(successes) * 100.0,np.mean(eliminations) * 100.0,))
    env.close()
