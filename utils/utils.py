import os
import re
import numpy as np
import torch
import logging
from environment.env import InsideBoxPushingEnv

# --- Logger setup ---
logger = logging.getLogger('inside_box')
logger.setLevel(logging.INFO)

LOG_FILE = 'training.log'

# File handler
fh = logging.FileHandler(LOG_FILE)
fh.setLevel(logging.INFO)

# Console handler
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# Formatter
fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
fh.setFormatter(fmt)
ch.setFormatter(fmt)

logger.addHandler(fh)
logger.addHandler(ch)
# --------------------

# Determine run number by scanning existing log
run_id = 1
if os.path.exists(LOG_FILE):
    with open(LOG_FILE, 'r') as f:
        data = f.read()
    matches = re.findall(r"=== Run (\d+) ===", data)
    if matches:
        run_id = max(int(m) for m in matches) + 1

# Log a header for this run
logger.info('')
logger.info(f"=== Run {run_id} ===")
logger.info('')

def run_episode(model, env, render=False):
    """
    Run one episode (no learning here).
    Returns (total_reward, success_flag:int, elimination_flag:int).
    """
    obs = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        actions = model.select_action(obs)
        next_obs, reward, done, _ = env.step(actions)
        obs = next_obs
        total_reward += reward
        if render:
            env.render()

    # success if box ever reached the goal (positive bonus)
    success = 1 if total_reward > 0 else 0
    # faulty agent index and alive flags come from the env
    elimination = int(env.agents_alive[env.faulty_agent] == 0)
    return total_reward, success, elimination

def train_qmix(model, num_episodes, env_params=None):
    """
    Train a QMIXAgent for num_episodes.
    Logs reward, success, and elimination rates.
    """
    env = InsideBoxPushingEnv(**env_params) if env_params else InsideBoxPushingEnv()

    rewards = []
    successes = []
    eliminations = []

    for ep in range(1, num_episodes + 1):
        r, s, e = run_episode(model, env, render=False)
        # one training step
        if hasattr(model, 'train_step'):
            _ = model.train_step()

        rewards.append(r)
        successes.append(s)
        eliminations.append(e)

        if ep % 10 == 0:
            avg_r = np.mean(rewards[-10:])
            sr = np.mean(successes[-10:]) * 100
            er = np.mean(eliminations[-10:]) * 100
            logger.info(f"Training Episode {ep}/{num_episodes} complete.")
            logger.info(f"Avg Reward: {avg_r:.2f} | Success Rate: {sr:.1f}% | Elimination Rate: {er:.1f}%")

    # final summary
    final_avg_r = np.mean(rewards)
    final_sr = np.mean(successes) * 100
    final_er = np.mean(eliminations) * 100
    logger.info("Training complete. Final Metrics:")
    logger.info(f"Avg Reward: {final_avg_r:.2f} | Success Rate: {final_sr:.1f}% | Elimination Rate: {final_er:.1f}%")
    env.close()

def evaluate_model(model, env_params=None, render=True):
    """
    Run one episode with rendering, report reward, pass/fail, and elimination.
    """
    env = InsideBoxPushingEnv(**env_params) if env_params else InsideBoxPushingEnv()

    logger.info("Evaluating model with interactive rendering...")
    r, s, e = run_episode(model, env, render=render)
    status = "Passed" if s else "Failed"
    elim = "Eliminated faulty agent" if e else "Did not eliminate faulty agent"
    logger.info(f"Evaluation Reward: {r:.2f} | {status} | {elim}")
    env.close()