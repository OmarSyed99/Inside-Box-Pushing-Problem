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

def run_episode(model, env, render: bool = False, *, epsilon: float = 0.05):
    """Run **one** episode.

    Parameters
    ----------
    model : QMIXAgent | RandomAgent | any policy with ``select_action``
    env   : InsideBoxPushingEnv instance (already created)
    render: bool, default False
        Whether to call ``env.render()`` every step.
    epsilon: float, default **0.05**
        Exploration rate passed straight to ``model.select_action``.  Use
        ``epsilon=0`` for fully‑greedy evaluation.

    Returns
    -------
    total_reward : float
    success      : int  (1 if reward > 0 else 0)
    elimination   : int  (1 if faulty agent removed else 0)
    """
    obs = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        # ←—— use the supplied epsilon (training ≈ 0.05, evaluation = 0.0)
        actions = model.select_action(obs, epsilon=epsilon)
        next_obs, reward, done, _ = env.step(actions)
        obs = next_obs
        total_reward += reward
        if render:
            env.render()

    # success if box ever reached the goal (positive bonus)
    success = 1 if total_reward > 0 else 0
    elimination = int(env.agents_alive[env.faulty_agent] == 0)
    return total_reward, success, elimination

def train_qmix(model, num_episodes: int, env_params: dict | None = None):
    """Train a *QMIXAgent* for ``num_episodes``.

    The function logs reward, success and elimination statistics every 10
    episodes and prints a final summary at the end.
    """
    env = InsideBoxPushingEnv(**env_params) if env_params else InsideBoxPushingEnv()

    rewards = []
    successes = []
    eliminations = []

    for ep in range(1, num_episodes + 1):
        # standard training exploration (epsilon left at default 0.05)
        r, s, e = run_episode(model, env, render=False)

        # one optimisation step per episode
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

def evaluate_model(model, env_params: dict | None = None, *, render: bool = True):
    """Evaluate *model* for **one** episode with deterministic (greedy) policy."""
    env = InsideBoxPushingEnv(**env_params) if env_params else InsideBoxPushingEnv()

    logger.info("Evaluating model with interactive rendering…")
    # fully‑greedy: epsilon = 0 disables any exploratory random actions
    r, s, e = run_episode(model, env, render=render, epsilon=0.0)

    status = "Passed" if s else "Failed"
    elim = "Eliminated faulty agent" if e else "Did not eliminate faulty agent"
    logger.info(f"Evaluation Reward: {r:.2f} | {status} | {elim}")
    env.close()
