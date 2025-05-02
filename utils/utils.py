import os
import re
import logging
from collections import deque
from typing import Tuple

import numpy as np
import torch

from environment.env import InsideBoxPushingEnv

# ---------- logger ----------
logger = logging.getLogger("inside_box")
logger.setLevel(logging.INFO)

LOG_FILE = "training.log"
fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

# file + console handlers
for h in (logging.FileHandler(LOG_FILE), logging.StreamHandler()):
    h.setFormatter(fmt)
    h.setLevel(logging.INFO)
    logger.addHandler(h)

# determine run‑id
run_id = 1
if os.path.exists(LOG_FILE):
    with open(LOG_FILE) as f:
        ids = re.findall(r"=== Run (\d+) ===", f.read())
        if ids:
            run_id = max(map(int, ids)) + 1

logger.info("\n=== Run %d ===\n", run_id)


# ---------- helpers ----------
def run_episode(
    model,
    env: InsideBoxPushingEnv,
    epsilon: float = 0.05,
    render: bool = False,
) -> Tuple[float, int, int]:
    """
    Executes one episode (no learning inside) and returns:
        total_reward, success_flag (0/1), elimination_flag (0/1)
    """
    obs = env.reset()
    done = False
    tot_r = 0.0

    while not done:
        # model API wrapper
        if hasattr(model, "select_action"):
            actions = model.select_action(obs, epsilon=epsilon)
        else:  # RandomAgent
            actions = model.get_actions(obs)

        obs, r, done, _ = env.step(actions)
        tot_r += r
        if render:
            env.render()

    success = 1 if tot_r > 0 else 0
    eliminated = int(env.agents_alive[env.faulty_agent] == 0)
    return tot_r, success, eliminated


def train_qmix(model, num_episodes: int, env_params: dict | None = None):
    """
    Trains QMIXAgent for `num_episodes` and logs 10‑episode rolling stats.
    Implements a simple ε‑greedy schedule: linearly decays from 1.0 ➜ 0.05.
    """
    env = InsideBoxPushingEnv(**env_params) if env_params else InsideBoxPushingEnv()

    eps_start, eps_end = 1.0, 0.05
    eps_decay = (eps_start - eps_end) / num_episodes

    rewards, successes, eliminations = [], [], []

    for ep in range(1, num_episodes + 1):
        epsilon = max(eps_end, eps_start - eps_decay * (ep - 1))

        r, s, e = run_episode(model, env, epsilon=epsilon, render=False)
        # one learning step
        if hasattr(model, "train_step"):
            _ = model.train_step()

        rewards.append(r)
        successes.append(s)
        eliminations.append(e)

        # every 10 episodes, log stats
        if ep % 10 == 0:
            avg_r = np.mean(rewards[-10:])
            sr = np.mean(successes[-10:]) * 100
            er = np.mean(eliminations[-10:]) * 100
            logger.info(
                "Training Episode %d/%d complete.", ep, num_episodes
            )
            logger.info(
                "Avg Reward: %.2f | Success Rate: %.1f%% | Elimination Rate: %.1f%%",
                avg_r,
                sr,
                er,
            )

    # final summary
    logger.info("Training complete. Final Metrics:")
    logger.info(
        "Avg Reward: %.2f | Success Rate: %.1f%% | Elimination Rate: %.1f%%",
        np.mean(rewards),
        np.mean(successes) * 100,
        np.mean(eliminations) * 100,
    )
    env.close()


def evaluate_model(model, env_params: dict | None = None, render: bool = True):
    """
    Runs one rendered episode with near‑greedy policy (ε = 0.05).
    """
    env = InsideBoxPushingEnv(**env_params) if env_params else InsideBoxPushingEnv()
    logger.info("Evaluating model with interactive rendering…")
    r, s, e = run_episode(model, env, epsilon=0.05, render=render)
    logger.info(
        "Evaluation Reward: %.2f | %s | %s",
        r,
        "Passed" if s else "Failed",
        "Eliminated faulty agent" if e else "Did not eliminate faulty agent",
    )
    env.close()
