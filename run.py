import argparse
import random
import numpy as np
import torch
from pathlib import Path
import time
import pickle

from environment.env import InsideBoxPushingEnv
from models.random_agent import RandomAgent
from models.qmix_agent import QMIXAgent
from utils.utils import run_episode, train_qmix, evaluate_model

def save_ckpt(agent, ckpt_dir: Path, fname: str):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    with open(ckpt_dir / fname, "wb") as f:
        pickle.dump(agent, f)

def latest_ckpt(ckpt_dir: Path, pattern: str):
    files = sorted(ckpt_dir.glob(pattern))
    return files[-1] if files else None


def train_with_curriculum(agent, episodes_per_phase: int, seed: int):
    phases = [
        dict(faulty_extra_mass=0.0, enable_voting=False, enable_obstacle=False), 
        dict(faulty_extra_mass=1.0, enable_voting=False, enable_obstacle=False), 
        dict(faulty_extra_mass=1.0, enable_voting=True,  enable_obstacle=False),  
        dict(faulty_extra_mass=1.0, enable_voting=True,  enable_obstacle=True),   
    ]
    base_env_kwargs = dict(
        grid_size=10,
        num_agents=4,
        max_steps=100,
        base_mass=1.0,
        push_strength=1.0,
        target_threshold=0.5,
    )

    for idx, extra in enumerate(phases, 1):
        print(f"\n=== Curriculum Phase {idx} ===")
        env_params = {**base_env_kwargs, **extra}
        train_qmix(agent, num_episodes=episodes_per_phase,
                   env_params=env_params, rng_seed=seed)


parser = argparse.ArgumentParser(description="Inside-Box Pushing runner")
parser.add_argument("--baseline", action="store_true", help="Perfect-team baseline (no faulty agent, no voting)")
parser.add_argument("--model", choices=["random", "qmix"], default="random", help="Which agent to run")
parser.add_argument("--episodes", type=int, default=50, help="Training episodes (per phase if --curriculum)")
parser.add_argument("--curriculum", action="store_true", help="Run the 4-phase curriculum (ignored with --baseline)")
parser.add_argument("--seed", type=int, default=None, help="RNG seed for reproducibility")
parser.add_argument("--checkpoint-dir", default="checkpoints", help="Directory for saving / loading model weights")
parser.add_argument("--tag", default=None, help="Optional tag; defaults to model+baseline+seed")
parser.add_argument("--fresh", action="store_true", help="Ignore existing checkpoints and start from scratch")
parser.add_argument("--no-eval", action="store_true", help="Skip the 10-episode greedy evaluation phase at the end")
args = parser.parse_args()


if args.seed is not None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

base_env_kwargs = dict(
    grid_size=10,
    num_agents=4,
    max_steps=100,
    base_mass=1.0,
    push_strength=1.0,
    target_threshold=0.5,
)

if args.model == "random":
    env_kwargs = {**base_env_kwargs}
    if args.baseline:
        env_kwargs.update(dict(faulty_extra_mass=0.0, enable_voting=False, enable_obstacle=False))
    else:
        env_kwargs.update(dict(faulty_extra_mass=1.0, enable_voting=True, enable_obstacle=True))

    env = InsideBoxPushingEnv(**env_kwargs)
    env.reset(seed=args.seed)
    agent = RandomAgent(env.action_space)
    run_episode(agent, env, train=False, render=True)
    env.close()
    exit(0)

ckpt_dir = Path(args.checkpoint_dir)
default_tag = f"{args.model}_{'baseline' if args.baseline else 'faulty'}_seed{args.seed}"
tag = args.tag or default_tag
ckpt_pattern = f"{tag}_*.pt"

print("Initializing QMIXAgent")
_tmp_env = InsideBoxPushingEnv(**base_env_kwargs)
_tmp_env.reset(seed=args.seed)
state_dim = _tmp_env.observation_space.shape[0]
_tmp_env.close()

agent = QMIXAgent(action_space=_tmp_env.action_space,n_agents=4, state_dim=state_dim)
if not args.fresh:
    ckpt = latest_ckpt(ckpt_dir, ckpt_pattern)
    if ckpt and not args.fresh:
        print(f"loading checkpoint {ckpt.name}")
        with open(ckpt, "rb") as f:
            agent = pickle.load(f)

# task configuration
if args.baseline:
    task_env_kwargs = {**base_env_kwargs, "faulty_extra_mass": 0.0, "enable_voting": False, "enable_obstacle": False}
else:
    task_env_kwargs = {}  

if args.curriculum and not args.baseline:
    train_with_curriculum(agent, args.episodes, args.seed)
else:
    train_qmix(agent, num_episodes=args.episodes, env_params=task_env_kwargs or None, rng_seed=args.seed)


print("\nEvaluating trained QMIXAgent")
timestamp = time.strftime("%Y%m%d-%H%M%S")
save_ckpt(agent, ckpt_dir, f"{tag}_{timestamp}.pt")
if not args.no_eval:
    evaluate_model(agent,render=True, env_params=task_env_kwargs or None, rng_seed=args.seed)
