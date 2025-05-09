#!/usr/bin/env python3
import argparse

from environment.env import InsideBoxPushingEnv
from models.random_agent import RandomAgent
from models.qmix_agent import QMIXAgent
from utils.utils import run_episode, train_qmix, evaluate_model


def train_with_curriculum(model, episodes_per_phase):
    phases = [
      
        dict(faulty_extra_mass=0.0, enable_voting=False, enable_obstacle=False),
        dict(faulty_extra_mass=1.0, enable_voting=False, enable_obstacle=False),
        dict(faulty_extra_mass=1.0, enable_voting=True, enable_obstacle=False),
        dict(faulty_extra_mass=1.0, enable_voting=True, enable_obstacle=True),
    ]

    for i, params in enumerate(phases, start=1):
        print(f"\n=== Curriculum Phase {i} ===")
        env_params = {
            'grid_size': 10,
            'num_agents': 4,
            'max_steps': 100,
            'base_mass': 1.0,
            'push_strength': 1.0,
            'target_threshold': 0.5,
            **params
        }
        train_qmix(model, num_episodes=episodes_per_phase, env_params=env_params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the Inside Box Pushing Env with a chosen model"
    )
    parser.add_argument(
        "--baseline", action="store_true",
        help="Perfect‑team baseline: no faulty agent, no voting"
    )
    parser.add_argument(
        "--model", type=str, default="random",
        choices=["random", "qmix"],
        help="Model to use: 'random' or 'qmix'"
    )
    parser.add_argument(
        "--episodes", type=int, default=50,
        help="Number of training episodes (per phase if using curriculum)"
    )
    parser.add_argument(
        "--curriculum", action="store_true",
        help="Train QMIX through the 4‑phase curriculum (ignored if --baseline)"
    )
    args = parser.parse_args()

    if args.model == "random":
        print("Running RandomAgent…")
        env = InsideBoxPushingEnv(
            grid_size=10,
            num_agents=4,
            max_steps=100,
            base_mass=1.0,
            faulty_extra_mass=(0.0 if args.baseline else 1.0),
            push_strength=1.0,
            target_threshold=0.5,
            enable_voting=not args.baseline,
            enable_obstacle=not args.baseline
        )
        model = RandomAgent(env.action_space)
        _ = run_episode(model, env, train=False, render=True)
        env.close()

    elif args.model == "qmix":
        print("Setting up QMIXAgent model…")
        temp = InsideBoxPushingEnv()
        state_dim = temp.observation_space.shape[0]
        model = QMIXAgent(temp.action_space, n_agents=4, state_dim=state_dim)
        temp.close()

        if args.curriculum and not args.baseline:
            print("Training QMIXAgent through curriculum…")
            train_with_curriculum(model, episodes_per_phase=args.episodes)
        else:
            print("Training QMIXAgent model…")
            env_kwargs = {}
            if args.baseline:
                env_kwargs = dict(
                    grid_size=10,
                    num_agents=4,
                    max_steps=100,
                    base_mass=1.0,
                    faulty_extra_mass=0.0,
                    push_strength=1.0,
                    target_threshold=0.5,
                    enable_voting=False,
                    enable_obstacle=False
                )
            train_qmix(
                model,
                num_episodes=args.episodes,
                env_params=env_kwargs if args.baseline else None
            )

        print("\nEvaluating trained QMIXAgent model…")
        evaluate_model(
            model,
            render=True,
            env_params=env_kwargs if args.baseline else None
        )
