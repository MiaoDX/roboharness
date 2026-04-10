#!/usr/bin/env python3
"""LeRobot evaluation plugin example — visual regression testing in one command.

Demonstrates how to use ``LeRobotEvaluator`` to run evaluation episodes on a
Gymnasium-compatible environment, capture visual checkpoints, and produce a
CI-compatible report.

Usage::

    pip install roboharness[lerobot]
    python examples/lerobot_eval.py
    python examples/lerobot_eval.py --episodes 20 --min-success-rate 0.8

After running, inspect the results::

    roboharness lerobot-eval ./lerobot_eval_output/autonomous_report.json

Requires: gymnasium, Pillow (included in ``roboharness[lerobot]``).
"""

from __future__ import annotations

import argparse
import logging
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(name)s — %(message)s")


def make_demo_env() -> Any:
    """Create a simple Gymnasium environment for demonstration.

    In a real workflow, replace this with your LeRobot env::

        import lerobot
        env = lerobot.envs.factory.make_env("lerobot/unitree-g1-mujoco")
    """
    import gymnasium as gym

    return gym.make("CartPole-v1", render_mode="rgb_array")


def make_demo_policy(env: Any) -> Any:
    """Create a trivial random policy for demonstration.

    In a real workflow, replace this with your trained policy::

        policy = load_my_policy("path/to/checkpoint")
    """
    action_space = env.action_space

    def policy(obs: Any) -> Any:
        return action_space.sample()

    return policy


def main() -> int:
    parser = argparse.ArgumentParser(description="LeRobot evaluation plugin example")
    parser.add_argument(
        "--episodes", type=int, default=5, help="Number of evaluation episodes (default: 5)"
    )
    parser.add_argument(
        "--min-success-rate",
        type=float,
        default=0.0,
        help="Minimum success rate for CI pass (0.0-1.0, default: 0.0)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./lerobot_eval_output",
        help="Output directory for reports and captures",
    )
    args = parser.parse_args()

    from roboharness.lerobot import LeRobotEvaluator

    # 1. Create environment and policy
    env = make_demo_env()
    policy = make_demo_policy(env)

    # 2. Create evaluator with checkpoint capture
    evaluator = LeRobotEvaluator(
        env=env,
        output_dir=args.output_dir,
        task_name="cartpole_demo",
        checkpoint_interval=50,  # capture every 50 steps
        max_steps=500,
        success_key="success",  # CartPole doesn't have this, so success stays False
    )

    # 3. Run evaluation
    report = evaluator.run(
        policy=policy,
        num_episodes=args.episodes,
        min_success_rate=args.min_success_rate,
    )

    # 4. Print summary
    print(f"\n{'=' * 50}")
    print(f"Task:           {report.task_name}")
    print(f"Episodes:       {report.num_episodes}")
    print(f"Success rate:   {report.success_rate:.0%}")
    print(f"Mean reward:    {report.mean_reward:.1f}")
    print(f"Mean length:    {report.mean_episode_length:.0f} steps")
    print(f"Report:         {report.output_dir}/autonomous_report.json")
    print(f"{'=' * 50}")

    # 5. Return CI exit code
    exit_code = report.exit_code(min_success_rate=args.min_success_rate)
    if exit_code == 0:
        print("PASS")
    else:
        print(f"FAIL — success rate {report.success_rate:.0%} < {args.min_success_rate:.0%}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
