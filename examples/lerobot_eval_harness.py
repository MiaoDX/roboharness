#!/usr/bin/env python3
"""LeRobot Evaluation Harness — visual regression testing for robot policies.

Evaluates a policy on a Gymnasium environment with roboharness visual checkpoints,
structured JSON output, and a CI-friendly pass/fail gate.

This example works with any Gymnasium environment. For full LeRobot integration,
install ``roboharness[lerobot]`` and pass a LeRobot policy checkpoint.

Requirements:
    pip install roboharness[demo]         # basic (Gymnasium env)
    pip install roboharness[lerobot]      # full LeRobot integration

Run (standalone — CartPole demo):
    python examples/lerobot_eval_harness.py --env CartPole-v1 --n-episodes 5

Run (with success threshold — CI gate):
    python examples/lerobot_eval_harness.py --env CartPole-v1 --n-episodes 10 \
        --min-success-rate 0.0 --assert-threshold

Output:
    ./harness_output/lerobot_eval/
        episode_000/  — checkpoint screenshots per episode
        episode_001/
        ...
        lerobot_eval_report.json  — structured evaluation report
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

from roboharness.evaluate.lerobot_plugin import (
    LeRobotEvalConfig,
    check_eval_threshold,
    evaluate_policy,
)


def _random_policy(obs: np.ndarray, action_space: Any = None) -> np.ndarray:
    """Fallback random policy when no trained policy is available."""
    if action_space is not None and hasattr(action_space, "sample"):
        return action_space.sample()
    return np.zeros(2, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="LeRobot evaluation harness")
    parser.add_argument(
        "--env",
        default="CartPole-v1",
        help="Gymnasium environment ID (default: CartPole-v1)",
    )
    parser.add_argument("--n-episodes", type=int, default=10, help="Number of episodes")
    parser.add_argument("--max-steps", type=int, default=500, help="Max steps per episode")
    parser.add_argument("--output-dir", default="./harness_output", help="Output directory")
    parser.add_argument(
        "--checkpoint-steps",
        type=int,
        nargs="*",
        default=[],
        help="Steps at which to capture checkpoints (e.g. 10 50 100)",
    )
    parser.add_argument(
        "--min-success-rate",
        type=float,
        default=0.0,
        help="Minimum success rate threshold (0.0 to 1.0)",
    )
    parser.add_argument(
        "--min-mean-reward",
        type=float,
        default=None,
        help="Minimum mean reward threshold",
    )
    parser.add_argument(
        "--assert-threshold",
        action="store_true",
        help="Exit non-zero if thresholds are not met (CI mode)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Roboharness: LeRobot Evaluation Harness")
    print("=" * 60)

    # 1. Create environment
    print(f"\n[1/3] Creating environment: {args.env}")
    try:
        import gymnasium as gym

        env = gym.make(args.env, render_mode="rgb_array")
    except ImportError:
        print("ERROR: gymnasium is required. Install with: pip install roboharness[demo]")
        sys.exit(1)

    print(f"      Obs space: {env.observation_space}")
    print(f"      Act space: {env.action_space}")

    # 2. Run evaluation
    print(f"[2/3] Evaluating ({args.n_episodes} episodes, max {args.max_steps} steps each) ...")
    output_dir = Path(args.output_dir) / "lerobot_eval"

    config = LeRobotEvalConfig(
        n_episodes=args.n_episodes,
        max_steps_per_episode=args.max_steps,
        checkpoint_steps=args.checkpoint_steps or [],
        output_dir=str(output_dir),
    )

    # Use random policy as fallback
    action_space = env.action_space

    def policy_fn(obs: np.ndarray) -> np.ndarray:
        return _random_policy(obs, action_space)

    report = evaluate_policy(env, policy_fn, config)
    env.close()

    # 3. Report results
    print("[3/3] Results:")
    print(f"      Episodes:        {report.n_episodes}")
    print(f"      Success rate:    {report.success_rate:.1%}")
    print(f"      Mean reward:     {report.mean_reward:.2f}")
    print(f"      Mean ep length:  {report.mean_episode_length:.1f}")
    print(f"      Wall time:       {report.wall_time:.2f}s")

    report_path = output_dir / "lerobot_eval_report.json"
    if report_path.exists():
        print(f"      Report saved:    {report_path}")

    # Print per-episode summary
    print("\n  Per-episode results:")
    for ep in report.episodes:
        status = "PASS" if ep.success else "FAIL"
        print(
            f"    Episode {ep.episode_id:3d}: [{status}]"
            f"  reward={ep.total_reward:7.2f}"
            f"  length={ep.episode_length:4d}"
        )

    # 4. CI gate
    if args.assert_threshold:
        passed = check_eval_threshold(
            report,
            min_success_rate=args.min_success_rate,
            min_mean_reward=args.min_mean_reward,
        )
        print(f"\n  Threshold check: {'PASSED' if passed else 'FAILED'}")
        if args.min_success_rate > 0:
            print(f"    success_rate >= {args.min_success_rate:.1%}: {report.success_rate:.1%}")
        if args.min_mean_reward is not None:
            print(f"    mean_reward >= {args.min_mean_reward:.2f}: {report.mean_reward:.2f}")

        if not passed:
            sys.exit(1)

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
