#!/usr/bin/env python3
"""Isaac Lab Integration — Using RobotHarnessWrapper with Isaac Lab environments.

This example demonstrates how to wrap an Isaac Lab Gymnasium environment
with Roboharness for checkpoint-based visual capture.

Requirements:
    - NVIDIA GPU with drivers >= 580.65.06
    - Isaac Sim + Isaac Lab installed (see https://isaac-sim.github.io/IsaacLab/)
    - pip install roboharness

Run:
    # From the Isaac Lab root (with Isaac Sim environment activated):
    python examples/isaac_lab_integration.py

    # With custom settings:
    python examples/isaac_lab_integration.py --task Isaac-Reach-Franka-v0 --num-envs 1

Note:
    Isaac Lab environments return PyTorch tensors (not NumPy arrays) for
    observations and expect tensors for actions. The RobotHarnessWrapper
    handles this transparently — it does not modify observations or actions.
"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Roboharness + Isaac Lab Integration")
    parser.add_argument(
        "--task",
        default="Isaac-Reach-Franka-v0",
        help="Isaac Lab task name (default: Isaac-Reach-Franka-v0)",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="Number of parallel environments (default: 1)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=200,
        help="Maximum steps per episode (default: 200)",
    )
    parser.add_argument(
        "--output-dir",
        default="./harness_output",
        help="Output directory (default: ./harness_output)",
    )
    args = parser.parse_args()

    # Isaac Lab requires isaaclab.app to be imported before gymnasium
    try:
        from isaaclab.app import AppLauncher

        launcher = AppLauncher(headless=True)
        launcher.app.start()
    except ImportError:
        print("ERROR: Isaac Lab is not installed or not available.")
        print()
        print("Isaac Lab requires:")
        print("  - NVIDIA GPU with drivers >= 580.65.06")
        print("  - Isaac Sim (Omniverse)")
        print("  - Isaac Lab (pip install)")
        print()
        print("See: https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html")
        sys.exit(1)

    import gymnasium as gym

    # Isaac Lab registers its envs on import
    import isaaclab_tasks  # noqa: F401

    from roboharness.wrappers import RobotHarnessWrapper

    print("=" * 60)
    print("  Roboharness: Isaac Lab Integration")
    print("=" * 60)

    # 1. Create the Isaac Lab environment via standard Gymnasium API
    print(f"\n[1/3] Creating environment: {args.task} (num_envs={args.num_envs})")
    env = gym.make(
        args.task,
        num_envs=args.num_envs,
        render_mode="rgb_array",
    )

    # 2. Wrap with RobotHarnessWrapper — zero changes to environment code
    print("[2/3] Wrapping with RobotHarnessWrapper ...")
    env = RobotHarnessWrapper(
        env,
        checkpoints=[
            {"name": "start", "step": 1},
            {"name": "quarter", "step": args.max_steps // 4},
            {"name": "mid", "step": args.max_steps // 2},
            {"name": "end", "step": args.max_steps},
        ],
        cameras=["default"],
        output_dir=args.output_dir,
        task_name=args.task.lower().replace("-", "_"),
    )

    # 3. Run the standard Gymnasium loop
    print(f"[3/3] Running for up to {args.max_steps} steps ...")
    obs, info = env.reset()
    total_reward = 0.0

    for step in range(args.max_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        # reward may be a torch tensor in Isaac Lab
        if hasattr(reward, "item"):
            total_reward += reward.item()
        else:
            total_reward += float(reward)

        if "checkpoint" in info:
            cp = info["checkpoint"]
            print(f"  Checkpoint '{cp['name']}' at step {cp['step']}")
            print(f"  Captures saved to: {cp['capture_dir']}")

        if terminated or truncated:
            # Isaac Lab vectorized envs auto-reset, but single-env may terminate
            if hasattr(terminated, "item"):
                terminated = terminated.item()
            if hasattr(truncated, "item"):
                truncated = truncated.item()
            if terminated or truncated:
                print(f"  Episode ended at step {step + 1}, total reward: {total_reward:.2f}")
                break

    env.close()
    launcher.app.close()

    print(f"\nDone! Total reward: {total_reward:.2f}")
    print(f"Check {args.output_dir}/ for captured screenshots and state.")
    print("=" * 60)


if __name__ == "__main__":
    main()
