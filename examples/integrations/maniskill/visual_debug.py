#!/usr/bin/env python3
"""ManiSkill + Roboharness: Visual Debugging for Manipulation Tasks.

A community example showing how to add checkpoint-based visual debugging
to ManiSkill environments using roboharness's Gymnasium wrapper.

This example demonstrates:
  - Wrapping a ManiSkill PickCube-v1 environment with RobotHarnessWrapper
  - Using semantic task protocols to define meaningful capture phases
  - Automatic screenshot capture at task-relevant checkpoints
  - Saving agent-consumable state JSON alongside visual captures
  - Working with ManiSkill's vectorized rewards and dict observations

The output directory contains checkpoint captures that an AI coding agent
can inspect to debug manipulation policies — no separate VLM needed.

Run (with ManiSkill installed):
    pip install roboharness gymnasium mani-skill
    python examples/integrations/maniskill/visual_debug.py

Run (without ManiSkill — uses built-in mock for demonstration):
    pip install roboharness gymnasium
    python examples/integrations/maniskill/visual_debug.py --mock

Output:
    ./harness_maniskill_output/pick_cube/trial_001/
        approach/    — gripper approaching the cube
        contact/     — gripper making contact
        lift/        — cube being lifted

Upstream target: ManiSkill examples
  https://github.com/haosulab/ManiSkill
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Mock environment (for running without ManiSkill installed)
# ---------------------------------------------------------------------------


def _create_mock_env() -> Any:
    """Create a lightweight ManiSkill-like mock environment for demonstration."""
    import gymnasium as gym
    from gymnasium import spaces

    class MockPickCubeEnv(gym.Env):
        """Minimal PickCube-like environment that mimics ManiSkill's interface.

        Returns vectorized rewards (shape: [1]) and dict observations,
        matching ManiSkill's CPUGymWrapper output format.
        """

        metadata: dict = {"render_modes": ["rgb_array"], "render_fps": 20}  # noqa: RUF012

        def __init__(self, render_mode: str = "rgb_array"):
            super().__init__()
            self.render_mode = render_mode
            self.observation_space = spaces.Dict(
                {
                    "agent": spaces.Box(-np.inf, np.inf, shape=(9,), dtype=np.float32),
                    "extra": spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32),
                }
            )
            self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
            self._step_count = 0

        def reset(
            self, *, seed: int | None = None, options: dict[str, Any] | None = None
        ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
            super().reset(seed=seed, options=options)
            self._step_count = 0
            return self._make_obs(), {}

        def step(
            self, action: Any
        ) -> tuple[dict[str, np.ndarray], np.ndarray, bool, bool, dict[str, Any]]:
            self._step_count += 1
            obs = self._make_obs()
            reward = np.array([0.1 * min(self._step_count / 100, 1.0)], dtype=np.float32)
            terminated = self._step_count >= 200
            return obs, reward, terminated, False, {}

        def render(self) -> np.ndarray:
            # Generate a simple colored frame that changes with steps
            frame = np.zeros((256, 256, 3), dtype=np.uint8)
            progress = min(self._step_count / 200, 1.0)
            # Blue background fading to green as task progresses
            frame[:, :, 1] = int(200 * progress)
            frame[:, :, 2] = int(200 * (1.0 - progress))
            # Add a "cube" indicator
            cx, cy = 128, int(128 - 60 * progress)
            frame[cy - 10 : cy + 10, cx - 10 : cx + 10] = [220, 50, 50]
            return frame

        def _make_obs(self) -> dict[str, np.ndarray]:
            return {
                "agent": np.random.randn(9).astype(np.float32) * 0.1,
                "extra": np.random.randn(7).astype(np.float32) * 0.1,
            }

    return MockPickCubeEnv(render_mode="rgb_array")


def _create_maniskill_env() -> Any:
    """Create a real ManiSkill PickCube-v1 environment."""
    import gymnasium as gym

    return gym.make(
        "PickCube-v1",
        obs_mode="state_dict",
        render_mode="rgb_array",
        max_episode_steps=200,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ManiSkill + Roboharness: Visual Debugging for Manipulation"
    )
    parser.add_argument(
        "--output-dir", default="./harness_maniskill_output", help="Output directory"
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock environment (no ManiSkill dependency required)",
    )
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to run")
    args = parser.parse_args()

    try:
        import gymnasium  # noqa: F401
    except ImportError:
        print("ERROR: gymnasium is required. Install with: pip install gymnasium")
        sys.exit(1)

    from roboharness.core.protocol import TaskPhase, TaskProtocol
    from roboharness.wrappers import RobotHarnessWrapper

    output_dir = Path(args.output_dir)

    print("=" * 60)
    print("  ManiSkill + Roboharness: Visual Debugging")
    print("=" * 60)

    # 1. Create the environment
    use_mock = args.mock
    if not use_mock:
        try:
            env = _create_maniskill_env()
            print("\n[1/4] Loaded ManiSkill PickCube-v1 environment")
        except Exception:
            print("\n[1/4] ManiSkill not available, falling back to mock environment")
            print("       (install mani-skill for the real environment, or use --mock)")
            use_mock = True

    if use_mock:
        env = _create_mock_env()
        print("\n[1/4] Using mock PickCube environment (--mock mode)")

    # 2. Define a semantic task protocol for pick-and-place debugging
    pick_protocol = TaskProtocol(
        name="pick_cube",
        description="PickCube manipulation task — visual debugging checkpoints",
        phases=[
            TaskPhase(
                "approach",
                "Gripper approaching the cube — check end-effector trajectory",
            ),
            TaskPhase(
                "contact",
                "Gripper making contact with cube — verify grasp alignment",
            ),
            TaskPhase(
                "lift",
                "Cube being lifted — confirm stable grasp and clearance",
            ),
        ],
    )

    # 3. Wrap with RobotHarnessWrapper
    print("[2/4] Wrapping environment with RobotHarnessWrapper ...")
    wrapped_env = RobotHarnessWrapper(
        env,
        protocol=pick_protocol,
        phase_steps={"approach": 50, "contact": 100, "lift": 150},
        cameras=["default"],
        output_dir=str(output_dir),
        task_name="pick_cube",
    )
    print(f"      Protocol: {wrapped_env.active_protocol.name}")
    print("      Checkpoint steps: approach=50, contact=100, lift=150")

    # 4. Run episodes
    print(f"[3/4] Running {args.episodes} episode(s) ...")

    for episode in range(args.episodes):
        _obs, info = wrapped_env.reset()
        total_reward = 0.0
        checkpoints_hit: list[str] = []

        steps_done = 0
        for _ in range(200):
            action = wrapped_env.action_space.sample()
            _obs, reward, terminated, truncated, info = wrapped_env.step(action)
            steps_done += 1
            total_reward += float(np.mean(reward))

            if "checkpoint" in info:
                cp = info["checkpoint"]
                checkpoints_hit.append(cp["name"])
                print(
                    f"      Episode {episode + 1} | Checkpoint '{cp['name']}' at step {cp['step']}"
                )
                print(f"        -> Captures saved to: {cp['capture_dir']}")

            if terminated or truncated:
                break

        print(
            f"      Episode {episode + 1} finished: {steps_done} steps,"
            f" reward={total_reward:.3f},"
            f" checkpoints={checkpoints_hit}"
        )

    wrapped_env.close()

    # 5. Summary
    print("\n[4/4] Done!")
    trial_dir = output_dir / "pick_cube" / "trial_001"
    if trial_dir.exists():
        print(f"      Captures saved to: {trial_dir}")
        print("\n  Output structure:")
        for cp_dir in sorted(trial_dir.iterdir()):
            if cp_dir.is_dir():
                files = sorted(f.name for f in cp_dir.iterdir() if f.is_file())
                print(f"    {cp_dir.name}/")
                for fname in files:
                    print(f"      {fname}")

    print()
    print("  How an AI agent uses these captures:")
    print("    1. Inspect approach/ screenshots to verify gripper trajectory")
    print("    2. Check contact/ state.json for grasp quality metrics")
    print("    3. Compare lift/ images to confirm cube clearance")
    print("    4. Iterate on the policy based on visual evidence")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
