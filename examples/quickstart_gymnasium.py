"""Quick Start: Using RobotHarnessWrapper with any Gymnasium environment.

This example shows how to wrap a standard Gymnasium environment
with Robot-Harness to add checkpoint-based visual capture.

Run:
    pip install robot-harness gymnasium
    python examples/quickstart_gymnasium.py
"""

import gymnasium as gym

from robot_harness.wrappers import RobotHarnessWrapper


def main() -> None:
    # 1. Create any Gymnasium environment with rgb_array rendering
    env = gym.make("CartPole-v1", render_mode="rgb_array")

    # 2. Wrap with RobotHarnessWrapper — zero changes to environment code
    env = RobotHarnessWrapper(
        env,
        checkpoints=[
            {"name": "early", "step": 10},
            {"name": "mid", "step": 50},
            {"name": "late", "step": 100},
        ],
        cameras=["default"],
        output_dir="./harness_output",
        task_name="cartpole_balance",
    )

    # 3. Run the standard Gymnasium loop
    obs, info = env.reset()
    total_reward = 0.0

    for step in range(200):
        action = env.action_space.sample()  # Replace with agent's action
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Check if a checkpoint was hit
        if "checkpoint" in info:
            cp = info["checkpoint"]
            print(f"  Checkpoint '{cp['name']}' at step {cp['step']}")
            print(f"  Captures saved to: {cp['capture_dir']}")

        if terminated or truncated:
            print(f"Episode ended at step {step + 1}, total reward: {total_reward}")
            break

    env.close()
    print("\nDone! Check ./harness_output/ for captured screenshots and state.")


if __name__ == "__main__":
    main()
