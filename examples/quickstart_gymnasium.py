"""Quick Start: Using RobotHarnessWrapper with any Gymnasium environment.

This example shows how to wrap a standard Gymnasium environment
with Roboharness to add checkpoint-based visual capture.

Run:
    pip install roboharness gymnasium
    python examples/quickstart_gymnasium.py
"""

import gymnasium as gym

from roboharness.core.protocol import TaskPhase, TaskProtocol
from roboharness.wrappers import RobotHarnessWrapper


def main() -> None:
    # 1. Create any Gymnasium environment with rgb_array rendering
    env = gym.make("CartPole-v1", render_mode="rgb_array")

    # 2. Wrap with RobotHarnessWrapper using a semantic task protocol
    balance_protocol = TaskProtocol(
        name="balance_monitor",
        description="CartPole balance monitoring across episode stages",
        phases=[
            TaskPhase("early", "Early balance — controller settling"),
            TaskPhase("mid", "Mid-episode — sustained balance test"),
            TaskPhase("late", "Late-episode — long-horizon stability"),
        ],
    )
    env = RobotHarnessWrapper(
        env,
        protocol=balance_protocol,
        phase_steps={"early": 10, "mid": 50, "late": 100},
        cameras=["default"],
        output_dir="./harness_output",
        task_name="cartpole_balance",
    )

    # 3. Run the standard Gymnasium loop
    _obs, info = env.reset()
    total_reward = 0.0

    for step in range(200):
        action = env.action_space.sample()  # Replace with agent's action
        _obs, reward, terminated, truncated, info = env.step(action)
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
