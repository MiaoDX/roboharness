"""Example: Using GraspTaskStore for organizing multi-position grasp experiments.

This shows the storage layout for a grasp task where an agent
tests multiple grasp positions, each with multiple retry trials.

Run:
    python examples/grasp_task_storage.py
"""

from roboharness.storage import GraspTaskStore


def main() -> None:
    # 1. Create a grasp task store
    store = GraspTaskStore(base_dir="./harness_output", task_name="pick_and_place")

    # 2. Save task-level config
    store.save_task_config(
        {
            "object": "red_cube",
            "robot": "franka_panda",
            "simulator": "mujoco",
            "cameras": ["front", "side", "top", "wrist"],
            "checkpoints": ["plan_start", "pre_grasp", "contact", "lift"],
        }
    )

    # 3. Register grasp positions (each position is a different place to grasp)
    positions = [
        {"xyz": (0.5, 0.0, 0.05), "quaternion": (1, 0, 0, 0), "object_name": "red_cube"},
        {"xyz": (0.5, 0.1, 0.05), "quaternion": (0.707, 0, 0, 0.707), "object_name": "red_cube"},
        {"xyz": (0.4, -0.1, 0.05), "quaternion": (1, 0, 0, 0), "object_name": "red_cube"},
    ]

    for i, pos in enumerate(positions, 1):
        store.add_grasp_position(position_id=i, **pos)
        print(f"Registered grasp position {i}: xyz={pos['xyz']}")

        # 4. Simulate agent trials at this position
        #    In practice, the agent would run the simulation and capture images
        for trial in range(1, 3):
            # Get checkpoint directories (where images would be saved)
            for cp in GraspTaskStore.CHECKPOINTS:
                cp_dir = store.get_grasp_checkpoint_dir(
                    position_id=i, trial_id=trial, checkpoint=cp
                )
                # In real usage, CaptureResult.save(cp_dir) writes images here
                print(f"    Position {i}, Trial {trial}, Checkpoint '{cp}': {cp_dir}")

    # 5. Generate report
    report = store.generate_report()
    print(f"\nReport saved. Total positions: {report['total_positions']}")
    print("\nDirectory structure created at ./harness_output/pick_and_place/")


if __name__ == "__main__":
    main()
