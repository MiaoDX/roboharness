#!/usr/bin/env python3
"""MuJoCo + Rerun Example — Capture `.rrd` files with Rerun Blueprint.

Demonstrates the Rerun integration in Roboharness:
  1. Load a MuJoCo model and configure the Harness with ``enable_rerun=True``
  2. Run a scripted grasp sequence through checkpoints
  3. Automatically write a `.rrd` file with RGB, depth, and state data
  4. Apply a Rerun Blueprint for a standardized debug layout

The resulting `.rrd` file can be opened with the Rerun Viewer::

    rerun harness_output/mujoco_grasp/trial_001/capture.rrd

Run:
    pip install roboharness[mujoco,rerun] Pillow
    MUJOCO_GL=osmesa python examples/mujoco_rerun.py

Output:
    ./harness_output/mujoco_grasp/trial_001/
        capture.rrd      — Rerun recording with all checkpoint data
        pre_grasp/       — gripper open, above the cube
        approach/        — gripper lowered onto the cube
        lift/            — cube lifted off the table
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from roboharness.backends.mujoco_meshcat import MuJoCoMeshcatBackend
from roboharness.core.harness import Harness
from roboharness.core.protocol import GRASP_PROTOCOL

# ---------------------------------------------------------------------------
# Inline MJCF model: table + cube + 2-finger gripper + 3 cameras
# ---------------------------------------------------------------------------
GRASP_MJCF = """\
<mujoco model="simple_grasp">
  <option gravity="0 0 -9.81" timestep="0.002"/>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.6 0.8 1.0" rgb2="0.2 0.3 0.5"
             width="256" height="256"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.9 0.9 0.9" rgb2="0.7 0.7 0.7"
             width="256" height="256"/>
    <material name="grid_mat" texture="grid" texrepeat="4 4" reflectance="0.1"/>
    <material name="table_mat" rgba="0.6 0.4 0.2 1"/>
    <material name="cube_mat" rgba="0.9 0.2 0.2 1"/>
    <material name="gripper_mat" rgba="0.3 0.3 0.7 1"/>
  </asset>

  <worldbody>
    <geom type="plane" size="1 1 0.01" material="grid_mat"/>
    <light pos="0 0 2" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <light pos="0.5 0.5 1.5" dir="-0.3 -0.3 -1" diffuse="0.4 0.4 0.4"/>

    <camera name="front" pos="0.75 0 0.55" xyaxes="0 1 0 -0.4 0 0.75"/>
    <camera name="side" pos="0 0.75 0.55" xyaxes="-1 0 0 0 -0.4 0.75"/>
    <camera name="top" pos="0 0 1.2" xyaxes="1 0 0 0 1 0"/>

    <body name="table" pos="0 0 0.2">
      <geom type="box" size="0.3 0.3 0.02" material="table_mat"/>
      <geom type="cylinder" size="0.015 0.1" pos=" 0.25  0.25 -0.12"/>
      <geom type="cylinder" size="0.015 0.1" pos="-0.25  0.25 -0.12"/>
      <geom type="cylinder" size="0.015 0.1" pos=" 0.25 -0.25 -0.12"/>
      <geom type="cylinder" size="0.015 0.1" pos="-0.25 -0.25 -0.12"/>
    </body>

    <body name="cube" pos="0 0 0.25">
      <joint type="free"/>
      <geom type="box" size="0.025 0.025 0.025" mass="0.02" material="cube_mat"
            friction="2.0 0.1 0.001" condim="4" solref="0.01 1" solimp="0.95 0.99 0.001"/>
    </body>

    <body name="gripper_base" pos="0 0 0.55">
      <joint name="gripper_z" type="slide" axis="0 0 1" range="-0.35 0.1" damping="50"/>
      <geom type="cylinder" size="0.02 0.03" material="gripper_mat"/>

      <body name="finger_left" pos="0 0.04 -0.06">
        <joint name="finger_left" type="slide" axis="0 1 0" range="-0.02 0.015" damping="0.5"/>
        <geom type="box" size="0.012 0.012 0.04" material="gripper_mat"
              friction="2.0 0.1 0.001" condim="4"/>
      </body>

      <body name="finger_right" pos="0 -0.04 -0.06">
        <joint name="finger_right" type="slide" axis="0 1 0" range="-0.015 0.02" damping="0.5"/>
        <geom type="box" size="0.012 0.012 0.04" material="gripper_mat"
              friction="2.0 0.1 0.001" condim="4"/>
      </body>
    </body>
  </worldbody>

  <actuator>
    <position name="gripper_z_ctrl" joint="gripper_z" kp="200" ctrlrange="-0.35 0.1"/>
    <position name="finger_left_ctrl" joint="finger_left" kp="100" ctrlrange="-0.02 0.015"/>
    <position name="finger_right_ctrl" joint="finger_right" kp="100" ctrlrange="-0.015 0.02"/>
  </actuator>
</mujoco>
"""


# ---------------------------------------------------------------------------
# Grasp action sequence (same as mujoco_grasp.py)
# ---------------------------------------------------------------------------


def make_action_sequence(
    target_z: float, finger_left: float, finger_right: float, n_steps: int
) -> list[np.ndarray]:
    action = np.array([target_z, finger_left, finger_right])
    return [action for _ in range(n_steps)]


def build_grasp_phases() -> dict[str, list[np.ndarray]]:
    left_open, left_closed = 0.015, -0.02
    right_open, right_closed = -0.015, 0.02

    return {
        "pre_grasp": make_action_sequence(0.05, left_open, right_open, 500),
        "approach": make_action_sequence(-0.24, left_open, right_open, 500),
        "grasp": make_action_sequence(-0.24, left_closed, right_closed, 800),
        "lift": make_action_sequence(-0.10, left_closed, right_closed, 800),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Roboharness MuJoCo + Rerun Example")
    parser.add_argument("--output-dir", default="./harness_output", help="Output directory")
    parser.add_argument("--width", type=int, default=640, help="Render width")
    parser.add_argument("--height", type=int, default=480, help="Render height")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    cameras = ["front", "side", "top"]

    print("=" * 60)
    print("  Roboharness: MuJoCo + Rerun Example")
    print("=" * 60)

    # 1. Create MuJoCo backend
    print("\n[1/4] Loading MuJoCo model ...")
    backend = MuJoCoMeshcatBackend(
        xml_string=GRASP_MJCF,
        cameras=cameras,
        render_width=args.width,
        render_height=args.height,
    )

    # 2. Set up harness with Rerun enabled and semantic grasp protocol
    print("[2/4] Setting up harness with Rerun capture logging ...")
    harness = Harness(
        backend,
        output_dir=str(output_dir),
        task_name="mujoco_grasp",
        enable_rerun=True,
        rerun_app_id="roboharness",
    )
    phases = build_grasp_phases()
    harness.load_protocol(GRASP_PROTOCOL, phases=["pre_grasp", "approach", "grasp", "lift"])
    print(f"      Protocol: {harness.active_protocol.name}")
    print(f"      Checkpoints: {harness.list_checkpoints()}")

    # 3. Run the grasp sequence — captures are logged to .rrd automatically
    print("[3/4] Running grasp simulation (logging to .rrd) ...")
    harness.reset()

    for phase_name, actions in phases.items():
        result = harness.run_to_next_checkpoint(actions)
        if result is None:
            print(f"      WARNING: No checkpoint for phase '{phase_name}'")
            continue

        n_views = len(result.views)
        print(
            f"      Checkpoint '{phase_name}': {n_views} views"
            f" | step={result.step} | sim_time={result.sim_time:.3f}s"
        )

    # 4. Summary
    rrd_path = output_dir / "mujoco_grasp" / "trial_001" / "capture.rrd"
    print("\n[4/4] Done!")
    print(f"      Rerun recording: {rrd_path}")
    if rrd_path.exists():
        size_mb = rrd_path.stat().st_size / (1024 * 1024)
        print(f"      File size: {size_mb:.1f} MB")
    print("\n  View the recording:")
    print(f"    rerun {rrd_path}")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
