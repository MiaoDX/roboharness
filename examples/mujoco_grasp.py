#!/usr/bin/env python3
"""MuJoCo Grasp Example — End-to-end demo of Roboharness.

A minimal, self-contained example that demonstrates the full harness workflow:
  1. Load a MuJoCo model (inline XML, no external files needed)
  2. Run a scripted grasp sequence
  3. Capture multi-view screenshots at each checkpoint
  4. Save everything to disk

Run:
    pip install roboharness[demo]
    python examples/mujoco_grasp.py

Output:
    ./harness_output/mujoco_grasp/trial_001/
        pre_grasp/   — gripper open, above the cube
        contact/     — gripper lowered onto the cube
        lift/        — cube lifted off the table
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from roboharness.backends.mujoco_meshcat import MuJoCoMeshcatBackend
from roboharness.backends.visualizer import MeshcatVisualizer
from roboharness.core.harness import Harness

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
    <!-- Ground -->
    <geom type="plane" size="1 1 0.01" material="grid_mat"/>
    <light pos="0 0 2" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <light pos="0.5 0.5 1.5" dir="-0.3 -0.3 -1" diffuse="0.4 0.4 0.4"/>

    <!-- Cameras -->
    <camera name="front" pos="0.75 0 0.55" xyaxes="0 1 0 -0.4 0 0.75"/>
    <camera name="side" pos="0 0.75 0.55" xyaxes="-1 0 0 0 -0.4 0.75"/>
    <camera name="top" pos="0 0 1.2" xyaxes="1 0 0 0 1 0"/>

    <!-- Table -->
    <body name="table" pos="0 0 0.2">
      <geom type="box" size="0.3 0.3 0.02" material="table_mat"/>
      <geom type="cylinder" size="0.015 0.1" pos=" 0.25  0.25 -0.12"/>
      <geom type="cylinder" size="0.015 0.1" pos="-0.25  0.25 -0.12"/>
      <geom type="cylinder" size="0.015 0.1" pos=" 0.25 -0.25 -0.12"/>
      <geom type="cylinder" size="0.015 0.1" pos="-0.25 -0.25 -0.12"/>
    </body>

    <!-- Cube (free body, graspable) -->
    <body name="cube" pos="0 0 0.25">
      <joint type="free"/>
      <geom type="box" size="0.025 0.025 0.025" mass="0.02" material="cube_mat"
            friction="2.0 0.1 0.001" condim="4" solref="0.01 1" solimp="0.95 0.99 0.001"/>
    </body>

    <!-- Gripper -->
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
# Grasp action sequence
# ---------------------------------------------------------------------------
# Controls: [gripper_z_position, finger_left, finger_right]
# Positive finger_left = open left, Positive finger_right = open right


def make_action_sequence(
    target_z: float, finger_left: float, finger_right: float, n_steps: int
) -> list[np.ndarray]:
    """Create a constant-action sequence for n_steps.

    Controls: [gripper_z, finger_left, finger_right]
      finger_left  range [-0.03, 0.01]: -0.03 = closed, 0.01 = open
      finger_right range [-0.01, 0.03]:  0.03 = closed, -0.01 = open
    """
    action = np.array([target_z, finger_left, finger_right])
    return [action for _ in range(n_steps)]


def build_grasp_phases() -> dict[str, list[np.ndarray]]:
    """Build the scripted grasp motion in phases.

    Returns a dict mapping phase_name -> action_sequence.
    """
    # Finger positions (asymmetric joint ranges)
    left_open, left_closed = 0.015, -0.02
    right_open, right_closed = -0.015, 0.02

    return {
        # Phase 1: Open gripper, hover above cube
        "pre_grasp": make_action_sequence(
            target_z=0.05,
            finger_left=left_open,
            finger_right=right_open,
            n_steps=500,
        ),
        # Phase 2: Lower onto cube, fingers still open
        "contact": make_action_sequence(
            target_z=-0.24,
            finger_left=left_open,
            finger_right=right_open,
            n_steps=500,
        ),
        # Phase 3: Close fingers around cube (stay at contact height)
        "grasp": make_action_sequence(
            target_z=-0.24,
            finger_left=left_closed,
            finger_right=right_closed,
            n_steps=800,
        ),
        # Phase 4: Lift with fingers closed
        "lift": make_action_sequence(
            target_z=-0.10,
            finger_left=left_closed,
            finger_right=right_closed,
            n_steps=800,
        ),
    }


# ---------------------------------------------------------------------------
# HTML report generator
# ---------------------------------------------------------------------------


def generate_html_report(output_dir: Path) -> Path:
    """Generate a self-contained HTML report showing all checkpoint captures."""
    from roboharness.reporting import generate_html_report as _generate

    return _generate(
        output_dir,
        "mujoco_grasp",
        title="Roboharness: MuJoCo Grasp Example",
        subtitle="Visual checkpoint captures from the scripted grasp sequence.",
        accent_color="#4a90d9",
        footer_text="Generated by <code>examples/mujoco_grasp.py --report</code>",
        meshcat_mode="iframe",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Roboharness MuJoCo Grasp Example")
    parser.add_argument(
        "--output-dir",
        default="./harness_output",
        help="Output directory (default: ./harness_output)",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate an HTML report after the run",
    )
    parser.add_argument("--width", type=int, default=640, help="Render width (default: 640)")
    parser.add_argument("--height", type=int, default=480, help="Render height (default: 480)")
    parser.add_argument(
        "--assert-success",
        action="store_true",
        help="Validate grasp success and exit non-zero on failure",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    cameras = ["front", "side", "top"]

    print("=" * 60)
    print("  Roboharness: MuJoCo Grasp Example")
    print("=" * 60)

    # 1. Create backend from inline XML
    print("\n[1/4] Loading MuJoCo model (inline XML) ...")
    backend = MuJoCoMeshcatBackend(
        xml_string=GRASP_MJCF,
        cameras=cameras,
        render_width=args.width,
        render_height=args.height,
    )
    print("      Model loaded. Actuators: 3 (z-slide, left-finger, right-finger)")

    # Create Meshcat visualizer for interactive 3D export (if meshcat available)
    meshcat_viz: MeshcatVisualizer | None = None
    if args.report:
        try:
            import mujoco as _mj  # noqa: F401

            meshcat_viz = MeshcatVisualizer(
                backend._model,
                backend._data,
                width=args.width,
                height=args.height,
            )
            print("      Meshcat visualizer ready for 3D scene export.")
        except ImportError:
            print("      Meshcat not installed — skipping interactive 3D export.")

    # 2. Set up harness with checkpoints
    print("[2/4] Setting up harness with checkpoints ...")
    harness = Harness(backend, output_dir=str(output_dir), task_name="mujoco_grasp")
    phases = build_grasp_phases()
    for phase_name in phases:
        harness.add_checkpoint(phase_name, cameras=cameras)
    print(f"      Checkpoints: {harness.list_checkpoints()}")

    # 3. Run the grasp sequence
    print("[3/4] Running grasp simulation ...")
    harness.reset()

    checkpoint_results: dict[str, object] = {}
    for phase_name, actions in phases.items():
        result = harness.run_to_next_checkpoint(actions)
        if result is None:
            print(f"      WARNING: No checkpoint for phase '{phase_name}'")
            continue

        checkpoint_results[phase_name] = result
        n_views = len(result.views)
        trial_dir = output_dir / "mujoco_grasp" / "trial_001" / phase_name
        print(
            f"      Checkpoint '{phase_name}': {n_views} views captured"
            f" | step={result.step} | sim_time={result.sim_time:.3f}s"
        )
        print(f"        -> {trial_dir}")

        # Export Meshcat interactive scene for this checkpoint
        if meshcat_viz is not None:
            meshcat_viz.sync()
            scene_path = trial_dir / "meshcat_scene.html"
            meshcat_viz.export_html(scene_path)
            print(f"        -> Meshcat 3D: {scene_path}")

    # 4. Summary
    print("\n[4/4] Done!")
    trial_dir = output_dir / "mujoco_grasp" / "trial_001"
    total_images = len(list(trial_dir.rglob("*_rgb.png"))) if trial_dir.exists() else 0
    print(f"      {total_images} images saved to: {trial_dir}")

    if args.report:
        report_path = generate_html_report(output_dir)
        print(f"      HTML report: {report_path}")

    # Print tree-like summary
    print("\n  Output structure:")
    if trial_dir.exists():
        for cp_dir in sorted(trial_dir.iterdir()):
            if cp_dir.is_dir():
                files = sorted(f.name for f in cp_dir.iterdir() if f.is_file())
                print(f"    {cp_dir.name}/")
                for fname in files:
                    print(f"      {fname}")

    print("\n" + "=" * 60)

    # 5. Assert success (optional CI gate)
    if args.assert_success:
        failures = assert_grasp_success(checkpoint_results, backend)
        if failures:
            print("\n" + "=" * 60)
            print("  ASSERT-SUCCESS: FAILED")
            print("=" * 60)
            for msg in failures:
                print(f"  FAIL: {msg}")
            sys.exit(1)
        else:
            print("\n" + "=" * 60)
            print("  ASSERT-SUCCESS: PASSED")
            print("=" * 60)

    return


# ---------------------------------------------------------------------------
# Success assertion helpers
# ---------------------------------------------------------------------------

# Table top z = table body z (0.2) + table half-height (0.02) = 0.22
# Cube half-size = 0.025, so cube center at rest on table = 0.245
TABLE_SURFACE_Z = 0.22
CUBE_LIFT_THRESHOLD = 0.005  # cube center must be >5mm above table surface
QVEL_MAX = 50.0  # maximum acceptable qvel magnitude (stability check)


def _get_cube_z(state: dict[str, object]) -> float:
    """Extract cube center z-position from simulator state.

    MuJoCo orders free joints before slide joints in qpos. In this model:
    qpos[0:3] = cube position (x, y, z), qpos[3:7] = cube quaternion,
    qpos[7] = gripper_z, qpos[8] = finger_left, qpos[9] = finger_right.
    So cube z = qpos[2].
    """
    qpos = state["qpos"]
    return float(qpos[2])


def assert_grasp_success(
    checkpoint_results: dict[str, object],
    backend: MuJoCoMeshcatBackend,
) -> list[str]:
    """Validate physical task success. Returns a list of failure messages (empty = pass)."""
    failures: list[str] = []

    # --- Check 1: Cube lifted above table at "lift" checkpoint ---
    lift_result = checkpoint_results.get("lift")
    if lift_result is None:
        failures.append("'lift' checkpoint not reached")
    else:
        cube_z = _get_cube_z(lift_result.state)
        min_z = TABLE_SURFACE_Z + CUBE_LIFT_THRESHOLD
        if cube_z <= min_z:
            failures.append(
                f"cube z={cube_z:.4f}m at lift, expected >{min_z:.4f}m "
                f"(>{CUBE_LIFT_THRESHOLD * 1000:.0f}mm above table)"
            )
        else:
            height_above = cube_z - TABLE_SURFACE_Z
            print(f"  CHECK: cube z={cube_z:.4f}m ({height_above * 1000:.1f}mm above table) — OK")

    # --- Check 2: Gripper has contact with cube at "lift" checkpoint ---
    if lift_result is not None:
        import mujoco

        model = backend._model
        data = backend._data
        cube_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        finger_left_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "finger_left")
        finger_right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "finger_right")

        # Map geom ids to their parent body ids
        has_contact = False
        for i in range(data.ncon):
            c = data.contact[i]
            body1 = model.geom_bodyid[c.geom1]
            body2 = model.geom_bodyid[c.geom2]
            bodies = {int(body1), int(body2)}
            if cube_body_id in bodies and (finger_left_id in bodies or finger_right_id in bodies):
                has_contact = True
                break

        if not has_contact:
            failures.append("no gripper-cube contact detected at lift checkpoint")
        else:
            print("  CHECK: gripper-cube contact at lift — OK")

    # --- Check 3: Simulation stability (qvel within bounds) ---
    for phase_name, result in checkpoint_results.items():
        qvel = np.array(result.state["qvel"])
        max_vel = float(np.max(np.abs(qvel)))
        if max_vel > QVEL_MAX:
            failures.append(
                f"unstable simulation at '{phase_name}': max |qvel|={max_vel:.2f}, limit={QVEL_MAX}"
            )
        else:
            print(f"  CHECK: stability at '{phase_name}' (max |qvel|={max_vel:.2f}) — OK")

    return failures


if __name__ == "__main__":
    main()
