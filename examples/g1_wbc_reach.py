#!/usr/bin/env python3
"""G1 WBC Reach Example — Humanoid IK-based reaching with Roboharness.

Demonstrates the Controller protocol with a Unitree G1 humanoid robot:
  1. Load G1 model from MuJoCo Menagerie (via robot_descriptions)
  2. Use WbcIkController (Pinocchio + Pink) to solve arm IK
  3. Reach toward target positions with both arms
  4. Capture multi-view screenshots at each checkpoint

Run:
    pip install roboharness[mujoco,wbc] robot_descriptions Pillow
    python examples/g1_wbc_reach.py

Output:
    ./harness_output/g1_wbc_reach/trial_001/
        stand/       — initial standing pose
        reach_left/  — left arm reaching toward target
        reach_both/  — both arms reaching
        retract/     — arms retracted to rest
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Scene builder: G1 + table + target objects + cameras
# ---------------------------------------------------------------------------


def build_scene_xml() -> str:
    """Build MJCF XML for G1 reaching scene.

    Includes the menagerie G1, a ground plane, a table with target
    objects, and multiple cameras.
    """
    from robot_descriptions import g1_mj_description

    g1_xml_path = g1_mj_description.MJCF_PATH
    g1_dir = str(Path(g1_xml_path).parent)

    return f"""\
<mujoco model="g1_reach_scene">
  <include file="{g1_xml_path}"/>

  <option gravity="0 0 -9.81" timestep="0.002"/>
  <compiler meshdir="{g1_dir}/assets"/>

  <statistic center="0 0 0.8" extent="1.5"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0.5 0.5 0.5"/>
    <rgba haze="0.15 0.25 0.35 1"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.6 0.8 1.0" rgb2="0.2 0.3 0.5"
             width="256" height="256"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.85 0.85 0.85" rgb2="0.65 0.65 0.65"
             width="256" height="256"/>
    <material name="grid_mat" texture="grid" texrepeat="8 8" reflectance="0.1"/>
    <material name="table_mat" rgba="0.55 0.35 0.2 1"/>
    <material name="target_red" rgba="0.9 0.15 0.15 1"/>
    <material name="target_green" rgba="0.15 0.8 0.15 1"/>
  </asset>

  <worldbody>
    <geom name="floor" type="plane" size="3 3 0.05" material="grid_mat"/>
    <light pos="1 1 3" dir="-0.3 -0.3 -1" diffuse="0.7 0.7 0.7"/>
    <light pos="-1 1 3" dir="0.3 -0.3 -1" diffuse="0.4 0.4 0.4"/>

    <!-- Cameras -->
    <camera name="front" pos="2.8 0 1.0" xyaxes="0 1 0 -0.2 0 1"/>
    <camera name="side" pos="0 3.0 1.0" xyaxes="-1 0 0 0 -0.2 1"/>
    <camera name="top" pos="0 0 4.5" xyaxes="1 0 0 0 1 0"/>
    <camera name="close_up" pos="1.2 0.5 1.2" xyaxes="-0.4 1 0 -0.2 -0.1 1"/>

    <!-- Table in front of robot -->
    <body name="table" pos="0.45 0 0.35">
      <geom type="box" size="0.25 0.35 0.02" material="table_mat"/>
      <geom type="cylinder" size="0.02 0.17" pos=" 0.2  0.3 -0.19"/>
      <geom type="cylinder" size="0.02 0.17" pos="-0.2  0.3 -0.19"/>
      <geom type="cylinder" size="0.02 0.17" pos=" 0.2 -0.3 -0.19"/>
      <geom type="cylinder" size="0.02 0.17" pos="-0.2 -0.3 -0.19"/>
    </body>

    <!-- Target objects on table -->
    <body name="target_left" pos="0.4 0.15 0.40">
      <joint type="free"/>
      <geom type="sphere" size="0.03" mass="0.05" material="target_red"/>
    </body>
    <body name="target_right" pos="0.4 -0.15 0.40">
      <joint type="free"/>
      <geom type="sphere" size="0.03" mass="0.05" material="target_green"/>
    </body>
  </worldbody>
</mujoco>
"""


# ---------------------------------------------------------------------------
# Joint mapping: MuJoCo <-> Pinocchio
# ---------------------------------------------------------------------------

# Arm joint names (7 per arm)
LEFT_ARM_JOINTS = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
]
RIGHT_ARM_JOINTS = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


class JointMapper:
    """Handles joint mapping between MuJoCo and Pinocchio models.

    MuJoCo qpos includes the floating base (7 DOF) plus all joints including
    free bodies for objects.  Pinocchio URDF has only the 29 robot joints.
    This class maps by actuator/joint name so the exact qpos layout doesn't
    matter.
    """

    def __init__(self, mj_model) -> None:
        import mujoco

        self._mj = mujoco
        self._mj_model = mj_model

        # Build MuJoCo joint name → qpos index mapping (for 1-DOF hinge/slide joints)
        self._mj_joint_qpos: dict[str, int] = {}
        for i in range(mj_model.njnt):
            name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if name and mj_model.jnt_type[i] in (2, 3):  # hinge or slide
                self._mj_joint_qpos[name] = mj_model.jnt_qposadr[i]

        # Build MuJoCo actuator name → ctrl index mapping
        self._mj_act_ctrl: dict[str, int] = {}
        for i in range(mj_model.nu):
            name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if name:
                self._mj_act_ctrl[name] = i

        # Build Pinocchio joint name → q index mapping
        import pinocchio as pin
        from robot_descriptions import g1_description

        pin_model = pin.buildModelFromUrdf(g1_description.URDF_PATH)
        self._pin_joint_q: dict[str, int] = {}
        idx = 0
        for i in range(1, pin_model.njoints):
            self._pin_joint_q[pin_model.names[i]] = idx
            idx += pin_model.joints[i].nq

        # Ordered list of actuated joint names (shared between MuJoCo and Pinocchio)
        self.actuated_joints = [name for name in self._mj_act_ctrl if name in self._pin_joint_q]

    def mj_qpos_to_pin_q(self, mj_qpos: np.ndarray) -> np.ndarray:
        """Extract Pinocchio q vector from MuJoCo qpos by joint name matching."""
        pin_q = np.zeros(len(self._pin_joint_q))
        for name, pin_idx in self._pin_joint_q.items():
            mj_addr = self._mj_joint_qpos.get(name)
            if mj_addr is not None:
                pin_q[pin_idx] = mj_qpos[mj_addr]
        return pin_q

    def pin_q_to_ctrl(
        self,
        pin_q: np.ndarray,
        current_ctrl: np.ndarray,
        joint_names: list[str],
    ) -> np.ndarray:
        """Update MuJoCo ctrl with Pinocchio q values for specific joints."""
        ctrl = current_ctrl.copy()
        for name in joint_names:
            pin_idx = self._pin_joint_q.get(name)
            act_idx = self._mj_act_ctrl.get(name)
            if pin_idx is not None and act_idx is not None:
                ctrl[act_idx] = pin_q[pin_idx]
        return ctrl

    def standing_ctrl(self, mj_qpos: np.ndarray) -> np.ndarray:
        """Get ctrl array that holds all actuated joints at their current qpos."""
        ctrl = np.zeros(self._mj_model.nu)
        for name, act_idx in self._mj_act_ctrl.items():
            mj_addr = self._mj_joint_qpos.get(name)
            if mj_addr is not None:
                ctrl[act_idx] = mj_qpos[mj_addr]
        return ctrl


# ---------------------------------------------------------------------------
# IK reach targets
# ---------------------------------------------------------------------------


def make_target_pose(pos: np.ndarray, approach_axis: str = "-z") -> np.ndarray:
    """Create a 4x4 homogeneous transform for a target end-effector pose.

    The orientation is set so the hand approaches from the given axis.
    """
    T = np.eye(4)
    T[:3, 3] = pos
    if approach_axis == "-z":
        # Hand pointing down (palm facing down)
        T[:3, :3] = np.array(
            [
                [1, 0, 0],
                [0, -1, 0],
                [0, 0, -1],
            ],
            dtype=np.float64,
        )
    elif approach_axis == "-x":
        # Hand reaching forward
        T[:3, :3] = np.array(
            [
                [0, 0, -1],
                [0, -1, 0],
                [-1, 0, 0],
            ],
            dtype=np.float64,
        )
    return T


# ---------------------------------------------------------------------------
# Reach motion controller
# ---------------------------------------------------------------------------


def world_targets_to_pin(
    targets: dict[str, np.ndarray],
    base_pos: np.ndarray,
) -> dict[str, np.ndarray]:
    """Transform world-frame targets to Pinocchio base frame.

    Pinocchio uses a fixed-base model, so its origin is at the robot pelvis.
    MuJoCo uses a floating base with the pelvis at ``base_pos`` in world frame.
    We must subtract the base position from target translations.
    """
    pin_targets = {}
    for frame_name, T_world in targets.items():
        T_pin = T_world.copy()
        T_pin[:3, 3] = T_world[:3, 3] - base_pos
        pin_targets[frame_name] = T_pin
    return pin_targets


def run_ik_reach(
    controller,
    mapper: JointMapper,
    state: dict,
    targets: dict[str, np.ndarray],
    current_ctrl: np.ndarray,
    arm_joints: list[str],
    base_pos: np.ndarray | None = None,
) -> np.ndarray:
    """Run IK controller and return MuJoCo ctrl for the target pose.

    Parameters
    ----------
    base_pos : np.ndarray | None
        World-frame position of the robot base (pelvis).  When provided,
        targets are transformed from world frame to Pinocchio's fixed-base
        frame before solving IK.
    """
    pin_q = mapper.mj_qpos_to_pin_q(np.asarray(state["qpos"]))

    # Transform targets from world frame to Pinocchio base frame
    if base_pos is not None:
        targets = world_targets_to_pin(targets, base_pos)

    ik_result = controller.compute(
        command=targets,
        state={"qpos": pin_q},
    )
    return mapper.pin_q_to_ctrl(ik_result, current_ctrl, arm_joints)


# ---------------------------------------------------------------------------
# HTML report generator
# ---------------------------------------------------------------------------


def generate_html_report(output_dir: Path, task_name: str = "g1_wbc_reach") -> Path:
    """Generate a self-contained HTML report for G1 WBC reach demo."""
    from roboharness.reporting import generate_html_report as _generate

    return _generate(
        output_dir,
        task_name,
        title="Roboharness: G1 Humanoid WBC Reach",
        subtitle=(
            "Unitree G1 reaching targets using differential IK (Pinocchio + Pink). "
            "Controller Protocol demo with multi-view checkpoint captures."
        ),
        accent_color="#d94a4a",
        footer_text="Generated by <code>examples/g1_wbc_reach.py --report</code>",
        meshcat_mode="link",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="G1 WBC Reach Example")
    parser.add_argument("--output-dir", default="./harness_output")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument(
        "--assert-success",
        action="store_true",
        help="Validate reach success and exit non-zero on failure",
    )
    args = parser.parse_args()

    import mujoco

    from roboharness.backends.mujoco_meshcat import MuJoCoMeshcatBackend
    from roboharness.backends.visualizer import MeshcatVisualizer
    from roboharness.controllers.wbc_ik import WbcIkController, WbcIkSettings
    from roboharness.core.harness import Harness

    output_dir = Path(args.output_dir)
    cameras = ["front", "side", "top", "close_up"]

    print("=" * 60)
    print("  Roboharness: G1 Humanoid WBC Reach Example")
    print("=" * 60)

    # 1. Load scene
    print("\n[1/5] Building G1 scene ...")
    scene_xml = build_scene_xml()

    # When generating a report, use MeshcatVisualizer as the backend visualizer.
    # It produces real multi-view images (via MuJoCo renderer fallback) AND
    # provides interactive 3D HTML export — best of both worlds.
    viz_choice: str | None = None
    if args.report:
        try:
            import meshcat as _meshcat  # noqa: F401

            viz_choice = "meshcat"
        except ImportError:
            print("      Meshcat not installed — using native renderer only.")

    backend = MuJoCoMeshcatBackend(
        xml_string=scene_xml,
        cameras=cameras,
        render_width=args.width,
        render_height=args.height,
        visualizer=viz_choice,
    )
    mj_model = backend._model
    mj_data = backend._data
    print(f"      Model loaded. nq={mj_model.nq}, nu={mj_model.nu}")

    # Get the MeshcatVisualizer for HTML export (if active)
    meshcat_viz: MeshcatVisualizer | None = None
    if isinstance(backend.visualizer, MeshcatVisualizer):
        meshcat_viz = backend.visualizer
        print("      Meshcat visualizer active — multi-view capture + 3D export.")

    # 2. Set standing keyframe and build joint mapper
    print("[2/5] Setting standing pose ...")
    key_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(mj_model, mj_data, key_id)
        mujoco.mj_forward(mj_model, mj_data)

    mapper = JointMapper(mj_model)
    stand_ctrl = mapper.standing_ctrl(mj_data.qpos)
    np.copyto(mj_data.ctrl, stand_ctrl)
    backend._visualizer.sync()
    print("      Standing keyframe applied.")

    # 3. Create WBC IK controller
    print("[3/5] Initializing WBC IK controller ...")
    from robot_descriptions import g1_description

    ee_frames = ["left_rubber_hand", "right_rubber_hand"]
    settings = WbcIkSettings(
        dt=0.02,
        num_iterations=20,
        position_cost=20.0,
        orientation_cost=0.1,
        posture_cost=1e-3,
        damping=1e-4,
    )

    stand_q = mapper.mj_qpos_to_pin_q(mj_data.qpos)
    controller = WbcIkController(
        urdf_path=g1_description.URDF_PATH,
        end_effector_frames=ee_frames,
        settings=settings,
        reference_configuration=stand_q,
    )
    print(f"      IK controller ready. EE frames: {ee_frames}")

    # 4. Set up harness and run reach sequence
    print("[4/5] Running reach sequence ...")
    task_name = "g1_wbc_reach"
    harness = Harness(backend, output_dir=str(output_dir), task_name=task_name)

    phase_names = ["stand", "reach_left", "reach_both", "retract"]
    for name in phase_names:
        harness.add_checkpoint(name, cameras=cameras)

    harness.reset()
    # Re-apply standing keyframe after reset
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(mj_model, mj_data, key_id)
        mujoco.mj_forward(mj_model, mj_data)
    np.copyto(mj_data.ctrl, stand_ctrl)
    backend._visualizer.sync()

    all_arm_joints = LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS

    # Extract floating-base position for world→Pinocchio frame transform.
    # MuJoCo qpos[0:3] is the pelvis world position for a floating-base robot.
    base_pos = np.array(mj_data.qpos[0:3])
    print(f"      Robot base (pelvis) at world position: {base_pos}")

    def _run_phase(actions, phase_name):
        result = harness.run_to_next_checkpoint(actions)
        _print_checkpoint(result)
        if meshcat_viz is not None and result is not None:
            meshcat_viz.sync()
            trial_dir = output_dir / task_name / "trial_001" / phase_name
            scene_path = trial_dir / "meshcat_scene.html"
            meshcat_viz.export_html(scene_path)
            print(f"        -> Meshcat 3D: {scene_path}")
        return result

    checkpoint_results: dict[str, object] = {}

    # -- Phase 1: Stand (capture initial pose)
    checkpoint_results["stand"] = _run_phase([stand_ctrl.copy() for _ in range(200)], "stand")

    # -- Phase 2: Left arm reach toward target
    # Targets are in world frame; run_ik_reach transforms them to Pinocchio base frame.
    # Target spheres are at world [0.4, 0.15, 0.40] and [0.4, -0.15, 0.40].
    # Aim slightly above them for a natural hovering/approach pose.
    state = harness.get_state()
    left_target_pos = np.array([0.38, 0.15, 0.48])
    left_target = make_target_pose(left_target_pos, approach_axis="-z")
    ctrl = run_ik_reach(
        controller,
        mapper,
        state,
        {"left_rubber_hand": left_target},
        mj_data.ctrl,
        all_arm_joints,
        base_pos=base_pos,
    )
    checkpoint_results["reach_left"] = _run_phase([ctrl for _ in range(600)], "reach_left")

    # -- Phase 3: Both arms reach
    state = harness.get_state()
    right_target_pos = np.array([0.38, -0.15, 0.48])
    right_target = make_target_pose(right_target_pos, approach_axis="-z")
    ctrl = run_ik_reach(
        controller,
        mapper,
        state,
        {"left_rubber_hand": left_target, "right_rubber_hand": right_target},
        mj_data.ctrl,
        all_arm_joints,
        base_pos=base_pos,
    )
    checkpoint_results["reach_both"] = _run_phase([ctrl for _ in range(600)], "reach_both")

    # -- Phase 4: Retract to standing
    checkpoint_results["retract"] = _run_phase([stand_ctrl.copy() for _ in range(600)], "retract")

    # 5. Summary
    print("\n[5/5] Done!")
    trial_dir = output_dir / task_name / "trial_001"
    total_images = len(list(trial_dir.rglob("*_rgb.png"))) if trial_dir.exists() else 0
    print(f"      {total_images} images saved to: {trial_dir}")

    if args.report:
        report_path = generate_html_report(output_dir, task_name)
        print(f"      HTML report: {report_path}")

    print("\n  Output structure:")
    if trial_dir.exists():
        for cp_dir in sorted(trial_dir.iterdir()):
            if cp_dir.is_dir():
                files = sorted(f.name for f in cp_dir.iterdir() if f.is_file())
                print(f"    {cp_dir.name}/")
                for fname in files:
                    print(f"      {fname}")

    print("\n" + "=" * 60)

    # 6. Assert success (optional CI gate)
    if args.assert_success:
        reach_targets = {
            "reach_left": {"left_rubber_hand": left_target_pos},
            "reach_both": {
                "left_rubber_hand": left_target_pos,
                "right_rubber_hand": right_target_pos,
            },
        }
        failures = assert_reach_success(checkpoint_results, backend, reach_targets)
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


def _print_checkpoint(result) -> None:
    if result:
        print(
            f"      Checkpoint '{result.checkpoint_name}': "
            f"{len(result.views)} views | step={result.step} | "
            f"sim_time={result.sim_time:.3f}s"
        )


# ---------------------------------------------------------------------------
# Success assertion helpers
# ---------------------------------------------------------------------------

# G1 floating base z is qpos[2]; if it drops below this, the robot fell
MIN_BASE_HEIGHT = 0.4  # meters (G1 standing is ~0.75m)
# End-effector must be within this distance of the target position
EE_DISTANCE_THRESHOLD = 0.15  # meters
QVEL_MAX = 50.0  # maximum acceptable qvel magnitude (stability check)


def assert_reach_success(
    checkpoint_results: dict[str, object],
    backend: object,
    reach_targets: dict[str, dict[str, np.ndarray]],
) -> list[str]:
    """Validate reach task success. Returns a list of failure messages (empty = pass)."""
    import mujoco

    failures: list[str] = []
    mj_model = backend._model
    mj_data = backend._data

    # --- Check 1: End-effector proximity to targets ---
    for phase_name, targets in reach_targets.items():
        result = checkpoint_results.get(phase_name)
        if result is None:
            failures.append(f"'{phase_name}' checkpoint not reached")
            continue

        for ee_name, target_pos in targets.items():
            # Get the site/body position for the end-effector
            site_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, ee_name)
            if site_id >= 0:
                # Restore state to this checkpoint, then read site_xpos
                qpos = np.array(result.state["qpos"])
                qvel = np.array(result.state["qvel"])
                np.copyto(mj_data.qpos, qpos)
                np.copyto(mj_data.qvel, qvel)
                mujoco.mj_forward(mj_model, mj_data)
                ee_pos = mj_data.site_xpos[site_id].copy()
            else:
                body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, ee_name)
                if body_id < 0:
                    failures.append(f"EE frame '{ee_name}' not found in model")
                    continue
                qpos = np.array(result.state["qpos"])
                qvel = np.array(result.state["qvel"])
                np.copyto(mj_data.qpos, qpos)
                np.copyto(mj_data.qvel, qvel)
                mujoco.mj_forward(mj_model, mj_data)
                ee_pos = mj_data.xpos[body_id].copy()

            dist = float(np.linalg.norm(ee_pos - target_pos))
            if dist > EE_DISTANCE_THRESHOLD:
                failures.append(
                    f"{ee_name} at '{phase_name}': distance={dist:.4f}m to target, "
                    f"expected <{EE_DISTANCE_THRESHOLD}m"
                )
            else:
                print(f"  CHECK: {ee_name} at '{phase_name}' (dist={dist:.4f}m) — OK")

    # --- Check 2: Robot stays upright ---
    for phase_name, result in checkpoint_results.items():
        if result is None:
            continue
        # G1 floating base: qpos[0:3] = xyz position
        base_z = float(result.state["qpos"][2])
        if base_z < MIN_BASE_HEIGHT:
            failures.append(
                f"robot fell at '{phase_name}': base z={base_z:.3f}m, min={MIN_BASE_HEIGHT}m"
            )
        else:
            print(f"  CHECK: upright at '{phase_name}' (base z={base_z:.3f}m) — OK")

    # --- Check 3: Simulation stability (qvel within bounds) ---
    for phase_name, result in checkpoint_results.items():
        if result is None:
            continue
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
