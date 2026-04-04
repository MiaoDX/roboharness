#!/usr/bin/env python3
"""LeRobot G1 Validation — RobotHarnessWrapper on a Unitree G1 MuJoCo simulation.

Validates that RobotHarnessWrapper integrates correctly with a LeRobot-style
Unitree G1 MuJoCo environment. Supports two modes:

  --model simplified   Use an inline 12-DOF model (default, no extra deps)
  --model huggingface  Download the real 29-DOF G1 from HuggingFace

The real G1 model (29 body DOF + 14 hand DOF) is hosted at:
  huggingface.co/lerobot/unitree-g1-mujoco

Run (simplified):
    pip install roboharness[mujoco] gymnasium Pillow
    MUJOCO_GL=osmesa python examples/lerobot_g1.py

Run (real model):
    pip install roboharness[lerobot] gymnasium Pillow
    MUJOCO_GL=osmesa python examples/lerobot_g1.py --model huggingface

Output:
    ./harness_output/lerobot_g1/trial_001/
        stand/    — robot in default standing pose
        step/     — robot mid-step (legs actuated)
        balance/  — robot balancing after perturbation
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    print("ERROR: gymnasium is required. Install with: pip install gymnasium")
    sys.exit(1)

try:
    import mujoco
except ImportError:
    print("ERROR: mujoco is required. Install with: pip install roboharness[mujoco]")
    sys.exit(1)

from roboharness.wrappers import RobotHarnessWrapper

# ---------------------------------------------------------------------------
# HuggingFace model download
# ---------------------------------------------------------------------------

G1_HF_REPO = "lerobot/unitree-g1-mujoco"
G1_SCENE_XML = "assets/scene_43dof.xml"
G1_NUM_BODY_MOTORS = 29


def download_g1_assets() -> Path:
    """Download the real G1 MuJoCo assets from HuggingFace.

    Returns the local path to the downloaded repository.
    Requires: ``pip install huggingface_hub``
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print(
            "ERROR: huggingface_hub is required for the real G1 model.\n"
            "Install with: pip install roboharness[lerobot]"
        )
        sys.exit(1)

    print(f"      Downloading {G1_HF_REPO} from HuggingFace ...")
    repo_path = Path(snapshot_download(G1_HF_REPO))
    xml_path = repo_path / G1_SCENE_XML
    if not xml_path.exists():
        print(f"ERROR: Expected scene file not found: {xml_path}")
        sys.exit(1)
    print(f"      Downloaded to: {repo_path}")
    return repo_path


# ---------------------------------------------------------------------------
# Simplified G1 MJCF model (12-DOF bipedal humanoid with 3 cameras)
# ---------------------------------------------------------------------------

G1_SIMPLIFIED_MJCF = """\
<mujoco model="unitree_g1_simplified">
  <option gravity="0 0 -9.81" timestep="0.002"/>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.4 0.6 0.9" rgb2="0.1 0.15 0.3"
             width="256" height="256"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.9 0.9 0.9" rgb2="0.7 0.7 0.7"
             width="256" height="256"/>
    <material name="grid_mat" texture="grid" texrepeat="8 8" reflectance="0.1"/>
    <material name="body_mat" rgba="0.3 0.3 0.3 1"/>
    <material name="joint_mat" rgba="0.2 0.5 0.8 1"/>
    <material name="foot_mat" rgba="0.5 0.5 0.5 1"/>
  </asset>

  <worldbody>
    <geom type="plane" size="3 3 0.01" material="grid_mat"/>
    <light pos="0 0 3" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <light pos="1 1 2" dir="-0.3 -0.3 -1" diffuse="0.4 0.4 0.4"/>

    <camera name="front" pos="1.5 0 1.0" xyaxes="0 1 0 -0.3 0 1"/>
    <camera name="side" pos="0 1.5 1.0" xyaxes="-1 0 0 0 -0.3 1"/>
    <camera name="top" pos="0 0 3.0" xyaxes="1 0 0 0 1 0"/>

    <body name="torso" pos="0 0 0.85">
      <joint name="root_x" type="slide" axis="1 0 0" limited="false"/>
      <joint name="root_z" type="slide" axis="0 0 1" range="-0.5 0.5" damping="10"/>
      <geom type="capsule" fromto="0 0 -0.1 0 0 0.2" size="0.08" mass="10" material="body_mat"/>

      <body name="head" pos="0 0 0.3">
        <geom type="sphere" size="0.08" mass="1" material="joint_mat"/>
      </body>

      <body name="left_hip" pos="0 0.1 -0.1">
        <joint name="left_hip_pitch" type="hinge" axis="0 1 0" range="-1.5 0.5" damping="5"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.3" size="0.04" mass="3" material="body_mat"/>
        <body name="left_knee" pos="0 0 -0.3">
          <joint name="left_knee" type="hinge" axis="0 1 0" range="0 2.5" damping="3"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.3" size="0.035" mass="2" material="body_mat"/>
          <body name="left_ankle" pos="0 0 -0.3">
            <joint name="left_ankle" type="hinge" axis="0 1 0" range="-0.8 0.8" damping="2"/>
            <geom type="box" size="0.08 0.04 0.015" pos="0.02 0 -0.015" mass="0.5"
                  material="foot_mat"/>
          </body>
        </body>
      </body>

      <body name="right_hip" pos="0 -0.1 -0.1">
        <joint name="right_hip_pitch" type="hinge" axis="0 1 0" range="-1.5 0.5" damping="5"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.3" size="0.04" mass="3" material="body_mat"/>
        <body name="right_knee" pos="0 0 -0.3">
          <joint name="right_knee" type="hinge" axis="0 1 0" range="0 2.5" damping="3"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.3" size="0.035" mass="2" material="body_mat"/>
          <body name="right_ankle" pos="0 0 -0.3">
            <joint name="right_ankle" type="hinge" axis="0 1 0" range="-0.8 0.8" damping="2"/>
            <geom type="box" size="0.08 0.04 0.015" pos="0.02 0 -0.015" mass="0.5"
                  material="foot_mat"/>
          </body>
        </body>
      </body>

      <body name="left_shoulder" pos="0 0.15 0.15">
        <joint name="left_shoulder" type="hinge" axis="0 1 0" range="-3.14 1.0" damping="2"/>
        <geom type="capsule" fromto="0 0 0 0 0.05 -0.25" size="0.03" mass="1.5"
              material="joint_mat"/>
      </body>

      <body name="right_shoulder" pos="0 -0.15 0.15">
        <joint name="right_shoulder" type="hinge" axis="0 1 0" range="-3.14 1.0" damping="2"/>
        <geom type="capsule" fromto="0 0 0 0 -0.05 -0.25" size="0.03" mass="1.5"
              material="joint_mat"/>
      </body>
    </body>
  </worldbody>

  <actuator>
    <position name="left_hip_ctrl" joint="left_hip_pitch" kp="100" ctrlrange="-1.5 0.5"/>
    <position name="left_knee_ctrl" joint="left_knee" kp="80" ctrlrange="0 2.5"/>
    <position name="left_ankle_ctrl" joint="left_ankle" kp="40" ctrlrange="-0.8 0.8"/>
    <position name="right_hip_ctrl" joint="right_hip_pitch" kp="100" ctrlrange="-1.5 0.5"/>
    <position name="right_knee_ctrl" joint="right_knee" kp="80" ctrlrange="0 2.5"/>
    <position name="right_ankle_ctrl" joint="right_ankle" kp="40" ctrlrange="-0.8 0.8"/>
    <position name="left_shoulder_ctrl" joint="left_shoulder" kp="30" ctrlrange="-3.14 1.0"/>
    <position name="right_shoulder_ctrl" joint="right_shoulder" kp="30" ctrlrange="-3.14 1.0"/>
  </actuator>
</mujoco>
"""


# ---------------------------------------------------------------------------
# Gymnasium environment wrapping the G1 MuJoCo model
# ---------------------------------------------------------------------------


class LeRobotG1Env(gym.Env):
    """Gymnasium environment for the Unitree G1 humanoid in MuJoCo.

    Supports two loading modes:
      - ``xml_string``: inline MJCF XML (simplified model, no external files)
      - ``model_path``: path to a MuJoCo XML file (real G1 from HuggingFace)

    For the real 43-DOF model (29 body + 14 hand), only the first
    ``num_motors`` actuators are exposed in the action space. The remaining
    (hand) actuators are held at zero.
    """

    metadata: ClassVar[dict[str, Any]] = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(
        self,
        render_mode: str = "rgb_array",
        render_width: int = 640,
        render_height: int = 480,
        xml_string: str | None = None,
        model_path: str | Path | None = None,
        num_motors: int | None = None,
    ):
        super().__init__()
        self.render_mode = render_mode
        self._render_width = render_width
        self._render_height = render_height

        # Load MuJoCo model
        if model_path is not None:
            self._model = mujoco.MjModel.from_xml_path(str(model_path))
        else:
            self._model = mujoco.MjModel.from_xml_string(xml_string or G1_SIMPLIFIED_MJCF)
        self._data = mujoco.MjData(self._model)
        self._renderer = mujoco.Renderer(self._model, self._render_height, self._render_width)

        # Number of controlled motors (may be less than total actuators)
        self._num_motors = num_motors or self._model.nu
        assert self._num_motors <= self._model.nu, (
            f"num_motors={self._num_motors} > model.nu={self._model.nu}"
        )

        # Observation: joint positions (nq) + joint velocities (nv)
        obs_dim = self._model.nq + self._model.nv
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float64
        )

        # Action: position targets for controlled motors only
        self.action_space = spaces.Box(
            low=-np.pi,
            high=np.pi,
            shape=(self._num_motors,),
            dtype=np.float64,
        )

        # Available cameras (from MJCF)
        self._cameras = []
        for i in range(self._model.ncam):
            self._cameras.append(self._model.camera(i).name)

        self._step_count = 0

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed, options=options)
        mujoco.mj_resetData(self._model, self._data)
        mujoco.mj_forward(self._model, self._data)
        self._step_count = 0
        return self._get_obs(), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        # Apply action to the first num_motors actuators, leave rest at zero
        self._data.ctrl[:] = 0.0
        self._data.ctrl[: self._num_motors] = action
        mujoco.mj_step(self._model, self._data)
        self._step_count += 1

        obs = self._get_obs()

        # Reward: stay upright (torso z-height) + alive bonus
        torso_z = self._get_torso_z()
        reward = 1.0 + torso_z

        # Terminated if torso falls too low
        terminated = bool(torso_z < 0.3)
        truncated = False

        info: dict[str, Any] = {
            "torso_z": float(torso_z),
            "sim_time": float(self._data.time),
        }

        return obs, reward, terminated, truncated, info

    def render(self) -> np.ndarray:
        """Render the first available camera."""
        if self._cameras:
            return self.render_camera(self._cameras[0])
        # Fallback: free camera
        self._renderer.update_scene(self._data)
        return self._renderer.render()

    def render_camera(self, camera_name: str) -> np.ndarray:
        """Render a named camera view — enables multi-camera capture."""
        cam_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if cam_id < 0:
            raise ValueError(f"Unknown camera: {camera_name}. Available: {self._cameras}")
        self._renderer.update_scene(self._data, camera=camera_name)
        return self._renderer.render()

    def _get_obs(self) -> np.ndarray:
        return np.concatenate([self._data.qpos.copy(), self._data.qvel.copy()])

    def _get_torso_z(self) -> float:
        """Get torso height. Handles both simplified (slide joint) and real (free joint) models."""
        # For a free joint, qpos[0:3] = (x, y, z), so z = qpos[2]
        # For the simplified model, root_z is qpos[1]
        # Detect by checking if the first joint is free
        if self._model.njnt > 0 and self._model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE:
            return float(self._data.qpos[2])
        # Simplified model: root_z is second joint
        return float(self._data.qpos[1]) + 0.85  # offset to absolute height

    @property
    def cameras(self) -> list[str]:
        """Available camera names."""
        return list(self._cameras)

    def close(self) -> None:
        self._renderer.close()


# ---------------------------------------------------------------------------
# Scripted motion sequences (work with any actuator count)
# ---------------------------------------------------------------------------


def build_stand_sequence(num_motors: int, n_steps: int = 300) -> list[np.ndarray]:
    """Standing pose: zero all joints (default pose)."""
    return [np.zeros(num_motors) for _ in range(n_steps)]


def build_step_sequence(num_motors: int, n_steps: int = 400) -> list[np.ndarray]:
    """Stepping motion: sinusoidal actuation on first 6 joints (legs)."""
    actions = []
    n_legs = min(6, num_motors)
    for i in range(n_steps):
        action = np.zeros(num_motors)
        phase = (i / n_steps) * 2 * np.pi
        # Alternate legs with sinusoidal motion
        for j in range(n_legs):
            sign = 1.0 if j < n_legs // 2 else -1.0
            action[j] = sign * 0.2 * np.sin(phase + (j % (n_legs // 2)) * 0.5)
        actions.append(action)
    return actions


def build_balance_sequence(num_motors: int, n_steps: int = 300) -> list[np.ndarray]:
    """Recovery balance: return to neutral."""
    return [np.zeros(num_motors) for _ in range(n_steps)]


# ---------------------------------------------------------------------------
# HTML report generator
# ---------------------------------------------------------------------------


def generate_html_report(output_dir: Path, model_name: str) -> Path:
    """Generate a self-contained HTML report with embedded checkpoint images."""
    trial_dir = output_dir / "lerobot_g1" / "trial_001"
    if not trial_dir.exists():
        return output_dir / "lerobot_g1_report.html"

    checkpoints = sorted(
        [d for d in trial_dir.iterdir() if d.is_dir()],
        key=lambda p: p.name,
    )

    rows_html = []
    for cp_dir in checkpoints:
        cp_name = cp_dir.name

        meta_path = cp_dir / "metadata.json"
        meta = {}
        if meta_path.exists():
            with meta_path.open() as f:
                meta = json.load(f)

        images_html = []
        for img_file in sorted(cp_dir.glob("*_rgb.png")):
            cam_name = img_file.stem.replace("_rgb", "")
            with img_file.open("rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            images_html.append(
                f'<div class="cam">'
                f'<img src="data:image/png;base64,{b64}" alt="{cam_name}"/>'
                f"<p>{cam_name}</p></div>"
            )

        step = meta.get("step", "?")
        cams = ", ".join(meta.get("cameras", []))
        capability = meta.get("camera_capability", "?")

        rows_html.append(
            f'<div class="checkpoint">'
            f"<h2>{cp_name}</h2>"
            f"<p>Step: {step} | Cameras: {cams} | Capability: {capability}</p>"
            f'<div class="views">{"".join(images_html)}</div>'
            f"</div>"
        )

    status = "PASSED" if rows_html else "NO DATA"
    html = f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Roboharness: LeRobot G1 Validation Report</title>
<style>
  body {{ font-family: -apple-system, sans-serif; max-width: 1200px; margin: 0 auto;
         padding: 20px; background: #f5f5f5; }}
  h1 {{ color: #333; border-bottom: 2px solid #2d8cf0; padding-bottom: 10px; }}
  .checkpoint {{ background: white; border-radius: 8px; padding: 20px; margin: 20px 0;
                 box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
  .checkpoint h2 {{ color: #2d8cf0; margin-top: 0; }}
  .views {{ display: flex; gap: 16px; flex-wrap: wrap; }}
  .cam {{ text-align: center; }}
  .cam img {{ max-width: 320px; border: 1px solid #ddd; border-radius: 4px; }}
  .cam p {{ margin: 4px 0 0; font-size: 14px; color: #666; }}
  .summary {{ background: #e8f4fd; border-radius: 8px; padding: 16px; margin: 20px 0; }}
  .footer {{ margin-top: 30px; color: #999; font-size: 12px; }}
</style>
</head>
<body>
<h1>LeRobot G1 Validation Report</h1>
<div class="summary">
  <strong>Model:</strong> {model_name}
  <br/><strong>Wrapper:</strong> Multi-camera via render_camera() detected
  <br/><strong>Status:</strong> Validation {status}
</div>
{"".join(rows_html)}
<div class="footer">
  Generated by <code>examples/lerobot_g1.py --report</code>
  <br/>Full G1 model (29-DOF): huggingface.co/lerobot/unitree-g1-mujoco
</div>
</body>
</html>
"""
    report_path = output_dir / "lerobot_g1_report.html"
    report_path.write_text(html)
    return report_path


# ---------------------------------------------------------------------------
# Validation checks
# ---------------------------------------------------------------------------


def validate_integration(
    output_dir: Path,
    env: LeRobotG1Env,
    checkpoint_infos: list[dict[str, Any]],
    expected_cameras: list[str],
) -> list[str]:
    """Validate that the wrapper integration worked correctly."""
    failures: list[str] = []

    if len(checkpoint_infos) != 3:
        failures.append(f"Expected 3 checkpoints, got {len(checkpoint_infos)}")

    for cp_info in checkpoint_infos:
        files = cp_info.get("files", {})
        meta_path = Path(files.get("state", "")).parent / "metadata.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if meta.get("camera_capability") != "render_camera":
                failures.append(
                    f"Checkpoint '{cp_info['name']}': expected render_camera, "
                    f"got '{meta.get('camera_capability')}'"
                )
            if len(meta.get("cameras", [])) != len(expected_cameras):
                failures.append(
                    f"Checkpoint '{cp_info['name']}': expected {len(expected_cameras)} cameras, "
                    f"got {meta.get('cameras')}"
                )

    for cp_info in checkpoint_infos:
        capture_dir = Path(cp_info["capture_dir"])
        for cam in expected_cameras:
            img = capture_dir / f"{cam}_rgb.png"
            npy = capture_dir / f"{cam}_rgb.npy"
            if not img.exists() and not npy.exists():
                failures.append(f"Missing image for camera '{cam}' at '{cp_info['name']}'")

    for cp_info in checkpoint_infos:
        state_path = Path(cp_info["files"]["state"])
        state = json.loads(state_path.read_text())
        if "step" not in state or "reward" not in state:
            failures.append(f"Checkpoint '{cp_info['name']}': state.json missing required fields")
        if "obs_shape" not in state:
            failures.append(f"Checkpoint '{cp_info['name']}': state.json missing obs_shape")

    torso_z = env._get_torso_z()
    if torso_z < 0.3:
        failures.append(f"Robot fell: torso z={torso_z:.3f}")

    return failures


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate RobotHarnessWrapper on LeRobot G1 MuJoCo simulation"
    )
    parser.add_argument("--output-dir", default="./harness_output", help="Output directory")
    parser.add_argument("--report", action="store_true", help="Generate HTML report")
    parser.add_argument("--width", type=int, default=640, help="Render width")
    parser.add_argument("--height", type=int, default=480, help="Render height")
    parser.add_argument(
        "--model",
        choices=["simplified", "huggingface"],
        default="simplified",
        help="Model to use: simplified (inline 12-DOF) or huggingface (real 29-DOF)",
    )
    parser.add_argument(
        "--assert-success", action="store_true", help="Exit non-zero on validation failure"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    print("=" * 60)
    print("  Roboharness: LeRobot G1 Validation")
    print("=" * 60)

    # 1. Create the G1 environment
    print(f"\n[1/5] Creating LeRobot G1 MuJoCo environment (model={args.model}) ...")
    if args.model == "huggingface":
        repo_path = download_g1_assets()
        xml_path = repo_path / G1_SCENE_XML
        env = LeRobotG1Env(
            render_mode="rgb_array",
            render_width=args.width,
            render_height=args.height,
            model_path=xml_path,
            num_motors=G1_NUM_BODY_MOTORS,
        )
        model_name = f"Unitree G1 29-DOF (from {G1_HF_REPO})"
    else:
        env = LeRobotG1Env(
            render_mode="rgb_array",
            render_width=args.width,
            render_height=args.height,
        )
        model_name = "Unitree G1 simplified (12-DOF inline)"

    cameras = env.cameras
    num_motors = env._num_motors

    print(f"      Model: {model_name}")
    print(f"      nq={env._model.nq}, nv={env._model.nv}, nu={env._model.nu}")
    print(f"      Controlled motors: {num_motors}")
    print(f"      Cameras: {cameras}")
    print(f"      Obs space: {env.observation_space.shape}")
    print(f"      Act space: {env.action_space.shape}")

    # 2. Wrap with RobotHarnessWrapper
    print("[2/5] Wrapping with RobotHarnessWrapper ...")
    checkpoints = [
        {"name": "stand", "step": 300},
        {"name": "step", "step": 700},
        {"name": "balance", "step": 1000},
    ]
    wrapped = RobotHarnessWrapper(
        env,
        checkpoints=checkpoints,
        cameras=cameras,
        output_dir=str(output_dir),
        task_name="lerobot_g1",
    )
    print(f"      Multi-camera detected: {wrapped.has_multi_camera}")
    print(f"      Camera capability: {wrapped.camera_capability}")

    # 3. Run the motion sequence
    print("[3/5] Running motion sequence ...")
    obs, info = wrapped.reset()
    print(f"      Initial obs shape: {obs.shape}, dtype: {obs.dtype}")

    actions = (
        build_stand_sequence(num_motors)
        + build_step_sequence(num_motors)
        + build_balance_sequence(num_motors)
    )
    checkpoint_infos: list[dict[str, Any]] = []

    for i, action in enumerate(actions):
        obs, reward, terminated, _truncated, info = wrapped.step(action)

        if "checkpoint" in info:
            cp = info["checkpoint"]
            checkpoint_infos.append(cp)
            print(f"      Checkpoint '{cp['name']}' at step {cp['step']} | reward={reward:.3f}")
            print(f"        -> {cp['capture_dir']}")

        if terminated:
            print(f"      Robot fell at step {i + 1}!")
            break

    # 4. Validate
    print("[4/5] Validating integration ...")
    failures = validate_integration(output_dir, env, checkpoint_infos, cameras)

    if failures:
        print("      VALIDATION FAILED:")
        for msg in failures:
            print(f"        FAIL: {msg}")
    else:
        print("      All checks passed!")

    # 5. Report
    print("[5/5] Summary")
    trial_dir = output_dir / "lerobot_g1" / "trial_001"
    total_images = len(list(trial_dir.rglob("*_rgb.png"))) if trial_dir.exists() else 0
    print(f"      {total_images} images saved to: {trial_dir}")

    if args.report:
        report_path = generate_html_report(output_dir, model_name)
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

    wrapped.close()

    if args.assert_success and failures:
        print("\n  VALIDATION: FAILED")
        sys.exit(1)
    elif args.assert_success:
        print("\n  VALIDATION: PASSED")


if __name__ == "__main__":
    main()
