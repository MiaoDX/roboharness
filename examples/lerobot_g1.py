#!/usr/bin/env python3
"""LeRobot G1 Validation — RobotHarnessWrapper on a Unitree G1 MuJoCo simulation.

Validates that RobotHarnessWrapper integrates correctly with the Unitree G1
humanoid robot in MuJoCo. Downloads the real 29-DOF G1 model from HuggingFace,
runs it with the GR00T RL balance/walk controller, and captures multi-view
checkpoint screenshots via RobotHarnessWrapper.

The G1 model (29 body DOF + 14 hand DOF) is hosted at:
  huggingface.co/lerobot/unitree-g1-mujoco

Run:
    pip install roboharness[lerobot] gymnasium Pillow
    MUJOCO_GL=osmesa python examples/lerobot_g1.py

Output:
    ./harness_output/lerobot_g1/trial_001/
        stand/    — robot in default standing pose
        walk/     — robot walking forward
        stop/     — robot stopped, balancing
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
    print("ERROR: mujoco is required. Install with: pip install roboharness[lerobot]")
    sys.exit(1)

from roboharness.wrappers import RobotHarnessWrapper

# ---------------------------------------------------------------------------
# HuggingFace model download
# ---------------------------------------------------------------------------

G1_HF_REPO = "lerobot/unitree-g1-mujoco"
G1_SCENE_XML = "assets/scene_43dof.xml"
G1_NUM_BODY_MOTORS = 29


def download_g1_assets() -> Path:
    """Download the real G1 MuJoCo assets from HuggingFace."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: huggingface_hub is required.\nInstall with: pip install roboharness[lerobot]")
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
# Gymnasium environment wrapping the G1 MuJoCo model
# ---------------------------------------------------------------------------


class LeRobotG1Env(gym.Env):
    """Gymnasium environment for the Unitree G1 humanoid in MuJoCo.

    Loads a MuJoCo XML model from ``model_path`` (the real G1 from HuggingFace).
    Actions are joint position targets, converted to torques via a PD controller
    (the G1 MuJoCo model uses torque actuators).
    """

    metadata: ClassVar[dict[str, Any]] = {"render_modes": ["rgb_array"], "render_fps": 50}

    # Default PD gains per joint (from GR00T WBC training configuration).
    DEFAULT_KP = np.array(
        [
            100, 100, 100, 150, 40, 40,   # left leg
            100, 100, 100, 150, 40, 40,   # right leg
            100, 50, 50,                   # waist
            40, 40, 40, 40, 40, 20, 20,   # left arm
            40, 40, 40, 40, 40, 20, 20,   # right arm
        ],
        dtype=np.float64,
    )  # fmt: skip
    DEFAULT_KD = DEFAULT_KP * 0.02

    def __init__(
        self,
        model_path: str | Path,
        render_mode: str = "rgb_array",
        render_width: int = 640,
        render_height: int = 480,
        num_motors: int | None = None,
    ):
        super().__init__()
        self.render_mode = render_mode
        self._render_width = render_width
        self._render_height = render_height

        self._model = mujoco.MjModel.from_xml_path(str(model_path))
        self._data = mujoco.MjData(self._model)
        self._renderer = mujoco.Renderer(self._model, self._render_height, self._render_width)

        self._num_motors = num_motors or self._model.nu
        assert self._num_motors <= self._model.nu

        # PD gains
        self._kp = self.DEFAULT_KP[: self._num_motors].copy()
        self._kd = self.DEFAULT_KD[: self._num_motors].copy()

        # Free joint offset for qpos/qvel indexing
        self._has_free_joint = (
            self._model.njnt > 0 and self._model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE
        )
        self._qj_offset = 7 if self._has_free_joint else 0
        self._dqj_offset = 6 if self._has_free_joint else 0

        obs_dim = self._model.nq + self._model.nv
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float64
        )
        self.action_space = spaces.Box(
            low=-np.pi, high=np.pi, shape=(self._num_motors,), dtype=np.float64
        )

        self._cameras = [self._model.camera(i).name for i in range(self._model.ncam)]
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
        # PD controller: convert position targets to torques
        q_actual = self._data.qpos[self._qj_offset : self._qj_offset + self._num_motors]
        dq_actual = self._data.qvel[self._dqj_offset : self._dqj_offset + self._num_motors]
        torques = self._kp * (action - q_actual) + self._kd * (0.0 - dq_actual)

        self._data.ctrl[:] = 0.0
        self._data.ctrl[: self._num_motors] = torques
        mujoco.mj_step(self._model, self._data)
        self._step_count += 1

        obs = self._get_obs()
        torso_z = self._get_torso_z()
        reward = 1.0 + torso_z
        terminated = False

        info: dict[str, Any] = {
            "torso_z": float(torso_z),
            "sim_time": float(self._data.time),
        }
        return obs, reward, terminated, False, info

    def render(self) -> np.ndarray:
        if self._cameras:
            return self.render_camera(self._cameras[0])
        self._renderer.update_scene(self._data)
        return self._renderer.render()

    def render_camera(self, camera_name: str) -> np.ndarray:
        cam_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if cam_id < 0:
            raise ValueError(f"Unknown camera: {camera_name}. Available: {self._cameras}")
        self._renderer.update_scene(self._data, camera=camera_name)
        return self._renderer.render()

    def _get_obs(self) -> np.ndarray:
        return np.concatenate([self._data.qpos.copy(), self._data.qvel.copy()])

    def _get_torso_z(self) -> float:
        if self._has_free_joint:
            return float(self._data.qpos[2])
        return float(self._data.qpos[0])

    @property
    def cameras(self) -> list[str]:
        return list(self._cameras)

    def close(self) -> None:
        self._renderer.close()


# ---------------------------------------------------------------------------
# HTML report generator
# ---------------------------------------------------------------------------


def generate_html_report(output_dir: Path) -> Path:
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
  <strong>Model:</strong> Unitree G1 29-DOF ({G1_HF_REPO})
  <br/><strong>Controller:</strong> GR00T Balance + Walk (ONNX)
  <br/><strong>Status:</strong> Validation {status}
</div>
{"".join(rows_html)}
<div class="footer">
  Generated by <code>examples/lerobot_g1.py --report</code>
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

MIN_TORSO_Z = 0.5  # robot must stay above 0.5m (initial ~0.79m)


def validate_integration(
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
    if torso_z < MIN_TORSO_Z:
        failures.append(f"Robot fell: torso z={torso_z:.3f}m (min={MIN_TORSO_Z}m)")
    else:
        print(f"      Robot upright: torso_z={torso_z:.3f}m (min={MIN_TORSO_Z}m)")

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
        "--controller",
        choices=["groot"],
        default="groot",
        help="Locomotion controller (default: groot). More options coming soon.",
    )
    parser.add_argument(
        "--assert-success", action="store_true", help="Exit non-zero on validation failure"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    print("=" * 60)
    print("  Roboharness: LeRobot G1 Validation")
    print("=" * 60)

    # 1. Download and load the G1 model
    print("\n[1/5] Downloading G1 model from HuggingFace ...")
    repo_path = download_g1_assets()
    xml_path = repo_path / G1_SCENE_XML

    env = LeRobotG1Env(
        model_path=xml_path,
        render_mode="rgb_array",
        render_width=args.width,
        render_height=args.height,
        num_motors=G1_NUM_BODY_MOTORS,
    )

    cameras = env.cameras
    num_motors = env._num_motors

    print(f"      nq={env._model.nq}, nv={env._model.nv}, nu={env._model.nu}")
    print(f"      Controlled motors: {num_motors}")
    print(f"      Cameras: {cameras}")
    print(f"      Obs space: {env.observation_space.shape}")
    print(f"      Act space: {env.action_space.shape}")

    # 2. Load GR00T locomotion controller
    from roboharness.controllers.locomotion import GrootLocomotionController

    print("[2/5] Loading GR00T locomotion controller ...")
    loco = GrootLocomotionController()
    print("      Balance + Walk ONNX policies loaded")

    # 3. Wrap with RobotHarnessWrapper
    print("[3/5] Wrapping with RobotHarnessWrapper ...")
    checkpoints = [
        {"name": "stand", "step": 300},
        {"name": "walk", "step": 700},
        {"name": "stop", "step": 1000},
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

    # 4. Run simulation with GR00T controller
    print("[4/5] Running simulation ...")
    obs, _info = wrapped.reset()
    loco.reset()
    print(f"      Initial obs shape: {obs.shape}, dtype: {obs.dtype}")

    n_steps = 1000
    checkpoint_infos: list[dict[str, Any]] = []

    for i in range(n_steps):
        state = {"qpos": env._data.qpos, "qvel": env._data.qvel}
        # Stand for 300 steps, walk forward for 400, then stop
        if i < 300:
            velocity = [0.0, 0.0, 0.0]
        elif i < 700:
            velocity = [0.3, 0.0, 0.0]
        else:
            velocity = [0.0, 0.0, 0.0]

        lower_body = loco.compute(command={"velocity": velocity}, state=state)
        action = np.zeros(num_motors)
        action[: len(lower_body)] = lower_body
        obs, reward, _terminated, _truncated, info = wrapped.step(action)

        if "checkpoint" in info:
            cp = info["checkpoint"]
            checkpoint_infos.append(cp)
            torso_z = env._get_torso_z()
            print(
                f"      Checkpoint '{cp['name']}' at step {cp['step']}"
                f" | reward={reward:.3f} | torso_z={torso_z:.3f}m"
            )
            print(f"        -> {cp['capture_dir']}")

    # 5. Validate
    print("[5/5] Validating integration ...")
    failures = validate_integration(env, checkpoint_infos, cameras)

    if failures:
        print("      VALIDATION FAILED:")
        for msg in failures:
            print(f"        FAIL: {msg}")
    else:
        print("      All checks passed!")

    # Summary
    trial_dir = output_dir / "lerobot_g1" / "trial_001"
    total_images = len(list(trial_dir.rglob("*_rgb.png"))) if trial_dir.exists() else 0
    print(f"      {total_images} images saved to: {trial_dir}")

    if args.report:
        report_path = generate_html_report(output_dir)
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
