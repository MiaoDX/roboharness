#!/usr/bin/env python3
"""SONIC Locomotion Demo — GEAR-SONIC planner on Unitree G1 in MuJoCo.

Runs the NVIDIA GEAR-SONIC locomotion controller in planner mode on the
Unitree G1 humanoid. The controller uses ONNX models (downloaded from
nvidia/GEAR-SONIC on HuggingFace) to generate full-body pose trajectories
from velocity commands.

The demo walks the G1 through: stand → walk forward → stop.

Run:
    pip install roboharness[demo]
    MUJOCO_GL=osmesa python examples/sonic_locomotion.py

Output:
    ./harness_output/sonic_locomotion/trial_001/
        initial/   — robot in default standing pose
        walking/   — robot walking forward
        stopping/  — robot decelerating
        terminal/  — robot stopped, balancing
"""

from __future__ import annotations

import argparse
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
    print("ERROR: mujoco is required. Install with: pip install roboharness[demo]")
    sys.exit(1)

from roboharness.core.protocol import TaskPhase, TaskProtocol
from roboharness.wrappers import RobotHarnessWrapper

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

G1_HF_REPO = "lerobot/unitree-g1-mujoco"
G1_SCENE_XML = "assets/scene_43dof.xml"
G1_NUM_BODY_MOTORS = 29

SONIC_PROTOCOL = TaskProtocol(
    name="sonic_locomotion",
    description="SONIC planner mode: stand → walk → stop",
    phases=[
        TaskPhase("initial", "Robot in default standing pose"),
        TaskPhase("walking", "Walking forward via SONIC planner"),
        TaskPhase("stopping", "Decelerating to stop"),
        TaskPhase("terminal", "Stopped, balancing"),
    ],
)

# ---------------------------------------------------------------------------
# G1 MuJoCo environment (same as lerobot_g1.py)
# ---------------------------------------------------------------------------


def _download_g1_assets() -> Path:
    """Download the real G1 MuJoCo assets from HuggingFace."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: huggingface_hub is required.\nInstall with: pip install roboharness[demo]")
        sys.exit(1)

    print(f"      Downloading {G1_HF_REPO} from HuggingFace ...")
    repo_path = Path(snapshot_download(G1_HF_REPO))
    xml_path = repo_path / G1_SCENE_XML
    if not xml_path.exists():
        print(f"ERROR: Expected scene file not found: {xml_path}")
        sys.exit(1)
    print(f"      Downloaded to: {repo_path}")
    return repo_path


class G1Env(gym.Env):
    """Minimal G1 MuJoCo env for SONIC demo (PD position control)."""

    metadata: ClassVar[dict[str, Any]] = {"render_modes": ["rgb_array"], "render_fps": 50}

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
        render_width: int = 640,
        render_height: int = 480,
    ):
        super().__init__()
        self.render_mode = "rgb_array"
        self._model = mujoco.MjModel.from_xml_path(str(model_path))
        self._data = mujoco.MjData(self._model)
        self._renderer = mujoco.Renderer(self._model, render_height, render_width)

        self._num_motors = G1_NUM_BODY_MOTORS
        self._kp = self.DEFAULT_KP[: self._num_motors].copy()
        self._kd = self.DEFAULT_KD[: self._num_motors].copy()

        self._has_free_joint = (
            self._model.njnt > 0 and self._model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE
        )
        self._qj = 7 if self._has_free_joint else 0
        self._dqj = 6 if self._has_free_joint else 0

        obs_dim = self._model.nq + self._model.nv
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float64)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self._num_motors,), dtype=np.float64)

        self._cameras = [self._model.camera(i).name for i in range(self._model.ncam)]

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self._model, self._data)
        mujoco.mj_forward(self._model, self._data)
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        qpos_target = action
        q = self._data.qpos[self._qj : self._qj + self._num_motors]
        dq = self._data.qvel[self._dqj : self._dqj + self._num_motors]
        torques = self._kp * (qpos_target - q) - self._kd * dq
        self._data.ctrl[: self._num_motors] = torques
        mujoco.mj_step(self._model, self._data)
        return self._get_obs(), 0.0, False, False, {}

    def _get_obs(self) -> np.ndarray:
        return np.concatenate([self._data.qpos, self._data.qvel])

    def render(self) -> np.ndarray:
        if self._cameras:
            return self.render_camera(self._cameras[0])
        self._renderer.update_scene(self._data)
        return self._renderer.render()

    def render_camera(self, camera_name: str) -> np.ndarray:
        if camera_name not in self._cameras:
            raise ValueError(f"Unknown camera: {camera_name}. Available: {self._cameras}")
        self._renderer.update_scene(self._data, camera=camera_name)
        return self._renderer.render()

    @property
    def cameras(self) -> list[str]:
        return list(self._cameras)

    def close(self):
        self._renderer.close()


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------


def generate_html_report(output_dir: Path) -> Path:
    from roboharness.reporting import generate_html_report as _generate

    return _generate(
        output_dir,
        "sonic_locomotion",
        title="SONIC Locomotion Report",
        subtitle="GEAR-SONIC planner mode on Unitree G1 — stand → walk → stop.",
        accent_color="#7c4dff",
        summary_html=(
            "<strong>Controller:</strong> SonicLocomotionController (ONNX, CPU)"
            "<br/><strong>Model:</strong> nvidia/GEAR-SONIC planner"
            "<br/><strong>Robot:</strong> Unitree G1 29-DOF (lerobot/unitree-g1-mujoco)"
        ),
        footer_text="Generated by <code>examples/sonic_locomotion.py --report</code>",
        meshcat_mode="none",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="SONIC locomotion demo on Unitree G1")
    parser.add_argument("--output-dir", default="./harness_output", help="Output directory")
    parser.add_argument("--report", action="store_true", help="Generate HTML report")
    parser.add_argument("--width", type=int, default=640, help="Render width")
    parser.add_argument("--height", type=int, default=480, help="Render height")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    n_steps = 1200

    print("=" * 60)
    print("  Roboharness: SONIC Locomotion Demo")
    print("=" * 60)

    # 1. Download G1 model
    print("\n[1/4] Downloading G1 model from HuggingFace ...")
    repo_path = _download_g1_assets()
    xml_path = repo_path / G1_SCENE_XML

    env = G1Env(model_path=xml_path, render_width=args.width, render_height=args.height)
    cameras = env.cameras
    print(f"      Cameras: {cameras}")
    print(f"      Obs space: {env.observation_space.shape}")

    # 2. Load SONIC controller
    from roboharness.controllers.locomotion import SonicLocomotionController

    print("[2/4] Loading SONIC locomotion controller ...")
    sonic = SonicLocomotionController()
    print("      SONIC planner ONNX model loaded")

    # 3. Wrap with RobotHarnessWrapper
    print("[3/4] Wrapping with RobotHarnessWrapper ...")
    wrapped = RobotHarnessWrapper(
        env,
        protocol=SONIC_PROTOCOL,
        phase_steps={"initial": 200, "walking": 600, "stopping": 900, "terminal": n_steps},
        cameras=cameras,
        output_dir=str(output_dir),
        task_name="sonic_locomotion",
    )
    print(f"      Protocol: {wrapped.active_protocol.name}")
    print(f"      Multi-camera: {wrapped.has_multi_camera}")

    # 4. Run simulation
    print(f"[4/4] Running simulation ({n_steps} steps) ...")
    _obs, _info = wrapped.reset()
    sonic.reset()

    checkpoint_infos: list[dict[str, Any]] = []
    for i in range(n_steps):
        state = {"qpos": env._data.qpos, "qvel": env._data.qvel}

        # Stand 200 steps → walk 400 steps → stop
        if i < 200:
            velocity = [0.0, 0.0, 0.0]
        elif i < 800:
            velocity = [0.3, 0.0, 0.0]
        else:
            velocity = [0.0, 0.0, 0.0]

        lower_body = sonic.compute(command={"velocity": velocity}, state=state)
        action = np.zeros(G1_NUM_BODY_MOTORS)
        action[: len(lower_body)] = lower_body
        _obs, _reward, _terminated, _truncated, info = wrapped.step(action)

        if "checkpoint" in info:
            cp = info["checkpoint"]
            checkpoint_infos.append(cp)
            torso_z = env._data.qpos[2] if env._has_free_joint else 0.0
            print(f"      Checkpoint '{cp['name']}' at step {cp['step']} | torso_z={torso_z:.3f}m")

    # Summary
    trial_dir = output_dir / "sonic_locomotion" / "trial_001"
    total_images = len(list(trial_dir.rglob("*_rgb.png"))) if trial_dir.exists() else 0
    print(f"\n      {len(checkpoint_infos)} checkpoints, {total_images} images")

    if args.report:
        report_path = generate_html_report(output_dir)
        print(f"      HTML report: {report_path}")

    print("\n" + "=" * 60)
    wrapped.close()


if __name__ == "__main__":
    main()
