"""Tests for LeRobot G1 compatibility — validates RobotHarnessWrapper with a G1-style env.

LeRobot's Unitree G1 MuJoCo environment (lerobot/unitree-g1-mujoco) provides:
  - 29-DOF body (+ 14 hand DOF), 99-dim observations (nq=50 + nv=49)
  - 29-dim action space (body motors only, hand held at zero)
  - Multi-camera rendering via render_camera(camera_name)
  - Standard Gymnasium reset/step/render interface

These tests use a lightweight mock environment (no MuJoCo required) that
matches the real G1 model dimensions.
"""

from __future__ import annotations

import json
from typing import Any, ClassVar

import numpy as np
import pytest

gym = pytest.importorskip("gymnasium", reason="gymnasium not installed")

from gymnasium import spaces  # noqa: E402

from roboharness.wrappers import RobotHarnessWrapper  # noqa: E402
from roboharness.wrappers.gymnasium_wrapper import MultiCameraCapability  # noqa: E402


class MockLeRobotG1Env(gym.Env):
    """Mock environment mimicking the real 29-DOF G1 from HuggingFace.

    Matches the actual G1 model dimensions:
      - Observation: nq (50) + nv (49) = 99-dim (43-DOF model with free joint)
      - Action: 29-dim (body motors only, hand motors held at zero)
      - Cameras: head_camera (on robot body), global_view (scene-level)
    """

    metadata: ClassVar[dict[str, Any]] = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, render_mode: str = "rgb_array"):
        super().__init__()
        self.render_mode = render_mode
        self._step_count = 0

        # Real G1 43-DOF model: free joint (7 qpos, 6 qvel) + 43 hinge joints
        # nq = 7 + 43 = 50, nv = 6 + 43 = 49, obs = 50 + 49 = 99
        obs_dim = 99
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float64
        )
        # Only 29 body motors exposed (out of 43 total actuators)
        self.action_space = spaces.Box(low=-np.pi, high=np.pi, shape=(29,), dtype=np.float64)

        self._cameras = ["head_camera", "global_view"]
        self._torso_z = 0.75  # G1 standing height

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed, options=options)
        self._step_count = 0
        self._torso_z = 0.75
        return np.zeros(self.observation_space.shape[0], dtype=np.float64), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self._step_count += 1
        obs = (
            np.random.default_rng(self._step_count).standard_normal(self.observation_space.shape[0])
            * 0.1
        )
        reward = 1.0 + self._torso_z
        info: dict[str, Any] = {"torso_z": self._torso_z, "sim_time": self._step_count * 0.004}
        return obs, reward, False, False, info

    def render(self) -> np.ndarray:
        return self.render_camera("head_camera")

    def render_camera(self, camera_name: str) -> np.ndarray:
        if camera_name not in self._cameras:
            raise ValueError(f"Unknown camera: {camera_name}. Available: {self._cameras}")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        color_map = {"head_camera": 0, "global_view": 1}
        frame[:, :, color_map[camera_name]] = 160
        return frame

    @property
    def cameras(self) -> list[str]:
        return list(self._cameras)


# ---------------------------------------------------------------------------
# Multi-camera detection
# ---------------------------------------------------------------------------


def test_g1_multi_camera_detected(tmp_path):
    """Wrapper should detect render_camera capability on G1 env."""
    env = MockLeRobotG1Env()
    wrapped = RobotHarnessWrapper(
        env,
        cameras=["head_camera", "global_view"],
        checkpoints=[{"name": "cp", "step": 1}],
        output_dir=tmp_path,
    )
    assert wrapped.camera_capability == MultiCameraCapability.RENDER_CAMERA
    assert wrapped.has_multi_camera is True


# ---------------------------------------------------------------------------
# Checkpoint capture
# ---------------------------------------------------------------------------


def test_g1_checkpoint_captures_all_cameras(tmp_path):
    """Both cameras should produce separate images at each checkpoint."""
    env = MockLeRobotG1Env()
    wrapped = RobotHarnessWrapper(
        env,
        cameras=["head_camera", "global_view"],
        checkpoints=[{"name": "stand", "step": 5}],
        output_dir=tmp_path,
        task_name="g1_test",
    )
    wrapped.reset()
    for _ in range(5):
        _, _, _, _, info = wrapped.step(env.action_space.sample())

    assert "checkpoint" in info
    files = info["checkpoint"]["files"]
    assert "head_camera_rgb" in files
    assert "global_view_rgb" in files

    capture_dir = tmp_path / "g1_test" / "trial_001" / "stand"
    for cam in ["head_camera", "global_view"]:
        assert (capture_dir / f"{cam}_rgb.png").exists() or (
            capture_dir / f"{cam}_rgb.npy"
        ).exists()


def test_g1_checkpoint_metadata(tmp_path):
    """Metadata JSON should record correct camera capability and camera list."""
    env = MockLeRobotG1Env()
    wrapped = RobotHarnessWrapper(
        env,
        cameras=["head_camera", "global_view"],
        checkpoints=[{"name": "stand", "step": 3}],
        output_dir=tmp_path,
        task_name="g1_meta",
    )
    wrapped.reset()
    for _ in range(3):
        _, _, _, _, _info = wrapped.step(env.action_space.sample())

    capture_dir = tmp_path / "g1_meta" / "trial_001" / "stand"
    meta = json.loads((capture_dir / "metadata.json").read_text())

    assert meta["camera_capability"] == "render_camera"
    assert set(meta["cameras"]) == {"head_camera", "global_view"}
    assert meta["step"] == 3
    assert meta["task"] == "g1_meta"
    assert meta["checkpoint"] == "stand"


def test_g1_state_json_has_obs_shape(tmp_path):
    """State JSON should record 99-dim observation shape."""
    env = MockLeRobotG1Env()
    wrapped = RobotHarnessWrapper(
        env,
        checkpoints=[{"name": "cp", "step": 2}],
        output_dir=tmp_path,
        task_name="g1_state",
    )
    wrapped.reset()
    for _ in range(2):
        _, _, _, _, _info = wrapped.step(env.action_space.sample())

    state_path = tmp_path / "g1_state" / "trial_001" / "cp" / "state.json"
    state = json.loads(state_path.read_text())

    assert state["obs_shape"] == [99]
    assert state["obs_dtype"] == "float64"
    assert state["step"] == 2
    assert isinstance(state["reward"], float)


# ---------------------------------------------------------------------------
# Gymnasium API compatibility
# ---------------------------------------------------------------------------


def test_g1_obs_passthrough(tmp_path):
    """Wrapper should not modify 99-dim numpy observations."""
    env = MockLeRobotG1Env()
    wrapped = RobotHarnessWrapper(
        env,
        checkpoints=[{"name": "cp", "step": 10}],
        output_dir=tmp_path,
    )
    obs, _info = wrapped.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (99,)
    assert obs.dtype == np.float64

    obs, reward, terminated, truncated, _info = wrapped.step(env.action_space.sample())
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (99,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_g1_action_space(tmp_path):
    """G1 env should have 29-dim action space (body motors only)."""
    env = MockLeRobotG1Env()
    assert env.action_space.shape == (29,)
    action = env.action_space.sample()
    assert action.shape == (29,)


def test_g1_reward_passthrough(tmp_path):
    """Reward values should pass through unchanged."""
    env = MockLeRobotG1Env()
    wrapped = RobotHarnessWrapper(
        env,
        checkpoints=[{"name": "cp", "step": 100}],
        output_dir=tmp_path,
    )
    wrapped.reset()
    _, reward, _, _, _ = wrapped.step(env.action_space.sample())
    # MockLeRobotG1Env returns 1.0 + torso_z (0.75) = 1.75
    assert abs(reward - 1.75) < 0.01


# ---------------------------------------------------------------------------
# Multiple checkpoints
# ---------------------------------------------------------------------------


def test_g1_multiple_checkpoints(tmp_path):
    """Multiple checkpoints should each trigger independently."""
    env = MockLeRobotG1Env()
    checkpoints = [
        {"name": "stand", "step": 5},
        {"name": "step", "step": 10},
        {"name": "balance", "step": 15},
    ]
    wrapped = RobotHarnessWrapper(
        env,
        cameras=["head_camera", "global_view"],
        checkpoints=checkpoints,
        output_dir=tmp_path,
        task_name="g1_multi",
    )
    wrapped.reset()

    captured = []
    for _ in range(15):
        _, _, _, _, info = wrapped.step(env.action_space.sample())
        if "checkpoint" in info:
            captured.append(info["checkpoint"]["name"])

    assert captured == ["stand", "step", "balance"]

    trial_dir = tmp_path / "g1_multi" / "trial_001"
    for name in ["stand", "step", "balance"]:
        assert (trial_dir / name).is_dir()
        assert (trial_dir / name / "state.json").exists()
        assert (trial_dir / name / "metadata.json").exists()
        # Both cameras should be captured
        files = list((trial_dir / name).glob("*_rgb.*"))
        assert len(files) >= 2


# ---------------------------------------------------------------------------
# Reset behavior
# ---------------------------------------------------------------------------


def test_g1_reset_increments_trial(tmp_path):
    """Each reset should create a new trial directory."""
    env = MockLeRobotG1Env()
    wrapped = RobotHarnessWrapper(
        env,
        checkpoints=[{"name": "cp", "step": 1}],
        output_dir=tmp_path,
        task_name="g1_trial",
    )

    wrapped.reset()
    wrapped.step(env.action_space.sample())

    wrapped.reset()
    wrapped.step(env.action_space.sample())

    assert (tmp_path / "g1_trial" / "trial_001" / "cp").is_dir()
    assert (tmp_path / "g1_trial" / "trial_002" / "cp").is_dir()


# ---------------------------------------------------------------------------
# Camera tests
# ---------------------------------------------------------------------------


def test_g1_camera_frames_are_distinct(tmp_path):
    """head_camera and global_view should produce visually distinct frames."""
    env = MockLeRobotG1Env()
    head = env.render_camera("head_camera")
    glob = env.render_camera("global_view")

    assert head[:, :, 0].mean() > head[:, :, 1].mean()  # head_camera = red
    assert glob[:, :, 1].mean() > glob[:, :, 0].mean()  # global_view = green


def test_g1_invalid_camera_raises(tmp_path):
    """Requesting an unknown camera should raise ValueError."""
    env = MockLeRobotG1Env()
    with pytest.raises(ValueError, match="Unknown camera"):
        env.render_camera("nonexistent")
