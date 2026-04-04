"""Tests for LeRobot G1 compatibility — validates RobotHarnessWrapper with a G1-style env.

LeRobot's Unitree G1 MuJoCo environments are Gymnasium-compatible and provide:
  - NumPy array observations (joint positions + velocities)
  - NumPy array actions (position targets for actuators)
  - Multi-camera rendering via render_camera(camera_name)
  - Standard Gymnasium reset/step/render interface

These tests use a lightweight mock environment (no MuJoCo required) to verify
the wrapper handles the G1 environment pattern correctly.
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
    """Mock environment mimicking a LeRobot G1 Gymnasium environment.

    Replicates the key interface of a Unitree G1 MuJoCo environment:
      - Observation: joint positions (nq=10) + velocities (nv=8) = 18-dim
      - Action: position targets for 8 actuators
      - render_camera(name) for multi-view capture
      - render() returns default (front) camera view
    """

    metadata: ClassVar[dict[str, Any]] = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, render_mode: str = "rgb_array"):
        super().__init__()
        self.render_mode = render_mode
        self._step_count = 0

        # G1-like observation: joint pos (10) + joint vel (8) = 18
        obs_dim = 18
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float64
        )

        # 8 actuators (6 leg + 2 arm)
        self.action_space = spaces.Box(low=-1.5, high=2.5, shape=(8,), dtype=np.float64)

        self._cameras = ["front", "side", "top"]
        self._torso_z = 0.85  # simulated torso height

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed, options=options)
        self._step_count = 0
        self._torso_z = 0.85
        return np.zeros(self.observation_space.shape[0], dtype=np.float64), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self._step_count += 1
        obs = (
            np.random.default_rng(self._step_count).standard_normal(self.observation_space.shape[0])
            * 0.1
        )
        reward = 1.0 + self._torso_z
        terminated = False
        truncated = False
        info: dict[str, Any] = {"torso_z": self._torso_z, "sim_time": self._step_count * 0.002}
        return obs, reward, terminated, truncated, info

    def render(self) -> np.ndarray:
        return self.render_camera("front")

    def render_camera(self, camera_name: str) -> np.ndarray:
        """Multi-camera rendering — same interface as LeRobotG1Env."""
        if camera_name not in self._cameras:
            raise ValueError(f"Unknown camera: {camera_name}. Available: {self._cameras}")
        # Return distinct frames per camera (different color channels)
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        color_map = {"front": 0, "side": 1, "top": 2}
        frame[:, :, color_map[camera_name]] = 180
        return frame

    @property
    def cameras(self) -> list[str]:
        return list(self._cameras)


# ---------------------------------------------------------------------------
# Multi-camera detection
# ---------------------------------------------------------------------------


def test_g1_multi_camera_detected(tmp_path):
    """Wrapper should detect render_camera capability on G1-style env."""
    env = MockLeRobotG1Env()
    wrapped = RobotHarnessWrapper(
        env,
        cameras=["front", "side", "top"],
        checkpoints=[{"name": "cp", "step": 1}],
        output_dir=tmp_path,
    )
    assert wrapped.camera_capability == MultiCameraCapability.RENDER_CAMERA
    assert wrapped.has_multi_camera is True


# ---------------------------------------------------------------------------
# Checkpoint capture
# ---------------------------------------------------------------------------


def test_g1_checkpoint_captures_all_cameras(tmp_path):
    """All 3 cameras should produce separate images at each checkpoint."""
    env = MockLeRobotG1Env()
    wrapped = RobotHarnessWrapper(
        env,
        cameras=["front", "side", "top"],
        checkpoints=[{"name": "stand", "step": 5}],
        output_dir=tmp_path,
        task_name="g1_test",
    )
    wrapped.reset()
    for _ in range(5):
        _, _, _, _, info = wrapped.step(env.action_space.sample())

    assert "checkpoint" in info
    files = info["checkpoint"]["files"]

    # Each camera should have its own RGB file
    assert "front_rgb" in files
    assert "side_rgb" in files
    assert "top_rgb" in files

    # Verify files on disk
    capture_dir = tmp_path / "g1_test" / "trial_001" / "stand"
    for cam in ["front", "side", "top"]:
        assert (capture_dir / f"{cam}_rgb.png").exists() or (
            capture_dir / f"{cam}_rgb.npy"
        ).exists()


def test_g1_checkpoint_metadata(tmp_path):
    """Metadata JSON should record correct camera capability and camera list."""
    env = MockLeRobotG1Env()
    wrapped = RobotHarnessWrapper(
        env,
        cameras=["front", "side", "top"],
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
    assert set(meta["cameras"]) == {"front", "side", "top"}
    assert meta["step"] == 3
    assert meta["task"] == "g1_meta"
    assert meta["checkpoint"] == "stand"


def test_g1_state_json_has_obs_shape(tmp_path):
    """State JSON should record observation shape for numpy array obs."""
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

    assert state["obs_shape"] == [18]
    assert state["obs_dtype"] == "float64"
    assert state["step"] == 2
    assert isinstance(state["reward"], float)


# ---------------------------------------------------------------------------
# Gymnasium API compatibility
# ---------------------------------------------------------------------------


def test_g1_obs_passthrough(tmp_path):
    """Wrapper should not modify numpy observations."""
    env = MockLeRobotG1Env()
    wrapped = RobotHarnessWrapper(
        env,
        checkpoints=[{"name": "cp", "step": 10}],
        output_dir=tmp_path,
    )
    obs, _info = wrapped.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (18,)
    assert obs.dtype == np.float64

    obs, reward, terminated, truncated, _info = wrapped.step(env.action_space.sample())
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (18,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


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
    # MockLeRobotG1Env returns 1.0 + torso_z (0.85) = 1.85
    assert abs(reward - 1.85) < 0.01


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
        cameras=["front", "side", "top"],
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

    # Verify all checkpoint dirs exist
    trial_dir = tmp_path / "g1_multi" / "trial_001"
    for name in ["stand", "step", "balance"]:
        assert (trial_dir / name).is_dir()
        assert (trial_dir / name / "state.json").exists()
        assert (trial_dir / name / "metadata.json").exists()


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

    # Trial 1
    wrapped.reset()
    wrapped.step(env.action_space.sample())

    # Trial 2
    wrapped.reset()
    wrapped.step(env.action_space.sample())

    assert (tmp_path / "g1_trial" / "trial_001" / "cp").is_dir()
    assert (tmp_path / "g1_trial" / "trial_002" / "cp").is_dir()


# ---------------------------------------------------------------------------
# Camera frame distinctness
# ---------------------------------------------------------------------------


def test_g1_camera_frames_are_distinct(tmp_path):
    """Different cameras should produce visually distinct frames."""
    env = MockLeRobotG1Env()
    front = env.render_camera("front")
    side = env.render_camera("side")
    top = env.render_camera("top")

    # Each camera has a different dominant color channel
    assert front[:, :, 0].mean() > front[:, :, 1].mean()  # front = red
    assert side[:, :, 1].mean() > side[:, :, 0].mean()  # side = green
    assert top[:, :, 2].mean() > top[:, :, 0].mean()  # top = blue


def test_g1_invalid_camera_raises(tmp_path):
    """Requesting an unknown camera should raise ValueError."""
    env = MockLeRobotG1Env()
    with pytest.raises(ValueError, match="Unknown camera"):
        env.render_camera("nonexistent")
