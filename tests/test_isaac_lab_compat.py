"""Tests for Isaac Lab compatibility — validates RobotHarnessWrapper with torch tensors.

Isaac Lab environments differ from standard Gymnasium envs in that:
  - Observations and actions are PyTorch tensors (not NumPy arrays)
  - The first dimension is the number of parallel environments
  - Rewards may be torch tensors
  - render() may return a torch tensor

These tests use a lightweight mock environment to verify the wrapper
handles these edge cases correctly, without requiring a GPU or Isaac Lab.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="PyTorch not installed")
gym = pytest.importorskip("gymnasium", reason="gymnasium not installed")

from gymnasium import spaces  # noqa: E402

from roboharness.wrappers import RobotHarnessWrapper  # noqa: E402


class MockIsaacLabEnv(gym.Env):
    """Mock environment that mimics Isaac Lab's tensor-based interface.

    Isaac Lab envs inherit from gymnasium.Env but return torch tensors
    instead of numpy arrays, and the first dimension is num_envs.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}

    def __init__(self, num_envs: int = 1, render_mode: str = "rgb_array"):
        super().__init__()
        self.num_envs = num_envs
        self.render_mode = render_mode
        self._step_count = 0

        # Isaac Lab uses Box spaces but actual data is torch tensors
        obs_dim = 12  # typical for reach tasks (joint pos + target pos)
        act_dim = 7  # 7-DOF arm
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        super().reset(seed=seed, options=options)
        self._step_count = 0
        obs = torch.zeros(self.num_envs, *self.observation_space.shape)
        return obs, {}

    def step(self, action: Any) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
        self._step_count += 1
        obs = torch.randn(self.num_envs, *self.observation_space.shape)
        reward = torch.tensor([0.5] * self.num_envs)
        terminated = torch.tensor([False] * self.num_envs)
        truncated = torch.tensor([False] * self.num_envs)
        return obs, reward, terminated, truncated, {}

    def render(self) -> np.ndarray:
        # Isaac Lab's render typically returns numpy RGB array even though obs are tensors
        return np.zeros((480, 640, 3), dtype=np.uint8)


class MockIsaacLabEnvDictObs(MockIsaacLabEnv):
    """Mock Isaac Lab env with dict observation space (common for RL tasks)."""

    def __init__(self, num_envs: int = 1, render_mode: str = "rgb_array"):
        super().__init__(num_envs=num_envs, render_mode=render_mode)
        self.observation_space = spaces.Dict(
            {
                "policy": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32
                ),
            }
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        gym.Env.reset(self, seed=seed, options=options)
        self._step_count = 0
        obs = {"policy": torch.zeros(self.num_envs, 12)}
        return obs, {}

    def step(self, action: Any) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
        self._step_count += 1
        obs = {"policy": torch.randn(self.num_envs, 12)}
        reward = torch.tensor([0.5] * self.num_envs)
        terminated = torch.tensor([False] * self.num_envs)
        truncated = torch.tensor([False] * self.num_envs)
        return obs, reward, terminated, truncated, {}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_wrapper_with_torch_tensor_obs(tmp_path):
    """Wrapper should pass through torch tensor observations unchanged."""
    env = MockIsaacLabEnv(num_envs=1)
    wrapped = RobotHarnessWrapper(
        env,
        checkpoints=[{"name": "cp1", "step": 5}],
        output_dir=tmp_path,
    )
    obs, info = wrapped.reset()
    assert isinstance(obs, torch.Tensor)

    for _ in range(5):
        obs, reward, terminated, truncated, info = wrapped.step(
            torch.zeros(1, *env.action_space.shape)
        )
    assert isinstance(obs, torch.Tensor)


def test_wrapper_checkpoint_with_torch_reward(tmp_path):
    """Checkpoint state.json should handle torch tensor rewards."""
    env = MockIsaacLabEnv(num_envs=1)
    wrapped = RobotHarnessWrapper(
        env,
        checkpoints=[{"name": "cp1", "step": 3}],
        output_dir=tmp_path,
    )
    wrapped.reset()
    for _ in range(3):
        obs, reward, terminated, truncated, info = wrapped.step(
            torch.zeros(1, *env.action_space.shape)
        )

    assert "checkpoint" in info
    assert info["checkpoint"]["name"] == "cp1"

    # Verify state.json was written and is valid
    import json
    from pathlib import Path

    state_path = Path(info["checkpoint"]["files"]["state"])
    state = json.loads(state_path.read_text())
    assert state["checkpoint"] == "cp1"
    assert state["step"] == 3


def test_wrapper_with_dict_observations(tmp_path):
    """Wrapper should handle dict observations (common in Isaac Lab)."""
    env = MockIsaacLabEnvDictObs(num_envs=1)
    wrapped = RobotHarnessWrapper(
        env,
        checkpoints=[{"name": "cp1", "step": 2}],
        output_dir=tmp_path,
    )
    obs, info = wrapped.reset()
    assert isinstance(obs, dict)
    assert "policy" in obs

    for _ in range(2):
        obs, reward, terminated, truncated, info = wrapped.step(
            torch.zeros(1, *env.action_space.shape)
        )
    assert isinstance(obs, dict)
    assert "checkpoint" in info

    # Verify state.json records obs_keys for dict obs
    import json
    from pathlib import Path

    state_path = Path(info["checkpoint"]["files"]["state"])
    state = json.loads(state_path.read_text())
    assert "obs_keys" in state
    assert "policy" in state["obs_keys"]


def test_wrapper_with_multi_env(tmp_path):
    """Wrapper should work with vectorized envs (num_envs > 1)."""
    env = MockIsaacLabEnv(num_envs=4)
    wrapped = RobotHarnessWrapper(
        env,
        checkpoints=[{"name": "cp1", "step": 1}],
        output_dir=tmp_path,
    )
    obs, _ = wrapped.reset()
    assert obs.shape[0] == 4

    obs, reward, terminated, truncated, info = wrapped.step(
        torch.zeros(4, *env.action_space.shape)
    )
    assert obs.shape[0] == 4
    assert "checkpoint" in info


def test_wrapper_render_capture_saved(tmp_path):
    """Wrapper should save render output as PNG at checkpoints."""
    env = MockIsaacLabEnv(num_envs=1)
    wrapped = RobotHarnessWrapper(
        env,
        checkpoints=[{"name": "init", "step": 1}],
        output_dir=tmp_path,
        task_name="test_isaac",
    )
    wrapped.reset()
    obs, reward, terminated, truncated, info = wrapped.step(
        torch.zeros(1, *env.action_space.shape)
    )
    assert "checkpoint" in info
    capture_dir = tmp_path / "test_isaac" / "trial_001" / "init"
    assert capture_dir.exists()
    assert (capture_dir / "state.json").exists()
    assert (capture_dir / "metadata.json").exists()


# ---------------------------------------------------------------------------
# Multi-camera tests
# ---------------------------------------------------------------------------


class MockMultiCameraEnv(MockIsaacLabEnv):
    """Mock environment that supports render_camera(camera_name) for multi-camera."""

    def __init__(self, num_envs: int = 1, render_mode: str = "rgb_array"):
        super().__init__(num_envs=num_envs, render_mode=render_mode)
        self._cameras = {"front", "wrist", "overhead"}

    def render_camera(self, camera_name: str) -> np.ndarray:
        """Render a named camera view."""
        if camera_name not in self._cameras:
            raise ValueError(f"Unknown camera: {camera_name}")
        # Return different colored frames per camera for distinguishability
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        color_map = {"front": 0, "wrist": 1, "overhead": 2}
        frame[:, :, color_map.get(camera_name, 0)] = 128
        return frame


class MockIsaacLabTiledCameraEnv(MockIsaacLabEnv):
    """Mock environment with Isaac Lab TiledCamera via scene attribute."""

    def __init__(self, num_envs: int = 1, render_mode: str = "rgb_array"):
        super().__init__(num_envs=num_envs, render_mode=render_mode)

        class _CameraData:
            def __init__(self, h: int = 480, w: int = 640):
                self.output = {"rgb": np.zeros((h, w, 3), dtype=np.uint8)}

        class _TiledCamera:
            __name__ = "TiledCamera"

            def __init__(self) -> None:
                self.data = _CameraData()

        class _Scene(dict):
            """Minimal scene mock that acts like an Isaac Lab InteractiveScene."""

        self.scene = _Scene({"tiled_camera": _TiledCamera()})


def test_detect_render_camera_capability(tmp_path):
    """Wrapper should detect render_camera method on environment."""
    from roboharness.wrappers.gymnasium_wrapper import (
        MultiCameraCapability,
    )

    env = MockMultiCameraEnv(num_envs=1)
    wrapped = RobotHarnessWrapper(
        env,
        cameras=["front", "wrist"],
        checkpoints=[{"name": "cp", "step": 1}],
        output_dir=tmp_path,
    )
    assert wrapped.camera_capability == MultiCameraCapability.RENDER_CAMERA
    assert wrapped.has_multi_camera is True


def test_detect_isaac_tiled_camera_capability(tmp_path):
    """Wrapper should detect Isaac Lab TiledCamera via scene."""
    from roboharness.wrappers.gymnasium_wrapper import (
        MultiCameraCapability,
    )

    env = MockIsaacLabTiledCameraEnv(num_envs=1)
    wrapped = RobotHarnessWrapper(
        env,
        cameras=["tiled_camera"],
        checkpoints=[{"name": "cp", "step": 1}],
        output_dir=tmp_path,
    )
    assert wrapped.camera_capability == MultiCameraCapability.ISAAC_TILED
    assert wrapped.has_multi_camera is True


def test_detect_no_multi_camera(tmp_path):
    """Standard env without multi-camera should be detected as NONE."""
    from roboharness.wrappers.gymnasium_wrapper import (
        MultiCameraCapability,
    )

    env = MockIsaacLabEnv(num_envs=1)
    wrapped = RobotHarnessWrapper(
        env,
        cameras=["default"],
        checkpoints=[{"name": "cp", "step": 1}],
        output_dir=tmp_path,
    )
    assert wrapped.camera_capability == MultiCameraCapability.NONE
    assert wrapped.has_multi_camera is False


def test_multi_camera_capture_render_camera(tmp_path):
    """Multi-camera env should save separate image per camera at checkpoint."""
    env = MockMultiCameraEnv(num_envs=1)
    wrapped = RobotHarnessWrapper(
        env,
        cameras=["front", "wrist"],
        checkpoints=[{"name": "grasp", "step": 1}],
        output_dir=tmp_path,
        task_name="multi_cam",
    )
    wrapped.reset()
    _, _, _, _, info = wrapped.step(torch.zeros(1, *env.action_space.shape))

    assert "checkpoint" in info
    files = info["checkpoint"]["files"]

    # Each camera should have its own RGB file
    assert "front_rgb" in files
    assert "wrist_rgb" in files

    capture_dir = tmp_path / "multi_cam" / "trial_001" / "grasp"
    assert (capture_dir / "front_rgb.png").exists() or (
        capture_dir / "front_rgb.npy"
    ).exists()
    assert (capture_dir / "wrist_rgb.png").exists() or (
        capture_dir / "wrist_rgb.npy"
    ).exists()

    # Metadata should list captured cameras
    import json

    meta = json.loads((capture_dir / "metadata.json").read_text())
    assert "front" in meta["cameras"]
    assert "wrist" in meta["cameras"]
    assert meta["camera_capability"] == "render_camera"


def test_multi_camera_capture_isaac_tiled(tmp_path):
    """Isaac Lab TiledCamera should be captured via scene sensor data."""
    env = MockIsaacLabTiledCameraEnv(num_envs=1)
    wrapped = RobotHarnessWrapper(
        env,
        cameras=["tiled_camera"],
        checkpoints=[{"name": "cp", "step": 1}],
        output_dir=tmp_path,
        task_name="isaac_tiled",
    )
    wrapped.reset()
    _, _, _, _, info = wrapped.step(torch.zeros(1, *env.action_space.shape))

    assert "checkpoint" in info
    files = info["checkpoint"]["files"]
    assert "tiled_camera_rgb" in files

    import json

    capture_dir = tmp_path / "isaac_tiled" / "trial_001" / "cp"
    meta = json.loads((capture_dir / "metadata.json").read_text())
    assert "tiled_camera" in meta["cameras"]
    assert meta["camera_capability"] == "isaac_tiled"


def test_single_camera_fallback_still_works(tmp_path):
    """Env without multi-camera should still capture default view."""
    env = MockIsaacLabEnv(num_envs=1)
    wrapped = RobotHarnessWrapper(
        env,
        cameras=["front", "side"],  # requested but env doesn't support named cameras
        checkpoints=[{"name": "cp", "step": 1}],
        output_dir=tmp_path,
        task_name="fallback",
    )
    wrapped.reset()
    _, _, _, _, info = wrapped.step(torch.zeros(1, *env.action_space.shape))

    assert "checkpoint" in info
    files = info["checkpoint"]["files"]
    # Should fall back to "default" since env doesn't support named cameras
    assert "default_rgb" in files

    import json

    capture_dir = tmp_path / "fallback" / "trial_001" / "cp"
    meta = json.loads((capture_dir / "metadata.json").read_text())
    assert meta["cameras"] == ["default"]
    assert meta["camera_capability"] == "none"
