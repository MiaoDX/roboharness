"""Tests for RobotHarnessWrapper edge cases — no torch/mujoco required.

Exercises code paths in gymnasium_wrapper.py that are normally only hit
with torch tensors, Isaac Lab scenes, or old-gym APIs.  Uses duck-typing
mocks so these run in every CI environment (only ``[dev]`` deps needed).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest

gym = pytest.importorskip("gymnasium", reason="gymnasium not installed")

from gymnasium import spaces  # noqa: E402

from roboharness.wrappers import RobotHarnessWrapper  # noqa: E402
from roboharness.wrappers.gymnasium_wrapper import (  # noqa: E402
    MultiCameraCapability,
    _capture_frame_from_env,
    _detect_camera_capability,
    _to_float,
    _to_numpy_rgb,
)

# ---------------------------------------------------------------------------
# Mock helpers — duck-typing mocks that quack like torch tensors
# ---------------------------------------------------------------------------


class FakeTensor:
    """Mock object that satisfies the duck-typing checks for torch tensors."""

    def __init__(self, data: np.ndarray):
        self._data = data
        self.shape = data.shape
        self.dtype = "float32"

    def detach(self) -> FakeTensor:
        return self

    def cpu(self) -> FakeTensor:
        return self

    def numpy(self) -> np.ndarray:
        return self._data

    def item(self) -> float:
        return float(self._data.flat[0])

    def float(self) -> FakeTensor:
        return self

    def mean(self) -> FakeTensor:
        return FakeTensor(np.array([float(self._data.mean())]))

    def numel(self) -> int:
        return self._data.size


class FakeCamera:
    """Mock Isaac Lab TiledCamera sensor."""

    class _Data:
        def __init__(self, rgb: np.ndarray):
            self.output = {"rgb": rgb}

    def __init__(self, rgb: np.ndarray | None = None):
        if rgb is None:
            rgb = np.zeros((64, 64, 3), dtype=np.uint8)
        self.data = self._Data(rgb)


class DictScene:
    """Mock Isaac Lab scene with dict-like access (has .keys())."""

    def __init__(self, sensors: dict[str, Any]):
        self._sensors = sensors

    def keys(self) -> Any:
        return self._sensors.keys()

    def __getitem__(self, key: str) -> Any:
        return self._sensors[key]

    def __contains__(self, key: str) -> bool:
        return key in self._sensors

    def __iter__(self):
        return iter(self._sensors)


class IterableScene:
    """Mock Isaac Lab scene with iterable-only access (no .keys())."""

    def __init__(self, sensors: list[Any]):
        self._sensors = sensors

    def __iter__(self):
        return iter(self._sensors)


class SimpleEnv(gym.Env):
    """Minimal Gymnasium env for testing."""

    metadata: ClassVar[dict[str, Any]] = {"render_modes": ["rgb_array"]}

    def __init__(self, obs_shape: tuple[int, ...] = (10,)):
        super().__init__()
        self.render_mode = "rgb_array"
        self.observation_space = spaces.Box(-1, 1, shape=obs_shape, dtype=np.float32)
        self.action_space = spaces.Box(-1, 1, shape=(2,), dtype=np.float32)

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(**kwargs)
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        return np.zeros(self.observation_space.shape, dtype=np.float32), 0.0, False, False, {}

    def render(self) -> np.ndarray:
        return np.zeros((64, 64, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Tests for _to_numpy_rgb
# ---------------------------------------------------------------------------


class TestToNumpyRgb:
    def test_none_returns_none(self):
        assert _to_numpy_rgb(None) is None

    def test_numpy_array_passthrough(self):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        result = _to_numpy_rgb(frame)
        assert result is frame

    def test_tensor_like_float_scaled_to_uint8(self):
        data = np.random.rand(64, 64, 3).astype(np.float32)  # values in [0, 1]
        fake = FakeTensor(data)
        result = _to_numpy_rgb(fake)
        assert result is not None
        assert result.dtype == np.uint8
        assert result.max() <= 255

    def test_tensor_like_large_float_cast_to_uint8(self):
        data = (np.random.rand(64, 64, 3) * 200 + 56).astype(np.float32)
        fake = FakeTensor(data)
        result = _to_numpy_rgb(fake)
        assert result is not None
        assert result.dtype == np.uint8

    def test_tensor_like_uint8_passthrough(self):
        data = np.zeros((64, 64, 3), dtype=np.uint8)
        fake = FakeTensor(data)
        result = _to_numpy_rgb(fake)
        assert result is not None
        assert result.dtype == np.uint8

    def test_unknown_type_returns_none(self):
        assert _to_numpy_rgb("not a frame") is None
        assert _to_numpy_rgb(42) is None


# ---------------------------------------------------------------------------
# Tests for _to_float
# ---------------------------------------------------------------------------


class TestToFloat:
    def test_python_int(self):
        assert _to_float(3) == 3.0

    def test_python_float(self):
        assert _to_float(2.5) == 2.5

    def test_numpy_scalar(self):
        assert _to_float(np.float64(1.5)) == 1.5
        assert _to_float(np.int32(7)) == 7.0

    def test_numpy_array_scalar(self):
        assert _to_float(np.array(3.14)) == pytest.approx(3.14)

    def test_numpy_array_vector_returns_mean(self):
        arr = np.array([1.0, 2.0, 3.0])
        assert _to_float(arr) == pytest.approx(2.0)

    def test_numpy_empty_array_returns_zero(self):
        assert _to_float(np.array([])) == 0.0

    def test_tensor_like_scalar(self):
        fake = FakeTensor(np.array([5.0]))
        assert _to_float(fake) == pytest.approx(5.0)

    def test_tensor_like_vector_returns_mean(self):
        fake = FakeTensor(np.array([1.0, 2.0, 3.0, 4.0]))
        assert _to_float(fake) == pytest.approx(2.5)

    def test_tensor_like_with_size_attr(self):
        """Object with .item() and .size (int) but no .numel() — e.g. numpy subclass."""

        class SizedTensor:
            size = 3

            def item(self):
                return 2.0  # pragma: no cover — not reached

            def mean(self):
                return 2.0

        assert _to_float(SizedTensor()) == pytest.approx(2.0)

    def test_unconvertible_returns_zero(self):
        assert _to_float(object()) == 0.0

    def test_string_number_fallback(self):
        # str "3.14" is convertible via float()
        assert _to_float("3.14") == pytest.approx(3.14)


# ---------------------------------------------------------------------------
# Tests for Isaac Lab camera detection
# ---------------------------------------------------------------------------


class TestIsaacLabDetection:
    def test_dict_scene_with_camera(self):
        scene = DictScene({"front_camera": FakeCamera()})
        env = SimpleEnv()
        env.unwrapped.scene = scene  # type: ignore[attr-defined]
        assert _detect_camera_capability(env) == MultiCameraCapability.ISAAC_TILED

    def test_dict_scene_without_camera(self):
        scene = DictScene({"lidar": type("Lidar", (), {})()})
        env = SimpleEnv()
        env.unwrapped.scene = scene  # type: ignore[attr-defined]
        assert _detect_camera_capability(env) == MultiCameraCapability.NONE

    def test_iterable_scene_with_camera(self):
        scene = IterableScene([FakeCamera()])
        env = SimpleEnv()
        env.unwrapped.scene = scene  # type: ignore[attr-defined]
        assert _detect_camera_capability(env) == MultiCameraCapability.ISAAC_TILED

    def test_iterable_scene_without_camera(self):
        scene = IterableScene([type("Lidar", (), {})()])
        env = SimpleEnv()
        env.unwrapped.scene = scene  # type: ignore[attr-defined]
        assert _detect_camera_capability(env) == MultiCameraCapability.NONE

    def test_render_camera_takes_priority(self):
        env = SimpleEnv()
        env.unwrapped.render_camera = lambda name: np.zeros((64, 64, 3), dtype=np.uint8)  # type: ignore[attr-defined]
        env.unwrapped.scene = DictScene({"cam": FakeCamera()})  # type: ignore[attr-defined]
        assert _detect_camera_capability(env) == MultiCameraCapability.RENDER_CAMERA


# ---------------------------------------------------------------------------
# Tests for Isaac Lab frame capture
# ---------------------------------------------------------------------------


class TestIsaacLabCapture:
    def test_isaac_tiled_capture(self):
        rgb = np.full((64, 64, 3), 42, dtype=np.uint8)
        scene = DictScene({"front": FakeCamera(rgb)})
        env = SimpleEnv()
        env.unwrapped.scene = scene  # type: ignore[attr-defined]

        frame = _capture_frame_from_env(env, "front", MultiCameraCapability.ISAAC_TILED)
        assert frame is not None
        assert frame.shape == (64, 64, 3)
        assert np.all(frame == 42)

    def test_isaac_tiled_missing_camera_falls_back(self):
        """If the camera name isn't in the scene, falls back to env.render()."""
        scene = DictScene({"front": FakeCamera()})
        env = SimpleEnv()
        env.unwrapped.scene = scene  # type: ignore[attr-defined]

        frame = _capture_frame_from_env(env, "nonexistent", MultiCameraCapability.ISAAC_TILED)
        # Falls back to env.render() which returns zeros
        assert frame is not None

    def test_capture_returns_none_on_failure(self):
        """If everything fails (no render), returns None."""
        env = SimpleEnv()
        env.render = lambda: None  # type: ignore[assignment]
        frame = _capture_frame_from_env(env, "default", MultiCameraCapability.NONE)
        assert frame is None


# ---------------------------------------------------------------------------
# Tests for old-gym single-value reset
# ---------------------------------------------------------------------------


class OldGymEnv(gym.Env):
    """Env whose reset() returns a non-tuple (old gym API)."""

    metadata: ClassVar[dict[str, Any]] = {"render_modes": ["rgb_array"]}

    def __init__(self):
        super().__init__()
        self.render_mode = "rgb_array"
        self.observation_space = spaces.Box(-1, 1, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Box(-1, 1, shape=(2,), dtype=np.float32)

    def reset(self, **kwargs: Any) -> np.ndarray:  # type: ignore[override]
        # Old gym: returns obs only, no info dict
        return np.zeros(4, dtype=np.float32)

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        return np.zeros(4, dtype=np.float32), 0.0, False, False, {}

    def render(self) -> np.ndarray:
        return np.zeros((64, 64, 3), dtype=np.uint8)


class TestOldGymCompat:
    def test_single_value_reset(self):
        env = OldGymEnv()
        wrapped = RobotHarnessWrapper(
            env, checkpoints=[{"name": "cp", "step": 1}], output_dir="/tmp/test_old_gym"
        )
        obs, info = wrapped.reset()
        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)


# ---------------------------------------------------------------------------
# Tests for tensor-like obs summary in checkpoint state.json
# ---------------------------------------------------------------------------


class TensorObsEnv(gym.Env):
    """Env that returns tensor-like observations."""

    metadata: ClassVar[dict[str, Any]] = {"render_modes": ["rgb_array"]}

    def __init__(self):
        super().__init__()
        self.render_mode = "rgb_array"
        self.observation_space = spaces.Box(-1, 1, shape=(10,), dtype=np.float32)
        self.action_space = spaces.Box(-1, 1, shape=(2,), dtype=np.float32)

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        return FakeTensor(np.zeros(10, dtype=np.float32)), {}

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        return FakeTensor(np.ones(10, dtype=np.float32)), 1.0, False, False, {}

    def render(self) -> np.ndarray:
        return np.zeros((64, 64, 3), dtype=np.uint8)


class TestTensorObsSummary:
    def test_checkpoint_records_tensor_obs_shape(self, tmp_path):
        env = TensorObsEnv()
        wrapped = RobotHarnessWrapper(
            env,
            checkpoints=[{"name": "cp1", "step": 1}],
            output_dir=str(tmp_path),
        )
        wrapped.reset()
        _obs, _reward, _term, _trunc, info = wrapped.step(np.zeros(2))

        assert "checkpoint" in info
        state_path = info["checkpoint"]["files"]["state"]
        state = json.loads(Path(state_path).read_text())
        assert state["obs_shape"] == [10]
        assert state["obs_dtype"] == "float32"


# ---------------------------------------------------------------------------
# Tests for tensor-like reward in _to_float via checkpoint
# ---------------------------------------------------------------------------


class TensorRewardEnv(gym.Env):
    """Env that returns tensor-like rewards."""

    metadata: ClassVar[dict[str, Any]] = {"render_modes": ["rgb_array"]}

    def __init__(self, reward_value: Any = 1.0):
        super().__init__()
        self.render_mode = "rgb_array"
        self.observation_space = spaces.Box(-1, 1, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Box(-1, 1, shape=(2,), dtype=np.float32)
        self._reward = reward_value

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action: Any) -> tuple[np.ndarray, Any, bool, bool, dict[str, Any]]:
        return np.zeros(4, dtype=np.float32), self._reward, False, False, {}

    def render(self) -> np.ndarray:
        return np.zeros((64, 64, 3), dtype=np.uint8)


class TestTensorRewardCheckpoint:
    def test_tensor_scalar_reward_in_state(self, tmp_path):
        reward = FakeTensor(np.array([3.14]))
        env = TensorRewardEnv(reward_value=reward)
        wrapped = RobotHarnessWrapper(
            env,
            checkpoints=[{"name": "cp1", "step": 1}],
            output_dir=str(tmp_path),
        )
        wrapped.reset()
        _obs, _reward, _term, _trunc, info = wrapped.step(np.zeros(2))

        state = json.loads(Path(info["checkpoint"]["files"]["state"]).read_text())
        assert state["reward"] == pytest.approx(3.14)

    def test_tensor_vector_reward_in_state(self, tmp_path):
        reward = FakeTensor(np.array([1.0, 2.0, 3.0, 4.0]))
        env = TensorRewardEnv(reward_value=reward)
        wrapped = RobotHarnessWrapper(
            env,
            checkpoints=[{"name": "cp1", "step": 1}],
            output_dir=str(tmp_path),
        )
        wrapped.reset()
        _obs, _reward, _term, _trunc, info = wrapped.step(np.zeros(2))

        state = json.loads(Path(info["checkpoint"]["files"]["state"]).read_text())
        assert state["reward"] == pytest.approx(2.5)

    def test_numpy_scalar_reward_in_state(self, tmp_path):
        env = TensorRewardEnv(reward_value=np.float64(7.5))
        wrapped = RobotHarnessWrapper(
            env,
            checkpoints=[{"name": "cp1", "step": 1}],
            output_dir=str(tmp_path),
        )
        wrapped.reset()
        _obs, _reward, _term, _trunc, info = wrapped.step(np.zeros(2))

        state = json.loads(Path(info["checkpoint"]["files"]["state"]).read_text())
        assert state["reward"] == pytest.approx(7.5)
