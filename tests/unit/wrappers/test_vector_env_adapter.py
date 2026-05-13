"""Tests for VectorEnvAdapter — validates batch-squeezing from VectorEnv to gym.Env.

Uses real Gymnasium SyncVectorEnv wrapping lightweight envs. No mocks.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import pytest

gym = pytest.importorskip("gymnasium", reason="gymnasium not installed")

from gymnasium import spaces  # noqa: E402

from roboharness.wrappers import RobotHarnessWrapper, VectorEnvAdapter  # noqa: E402

# ---------------------------------------------------------------------------
# Lightweight test environments
# ---------------------------------------------------------------------------


class FlatEnv(gym.Env):
    """Simple env with flat Box obs/act spaces (robot-like dimensions)."""

    metadata: ClassVar[dict[str, Any]] = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, render_mode: str = "rgb_array"):
        super().__init__()
        self.render_mode = render_mode
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float64)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float64)
        self._step_count = 0
        self.custom_attr = "hello"

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed, options=options)
        self._step_count = 0
        return np.zeros(12, dtype=np.float64), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self._step_count += 1
        obs = np.ones(12, dtype=np.float64) * self._step_count * 0.1
        terminated = self._step_count >= 100
        return obs, 1.5, terminated, False, {"sim_time": self._step_count * 0.02}

    def render(self) -> np.ndarray:
        return np.full((64, 64, 3), self._step_count, dtype=np.uint8)


class DictObsEnv(gym.Env):
    """Env with Dict observation space."""

    metadata: ClassVar[dict[str, Any]] = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, render_mode: str = "rgb_array"):
        super().__init__()
        self.render_mode = render_mode
        self.observation_space = spaces.Dict(
            {
                "joints": spaces.Box(low=-np.pi, high=np.pi, shape=(6,), dtype=np.float32),
                "image": spaces.Box(low=0, high=255, shape=(32, 32, 3), dtype=np.uint8),
            }
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        self._step_count = 0

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed, options=options)
        self._step_count = 0
        return {
            "joints": np.zeros(6, dtype=np.float32),
            "image": np.zeros((32, 32, 3), dtype=np.uint8),
        }, {}

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        self._step_count += 1
        return (
            {
                "joints": np.ones(6, dtype=np.float32) * 0.5,
                "image": np.ones((32, 32, 3), dtype=np.uint8) * 100,
            },
            2.0,
            False,
            False,
            {},
        )

    def render(self) -> np.ndarray:
        return np.zeros((32, 32, 3), dtype=np.uint8)


class CameraEnv(FlatEnv):
    """FlatEnv with render_camera support for multi-camera testing."""

    def __init__(self, render_mode: str = "rgb_array"):
        super().__init__(render_mode=render_mode)
        self.cameras = ["front", "side"]

    def render_camera(self, camera_name: str) -> np.ndarray:
        if camera_name not in self.cameras:
            raise ValueError(f"Unknown camera: {camera_name}")
        return np.full((64, 64, 3), self.cameras.index(camera_name) * 50, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_vec_env(env_cls: type, **kwargs: Any) -> gym.vector.SyncVectorEnv:
    """Create a SyncVectorEnv with num_envs=1."""
    return gym.vector.SyncVectorEnv([lambda: env_cls(**kwargs)])


# ---------------------------------------------------------------------------
# Tests: construction
# ---------------------------------------------------------------------------


class TestConstruction:
    """VectorEnvAdapter construction and validation."""

    def test_accepts_sync_vector_env(self):
        vec = _make_vec_env(FlatEnv)
        adapter = VectorEnvAdapter(vec)
        assert adapter.observation_space.shape == (12,)
        assert adapter.action_space.shape == (4,)
        adapter.close()

    def test_rejects_multi_env(self):
        vec = gym.vector.SyncVectorEnv([lambda: FlatEnv(), lambda: FlatEnv()])
        with pytest.raises(ValueError, match="num_envs=1"):
            VectorEnvAdapter(vec)
        vec.close()

    def test_spaces_match_single_env(self):
        base = FlatEnv()
        vec = _make_vec_env(FlatEnv)
        adapter = VectorEnvAdapter(vec)

        assert adapter.observation_space == base.observation_space
        assert adapter.action_space == base.action_space
        adapter.close()

    def test_dict_obs_spaces(self):
        vec = _make_vec_env(DictObsEnv)
        adapter = VectorEnvAdapter(vec)

        assert isinstance(adapter.observation_space, spaces.Dict)
        assert "joints" in adapter.observation_space.spaces
        assert adapter.observation_space["joints"].shape == (6,)
        adapter.close()


# ---------------------------------------------------------------------------
# Tests: reset / step
# ---------------------------------------------------------------------------


class TestResetStep:
    """Core gym.Env interface via VectorEnvAdapter."""

    def test_reset_returns_unbatched_obs(self):
        vec = _make_vec_env(FlatEnv)
        adapter = VectorEnvAdapter(vec)

        obs, _info = adapter.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (12,)
        np.testing.assert_array_equal(obs, np.zeros(12))
        adapter.close()

    def test_step_returns_scalar_reward(self):
        vec = _make_vec_env(FlatEnv)
        adapter = VectorEnvAdapter(vec)
        adapter.reset()

        obs, reward, terminated, truncated, _info = adapter.step(np.zeros(4))

        assert obs.shape == (12,)
        assert reward == 1.5
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert not terminated
        assert not truncated
        adapter.close()

    def test_obs_values_match_underlying_env(self):
        vec = _make_vec_env(FlatEnv)
        adapter = VectorEnvAdapter(vec)
        adapter.reset()

        obs, _, _, _, _ = adapter.step(np.zeros(4))
        # FlatEnv step 1: obs = ones(12) * 0.1
        np.testing.assert_allclose(obs, np.ones(12) * 0.1)
        adapter.close()

    def test_terminated_propagates(self):
        vec = _make_vec_env(FlatEnv)
        adapter = VectorEnvAdapter(vec)
        adapter.reset()

        # FlatEnv terminates at step 100
        for _i in range(100):
            _, _, terminated, _, _ = adapter.step(np.zeros(4))
        assert terminated
        adapter.close()

    def test_dict_obs_squeezed(self):
        vec = _make_vec_env(DictObsEnv)
        adapter = VectorEnvAdapter(vec)

        obs, _ = adapter.reset()
        assert isinstance(obs, dict)
        assert obs["joints"].shape == (6,)
        assert obs["image"].shape == (32, 32, 3)

        obs, reward, _, _, _ = adapter.step(np.zeros(6, dtype=np.float32))
        assert obs["joints"].shape == (6,)
        assert reward == 2.0
        adapter.close()

    def test_info_squeezed(self):
        vec = _make_vec_env(FlatEnv)
        adapter = VectorEnvAdapter(vec)
        adapter.reset()

        _, _, _, _, info = adapter.step(np.zeros(4))
        # FlatEnv returns sim_time as a float, not array
        if "sim_time" in info:
            assert isinstance(info["sim_time"], (int, float, np.floating))
        adapter.close()


# ---------------------------------------------------------------------------
# Tests: render
# ---------------------------------------------------------------------------


class TestRender:
    """Render pass-through."""

    def test_render_returns_single_frame(self):
        vec = _make_vec_env(FlatEnv, render_mode="rgb_array")
        adapter = VectorEnvAdapter(vec)
        adapter.reset()

        frame = adapter.render()
        assert isinstance(frame, np.ndarray)
        # Should be 3D (H, W, C), not 4D (batch, H, W, C)
        assert frame.ndim == 3
        adapter.close()


# ---------------------------------------------------------------------------
# Tests: attribute proxying
# ---------------------------------------------------------------------------


class TestAttributeProxy:
    """Attribute access proxied to the underlying single env."""

    def test_custom_attr(self):
        vec = _make_vec_env(FlatEnv)
        adapter = VectorEnvAdapter(vec)
        assert adapter.custom_attr == "hello"
        adapter.close()

    def test_unwrapped_returns_single_env(self):
        vec = _make_vec_env(FlatEnv)
        adapter = VectorEnvAdapter(vec)
        assert isinstance(adapter.unwrapped, FlatEnv)
        adapter.close()

    def test_render_camera_proxied(self):
        vec = _make_vec_env(CameraEnv, render_mode="rgb_array")
        adapter = VectorEnvAdapter(vec)
        assert hasattr(adapter.unwrapped, "render_camera")
        frame = adapter.unwrapped.render_camera("front")
        assert frame.shape == (64, 64, 3)
        adapter.close()

    def test_cameras_proxied(self):
        vec = _make_vec_env(CameraEnv, render_mode="rgb_array")
        adapter = VectorEnvAdapter(vec)
        assert adapter.unwrapped.cameras == ["front", "side"]
        adapter.close()

    def test_missing_attr_raises(self):
        vec = _make_vec_env(FlatEnv)
        adapter = VectorEnvAdapter(vec)
        with pytest.raises(AttributeError):
            _ = adapter.nonexistent_attribute
        adapter.close()


# ---------------------------------------------------------------------------
# Tests: integration with RobotHarnessWrapper
# ---------------------------------------------------------------------------


class TestWrapperIntegration:
    """VectorEnvAdapter works as a drop-in for RobotHarnessWrapper."""

    def test_checkpoint_through_adapter(self, tmp_path):
        vec = _make_vec_env(FlatEnv, render_mode="rgb_array")
        adapter = VectorEnvAdapter(vec)

        wrapped = RobotHarnessWrapper(
            adapter,
            checkpoints=[{"name": "cp1", "step": 2}, {"name": "cp2", "step": 5}],
            output_dir=tmp_path,
            task_name="adapter_test",
        )
        wrapped.reset()

        captured = []
        for _ in range(5):
            _, _, _, _, info = wrapped.step(np.zeros(4))
            if "checkpoint" in info:
                captured.append(info["checkpoint"]["name"])

        assert captured == ["cp1", "cp2"]
        wrapped.close()

    def test_multi_camera_through_adapter(self, tmp_path):
        vec = _make_vec_env(CameraEnv, render_mode="rgb_array")
        adapter = VectorEnvAdapter(vec)

        wrapped = RobotHarnessWrapper(
            adapter,
            checkpoints=[{"name": "cp", "step": 1}],
            cameras=["front", "side"],
            output_dir=tmp_path,
            task_name="multicam_test",
        )
        wrapped.reset()
        wrapped.step(np.zeros(4))

        capture_dir = tmp_path / "multicam_test" / "trial_001" / "cp"
        image_files = sorted(f.name for f in capture_dir.glob("*_rgb.*"))
        assert len(image_files) == 2
        wrapped.close()

    def test_dict_obs_through_adapter(self, tmp_path):
        vec = _make_vec_env(DictObsEnv, render_mode="rgb_array")
        adapter = VectorEnvAdapter(vec)

        wrapped = RobotHarnessWrapper(
            adapter,
            checkpoints=[{"name": "cp", "step": 1}],
            output_dir=tmp_path,
            task_name="dict_test",
        )
        obs, _ = wrapped.reset()
        assert isinstance(obs, dict)
        assert obs["joints"].shape == (6,)

        obs, reward, _, _, info = wrapped.step(np.zeros(6, dtype=np.float32))
        assert isinstance(obs, dict)
        assert reward == 2.0
        assert "checkpoint" in info
        wrapped.close()
