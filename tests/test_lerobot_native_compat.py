"""Tests for native LeRobot integration — validates VectorEnv adapter + RobotHarnessWrapper.

The native LeRobot path uses ``make_env()`` which returns a ``VectorEnv``.
The ``_VectorEnvAdapter`` in ``examples/lerobot_g1_native.py`` unbatches it
into a standard Gymnasium Env.

These tests use real Gymnasium VectorEnvs (no mocks) to validate:
  - VectorEnv (n=1) is correctly unbatched to single-env interface
  - Dict observations are unbatched per-key
  - RobotHarnessWrapper captures checkpoints through the adapter
  - Validation logic in the example works correctly

No LeRobot or MuJoCo installation is required.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest

gym = pytest.importorskip("gymnasium", reason="gymnasium not installed")

from gymnasium import spaces  # noqa: E402

from roboharness.wrappers import RobotHarnessWrapper  # noqa: E402

# ---------------------------------------------------------------------------
# Real lightweight Gymnasium envs for testing
# ---------------------------------------------------------------------------


class SimpleG1Env(gym.Env):
    """Lightweight single env matching G1 dimensions (99-dim obs, 29-dim act)."""

    metadata: ClassVar[dict[str, Any]] = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, render_mode: str = "rgb_array"):
        super().__init__()
        self.render_mode = render_mode
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(99,), dtype=np.float64)
        self.action_space = spaces.Box(low=-np.pi, high=np.pi, shape=(29,), dtype=np.float64)
        self._step_count = 0

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed, options=options)
        self._step_count = 0
        return np.zeros(99, dtype=np.float64), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self._step_count += 1
        rng = np.random.default_rng(self._step_count)
        obs = rng.standard_normal(99).astype(np.float64) * 0.1
        return obs, 1.75, False, False, {"sim_time": self._step_count * 0.004}

    def render(self) -> np.ndarray:
        return np.zeros((480, 640, 3), dtype=np.uint8)


class DictObsEnv(gym.Env):
    """Env with dict observations (common in LeRobot policy envs)."""

    metadata: ClassVar[dict[str, Any]] = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, render_mode: str = "rgb_array"):
        super().__init__()
        self.render_mode = render_mode
        self.observation_space = spaces.Dict(
            {
                "observation.state": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(29,), dtype=np.float32
                ),
                "observation.images.head": spaces.Box(
                    low=0, high=255, shape=(64, 64, 3), dtype=np.uint8
                ),
            }
        )
        self.action_space = spaces.Box(low=-np.pi, high=np.pi, shape=(29,), dtype=np.float64)
        self._step_count = 0

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed, options=options)
        self._step_count = 0
        obs = {
            "observation.state": np.zeros(29, dtype=np.float32),
            "observation.images.head": np.zeros((64, 64, 3), dtype=np.uint8),
        }
        return obs, {}

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        self._step_count += 1
        obs = {
            "observation.state": np.ones(29, dtype=np.float32) * 0.1,
            "observation.images.head": np.ones((64, 64, 3), dtype=np.uint8) * 128,
        }
        return obs, 1.0, False, False, {}

    def render(self) -> np.ndarray:
        return np.zeros((480, 640, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Fixture: import the example module
# ---------------------------------------------------------------------------


@pytest.fixture()
def native_module():
    """Import examples/lerobot_g1_native.py as a module."""
    example_path = Path(__file__).parent.parent / "examples"
    sys.path.insert(0, str(example_path))
    try:
        import lerobot_g1_native

        yield lerobot_g1_native
    finally:
        sys.path.pop(0)
        sys.modules.pop("lerobot_g1_native", None)


# ---------------------------------------------------------------------------
# Tests: VectorEnv adapter unbatching
# ---------------------------------------------------------------------------


def _make_vector_env(env_fn):
    """Create a real SyncVectorEnv with n=1 from an env factory."""
    return gym.vector.SyncVectorEnv([env_fn])


class TestVectorEnvAdapter:
    """Tests for _VectorEnvAdapter with real SyncVectorEnv."""

    def test_reset_unbatches_flat_obs(self, native_module):
        vec_env = _make_vector_env(SimpleG1Env)
        adapter = native_module._VectorEnvAdapter(vec_env)
        obs, _info = adapter.reset()

        assert isinstance(obs, np.ndarray)
        assert obs.shape == (99,)

    def test_step_returns_scalars(self, native_module):
        vec_env = _make_vector_env(SimpleG1Env)
        adapter = native_module._VectorEnvAdapter(vec_env)
        adapter.reset()

        action = np.zeros(29, dtype=np.float64)
        obs, reward, terminated, truncated, _info = adapter.step(action)

        assert obs.shape == (99,)
        assert isinstance(reward, float)
        assert abs(reward - 1.75) < 0.01
        assert terminated is False
        assert truncated is False

    def test_render_single_frame(self, native_module):
        vec_env = _make_vector_env(SimpleG1Env)
        adapter = native_module._VectorEnvAdapter(vec_env)
        adapter.reset()
        frame = adapter.render()

        assert isinstance(frame, np.ndarray)
        assert frame.shape == (480, 640, 3)

    def test_dict_obs_unbatched(self, native_module):
        vec_env = _make_vector_env(DictObsEnv)
        adapter = native_module._VectorEnvAdapter(vec_env)
        obs, _info = adapter.reset()

        assert isinstance(obs, dict)
        assert obs["observation.state"].shape == (29,)
        assert obs["observation.images.head"].shape == (64, 64, 3)

    def test_spaces_are_single(self, native_module):
        vec_env = _make_vector_env(SimpleG1Env)
        adapter = native_module._VectorEnvAdapter(vec_env)

        assert adapter.observation_space.shape == (99,)
        assert adapter.action_space.shape == (29,)

    def test_close(self, native_module):
        vec_env = _make_vector_env(SimpleG1Env)
        adapter = native_module._VectorEnvAdapter(vec_env)
        adapter.close()  # should not raise


# ---------------------------------------------------------------------------
# Tests: RobotHarnessWrapper through adapter
# ---------------------------------------------------------------------------


class TestWrapperWithAdapter:
    """RobotHarnessWrapper integration with VectorEnvAdapter."""

    def test_checkpoint_captured(self, native_module, tmp_path):
        vec_env = _make_vector_env(SimpleG1Env)
        adapter = native_module._VectorEnvAdapter(vec_env)
        wrapped = RobotHarnessWrapper(
            adapter,
            checkpoints=[{"name": "cp", "step": 3}],
            output_dir=tmp_path,
            task_name="native_test",
        )
        wrapped.reset()
        for _ in range(3):
            _, _, _, _, info = wrapped.step(np.zeros(29))

        assert "checkpoint" in info
        state_path = tmp_path / "native_test" / "trial_001" / "cp" / "state.json"
        assert state_path.exists()
        state = json.loads(state_path.read_text())
        assert state["step"] == 3
        assert isinstance(state["reward"], float)

    def test_obs_passthrough(self, native_module, tmp_path):
        vec_env = _make_vector_env(SimpleG1Env)
        adapter = native_module._VectorEnvAdapter(vec_env)
        wrapped = RobotHarnessWrapper(
            adapter,
            checkpoints=[{"name": "cp", "step": 100}],
            output_dir=tmp_path,
        )
        obs, _ = wrapped.reset()
        assert obs.shape == (99,)

        obs, reward, _, _, _ = wrapped.step(np.zeros(29))
        assert obs.shape == (99,)
        assert isinstance(reward, float)

    def test_dict_obs_records_keys(self, native_module, tmp_path):
        vec_env = _make_vector_env(DictObsEnv)
        adapter = native_module._VectorEnvAdapter(vec_env)
        wrapped = RobotHarnessWrapper(
            adapter,
            checkpoints=[{"name": "cp", "step": 1}],
            output_dir=tmp_path,
            task_name="dict_obs",
        )
        wrapped.reset()
        wrapped.step(np.zeros(29))

        state_path = tmp_path / "dict_obs" / "trial_001" / "cp" / "state.json"
        state = json.loads(state_path.read_text())
        assert "obs_keys" in state
        assert "observation.state" in state["obs_keys"]

    def test_multiple_checkpoints(self, native_module, tmp_path):
        vec_env = _make_vector_env(SimpleG1Env)
        adapter = native_module._VectorEnvAdapter(vec_env)
        wrapped = RobotHarnessWrapper(
            adapter,
            checkpoints=[
                {"name": "initial", "step": 1},
                {"name": "mid", "step": 5},
                {"name": "final", "step": 10},
            ],
            output_dir=tmp_path,
            task_name="multi_cp",
        )
        wrapped.reset()

        captured = []
        for _ in range(10):
            _, _, _, _, info = wrapped.step(np.zeros(29))
            if "checkpoint" in info:
                captured.append(info["checkpoint"]["name"])

        assert captured == ["initial", "mid", "final"]

    def test_render_image_saved(self, native_module, tmp_path):
        vec_env = _make_vector_env(SimpleG1Env)
        adapter = native_module._VectorEnvAdapter(vec_env)
        wrapped = RobotHarnessWrapper(
            adapter,
            checkpoints=[{"name": "cp", "step": 1}],
            output_dir=tmp_path,
            task_name="render_test",
        )
        wrapped.reset()
        wrapped.step(np.zeros(29))

        capture_dir = tmp_path / "render_test" / "trial_001" / "cp"
        image_files = list(capture_dir.glob("*_rgb.*"))
        assert len(image_files) > 0


# ---------------------------------------------------------------------------
# Tests: validation logic
# ---------------------------------------------------------------------------


class TestValidation:
    """Tests for validate_integration in the example."""

    def test_valid_checkpoints_pass(self, native_module, tmp_path):
        cp_dir = tmp_path / "cp1"
        cp_dir.mkdir()
        (cp_dir / "state.json").write_text(json.dumps({"step": 1, "reward": 1.5}))

        infos = [{"name": "cp1", "step": 1, "files": {"state": str(cp_dir / "state.json")}}]
        failures = native_module.validate_integration(infos, expected_count=1)
        assert failures == []

    def test_wrong_count_fails(self, native_module):
        failures = native_module.validate_integration([], expected_count=3)
        assert any("Expected 3" in f for f in failures)

    def test_missing_state_fails(self, native_module):
        infos = [{"name": "cp1", "step": 1, "files": {"state": "/nonexistent/state.json"}}]
        failures = native_module.validate_integration(infos, expected_count=1)
        assert any("missing state.json" in f for f in failures)
