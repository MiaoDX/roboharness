"""Tests for native LeRobot integration — validates RobotHarnessWrapper with G1-like envs.

The native LeRobot path imports the hub env module directly and wraps the
resulting Gymnasium Env with RobotHarnessWrapper.

These tests use real Gymnasium envs (no mocks) to validate:
  - RobotHarnessWrapper captures checkpoints through a G1-like env
  - Dict observations are handled correctly
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


class MismatchedObsSpaceEnv(gym.Env):
    """Env that declares obs space (97,) but returns (100,) — mimics upstream G1 bug.

    The lerobot/unitree-g1-mujoco env declares shape=(num_joints * 3 + 10,) = (97,)
    but _get_obs() returns 100 elements because floating_base_acc is 6-D not 3-D.
    See: https://github.com/MiaoDX/roboharness/issues/110
    """

    metadata: ClassVar[dict[str, Any]] = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, render_mode: str = "rgb_array"):
        super().__init__()
        self.render_mode = render_mode
        # Deliberately wrong: declares 97 but returns 100
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(97,), dtype=np.float64)
        self.action_space = spaces.Box(low=-np.pi, high=np.pi, shape=(29,), dtype=np.float64)
        self._step_count = 0

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed, options=options)
        self._step_count = 0
        return np.zeros(100, dtype=np.float64), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self._step_count += 1
        return np.zeros(100, dtype=np.float64), 1.0, False, False, {}

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
# Tests: RobotHarnessWrapper with G1-like envs
# ---------------------------------------------------------------------------


class TestWrapperWithG1Env:
    """RobotHarnessWrapper integration with G1-like Gymnasium envs."""

    def test_checkpoint_captured(self, tmp_path):
        env = SimpleG1Env()
        wrapped = RobotHarnessWrapper(
            env,
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

    def test_obs_passthrough(self, tmp_path):
        env = SimpleG1Env()
        wrapped = RobotHarnessWrapper(
            env,
            checkpoints=[{"name": "cp", "step": 100}],
            output_dir=tmp_path,
        )
        obs, _ = wrapped.reset()
        assert obs.shape == (99,)

        obs, reward, _, _, _ = wrapped.step(np.zeros(29))
        assert obs.shape == (99,)
        assert isinstance(reward, float)

    def test_dict_obs_records_keys(self, tmp_path):
        env = DictObsEnv()
        wrapped = RobotHarnessWrapper(
            env,
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

    def test_multiple_checkpoints(self, tmp_path):
        env = SimpleG1Env()
        wrapped = RobotHarnessWrapper(
            env,
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

    def test_render_image_saved(self, tmp_path):
        env = SimpleG1Env()
        wrapped = RobotHarnessWrapper(
            env,
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
# Tests: obs-space mismatch auto-fix (issue #110)
# ---------------------------------------------------------------------------


class TestObsSpaceAutoFix:
    """RobotHarnessWrapper auto_fix_obs_space for upstream shape mismatches."""

    def test_mismatch_fixed_on_first_reset(self, tmp_path):
        """With auto_fix_obs_space=True, obs space is corrected on first reset."""
        env = MismatchedObsSpaceEnv()
        assert env.observation_space.shape == (97,)

        wrapped = RobotHarnessWrapper(
            env,
            checkpoints=[{"name": "cp", "step": 1}],
            output_dir=tmp_path,
            auto_fix_obs_space=True,
        )
        assert wrapped.observation_space.shape == (97,)  # Before reset: still wrong

        obs, _ = wrapped.reset()
        assert obs.shape == (100,)
        assert wrapped.observation_space.shape == (100,)  # After reset: corrected

    def test_mismatch_not_fixed_without_flag(self, tmp_path):
        """Without auto_fix_obs_space, obs space stays at declared (wrong) shape."""
        env = MismatchedObsSpaceEnv()
        wrapped = RobotHarnessWrapper(
            env,
            checkpoints=[{"name": "cp", "step": 1}],
            output_dir=tmp_path,
        )
        wrapped.reset()
        assert wrapped.observation_space.shape == (97,)  # Still wrong — no auto-fix

    def test_no_fix_when_shapes_match(self, tmp_path):
        """When declared and actual shapes match, no fix is applied."""
        env = SimpleG1Env()
        wrapped = RobotHarnessWrapper(
            env,
            checkpoints=[{"name": "cp", "step": 1}],
            output_dir=tmp_path,
            auto_fix_obs_space=True,
        )
        wrapped.reset()
        assert wrapped.observation_space.shape == (99,)  # Unchanged

    def test_dict_obs_skipped(self, tmp_path):
        """Dict observations are skipped by auto-fix (no shape to compare)."""
        env = DictObsEnv()
        wrapped = RobotHarnessWrapper(
            env,
            checkpoints=[{"name": "cp", "step": 1}],
            output_dir=tmp_path,
            auto_fix_obs_space=True,
        )
        obs, _ = wrapped.reset()
        assert isinstance(obs, dict)
        # No crash, observation_space unchanged
        assert isinstance(wrapped.observation_space, spaces.Dict)

    def test_fix_persists_across_resets(self, tmp_path):
        """The fix is applied once and persists across subsequent resets."""
        env = MismatchedObsSpaceEnv()
        wrapped = RobotHarnessWrapper(
            env,
            checkpoints=[{"name": "cp", "step": 1}],
            output_dir=tmp_path,
            auto_fix_obs_space=True,
        )
        wrapped.reset()
        assert wrapped.observation_space.shape == (100,)

        wrapped.reset()
        assert wrapped.observation_space.shape == (100,)

    def test_checkpoint_captures_correct_shape(self, tmp_path):
        """Checkpoints record the actual obs shape after auto-fix."""
        env = MismatchedObsSpaceEnv()
        wrapped = RobotHarnessWrapper(
            env,
            checkpoints=[{"name": "cp", "step": 1}],
            output_dir=tmp_path,
            task_name="shape_fix",
            auto_fix_obs_space=True,
        )
        wrapped.reset()
        _, _, _, _, _info = wrapped.step(np.zeros(29))

        state_path = tmp_path / "shape_fix" / "trial_001" / "cp" / "state.json"
        state = json.loads(state_path.read_text())
        assert state["obs_shape"] == [100]


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
