"""Tests for ManiSkill compatibility via Gymnasium wrapper.

These tests validate common ManiSkill-like patterns without requiring
ManiSkill as a dependency:
  - NumPy-based vectorized rewards (shape: [num_envs])
  - CPUGymWrapper-style single-env observations
  - Dict observations used in many manipulation tasks
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import pytest

gym = pytest.importorskip("gymnasium", reason="gymnasium not installed")

from gymnasium import spaces  # noqa: E402

from roboharness.wrappers import RobotHarnessWrapper  # noqa: E402


class MockManiSkillEnv(gym.Env):
    """Lightweight ManiSkill-like env returning NumPy data."""

    metadata: ClassVar[dict] = {"render_modes": ["rgb_array"], "render_fps": 60}

    def __init__(self, num_envs: int = 1, render_mode: str = "rgb_array"):
        super().__init__()
        self.num_envs = num_envs
        self.render_mode = render_mode
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed, options=options)
        if self.num_envs == 1:
            return np.zeros(self.observation_space.shape, dtype=np.float32), {}
        return np.zeros((self.num_envs, *self.observation_space.shape), dtype=np.float32), {}

    def step(
        self, action: Any
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        if self.num_envs == 1:
            obs = np.ones(self.observation_space.shape, dtype=np.float32)
        else:
            obs = np.ones((self.num_envs, *self.observation_space.shape), dtype=np.float32)
        reward = np.full((self.num_envs,), 0.5, dtype=np.float32)
        terminated = np.zeros((self.num_envs,), dtype=bool)
        truncated = np.zeros((self.num_envs,), dtype=bool)
        return obs, reward, terminated, truncated, {}

    def render(self) -> np.ndarray:
        return np.zeros((256, 256, 3), dtype=np.uint8)


class MockManiSkillDictObsEnv(MockManiSkillEnv):
    """ManiSkill-like env with dict observation space."""

    def __init__(self, num_envs: int = 1, render_mode: str = "rgb_array"):
        super().__init__(num_envs=num_envs, render_mode=render_mode)
        self.observation_space = spaces.Dict(
            {
                "state": spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32),
                "extra": spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
            }
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        gym.Env.reset(self, seed=seed, options=options)
        return {
            "state": np.zeros((10,), dtype=np.float32),
            "extra": np.zeros((2,), dtype=np.float32),
        }, {}

    def step(
        self, action: Any
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        obs = {
            "state": np.ones((10,), dtype=np.float32),
            "extra": np.ones((2,), dtype=np.float32),
        }
        reward = np.array([0.25], dtype=np.float32)
        terminated = np.array([False], dtype=bool)
        truncated = np.array([False], dtype=bool)
        return obs, reward, terminated, truncated, {}


def test_maniskill_vector_reward_saved_as_scalar_mean(tmp_path):
    """Vector reward arrays should serialize to a meaningful scalar in state.json."""
    env = MockManiSkillEnv(num_envs=8)
    wrapped = RobotHarnessWrapper(
        env,
        checkpoints=[{"name": "cp", "step": 1}],
        output_dir=tmp_path,
        task_name="maniskill_vec",
    )
    wrapped.reset()
    _obs, _reward, _terminated, _truncated, info = wrapped.step(
        np.zeros((8, *env.action_space.shape), dtype=np.float32)
    )

    import json
    from pathlib import Path

    state_path = Path(info["checkpoint"]["files"]["state"])
    state = json.loads(state_path.read_text())
    assert state["reward"] == pytest.approx(0.5)


def test_maniskill_dict_obs_records_obs_keys(tmp_path):
    """Dict observations should be summarized via obs_keys in state.json."""
    env = MockManiSkillDictObsEnv(num_envs=1)
    wrapped = RobotHarnessWrapper(
        env,
        checkpoints=[{"name": "cp", "step": 1}],
        output_dir=tmp_path,
        task_name="maniskill_dict",
    )
    wrapped.reset()
    _obs, _reward, _terminated, _truncated, info = wrapped.step(
        np.zeros(env.action_space.shape, dtype=np.float32)
    )

    import json
    from pathlib import Path

    state_path = Path(info["checkpoint"]["files"]["state"])
    state = json.loads(state_path.read_text())
    assert "state" in state["obs_keys"]
    assert "extra" in state["obs_keys"]


def test_maniskill_single_env_capture_fallback(tmp_path):
    """Single-camera render fallback should still produce default capture files."""
    env = MockManiSkillEnv(num_envs=1)
    wrapped = RobotHarnessWrapper(
        env,
        cameras=["front", "wrist"],
        checkpoints=[{"name": "cp", "step": 1}],
        output_dir=tmp_path,
        task_name="maniskill_capture",
    )
    wrapped.reset()
    _obs, _reward, _terminated, _truncated, info = wrapped.step(
        np.zeros(env.action_space.shape, dtype=np.float32)
    )

    files = info["checkpoint"]["files"]
    assert "default_rgb" in files
