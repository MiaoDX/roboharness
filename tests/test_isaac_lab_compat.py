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

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

import gymnasium as gym
from gymnasium import spaces

from roboharness.wrappers import RobotHarnessWrapper

pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")


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
