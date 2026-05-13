"""Tests for upstream contribution examples.

These tests validate the contrib examples work correctly using mock
environments and mock dependencies, ensuring the examples are ready
for upstream submission.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

gym = pytest.importorskip("gymnasium", reason="gymnasium not installed")

from roboharness.core.protocol import TaskPhase, TaskProtocol  # noqa: E402
from roboharness.wrappers import RobotHarnessWrapper  # noqa: E402

# ---------------------------------------------------------------------------
# ManiSkill contrib example tests
# ---------------------------------------------------------------------------


class MockPickCubeEnv(gym.Env):
    """Reproduces the mock from contrib_maniskill_visual_debug.py for testing."""

    metadata: dict = {"render_modes": ["rgb_array"], "render_fps": 20}  # noqa: RUF012

    def __init__(self, render_mode: str = "rgb_array"):
        super().__init__()
        self.render_mode = render_mode
        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(-np.inf, np.inf, shape=(9,), dtype=np.float32),
                "extra": gym.spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32),
            }
        )
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        self._step_count = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self._step_count = 0
        return {
            "agent": np.zeros(9, dtype=np.float32),
            "extra": np.zeros(7, dtype=np.float32),
        }, {}

    def step(self, action):
        self._step_count += 1
        obs = {
            "agent": np.random.randn(9).astype(np.float32) * 0.1,
            "extra": np.random.randn(7).astype(np.float32) * 0.1,
        }
        reward = np.array([0.1 * min(self._step_count / 100, 1.0)], dtype=np.float32)
        terminated = self._step_count >= 200
        return obs, reward, terminated, False, {}

    def render(self):
        frame = np.zeros((256, 256, 3), dtype=np.uint8)
        frame[:, :, 2] = 128
        return frame


def test_maniskill_contrib_wrapper_captures_checkpoints(tmp_path):
    """The ManiSkill contrib example pattern captures at all 3 checkpoints."""
    env = MockPickCubeEnv()
    protocol = TaskProtocol(
        name="pick_cube",
        description="PickCube manipulation task",
        phases=[
            TaskPhase("approach", "Gripper approaching the cube"),
            TaskPhase("contact", "Gripper making contact"),
            TaskPhase("lift", "Cube being lifted"),
        ],
    )
    wrapped = RobotHarnessWrapper(
        env,
        protocol=protocol,
        phase_steps={"approach": 50, "contact": 100, "lift": 150},
        cameras=["default"],
        output_dir=str(tmp_path),
        task_name="pick_cube",
    )

    wrapped.reset()
    checkpoints_hit = []
    for _ in range(200):
        action = wrapped.action_space.sample()
        _obs, _reward, terminated, truncated, info = wrapped.step(action)
        if "checkpoint" in info:
            checkpoints_hit.append(info["checkpoint"]["name"])
        if terminated or truncated:
            break

    wrapped.close()

    assert checkpoints_hit == ["approach", "contact", "lift"]


def test_maniskill_contrib_saves_state_json(tmp_path):
    """Checkpoint state.json contains expected fields for agent consumption."""
    env = MockPickCubeEnv()
    protocol = TaskProtocol(
        name="pick_cube",
        description="PickCube task",
        phases=[TaskPhase("approach", "Approaching")],
    )
    wrapped = RobotHarnessWrapper(
        env,
        protocol=protocol,
        phase_steps={"approach": 5},
        cameras=["default"],
        output_dir=str(tmp_path),
        task_name="pick_cube",
    )

    wrapped.reset()
    for _ in range(10):
        _obs, _reward, _term, _trunc, info = wrapped.step(wrapped.action_space.sample())
        if "checkpoint" in info:
            break

    wrapped.close()

    state_path = Path(info["checkpoint"]["files"]["state"])
    state = json.loads(state_path.read_text())

    assert state["step"] == 5
    assert state["checkpoint"] == "approach"
    assert "obs_keys" in state
    assert "agent" in state["obs_keys"]
    assert "extra" in state["obs_keys"]
    assert isinstance(state["reward"], float)


def test_maniskill_contrib_captures_rgb_image(tmp_path):
    """Checkpoint captures include an RGB image file (png or npy fallback)."""
    env = MockPickCubeEnv()
    wrapped = RobotHarnessWrapper(
        env,
        checkpoints=[{"name": "cp", "step": 1}],
        cameras=["default"],
        output_dir=str(tmp_path),
        task_name="pick_cube",
    )

    wrapped.reset()
    _obs, _reward, _term, _trunc, info = wrapped.step(wrapped.action_space.sample())
    wrapped.close()

    assert "default_rgb" in info["checkpoint"]["files"]
    rgb_path = Path(info["checkpoint"]["files"]["default_rgb"])
    # PIL may not be installed, falling back to .npy — check either exists
    assert rgb_path.exists() or rgb_path.with_suffix(".npy").exists()


def test_maniskill_contrib_dict_obs_records_keys(tmp_path):
    """Dict observations from ManiSkill-like envs record obs_keys in state."""
    env = MockPickCubeEnv()
    wrapped = RobotHarnessWrapper(
        env,
        checkpoints=[{"name": "cp", "step": 1}],
        output_dir=str(tmp_path),
        task_name="test",
    )

    wrapped.reset()
    _obs, _reward, _term, _trunc, info = wrapped.step(wrapped.action_space.sample())
    wrapped.close()

    state = json.loads(Path(info["checkpoint"]["files"]["state"]).read_text())
    assert set(state["obs_keys"]) == {"agent", "extra"}


def test_maniskill_contrib_vectorized_reward_serialized(tmp_path):
    """ManiSkill vectorized rewards (shape [1]) serialize to a float scalar."""
    env = MockPickCubeEnv()
    wrapped = RobotHarnessWrapper(
        env,
        checkpoints=[{"name": "cp", "step": 1}],
        output_dir=str(tmp_path),
        task_name="test",
    )

    wrapped.reset()
    _obs, _reward, _term, _trunc, info = wrapped.step(wrapped.action_space.sample())
    wrapped.close()

    state = json.loads(Path(info["checkpoint"]["files"]["state"]).read_text())
    assert isinstance(state["reward"], float)
    assert state["reward"] > 0.0
