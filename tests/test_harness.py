"""Tests for the core Harness class."""

from typing import Any

import numpy as np

from roboharness.core.capture import CameraView
from roboharness.core.harness import Harness


class MockBackend:
    """A mock simulator backend for testing."""

    def __init__(self) -> None:
        self._time = 0.0
        self._state: dict[str, Any] = {"qpos": [0.0], "qvel": [0.0]}
        self._saved_states: dict[str, dict[str, Any]] = {}

    def step(self, action: Any) -> dict[str, Any]:
        self._time += 0.01
        self._state["qpos"] = [self._state["qpos"][0] + 0.1]
        return self.get_state()

    def get_state(self) -> dict[str, Any]:
        return {**self._state, "time": self._time}

    def save_state(self) -> dict[str, Any]:
        return {"state": {**self._state}, "time": self._time}

    def restore_state(self, state: dict[str, Any]) -> None:
        self._state = {**state["state"]}
        self._time = state["time"]

    def capture_camera(self, camera_name: str) -> CameraView:
        return CameraView(
            name=camera_name,
            rgb=np.zeros((64, 64, 3), dtype=np.uint8),
        )

    def get_sim_time(self) -> float:
        return self._time

    def reset(self) -> dict[str, Any]:
        self._time = 0.0
        self._state = {"qpos": [0.0], "qvel": [0.0]}
        return self.get_state()


def test_mock_implements_protocol():
    """Verify MockBackend structurally matches SimulatorBackend Protocol."""
    backend = MockBackend()
    assert callable(getattr(backend, "step", None))
    assert callable(getattr(backend, "get_state", None))
    assert callable(getattr(backend, "reset", None))


def test_harness_add_checkpoints(tmp_path):
    harness = Harness(MockBackend(), output_dir=tmp_path)
    harness.add_checkpoint("cp1", cameras=["front"])
    harness.add_checkpoint("cp2", cameras=["front", "side"])
    assert harness.list_checkpoints() == ["cp1", "cp2"]


def test_harness_reset(tmp_path):
    harness = Harness(MockBackend(), output_dir=tmp_path)
    state = harness.reset()
    assert state["qpos"] == [0.0]
    assert state["time"] == 0.0


def test_harness_step(tmp_path):
    harness = Harness(MockBackend(), output_dir=tmp_path)
    harness.reset()
    state = harness.step(None)
    assert state["qpos"][0] > 0.0


def test_harness_run_to_checkpoint(tmp_path):
    harness = Harness(MockBackend(), output_dir=tmp_path)
    harness.add_checkpoint("cp1", cameras=["front"])
    harness.reset()

    actions = [None] * 10
    result = harness.run_to_next_checkpoint(actions)
    assert result is not None
    assert result.checkpoint_name == "cp1"
    assert len(result.views) == 1
    assert result.views[0].name == "front"


def test_harness_checkpoint_save_restore(tmp_path):
    harness = Harness(MockBackend(), output_dir=tmp_path)
    harness.add_checkpoint("cp1", cameras=["front"])
    harness.add_checkpoint("cp2", cameras=["front"])
    harness.reset()

    # Run to cp1
    harness.run_to_next_checkpoint([None] * 5)
    state_at_cp1 = harness.get_state()

    # Run to cp2 (state changes)
    harness.run_to_next_checkpoint([None] * 5)
    state_at_cp2 = harness.get_state()
    assert state_at_cp2["qpos"] != state_at_cp1["qpos"]

    # Restore to cp1
    harness.restore_checkpoint("cp1")
    restored_state = harness.get_state()
    assert restored_state["qpos"] == state_at_cp1["qpos"]


def test_harness_no_more_checkpoints(tmp_path):
    harness = Harness(MockBackend(), output_dir=tmp_path)
    harness.add_checkpoint("only_one", cameras=["front"])
    harness.reset()

    result1 = harness.run_to_next_checkpoint([None])
    assert result1 is not None

    result2 = harness.run_to_next_checkpoint([None])
    assert result2 is None
