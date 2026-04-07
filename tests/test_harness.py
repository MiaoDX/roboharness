"""Tests for the core Harness class."""

import pytest

from roboharness.core.harness import Harness, SimulatorBackend

from .conftest import MockBackend


def test_mock_implements_protocol():
    """Verify MockBackend structurally matches SimulatorBackend Protocol."""
    backend = MockBackend()
    assert isinstance(backend, SimulatorBackend)


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
    assert state["qpos"][0] == pytest.approx(0.1)


def test_harness_run_to_checkpoint(tmp_path):
    harness = Harness(MockBackend(), output_dir=tmp_path)
    harness.add_checkpoint("cp1", cameras=["front"])
    harness.reset()

    actions = [None] * 10
    result = harness.run_to_next_checkpoint(actions)
    assert result is not None
    assert result.step == 10
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
