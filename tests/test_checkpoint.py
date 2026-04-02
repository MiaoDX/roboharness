"""Tests for checkpoint management."""

import pytest

from robot_harness.core.checkpoint import Checkpoint, CheckpointStore


def test_checkpoint_creation():
    cp = Checkpoint(name="test", cameras=["front", "side"])
    assert cp.name == "test"
    assert cp.cameras == ["front", "side"]
    assert cp.trigger_step is None


def test_checkpoint_store_save_restore(tmp_path):
    store = CheckpointStore(base_dir=tmp_path / "checkpoints")
    state = {"qpos": [1.0, 2.0, 3.0], "time": 0.5}

    store.save("cp1", state)
    assert store.has("cp1")
    assert not store.has("cp2")

    restored = store.restore("cp1")
    assert restored == state


def test_checkpoint_store_restore_missing(tmp_path):
    store = CheckpointStore(base_dir=tmp_path / "checkpoints")
    with pytest.raises(KeyError, match="not_saved"):
        store.restore("not_saved")


def test_checkpoint_store_list(tmp_path):
    store = CheckpointStore(base_dir=tmp_path / "checkpoints")
    store.save("a", {"x": 1})
    store.save("b", {"x": 2})
    assert store.list_checkpoints() == ["a", "b"]
