"""Tests for task storage."""

import json

from robot_harness.storage.task_store import GraspTaskStore, TaskStore, TrialResult


def test_task_store_variant_dirs(tmp_path):
    store = TaskStore(tmp_path, "test_task")
    d = store.get_variant_dir("variant_1")
    assert d.exists()
    assert d.name == "variant_1"


def test_task_store_trial_dirs(tmp_path):
    store = TaskStore(tmp_path, "test_task")
    d = store.get_trial_dir("variant_1", 1)
    assert d.exists()
    assert d.name == "trial_001"


def test_task_store_checkpoint_dirs(tmp_path):
    store = TaskStore(tmp_path, "test_task")
    d = store.get_checkpoint_dir("variant_1", 1, "contact")
    assert d.exists()
    assert d.name == "contact"


def test_task_store_save_config(tmp_path):
    store = TaskStore(tmp_path, "test_task")
    path = store.save_task_config({"robot": "franka", "task": "pick"})
    assert path.exists()
    with open(path) as f:
        config = json.load(f)
    assert config["robot"] == "franka"


def test_task_store_save_trial_result(tmp_path):
    store = TaskStore(tmp_path, "test_task")
    result = TrialResult(trial_id=1, success=True, reason="grasped successfully")
    path = store.save_trial_result("variant_1", result)
    assert path.exists()
    with open(path) as f:
        data = json.load(f)
    assert data["success"] is True


def test_grasp_task_store_add_position(tmp_path):
    store = GraspTaskStore(tmp_path, "grasp")
    store.add_grasp_position(
        position_id=1,
        xyz=(0.5, 0.0, 0.05),
        quaternion=(1, 0, 0, 0),
        object_name="cube",
    )
    pos_path = tmp_path / "grasp" / "grasp_position_001" / "position.json"
    assert pos_path.exists()
    with open(pos_path) as f:
        data = json.load(f)
    assert data["xyz"] == [0.5, 0.0, 0.05]
    assert data["object_name"] == "cube"


def test_grasp_task_store_checkpoint_dir(tmp_path):
    store = GraspTaskStore(tmp_path)
    d = store.get_grasp_checkpoint_dir(
        position_id=2, trial_id=1, checkpoint="contact"
    )
    assert d.exists()
    assert "grasp_position_002" in str(d)
    assert "trial_001" in str(d)
    assert d.name == "contact"


def test_grasp_task_store_list_variants(tmp_path):
    store = GraspTaskStore(tmp_path)
    store.add_grasp_position(1, xyz=(0.5, 0, 0))
    store.add_grasp_position(2, xyz=(0.6, 0, 0))
    variants = store.list_variants()
    assert len(variants) == 2
