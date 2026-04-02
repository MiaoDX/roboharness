"""Tests for capture functionality."""

import json

import numpy as np

from robot_harness.core.capture import CameraView, CaptureResult


def test_camera_view_save_rgb(tmp_path):
    rgb = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    view = CameraView(name="front", rgb=rgb)
    files = view.save(tmp_path / "test_view")

    assert "rgb" in files


def test_camera_view_save_with_depth(tmp_path):
    rgb = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    depth = np.random.rand(64, 64).astype(np.float32)
    view = CameraView(name="front", rgb=rgb, depth=depth)
    files = view.save(tmp_path / "test_view")

    assert "rgb" in files
    assert "depth" in files
    assert "depth_viz" in files


def test_capture_result_save(tmp_path):
    views = [
        CameraView(
            name="front",
            rgb=np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8),
        ),
        CameraView(
            name="side",
            rgb=np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8),
        ),
    ]
    result = CaptureResult(
        checkpoint_name="test_cp",
        step=42,
        sim_time=1.5,
        views=views,
        state={"qpos": [1.0, 2.0]},
    )

    save_dir = tmp_path / "capture"
    result.save(save_dir)

    # Check metadata was saved
    meta_path = save_dir / "metadata.json"
    assert meta_path.exists()
    with open(meta_path) as f:
        meta = json.load(f)
    assert meta["checkpoint"] == "test_cp"
    assert meta["step"] == 42
    assert "front" in meta["cameras"]
    assert "side" in meta["cameras"]

    # Check state was saved
    state_path = save_dir / "state.json"
    assert state_path.exists()
    with open(state_path) as f:
        state = json.load(f)
    assert state["qpos"] == [1.0, 2.0]
