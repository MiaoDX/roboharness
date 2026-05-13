"""Tests for internal utilities."""

from __future__ import annotations

import json

import numpy as np
import pytest

from roboharness._utils import NumpyEncoder, save_image


def test_save_image_without_pil(tmp_path, monkeypatch):
    """Fallback saves .npy when PIL is not available."""
    import builtins

    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if "PIL" in name:
            raise ImportError("no PIL")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    arr = np.zeros((2, 2, 3), dtype=np.uint8)
    save_image(arr, tmp_path / "test.png")

    npy_path = tmp_path / "test.npy"
    assert npy_path.exists()
    loaded = np.load(npy_path)
    np.testing.assert_array_equal(loaded, arr)


def test_numpy_encoder_ndarray():
    """NumpyEncoder converts ndarray to list."""
    arr = np.array([1.0, 2.0, 3.0])
    result = json.dumps(arr, cls=NumpyEncoder)
    assert json.loads(result) == [1.0, 2.0, 3.0]


def test_numpy_encoder_integer():
    """NumpyEncoder converts numpy integers."""
    val = np.int64(42)
    result = json.dumps({"val": val}, cls=NumpyEncoder)
    assert json.loads(result)["val"] == 42


def test_numpy_encoder_floating():
    """NumpyEncoder converts numpy floats."""
    val = np.float32(3.14)
    result = json.dumps({"val": val}, cls=NumpyEncoder)
    assert json.loads(result)["val"] == pytest.approx(3.14, rel=1e-5)
