"""Shared pytest configuration and fixtures."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from roboharness.core.capture import CameraView


class MockBackend:
    """A mock simulator backend shared across test modules.

    Implements the ``SimulatorBackend`` protocol with simple in-memory state.
    """

    def __init__(self) -> None:
        self._time = 0.0
        self._state: dict[str, Any] = {"qpos": [0.0], "qvel": [0.0]}

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


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Auto-skip GPU-marked tests when CUDA is not available."""
    try:
        import torch

        has_cuda = torch.cuda.is_available()
    except ImportError:
        has_cuda = False

    if not has_cuda:
        skip_gpu = pytest.mark.skip(reason="CUDA not available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)
