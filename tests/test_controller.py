"""Tests for the Controller protocol and WbcIkController."""

from typing import Any

import numpy as np

from roboharness.core.controller import Controller


class MockController:
    """A mock controller for protocol compliance testing."""

    def compute(self, command: dict[str, Any], state: dict[str, Any]) -> Any:
        return np.zeros(7)


class NotAController:
    """Does not implement the Controller protocol."""

    def do_something(self) -> None:
        pass


def test_mock_implements_controller_protocol():
    controller = MockController()
    assert isinstance(controller, Controller)


def test_non_controller_rejected():
    obj = NotAController()
    assert not isinstance(obj, Controller)


def test_mock_controller_compute():
    controller = MockController()
    action = controller.compute(
        command={"target": np.eye(4)},
        state={"qpos": np.zeros(7)},
    )
    assert isinstance(action, np.ndarray)
    assert action.shape == (7,)
