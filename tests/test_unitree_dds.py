"""Tests for Unitree DDS backend — validates communication layer with mocked SDK.

The Unitree DDS backend wraps ``unitree_sdk2py`` to provide a bridge between
roboharness and real Unitree robots (G1, H1, Go2, B2) via DDS messaging.

These tests use mocked SDK objects since the real SDK requires a physical
robot or DDS network. They validate:
  - Channel initialization and cleanup
  - State subscription and reading
  - Command publishing (high-level loco + low-level motor)
  - Error handling for missing SDK or disconnected robot
"""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def _mock_unitree_sdk():
    """Inject a fake unitree_sdk2py into sys.modules for all tests."""
    modules: dict[str, Any] = {}

    # Build a minimal mock module tree
    sdk = MagicMock()
    modules["unitree_sdk2py"] = sdk
    modules["unitree_sdk2py.core"] = sdk.core
    modules["unitree_sdk2py.core.channel"] = sdk.core.channel
    modules["unitree_sdk2py.g1"] = sdk.g1
    modules["unitree_sdk2py.g1.loco"] = sdk.g1.loco
    modules["unitree_sdk2py.g1.loco.g1_loco_client"] = sdk.g1.loco.g1_loco_client
    modules["unitree_sdk2py.idl"] = sdk.idl
    modules["unitree_sdk2py.idl.default"] = sdk.idl.default
    modules["unitree_sdk2py.idl.unitree_hg"] = sdk.idl.unitree_hg
    modules["unitree_sdk2py.idl.unitree_hg.msg"] = sdk.idl.unitree_hg.msg
    modules["unitree_sdk2py.idl.unitree_hg.msg.dds_"] = sdk.idl.unitree_hg.msg.dds_
    modules["unitree_sdk2py.utils"] = sdk.utils
    modules["unitree_sdk2py.utils.crc"] = sdk.utils.crc

    # Provide concrete mock callables (use MagicMock() instances so that
    # calling them returns a plain MagicMock without a restrictive spec).
    sdk.core.channel.ChannelFactoryInitialize = MagicMock()
    sdk.core.channel.ChannelSubscriber = MagicMock()
    sdk.core.channel.ChannelPublisher = MagicMock()
    sdk.g1.loco.g1_loco_client.LocoClient = MagicMock()
    sdk.idl.unitree_hg.msg.dds_.LowState_ = MagicMock()
    sdk.idl.unitree_hg.msg.dds_.LowCmd_ = MagicMock()
    sdk.idl.default.unitree_hg_msg_dds__LowState_ = MagicMock()
    sdk.idl.default.unitree_hg_msg_dds__LowCmd_ = MagicMock()
    sdk.utils.crc.CRC = MagicMock()

    saved = {}
    for name, mod in modules.items():
        saved[name] = sys.modules.get(name)
        sys.modules[name] = mod

    yield sdk

    # Restore original state
    for name, original in saved.items():
        if original is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = original


# ---------------------------------------------------------------------------
# Import after mock is in place (module-level would fail without SDK)
# ---------------------------------------------------------------------------


def _import_backend():
    """Import the backend module (must be called inside a test with the mock active)."""
    from roboharness.backends.unitree_dds import UnitreeDDSBackend

    return UnitreeDDSBackend


# ---------------------------------------------------------------------------
# Construction / initialization
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_default_construction(self):
        cls = _import_backend()
        backend = cls(network_interface="eth0")
        assert backend.network_interface == "eth0"
        assert backend.robot_type == "g1"

    def test_custom_robot_type(self):
        cls = _import_backend()
        backend = cls(network_interface="eth0", robot_type="h1")
        assert backend.robot_type == "h1"

    def test_invalid_robot_type_raises(self):
        cls = _import_backend()
        with pytest.raises(ValueError, match="Unsupported robot type"):
            cls(network_interface="eth0", robot_type="invalid_robot")

    def test_connect_initializes_channel(self, _mock_unitree_sdk):
        cls = _import_backend()
        backend = cls(network_interface="eth0")
        backend.connect()

        _mock_unitree_sdk.core.channel.ChannelFactoryInitialize.assert_called_once_with(0, "eth0")
        assert backend.connected

    def test_disconnect(self, _mock_unitree_sdk):
        cls = _import_backend()
        backend = cls(network_interface="eth0")
        backend.connect()
        backend.disconnect()
        assert not backend.connected


# ---------------------------------------------------------------------------
# High-level locomotion commands
# ---------------------------------------------------------------------------


class TestHighLevelLoco:
    def test_move_delegates_to_loco_client(self, _mock_unitree_sdk):
        cls = _import_backend()
        backend = cls(network_interface="eth0")
        backend.connect()

        backend.move(vx=0.3, vy=0.0, vyaw=0.0)
        backend._loco_client.Move.assert_called_once_with(0.3, 0.0, 0.0)

    def test_stop_calls_stop_move(self, _mock_unitree_sdk):
        cls = _import_backend()
        backend = cls(network_interface="eth0")
        backend.connect()

        backend.stop()
        backend._loco_client.StopMove.assert_called_once()

    def test_stand_up(self, _mock_unitree_sdk):
        cls = _import_backend()
        backend = cls(network_interface="eth0")
        backend.connect()

        backend.stand_up()
        backend._loco_client.Squat2StandUp.assert_called_once()

    def test_damp(self, _mock_unitree_sdk):
        cls = _import_backend()
        backend = cls(network_interface="eth0")
        backend.connect()

        backend.damp()
        backend._loco_client.Damp.assert_called_once()

    def test_command_before_connect_raises(self):
        cls = _import_backend()
        backend = cls(network_interface="eth0")

        with pytest.raises(RuntimeError, match="Not connected"):
            backend.move(vx=0.1, vy=0.0, vyaw=0.0)


# ---------------------------------------------------------------------------
# State reading
# ---------------------------------------------------------------------------


class TestStateReading:
    def test_get_state_returns_dict(self, _mock_unitree_sdk):
        cls = _import_backend()
        backend = cls(network_interface="eth0")
        backend.connect()

        # Simulate a state callback having been received
        backend._latest_state = MagicMock()
        backend._latest_state.motor_state = [MagicMock(q=0.1, dq=0.01) for _ in range(29)]
        backend._latest_state.imu_state = MagicMock(
            quaternion=[1.0, 0.0, 0.0, 0.0],
            gyroscope=[0.0, 0.0, 0.0],
            accelerometer=[0.0, 0.0, 9.81],
        )

        state = backend.get_state()

        assert "motor_positions" in state
        assert "motor_velocities" in state
        assert "imu_quaternion" in state
        assert "imu_gyroscope" in state
        assert "imu_accelerometer" in state
        assert len(state["motor_positions"]) == 29
        assert len(state["motor_velocities"]) == 29

    def test_get_state_before_data_returns_none(self, _mock_unitree_sdk):
        cls = _import_backend()
        backend = cls(network_interface="eth0")
        backend.connect()

        state = backend.get_state()
        assert state is None


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


class TestContextManager:
    def test_context_manager_connects_and_disconnects(self, _mock_unitree_sdk):
        cls = _import_backend()
        backend = cls(network_interface="eth0")

        with backend:
            assert backend.connected
            _mock_unitree_sdk.core.channel.ChannelFactoryInitialize.assert_called_once()

        assert not backend.connected


# ---------------------------------------------------------------------------
# Import error when SDK is missing
# ---------------------------------------------------------------------------


class TestMissingSDK:
    def test_import_error_message(self):
        """Importing with SDK missing gives a clear error."""
        # Temporarily remove the mock SDK from sys.modules
        saved = {}
        sdk_modules = [k for k in sys.modules if k.startswith("unitree_sdk2py")]
        for name in sdk_modules:
            saved[name] = sys.modules.pop(name)

        try:
            # Force reimport
            if "roboharness.backends.unitree_dds" in sys.modules:
                del sys.modules["roboharness.backends.unitree_dds"]

            from roboharness.backends.unitree_dds import UnitreeDDSBackend

            backend = UnitreeDDSBackend(network_interface="eth0")
            with pytest.raises(ImportError, match="unitree_sdk2py"):
                backend.connect()
        finally:
            # Restore
            for name, mod in saved.items():
                sys.modules[name] = mod
