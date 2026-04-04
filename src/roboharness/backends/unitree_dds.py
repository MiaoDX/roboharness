"""Unitree DDS backend — bridge between roboharness and real Unitree robots.

Communicates with Unitree robots (G1, H1, Go2, B2) over DDS using the
``unitree_sdk2py`` package. Provides both high-level locomotion commands
and low-level motor state reading.

Requires: pip install roboharness[unitree]

The SDK dependency is resolved via the community fork which fixes
CycloneDDS compatibility on Python 3.10+:
  https://github.com/MiaoDX-fork-and-pruning/unitree_sdk2_python_uv

Usage:
    from roboharness.backends.unitree_dds import UnitreeDDSBackend

    with UnitreeDDSBackend(network_interface="eth0") as robot:
        robot.stand_up()
        robot.move(vx=0.3, vy=0.0, vyaw=0.0)
        state = robot.get_state()
        print(state["motor_positions"])
        robot.stop()
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

SUPPORTED_ROBOTS = ("g1", "h1", "go2", "b2")

# Number of motors per robot type (body DOFs)
MOTOR_COUNTS: dict[str, int] = {
    "g1": 29,
    "h1": 19,
    "go2": 12,
    "b2": 12,
}


class UnitreeDDSBackend:
    """Backend for communicating with Unitree robots via DDS.

    Parameters
    ----------
    network_interface:
        Network interface name for DDS communication (e.g. ``"eth0"``).
    robot_type:
        Robot model identifier. One of ``"g1"``, ``"h1"``, ``"go2"``, ``"b2"``.
    timeout:
        RPC timeout in seconds for high-level commands.
    """

    def __init__(
        self,
        network_interface: str,
        robot_type: str = "g1",
        timeout: float = 10.0,
    ) -> None:
        if robot_type not in SUPPORTED_ROBOTS:
            raise ValueError(
                f"Unsupported robot type {robot_type!r}. Must be one of {SUPPORTED_ROBOTS}"
            )

        self.network_interface = network_interface
        self.robot_type = robot_type
        self.timeout = timeout

        self.connected = False
        self._loco_client: Any = None
        self._state_subscriber: Any = None
        self._latest_state: Any = None

    def connect(self) -> None:
        """Initialize DDS channels and connect to the robot."""
        try:
            from unitree_sdk2py.core.channel import (
                ChannelFactoryInitialize,
                ChannelSubscriber,
            )
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
        except ImportError as exc:
            raise ImportError(
                "unitree_sdk2py is required for the Unitree DDS backend. "
                "Install with: pip install roboharness[unitree]\n"
                "See: https://github.com/MiaoDX-fork-and-pruning/unitree_sdk2_python_uv"
            ) from exc

        ChannelFactoryInitialize(0, self.network_interface)

        # Set up state subscriber
        self._state_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self._state_subscriber.Init(self._on_state_received, 10)

        # Set up high-level locomotion client (G1/H1)
        if self.robot_type in ("g1", "h1"):
            from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient

            self._loco_client = LocoClient()
            self._loco_client.SetTimeout(self.timeout)
            self._loco_client.Init()

        self.connected = True
        logger.info(
            "Connected to %s robot via DDS on %s",
            self.robot_type,
            self.network_interface,
        )

    def disconnect(self) -> None:
        """Disconnect from the robot and clean up resources."""
        self._loco_client = None
        self._state_subscriber = None
        self._latest_state = None
        self.connected = False
        logger.info("Disconnected from %s robot", self.robot_type)

    def _require_connected(self) -> None:
        if not self.connected:
            raise RuntimeError("Not connected. Call connect() or use as a context manager first.")

    # -- Context manager ----------------------------------------------------

    def __enter__(self) -> UnitreeDDSBackend:
        self.connect()
        return self

    def __exit__(self, *exc: object) -> None:
        self.disconnect()

    # -- State reading ------------------------------------------------------

    def _on_state_received(self, msg: Any) -> None:
        """Callback invoked by the DDS subscriber when a new state arrives."""
        self._latest_state = msg

    def get_state(self) -> dict[str, Any] | None:
        """Return the latest robot state as a dict, or None if no data yet.

        Returns
        -------
        dict or None
            Keys: ``motor_positions``, ``motor_velocities``,
            ``imu_quaternion``, ``imu_gyroscope``, ``imu_accelerometer``.
            All values are lists of floats.
        """
        self._require_connected()

        if self._latest_state is None:
            return None

        n_motors = MOTOR_COUNTS[self.robot_type]
        state = self._latest_state

        return {
            "motor_positions": [state.motor_state[i].q for i in range(n_motors)],
            "motor_velocities": [state.motor_state[i].dq for i in range(n_motors)],
            "imu_quaternion": list(state.imu_state.quaternion),
            "imu_gyroscope": list(state.imu_state.gyroscope),
            "imu_accelerometer": list(state.imu_state.accelerometer),
        }

    # -- High-level locomotion commands ------------------------------------

    def move(self, vx: float, vy: float, vyaw: float) -> None:
        """Command the robot to move with the given velocities.

        Parameters
        ----------
        vx: Forward velocity (m/s).
        vy: Lateral velocity (m/s).
        vyaw: Yaw rate (rad/s).
        """
        self._require_connected()
        self._loco_client.Move(vx, vy, vyaw)

    def stop(self) -> None:
        """Stop all movement."""
        self._require_connected()
        self._loco_client.StopMove()

    def stand_up(self) -> None:
        """Transition from squat to standing position."""
        self._require_connected()
        self._loco_client.Squat2StandUp()

    def damp(self) -> None:
        """Enter damping (low-energy safe) mode."""
        self._require_connected()
        self._loco_client.Damp()

    def high_stand(self) -> None:
        """Stand at maximum height."""
        self._require_connected()
        self._loco_client.HighStand()

    def low_stand(self) -> None:
        """Stand at minimum height."""
        self._require_connected()
        self._loco_client.LowStand()
