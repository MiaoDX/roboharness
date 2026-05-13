"""Tests for the SONIC locomotion example helpers."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mujoco", reason="mujoco not installed")

from examples.sonic_locomotion import (
    G1_BODY_TORQUE_LIMITS,
    VirtualSupportHarness,
    _quat_to_rotvec,
    command_for_step,
)
from roboharness.controllers.locomotion import SonicMode


def test_quat_to_rotvec_identity_returns_zero() -> None:
    rotvec = _quat_to_rotvec(np.array([1.0, 0.0, 0.0, 0.0]))
    np.testing.assert_allclose(rotvec, np.zeros(3), atol=1e-8)


def test_quat_to_rotvec_90deg_x_axis() -> None:
    quat = np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0])
    rotvec = _quat_to_rotvec(quat)
    np.testing.assert_allclose(rotvec, np.array([np.pi / 2, 0.0, 0.0]), atol=1e-6)


def test_virtual_support_harness_only_applies_vertical_force_at_identity() -> None:
    harness = VirtualSupportHarness(target_z=1.0, kp_z=100.0, kd_z=10.0, kp_ang=50.0, kd_ang=5.0)
    pose = np.array(
        [
            0.3,
            -0.4,
            0.8,
            1.0,
            0.0,
            0.0,
            0.0,
            0.2,
            -0.1,
            -0.5,
            0.0,
            0.0,
            0.0,
        ]
    )
    wrench = harness.compute_wrench(pose)
    np.testing.assert_allclose(wrench[:2], np.zeros(2), atol=1e-8)
    assert wrench[2] > 0.0
    np.testing.assert_allclose(wrench[3:], np.zeros(3), atol=1e-8)


def test_virtual_support_harness_adds_corrective_orientation_torque() -> None:
    harness = VirtualSupportHarness(target_z=1.0, kp_z=100.0, kd_z=10.0, kp_ang=50.0, kd_ang=5.0)
    pose = np.array(
        [
            0.0,
            0.0,
            1.0,
            np.cos(np.pi / 4),
            np.sin(np.pi / 4),
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )
    wrench = harness.compute_wrench(pose)
    assert wrench[3] < 0.0
    np.testing.assert_allclose(wrench[[0, 1, 2, 4, 5]], np.zeros(5), atol=1e-8)


def test_command_for_step_uses_idle_outside_walking_window() -> None:
    initial = command_for_step(0)
    walking = command_for_step(400)
    stopping = command_for_step(1100)

    assert initial["mode"] == SonicMode.IDLE
    assert initial["velocity"] == [0.0, 0.0, 0.0]
    assert walking["mode"] == SonicMode.WALK
    assert walking["velocity"] == [0.3, 0.0, 0.0]
    assert stopping["mode"] == SonicMode.IDLE
    assert stopping["velocity"] == [0.0, 0.0, 0.0]


def test_body_torque_limits_cover_all_controlled_joints() -> None:
    assert G1_BODY_TORQUE_LIMITS.shape == (29,)
