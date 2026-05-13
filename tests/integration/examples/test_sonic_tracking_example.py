"""Tests for the SONIC tracking example helpers."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mujoco", reason="mujoco not installed")

from examples.sonic_tracking import (
    TRACKING_PROTOCOL,
    build_phase_steps,
    build_tracking_evaluation,
    make_tracking_clip,
    summarize_tracking_run,
)
from roboharness.evaluate.result import Verdict


def test_make_tracking_clip_matches_expected_contract() -> None:
    clip = make_tracking_clip(num_frames=120)

    assert clip.joint_positions.shape == (120, 29)
    assert clip.joint_velocities.shape == (120, 29)
    assert clip.root_height.shape == (120,)
    assert clip.root_rotation_6d.shape == (120, 6)
    assert clip.name == "deterministic_march"
    assert np.ptp(clip.joint_positions[:, 0]) > 0.15
    assert np.ptp(clip.joint_positions[:, 6]) > 0.15


def test_build_phase_steps_stays_ordered_and_hits_final_frame() -> None:
    phase_steps = build_phase_steps(180)

    assert list(phase_steps) == TRACKING_PROTOCOL.phase_names()
    assert phase_steps["initial"] < phase_steps["left_stride"]
    assert phase_steps["left_stride"] < phase_steps["right_stride"]
    assert phase_steps["right_stride"] < phase_steps["finale"]
    assert phase_steps["finale"] == 180


def test_summarize_tracking_run_reports_progress_and_motion_span() -> None:
    joint_history = [
        np.zeros(29, dtype=np.float32),
        np.array([0.18, 0.0, 0.0, 0.32, 0.0, 0.0, -0.15, 0.0, 0.0, 0.28] + [0.0] * 19),
        np.array([-0.14, 0.0, 0.0, 0.10, 0.0, 0.0, 0.16, 0.0, 0.0, 0.34] + [0.0] * 19),
    ]

    summary = summarize_tracking_run(
        tracking_frames=[0, 45, 90],
        torso_heights=[0.74, 0.71, 0.73],
        rms_joint_errors=[0.22, 0.35, 0.28],
        joint_history=joint_history,
    )

    assert summary["min_torso_z"] == pytest.approx(0.71)
    assert summary["final_tracking_frame"] == pytest.approx(90.0)
    assert summary["mean_rms_joint_error"] == pytest.approx((0.22 + 0.35 + 0.28) / 3.0)
    assert summary["left_hip_pitch_span"] == pytest.approx(0.32)
    assert summary["right_knee_span"] == pytest.approx(0.34)


def test_build_tracking_evaluation_passes_and_degrades_expected_cases() -> None:
    summary_metrics = {
        "min_torso_z": 0.71,
        "max_torso_z": 0.76,
        "final_tracking_frame": 119.0,
        "mean_rms_joint_error": 0.34,
        "max_rms_joint_error": 0.52,
        "left_hip_pitch_span": 0.18,
        "right_hip_pitch_span": 0.19,
        "left_knee_span": 0.24,
        "right_knee_span": 0.25,
    }
    snapshot_metrics = {
        "initial": {"torso_z": 0.74, "rms_joint_error": 0.21},
        "left_stride": {"torso_z": 0.70, "rms_joint_error": 0.48},
        "right_stride": {"torso_z": 0.70, "rms_joint_error": 0.50},
        "finale": {"torso_z": 0.72, "rms_joint_error": 0.24},
    }

    passing = build_tracking_evaluation(summary_metrics, snapshot_metrics, clip_frames=120)
    degraded = build_tracking_evaluation(
        {**summary_metrics, "left_hip_pitch_span": 0.02},
        snapshot_metrics,
        clip_frames=120,
    )

    assert passing.verdict is Verdict.PASS
    assert degraded.verdict is Verdict.DEGRADED
