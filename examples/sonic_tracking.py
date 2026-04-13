#!/usr/bin/env python3
"""SONIC Motion Tracking Demo — real encoder+decoder ONNX tracking on Unitree G1.

Runs the NVIDIA GEAR-SONIC tracking stack on the Unitree G1 humanoid in MuJoCo.
Unlike the planner demo, this example exercises the published encoder+decoder
ONNX contracts (``model_encoder.onnx`` + ``model_decoder.onnx``) against a real
motion clip and publishes the result as the live ``/sonic/`` report.

The clip is a deterministic in-place marching sequence designed to make the
tracking motion visually obvious while staying stable enough for CI.

Run:
    pip install roboharness[demo]
    MUJOCO_GL=osmesa python examples/sonic_tracking.py --report --assert-success

Output:
    ./harness_output/sonic_tracking/trial_001/
        initial/       — stable ready posture at clip start
        left_stride/   — left leg swing / right arm counter-swing
        right_stride/  — right leg swing / left arm counter-swing
        finale/        — settled final pose after the clip completes
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from roboharness.core.protocol import TaskPhase, TaskProtocol
from roboharness.evaluate.assertions import AssertionEngine, MetricAssertion
from roboharness.evaluate.result import EvaluationResult, Operator, Severity
from roboharness.robots.unitree_g1.locomotion import (
    SONIC_TRACKING_DEFAULT_ANGLES,
    MotionClip,
    SonicLocomotionController,
)
from roboharness.wrappers import RobotHarnessWrapper

if __package__ in (None, ""):
    from sonic_locomotion import (  # type: ignore[import-not-found]
        G1_SCENE_XML,
        SONIC_MIN_TORSO_Z,
        G1Env,
        _download_g1_assets,
    )
else:
    from .sonic_locomotion import (
        G1_SCENE_XML,
        SONIC_MIN_TORSO_Z,
        G1Env,
        _download_g1_assets,
    )

TRACKING_TASK_NAME = "sonic_tracking"
TRACKING_PHASE_NAMES = ["initial", "left_stride", "right_stride", "finale"]
TRACKING_PROTOCOL = TaskProtocol(
    name=TRACKING_TASK_NAME,
    description="SONIC tracking mode: deterministic marching clip replay",
    phases=[
        TaskPhase("initial", "Stable ready pose at the start of the tracking clip"),
        TaskPhase("left_stride", "Left leg swing with visible counter-swing in the arms"),
        TaskPhase("right_stride", "Right leg swing with visible counter-swing in the arms"),
        TaskPhase("finale", "Clip complete and the robot settles back into the end pose"),
    ],
)


def _yaw_to_sixd(yaw: float) -> np.ndarray:
    """Encode a yaw-only rotation into SONIC's row-wise 6D format."""
    cy = np.cos(yaw)
    sy = np.sin(yaw)
    return np.array([cy, -sy, sy, cy, 0.0, 0.0], dtype=np.float32)


def make_tracking_clip(num_frames: int = 180) -> MotionClip:
    """Build a deterministic marching clip for the real SONIC tracking stack."""
    t = np.linspace(0.0, 1.0, num_frames, dtype=np.float32)
    phase = 2.0 * np.pi * t
    envelope = np.clip(np.sin(np.pi * t), 0.0, None).astype(np.float32)
    envelope = envelope**1.2

    joint_positions = np.tile(SONIC_TRACKING_DEFAULT_ANGLES.astype(np.float32), (num_frames, 1))
    swing = envelope * np.sin(phase)
    lift_left = envelope * np.clip(np.sin(phase), 0.0, None)
    lift_right = envelope * np.clip(-np.sin(phase), 0.0, None)

    joint_positions[:, 0] += 0.24 * swing
    joint_positions[:, 6] -= 0.24 * swing
    joint_positions[:, 3] += 0.34 * lift_left
    joint_positions[:, 9] += 0.34 * lift_right
    joint_positions[:, 4] -= 0.14 * lift_left
    joint_positions[:, 10] -= 0.14 * lift_right
    joint_positions[:, 12] += 0.06 * swing
    joint_positions[:, 15] -= 0.22 * swing
    joint_positions[:, 22] += 0.22 * swing
    joint_positions[:, 18] += 0.10 * envelope
    joint_positions[:, 25] += 0.10 * envelope

    joint_velocities = np.gradient(joint_positions, axis=0).astype(np.float32) * 50.0
    root_height = (0.74 + 0.012 * envelope * (1.0 - np.cos(2.0 * phase))).astype(np.float32)
    yaw = 0.08 * swing
    root_rotation_6d = np.stack([_yaw_to_sixd(float(v)) for v in yaw], axis=0)

    return MotionClip(
        joint_positions=joint_positions,
        joint_velocities=joint_velocities,
        root_height=root_height,
        root_rotation_6d=root_rotation_6d,
        fps=50.0,
        name="deterministic_march",
    )


def build_phase_steps(num_frames: int) -> dict[str, int]:
    """Choose semantically useful checkpoints across the tracking clip."""
    initial = max(1, num_frames // 8)
    left_stride = max(initial + 1, num_frames // 3)
    right_stride = max(left_stride + 1, (2 * num_frames) // 3)
    finale = max(right_stride + 1, num_frames)
    return {
        "initial": initial,
        "left_stride": left_stride,
        "right_stride": right_stride,
        "finale": finale,
    }


def summarize_tracking_run(
    tracking_frames: list[int],
    torso_heights: list[float],
    rms_joint_errors: list[float],
    joint_history: list[np.ndarray],
) -> dict[str, float]:
    """Compute run-level metrics that prove the robot stayed up and actually moved."""
    history = np.asarray(joint_history, dtype=np.float32)
    if history.size == 0:
        return {
            "min_torso_z": 0.0,
            "max_torso_z": 0.0,
            "final_tracking_frame": 0.0,
            "mean_rms_joint_error": float("inf"),
            "max_rms_joint_error": float("inf"),
            "left_hip_pitch_span": 0.0,
            "right_hip_pitch_span": 0.0,
            "left_knee_span": 0.0,
            "right_knee_span": 0.0,
        }

    return {
        "min_torso_z": float(np.min(torso_heights)),
        "max_torso_z": float(np.max(torso_heights)),
        "final_tracking_frame": float(tracking_frames[-1]),
        "mean_rms_joint_error": float(np.mean(rms_joint_errors)),
        "max_rms_joint_error": float(np.max(rms_joint_errors)),
        "left_hip_pitch_span": float(np.ptp(history[:, 0])),
        "right_hip_pitch_span": float(np.ptp(history[:, 6])),
        "left_knee_span": float(np.ptp(history[:, 3])),
        "right_knee_span": float(np.ptp(history[:, 9])),
    }


def build_tracking_evaluation(
    summary_metrics: dict[str, float],
    snapshot_metrics: dict[str, dict[str, float]],
    clip_frames: int,
) -> EvaluationResult:
    """Evaluate the tracking run with explicit pass/fail thresholds."""
    assertions = [
        MetricAssertion("min_torso_z", Operator.GE, SONIC_MIN_TORSO_Z, Severity.CRITICAL),
        MetricAssertion("final_tracking_frame", Operator.GE, clip_frames - 1, Severity.CRITICAL),
        MetricAssertion("mean_rms_joint_error", Operator.LE, 0.60, Severity.MAJOR),
        MetricAssertion("max_rms_joint_error", Operator.LE, 0.95, Severity.MAJOR),
        MetricAssertion("left_hip_pitch_span", Operator.GE, 0.10, Severity.MAJOR),
        MetricAssertion("right_hip_pitch_span", Operator.GE, 0.10, Severity.MAJOR),
        MetricAssertion("left_knee_span", Operator.GE, 0.10, Severity.MAJOR),
        MetricAssertion("right_knee_span", Operator.GE, 0.10, Severity.MAJOR),
    ]
    for phase_name in TRACKING_PHASE_NAMES:
        assertions.extend(
            [
                MetricAssertion(
                    "torso_z",
                    Operator.GE,
                    SONIC_MIN_TORSO_Z,
                    Severity.CRITICAL,
                    phase=phase_name,
                ),
                MetricAssertion(
                    "rms_joint_error",
                    Operator.LE,
                    0.85,
                    Severity.MAJOR,
                    phase=phase_name,
                ),
            ]
        )
    report = {
        "summary_metrics": summary_metrics,
        "snapshot_metrics": snapshot_metrics,
    }
    return AssertionEngine(assertions).evaluate(report)


def _write_checkpoint_metrics(state_path: Path, metrics: dict[str, float]) -> None:
    """Augment a checkpoint state.json with tracking-specific metrics."""
    state = json.loads(state_path.read_text())
    state.update(metrics)
    state_path.write_text(json.dumps(state, indent=2))


def generate_html_report(
    output_dir: Path,
    clip: MotionClip,
    summary_metrics: dict[str, float],
    evaluation_result: EvaluationResult,
) -> Path:
    """Generate a self-contained HTML report for the tracking demo."""
    from roboharness.reporting import generate_html_report as _generate

    return _generate(
        output_dir,
        TRACKING_TASK_NAME,
        title="SONIC Motion Tracking Report",
        subtitle=(
            "GEAR-SONIC encoder+decoder tracking on Unitree G1 using the published "
            "HuggingFace ONNX models. The clip is a deterministic marching fixture run "
            "through the real tracking stack, not a mock decoder path."
        ),
        accent_color="#0f766e",
        summary_html=(
            "<strong>Controller:</strong> "
            "SonicLocomotionController tracking mode (real ONNX, CPU)"
            "<br/><strong>Models:</strong> "
            "nvidia/GEAR-SONIC model_encoder.onnx + model_decoder.onnx"
            "<br/><strong>Robot:</strong> "
            "Unitree G1 29-DOF (lerobot/unitree-g1-mujoco)"
            f"<br/><strong>Clip:</strong> {clip.name} "
            f"({clip.num_frames} frames @ {clip.fps:.0f} Hz)"
            f"<br/><strong>Observed:</strong> "
            f"min torso z={summary_metrics['min_torso_z']:.3f}m, "
            f"mean RMS joint error={summary_metrics['mean_rms_joint_error']:.3f}rad"
            "<br/><strong>Hip spans L/R:</strong> "
            f"{summary_metrics['left_hip_pitch_span']:.3f}/"
            f"{summary_metrics['right_hip_pitch_span']:.3f}rad"
        ),
        footer_text="Generated by <code>examples/sonic_tracking.py --report</code>",
        meshcat_mode="none",
        evaluation_result=evaluation_result,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="SONIC motion tracking demo on Unitree G1")
    parser.add_argument("--output-dir", default="./harness_output", help="Output directory")
    parser.add_argument("--report", action="store_true", help="Generate HTML report")
    parser.add_argument("--width", type=int, default=640, help="Render width")
    parser.add_argument("--height", type=int, default=480, help="Render height")
    parser.add_argument(
        "--assert-success",
        action="store_true",
        help="Exit non-zero if the robot falls or tracking quality regresses",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    clip = make_tracking_clip()
    phase_steps = build_phase_steps(clip.num_frames)
    n_steps = phase_steps["finale"]

    print("=" * 60)
    print("  Roboharness: SONIC Motion Tracking Demo")
    print("=" * 60)

    print("\n[1/4] Downloading G1 model from HuggingFace ...")
    repo_path = _download_g1_assets()
    xml_path = repo_path / G1_SCENE_XML

    env = G1Env(model_path=xml_path, render_width=args.width, render_height=args.height)
    cameras = env.cameras
    print(f"      Cameras: {cameras}")
    print(f"      Obs space: {env.observation_space.shape}")

    print("[2/4] Loading SONIC tracking controller ...")
    sonic = SonicLocomotionController()
    sonic.set_tracking_clip(clip)
    print("      SONIC planner + tracking ONNX models ready (encoder/decoder load lazily)")

    print("[3/4] Wrapping with RobotHarnessWrapper ...")
    wrapped = RobotHarnessWrapper(
        env,
        protocol=TRACKING_PROTOCOL,
        phase_steps=phase_steps,
        cameras=cameras,
        output_dir=str(output_dir),
        task_name=TRACKING_TASK_NAME,
    )
    print(f"      Protocol: {wrapped.active_protocol.name}")
    print(f"      Multi-camera: {wrapped.has_multi_camera}")
    print(f"      Clip: {clip.name} ({clip.num_frames} frames)")

    print(f"[4/4] Running tracking simulation ({n_steps} steps) ...")
    _obs, _info = wrapped.reset()
    sonic.reset()

    checkpoint_infos: list[dict[str, Any]] = []
    snapshot_metrics: dict[str, dict[str, float]] = {}
    tracking_frames: list[int] = []
    torso_heights: list[float] = []
    rms_joint_errors: list[float] = []
    joint_history: list[np.ndarray] = []

    for _step in range(n_steps):
        state = {"qpos": env._data.qpos, "qvel": env._data.qvel}
        action = sonic.compute(command={"tracking": True}, state=state)
        _obs, _reward, _terminated, _truncated, info = wrapped.step(action)

        frame_index = int(sonic._tracking_frame_index)
        ref_index = min(frame_index, clip.num_frames - 1)
        actual_joints = np.asarray(env._data.qpos[7:36], dtype=np.float32).copy()
        ref_joints = clip.joint_positions[ref_index]
        rms_joint_error = float(np.sqrt(np.mean(np.square(actual_joints - ref_joints))))
        torso_z = env._get_torso_z()

        tracking_frames.append(frame_index)
        torso_heights.append(torso_z)
        rms_joint_errors.append(rms_joint_error)
        joint_history.append(actual_joints)

        if "checkpoint" in info:
            cp = info["checkpoint"]
            checkpoint_infos.append(cp)
            phase_metrics = {
                "torso_z": torso_z,
                "tracking_frame_index": float(frame_index),
                "rms_joint_error": rms_joint_error,
                "left_hip_pitch": float(actual_joints[0]),
                "right_hip_pitch": float(actual_joints[6]),
            }
            snapshot_metrics[cp["name"]] = phase_metrics
            _write_checkpoint_metrics(Path(cp["files"]["state"]), phase_metrics)
            print(
                f"      Checkpoint '{cp['name']}' at step {cp['step']} | "
                f"torso_z={torso_z:.3f}m | frame={frame_index} | error={rms_joint_error:.3f}rad"
            )

    summary_metrics = summarize_tracking_run(
        tracking_frames, torso_heights, rms_joint_errors, joint_history
    )
    evaluation_result = build_tracking_evaluation(
        summary_metrics, snapshot_metrics, clip.num_frames
    )

    failures: list[str] = []
    if len(checkpoint_infos) != len(TRACKING_PROTOCOL.phases):
        failures.append(
            f"Expected {len(TRACKING_PROTOCOL.phases)} checkpoints, got {len(checkpoint_infos)}"
        )

    required_state_fields = {"torso_z", "tracking_frame_index", "rms_joint_error"}
    for cp in checkpoint_infos:
        state_path = Path(cp["files"]["state"])
        state = json.loads(state_path.read_text())
        missing = sorted(required_state_fields - state.keys())
        if missing:
            failures.append(f"Checkpoint '{cp['name']}': state.json missing {missing}")

    for result in evaluation_result.failed:
        failures.append(result.message)

    trial_dir = output_dir / TRACKING_TASK_NAME / "trial_001"
    total_images = len(list(trial_dir.rglob("*_rgb.png"))) if trial_dir.exists() else 0
    print(f"\n      {len(checkpoint_infos)} checkpoints, {total_images} images")
    print(
        "      Summary: "
        f"min_torso_z={summary_metrics['min_torso_z']:.3f}m, "
        f"mean_error={summary_metrics['mean_rms_joint_error']:.3f}rad, "
        f"frame={summary_metrics['final_tracking_frame']:.0f}, "
        f"hip_spans={summary_metrics['left_hip_pitch_span']:.3f}/"
        f"{summary_metrics['right_hip_pitch_span']:.3f}rad"
    )

    if failures:
        print("      VALIDATION FAILED:")
        for msg in failures:
            print(f"        FAIL: {msg}")
    else:
        print(f"      VALIDATION PASSED: verdict={evaluation_result.verdict.value.upper()}")

    if args.report:
        report_path = generate_html_report(output_dir, clip, summary_metrics, evaluation_result)
        print(f"      HTML report: {report_path}")

    print("\n" + "=" * 60)
    wrapped.close()

    if args.assert_success and failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
