"""Alarmed-grasp wedge helpers kept local to the MuJoCo example."""

from __future__ import annotations

import base64
import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from roboharness._utils import save_json
from roboharness.evaluate.assertions import AssertionEngine, MetricAssertion
from roboharness.evaluate.result import EvaluationResult, Operator, Severity

try:
    from examples._mujoco_grasp_fixture import (
        MUJOCO_GRASP_PHASE_LABELS,
        MUJOCO_GRASP_PHASE_ORDER,
        MUJOCO_GRASP_PRIMARY_VIEWS,
        MUJOCO_GRASP_TASK,
    )
except ModuleNotFoundError:  # pragma: no cover - script execution path
    from _mujoco_grasp_fixture import (  # type: ignore[no-redef]
        MUJOCO_GRASP_PHASE_LABELS,
        MUJOCO_GRASP_PHASE_ORDER,
        MUJOCO_GRASP_PRIMARY_VIEWS,
        MUJOCO_GRASP_TASK,
    )

TABLE_SURFACE_Z = 0.22
FINGER_BODY_WIDTH_M = 0.024
CUBE_WIDTH_M = 0.05
CANONICAL_REPORT_NAME = "report.html"
ASSET_ROOT = Path(__file__).resolve().parents[1] / "assets" / "example_mujoco_grasp"
BASELINE_REPORT_PATH = ASSET_ROOT / "baseline_autonomous_report.json"
BASELINE_VISUAL_ROOT = ASSET_ROOT / "baseline_visual"
KNOWN_BAD_VISUAL_ROOT = ASSET_ROOT / "known_bad_visual"
PHASE_COMPARISON_METRICS = (
    "grip_center_error_mm",
    "pinch_gap_error_mm",
    "pinch_elevation_deg",
    "gripper_skew_deg",
    "cube_height_mm",
    "contact_count",
    "phase_runtime_s",
)
ROOT_CAUSE_BY_METRIC = {
    "grip_center_error_mm_abs_delta": "approach_alignment_regression",
    "pinch_gap_error_mm_abs_delta": "contact_asymmetry",
    "pinch_elevation_deg_abs_delta": "contact_asymmetry",
    "gripper_skew_deg_abs_delta": "approach_alignment_regression",
    "contact_count": "grasp_drop",
    "cube_height_mm": "lift_regression",
    "cube_height_mm_abs_delta": "lift_regression",
    "loop_runtime_s_abs_delta": "runtime_regression",
    "loop_runtime_s": "runtime_regression",
    "max_abs_qvel": "stability_regression",
}
ROOT_CAUSE_CATEGORY = {
    "approach_alignment_regression": "trajectory_failure",
    "contact_asymmetry": "contact_failure",
    "grasp_drop": "lift_failure",
    "lift_regression": "lift_failure",
    "runtime_regression": "runtime_regression",
    "stability_regression": "stability_failure",
    "trajectory_regression": "trajectory_failure",
}
PRIMARY_VIEWS_BY_ROOT_CAUSE = {
    "approach_alignment_regression": ["side", "top"],
    "contact_asymmetry": ["front", "side"],
    "grasp_drop": ["front", "side"],
    "lift_regression": ["front", "side"],
    "runtime_regression": ["front"],
    "stability_regression": ["side"],
    "trajectory_regression": ["side", "top"],
}
AMBIGUOUS_STILL_IMAGE_METRICS = {"phase_runtime_s", "loop_runtime_s", "max_abs_qvel"}
MUJOCO_GRASP_ASSERTIONS = (
    MetricAssertion(
        metric="loop_runtime_s",
        operator=Operator.LT,
        threshold=10.0,
        severity=Severity.MAJOR,
    ),
    MetricAssertion(
        metric="loop_runtime_s_abs_delta",
        operator=Operator.LT,
        threshold=1.0,
        severity=Severity.MAJOR,
    ),
    MetricAssertion(
        metric="max_abs_qvel",
        operator=Operator.LT,
        threshold=50.0,
        severity=Severity.MAJOR,
    ),
    MetricAssertion(
        metric="grip_center_error_mm_abs_delta",
        operator=Operator.LT,
        threshold=12.0,
        severity=Severity.CRITICAL,
        phase="approach",
    ),
    MetricAssertion(
        metric="gripper_skew_deg_abs_delta",
        operator=Operator.LT,
        threshold=8.0,
        severity=Severity.MAJOR,
        phase="approach",
    ),
    MetricAssertion(
        metric="pinch_gap_error_mm_abs_delta",
        operator=Operator.LT,
        threshold=10.0,
        severity=Severity.MAJOR,
        phase="grasp",
    ),
    MetricAssertion(
        metric="pinch_elevation_deg_abs_delta",
        operator=Operator.LT,
        threshold=10.0,
        severity=Severity.MAJOR,
        phase="grasp",
    ),
    MetricAssertion(
        metric="contact_count",
        operator=Operator.GE,
        threshold=1.0,
        severity=Severity.CRITICAL,
        phase="lift",
    ),
    MetricAssertion(
        metric="cube_height_mm",
        operator=Operator.GT,
        threshold=5.0,
        severity=Severity.CRITICAL,
        phase="lift",
    ),
    MetricAssertion(
        metric="cube_height_mm_abs_delta",
        operator=Operator.LT,
        threshold=12.0,
        severity=Severity.CRITICAL,
        phase="lift",
    ),
)


@dataclass
class AlarmRecord:
    """Evaluator-derived alarm or OK status for one assertion."""

    code: str
    title: str
    status: str
    severity: str
    phase_id: str | None
    phase_label: str
    metric: str
    actual_value: float | None
    expected: str
    baseline_value: float | None
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "title": self.title,
            "status": self.status,
            "severity": self.severity,
            "phase_id": self.phase_id,
            "phase": self.phase_label,
            "metric": self.metric,
            "actual_value": self.actual_value,
            "expected": self.expected,
            "baseline_value": self.baseline_value,
            "message": self.message,
        }


@dataclass
class PhaseStatus:
    """Machine-readable status summary for one grasp phase."""

    phase_id: str
    phase_label: str
    status: str
    step: int
    sim_time_s: float
    regressions: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase_id": self.phase_id,
            "phase": self.phase_label,
            "status": self.status,
            "step": self.step,
            "sim_time_s": self.sim_time_s,
            "regressions": list(self.regressions),
        }


@dataclass
class PhaseManifest:
    """Agent-facing manifest for the next diagnostic iteration."""

    task: str
    verdict: str
    failed_phase_id: str | None
    failed_phase: str | None
    suspected_root_cause: str
    primary_views: list[str]
    regressions: list[str]
    rerun_hint: str
    agent_next_action: str
    evidence_paths: list[str]
    phase_aliases: dict[str, str]
    phase_statuses: list[PhaseStatus]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "task": self.task,
            "verdict": self.verdict,
            "failed_phase_id": self.failed_phase_id,
            "failed_phase": self.failed_phase,
            "suspected_root_cause": self.suspected_root_cause,
            "primary_views": list(self.primary_views),
            "regressions": list(self.regressions),
            "rerun_hint": self.rerun_hint,
            "agent_next_action": self.agent_next_action,
            "evidence_paths": list(self.evidence_paths),
            "phase_aliases": dict(self.phase_aliases),
            "phase_statuses": [status.to_dict() for status in self.phase_statuses],
        }


@dataclass
class MetricExplanation:
    """Single metric chip rendered in the summary evidence cards."""

    metric: str
    copy: str


@dataclass
class EvidencePair:
    """Current-vs-baseline evidence pair for one selected phase/view."""

    phase_id: str
    phase_label: str
    view_name: str
    current_image_path: Path | None
    baseline_image_path: Path | None
    status: str
    metric_explanations: list[MetricExplanation]
    interpretation_caption: str
    diagnostic_message: str | None = None


def load_blessed_baseline(path: str | Path | None = None) -> dict[str, Any]:
    """Load the blessed deterministic baseline fixture."""
    baseline_path = Path(path) if path is not None else BASELINE_REPORT_PATH
    if not baseline_path.exists():
        raise FileNotFoundError(f"Blessed baseline report not found: {baseline_path}")
    with baseline_path.open() as fh:
        return json.load(fh)


def collect_phase_metrics(
    harness: Any,
    backend: Any,
    checkpoint_results: dict[str, Any],
) -> dict[str, dict[str, float]]:
    """Collect deterministic phase metrics by restoring each saved checkpoint."""
    try:
        import mujoco
    except ImportError as exc:
        raise ImportError("MuJoCo is required to build grasp-loop metrics.") from exc

    missing = [phase for phase in MUJOCO_GRASP_PHASE_ORDER if phase not in checkpoint_results]
    if missing:
        raise ValueError(f"Missing checkpoint results for phases: {missing}")

    model = backend._model
    data = backend._data
    cube_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    finger_left_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "finger_left")
    finger_right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "finger_right")

    metrics: dict[str, dict[str, float]] = {}
    previous_sim_time = 0.0

    for phase in MUJOCO_GRASP_PHASE_ORDER:
        harness.restore_checkpoint(phase)
        mujoco.mj_forward(model, data)
        capture = checkpoint_results[phase]

        cube_pos = np.asarray(data.xpos[cube_body_id], dtype=np.float64)
        left_pos = np.asarray(data.xpos[finger_left_id], dtype=np.float64)
        right_pos = np.asarray(data.xpos[finger_right_id], dtype=np.float64)

        finger_vector = left_pos - right_pos
        grip_center = (left_pos + right_pos) / 2.0
        gap_m = max(float(np.linalg.norm(finger_vector)) - FINGER_BODY_WIDTH_M, 0.0)
        lateral_span = max(abs(float(finger_vector[1])), 1e-9)

        metrics[phase] = {
            "step": float(capture.step),
            "sim_time_s": float(capture.sim_time),
            "phase_runtime_s": float(capture.sim_time - previous_sim_time),
            "cube_height_mm": float((cube_pos[2] - TABLE_SURFACE_Z) * 1000.0),
            "grip_center_error_mm": float(np.linalg.norm(grip_center - cube_pos) * 1000.0),
            "pinch_gap_error_mm": float(abs(gap_m - CUBE_WIDTH_M) * 1000.0),
            "pinch_elevation_deg": float(
                np.degrees(np.arctan2(abs(float(finger_vector[2])), lateral_span))
            ),
            "gripper_skew_deg": float(
                np.degrees(np.arctan2(abs(float(finger_vector[0])), lateral_span))
            ),
            "contact_count": float(
                _count_cube_finger_contacts(
                    model=model,
                    data=data,
                    cube_body_id=cube_body_id,
                    finger_left_id=finger_left_id,
                    finger_right_id=finger_right_id,
                )
            ),
            "max_abs_qvel": float(np.max(np.abs(np.asarray(data.qvel, dtype=np.float64)))),
        }
        previous_sim_time = float(capture.sim_time)

    harness.restore_checkpoint(MUJOCO_GRASP_PHASE_ORDER[-1])
    return metrics


def build_autonomous_report(
    *,
    snapshot_metrics: dict[str, dict[str, float]],
    baseline_report: dict[str, Any],
    baseline_source: str,
) -> dict[str, Any]:
    """Build the report dict consumed by the evaluator and downstream artifacts."""
    baseline_snapshot_metrics = baseline_report.get("snapshot_metrics", {})
    baseline_summary_metrics = baseline_report.get("summary_metrics", {})

    missing_baseline_phases = [
        phase for phase in MUJOCO_GRASP_PHASE_ORDER if phase not in baseline_snapshot_metrics
    ]
    if missing_baseline_phases:
        raise KeyError(f"Baseline snapshot metrics missing phases: {missing_baseline_phases}")

    augmented_snapshot_metrics: dict[str, dict[str, float]] = {}
    for phase in MUJOCO_GRASP_PHASE_ORDER:
        current = dict(snapshot_metrics[phase])
        baseline_metrics = baseline_snapshot_metrics[phase]
        for metric in PHASE_COMPARISON_METRICS:
            if metric not in current:
                raise KeyError(f"Current snapshot metrics missing '{metric}' for phase '{phase}'")
            if metric not in baseline_metrics:
                raise KeyError(f"Baseline snapshot metrics missing '{metric}' for phase '{phase}'")
            baseline_value = float(baseline_metrics[metric])
            current[f"{metric}_baseline"] = baseline_value
            current[f"{metric}_abs_delta"] = abs(float(current[metric]) - baseline_value)
        augmented_snapshot_metrics[phase] = current

    loop_runtime_s = float(snapshot_metrics[MUJOCO_GRASP_PHASE_ORDER[-1]]["sim_time_s"])
    final_cube_height_mm = float(snapshot_metrics["lift"]["cube_height_mm"])
    final_contact_count = float(snapshot_metrics["lift"]["contact_count"])
    max_abs_qvel = max(float(phase["max_abs_qvel"]) for phase in snapshot_metrics.values())

    loop_runtime_baseline = float(baseline_summary_metrics.get("loop_runtime_s", loop_runtime_s))
    final_cube_height_baseline = float(
        baseline_summary_metrics.get(
            "final_cube_height_mm",
            baseline_snapshot_metrics["lift"]["cube_height_mm"],
        )
    )
    final_contact_count_baseline = float(
        baseline_summary_metrics.get(
            "final_contact_count",
            baseline_snapshot_metrics["lift"]["contact_count"],
        )
    )
    max_abs_qvel_baseline = float(
        baseline_summary_metrics.get(
            "max_abs_qvel",
            max(float(phase["max_abs_qvel"]) for phase in baseline_snapshot_metrics.values()),
        )
    )

    summary_metrics = {
        "loop_runtime_s": loop_runtime_s,
        "loop_runtime_s_baseline": loop_runtime_baseline,
        "loop_runtime_s_abs_delta": abs(loop_runtime_s - loop_runtime_baseline),
        "final_cube_height_mm": final_cube_height_mm,
        "final_cube_height_mm_baseline": final_cube_height_baseline,
        "final_cube_height_mm_abs_delta": abs(final_cube_height_mm - final_cube_height_baseline),
        "final_contact_count": final_contact_count,
        "final_contact_count_baseline": final_contact_count_baseline,
        "final_contact_count_abs_delta": abs(final_contact_count - final_contact_count_baseline),
        "max_abs_qvel": max_abs_qvel,
        "max_abs_qvel_baseline": max_abs_qvel_baseline,
        "max_abs_qvel_abs_delta": abs(max_abs_qvel - max_abs_qvel_baseline),
    }

    return {
        "schema_version": 1,
        "task": MUJOCO_GRASP_TASK,
        "case_id": "deterministic_mujoco_grasp",
        "baseline_source": baseline_source,
        "phase_order": list(MUJOCO_GRASP_PHASE_ORDER),
        "phase_aliases": dict(MUJOCO_GRASP_PHASE_LABELS),
        "primary_views": dict(MUJOCO_GRASP_PRIMARY_VIEWS),
        "summary_metrics": summary_metrics,
        "snapshot_metrics": augmented_snapshot_metrics,
        "baseline_summary_metrics": baseline_summary_metrics,
        "baseline_snapshot_metrics": baseline_snapshot_metrics,
        "failure_taxonomy": [],
        "verdict_reasons": [],
    }


def evaluate_autonomous_report(
    report: dict[str, Any],
    *,
    report_path: str = "",
) -> EvaluationResult:
    """Evaluate the MuJoCo wedge report using one shared verdict path."""
    return AssertionEngine(MUJOCO_GRASP_ASSERTIONS).evaluate(report, report_path=report_path)


def build_alarms(report: dict[str, Any], evaluation_result: EvaluationResult) -> list[AlarmRecord]:
    """Convert evaluator results into alarm cards and machine-readable records."""
    alarms: list[AlarmRecord] = []
    for result in evaluation_result.results:
        phase_id = None if result.phase == "*" else result.phase
        phase_label = "loop" if phase_id is None else MUJOCO_GRASP_PHASE_LABELS[result.phase]
        status = "ok" if result.passed else "alarm"
        code = f"{phase_id or 'summary'}:{result.metric}"
        alarms.append(
            AlarmRecord(
                code=code,
                title=_metric_title(result.metric),
                status=status,
                severity=result.severity.value,
                phase_id=phase_id,
                phase_label=phase_label,
                metric=result.metric,
                actual_value=result.actual_value,
                expected=_format_expected(result.operator.value, result.threshold),
                baseline_value=_lookup_baseline_value(report, result.phase, result.metric),
                message=result.message,
            )
        )
    return alarms


def build_phase_manifest(
    report: dict[str, Any],
    evaluation_result: EvaluationResult,
    alarms: list[AlarmRecord],
) -> PhaseManifest:
    """Build the next-action manifest from the evaluator output."""
    phase_statuses = _build_phase_statuses(report, evaluation_result)
    failed_phase_id = _resolve_failed_phase_id(phase_statuses, evaluation_result)
    failed_phase = (
        MUJOCO_GRASP_PHASE_LABELS[failed_phase_id] if failed_phase_id is not None else None
    )
    failed_results = [result for result in evaluation_result.failed if result.phase != "*"]
    if failed_results:
        primary_failure = failed_results[0]
    elif evaluation_result.failed:
        primary_failure = evaluation_result.failed[0]
    else:
        primary_failure = None
    if primary_failure is None:
        root_cause = "none"
        primary_views = list(MUJOCO_GRASP_PRIMARY_VIEWS["approach"])
        rerun_hint = "not_required"
        evidence_paths: list[str] = []
    else:
        root_cause = ROOT_CAUSE_BY_METRIC.get(primary_failure.metric, "trajectory_regression")
        primary_views = PRIMARY_VIEWS_BY_ROOT_CAUSE.get(
            root_cause,
            MUJOCO_GRASP_PRIMARY_VIEWS.get(failed_phase_id or "lift", ["front"]),
        )
        rerun_hint = _build_rerun_hint(failed_phase_id)
        evidence_paths = [f"{failed_phase_id}/{view}_rgb.png" for view in primary_views]
    regressions = [alarm.metric for alarm in alarms if alarm.status == "alarm"]
    agent_next_action = _build_agent_next_action(
        failed_phase_id=failed_phase_id,
        failed_phase=failed_phase,
        primary_views=primary_views,
        primary_failure=primary_failure,
    )

    return PhaseManifest(
        task=MUJOCO_GRASP_TASK,
        verdict=evaluation_result.verdict.value,
        failed_phase_id=failed_phase_id,
        failed_phase=failed_phase,
        suspected_root_cause=root_cause,
        primary_views=primary_views,
        regressions=regressions,
        rerun_hint=rerun_hint,
        agent_next_action=agent_next_action,
        evidence_paths=evidence_paths,
        phase_aliases=dict(MUJOCO_GRASP_PHASE_LABELS),
        phase_statuses=phase_statuses,
    )


def resolve_evidence_pairs(
    *,
    trial_dir: Path,
    baseline_visual_root: Path,
    manifest: PhaseManifest,
    report: dict[str, Any],
) -> list[EvidencePair]:
    """Resolve deterministic current-vs-baseline evidence pairs for the failing phase."""
    failed_phase_id = manifest.failed_phase_id
    failed_phase = manifest.failed_phase
    if failed_phase_id is None or failed_phase is None:
        return []

    canonical_views = list(report.get("primary_views", {}).get(failed_phase_id, []))
    metric_explanations = _build_metric_explanations(report, failed_phase_id)
    pairs: list[EvidencePair] = []

    for view_name in manifest.primary_views[:2]:
        expected_rel = f"{failed_phase_id}/{view_name}_rgb.png"
        if view_name not in canonical_views or expected_rel not in manifest.evidence_paths:
            message = (
                "Manifest/view contract mismatch. "
                f"The report requested {failed_phase_id}/{view_name}, but no matching asset "
                "exists under the blessed baseline root."
            )
            pairs.append(
                EvidencePair(
                    phase_id=failed_phase_id,
                    phase_label=failed_phase,
                    view_name=view_name,
                    current_image_path=None,
                    baseline_image_path=None,
                    status="mismatch",
                    metric_explanations=[],
                    interpretation_caption=message,
                    diagnostic_message=message,
                )
            )
            continue

        current_image_path = _resolve_visual_image(trial_dir, failed_phase_id, view_name)
        baseline_image_path = _resolve_visual_image(
            baseline_visual_root,
            failed_phase_id,
            view_name,
        )
        if current_image_path is None or baseline_image_path is None:
            message = (
                "Manifest/view contract mismatch. "
                f"The report requested {failed_phase_id}/{view_name}, but the resolved asset "
                "path escaped the allowed evidence roots."
            )
            pairs.append(
                EvidencePair(
                    phase_id=failed_phase_id,
                    phase_label=failed_phase,
                    view_name=view_name,
                    current_image_path=None,
                    baseline_image_path=None,
                    status="mismatch",
                    metric_explanations=[],
                    interpretation_caption=message,
                    diagnostic_message=message,
                )
            )
            continue

        current_exists = current_image_path.exists()
        baseline_exists = baseline_image_path.exists()
        if current_exists and baseline_exists:
            status = "ambiguous" if _metric_set_is_ambiguous(metric_explanations) else "full"
            diagnostic_message = (
                f"Still-image evidence is suggestive for {failed_phase_id}/{view_name}, "
                f"but temporal proof is weak. Re-run from {manifest.rerun_hint} if motion "
                "timing is part of the diagnosis."
                if status == "ambiguous"
                else None
            )
            pairs.append(
                EvidencePair(
                    phase_id=failed_phase_id,
                    phase_label=failed_phase,
                    view_name=view_name,
                    current_image_path=current_image_path,
                    baseline_image_path=baseline_image_path,
                    status=status,
                    metric_explanations=metric_explanations,
                    interpretation_caption=_build_interpretation_caption(
                        phase_id=failed_phase_id,
                        phase_label=failed_phase,
                        view_name=view_name,
                        status=status,
                    ),
                    diagnostic_message=diagnostic_message,
                )
            )
            continue

        if current_exists or baseline_exists:
            if not baseline_exists:
                diagnostic_message = (
                    f"Baseline image missing for {failed_phase_id}/{view_name}. Rebuild or "
                    "restore the blessed baseline pack."
                )
                current_path: Path | None = current_image_path
                baseline_path: Path | None = None
            else:
                diagnostic_message = (
                    f"Current capture missing for {failed_phase_id}/{view_name}. Re-run from "
                    f"{manifest.rerun_hint} to rebuild evidence."
                )
                current_path = None
                baseline_path = baseline_image_path
            pairs.append(
                EvidencePair(
                    phase_id=failed_phase_id,
                    phase_label=failed_phase,
                    view_name=view_name,
                    current_image_path=current_path,
                    baseline_image_path=baseline_path,
                    status="partial",
                    metric_explanations=metric_explanations,
                    interpretation_caption=_build_interpretation_caption(
                        phase_id=failed_phase_id,
                        phase_label=failed_phase,
                        view_name=view_name,
                        status="partial",
                    ),
                    diagnostic_message=diagnostic_message,
                )
            )
            continue

        diagnostic_message = (
            "No visual evidence available for the failing phase. "
            f"Re-run from {manifest.rerun_hint} to rebuild evidence."
        )
        pairs.append(
            EvidencePair(
                phase_id=failed_phase_id,
                phase_label=failed_phase,
                view_name=view_name,
                current_image_path=None,
                baseline_image_path=None,
                status="empty",
                metric_explanations=metric_explanations,
                interpretation_caption=_build_interpretation_caption(
                    phase_id=failed_phase_id,
                    phase_label=failed_phase,
                    view_name=view_name,
                    status="empty",
                ),
                diagnostic_message=diagnostic_message,
            )
        )

    return pairs


def write_artifact_pack(
    *,
    trial_dir: Path,
    report: dict[str, Any],
    evaluation_result: EvaluationResult,
    alarms: list[AlarmRecord],
    manifest: PhaseManifest,
    report_generated: bool,
) -> None:
    """Write the three machine-readable wedge artifacts into the trial directory."""
    report_with_evaluation = dict(report)
    taxonomy, verdict_reasons = _build_failure_taxonomy(evaluation_result)
    report_with_evaluation["evaluation"] = evaluation_result.to_dict()
    report_with_evaluation["verdict"] = evaluation_result.verdict.value
    report_with_evaluation["verdict_reasons"] = verdict_reasons
    report_with_evaluation["failure_taxonomy"] = taxonomy
    artifacts: dict[str, str] = {
        "autonomous_report": "autonomous_report.json",
        "alarms": "alarms.json",
        "phase_manifest": "phase_manifest.json",
    }
    if report_generated:
        artifacts["report_html"] = CANONICAL_REPORT_NAME
    report_with_evaluation["artifacts"] = artifacts

    save_json(report_with_evaluation, trial_dir / "autonomous_report.json")
    save_json(
        {
            "schema_version": 1,
            "task": MUJOCO_GRASP_TASK,
            "verdict": evaluation_result.verdict.value,
            "failed_alarm_count": sum(1 for alarm in alarms if alarm.status == "alarm"),
            "alarms": [alarm.to_dict() for alarm in alarms],
        },
        trial_dir / "alarms.json",
    )
    save_json(manifest.to_dict(), trial_dir / "phase_manifest.json")


def build_summary_html(
    report: dict[str, Any],
    alarms: list[AlarmRecord],
    manifest: PhaseManifest,
    evidence_pairs: list[EvidencePair],
) -> str:
    """Build the alarm-first HTML summary block for the shared report renderer."""
    visible_alarms = alarms[:4]
    alarm_cards = "".join(_render_alarm_card(alarm) for alarm in visible_alarms)
    evidence_state = _summary_evidence_state(manifest, evidence_pairs)
    evidence_section = _render_evidence_section(manifest, evidence_pairs)

    phase_cards = "".join(_render_phase_card(report, status) for status in manifest.phase_statuses)
    baseline_name = html.escape(Path(str(report["baseline_source"])).name)
    failed_phase_text = html.escape(manifest.failed_phase or "none")
    regressions = ", ".join(manifest.regressions) if manifest.regressions else "none"
    selected_views = ", ".join(manifest.primary_views[:2]) if manifest.primary_views else "none"
    diagnostics = _collect_diagnostic_messages(evidence_pairs)
    diagnostics_html = "".join(f"<p>{html.escape(message)}</p>" for message in diagnostics)
    root_cause_display = manifest.suspected_root_cause if manifest.failed_phase_id is not None else "none"
    rerun_hint_row = (
        f"<tr><th>Rerun hint</th><td><code>{html.escape(manifest.rerun_hint)}</code></td></tr>"
        if manifest.failed_phase_id is not None
        else "<tr><th>Rerun hint</th><td>No rerun required</td></tr>"
    )
    if manifest.failed_phase_id is None:
        agent_context_html = ""
    else:
        agent_context_html = (
            "<p><strong>Why this proof:</strong> "
            f"first failing phase <code>{html.escape(manifest.failed_phase_id)}</code> "
            f"with manifest-selected views <code>{html.escape(selected_views)}</code>. "
            f"State: {html.escape(evidence_state)}.</p>"
            f"<p><strong>Rerun hint:</strong> <code>{html.escape(manifest.rerun_hint)}</code></p>"
        )

    return (
        '<div class="alarm-grid">'
        f"{alarm_cards}"
        "</div>"
        f"{evidence_section}"
        '<div class="agent-panel">'
        "<strong>Agent Next Action</strong>"
        f"<p>{html.escape(manifest.agent_next_action)}</p>"
        f"{agent_context_html}"
        f"{diagnostics_html}"
        "</div>"
        '<div class="report-grid">'
        '<div class="summary-card">'
        "<h3>Phase Timeline</h3>"
        f'<div class="phase-timeline">{phase_cards}</div>'
        "</div>"
        '<div class="summary-card">'
        "<h3>Artifact Pack</h3>"
        '<div class="table-scroll"><table class="meta-table">'
        f"<tr><th>Baseline</th><td><code>{baseline_name}</code></td></tr>"
        f"<tr><th>Verdict</th><td><strong>{html.escape(manifest.verdict.upper())}</strong></td></tr>"
        f"<tr><th>Evidence state</th><td>{html.escape(evidence_state)}</td></tr>"
        f"<tr><th>Failed phase</th><td>{failed_phase_text}</td></tr>"
        f"<tr><th>Selected views</th><td><code>{html.escape(selected_views)}</code></td></tr>"
        "<tr><th>Root cause</th><td><code>"
        f"{html.escape(root_cause_display)}</code></td></tr>"
        f"{rerun_hint_row}"
        f"<tr><th>Regressions</th><td><code>{html.escape(regressions)}</code></td></tr>"
        "</table></div>"
        "</div>"
        "</div>"
    )


def _count_cube_finger_contacts(
    *,
    model: Any,
    data: Any,
    cube_body_id: int,
    finger_left_id: int,
    finger_right_id: int,
) -> int:
    touching_fingers: set[int] = set()
    for contact_idx in range(int(data.ncon)):
        contact = data.contact[contact_idx]
        bodies = {
            int(model.geom_bodyid[contact.geom1]),
            int(model.geom_bodyid[contact.geom2]),
        }
        if cube_body_id not in bodies:
            continue
        if finger_left_id in bodies:
            touching_fingers.add(finger_left_id)
        if finger_right_id in bodies:
            touching_fingers.add(finger_right_id)
    return len(touching_fingers)


def _build_phase_statuses(
    report: dict[str, Any],
    evaluation_result: EvaluationResult,
) -> list[PhaseStatus]:
    statuses: list[PhaseStatus] = []
    for phase in MUJOCO_GRASP_PHASE_ORDER:
        phase_results = [result for result in evaluation_result.results if result.phase == phase]
        regressions = [result.metric for result in phase_results if not result.passed]
        if any(
            not result.passed and result.severity == Severity.CRITICAL for result in phase_results
        ):
            status = "fail"
        elif any(not result.passed for result in phase_results):
            status = "degraded"
        else:
            status = "ok"
        metrics = report["snapshot_metrics"][phase]
        statuses.append(
            PhaseStatus(
                phase_id=phase,
                phase_label=MUJOCO_GRASP_PHASE_LABELS[phase],
                status=status,
                step=round(float(metrics["step"])),
                sim_time_s=float(metrics["sim_time_s"]),
                regressions=regressions,
            )
        )
    return statuses


def _resolve_failed_phase_id(
    phase_statuses: list[PhaseStatus],
    evaluation_result: EvaluationResult,
) -> str | None:
    for status in phase_statuses:
        if status.status != "ok":
            return status.phase_id
    if evaluation_result.failed:
        return MUJOCO_GRASP_PHASE_ORDER[-1]
    return None


def _build_rerun_hint(failed_phase_id: str | None) -> str:
    restore_map = {
        None: "restore:plan",
        "plan": "restore:plan",
        "pre_grasp": "restore:plan",
        "approach": "restore:pre_grasp",
        "grasp": "restore:approach",
        "lift": "restore:grasp",
    }
    return restore_map[failed_phase_id]


def _build_agent_next_action(
    *,
    failed_phase_id: str | None,
    failed_phase: str | None,
    primary_views: list[str],
    primary_failure: Any,
) -> str:
    if failed_phase_id is None or failed_phase is None:
        return (
            "No rerun required. The canonical primary views match baseline; inspect the "
            "final lift captures only if you are chasing a visual false negative."
        )
    view = primary_views[0]
    metric = primary_failure.metric if primary_failure is not None else "phase regression"
    return (
        f"The first divergence is {failed_phase} (checkpoint '{failed_phase_id}'). "
        f"Inspect {failed_phase_id}/{view}_rgb.png first, then re-tune the code path that drives "
        f"{metric} before touching later phases."
    )


def _build_failure_taxonomy(
    evaluation_result: EvaluationResult,
) -> tuple[list[dict[str, str]], list[str]]:
    taxonomy: list[dict[str, str]] = []
    verdict_reasons: list[str] = []
    for result in evaluation_result.failed:
        code = ROOT_CAUSE_BY_METRIC.get(result.metric, "trajectory_regression")
        if code in verdict_reasons:
            continue
        verdict_reasons.append(code)
        taxonomy.append(
            {
                "category": ROOT_CAUSE_CATEGORY[code],
                "code": code,
                "detail": result.metric,
            }
        )
    return taxonomy, verdict_reasons


def _metric_title(metric: str) -> str:
    titles = {
        "loop_runtime_s": "Loop Runtime",
        "loop_runtime_s_abs_delta": "Runtime Drift",
        "max_abs_qvel": "Velocity Stability",
        "grip_center_error_mm_abs_delta": "Approach Drift",
        "gripper_skew_deg_abs_delta": "Approach Skew",
        "pinch_gap_error_mm_abs_delta": "Contact Gap Drift",
        "pinch_elevation_deg_abs_delta": "Contact Tilt Drift",
        "contact_count": "Lift Contact",
        "cube_height_mm": "Lift Height",
        "cube_height_mm_abs_delta": "Lift Height Drift",
    }
    return titles.get(metric, metric.replace("_", " "))


def _format_expected(operator: str, threshold: float | tuple[float, float]) -> str:
    if isinstance(threshold, tuple):
        return f"{operator} {threshold}"
    if operator == Operator.LT.value:
        return f"< {threshold}"
    if operator == Operator.LE.value:
        return f"<= {threshold}"
    if operator == Operator.GT.value:
        return f"> {threshold}"
    if operator == Operator.GE.value:
        return f">= {threshold}"
    return f"{operator} {threshold}"


def _lookup_baseline_value(report: dict[str, Any], phase: str, metric: str) -> float | None:
    if phase == "*":
        baseline_key = f"{metric}_baseline"
        value = report["summary_metrics"].get(baseline_key)
    else:
        baseline_key = f"{metric}_baseline"
        value = report["snapshot_metrics"][phase].get(baseline_key)
    if value is None:
        return None
    return float(value)


def _render_alarm_card(alarm: AlarmRecord) -> str:
    css_class = "alarm-card ok" if alarm.status == "ok" else "alarm-card"
    actual_value = (
        f"{alarm.actual_value:.2f}" if isinstance(alarm.actual_value, (int, float)) else "missing"
    )
    baseline = (
        f"Baseline {alarm.baseline_value:.2f}"
        if isinstance(alarm.baseline_value, (int, float))
        else "No baseline"
    )
    return (
        f'<article class="{css_class}">'
        f"<small>{html.escape(alarm.phase_label)} · {html.escape(alarm.severity)}</small>"
        f"<strong>{html.escape(alarm.title)}</strong>"
        f"<p>Actual {html.escape(actual_value)} | Expected {html.escape(alarm.expected)}</p>"
        f"<p>{html.escape(baseline)}</p>"
        "</article>"
    )


def _render_phase_card(report: dict[str, Any], status: PhaseStatus) -> str:
    metrics = report["snapshot_metrics"][status.phase_id]
    css_class = f"phase-card phase-card-{status.status}"
    regressions = ", ".join(status.regressions) if status.regressions else "No alarms"
    title = status.phase_label
    if status.phase_label != status.phase_id:
        title = f"{status.phase_label} ({status.phase_id})"
    return (
        f'<article class="{css_class}">'
        f"<h4>{html.escape(title)}</h4>"
        f"<p>step {status.step} · t={status.sim_time_s:.3f}s</p>"
        f"<p>grip err {metrics['grip_center_error_mm']:.2f} mm · "
        f"contacts {metrics['contact_count']:.0f}</p>"
        f"<p>{html.escape(regressions)}</p>"
        "</article>"
    )


def _resolve_visual_image(root: Path, phase_id: str, view_name: str) -> Path | None:
    root_resolved = root.resolve()
    candidate = (root / phase_id / f"{view_name}_rgb.png").resolve()
    try:
        candidate.relative_to(root_resolved)
    except ValueError:
        return None
    return candidate


def _build_metric_explanations(report: dict[str, Any], phase_id: str) -> list[MetricExplanation]:
    phase_candidates = _collect_assertion_metric_explanations(
        report,
        phase=phase_id,
        metrics=report["snapshot_metrics"][phase_id],
        baseline_metrics=report["baseline_snapshot_metrics"][phase_id],
    )
    if phase_candidates:
        return phase_candidates

    summary_candidates = _collect_assertion_metric_explanations(
        report,
        phase="*",
        metrics=report["summary_metrics"],
        baseline_metrics=report.get("baseline_summary_metrics", {}),
    )
    if summary_candidates:
        return summary_candidates

    return _collect_fallback_metric_explanations(report, phase_id)


def _collect_assertion_metric_explanations(
    report: dict[str, Any],
    *,
    phase: str,
    metrics: dict[str, Any],
    baseline_metrics: dict[str, Any],
) -> list[MetricExplanation]:
    ranked: list[tuple[tuple[int, float, str], MetricExplanation]] = []
    for assertion in MUJOCO_GRASP_ASSERTIONS:
        if assertion.phase != phase:
            continue
        raw_actual = metrics.get(assertion.metric)
        if raw_actual is None:
            continue
        actual = float(raw_actual)
        if not _assertion_failed(assertion, actual):
            continue
        base_metric = _base_metric_name(assertion.metric)
        current_value = float(metrics.get(base_metric, actual))
        baseline_value = float(
            metrics.get(f"{base_metric}_baseline", baseline_metrics.get(base_metric, current_value))
        )
        delta = current_value - baseline_value
        explanation = MetricExplanation(
            metric=base_metric,
            copy=(
                f"{base_metric}: {_format_metric_number(baseline_value)} -> "
                f"{_format_metric_number(current_value)} "
                f"({_format_signed_metric_number(delta)}, threshold "
                f"{_format_threshold_number(assertion.threshold)})"
            ),
        )
        ranked.append(
            (
                (
                    _severity_rank(assertion.severity),
                    -abs(delta),
                    base_metric,
                ),
                explanation,
            )
        )
    ranked.sort(key=lambda item: item[0])
    return [item[1] for item in ranked[:2]]


def _collect_fallback_metric_explanations(
    report: dict[str, Any],
    phase_id: str,
) -> list[MetricExplanation]:
    metrics = report["snapshot_metrics"][phase_id]
    baseline_metrics = report["baseline_snapshot_metrics"][phase_id]
    ranked: list[tuple[tuple[float, str], MetricExplanation]] = []
    for metric in PHASE_COMPARISON_METRICS:
        current_raw = metrics.get(metric)
        baseline_raw = metrics.get(f"{metric}_baseline", baseline_metrics.get(metric))
        if current_raw is None or baseline_raw is None:
            continue
        current_value = float(current_raw)
        baseline_value = float(baseline_raw)
        delta = current_value - baseline_value
        if abs(delta) < 1e-6:
            continue
        threshold = _lookup_threshold(metric, phase_id)
        threshold_text = _format_threshold_number(threshold) if threshold is not None else "n/a"
        ranked.append(
            (
                (-abs(delta), metric),
                MetricExplanation(
                    metric=metric,
                    copy=(
                        f"{metric}: {_format_metric_number(baseline_value)} -> "
                        f"{_format_metric_number(current_value)} "
                        f"({_format_signed_metric_number(delta)}, threshold {threshold_text})"
                    ),
                ),
            )
        )
    ranked.sort(key=lambda item: item[0])
    return [item[1] for item in ranked[:2]]


def _assertion_failed(assertion: MetricAssertion, actual: float) -> bool:
    threshold = assertion.threshold
    if assertion.operator == Operator.LT:
        return actual >= float(threshold)
    if assertion.operator == Operator.LE:
        return actual > float(threshold)
    if assertion.operator == Operator.EQ:
        return actual != float(threshold)
    if assertion.operator == Operator.GT:
        return actual <= float(threshold)
    if assertion.operator == Operator.GE:
        return actual < float(threshold)
    if assertion.operator == Operator.IN_RANGE:
        low, high = threshold  # type: ignore[misc]
        return not (float(low) <= actual <= float(high))
    raise ValueError(f"Unknown operator: {assertion.operator}")


def _severity_rank(severity: Severity) -> int:
    ranks = {
        Severity.CRITICAL: 0,
        Severity.MAJOR: 1,
        Severity.MINOR: 2,
        Severity.INFO: 3,
    }
    return ranks[severity]


def _base_metric_name(metric: str) -> str:
    if metric.endswith("_abs_delta"):
        return metric[: -len("_abs_delta")]
    return metric


def _format_metric_number(value: float) -> str:
    return f"{value:.1f}"


def _format_signed_metric_number(value: float) -> str:
    return f"{value:+.1f}"


def _format_threshold_number(threshold: float | tuple[float, float] | None) -> str:
    if threshold is None:
        return "n/a"
    if isinstance(threshold, tuple):
        return (
            f"[{_format_metric_number(float(threshold[0]))}, "
            f"{_format_metric_number(float(threshold[1]))}]"
        )
    return _format_metric_number(float(threshold))


def _lookup_threshold(metric: str, phase_id: str) -> float | tuple[float, float] | None:
    metric_candidates = {metric, f"{metric}_abs_delta"}
    for assertion in MUJOCO_GRASP_ASSERTIONS:
        if assertion.phase not in {phase_id, "*"}:
            continue
        if assertion.metric in metric_candidates:
            return assertion.threshold
    return None


def _metric_set_is_ambiguous(metric_explanations: list[MetricExplanation]) -> bool:
    if not metric_explanations:
        return False
    return all(
        explanation.metric in AMBIGUOUS_STILL_IMAGE_METRICS for explanation in metric_explanations
    )


def _build_interpretation_caption(
    *,
    phase_id: str,
    phase_label: str,
    view_name: str,
    status: str,
) -> str:
    if status == "ambiguous":
        return (
            f"{phase_label} / {view_name} hints at the first divergence, but the still frame "
            "alone is not definitive."
        )
    if status == "partial":
        return (
            f"{phase_label} / {view_name} still points at the first divergence, "
            "but one side of the comparison is missing."
        )
    if status == "empty":
        return f"No visual evidence is currently available for {phase_label} / {view_name}."
    return (
        f"{phase_label} / {view_name} is the first manifest-selected proof surface against the "
        "blessed baseline."
    )


def _summary_evidence_state(
    manifest: PhaseManifest,
    evidence_pairs: list[EvidencePair],
) -> str:
    if manifest.failed_phase_id is None:
        return "PASS/no failed phase"
    if not evidence_pairs or all(pair.status == "empty" for pair in evidence_pairs):
        return "FAIL/empty evidence"
    statuses = {pair.status for pair in evidence_pairs}
    if "mismatch" in statuses:
        return "FAIL/manifest mismatch"
    if "ambiguous" in statuses:
        return "FAIL/ambiguous still-image evidence"
    if "partial" in statuses or "empty" in statuses:
        return "FAIL/partial evidence"
    return "FAIL/full evidence"


def _summary_evidence_message(
    manifest: PhaseManifest,
    evidence_pairs: list[EvidencePair],
) -> str:
    state = _summary_evidence_state(manifest, evidence_pairs)
    diagnostics = _collect_diagnostic_messages(evidence_pairs)
    if state == "PASS/no failed phase":
        return "No visual regression detected for the canonical primary views."
    if state == "FAIL/manifest mismatch" and diagnostics:
        return diagnostics[0]
    if state == "FAIL/empty evidence":
        return (
            "No visual evidence available for the failing phase. "
            f"Re-run from {manifest.rerun_hint} to rebuild evidence."
        )
    if state == "FAIL/ambiguous still-image evidence":
        if diagnostics:
            return diagnostics[0]
        return (
            "Still-image evidence is suggestive, but temporal proof is weak. "
            f"Re-run from {manifest.rerun_hint} if motion timing matters."
        )
    if state == "FAIL/partial evidence" and diagnostics:
        return diagnostics[0]
    if manifest.failed_phase_id is None:
        return "No visual regression detected for the canonical primary views."
    selected_views = ", ".join(manifest.primary_views[:2]) if manifest.primary_views else "none"
    return (
        f"Showing {manifest.failed_phase_id} in manifest-selected order ({selected_views}) for the "
        "first failing phase."
    )


def _collect_diagnostic_messages(evidence_pairs: list[EvidencePair]) -> list[str]:
    diagnostics: list[str] = []
    for pair in evidence_pairs:
        if pair.diagnostic_message and pair.diagnostic_message not in diagnostics:
            diagnostics.append(pair.diagnostic_message)
    return diagnostics


def _render_evidence_section(
    manifest: PhaseManifest,
    evidence_pairs: list[EvidencePair],
) -> str:
    state = _summary_evidence_state(manifest, evidence_pairs)
    message = _summary_evidence_message(manifest, evidence_pairs)
    selected_views = ", ".join(manifest.primary_views[:2]) if manifest.primary_views else "none"

    cards = "".join(
        _render_evidence_card(pair) for pair in evidence_pairs if pair.status != "mismatch"
    )
    mismatch_banners = "".join(
        _render_evidence_banner("mismatch", pair.diagnostic_message or pair.interpretation_caption)
        for pair in evidence_pairs
        if pair.status == "mismatch"
    )
    body = cards or ""
    if manifest.failed_phase_id is None:
        body = ""

    return (
        '<section class="evidence-section">'
        '<div class="evidence-section-head">'
        "<div>"
        "<h3>Current vs Baseline</h3>"
        f"<p>First failing phase: <code>{html.escape(manifest.failed_phase_id or 'none')}</code> · "
        f"views: <code>{html.escape(selected_views)}</code></p>"
        "</div>"
        f'<p class="evidence-state-label">{html.escape(state)}</p>'
        "</div>"
        f"{_render_evidence_banner(_banner_variant_for_state(state), message)}"
        f"{mismatch_banners}"
        f'<div class="evidence-grid">{body}</div>'
        "</section>"
    )


def _banner_variant_for_state(state: str) -> str:
    if state.startswith("PASS"):
        return "pass"
    if state.endswith("manifest mismatch"):
        return "mismatch"
    if state.endswith("partial evidence"):
        return "partial"
    if state.endswith("empty evidence"):
        return "empty"
    if state.endswith("ambiguous still-image evidence"):
        return "ambiguous"
    return "full"


def _render_evidence_banner(variant: str, message: str) -> str:
    return (
        f'<div class="evidence-banner evidence-banner-{html.escape(variant)}">'
        f"<p>{html.escape(message)}</p>"
        "</div>"
    )


def _render_evidence_card(pair: EvidencePair) -> str:
    chips = "".join(_render_metric_chip(explanation) for explanation in pair.metric_explanations)
    diagnostic = (
        f'<p class="evidence-diagnostic">{html.escape(pair.diagnostic_message)}</p>'
        if pair.diagnostic_message
        else ""
    )
    return (
        f'<article class="evidence-card evidence-card-{html.escape(pair.status)}">'
        '<div class="evidence-card-head">'
        f'<span class="evidence-badge">Phase {html.escape(pair.phase_label)}</span>'
        f'<span class="evidence-badge">View {html.escape(pair.view_name)}</span>'
        f'<span class="evidence-badge evidence-badge-status">{html.escape(pair.status)}</span>'
        "</div>"
        '<div class="evidence-compare-grid">'
        f"{_render_evidence_media(pair, role='Current')}"
        f"{_render_evidence_media(pair, role='Baseline')}"
        "</div>"
        f'<div class="evidence-chip-row">{chips}</div>'
        f'<p class="evidence-caption">{html.escape(pair.interpretation_caption)}</p>'
        f"{diagnostic}"
        "</article>"
    )


def _render_metric_chip(explanation: MetricExplanation) -> str:
    return f'<span class="metric-chip">{html.escape(explanation.copy)}</span>'


def _render_evidence_media(pair: EvidencePair, *, role: str) -> str:
    if role == "Current":
        image_path = pair.current_image_path
        fallback = _missing_role_message(pair, role)
    else:
        image_path = pair.baseline_image_path
        fallback = _missing_role_message(pair, role)

    label = f"{role}"
    alt = f"{pair.phase_label} ({pair.phase_id}) {pair.view_name} {role.lower()} evidence"
    if image_path is not None and image_path.exists():
        return (
            '<figure class="evidence-figure">'
            f'<div class="evidence-role">{html.escape(label)}</div>'
            f'<img src="{_image_data_uri(image_path)}" '
            f'alt="{html.escape(alt)}" loading="lazy"/>'
            f"<figcaption>{html.escape(role)} {html.escape(pair.phase_label)} / "
            f"{html.escape(pair.view_name)}</figcaption>"
            "</figure>"
        )
    return (
        '<figure class="evidence-figure">'
        f'<div class="evidence-role">{html.escape(label)}</div>'
        f'<div class="evidence-placeholder" role="img" aria-label="{html.escape(fallback)}">'
        f"{html.escape(role)}"
        "</div>"
        f"<figcaption>{html.escape(fallback)}</figcaption>"
        "</figure>"
    )


def _missing_role_message(pair: EvidencePair, role: str) -> str:
    if pair.diagnostic_message:
        if role == "Current" and "Current capture missing" in pair.diagnostic_message:
            return pair.diagnostic_message
        if role == "Baseline" and "Baseline image missing" in pair.diagnostic_message:
            return pair.diagnostic_message
        if pair.status in {"empty", "mismatch"}:
            return pair.diagnostic_message
    return f"{role} image missing for {pair.phase_id}/{pair.view_name}."


def _image_data_uri(path: Path) -> str:
    mime = "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"
