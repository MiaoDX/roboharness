"""Alarmed-grasp wedge helpers kept local to the MuJoCo example."""

from __future__ import annotations

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
BASELINE_REPORT_PATH = (
    Path(__file__).resolve().parents[1]
    / "assets"
    / "example_mujoco_grasp"
    / "baseline_autonomous_report.json"
)
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
    root_cause = (
        ROOT_CAUSE_BY_METRIC.get(primary_failure.metric, "trajectory_regression")
        if primary_failure is not None
        else "trajectory_regression"
    )
    primary_views = PRIMARY_VIEWS_BY_ROOT_CAUSE.get(
        root_cause,
        MUJOCO_GRASP_PRIMARY_VIEWS.get(failed_phase_id or "lift", ["front"]),
    )
    regressions = [alarm.metric for alarm in alarms if alarm.status == "alarm"]
    rerun_hint = _build_rerun_hint(failed_phase_id)
    evidence_paths = (
        [f"{failed_phase_id}/{view}_rgb.png" for view in primary_views]
        if failed_phase_id is not None
        else [f"lift/{primary_views[0]}_rgb.png"]
    )
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
) -> str:
    """Build the alarm-first HTML summary block for the shared report renderer."""
    visible_alarms = alarms[:4]
    alarm_cards = "".join(_render_alarm_card(alarm) for alarm in visible_alarms)

    phase_cards = "".join(_render_phase_card(report, status) for status in manifest.phase_statuses)
    baseline_name = html.escape(Path(str(report["baseline_source"])).name)
    failed_phase_text = html.escape(manifest.failed_phase or "none")
    regressions = ", ".join(manifest.regressions) if manifest.regressions else "none"

    return (
        '<div class="alarm-grid">'
        f"{alarm_cards}"
        "</div>"
        '<div class="agent-panel">'
        "<strong>Agent Next Action</strong>"
        f"<p>{html.escape(manifest.agent_next_action)}</p>"
        "</div>"
        '<div class="report-grid">'
        '<div class="summary-card">'
        "<h3>Artifact Pack</h3>"
        '<table class="meta-table">'
        f"<tr><th>Baseline</th><td><code>{baseline_name}</code></td></tr>"
        f"<tr><th>Verdict</th><td><strong>{html.escape(manifest.verdict.upper())}</strong></td></tr>"
        f"<tr><th>Failed phase</th><td>{failed_phase_text}</td></tr>"
        "<tr><th>Root cause</th><td><code>"
        f"{html.escape(manifest.suspected_root_cause)}</code></td></tr>"
        f"<tr><th>Rerun hint</th><td><code>{html.escape(manifest.rerun_hint)}</code></td></tr>"
        f"<tr><th>Regressions</th><td><code>{html.escape(regressions)}</code></td></tr>"
        "</table>"
        "</div>"
        '<div class="summary-card">'
        "<h3>Phase Timeline</h3>"
        f'<div class="phase-timeline">{phase_cards}</div>'
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
        return "Inspect the final lift captures first, then compare loop metrics against baseline."
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
