"""Alarmed-grasp wedge helpers kept local to the MuJoCo example."""

from __future__ import annotations

import html
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from roboharness._utils import save_json
from roboharness.approval.evidence import (
    EvidencePair,
    EvidenceTarget,
    MetricExplanation,
)
from roboharness.approval.evidence import (
    render_lightbox_shell as _render_lightbox_shell,
)
from roboharness.approval.evidence import (
    render_zoomable_image as _render_zoomable_image,
)
from roboharness.approval.evidence import (
    resolve_evidence_pairs as resolve_shared_evidence_pairs,
)
from roboharness.evaluate.assertions import AssertionEngine, MetricAssertion
from roboharness.evaluate.result import EvaluationResult, Operator, Severity

try:
    from examples.demos.mujoco.fixture import (
        MUJOCO_GRASP_CAMERAS,
        MUJOCO_GRASP_PHASE_LABELS,
        MUJOCO_GRASP_PHASE_ORDER,
        MUJOCO_GRASP_PRIMARY_VIEWS,
        MUJOCO_GRASP_TASK,
    )
except ModuleNotFoundError:  # pragma: no cover - script execution path
    from fixture import (  # type: ignore[no-redef]
        MUJOCO_GRASP_CAMERAS,
        MUJOCO_GRASP_PHASE_LABELS,
        MUJOCO_GRASP_PHASE_ORDER,
        MUJOCO_GRASP_PRIMARY_VIEWS,
        MUJOCO_GRASP_TASK,
    )

TABLE_SURFACE_Z = 0.22
FINGER_BODY_WIDTH_M = 0.024
CUBE_WIDTH_M = 0.05
CANONICAL_REPORT_NAME = "report.html"
ASSET_ROOT = Path(__file__).resolve().parents[3] / "assets" / "example_mujoco_grasp"
BASELINE_REPORT_PATH = ASSET_ROOT / "baseline_autonomous_report.json"
BASELINE_VISUAL_ROOT = ASSET_ROOT / "baseline_visual"
KNOWN_BAD_VISUAL_ROOT = ASSET_ROOT / "known_bad_visual"
CONTRACT_SCHEMA_VERSION = "roboharness_contract/v1"
APPROVAL_REPORT_SCHEMA_VERSION = "roboharness_report/v1"
DEFAULT_CONTRACT_PRESET = "mujoco_regression_v1"
MIGRATION_CONTRACT_PRESET = "mujoco_migration_guarded_v1"
SUPPORTED_CONTRACT_PRESETS = (
    DEFAULT_CONTRACT_PRESET,
    MIGRATION_CONTRACT_PRESET,
)
CONTRACT_DOCS_URL = "docs/designs/unattended-refactor-harness-v1.md"
REGRESSION_PROMPT_MARKERS = (
    "regression",
    "same behavior",
    "no behavior change",
    "keep baseline",
    "keep aligned",
    "unchanged",
)
MIGRATION_PROMPT_MARKERS = (
    "migration",
    "intended change",
    "manual blessing",
    "requires review",
    "review against the old baseline",
    "bless",
)
UNSUPPORTED_PROMPT_MARKERS = (
    "visual_goal",
    "anti_goal",
    "anti-goal",
    "top-down",
    "top down",
    "ball grasp",
    "palm-down",
    "palm down",
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
SUPPORTED_CONTRACT_PHASES = frozenset(["all", *MUJOCO_GRASP_PHASE_ORDER])
_DEFAULT_ASSERTION_BY_CONTRACT_KEY = {
    ("all" if assertion.phase == "*" else assertion.phase, assertion.metric): assertion
    for assertion in MUJOCO_GRASP_ASSERTIONS
}


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
class ProofPack:
    """Complete MuJoCo trust-loop proof pack assembled from one evaluator run."""

    contract: dict[str, Any]
    report: dict[str, Any]
    evaluation_result: EvaluationResult
    alarms: list[AlarmRecord]
    manifest: PhaseManifest
    evidence_pairs: list[EvidencePair]
    approval_report: dict[str, Any]


class EvidenceState(str, Enum):
    """Canonical evidence state labels emitted by the MuJoCo approval report."""

    PASS_NO_FAILED_PHASE = "PASS/no failed phase"
    EMPTY = "FAIL/empty evidence"
    MANIFEST_MISMATCH = "FAIL/manifest mismatch"
    AMBIGUOUS_STILL_IMAGE = "FAIL/ambiguous still-image evidence"
    PARTIAL = "FAIL/partial evidence"
    FULL = "FAIL/full evidence"

    @property
    def label(self) -> str:
        return self.value

    @property
    def banner_variant(self) -> str:
        if self is EvidenceState.PASS_NO_FAILED_PHASE:
            return "pass"
        if self is EvidenceState.MANIFEST_MISMATCH:
            return "mismatch"
        if self is EvidenceState.PARTIAL:
            return "partial"
        if self is EvidenceState.EMPTY:
            return "empty"
        if self is EvidenceState.AMBIGUOUS_STILL_IMAGE:
            return "ambiguous"
        return "full"


@dataclass(frozen=True)
class EvidenceSummary:
    """Evidence state and reviewer-facing copy derived from one manifest."""

    state: EvidenceState
    message: str

    @property
    def label(self) -> str:
        return self.state.label

    @property
    def banner_variant(self) -> str:
        return self.state.banner_variant


@dataclass(frozen=True)
class ApprovalDecision:
    """Central approval semantics for one MuJoCo proof pack."""

    evidence: EvidenceSummary
    overall_verdict: str
    surfaces_case: bool
    surfaced_case_status: str
    material_reasons: tuple[str, ...]
    stop_reason: str
    run_state: str
    run_state_title: str
    run_state_message: str
    needs_baseline_blessing: bool
    reruns: int


@dataclass(frozen=True)
class GroundedContractRule:
    """Validated contract rule tied to one evaluator assertion template."""

    rule_id: str
    phase_id: str
    metric: str
    operator: Operator
    threshold: float | tuple[float, float]
    template: MetricAssertion

    def to_assertion(self) -> MetricAssertion:
        return MetricAssertion(
            metric=self.template.metric,
            operator=self.operator,
            threshold=self.threshold,
            severity=self.template.severity,
            phase=self.template.phase,
        )


@dataclass(frozen=True)
class ErrorEnvelope:
    """User-facing error contract shared across contract and report failures."""

    problem: str
    cause: str
    fix: str
    docs_url: str
    recoverable: bool
    next_action: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "problem": self.problem,
            "cause": self.cause,
            "fix": self.fix,
            "docs_url": self.docs_url,
            "recoverable": self.recoverable,
            "next_action": self.next_action,
        }


class ContractCompileError(ValueError):
    """Raised when a wedge contract cannot be grounded safely."""

    def __init__(self, envelope: ErrorEnvelope):
        super().__init__(f"{envelope.problem} {envelope.cause}")
        self.envelope = envelope


def available_contract_presets() -> tuple[str, ...]:
    """Return the grounded preset names supported by the MuJoCo wedge."""
    return SUPPORTED_CONTRACT_PRESETS


def build_default_contract(*, baseline_source: str) -> dict[str, Any]:
    """Build the reviewed regression contract for the deterministic MuJoCo wedge."""
    return build_contract_from_preset(
        baseline_source=baseline_source,
        contract_preset=DEFAULT_CONTRACT_PRESET,
    )


def build_contract_from_preset(
    *,
    baseline_source: str,
    contract_preset: str,
    source_prompt: str | None = None,
) -> dict[str, Any]:
    """Build a grounded contract from one of the reviewed MuJoCo presets."""
    if contract_preset not in SUPPORTED_CONTRACT_PRESETS:
        raise _contract_error(
            cause=(
                f"unsupported contract preset {contract_preset!r}. "
                f"Supported presets: {list(SUPPORTED_CONTRACT_PRESETS)!r}."
            ),
            fix="Choose one of the reviewed preset names for this wedge.",
        )
    if contract_preset == MIGRATION_CONTRACT_PRESET:
        contract_id = "mujoco-grasp-migration-guarded-v1"
        mode = "migration"
        default_prompt = (
            "Treat this MuJoCo wedge run as an intended change, keep the evaluator "
            "metric gates authoritative, and require explicit human review before "
            "blessing any new baseline."
        )
    else:
        contract_id = "mujoco-grasp-regression-v1"
        mode = "regression"
        default_prompt = (
            "Keep the deterministic MuJoCo grasp loop aligned with the blessed baseline and "
            "surface only materially changed cases for review."
        )
    return {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "contract_id": contract_id,
        "contract_preset": contract_preset,
        "mode": mode,
        "source_prompt": source_prompt or default_prompt,
        "cases": {
            "source": "deterministic_mujoco_grasp",
            "immutable": True,
        },
        "baseline_source": str(baseline_source),
        "compile_policy": {
            "on_ambiguity": "fail_closed",
            "require_grounded_rules": True,
        },
        "runtime_policy": {
            "on_ambiguous_verdict": "gather_more_evidence_but_never_self_pass",
            "soft_stop": "agent_may_stop_when_goal_unreachable",
            "hard_stop": {
                "repeat_failure_signature_limit": 2,
                "max_reruns": 12,
            },
            "failure_signature": [
                "case_id",
                "phase_id",
                "violated_rule_id",
            ],
        },
        "approval_policy": {
            "surface_changed_cases_only": True,
            "show_unchanged_case_count": True,
            "require_user_blessing_for_new_baseline": True,
        },
        "rules": [_build_contract_rule(assertion) for assertion in MUJOCO_GRASP_ASSERTIONS],
    }


def compile_contract_prompt(
    *,
    baseline_source: str,
    contract_prompt: str,
) -> dict[str, Any]:
    """Compile a constrained natural-language prompt into a reviewed preset."""
    normalized_prompt = " ".join(contract_prompt.lower().split())
    if any(marker in normalized_prompt for marker in UNSUPPORTED_PROMPT_MARKERS):
        raise _contract_error(
            cause=(
                "contract prompt asks for visual or open-ended rule authoring that this wedge "
                "does not ground safely."
            ),
            fix=(
                "Use --contract-preset for reviewed preset selection, or pass "
                "--contract-json with explicit metric_gate rules."
            ),
        )

    is_regression = any(marker in normalized_prompt for marker in REGRESSION_PROMPT_MARKERS)
    is_migration = any(marker in normalized_prompt for marker in MIGRATION_PROMPT_MARKERS)
    if is_regression and is_migration:
        raise _contract_error(
            cause="contract prompt mixes regression and migration intent in one request.",
            fix="Choose one approval mode per run, or use --contract-preset explicitly.",
        )
    if is_migration:
        preset = MIGRATION_CONTRACT_PRESET
    elif is_regression:
        preset = DEFAULT_CONTRACT_PRESET
    else:
        raise _contract_error(
            cause=(
                "contract prompt could not be grounded to a reviewed preset. "
                "This wedge only supports prompt-assisted preset selection today."
            ),
            fix=(
                "Use prompt text that clearly says regression or migration intent, or pass "
                "--contract-preset / --contract-json."
            ),
        )

    return build_contract_from_preset(
        baseline_source=baseline_source,
        contract_preset=preset,
        source_prompt=contract_prompt,
    )


def compile_contract(
    *,
    baseline_source: str,
    contract_path: str | Path | None = None,
    contract_preset: str = DEFAULT_CONTRACT_PRESET,
    contract_prompt: str | None = None,
) -> dict[str, Any]:
    """Load or build the contract and fail closed if it cannot be grounded safely."""
    if contract_path is None:
        if contract_prompt is not None:
            contract = compile_contract_prompt(
                baseline_source=baseline_source,
                contract_prompt=contract_prompt,
            )
        else:
            contract = build_contract_from_preset(
                baseline_source=baseline_source,
                contract_preset=contract_preset,
            )
    else:
        try:
            with Path(contract_path).open() as fh:
                contract = json.load(fh)
        except FileNotFoundError as exc:
            raise _contract_error(
                cause=f"contract file not found: {contract_path}",
                fix="Pass an existing JSON file or omit --contract-json to use the default preset.",
            ) from exc
        except json.JSONDecodeError as exc:
            raise _contract_error(
                cause=f"contract JSON is invalid: {exc}",
                fix="Fix the JSON syntax or omit --contract-json to use the default preset.",
            ) from exc

    validate_contract(contract)
    return contract


def validate_contract(contract: dict[str, Any]) -> None:
    """Validate that the contract can be grounded by the current MuJoCo wedge."""
    _ground_contract_rules(contract)


def contract_assertions(contract: dict[str, Any]) -> tuple[MetricAssertion, ...]:
    """Compile the validated contract into the assertion set used by the evaluator."""
    return tuple(rule.to_assertion() for rule in _ground_contract_rules(contract))


def build_approval_report(
    *,
    contract: dict[str, Any],
    report: dict[str, Any],
    evaluation_result: EvaluationResult,
    manifest: PhaseManifest,
    evidence_pairs: list[EvidencePair],
) -> dict[str, Any]:
    """Build the single-case approval artifact returned by the MuJoCo wedge."""
    case_id = str(report.get("case_id", "deterministic_mujoco_grasp"))
    evidence = _build_evidence_summary(manifest, evidence_pairs)
    decision = _build_approval_decision(
        mode=contract["mode"],
        evaluation_result=evaluation_result,
        evidence=evidence,
        has_ungrounded_rules=bool(_ungrounded_rule_ids(contract, evaluation_result)),
    )
    surfaced_cases: list[dict[str, Any]] = []
    suppressed_cases: list[dict[str, Any]] = []

    if decision.surfaces_case:
        surfaced_cases.append(
            _build_surfaced_case(
                contract=contract,
                report=report,
                evaluation_result=evaluation_result,
                manifest=manifest,
                evidence_pairs=evidence_pairs,
                case_id=case_id,
                decision=decision,
            )
        )
    else:
        suppressed_cases.append(
            {
                "case_id": case_id,
                "status": "UNCHANGED",
                "reason": "no_material_change",
            }
        )

    return {
        "schema_version": APPROVAL_REPORT_SCHEMA_VERSION,
        "contract_id": contract["contract_id"],
        "mode": contract["mode"],
        "overall_verdict": decision.overall_verdict,
        "stop_reason": decision.stop_reason,
        "run_state": decision.run_state,
        "run_state_title": decision.run_state_title,
        "run_state_message": decision.run_state_message,
        "baseline_authority": _baseline_authority_copy(contract["mode"]),
        "summary": {
            "cases_total": 1,
            "cases_surfaced": len(surfaced_cases),
            "cases_suppressed": len(suppressed_cases),
            "cases_unchanged": len(suppressed_cases),
            "reruns": decision.reruns,
        },
        "surfaced_cases": surfaced_cases,
        "suppressed_cases": suppressed_cases,
        "unchanged": {
            "count": len(suppressed_cases),
        },
        "user_action": {
            "needs_review": bool(surfaced_cases),
            "needs_baseline_blessing": decision.needs_baseline_blessing,
            "review_case_ids": [case["case_id"] for case in surfaced_cases],
        },
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
    contract: dict[str, Any] | None = None,
) -> EvaluationResult:
    """Evaluate the MuJoCo wedge report using one shared verdict path."""
    assertions = contract_assertions(contract) if contract is not None else MUJOCO_GRASP_ASSERTIONS
    return AssertionEngine(assertions).evaluate(report, report_path=report_path)


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
    targets: list[EvidenceTarget] = []

    for view_name in manifest.primary_views[:2]:
        expected_rel = f"{failed_phase_id}/{view_name}_rgb.png"
        if view_name not in canonical_views or expected_rel not in manifest.evidence_paths:
            message = (
                "Manifest/view contract mismatch. "
                f"The report requested {failed_phase_id}/{view_name}, but no matching asset "
                "exists under the blessed baseline root."
            )
            targets.append(
                EvidenceTarget(
                    phase_id=failed_phase_id,
                    phase_label=failed_phase,
                    view_name=view_name,
                    current_relative_path=expected_rel,
                    baseline_relative_path=expected_rel,
                    forced_mismatch_message=message,
                )
            )
            continue

        targets.append(
            EvidenceTarget(
                phase_id=failed_phase_id,
                phase_label=failed_phase,
                view_name=view_name,
                current_relative_path=expected_rel,
                baseline_relative_path=expected_rel,
                missing_baseline_message=(
                    f"Baseline image missing for {failed_phase_id}/{view_name}. Rebuild or "
                    "restore the blessed baseline pack."
                ),
                missing_current_message=(
                    f"Current capture missing for {failed_phase_id}/{view_name}. Re-run from "
                    f"{manifest.rerun_hint} to rebuild evidence."
                ),
                empty_message=(
                    "No visual evidence available for the failing phase. "
                    f"Re-run from {manifest.rerun_hint} to rebuild evidence."
                ),
                ambiguous_message=(
                    f"Still-image evidence is suggestive for {failed_phase_id}/{view_name}, "
                    f"but temporal proof is weak. Re-run from {manifest.rerun_hint} if motion "
                    "timing is part of the diagnosis."
                ),
            )
        )

    return resolve_shared_evidence_pairs(
        current_root=trial_dir,
        baseline_root=baseline_visual_root,
        targets=targets,
        metric_explanations=metric_explanations,
        caption_builder=lambda target, status: _build_interpretation_caption(
            phase_id=target.phase_id,
            phase_label=target.phase_label,
            view_name=target.view_name,
            status=status,
        ),
        ambiguity_selector=(
            (lambda _target: True) if _metric_set_is_ambiguous(metric_explanations) else None
        ),
    )


def build_proof_pack(
    *,
    contract: dict[str, Any],
    snapshot_metrics: dict[str, dict[str, float]],
    baseline_report: dict[str, Any],
    baseline_source: str,
    report_path: str = "",
    trial_dir: Path | None = None,
    baseline_visual_root: Path | None = None,
    include_evidence: bool = True,
) -> ProofPack:
    """Build the full MuJoCo approval proof pack in canonical dependency order."""
    report = build_autonomous_report(
        snapshot_metrics=snapshot_metrics,
        baseline_report=baseline_report,
        baseline_source=baseline_source,
    )
    evaluation_result = evaluate_autonomous_report(
        report,
        report_path=report_path,
        contract=contract,
    )
    alarms = build_alarms(report, evaluation_result)
    manifest = build_phase_manifest(report, evaluation_result, alarms)
    evidence_pairs: list[EvidencePair] = []
    if include_evidence and trial_dir is not None and baseline_visual_root is not None:
        evidence_pairs = resolve_evidence_pairs(
            trial_dir=trial_dir,
            baseline_visual_root=baseline_visual_root,
            manifest=manifest,
            report=report,
        )
    approval_report = build_approval_report(
        contract=contract,
        report=report,
        evaluation_result=evaluation_result,
        manifest=manifest,
        evidence_pairs=evidence_pairs,
    )
    return ProofPack(
        contract=contract,
        report=report,
        evaluation_result=evaluation_result,
        alarms=alarms,
        manifest=manifest,
        evidence_pairs=evidence_pairs,
        approval_report=approval_report,
    )


def write_artifact_pack(
    *,
    trial_dir: Path,
    contract: dict[str, Any],
    approval_report: dict[str, Any],
    report: dict[str, Any],
    evaluation_result: EvaluationResult,
    alarms: list[AlarmRecord],
    manifest: PhaseManifest,
    report_generated: bool,
) -> None:
    """Write the machine-readable wedge artifacts into the trial directory."""
    report_with_evaluation = dict(report)
    taxonomy, verdict_reasons = _build_failure_taxonomy(evaluation_result)
    report_with_evaluation["evaluation"] = evaluation_result.to_dict()
    report_with_evaluation["verdict"] = evaluation_result.verdict.value
    report_with_evaluation["verdict_reasons"] = verdict_reasons
    report_with_evaluation["failure_taxonomy"] = taxonomy
    artifacts: dict[str, str] = {
        "contract": "contract.json",
        "autonomous_report": "autonomous_report.json",
        "alarms": "alarms.json",
        "phase_manifest": "phase_manifest.json",
        "approval_report": "approval_report.json",
    }
    if report_generated:
        artifacts["report_html"] = CANONICAL_REPORT_NAME
    report_with_evaluation["artifacts"] = artifacts

    save_json(contract, trial_dir / "contract.json")
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
    save_json(approval_report, trial_dir / "approval_report.json")


def build_summary_html(
    report: dict[str, Any],
    alarms: list[AlarmRecord],
    manifest: PhaseManifest,
    evidence_pairs: list[EvidencePair],
    *,
    contract: dict[str, Any],
    approval_report: dict[str, Any],
) -> str:
    """Build the alarm-first HTML summary block for the shared report renderer."""
    visible_alarms = alarms[:4]
    alarm_cards = "".join(_render_alarm_card(alarm) for alarm in visible_alarms)
    evidence = _build_evidence_summary(manifest, evidence_pairs)
    evidence_section = _render_evidence_section(manifest, evidence_pairs, evidence)
    decision_banner = _render_run_decision_banner(approval_report)
    counts_strip = _render_case_counts(approval_report)
    queue_section = _render_approval_queue(approval_report)
    contract_section = _render_contract_section(contract)
    baseline_section = _render_baseline_section(approval_report)
    phase_cards = "".join(_render_phase_card(report, status) for status in manifest.phase_statuses)
    baseline_name = html.escape(Path(str(report["baseline_source"])).name)
    failed_phase_text = html.escape(manifest.failed_phase or "none")
    regressions = ", ".join(manifest.regressions) if manifest.regressions else "none"
    selected_views = ", ".join(manifest.primary_views[:2]) if manifest.primary_views else "none"
    diagnostics = _collect_diagnostic_messages(evidence_pairs)
    diagnostics_html = "".join(f"<p>{html.escape(message)}</p>" for message in diagnostics)
    root_cause_display = (
        manifest.suspected_root_cause if manifest.failed_phase_id is not None else "none"
    )
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
            f"State: {html.escape(evidence.label)}.</p>"
            f"<p><strong>Rerun hint:</strong> <code>{html.escape(manifest.rerun_hint)}</code></p>"
        )

    return (
        f"{decision_banner}"
        f"{counts_strip}"
        f"{queue_section}"
        '<section class="summary-card metric-results">'
        "<h3>Hard Metric Results</h3>"
        '<div class="alarm-grid">'
        f"{alarm_cards}"
        "</div>"
        "</section>"
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
        f"<tr><th>Evidence state</th><td>{html.escape(evidence.label)}</td></tr>"
        f"<tr><th>Failed phase</th><td>{failed_phase_text}</td></tr>"
        f"<tr><th>Selected views</th><td><code>{html.escape(selected_views)}</code></td></tr>"
        "<tr><th>Root cause</th><td><code>"
        f"{html.escape(root_cause_display)}</code></td></tr>"
        f"{rerun_hint_row}"
        f"<tr><th>Regressions</th><td><code>{html.escape(regressions)}</code></td></tr>"
        "</table></div>"
        "</div>"
        f"{contract_section}"
        f"{baseline_section}"
        "</div>"
        f"{_render_lightbox_shell()}"
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


def _build_contract_rule(assertion: MetricAssertion) -> dict[str, Any]:
    phase_id = "all" if assertion.phase == "*" else assertion.phase
    evidence_at: dict[str, str] = {"phase": phase_id}
    if assertion.phase != "*" and assertion.phase in MUJOCO_GRASP_PRIMARY_VIEWS:
        evidence_at["view"] = MUJOCO_GRASP_PRIMARY_VIEWS[assertion.phase][0]
    return {
        "id": f"{phase_id}:{assertion.metric}",
        "type": "metric_gate",
        "judge": "metric",
        "evidence_at": [evidence_at],
        "pass_if": {
            "metric": assertion.metric,
            "op": assertion.operator.value,
            "value": _serialize_threshold(assertion.threshold),
        },
        "severity": "fail",
    }


def _serialize_threshold(threshold: float | tuple[float, float]) -> float | list[float]:
    if isinstance(threshold, tuple):
        return [float(threshold[0]), float(threshold[1])]
    return float(threshold)


def _contract_error(*, cause: str, fix: str) -> ContractCompileError:
    return ContractCompileError(
        ErrorEnvelope(
            problem="Contract blocked.",
            cause=cause,
            fix=fix,
            docs_url=CONTRACT_DOCS_URL,
            recoverable=True,
            next_action="Fix contract",
        )
    )


def _approval_overall_verdict(
    evaluation_result: EvaluationResult,
    evidence_state: EvidenceState,
    *,
    has_ungrounded_rules: bool,
) -> str:
    if has_ungrounded_rules:
        return "CONTRACT_INVALID"
    if evidence_state is EvidenceState.AMBIGUOUS_STILL_IMAGE:
        return "AMBIGUOUS"
    if evaluation_result.verdict.value == "pass":
        return "PASS"
    return "FAIL"


def _build_approval_decision(
    *,
    mode: str,
    evaluation_result: EvaluationResult,
    evidence: EvidenceSummary,
    has_ungrounded_rules: bool,
) -> ApprovalDecision:
    overall_verdict = _approval_overall_verdict(
        evaluation_result,
        evidence.state,
        has_ungrounded_rules=has_ungrounded_rules,
    )
    surfaces_case = overall_verdict != "PASS" or mode == "migration"
    run_state, run_title, run_message = _run_fields_for_decision(
        overall_verdict=overall_verdict,
        evidence_state=evidence.state,
        mode=mode,
        surfaces_case=surfaces_case,
    )
    return ApprovalDecision(
        evidence=evidence,
        overall_verdict=overall_verdict,
        surfaces_case=surfaces_case,
        surfaced_case_status=_surfaced_case_status_for_decision(mode, overall_verdict),
        material_reasons=_material_reasons_for_decision(mode, overall_verdict, evidence.state),
        stop_reason=_stop_reason_for_decision(
            mode=mode,
            overall_verdict=overall_verdict,
            evidence_state=evidence.state,
            surfaces_case=surfaces_case,
        ),
        run_state=run_state,
        run_state_title=run_title,
        run_state_message=run_message,
        needs_baseline_blessing=mode == "migration" and overall_verdict == "PASS" and surfaces_case,
        reruns=0 if overall_verdict == "PASS" else 1,
    )


def _run_fields_for_decision(
    *,
    overall_verdict: str,
    evidence_state: EvidenceState,
    mode: str,
    surfaces_case: bool,
) -> tuple[str, str, str]:
    if overall_verdict == "CONTRACT_INVALID":
        return (
            "contract_invalid",
            "Contract invalid",
            (
                "The compiled contract does not match the current MuJoCo wedge. "
                "Do not trust or bless this run."
            ),
        )
    if overall_verdict == "PASS" and surfaces_case and mode == "migration":
        return (
            "review_ready_migration",
            "Review intended change",
            (
                "Contract rules passed, but migration runs still require review "
                "against the old baseline before blessing a new one."
            ),
        )
    if overall_verdict == "PASS" and not surfaces_case:
        return (
            "review_ready_success",
            "No surfaced cases",
            "No material changes surfaced. Old baseline remains authoritative.",
        )
    if evidence_state in {EvidenceState.PARTIAL, EvidenceState.EMPTY}:
        return (
            "partial_reviewable",
            "Partial reviewable",
            "Some review is possible, but one side of the proof is missing. Review cautiously.",
        )
    if evidence_state in {EvidenceState.MANIFEST_MISMATCH, EvidenceState.AMBIGUOUS_STILL_IMAGE}:
        return (
            "evidence_degraded",
            "Evidence degraded",
            "Evidence is incomplete or weak. Review the surfaced case, but do not self-approve it.",
        )
    return (
        "review_ready_surfaced",
        "Review surfaced cases",
        "Review surfaced cases against the old baseline.",
    )


def _stop_reason_for_decision(
    *,
    mode: str,
    overall_verdict: str,
    evidence_state: EvidenceState,
    surfaces_case: bool,
) -> str:
    if overall_verdict == "CONTRACT_INVALID":
        return "contract_invalid"
    if overall_verdict == "PASS" and mode == "migration" and surfaces_case:
        return "awaiting_user_blessing"
    if overall_verdict == "PASS":
        return "all_rules_satisfied"
    if overall_verdict == "AMBIGUOUS":
        return "visual_intent_unclear"
    if evidence_state is EvidenceState.MANIFEST_MISMATCH:
        return "evidence_contract_mismatch"
    if evidence_state is EvidenceState.PARTIAL:
        return "evidence_incomplete"
    if evidence_state is EvidenceState.EMPTY:
        return "evidence_missing"
    return "hard_metric_failed"


def _surfaced_case_status_for_decision(mode: str, overall_verdict: str) -> str:
    if overall_verdict == "CONTRACT_INVALID":
        return "CONTRACT_INVALID"
    if overall_verdict == "AMBIGUOUS":
        return "AMBIGUOUS"
    if mode == "migration" and overall_verdict == "PASS":
        return "INTENDED_CHANGE_CONFIRMED"
    return "REGRESSION"


def _material_reasons_for_decision(
    mode: str,
    overall_verdict: str,
    evidence_state: EvidenceState,
) -> tuple[str, ...]:
    if overall_verdict == "CONTRACT_INVALID":
        return ("contract_invalid",)
    if overall_verdict == "AMBIGUOUS":
        return ("visual_intent_unclear",)
    if mode == "migration" and overall_verdict == "PASS":
        return ("intended_change_requires_review",)
    if evidence_state is EvidenceState.PARTIAL:
        return ("hard_metric_failed", "partial_evidence")
    if evidence_state is EvidenceState.EMPTY:
        return ("hard_metric_failed", "missing_evidence")
    if evidence_state is EvidenceState.MANIFEST_MISMATCH:
        return ("hard_metric_failed", "evidence_contract_mismatch")
    return ("hard_metric_failed",)


def _baseline_authority_copy(mode: str) -> str:
    if mode == "migration":
        return (
            "Migration mode. Old baseline remains authoritative until surfaced cases are "
            "reviewed and a proposed baseline is explicitly blessed."
        )
    return (
        "Regression mode. Old baseline remains authoritative. "
        "No new baseline is available to bless."
    )


def _build_surfaced_case(
    *,
    contract: dict[str, Any],
    report: dict[str, Any],
    evaluation_result: EvaluationResult,
    manifest: PhaseManifest,
    evidence_pairs: list[EvidencePair],
    case_id: str,
    decision: ApprovalDecision,
) -> dict[str, Any]:
    proof_pair = next((pair for pair in evidence_pairs if pair.status != "mismatch"), None)
    rules = _build_rule_outcomes(contract, evaluation_result, decision.overall_verdict)
    metrics = _build_metric_observations(evaluation_result, report)
    return {
        "case_id": case_id,
        "status": decision.surfaced_case_status,
        "material_reason": list(decision.material_reasons),
        "proof_panel": {
            "phase_id": proof_pair.phase_id if proof_pair is not None else manifest.failed_phase_id,
            "view": proof_pair.view_name if proof_pair is not None else None,
            "status": proof_pair.status if proof_pair is not None else "empty",
            "current_image": (
                str(proof_pair.current_image_path)
                if proof_pair and proof_pair.current_image_path
                else None
            ),
            "baseline_image": (
                str(proof_pair.baseline_image_path)
                if proof_pair and proof_pair.baseline_image_path
                else None
            ),
        },
        "rules": rules,
        "metrics": metrics,
        "caption": decision.evidence.message,
    }


def _build_rule_outcomes(
    contract: dict[str, Any],
    evaluation_result: EvaluationResult,
    overall_verdict: str,
) -> dict[str, list[str]]:
    result_lookup = {
        (
            "all" if result.phase == "*" else result.phase,
            result.metric,
        ): result
        for result in evaluation_result.results
    }
    passed: list[str] = []
    failed: list[str] = []
    ambiguous: list[str] = []
    try:
        grounded_rules = _ground_contract_rules(contract)
    except ContractCompileError:
        grounded_rules = ()
        ambiguous.append("<contract-invalid>")
    for rule in grounded_rules:
        result = result_lookup.get((rule.phase_id, rule.metric))
        if result is None:
            ambiguous.append(rule.rule_id)
            continue
        if result.passed:
            passed.append(rule.rule_id)
        else:
            failed.append(rule.rule_id)
    if overall_verdict == "AMBIGUOUS":
        ambiguous.append("still_image_review_required")
    return {
        "passed": passed,
        "failed": failed,
        "ambiguous": ambiguous,
    }


def _build_metric_observations(
    evaluation_result: EvaluationResult,
    report: dict[str, Any],
) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    for result in evaluation_result.failed[:2]:
        observations.append(
            {
                "id": result.metric,
                "verdict": "FAIL",
                "observed": result.actual_value,
                "baseline": _lookup_baseline_value(report, result.phase, result.metric),
            }
        )
    return observations


def _ground_contract_rules(contract: dict[str, Any]) -> tuple[GroundedContractRule, ...]:
    """Validate contract shape and ground rules to evaluator assertion templates."""
    if contract.get("schema_version") != CONTRACT_SCHEMA_VERSION:
        raise _contract_error(
            cause=(
                "schema_version must be "
                f"'{CONTRACT_SCHEMA_VERSION}', got {contract.get('schema_version')!r}."
            ),
            fix="Use the reviewed v1 contract schema for this wedge.",
        )

    if contract.get("mode") not in {"regression", "migration"}:
        raise _contract_error(
            cause=f"mode must be 'regression' or 'migration', got {contract.get('mode')!r}.",
            fix="Set mode to 'regression' or 'migration'.",
        )

    rules = contract.get("rules")
    if not isinstance(rules, list) or not rules:
        raise _contract_error(
            cause="rules must be a non-empty list.",
            fix="Provide at least one grounded rule in the contract.",
        )
    return tuple(_ground_contract_rule(rule) for rule in rules)


def _ground_contract_rule(rule: dict[str, Any]) -> GroundedContractRule:
    """Validate one metric gate and tie it to the MuJoCo evaluator corpus."""
    supported_metrics = {assertion.metric for assertion in MUJOCO_GRASP_ASSERTIONS}
    raw_rule_id = rule.get("id", "<missing-id>")
    rule_id = raw_rule_id if isinstance(raw_rule_id, str) else str(raw_rule_id)
    rule_type = rule.get("type")
    if rule_type != "metric_gate":
        raise _contract_error(
            cause=(
                f"rule '{rule_id}' uses type {rule_type!r}. The MuJoCo wedge grounds only "
                "metric_gate rules today."
            ),
            fix="Use metric_gate rules for this wedge or fall back to the default preset.",
        )
    judge = rule.get("judge")
    if judge != "metric":
        raise _contract_error(
            cause=(
                f"rule '{rule_id}' uses judge {judge!r}. The MuJoCo wedge grounds only "
                "metric rules today."
            ),
            fix="Use metric rules for this wedge or fall back to the default preset.",
        )

    evidence_at = rule.get("evidence_at")
    if not isinstance(evidence_at, list) or not evidence_at:
        raise _contract_error(
            cause=f"rule '{rule_id}' is missing a non-empty evidence_at list.",
            fix="Add at least one phase grounding to evidence_at.",
        )
    if len(evidence_at) != 1:
        raise _contract_error(
            cause=(
                f"rule '{rule_id}' declares {len(evidence_at)} evidence locations. "
                "The current MuJoCo wedge supports exactly one grounded location per rule."
            ),
            fix="Use one evidence_at entry per rule for this wedge.",
        )
    location = evidence_at[0]
    if not isinstance(location, dict) or not location.get("phase"):
        raise _contract_error(
            cause=f"rule '{rule_id}' has an invalid evidence_at entry: {location!r}.",
            fix="Each evidence_at entry needs at least a phase string.",
        )
    phase_id = location["phase"]
    if not isinstance(phase_id, str) or phase_id not in SUPPORTED_CONTRACT_PHASES:
        raise _contract_error(
            cause=(
                f"rule '{rule_id}' references unsupported phase {phase_id!r}. "
                f"Supported phases: {sorted(SUPPORTED_CONTRACT_PHASES)!r}."
            ),
            fix="Use one of the wedge's known phase ids or 'all' for summary metrics.",
        )
    view_name = location.get("view")
    if view_name is not None and view_name not in MUJOCO_GRASP_CAMERAS:
        raise _contract_error(
            cause=(
                f"rule '{rule_id}' references unsupported view {view_name!r}. "
                f"Supported views: {MUJOCO_GRASP_CAMERAS!r}."
            ),
            fix="Use one of the MuJoCo wedge camera names for evidence_at.view.",
        )

    pass_if = rule.get("pass_if")
    if not isinstance(pass_if, dict):
        raise _contract_error(
            cause=f"rule '{rule_id}' is missing pass_if.",
            fix="Add pass_if with metric, op, and value fields.",
        )
    metric = pass_if.get("metric")
    if not isinstance(metric, str) or metric not in supported_metrics:
        raise _contract_error(
            cause=(
                f"rule '{rule_id}' references unsupported metric {metric!r}. "
                f"Supported metrics: {sorted(supported_metrics)!r}."
            ),
            fix="Use one of the wedge's supported evaluator metrics or omit --contract-json.",
        )
    operator = _contract_operator(rule_id, pass_if.get("op"))
    threshold = _contract_threshold(rule_id, pass_if.get("op"), pass_if.get("value"))
    template = _contract_assertion_template(metric, phase_id)
    if template is None:
        raise _contract_error(
            cause=(
                f"rule '{rule_id}' cannot be grounded to the MuJoCo wedge evaluator for "
                f"phase {phase_id!r} and metric {metric!r}."
            ),
            fix=(
                "Use the phase/metric combinations from the reviewed preset, or omit "
                "--contract-json to use the default contract."
            ),
        )
    return GroundedContractRule(
        rule_id=rule_id,
        phase_id=phase_id,
        metric=metric,
        operator=operator,
        threshold=threshold,
        template=template,
    )


def _contract_assertion_template(metric: Any, phase_id: Any) -> MetricAssertion | None:
    if not isinstance(metric, str) or not isinstance(phase_id, str):
        return None
    return _DEFAULT_ASSERTION_BY_CONTRACT_KEY.get((phase_id, metric))


def _contract_operator(rule_id: str, op_value: Any) -> Operator:
    if not isinstance(op_value, str):
        raise _contract_error(
            cause=f"rule '{rule_id}' is missing a valid pass_if.op string.",
            fix="Set pass_if.op to one of the wedge's supported operators.",
        )
    try:
        return Operator(op_value)
    except ValueError as exc:
        raise _contract_error(
            cause=f"rule '{rule_id}' uses unsupported operator {op_value!r}.",
            fix=f"Use one of {[operator.value for operator in Operator]!r}.",
        ) from exc


def _contract_threshold(
    rule_id: str,
    op_value: Any,
    threshold_value: Any,
) -> float | tuple[float, float]:
    operator = _contract_operator(rule_id, op_value)
    if operator == Operator.IN_RANGE:
        if (
            not isinstance(threshold_value, (list, tuple))
            or len(threshold_value) != 2
            or any(
                not isinstance(value, (int, float)) or isinstance(value, bool)
                for value in threshold_value
            )
        ):
            raise _contract_error(
                cause=(
                    f"rule '{rule_id}' must use a two-number pass_if.value for "
                    f"operator {operator.value!r}."
                ),
                fix="Set pass_if.value to [low, high] for in_range rules.",
            )
        return (float(threshold_value[0]), float(threshold_value[1]))
    if not isinstance(threshold_value, (int, float)) or isinstance(threshold_value, bool):
        raise _contract_error(
            cause=f"rule '{rule_id}' must use a numeric pass_if.value.",
            fix="Set pass_if.value to a number for this operator.",
        )
    return float(threshold_value)


def _ungrounded_rule_ids(
    contract: dict[str, Any],
    evaluation_result: EvaluationResult,
) -> list[str]:
    result_lookup = {
        ("all" if result.phase == "*" else result.phase, result.metric): result
        for result in evaluation_result.results
    }
    try:
        grounded_rules = _ground_contract_rules(contract)
    except ContractCompileError:
        return ["<contract-invalid>"]
    missing: list[str] = []
    for rule in grounded_rules:
        if (rule.phase_id, rule.metric) not in result_lookup:
            missing.append(rule.rule_id)
    return missing


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
    return _build_evidence_summary(manifest, evidence_pairs).label


def _build_evidence_summary(
    manifest: PhaseManifest,
    evidence_pairs: list[EvidencePair],
) -> EvidenceSummary:
    state = _resolve_evidence_state(manifest, evidence_pairs)
    return EvidenceSummary(
        state=state,
        message=_summary_evidence_message_for_state(state, manifest, evidence_pairs),
    )


def _resolve_evidence_state(
    manifest: PhaseManifest,
    evidence_pairs: list[EvidencePair],
) -> EvidenceState:
    if manifest.failed_phase_id is None:
        return EvidenceState.PASS_NO_FAILED_PHASE
    if not evidence_pairs or all(pair.status == "empty" for pair in evidence_pairs):
        return EvidenceState.EMPTY
    statuses = {pair.status for pair in evidence_pairs}
    if "mismatch" in statuses:
        return EvidenceState.MANIFEST_MISMATCH
    if "ambiguous" in statuses:
        return EvidenceState.AMBIGUOUS_STILL_IMAGE
    if "partial" in statuses or "empty" in statuses:
        return EvidenceState.PARTIAL
    return EvidenceState.FULL


def _summary_evidence_message(
    manifest: PhaseManifest,
    evidence_pairs: list[EvidencePair],
) -> str:
    return _build_evidence_summary(manifest, evidence_pairs).message


def _summary_evidence_message_for_state(
    state: EvidenceState,
    manifest: PhaseManifest,
    evidence_pairs: list[EvidencePair],
) -> str:
    diagnostics = _collect_diagnostic_messages(evidence_pairs)
    if state is EvidenceState.PASS_NO_FAILED_PHASE:
        return "No visual regression detected for the canonical primary views."
    if state is EvidenceState.MANIFEST_MISMATCH and diagnostics:
        return diagnostics[0]
    if state is EvidenceState.EMPTY:
        return (
            "No visual evidence available for the failing phase. "
            f"Re-run from {manifest.rerun_hint} to rebuild evidence."
        )
    if state is EvidenceState.AMBIGUOUS_STILL_IMAGE:
        if diagnostics:
            return diagnostics[0]
        return (
            "Still-image evidence is suggestive, but temporal proof is weak. "
            f"Re-run from {manifest.rerun_hint} if motion timing matters."
        )
    if state is EvidenceState.PARTIAL and diagnostics:
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


def _render_run_decision_banner(approval_report: dict[str, Any]) -> str:
    return (
        f'<section class="run-decision run-decision-{html.escape(approval_report["run_state"])}">'
        '<div class="run-decision-main">'
        '<p class="run-decision-kicker">Run Decision</p>'
        f"<h3>{html.escape(approval_report['run_state_title'])}</h3>"
        f"<p>{html.escape(approval_report['run_state_message'])}</p>"
        '<p class="run-baseline-authority">'
        f"{html.escape(approval_report['baseline_authority'])}"
        "</p>"
        "</div>"
        '<div class="run-decision-meta">'
        f"<p><strong>Verdict</strong><span>{html.escape(approval_report['overall_verdict'])}</span></p>"
        f"<p><strong>Mode</strong><span>{html.escape(approval_report['mode'])}</span></p>"
        "<p><strong>Stop reason</strong><span>"
        f"{html.escape(approval_report['stop_reason'])}"
        "</span></p>"
        "</div>"
        "</section>"
    )


def _render_case_counts(approval_report: dict[str, Any]) -> str:
    summary = approval_report["summary"]
    counts = (
        ("Surfaced", summary["cases_surfaced"]),
        ("Suppressed", summary["cases_suppressed"]),
        ("Unchanged", summary["cases_unchanged"]),
        ("Total", summary["cases_total"]),
    )
    pills = "".join(
        (f'<div class="count-pill"><span>{html.escape(label)}</span><strong>{count}</strong></div>')
        for label, count in counts
    )
    return f'<section class="counts-strip">{pills}</section>'


def _render_approval_queue(approval_report: dict[str, Any]) -> str:
    surfaced_cases = approval_report["surfaced_cases"]
    if surfaced_cases:
        cards = "".join(_render_queue_card(case, approval_report) for case in surfaced_cases)
    else:
        cards = (
            '<article class="queue-card queue-card-success">'
            "<strong>No material changes surfaced.</strong>"
            "<p>Old baseline remains authoritative.</p>"
            "</article>"
        )
    suppressed_summary = ""
    if approval_report["suppressed_cases"]:
        reason = approval_report["suppressed_cases"][0]["reason"].replace("_", " ")
        suppressed_summary = (
            '<p class="queue-suppressed-copy">'
            f"Suppressed cases stay out of the first screen because: {html.escape(reason)}."
            "</p>"
        )
    return (
        '<section class="approval-queue">'
        "<div>"
        "<h3>Approval Queue</h3>"
        "<p>Changed or ambiguous cases only. Unchanged cases stay off the first screen.</p>"
        "</div>"
        f'<div class="queue-list">{cards}</div>'
        f"{suppressed_summary}"
        "</section>"
    )


def _render_queue_card(case: dict[str, Any], approval_report: dict[str, Any]) -> str:
    reasons = ", ".join(reason.replace("_", " ") for reason in case["material_reason"])
    rules = case["rules"]
    return (
        '<article class="queue-card queue-card-surfaced">'
        '<div class="queue-card-head">'
        f'<span class="queue-badge">{html.escape(case["status"])}</span>'
        f'<span class="queue-badge queue-badge-case">{html.escape(case["case_id"])}</span>'
        "</div>"
        f"<p><strong>Why surfaced:</strong> {html.escape(reasons)}</p>"
        f"<p>{html.escape(case['caption'])}</p>"
        f"<p><strong>Rules:</strong> passed {len(rules['passed'])}, failed {len(rules['failed'])}, "
        f"ambiguous {len(rules['ambiguous'])}</p>"
        '<p class="queue-baseline-authority">'
        f"{html.escape(approval_report['baseline_authority'])}"
        "</p>"
        "</article>"
    )


def _render_contract_section(contract: dict[str, Any]) -> str:
    return (
        '<div class="summary-card">'
        "<h3>Compiled Contract</h3>"
        '<div class="table-scroll"><table class="meta-table">'
        f"<tr><th>Contract</th><td><code>{html.escape(contract['contract_id'])}</code></td></tr>"
        "<tr><th>Preset</th><td><code>"
        f"{html.escape(contract.get('contract_preset', 'custom'))}"
        "</code></td></tr>"
        f"<tr><th>Mode</th><td><code>{html.escape(contract['mode'])}</code></td></tr>"
        f"<tr><th>Rules</th><td>{len(contract['rules'])}</td></tr>"
        f"<tr><th>Cases</th><td><code>{html.escape(str(contract['cases']['source']))}</code></td></tr>"
        "</table></div>"
        "</div>"
    )


def _render_baseline_section(approval_report: dict[str, Any]) -> str:
    user_action = approval_report["user_action"]
    if user_action["needs_baseline_blessing"]:
        body = (
            "A proposed baseline exists, but it remains inactive until every surfaced "
            "case is reviewed and the reviewer explicitly blesses it."
        )
    else:
        body = "No new baseline is available to bless for this run."
    return (
        '<div class="summary-card">'
        "<h3>Baseline Promotion</h3>"
        f"<p>{html.escape(approval_report['baseline_authority'])}</p>"
        f"<p>{html.escape(body)}</p>"
        "</div>"
    )


def _render_evidence_section(
    manifest: PhaseManifest,
    evidence_pairs: list[EvidencePair],
    evidence: EvidenceSummary,
) -> str:
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
        f'<p class="evidence-state-label">{html.escape(evidence.label)}</p>'
        "</div>"
        f"{_render_evidence_banner(evidence.banner_variant, evidence.message)}"
        f"{mismatch_banners}"
        f'<div class="evidence-grid">{body}</div>'
        "</section>"
    )


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
    temporal_evidence = _render_temporal_evidence(pair)
    return (
        f'<article class="evidence-card evidence-card-{html.escape(pair.status)}">'
        '<div class="evidence-card-head">'
        f'<span class="evidence-badge">Phase {html.escape(pair.phase_label)}</span>'
        f'<span class="evidence-badge">View {html.escape(pair.view_name)}</span>'
        f'<span class="evidence-badge evidence-badge-status">{html.escape(pair.status)}</span>'
        "</div>"
        '<div class="evidence-compare-grid">'
        f"{_render_evidence_media(pair, side='current')}"
        f"{_render_evidence_media(pair, side='baseline')}"
        "</div>"
        f'<div class="evidence-chip-row">{chips}</div>'
        f"{temporal_evidence}"
        f'<p class="evidence-caption">{html.escape(pair.interpretation_caption)}</p>'
        f"{diagnostic}"
        "</article>"
    )


def _render_metric_chip(explanation: MetricExplanation) -> str:
    return f'<span class="metric-chip">{html.escape(explanation.copy)}</span>'


def _render_evidence_media(pair: EvidencePair, *, side: str) -> str:
    if side == "current":
        image_path = pair.current_image_path
        fallback = _missing_role_message(pair, side)
        label = pair.current_label
    else:
        image_path = pair.baseline_image_path
        fallback = _missing_role_message(pair, side)
        label = pair.baseline_label

    alt = f"{pair.phase_label} ({pair.phase_id}) {pair.view_name} {label} evidence"
    if image_path is not None and image_path.exists():
        caption = f"{label} {pair.phase_label} / {pair.view_name}"
        return (
            '<figure class="evidence-figure">'
            f'<div class="evidence-role">{html.escape(label)}</div>'
            f"{_render_zoomable_image(image_path, alt=alt, caption=caption)}"
            f"<figcaption>{html.escape(caption)}</figcaption>"
            "</figure>"
        )
    return (
        '<figure class="evidence-figure">'
        f'<div class="evidence-role">{html.escape(label)}</div>'
        f'<div class="evidence-placeholder" role="img" aria-label="{html.escape(fallback)}">'
        f"{html.escape(label)}"
        "</div>"
        f"<figcaption>{html.escape(fallback)}</figcaption>"
        "</figure>"
    )


def _missing_role_message(pair: EvidencePair, side: str) -> str:
    if pair.diagnostic_message:
        if side == "current" and "Current capture missing" in pair.diagnostic_message:
            return pair.diagnostic_message
        if side == "baseline" and "Baseline image missing" in pair.diagnostic_message:
            return pair.diagnostic_message
        if pair.status in {"empty", "mismatch"}:
            return pair.diagnostic_message
    label = pair.current_label if side == "current" else pair.baseline_label
    return f"{label} image missing for {pair.phase_id}/{pair.view_name}."


def _render_temporal_evidence(pair: EvidencePair) -> str:
    if pair.status != "ambiguous":
        return ""
    current_root = _image_root_from_path(pair.current_image_path)
    baseline_root = _image_root_from_path(pair.baseline_image_path)
    if current_root is None or baseline_root is None:
        return ""
    return (
        '<div class="temporal-evidence">'
        '<div class="temporal-evidence-head">'
        "<strong>Temporal Evidence</strong>"
        "<p>Checkpoint order adds motion context for this view. It is still not "
        "continuous video.</p>"
        "</div>"
        f"{_render_temporal_row('Current checkpoints', current_root, pair.view_name, 'current')}"
        f"{_render_temporal_row('Baseline checkpoints', baseline_root, pair.view_name, 'baseline')}"
        "</div>"
    )


def _render_temporal_row(
    title: str,
    visual_root: Path,
    view_name: str,
    role: str,
) -> str:
    frames = "".join(
        _render_temporal_frame(
            visual_root=visual_root,
            phase_id=phase_id,
            view_name=view_name,
            role=role,
        )
        for phase_id in MUJOCO_GRASP_PHASE_ORDER
    )
    return (
        '<div class="temporal-row">'
        f'<div class="temporal-row-head">{html.escape(title)}</div>'
        f'<div class="temporal-grid">{frames}</div>'
        "</div>"
    )


def _render_temporal_frame(
    *,
    visual_root: Path,
    phase_id: str,
    view_name: str,
    role: str,
) -> str:
    phase_label = MUJOCO_GRASP_PHASE_LABELS[phase_id]
    image_path = visual_root / phase_id / f"{view_name}_rgb.png"
    alt = f"{role} {phase_label} ({phase_id}) {view_name} temporal evidence"
    caption = f"{role.title()} {phase_label} / {view_name}"
    if image_path.exists():
        body = _render_zoomable_image(
            image_path,
            alt=alt,
            caption=caption,
            class_name="evidence-zoom-button temporal-zoom-button",
        )
    else:
        body = (
            f'<div class="temporal-placeholder" role="img" '
            f'aria-label="{html.escape(caption)} missing">'
            "Missing"
            "</div>"
        )
    return (
        '<figure class="temporal-figure">'
        f'<div class="temporal-frame-label">{html.escape(phase_label)}</div>'
        f"{body}"
        f"<figcaption>{html.escape(caption)}</figcaption>"
        "</figure>"
    )


def _image_root_from_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    parents = path.parents
    if len(parents) < 2:
        return None
    return parents[1]
