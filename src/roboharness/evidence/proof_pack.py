"""Proof-pack assembly for downstream visual harness evidence."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from roboharness._utils import load_json, save_json
from roboharness.approval.visual_review import MANIFEST_SCHEMA_VERSION
from roboharness.evidence.artifacts import (
    AutonomousEvidenceReport,
    RenderedImage,
    RendererReport,
    SemanticSnapshotBundle,
    load_autonomous_evidence_report,
    load_renderer_report,
    load_semantic_snapshot_bundle,
)

CASE_PROOF_PACK_SCHEMA_VERSION = "roboharness_case_proof_pack/v1"
SUITE_PROOF_PACK_SCHEMA_VERSION = "roboharness_suite_proof_pack/v1"
VISUAL_REVIEW_QUEUE_SCHEMA_VERSION = "roboharness_visual_review_queue/v1"
STATIC_VISUAL_DIMENSIONS = (
    "robot_posture",
    "hand_pose",
    "object_relative_position",
    "obvious_collision_or_penetration",
    "task_success_visual_check",
)
DEFAULT_STATIC_REVIEW_VIEWS = ("front2back", "left2right", "top2down")
DEFAULT_METRIC_SUMMARY_KEYS = (
    "semantic_visual_ok",
    "workspace_framing_ok",
    "snapshot_state_progress_ok",
    "final_snapshot_name",
    "grasp_accuracy_snapshot_name",
    "holding_snapshot_name",
    "grip_center_error_mm",
    "holding_grip_center_error_mm",
    "pinch_gap_error_mm",
    "pinch_elevation_deg",
    "index_middle_vertical_deg",
    "pelvis_roll_deg",
    "pelvis_pitch_deg",
    "pelvis_height_m",
    "non_primary_arm_drift_deg",
    "cmd_vs_actual_max_delta_deg",
    "waypoint_wrist_gap_mm",
    "waypoint_first_segment_mm",
    "planning_success",
    "pregrasp_converged",
    "approach_converged",
    "lift_converged",
    "holding_reached",
    "control_backend",
    "runtime_surface",
    "render_mujoco_enabled",
    "render_meshcat_capture_s",
    "render_mujoco_capture_s",
    "render_live_total_s",
    "render_total_s",
    "whole_case_wall_time_s",
)


@dataclass(frozen=True)
class ProofPackArtifact:
    """A case-local artifact consumed by a proof pack."""

    id: str
    path: str
    kind: str

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "path": self.path, "kind": self.kind}


@dataclass(frozen=True)
class ProofPackImageRef:
    """One case-local rendered image reference selected for review."""

    renderer: str
    phase: str
    view: str
    path: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "renderer": self.renderer,
            "phase": self.phase,
            "view": self.view,
            "path": self.path,
        }
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True)
class CaseProofPack:
    """Compact evidence bundle prepared from one downstream visual harness case."""

    case_id: str
    output_dir: str
    verdict: str
    verdict_reasons: tuple[str, ...]
    failure_taxonomy: tuple[dict[str, Any], ...]
    snapshot_order: tuple[str, ...]
    selected_phase: str
    metric_summary: dict[str, Any]
    renderer_evidence: tuple[ProofPackImageRef, ...]
    artifacts: tuple[ProofPackArtifact, ...]
    review_mode: str = "current_only"
    schema_version: str = CASE_PROOF_PACK_SCHEMA_VERSION

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CaseProofPack:
        return cls(
            case_id=str(data["case_id"]),
            output_dir=str(data.get("output_dir") or ""),
            verdict=str(data.get("verdict") or ""),
            verdict_reasons=tuple(str(reason) for reason in data.get("verdict_reasons", ())),
            failure_taxonomy=tuple(dict(item) for item in data.get("failure_taxonomy", ())),
            snapshot_order=tuple(str(name) for name in data.get("snapshot_order", ())),
            selected_phase=str(data.get("selected_phase") or ""),
            metric_summary=dict(data.get("metric_summary") or {}),
            renderer_evidence=tuple(
                ProofPackImageRef(
                    renderer=str(item["renderer"]),
                    phase=str(item["phase"]),
                    view=str(item["view"]),
                    path=str(item["path"]),
                    metadata=dict(item.get("metadata") or {}),
                )
                for item in data.get("renderer_evidence", ())
            ),
            artifacts=tuple(
                ProofPackArtifact(
                    id=str(item["id"]),
                    path=str(item["path"]),
                    kind=str(item["kind"]),
                )
                for item in data.get("artifacts", ())
            ),
            review_mode=str(data.get("review_mode") or "current_only"),
            schema_version=str(data.get("schema_version") or CASE_PROOF_PACK_SCHEMA_VERSION),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "case_id": self.case_id,
            "output_dir": self.output_dir,
            "verdict": self.verdict,
            "verdict_reasons": list(self.verdict_reasons),
            "failure_taxonomy": [dict(item) for item in self.failure_taxonomy],
            "snapshot_order": list(self.snapshot_order),
            "selected_phase": self.selected_phase,
            "metric_summary": dict(self.metric_summary),
            "renderer_evidence": [item.to_dict() for item in self.renderer_evidence],
            "artifacts": [item.to_dict() for item in self.artifacts],
            "review_mode": self.review_mode,
        }

    def write_json(self, path: str | Path) -> Path:
        return write_case_proof_pack(self, path)


@dataclass(frozen=True)
class SuiteProofPackCase:
    """One case entry inside a suite proof pack."""

    case_id: str
    case_dir: str
    status: str
    proof_pack_path: str | None
    visual_review_manifest_path: str | None
    selected_phase: str | None = None
    verdict: str | None = None
    renderer_evidence_count: int = 0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "case_id": self.case_id,
            "case_dir": self.case_dir,
            "status": self.status,
            "proof_pack_path": self.proof_pack_path,
            "visual_review_manifest_path": self.visual_review_manifest_path,
            "renderer_evidence_count": int(self.renderer_evidence_count),
        }
        if self.selected_phase is not None:
            payload["selected_phase"] = self.selected_phase
        if self.verdict is not None:
            payload["verdict"] = self.verdict
        if self.error is not None:
            payload["error"] = self.error
        return payload


@dataclass(frozen=True)
class SuiteProofPack:
    """Suite-level index over case proof packs prepared for visual review."""

    suite_name: str
    suite_dir: str
    suite_report_path: str
    cases: tuple[SuiteProofPackCase, ...]
    schema_version: str = SUITE_PROOF_PACK_SCHEMA_VERSION

    @property
    def reviewable_count(self) -> int:
        return sum(1 for case in self.cases if case.status == "reviewable")

    @property
    def skipped_count(self) -> int:
        return sum(1 for case in self.cases if case.status != "reviewable")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "suite_name": self.suite_name,
            "suite_dir": self.suite_dir,
            "suite_report_path": self.suite_report_path,
            "total_cases": len(self.cases),
            "reviewable_count": self.reviewable_count,
            "skipped_count": self.skipped_count,
            "cases": [case.to_dict() for case in self.cases],
        }

    def write_json(self, path: str | Path) -> Path:
        return write_suite_proof_pack(self, path)


@dataclass(frozen=True)
class VisualReviewQueueItem:
    """One bounded visual review item selected from a suite proof pack."""

    case_id: str
    case_dir: str
    visual_review_manifest_path: str
    proof_pack_path: str
    selected_phase: str
    verdict: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "case_dir": self.case_dir,
            "visual_review_manifest_path": self.visual_review_manifest_path,
            "proof_pack_path": self.proof_pack_path,
            "selected_phase": self.selected_phase,
            "verdict": self.verdict,
        }


@dataclass(frozen=True)
class VisualReviewQueue:
    """Review queue derived from a suite proof pack."""

    suite_name: str
    suite_dir: str
    items: tuple[VisualReviewQueueItem, ...]
    schema_version: str = VISUAL_REVIEW_QUEUE_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "suite_name": self.suite_name,
            "suite_dir": self.suite_dir,
            "total_items": len(self.items),
            "items": [item.to_dict() for item in self.items],
        }

    def write_json(self, path: str | Path) -> Path:
        return write_visual_review_queue(self, path)


def build_case_proof_pack(
    case_dir: str | Path,
    *,
    preferred_renderers: tuple[str, ...] = ("mujoco", "meshcat"),
    preferred_views: tuple[str, ...] = DEFAULT_STATIC_REVIEW_VIEWS,
    metric_keys: tuple[str, ...] = DEFAULT_METRIC_SUMMARY_KEYS,
) -> CaseProofPack:
    """Build a case-level proof pack from a downstream visual harness case directory."""

    root = Path(case_dir)
    report_path = root / "autonomous_report.json"
    if not report_path.exists():
        raise FileNotFoundError(f"autonomous report not found: {report_path}")
    report = load_autonomous_evidence_report(report_path)

    bundle_path = root / "snapshot_bundle.json"
    bundle = load_semantic_snapshot_bundle(bundle_path) if bundle_path.exists() else None
    renderer_reports = _load_case_renderer_reports(root, report)
    snapshot_order = _snapshot_order(report, bundle)
    selected_phase = _select_review_phase(report, snapshot_order)
    metric_summary = _metric_summary(report, metric_keys)
    renderer_evidence = _select_renderer_evidence(
        root,
        renderer_reports,
        selected_phase=selected_phase,
        preferred_renderers=preferred_renderers,
        preferred_views=preferred_views,
    )
    artifacts = _artifact_refs(
        root,
        report,
        bundle_path=bundle_path,
        renderer_reports=renderer_reports,
    )

    return CaseProofPack(
        case_id=report.case_id,
        output_dir=_case_relative_path(root, Path(report.output_dir)) or root.as_posix(),
        verdict=report.verdict,
        verdict_reasons=report.verdict_reasons,
        failure_taxonomy=report.failure_taxonomy,
        snapshot_order=snapshot_order,
        selected_phase=selected_phase,
        metric_summary=metric_summary,
        renderer_evidence=renderer_evidence,
        artifacts=artifacts,
    )


def build_suite_proof_pack(
    suite_report_path: str | Path,
    *,
    write_missing_case_artifacts: bool = True,
    task_intent: str | None = None,
) -> SuiteProofPack:
    """Build a suite-level index over case proof packs and visual manifests."""

    report_path = Path(suite_report_path)
    suite_dir = report_path.parent
    suite_report = load_json(report_path)
    suite_name = str(
        suite_report.get("suite_name") or report_path.stem.removeprefix("suite_report_")
    )
    cases: list[SuiteProofPackCase] = []
    for result in _suite_results(suite_report):
        case_id = str(result.get("case_id") or "")
        case_dir = _suite_case_dir(suite_dir, result)
        proof_pack_path = case_dir / "proof_pack.json"
        manifest_path = case_dir / "visual_review_manifest.json"
        if not case_id:
            cases.append(
                SuiteProofPackCase(
                    case_id="",
                    case_dir=_suite_relative_path(suite_dir, case_dir) or case_dir.as_posix(),
                    status="skipped",
                    proof_pack_path=None,
                    visual_review_manifest_path=None,
                    error="suite result is missing case_id",
                )
            )
            continue
        if not case_dir.exists():
            cases.append(
                SuiteProofPackCase(
                    case_id=case_id,
                    case_dir=_suite_relative_path(suite_dir, case_dir) or case_dir.as_posix(),
                    status="skipped",
                    proof_pack_path=None,
                    visual_review_manifest_path=None,
                    error="case directory does not exist",
                )
            )
            continue
        if write_missing_case_artifacts and not proof_pack_path.exists():
            try:
                proof_pack = build_case_proof_pack(case_dir)
                write_case_proof_pack(proof_pack, proof_pack_path)
                write_static_visual_review_manifest(
                    proof_pack,
                    manifest_path,
                    task_intent=task_intent or _default_task_intent(proof_pack),
                )
            except Exception as exc:  # pragma: no cover - exact failures are data dependent
                cases.append(
                    SuiteProofPackCase(
                        case_id=case_id,
                        case_dir=_suite_relative_path(suite_dir, case_dir) or case_dir.as_posix(),
                        status="skipped",
                        proof_pack_path=None,
                        visual_review_manifest_path=None,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
                continue
        if not proof_pack_path.exists() or not manifest_path.exists():
            cases.append(
                SuiteProofPackCase(
                    case_id=case_id,
                    case_dir=_suite_relative_path(suite_dir, case_dir) or case_dir.as_posix(),
                    status="skipped",
                    proof_pack_path=(
                        _suite_relative_path(suite_dir, proof_pack_path)
                        if proof_pack_path.exists()
                        else None
                    ),
                    visual_review_manifest_path=(
                        _suite_relative_path(suite_dir, manifest_path)
                        if manifest_path.exists()
                        else None
                    ),
                    error="case proof pack or visual review manifest is missing",
                )
            )
            continue

        proof_pack = load_case_proof_pack(proof_pack_path)
        cases.append(
            SuiteProofPackCase(
                case_id=case_id,
                case_dir=_suite_relative_path(suite_dir, case_dir) or case_dir.as_posix(),
                status="reviewable",
                proof_pack_path=_suite_relative_path(suite_dir, proof_pack_path),
                visual_review_manifest_path=_suite_relative_path(suite_dir, manifest_path),
                selected_phase=proof_pack.selected_phase,
                verdict=proof_pack.verdict,
                renderer_evidence_count=len(proof_pack.renderer_evidence),
            )
        )
    return SuiteProofPack(
        suite_name=suite_name,
        suite_dir=suite_dir.as_posix(),
        suite_report_path=_suite_relative_path(suite_dir, report_path) or report_path.name,
        cases=tuple(cases),
    )


def write_suite_proof_pack(suite_proof_pack: SuiteProofPack, path: str | Path) -> Path:
    """Write a suite proof pack to JSON."""

    output_path = Path(path)
    save_json(suite_proof_pack.to_dict(), output_path)
    return output_path


def load_suite_proof_pack(path: str | Path) -> SuiteProofPack:
    """Load a suite proof pack from JSON."""

    data = load_json(Path(path))
    return SuiteProofPack(
        suite_name=str(data["suite_name"]),
        suite_dir=str(data.get("suite_dir") or ""),
        suite_report_path=str(data.get("suite_report_path") or ""),
        cases=tuple(
            SuiteProofPackCase(
                case_id=str(item.get("case_id") or ""),
                case_dir=str(item.get("case_dir") or ""),
                status=str(item.get("status") or ""),
                proof_pack_path=(
                    None
                    if item.get("proof_pack_path") is None
                    else str(item.get("proof_pack_path"))
                ),
                visual_review_manifest_path=(
                    None
                    if item.get("visual_review_manifest_path") is None
                    else str(item.get("visual_review_manifest_path"))
                ),
                selected_phase=(
                    None if item.get("selected_phase") is None else str(item.get("selected_phase"))
                ),
                verdict=None if item.get("verdict") is None else str(item.get("verdict")),
                renderer_evidence_count=int(item.get("renderer_evidence_count") or 0),
                error=None if item.get("error") is None else str(item.get("error")),
            )
            for item in data.get("cases", ())
        ),
        schema_version=str(data.get("schema_version") or SUITE_PROOF_PACK_SCHEMA_VERSION),
    )


def build_visual_review_queue(suite_proof_pack: SuiteProofPack) -> VisualReviewQueue:
    """Build a review queue from reviewable case entries in a suite proof pack."""

    items: list[VisualReviewQueueItem] = []
    for case in suite_proof_pack.cases:
        if (
            case.status != "reviewable"
            or case.proof_pack_path is None
            or case.visual_review_manifest_path is None
            or case.selected_phase is None
            or case.verdict is None
        ):
            continue
        items.append(
            VisualReviewQueueItem(
                case_id=case.case_id,
                case_dir=case.case_dir,
                visual_review_manifest_path=case.visual_review_manifest_path,
                proof_pack_path=case.proof_pack_path,
                selected_phase=case.selected_phase,
                verdict=case.verdict,
            )
        )
    return VisualReviewQueue(
        suite_name=suite_proof_pack.suite_name,
        suite_dir=suite_proof_pack.suite_dir,
        items=tuple(items),
    )


def write_visual_review_queue(queue: VisualReviewQueue, path: str | Path) -> Path:
    """Write a suite visual review queue to JSON."""

    output_path = Path(path)
    save_json(queue.to_dict(), output_path)
    return output_path


def write_case_proof_pack(proof_pack: CaseProofPack, path: str | Path) -> Path:
    """Write a case proof pack to JSON."""

    output_path = Path(path)
    save_json(proof_pack.to_dict(), output_path)
    return output_path


def load_case_proof_pack(path: str | Path) -> CaseProofPack:
    """Load a case proof pack from JSON."""

    from roboharness._utils import load_json

    return CaseProofPack.from_dict(load_json(Path(path)))


def build_static_visual_review_manifest(
    proof_pack: CaseProofPack,
    *,
    task_intent: str,
    dimensions: tuple[str, ...] = STATIC_VISUAL_DIMENSIONS,
    preferred_renderer: str | None = None,
    required: bool = True,
) -> dict[str, Any]:
    """Build a current-only static keyframe visual review manifest from a proof pack."""

    evidence = _manifest_evidence(proof_pack, preferred_renderer=preferred_renderer)
    if not evidence:
        raise ValueError(f"proof pack {proof_pack.case_id!r} has no renderer evidence")
    views = _dedupe(ref.view for ref in evidence)
    current_paths = _dedupe(ref.path for ref in evidence)
    metric_fallback = tuple(proof_pack.metric_summary)
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "case_id": proof_pack.case_id,
        "mode": "current_only",
        "task_intent": task_intent,
        "dimensions": [
            {
                "id": dimension,
                "required": required,
                "phase": proof_pack.selected_phase,
                "evidence_type": "current_static_keyframe",
                "views": list(views),
                "current": list(current_paths),
                "metric_fallback": list(metric_fallback),
                "why_not_metricized": (
                    "Static rendered keyframes catch posture, contact, and task-agreement "
                    "issues that scalar metrics do not fully encode."
                ),
            }
            for dimension in dimensions
        ],
        "metric_summary": dict(proof_pack.metric_summary),
        "review_policy": {
            "requires_paired_evidence": False,
            "allow_automatic_visual_pass": False,
            "human_escalation_reasons": ["current_only_review_cannot_auto_pass"],
        },
        "proof_pack": {
            "schema_version": proof_pack.schema_version,
            "selected_phase": proof_pack.selected_phase,
            "renderer_evidence": [ref.to_dict() for ref in evidence],
            "artifacts": [artifact.to_dict() for artifact in proof_pack.artifacts],
        },
    }


def build_paired_visual_review_manifest(
    current_proof_pack: CaseProofPack,
    baseline_proof_pack: CaseProofPack,
    *,
    task_intent: str,
    dimensions: tuple[str, ...] = STATIC_VISUAL_DIMENSIONS,
    preferred_renderer: str | None = None,
    required: bool = True,
    mode: str = "regression",
) -> dict[str, Any]:
    """Build an explicit current-vs-baseline static visual review manifest."""

    if mode not in {"regression", "migration"}:
        raise ValueError("paired visual review mode must be 'regression' or 'migration'")
    if current_proof_pack.case_id != baseline_proof_pack.case_id:
        raise ValueError(
            "paired visual review requires matching case_id values, got "
            f"{current_proof_pack.case_id!r} and {baseline_proof_pack.case_id!r}"
        )
    current_evidence = _manifest_evidence(
        current_proof_pack,
        preferred_renderer=preferred_renderer,
    )
    baseline_evidence = _manifest_evidence(
        baseline_proof_pack,
        preferred_renderer=preferred_renderer,
    )
    if not current_evidence:
        raise ValueError(
            f"current proof pack {current_proof_pack.case_id!r} has no renderer evidence"
        )
    if not baseline_evidence:
        raise ValueError(
            f"baseline proof pack {baseline_proof_pack.case_id!r} has no renderer evidence"
        )
    pairs = _paired_manifest_evidence(current_evidence, baseline_evidence)
    if not pairs:
        raise ValueError(f"no paired renderer evidence for case {current_proof_pack.case_id!r}")
    views = _dedupe(pair[0].view for pair in pairs)
    current_paths = _dedupe(pair[0].path for pair in pairs)
    baseline_paths = _dedupe(pair[1].path for pair in pairs)
    metric_fallback = tuple(current_proof_pack.metric_summary)
    human_reasons: list[str] = []
    allow_automatic_visual_pass = mode == "regression"
    if mode == "migration":
        human_reasons.append("baseline_blessing_required")
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "case_id": current_proof_pack.case_id,
        "mode": mode,
        "task_intent": task_intent,
        "dimensions": [
            {
                "id": dimension,
                "required": required,
                "phase": current_proof_pack.selected_phase,
                "evidence_type": "paired_keyframe",
                "views": list(views),
                "current": list(current_paths),
                "baseline": list(baseline_paths),
                "metric_fallback": list(metric_fallback),
                "why_not_metricized": (
                    "Paired static keyframes compare posture, contact, and "
                    "task-agreement evidence against an explicit baseline pack."
                ),
            }
            for dimension in dimensions
        ],
        "metric_summary": dict(current_proof_pack.metric_summary),
        "review_policy": {
            "requires_paired_evidence": True,
            "allow_automatic_visual_pass": allow_automatic_visual_pass,
            "human_escalation_reasons": human_reasons,
        },
        "proof_pack": {
            "schema_version": current_proof_pack.schema_version,
            "selected_phase": current_proof_pack.selected_phase,
            "baseline_selected_phase": baseline_proof_pack.selected_phase,
            "renderer_evidence": [current.to_dict() for current, _ in pairs],
            "baseline_renderer_evidence": [baseline.to_dict() for _, baseline in pairs],
            "artifacts": [artifact.to_dict() for artifact in current_proof_pack.artifacts],
            "baseline_artifacts": [
                artifact.to_dict() for artifact in baseline_proof_pack.artifacts
            ],
        },
    }


def write_paired_visual_review_manifest(
    current_proof_pack: CaseProofPack,
    baseline_proof_pack: CaseProofPack,
    path: str | Path,
    *,
    task_intent: str,
    dimensions: tuple[str, ...] = STATIC_VISUAL_DIMENSIONS,
    preferred_renderer: str | None = None,
    required: bool = True,
    mode: str = "regression",
) -> Path:
    """Write an explicit current-vs-baseline visual review manifest to JSON."""

    manifest = build_paired_visual_review_manifest(
        current_proof_pack,
        baseline_proof_pack,
        task_intent=task_intent,
        dimensions=dimensions,
        preferred_renderer=preferred_renderer,
        required=required,
        mode=mode,
    )
    output_path = Path(path)
    save_json(manifest, output_path)
    return output_path


def write_static_visual_review_manifest(
    proof_pack: CaseProofPack,
    path: str | Path,
    *,
    task_intent: str,
    dimensions: tuple[str, ...] = STATIC_VISUAL_DIMENSIONS,
    preferred_renderer: str | None = None,
    required: bool = True,
) -> Path:
    """Write a current-only static visual review manifest to JSON."""

    manifest = build_static_visual_review_manifest(
        proof_pack,
        task_intent=task_intent,
        dimensions=dimensions,
        preferred_renderer=preferred_renderer,
        required=required,
    )
    output_path = Path(path)
    save_json(manifest, output_path)
    return output_path


def _load_case_renderer_reports(
    root: Path,
    report: AutonomousEvidenceReport,
) -> dict[str, RendererReport]:
    renderer_reports: dict[str, RendererReport] = {}
    for name in report.renderer_reports:
        renderer_report_path = root / name / "report.json"
        renderer_reports[name] = (
            load_renderer_report(renderer_report_path)
            if renderer_report_path.exists()
            else report.renderer_reports[name]
        )
    for renderer_report_path in sorted(root.glob("*/report.json")):
        name = renderer_report_path.parent.name
        if name not in renderer_reports:
            renderer_reports[name] = load_renderer_report(renderer_report_path)
    return renderer_reports


def _snapshot_order(
    report: AutonomousEvidenceReport,
    bundle: SemanticSnapshotBundle | None,
) -> tuple[str, ...]:
    raw_order = report.extra.get("snapshot_order")
    if isinstance(raw_order, list) and all(isinstance(item, str) for item in raw_order):
        return tuple(raw_order)
    if bundle is not None:
        return bundle.snapshot_order
    if report.snapshot_metrics:
        return tuple(report.snapshot_metrics)
    return ()


def _select_review_phase(
    report: AutonomousEvidenceReport,
    snapshot_order: tuple[str, ...],
) -> str:
    final_snapshot = report.summary_metrics.get("final_snapshot_name")
    if isinstance(final_snapshot, str) and final_snapshot:
        return final_snapshot
    failure_phase = _failure_phase(report.failure_taxonomy)
    if failure_phase is not None:
        return failure_phase
    return snapshot_order[-1] if snapshot_order else "unknown"


def _failure_phase(failure_taxonomy: tuple[dict[str, Any], ...]) -> str | None:
    for item in failure_taxonomy:
        for key in ("phase", "snapshot", "snapshot_name"):
            value = item.get(key)
            if isinstance(value, str) and value:
                return value
    return None


def _metric_summary(
    report: AutonomousEvidenceReport,
    metric_keys: tuple[str, ...],
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for key in metric_keys:
        if key in report.summary_metrics:
            metrics[key] = report.summary_metrics[key]
        elif key in report.extra:
            metrics[key] = report.extra[key]
    metrics["verdict"] = report.verdict
    metrics["verdict_reasons"] = list(report.verdict_reasons)
    return metrics


def _select_renderer_evidence(
    root: Path,
    renderer_reports: dict[str, RendererReport],
    *,
    selected_phase: str,
    preferred_renderers: tuple[str, ...],
    preferred_views: tuple[str, ...],
) -> tuple[ProofPackImageRef, ...]:
    ordered_renderer_names = [name for name in preferred_renderers if name in renderer_reports] + [
        name for name in renderer_reports if name not in preferred_renderers
    ]
    refs: list[ProofPackImageRef] = []
    for renderer_name in ordered_renderer_names:
        renderer_report = renderer_reports[renderer_name]
        snapshot = _find_snapshot(renderer_report, selected_phase)
        if snapshot is None:
            continue
        images_by_view = _images_by_view(snapshot.images)
        for view in preferred_views:
            image = images_by_view.get(view)
            if image is None:
                continue
            refs.append(
                ProofPackImageRef(
                    renderer=renderer_report.renderer or renderer_name,
                    phase=snapshot.name,
                    view=view,
                    path=_case_relative_path(root, Path(image.path)) or image.path,
                    metadata=_image_metadata(image),
                )
            )
    return tuple(refs)


def _find_snapshot(renderer_report: RendererReport, name: str) -> Any | None:
    for snapshot in renderer_report.snapshots:
        if snapshot.name == name:
            return snapshot
    return renderer_report.snapshots[-1] if renderer_report.snapshots else None


def _images_by_view(images: tuple[RenderedImage, ...]) -> dict[str, RenderedImage]:
    result: dict[str, RenderedImage] = {}
    for image in images:
        view = image.camera or image.view
        if view is not None:
            result[view] = image
    return result


def _image_metadata(image: RenderedImage) -> dict[str, Any]:
    metadata = dict(image.metadata)
    for key in (
        "unique_colors",
        "foreground_fraction",
        "workspace_visible",
        "workspace_center_xy",
        "diff_from_previous",
    ):
        if key in image.extra:
            metadata[key] = image.extra[key]
    return metadata


def _artifact_refs(
    root: Path,
    report: AutonomousEvidenceReport,
    *,
    bundle_path: Path,
    renderer_reports: dict[str, RendererReport],
) -> tuple[ProofPackArtifact, ...]:
    artifacts = [
        ProofPackArtifact(
            id="autonomous_report",
            path="autonomous_report.json",
            kind="autonomous_evidence_report",
        )
    ]
    if bundle_path.exists():
        artifacts.append(
            ProofPackArtifact(
                id="snapshot_bundle",
                path="snapshot_bundle.json",
                kind="semantic_snapshot_bundle",
            )
        )
    for name, renderer_report in renderer_reports.items():
        renderer_path = root / name / "report.json"
        artifacts.append(
            ProofPackArtifact(
                id=f"{name}_renderer_report",
                path=(
                    _case_relative_path(root, renderer_path)
                    or _case_relative_path(root, Path(renderer_report.output_dir) / "report.json")
                    or f"{name}/report.json"
                ),
                kind="renderer_report",
            )
        )
    return tuple(artifacts)


def _suite_results(suite_report: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    results = suite_report.get("results")
    if not isinstance(results, list):
        return ()
    return tuple(item for item in results if isinstance(item, dict))


def _suite_case_dir(suite_dir: Path, result: dict[str, Any]) -> Path:
    output_dir = result.get("output_dir")
    if isinstance(output_dir, str) and output_dir:
        path = Path(output_dir)
        if path.is_absolute():
            return path
        if path.exists():
            return path
        suite_relative = suite_dir / path
        if suite_relative.exists():
            return suite_relative
        return path
    artifact_dir_name = result.get("artifact_dir_name")
    if isinstance(artifact_dir_name, str) and artifact_dir_name:
        return suite_dir / artifact_dir_name
    return suite_dir / str(result.get("case_id") or "")


def _suite_relative_path(root: Path, path: Path) -> str | None:
    if not path.as_posix():
        return None
    root_resolved = root.resolve()
    candidate = path if path.is_absolute() else Path.cwd() / path
    try:
        return candidate.resolve().relative_to(root_resolved).as_posix()
    except ValueError:
        return None


def _default_task_intent(proof_pack: CaseProofPack) -> str:
    return (
        f"Review {proof_pack.case_id} static visual harness keyframes against "
        "the bounded robot evidence and metric summary."
    )


def _case_relative_path(root: Path, path: Path) -> str | None:
    if not path.as_posix():
        return None
    root_resolved = root.resolve()
    candidate = path
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    try:
        return candidate.resolve().relative_to(root_resolved).as_posix()
    except ValueError:
        return None


def _manifest_evidence(
    proof_pack: CaseProofPack,
    *,
    preferred_renderer: str | None,
) -> tuple[ProofPackImageRef, ...]:
    if preferred_renderer is None:
        if any(ref.renderer == "mujoco" for ref in proof_pack.renderer_evidence):
            preferred_renderer = "mujoco"
        elif proof_pack.renderer_evidence:
            preferred_renderer = proof_pack.renderer_evidence[0].renderer
    selected = tuple(
        ref for ref in proof_pack.renderer_evidence if ref.renderer == preferred_renderer
    )
    return selected or proof_pack.renderer_evidence


def _paired_manifest_evidence(
    current_evidence: tuple[ProofPackImageRef, ...],
    baseline_evidence: tuple[ProofPackImageRef, ...],
) -> tuple[tuple[ProofPackImageRef, ProofPackImageRef], ...]:
    baseline_by_key = {(ref.renderer, ref.view): ref for ref in baseline_evidence}
    pairs: list[tuple[ProofPackImageRef, ProofPackImageRef]] = []
    for current in current_evidence:
        baseline = baseline_by_key.get((current.renderer, current.view))
        if baseline is not None:
            pairs.append((current, baseline))
    return tuple(pairs)


def _dedupe(values: Any) -> tuple[str, ...]:
    deduped: list[str] = []
    for value in values:
        text = str(value)
        if text not in deduped:
            deduped.append(text)
    return tuple(deduped)
