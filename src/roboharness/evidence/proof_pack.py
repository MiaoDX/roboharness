"""Proof-pack assembly for downstream visual harness evidence."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from roboharness._utils import save_json
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


def _dedupe(values: Any) -> tuple[str, ...]:
    deduped: list[str] = []
    for value in values:
        text = str(value)
        if text not in deduped:
            deduped.append(text)
    return tuple(deduped)
