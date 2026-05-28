"""Typed evidence artifacts shared by downstream robot harnesses."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from roboharness._utils import load_json, save_json

SEMANTIC_SNAPSHOT_BUNDLE_SCHEMA_VERSION = "roboharness_semantic_snapshot_bundle/v1"
AUTONOMOUS_EVIDENCE_REPORT_SCHEMA_VERSION = "roboharness_autonomous_evidence_report/v1"


def _payload_without(data: dict[str, Any], keys: set[str]) -> dict[str, Any]:
    return {key: value for key, value in data.items() if key not in keys}


def _put_if_not_none(payload: dict[str, Any], key: str, value: Any) -> None:
    if value is not None:
        payload[key] = value


@dataclass(frozen=True)
class SemanticSnapshot:
    """One named semantic phase sample emitted by an evidence producer."""

    name: str
    state: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SemanticSnapshot:
        reserved = {"name", "state", "metrics", "metadata"}
        return cls(
            name=str(data["name"]),
            state=dict(data.get("state") or {}),
            metrics=dict(data.get("metrics") or {}),
            metadata=dict(data.get("metadata") or {}),
            extra=_payload_without(data, reserved),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = dict(self.extra)
        payload["name"] = self.name
        if self.state:
            payload["state"] = dict(self.state)
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        payload["metrics"] = dict(self.metrics)
        return payload


@dataclass(frozen=True)
class SemanticSnapshotBundle:
    """Ordered semantic snapshots plus run metadata for replay or review."""

    snapshots: tuple[SemanticSnapshot, ...]
    snapshot_order: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str | int = SEMANTIC_SNAPSHOT_BUNDLE_SCHEMA_VERSION
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SemanticSnapshotBundle:
        snapshots = tuple(SemanticSnapshot.from_dict(item) for item in data.get("snapshots", ()))
        order = tuple(str(name) for name in data.get("snapshot_order") or ())
        if not order:
            order = tuple(snapshot.name for snapshot in snapshots)
        reserved = {"schema_version", "snapshot_order", "snapshots", "metadata"}
        return cls(
            snapshots=snapshots,
            snapshot_order=order,
            metadata=dict(data.get("metadata") or {}),
            schema_version=data.get("schema_version", SEMANTIC_SNAPSHOT_BUNDLE_SCHEMA_VERSION),
            extra=_payload_without(data, reserved),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = dict(self.extra)
        payload["schema_version"] = self.schema_version
        payload["snapshot_order"] = list(
            self.snapshot_order or (snapshot.name for snapshot in self.snapshots)
        )
        payload["snapshots"] = [snapshot.to_dict() for snapshot in self.snapshots]
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload

    def write_json(self, path: str | Path) -> Path:
        return write_semantic_snapshot_bundle(self, path)


@dataclass(frozen=True)
class RenderedImage:
    """Single rendered image reference inside a renderer report."""

    path: str
    camera: str | None = None
    view: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RenderedImage:
        reserved = {"path", "camera", "view", "metadata"}
        return cls(
            path=str(data["path"]),
            camera=None if data.get("camera") is None else str(data["camera"]),
            view=None if data.get("view") is None else str(data["view"]),
            metadata=dict(data.get("metadata") or {}),
            extra=_payload_without(data, reserved),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = dict(self.extra)
        _put_if_not_none(payload, "camera", self.camera)
        _put_if_not_none(payload, "view", self.view)
        payload["path"] = self.path
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True)
class RendererSnapshot:
    """Renderer evidence for one semantic snapshot."""

    name: str
    images: tuple[RenderedImage, ...] = ()
    metrics: dict[str, Any] = field(default_factory=dict)
    capture_ok: bool | None = None
    motion_ok: bool | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RendererSnapshot:
        reserved = {"name", "images", "metrics", "capture_ok", "motion_ok", "metadata"}
        return cls(
            name=str(data["name"]),
            images=tuple(RenderedImage.from_dict(item) for item in data.get("images", ())),
            metrics=dict(data.get("metrics") or {}),
            capture_ok=data.get("capture_ok"),
            motion_ok=data.get("motion_ok"),
            metadata=dict(data.get("metadata") or {}),
            extra=_payload_without(data, reserved),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = dict(self.extra)
        payload["name"] = self.name
        _put_if_not_none(payload, "capture_ok", self.capture_ok)
        _put_if_not_none(payload, "motion_ok", self.motion_ok)
        payload["metrics"] = dict(self.metrics)
        payload["images"] = [image.to_dict() for image in self.images]
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True)
class RendererReport:
    """Structured output from rendering a semantic snapshot bundle."""

    output_dir: str
    renderer: str
    capture_ok: bool | None = None
    motion_ok: bool | None = None
    snapshots: tuple[RendererSnapshot, ...] = ()
    flags: tuple[str, ...] = ()
    trustworthiness_flags: tuple[dict[str, Any], ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RendererReport:
        reserved = {
            "schema_version",
            "output_dir",
            "renderer",
            "capture_ok",
            "motion_ok",
            "snapshots",
            "flags",
            "trustworthiness_flags",
            "metadata",
        }
        return cls(
            output_dir=str(data["output_dir"]),
            renderer=str(data.get("renderer") or "unknown"),
            capture_ok=data.get("capture_ok"),
            motion_ok=data.get("motion_ok"),
            snapshots=tuple(RendererSnapshot.from_dict(item) for item in data.get("snapshots", ())),
            flags=tuple(str(flag) for flag in data.get("flags", ())),
            trustworthiness_flags=tuple(
                dict(flag) for flag in data.get("trustworthiness_flags", ())
            ),
            metadata=dict(data.get("metadata") or {}),
            schema_version=data.get("schema_version"),
            extra=_payload_without(data, reserved),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = dict(self.extra)
        _put_if_not_none(payload, "schema_version", self.schema_version)
        payload["output_dir"] = self.output_dir
        payload["renderer"] = self.renderer
        _put_if_not_none(payload, "capture_ok", self.capture_ok)
        _put_if_not_none(payload, "motion_ok", self.motion_ok)
        payload["flags"] = list(self.flags)
        payload["trustworthiness_flags"] = [dict(flag) for flag in self.trustworthiness_flags]
        payload["metadata"] = dict(self.metadata)
        payload["snapshots"] = [snapshot.to_dict() for snapshot in self.snapshots]
        return payload

    def write_json(self, path: str | Path) -> Path:
        return write_renderer_report(self, path)


@dataclass(frozen=True)
class AutonomousEvidenceReport:
    """Machine-readable run report before approval or human escalation."""

    case_id: str
    output_dir: str
    verdict: str
    verdict_reasons: tuple[str, ...] = ()
    summary_metrics: dict[str, Any] = field(default_factory=dict)
    snapshot_metrics: dict[str, dict[str, Any]] = field(default_factory=dict)
    renderer_reports: dict[str, RendererReport] = field(default_factory=dict)
    failure_taxonomy: tuple[dict[str, Any], ...] = ()
    runtime: dict[str, Any] = field(default_factory=dict)
    plan: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AutonomousEvidenceReport:
        reserved = {
            "schema_version",
            "case_id",
            "output_dir",
            "verdict",
            "verdict_reasons",
            "summary_metrics",
            "snapshot_metrics",
            "renderer_reports",
            "failure_taxonomy",
            "runtime",
            "plan",
            "metadata",
        }
        renderer_reports = {
            str(name): RendererReport.from_dict(report)
            for name, report in dict(data.get("renderer_reports") or {}).items()
        }
        return cls(
            case_id=str(data["case_id"]),
            output_dir=str(data.get("output_dir") or ""),
            verdict=str(data.get("verdict") or ""),
            verdict_reasons=tuple(str(reason) for reason in data.get("verdict_reasons", ())),
            summary_metrics=dict(data.get("summary_metrics") or {}),
            snapshot_metrics={
                str(name): dict(metrics)
                for name, metrics in dict(data.get("snapshot_metrics") or {}).items()
            },
            renderer_reports=renderer_reports,
            failure_taxonomy=tuple(dict(item) for item in data.get("failure_taxonomy", ())),
            runtime=dict(data.get("runtime") or {}),
            plan=dict(data.get("plan") or {}),
            metadata=dict(data.get("metadata") or {}),
            schema_version=data.get("schema_version"),
            extra=_payload_without(data, reserved),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = dict(self.extra)
        _put_if_not_none(payload, "schema_version", self.schema_version)
        payload["case_id"] = self.case_id
        payload["output_dir"] = self.output_dir
        payload["verdict"] = self.verdict
        payload["verdict_reasons"] = list(self.verdict_reasons)
        payload["failure_taxonomy"] = [dict(item) for item in self.failure_taxonomy]
        payload["runtime"] = dict(self.runtime)
        payload["plan"] = dict(self.plan)
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        payload["summary_metrics"] = dict(self.summary_metrics)
        payload["snapshot_metrics"] = {
            name: dict(metrics) for name, metrics in self.snapshot_metrics.items()
        }
        payload["renderer_reports"] = {
            name: report.to_dict() for name, report in self.renderer_reports.items()
        }
        return payload

    def write_json(self, path: str | Path) -> Path:
        return write_autonomous_evidence_report(self, path)


def load_semantic_snapshot_bundle(path: str | Path) -> SemanticSnapshotBundle:
    """Load a semantic snapshot bundle from JSON."""
    return SemanticSnapshotBundle.from_dict(load_json(Path(path)))


def write_semantic_snapshot_bundle(bundle: SemanticSnapshotBundle, path: str | Path) -> Path:
    """Write a semantic snapshot bundle to JSON."""
    output_path = Path(path)
    save_json(bundle.to_dict(), output_path)
    return output_path


def load_renderer_report(path: str | Path) -> RendererReport:
    """Load a renderer report from JSON."""
    return RendererReport.from_dict(load_json(Path(path)))


def write_renderer_report(report: RendererReport, path: str | Path) -> Path:
    """Write a renderer report to JSON."""
    output_path = Path(path)
    save_json(report.to_dict(), output_path)
    return output_path


def load_autonomous_evidence_report(path: str | Path) -> AutonomousEvidenceReport:
    """Load an autonomous evidence report from JSON."""
    return AutonomousEvidenceReport.from_dict(load_json(Path(path)))


def write_autonomous_evidence_report(
    report: AutonomousEvidenceReport,
    path: str | Path,
) -> Path:
    """Write an autonomous evidence report to JSON."""
    output_path = Path(path)
    save_json(report.to_dict(), output_path)
    return output_path
