"""Library-first visual harness lifecycle for downstream robot projects.

This module intentionally does not own robot runtime, simulation stepping,
renderer execution, or live safety semantics.  It gives downstream projects a
typed place to assemble visual evidence and delegates proof-pack/review artifact
generation to :mod:`roboharness.evidence`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from roboharness._utils import load_json, save_json
from roboharness.approval.visual_review import validate_visual_review_manifest
from roboharness.evidence import (
    AutonomousEvidenceReport,
    RendererReport,
    SemanticSnapshotBundle,
    build_case_proof_pack,
    build_suite_proof_pack,
    build_visual_review_queue,
    load_autonomous_evidence_report,
    load_renderer_report,
    load_semantic_snapshot_bundle,
    write_autonomous_evidence_report,
    write_case_proof_pack,
    write_static_visual_review_manifest,
    write_suite_proof_pack,
    write_visual_review_queue,
)


@dataclass(frozen=True)
class VisualCaseArtifacts:
    """Paths written or verified for one visual case."""

    case_dir: Path
    autonomous_report_path: Path
    proof_pack_path: Path
    visual_review_manifest_path: Path
    snapshot_bundle_path: Path | None = None
    renderer_report_paths: dict[str, Path] = field(default_factory=dict)


@dataclass(frozen=True)
class VisualSuiteArtifacts:
    """Paths written for one visual suite."""

    suite_dir: Path
    suite_report_path: Path
    suite_proof_pack_path: Path
    visual_review_queue_path: Path


@dataclass(frozen=True)
class VisualCaseSpec:
    """A suite case identifier plus optional downstream-owned payload."""

    case_id: str
    payload: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VisualCaseResult:
    """One downstream case result collected by the embedded suite executor."""

    result: dict[str, Any]
    case_run: VisualCaseRun | None = None


@dataclass(frozen=True)
class VisualSuiteSpec:
    """A suite name and ordered case list for embedded suite execution."""

    suite_name: str
    cases: Sequence[VisualCaseSpec]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VisualSuiteOptions:
    """Options for embedded suite execution and artifact writing."""

    task_intent: str | None = None
    suite_report_filename: str | None = None
    continue_on_error: bool = True
    write_case_artifacts: bool = True


@dataclass
class VisualCaseRun:
    """Mutable builder for one downstream visual-harness case run."""

    case_id: str
    output_dir: str | Path
    robot_type: str | None = None
    runner: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    runtime: dict[str, Any] = field(default_factory=dict)
    plan: dict[str, Any] = field(default_factory=dict)
    verdict: str = "unknown"
    verdict_reasons: list[str] = field(default_factory=list)
    failure_taxonomy: list[dict[str, Any]] = field(default_factory=list)
    summary_metrics: dict[str, Any] = field(default_factory=dict)
    snapshot_metrics: dict[str, dict[str, Any]] = field(default_factory=dict)
    snapshot_order: list[str] = field(default_factory=list)
    snapshot_bundle: SemanticSnapshotBundle | None = None
    renderer_reports: dict[str, RendererReport] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def case_dir(self) -> Path:
        """Return the case artifact directory."""

        return Path(self.output_dir)

    @classmethod
    def from_artifact_dir(cls, case_dir: str | Path) -> VisualCaseRun:
        """Load a case run builder from an existing visual artifact directory."""

        root = Path(case_dir)
        report = load_autonomous_evidence_report(root / "autonomous_report.json")
        snapshot_bundle_path = root / "snapshot_bundle.json"
        snapshot_bundle = (
            load_semantic_snapshot_bundle(snapshot_bundle_path)
            if snapshot_bundle_path.exists()
            else None
        )
        renderer_reports: dict[str, RendererReport] = {}
        for name in report.renderer_reports:
            renderer_path = root / name / "report.json"
            renderer_reports[name] = (
                load_renderer_report(renderer_path)
                if renderer_path.exists()
                else report.renderer_reports[name]
            )
        for renderer_path in sorted(root.glob("*/report.json")):
            renderer_name = renderer_path.parent.name
            if renderer_name not in renderer_reports:
                renderer_reports[renderer_name] = load_renderer_report(renderer_path)

        extra = dict(report.extra)
        robot_type = None if extra.get("robot_type") is None else str(extra.pop("robot_type"))
        runner_payload = extra.pop("runner", {})
        snapshot_order = extra.pop("snapshot_order", [])
        return cls(
            case_id=report.case_id,
            output_dir=root,
            robot_type=robot_type,
            runner=dict(runner_payload) if isinstance(runner_payload, dict) else {},
            metadata=dict(report.metadata),
            runtime=dict(report.runtime),
            plan=dict(report.plan),
            verdict=report.verdict,
            verdict_reasons=list(report.verdict_reasons),
            failure_taxonomy=[dict(item) for item in report.failure_taxonomy],
            summary_metrics=dict(report.summary_metrics),
            snapshot_metrics={
                name: dict(metrics) for name, metrics in report.snapshot_metrics.items()
            },
            snapshot_order=[str(item) for item in snapshot_order]
            if isinstance(snapshot_order, list)
            else [],
            snapshot_bundle=snapshot_bundle,
            renderer_reports=renderer_reports,
            extra=extra,
        )

    def set_snapshot_bundle(
        self,
        snapshot_bundle: SemanticSnapshotBundle | dict[str, Any],
    ) -> VisualCaseRun:
        """Set or replace the semantic snapshot bundle."""

        self.snapshot_bundle = _coerce_snapshot_bundle(snapshot_bundle)
        if not self.snapshot_order:
            self.snapshot_order = list(self.snapshot_bundle.snapshot_order)
        return self

    def add_renderer_report(
        self,
        name: str,
        renderer_report: RendererReport | dict[str, Any],
    ) -> VisualCaseRun:
        """Attach one renderer report to this case."""

        self.renderer_reports[str(name)] = _coerce_renderer_report(renderer_report)
        return self

    def set_metrics(
        self,
        summary_metrics: dict[str, Any],
        *,
        snapshot_metrics: dict[str, dict[str, Any]] | None = None,
    ) -> VisualCaseRun:
        """Set metric payloads for the autonomous evidence report."""

        self.summary_metrics = dict(summary_metrics)
        if snapshot_metrics is not None:
            self.snapshot_metrics = {
                str(name): dict(metrics) for name, metrics in snapshot_metrics.items()
            }
        return self

    def set_verdict(
        self,
        verdict: str,
        *,
        reasons: Sequence[str] = (),
        taxonomy: Sequence[dict[str, Any]] = (),
    ) -> VisualCaseRun:
        """Set the case verdict and failure taxonomy."""

        self.verdict = str(verdict)
        self.verdict_reasons = [str(reason) for reason in reasons]
        self.failure_taxonomy = [dict(item) for item in taxonomy]
        return self

    def to_autonomous_evidence_report(self) -> AutonomousEvidenceReport:
        """Build the autonomous evidence report for this case."""

        extra = dict(self.extra)
        if self.robot_type is not None:
            extra["robot_type"] = self.robot_type
        if self.runner:
            extra["runner"] = dict(self.runner)
        if self.snapshot_order:
            extra["snapshot_order"] = list(self.snapshot_order)
        return AutonomousEvidenceReport(
            case_id=self.case_id,
            output_dir=self.case_dir.as_posix(),
            verdict=self.verdict,
            verdict_reasons=tuple(self.verdict_reasons),
            summary_metrics=dict(self.summary_metrics),
            snapshot_metrics={
                name: dict(metrics) for name, metrics in self.snapshot_metrics.items()
            },
            renderer_reports=dict(self.renderer_reports),
            failure_taxonomy=tuple(dict(item) for item in self.failure_taxonomy),
            runtime=dict(self.runtime),
            plan=dict(self.plan),
            metadata=dict(self.metadata),
            extra=extra,
        )

    def write_artifacts(
        self,
        *,
        task_intent: str | None = None,
        write_snapshot_bundle: bool = True,
        write_renderer_reports: bool = True,
    ) -> VisualCaseArtifacts:
        """Write case evidence and visual-review artifacts."""

        root = self.case_dir
        root.mkdir(parents=True, exist_ok=True)
        snapshot_bundle_path: Path | None = None
        if write_snapshot_bundle and self.snapshot_bundle is not None:
            snapshot_bundle_path = self.snapshot_bundle.write_json(root / "snapshot_bundle.json")

        renderer_report_paths: dict[str, Path] = {}
        if write_renderer_reports:
            for name, report in self.renderer_reports.items():
                renderer_dir = root / name
                renderer_dir.mkdir(parents=True, exist_ok=True)
                renderer_report_paths[name] = report.write_json(renderer_dir / "report.json")

        autonomous_report_path = write_autonomous_evidence_report(
            self.to_autonomous_evidence_report(),
            root / "autonomous_report.json",
        )
        proof_pack_path, manifest_path = write_case_visual_artifacts(
            root,
            task_intent=task_intent,
        )
        return VisualCaseArtifacts(
            case_dir=root,
            autonomous_report_path=autonomous_report_path,
            proof_pack_path=proof_pack_path,
            visual_review_manifest_path=manifest_path,
            snapshot_bundle_path=snapshot_bundle_path,
            renderer_report_paths=renderer_report_paths,
        )

    def write_review_artifacts(self, *, task_intent: str | None = None) -> VisualCaseArtifacts:
        """Write proof-pack and manifest artifacts for an existing case directory."""

        proof_pack_path, manifest_path = write_case_visual_artifacts(
            self.case_dir,
            task_intent=task_intent,
        )
        return VisualCaseArtifacts(
            case_dir=self.case_dir,
            autonomous_report_path=self.case_dir / "autonomous_report.json",
            proof_pack_path=proof_pack_path,
            visual_review_manifest_path=manifest_path,
            snapshot_bundle_path=(
                self.case_dir / "snapshot_bundle.json"
                if (self.case_dir / "snapshot_bundle.json").exists()
                else None
            ),
            renderer_report_paths={
                name: self.case_dir / name / "report.json" for name in self.renderer_reports
            },
        )


@dataclass
class VisualSuiteRun:
    """Mutable builder for a downstream visual-harness suite run."""

    suite_name: str
    output_root: str | Path
    metadata: dict[str, Any] = field(default_factory=dict)
    results: list[dict[str, Any]] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)
    suite_proof_pack_path: str | None = None
    visual_review_queue_path: str | None = None

    @property
    def suite_dir(self) -> Path:
        """Return the suite artifact directory."""

        return Path(self.output_root)

    @classmethod
    def from_report_path(cls, suite_report_path: str | Path) -> VisualSuiteRun:
        """Load a suite builder from an existing suite report."""

        path = Path(suite_report_path)
        payload = load_json(path)
        extra = {
            key: value
            for key, value in payload.items()
            if key
            not in {
                "suite_name",
                "output_root",
                "metadata",
                "results",
                "suite_proof_pack_path",
                "visual_review_queue_path",
            }
        }
        return cls(
            suite_name=str(payload.get("suite_name") or path.stem.removeprefix("suite_report_")),
            output_root=payload.get("output_root") or path.parent.as_posix(),
            metadata=dict(payload.get("metadata") or {}),
            results=[dict(item) for item in payload.get("results", ()) if isinstance(item, dict)],
            extra=extra,
            suite_proof_pack_path=payload.get("suite_proof_pack_path"),
            visual_review_queue_path=payload.get("visual_review_queue_path"),
        )

    def add_case(
        self,
        case_run: VisualCaseRun,
        *,
        status: str | None = None,
        report_json: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> VisualSuiteRun:
        """Append one case result entry."""

        case_dir = case_run.case_dir
        entry: dict[str, Any] = {
            "case_id": case_run.case_id,
            "output_dir": case_dir.as_posix(),
            "status": status or _status_from_verdict(case_run.verdict),
            "report_json": report_json or (case_dir / "autonomous_report.json").as_posix(),
        }
        if extra:
            entry.update(dict(extra))
        self.results.append(entry)
        return self

    def add_result(self, result: dict[str, Any]) -> VisualSuiteRun:
        """Append a downstream-owned suite result entry."""

        self.results.append(dict(result))
        return self

    def to_dict(self) -> dict[str, Any]:
        """Serialize the suite report payload."""

        payload = dict(self.extra)
        payload["suite_name"] = self.suite_name
        payload["output_root"] = self.suite_dir.as_posix()
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        payload["results"] = [dict(result) for result in self.results]
        payload.setdefault("total_cases", len(self.results))
        payload.setdefault(
            "pass_count",
            sum(1 for result in self.results if result.get("status") == "pass"),
        )
        payload.setdefault(
            "fail_count",
            sum(1 for result in self.results if result.get("status") == "fail"),
        )
        payload.setdefault(
            "execution_error_count",
            sum(1 for result in self.results if result.get("status") == "execution_error"),
        )
        payload.setdefault(
            "suite_verdict",
            "fail" if payload["fail_count"] or payload["execution_error_count"] else "pass",
        )
        payload["suite_proof_pack_path"] = self.suite_proof_pack_path
        payload["visual_review_queue_path"] = self.visual_review_queue_path
        return payload

    def write_artifacts(
        self,
        *,
        task_intent: str | None = None,
        filename: str | None = None,
    ) -> VisualSuiteArtifacts:
        """Write the suite report, proof pack, and review queue."""

        self.suite_dir.mkdir(parents=True, exist_ok=True)
        report_filename = filename or f"suite_report_{self.suite_name}.json"
        suite_report_path = self.suite_dir / report_filename
        save_json(self.to_dict(), suite_report_path)
        suite_proof_pack_path, queue_path = write_suite_visual_artifacts(
            suite_report_path,
            task_intent=task_intent,
        )
        self.suite_proof_pack_path = suite_proof_pack_path.as_posix()
        self.visual_review_queue_path = queue_path.as_posix()
        save_json(self.to_dict(), suite_report_path)
        return VisualSuiteArtifacts(
            suite_dir=self.suite_dir,
            suite_report_path=suite_report_path,
            suite_proof_pack_path=suite_proof_pack_path,
            visual_review_queue_path=queue_path,
        )

    def write_review_artifacts(
        self,
        suite_report_path: str | Path,
        *,
        task_intent: str | None = None,
    ) -> VisualSuiteArtifacts:
        """Write proof-pack and queue artifacts for an existing suite report."""

        report_path = Path(suite_report_path)
        suite_proof_pack_path, queue_path = write_suite_visual_artifacts(
            report_path,
            task_intent=task_intent,
        )
        payload = load_json(report_path)
        payload["suite_proof_pack_path"] = suite_proof_pack_path.as_posix()
        payload["visual_review_queue_path"] = queue_path.as_posix()
        save_json(payload, report_path)
        self.suite_proof_pack_path = suite_proof_pack_path.as_posix()
        self.visual_review_queue_path = queue_path.as_posix()
        return VisualSuiteArtifacts(
            suite_dir=report_path.parent,
            suite_report_path=report_path,
            suite_proof_pack_path=suite_proof_pack_path,
            visual_review_queue_path=queue_path,
        )


def write_case_visual_artifacts(
    case_dir: str | Path,
    *,
    task_intent: str | None = None,
) -> tuple[Path, Path]:
    """Write a proof pack and current-only static visual-review manifest."""

    root = Path(case_dir)
    proof_pack = build_case_proof_pack(root)
    proof_pack_path = write_case_proof_pack(proof_pack, root / "proof_pack.json")
    manifest_path = write_static_visual_review_manifest(
        proof_pack,
        root / "visual_review_manifest.json",
        task_intent=task_intent or _default_case_task_intent(proof_pack.case_id),
    )
    manifest = load_json(manifest_path)
    validate_visual_review_manifest(manifest, current_root=root)
    return proof_pack_path, manifest_path


def write_suite_visual_artifacts(
    suite_report_path: str | Path,
    *,
    task_intent: str | None = None,
) -> tuple[Path, Path]:
    """Write suite proof-pack and visual review queue artifacts."""

    report_path = Path(suite_report_path)
    suite_proof_pack = build_suite_proof_pack(report_path, task_intent=task_intent)
    suite_proof_pack_path = write_suite_proof_pack(
        suite_proof_pack,
        report_path.parent / "suite_proof_pack.json",
    )
    queue = build_visual_review_queue(suite_proof_pack)
    queue_path = write_visual_review_queue(queue, report_path.parent / "visual_review_queue.json")
    return suite_proof_pack_path, queue_path


def run_visual_suite(
    suite_spec: VisualSuiteSpec,
    *,
    case_runner: Callable[
        [VisualCaseSpec, Path], VisualCaseRun | VisualCaseResult | dict[str, Any]
    ],
    output_root: str | Path,
    options: VisualSuiteOptions | None = None,
) -> VisualSuiteArtifacts:
    """Run a downstream-owned case runner through RoboHarness suite orchestration."""

    suite = collect_visual_suite(
        suite_spec,
        case_runner=case_runner,
        output_root=output_root,
        options=options,
    )
    resolved_options = options or VisualSuiteOptions()
    return suite.write_artifacts(
        task_intent=resolved_options.task_intent,
        filename=resolved_options.suite_report_filename,
    )


def collect_visual_suite(
    suite_spec: VisualSuiteSpec,
    *,
    case_runner: Callable[
        [VisualCaseSpec, Path], VisualCaseRun | VisualCaseResult | dict[str, Any]
    ],
    output_root: str | Path,
    options: VisualSuiteOptions | None = None,
    error_result_builder: Callable[[VisualCaseSpec, Path, Exception], dict[str, Any]] | None = None,
) -> VisualSuiteRun:
    """Collect downstream case results with common suite-loop semantics.

    This is the library-first executor path for projects that must keep their
    own suite report schema.  The downstream runner owns robot/runtime work and
    result-row semantics; RoboHarness owns ordered case iteration, output
    directory selection, error capture, and status accounting through
    :class:`VisualSuiteRun`.
    """

    resolved_options = options or VisualSuiteOptions()
    suite = VisualSuiteRun(
        suite_name=suite_spec.suite_name,
        output_root=output_root,
        metadata=dict(suite_spec.metadata),
    )
    output_dir = Path(output_root)
    for case_spec in suite_spec.cases:
        case_dir = output_dir / case_spec.case_id
        try:
            result = case_runner(case_spec, case_dir)
            _add_visual_case_result(
                suite,
                result,
                task_intent=resolved_options.task_intent,
                write_case_artifacts=resolved_options.write_case_artifacts,
            )
        except Exception as exc:
            if not resolved_options.continue_on_error:
                raise
            if error_result_builder is None:
                error_result = {
                    "case_id": case_spec.case_id,
                    "output_dir": case_dir.as_posix(),
                    "status": "execution_error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            else:
                error_result = error_result_builder(case_spec, case_dir, exc)
            suite.add_result(error_result)
    return suite


def _add_visual_case_result(
    suite: VisualSuiteRun,
    result: VisualCaseRun | VisualCaseResult | dict[str, Any],
    *,
    task_intent: str | None,
    write_case_artifacts: bool,
) -> None:
    if isinstance(result, VisualCaseRun):
        if write_case_artifacts:
            result.write_artifacts(task_intent=task_intent)
        suite.add_case(result)
        return
    if isinstance(result, VisualCaseResult):
        if result.case_run is not None and write_case_artifacts:
            result.case_run.write_artifacts(task_intent=task_intent)
        suite.add_result(result.result)
        return
    suite.add_result(result)


def _coerce_snapshot_bundle(
    snapshot_bundle: SemanticSnapshotBundle | dict[str, Any],
) -> SemanticSnapshotBundle:
    if isinstance(snapshot_bundle, SemanticSnapshotBundle):
        return snapshot_bundle
    return SemanticSnapshotBundle.from_dict(snapshot_bundle)


def _coerce_renderer_report(renderer_report: RendererReport | dict[str, Any]) -> RendererReport:
    if isinstance(renderer_report, RendererReport):
        return renderer_report
    return RendererReport.from_dict(renderer_report)


def _status_from_verdict(verdict: str) -> str:
    if verdict == "pass":
        return "pass"
    if verdict in {"fail", "execution_error"}:
        return verdict
    return "fail"


def _default_case_task_intent(case_id: str) -> str:
    return (
        f"Review {case_id} static visual harness keyframes against the bounded "
        "robot evidence and metric summary."
    )
