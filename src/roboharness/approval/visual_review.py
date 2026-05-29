"""Agent visual review package and record validation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from roboharness._utils import save_json
from roboharness.approval.evidence import resolve_evidence_path

MANIFEST_SCHEMA_VERSION = "roboharness_visual_review_manifest/v1"
RECORD_SCHEMA_VERSION = "roboharness_visual_review/v1"
VISUAL_REVIEW_SUMMARY_SCHEMA_VERSION = "roboharness_visual_review_summary/v1"

SUPPORTED_DIMENSIONS = frozenset(
    {
        "robot_posture",
        "hand_pose",
        "object_relative_position",
        "obvious_collision_or_penetration",
        "task_success_visual_check",
    }
)
TEMPORAL_DIMENSIONS = frozenset(
    {
        "trajectory_naturalness",
        "smoothness",
        "contact_sequence",
        "late_sharp_motion",
        "locomotion_quality",
    }
)
MANIFEST_MODES = ("regression", "migration", "current_only")
DIMENSION_VERDICTS = (
    "PASS",
    "FAIL",
    "INSUFFICIENT_EVIDENCE",
    "NEEDS_HUMAN",
    "NOT_APPLICABLE",
)
OVERALL_VISUAL_VERDICTS = (
    "PASS",
    "FAIL",
    "INSUFFICIENT_EVIDENCE",
    "NEEDS_HUMAN",
)
CONFIDENCE_VALUES = ("high", "medium", "low")
HUMAN_REASON_VALUES = (
    "missing_required_evidence",
    "view_conflict",
    "low_confidence_high_risk",
    "baseline_blessing_required",
    "migration_intent_confirmation_required",
    "unsupported_temporal_dimension",
    "current_only_review_cannot_auto_pass",
)

EffectiveVisualVerdict = Literal["PASS", "FAIL", "NEEDS_HUMAN", "REVIEW_INVALID"]


@dataclass(frozen=True)
class VisualReviewPackage:
    """Files prepared for one bounded agent visual review invocation."""

    manifest_path: Path
    prompt_path: Path
    schema_path: Path

    def artifact_paths(self, root: Path) -> dict[str, str]:
        """Return package file paths relative to *root*."""
        return {
            "visual_review_manifest": self.manifest_path.relative_to(root).as_posix(),
            "visual_review_prompt": self.prompt_path.relative_to(root).as_posix(),
            "visual_review_schema": self.schema_path.relative_to(root).as_posix(),
        }


@dataclass(frozen=True)
class VisualReviewResult:
    """Validated visual review aggregation ready for an approval report."""

    effective_visual_verdict: EffectiveVisualVerdict
    summary: dict[str, Any]
    validation_errors: tuple[str, ...] = ()

    @property
    def is_valid(self) -> bool:
        """Whether the reviewer record was valid and trustworthy."""
        return self.effective_visual_verdict != "REVIEW_INVALID"


class VisualReviewValidationError(ValueError):
    """Raised when a visual review manifest or record is invalid."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(errors)
        super().__init__("; ".join(self.errors))


def build_visual_review_summary(
    manifest: Mapping[str, Any],
    record: Mapping[str, Any],
    *,
    manifest_path: str = "visual_review_manifest.json",
    record_path: str = "visual_review.json",
) -> dict[str, Any]:
    """Build a persisted summary from a bounded visual review record."""

    result = ingest_visual_review_record(
        manifest,
        record,
        manifest_path=manifest_path,
        record_path=record_path,
    )
    case_id = manifest.get("case_id") if isinstance(manifest.get("case_id"), str) else ""
    if not case_id and isinstance(record.get("case_id"), str):
        case_id = str(record["case_id"])
    return {
        "schema_version": VISUAL_REVIEW_SUMMARY_SCHEMA_VERSION,
        "case_id": case_id,
        "is_valid": result.is_valid,
        "effective_visual_verdict": result.effective_visual_verdict,
        "summary": dict(result.summary),
    }


def write_visual_review_summary(
    manifest: Mapping[str, Any],
    record: Mapping[str, Any],
    path: str | Path,
    *,
    manifest_path: str = "visual_review_manifest.json",
    record_path: str = "visual_review.json",
) -> Path:
    """Write a persisted summary from a bounded visual review record."""

    output_path = Path(path)
    save_json(
        build_visual_review_summary(
            manifest,
            record,
            manifest_path=manifest_path,
            record_path=record_path,
        ),
        output_path,
    )
    return output_path


def build_visual_review_schema() -> dict[str, Any]:
    """Return the JSON schema given to the visual reviewer."""
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Roboharness Agent Visual Review Record",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema_version",
            "case_id",
            "reviewer_context",
            "overall_visual_verdict",
            "dimensions",
            "needs_human_reasons",
        ],
        "properties": {
            "schema_version": {"const": RECORD_SCHEMA_VERSION},
            "case_id": {"type": "string", "minLength": 1},
            "reviewer_context": {"type": "string", "minLength": 1},
            "overall_visual_verdict": {"enum": list(OVERALL_VISUAL_VERDICTS)},
            "dimensions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["id", "verdict", "confidence", "evidence", "rationale"],
                    "properties": {
                        "id": {"type": "string"},
                        "verdict": {"enum": list(DIMENSION_VERDICTS)},
                        "confidence": {"enum": list(CONFIDENCE_VALUES)},
                        "evidence": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "rationale": {"type": "string", "minLength": 1},
                    },
                },
            },
            "needs_human_reasons": {
                "type": "array",
                "items": {"enum": list(HUMAN_REASON_VALUES)},
            },
        },
    }


def build_visual_review_prompt(manifest: Mapping[str, Any]) -> str:
    """Build the bounded prompt that accompanies a visual review manifest."""
    dimension_lines = []
    for dimension in _manifest_dimensions(manifest):
        views = ", ".join(_string_list(dimension.get("views"), field_name="views"))
        required = "required" if dimension.get("required") is True else "optional"
        dimension_lines.append(
            f"- {dimension['id']} ({required}) at phase `{dimension['phase']}`; views: {views}"
        )

    return "\n".join(
        [
            "# Agent Visual Review",
            "",
            "Review only the dimensions declared in `visual_review_manifest.json`.",
            "Return only JSON matching `visual_review_schema.json`.",
            "",
            "Rules:",
            "- Do not invent task criteria outside the manifest.",
            "- Do not infer unseen motion from still frames.",
            "- Do not use implementation intent to fill evidence gaps.",
            "- Return `INSUFFICIENT_EVIDENCE` when evidence is inadequate.",
            "- Return `NEEDS_HUMAN` when semantic approval is outside your authority.",
            "- Keep each dimension rationale to one sentence.",
            "",
            f"Case: `{manifest.get('case_id', '')}`",
            f"Mode: `{manifest.get('mode', '')}`",
            f"Task intent: {manifest.get('task_intent', '')}",
            "",
            "Dimensions:",
            *dimension_lines,
            "",
        ]
    )


def write_visual_review_package(
    package_dir: Path,
    manifest: Mapping[str, Any],
    *,
    current_root: Path | None = None,
    baseline_root: Path | None = None,
) -> VisualReviewPackage:
    """Validate and write manifest, prompt, and schema files for review."""
    package_dir.mkdir(parents=True, exist_ok=True)
    manifest_dict = dict(manifest)
    validate_visual_review_manifest(
        manifest_dict,
        current_root=current_root or package_dir,
        baseline_root=baseline_root or current_root or package_dir,
    )

    package = VisualReviewPackage(
        manifest_path=package_dir / "visual_review_manifest.json",
        prompt_path=package_dir / "visual_review_prompt.md",
        schema_path=package_dir / "visual_review_schema.json",
    )
    save_json(manifest_dict, package.manifest_path)
    package.prompt_path.write_text(build_visual_review_prompt(manifest_dict), encoding="utf-8")
    save_json(build_visual_review_schema(), package.schema_path)
    return package


def validate_visual_review_manifest(
    manifest: Mapping[str, Any],
    *,
    current_root: Path | None = None,
    baseline_root: Path | None = None,
) -> None:
    """Validate the reviewer boundary before an agent invocation."""
    errors: list[str] = []

    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        errors.append(
            "schema_version must be "
            f"{MANIFEST_SCHEMA_VERSION!r}, got {manifest.get('schema_version')!r}"
        )
    if not isinstance(manifest.get("case_id"), str) or not manifest.get("case_id"):
        errors.append("case_id must be a non-empty string")
    mode = manifest.get("mode")
    if mode not in MANIFEST_MODES:
        errors.append(f"mode must be one of {list(MANIFEST_MODES)!r}")
    if not isinstance(manifest.get("task_intent"), str) or not manifest.get("task_intent"):
        errors.append("task_intent must be a non-empty string")
    if not isinstance(manifest.get("metric_summary"), Mapping):
        errors.append("metric_summary must be an object")

    review_policy = manifest.get("review_policy")
    if not isinstance(review_policy, Mapping):
        errors.append("review_policy must be an object")
        review_policy = {}

    raw_dimensions = manifest.get("dimensions")
    if not isinstance(raw_dimensions, list) or not raw_dimensions:
        errors.append("dimensions must be a non-empty list")
        raw_dimensions = []

    requires_baseline = _requires_baseline(mode, review_policy)
    seen_ids: set[str] = set()
    for index, raw_dimension in enumerate(raw_dimensions):
        if not isinstance(raw_dimension, Mapping):
            errors.append(f"dimensions[{index}] must be an object")
            continue
        errors.extend(
            _validate_manifest_dimension(
                raw_dimension,
                index=index,
                current_root=current_root,
                baseline_root=baseline_root,
                requires_baseline=requires_baseline,
                review_policy=review_policy,
            )
        )
        dimension_id = raw_dimension.get("id")
        if isinstance(dimension_id, str):
            if dimension_id in seen_ids:
                errors.append(f"dimensions[{index}].id duplicates {dimension_id!r}")
            seen_ids.add(dimension_id)

    if errors:
        raise VisualReviewValidationError(errors)


def validate_visual_review_record(
    manifest: Mapping[str, Any],
    record: Mapping[str, Any],
) -> None:
    """Validate a reviewer record against its manifest boundary."""
    errors: list[str] = []
    try:
        validate_visual_review_manifest(manifest)
    except VisualReviewValidationError as exc:
        errors.extend(f"manifest: {error}" for error in exc.errors)

    if record.get("schema_version") != RECORD_SCHEMA_VERSION:
        errors.append(
            f"schema_version must be {RECORD_SCHEMA_VERSION!r}, "
            f"got {record.get('schema_version')!r}"
        )
    if record.get("case_id") != manifest.get("case_id"):
        errors.append("case_id must match the visual review manifest")
    if not isinstance(record.get("reviewer_context"), str) or not record.get("reviewer_context"):
        errors.append("reviewer_context must be a non-empty string")
    if record.get("overall_visual_verdict") not in OVERALL_VISUAL_VERDICTS:
        errors.append("overall_visual_verdict has an illegal value")

    needs_human_reasons = record.get("needs_human_reasons")
    if not isinstance(needs_human_reasons, list) or any(
        reason not in HUMAN_REASON_VALUES for reason in needs_human_reasons
    ):
        errors.append("needs_human_reasons must use the fixed taxonomy")

    declared_dimensions = _declared_dimension_map(manifest)
    required_dimension_ids = {
        dimension_id
        for dimension_id, dimension in declared_dimensions.items()
        if dimension.get("required") is True
    }
    seen_record_ids: set[str] = set()
    raw_dimensions = record.get("dimensions")
    if not isinstance(raw_dimensions, list):
        errors.append("dimensions must be a list")
        raw_dimensions = []

    for index, raw_dimension in enumerate(raw_dimensions):
        if not isinstance(raw_dimension, Mapping):
            errors.append(f"dimensions[{index}] must be an object")
            continue
        dimension_id = raw_dimension.get("id")
        if not isinstance(dimension_id, str):
            errors.append(f"dimensions[{index}].id must be a string")
            continue
        if dimension_id not in declared_dimensions:
            errors.append(f"dimensions[{index}].id {dimension_id!r} is not declared")
            continue
        if dimension_id in seen_record_ids:
            errors.append(f"dimensions[{index}].id duplicates {dimension_id!r}")
        seen_record_ids.add(dimension_id)
        errors.extend(
            _validate_record_dimension(
                raw_dimension,
                manifest_dimension=declared_dimensions[dimension_id],
                index=index,
            )
        )

    missing_required = required_dimension_ids - seen_record_ids
    if missing_required:
        errors.append(f"missing required dimension reviews: {sorted(missing_required)!r}")

    if errors:
        raise VisualReviewValidationError(errors)


def ingest_visual_review_record(
    manifest: Mapping[str, Any],
    record: Mapping[str, Any],
    *,
    manifest_path: str = "visual_review_manifest.json",
    record_path: str = "visual_review.json",
) -> VisualReviewResult:
    """Validate and aggregate a reviewer record for approval."""
    try:
        validate_visual_review_record(manifest, record)
    except VisualReviewValidationError as exc:
        summary = {
            "manifest_path": manifest_path,
            "record_path": record_path,
            "overall_visual_verdict": "REVIEW_INVALID",
            "effective_visual_verdict": "REVIEW_INVALID",
            "blocking_dimensions": [],
            "needs_human_reasons": [],
            "metric_findings": [],
            "validation_errors": list(exc.errors),
        }
        return VisualReviewResult(
            effective_visual_verdict="REVIEW_INVALID",
            summary=summary,
            validation_errors=exc.errors,
        )

    summary = _aggregate_valid_record(
        manifest,
        record,
        manifest_path=manifest_path,
        record_path=record_path,
    )
    return VisualReviewResult(
        effective_visual_verdict=cast(
            "EffectiveVisualVerdict", summary["effective_visual_verdict"]
        ),
        summary=summary,
    )


def _validate_manifest_dimension(
    dimension: Mapping[str, Any],
    *,
    index: int,
    current_root: Path | None,
    baseline_root: Path | None,
    requires_baseline: bool,
    review_policy: Mapping[str, Any],
) -> list[str]:
    errors: list[str] = []
    prefix = f"dimensions[{index}]"
    dimension_id = dimension.get("id")
    if not isinstance(dimension_id, str) or not dimension_id:
        errors.append(f"{prefix}.id must be a non-empty string")
    elif dimension_id not in SUPPORTED_DIMENSIONS and dimension_id not in TEMPORAL_DIMENSIONS:
        errors.append(f"{prefix}.id {dimension_id!r} is not supported by v1")
    elif dimension_id in TEMPORAL_DIMENSIONS and (
        dimension.get("required") is True or review_policy.get("allow_automatic_visual_pass")
    ):
        errors.append(
            f"{prefix}.id {dimension_id!r} is temporal and cannot participate "
            "in automatic v1 visual pass"
        )

    if not isinstance(dimension.get("required"), bool):
        errors.append(f"{prefix}.required must be a boolean")
    if not isinstance(dimension.get("phase"), str) or not dimension.get("phase"):
        errors.append(f"{prefix}.phase must be a non-empty string")
    if not isinstance(dimension.get("evidence_type"), str) or not dimension.get("evidence_type"):
        errors.append(f"{prefix}.evidence_type must be a non-empty string")

    views = _optional_string_list(dimension.get("views"))
    if views is None or not views:
        errors.append(f"{prefix}.views must be a non-empty string list")

    current_paths = _optional_string_list(dimension.get("current"))
    if current_paths is None or not current_paths:
        errors.append(f"{prefix}.current must be a non-empty string list")
    elif current_root is not None:
        errors.extend(_validate_declared_paths(current_root, current_paths, f"{prefix}.current"))

    baseline_paths = _optional_string_list(dimension.get("baseline"))
    if requires_baseline:
        if baseline_paths is None or not baseline_paths:
            errors.append(f"{prefix}.baseline is required for paired review")
        elif baseline_root is not None:
            errors.extend(
                _validate_declared_paths(baseline_root, baseline_paths, f"{prefix}.baseline")
            )
    elif baseline_paths is not None and baseline_root is not None:
        errors.extend(_validate_declared_paths(baseline_root, baseline_paths, f"{prefix}.baseline"))

    participates = dimension.get("participates_in_verdict", True)
    if not isinstance(participates, bool):
        errors.append(f"{prefix}.participates_in_verdict must be a boolean when present")
        participates = True
    has_metric_fallback = bool(_optional_string_list(dimension.get("metric_fallback")))
    has_rationale = isinstance(dimension.get("why_not_metricized"), str) and bool(
        str(dimension.get("why_not_metricized")).strip()
    )
    if (dimension.get("required") is True or participates) and not (
        has_metric_fallback or has_rationale
    ):
        errors.append(f"{prefix} must declare metric_fallback or why_not_metricized before review")

    return errors


def _validate_record_dimension(
    dimension: Mapping[str, Any],
    *,
    manifest_dimension: Mapping[str, Any],
    index: int,
) -> list[str]:
    errors: list[str] = []
    prefix = f"dimensions[{index}]"
    if dimension.get("verdict") not in DIMENSION_VERDICTS:
        errors.append(f"{prefix}.verdict has an illegal value")
    if dimension.get("confidence") not in CONFIDENCE_VALUES:
        errors.append(f"{prefix}.confidence has an illegal value")
    if not isinstance(dimension.get("rationale"), str) or not dimension.get("rationale"):
        errors.append(f"{prefix}.rationale must be a non-empty string")

    evidence = _optional_string_list(dimension.get("evidence"))
    if evidence is None:
        errors.append(f"{prefix}.evidence must be a string list")
        evidence = []

    declared_evidence = set(_string_list(manifest_dimension.get("current"), field_name="current"))
    declared_evidence.update(_optional_string_list(manifest_dimension.get("baseline")) or [])
    outside_manifest = [path for path in evidence if path not in declared_evidence]
    if outside_manifest:
        errors.append(f"{prefix}.evidence references undeclared paths: {outside_manifest!r}")
    if dimension.get("verdict") in {"PASS", "FAIL"} and not evidence:
        errors.append(f"{prefix}.evidence is required for PASS or FAIL verdicts")
    return errors


def _aggregate_valid_record(
    manifest: Mapping[str, Any],
    record: Mapping[str, Any],
    *,
    manifest_path: str,
    record_path: str,
) -> dict[str, Any]:
    declared_dimensions = _declared_dimension_map(manifest)
    record_dimensions = {
        str(dimension["id"]): dimension for dimension in _record_dimensions(record)
    }
    effective: EffectiveVisualVerdict = _effective_from_overall(record["overall_visual_verdict"])
    blocking_dimensions: list[str] = []
    needs_human_reasons = set(_string_list(record.get("needs_human_reasons"), field_name="reasons"))
    metric_findings: list[dict[str, Any]] = []

    for dimension_id, manifest_dimension in declared_dimensions.items():
        record_dimension = record_dimensions.get(dimension_id)
        if record_dimension is None:
            continue
        participates = bool(manifest_dimension.get("participates_in_verdict", True))
        required = manifest_dimension.get("required") is True
        verdict = str(record_dimension["verdict"])
        confidence = str(record_dimension["confidence"])
        if participates and verdict == "FAIL":
            effective = "FAIL"
            blocking_dimensions.append(dimension_id)
        elif participates and verdict in {"INSUFFICIENT_EVIDENCE", "NEEDS_HUMAN"}:
            if effective != "FAIL":
                effective = "NEEDS_HUMAN"
            blocking_dimensions.append(dimension_id)
            if required and verdict == "INSUFFICIENT_EVIDENCE":
                needs_human_reasons.add("missing_required_evidence")
        elif required and verdict == "PASS" and confidence == "low":
            if effective != "FAIL":
                effective = "NEEDS_HUMAN"
            blocking_dimensions.append(dimension_id)
            needs_human_reasons.add("low_confidence_high_risk")

        metric_findings.append(
            {
                "id": f"visual.{dimension_id}",
                "dimension_id": dimension_id,
                "phase": manifest_dimension["phase"],
                "verdict": verdict,
                "confidence": confidence,
                "metric_fallback": list(
                    _optional_string_list(manifest_dimension.get("metric_fallback")) or []
                ),
            }
        )

    mode = manifest.get("mode")
    if mode == "current_only" and effective == "PASS":
        effective = "NEEDS_HUMAN"
        needs_human_reasons.add("current_only_review_cannot_auto_pass")
    if mode == "migration" and effective == "PASS":
        effective = "NEEDS_HUMAN"
        needs_human_reasons.add("baseline_blessing_required")

    return {
        "manifest_path": manifest_path,
        "record_path": record_path,
        "overall_visual_verdict": record["overall_visual_verdict"],
        "effective_visual_verdict": effective,
        "blocking_dimensions": _dedupe(blocking_dimensions),
        "needs_human_reasons": _ordered_human_reasons(needs_human_reasons),
        "metric_findings": metric_findings,
    }


def _effective_from_overall(overall: str) -> EffectiveVisualVerdict:
    if overall == "FAIL":
        return "FAIL"
    if overall in {"INSUFFICIENT_EVIDENCE", "NEEDS_HUMAN"}:
        return "NEEDS_HUMAN"
    return "PASS"


def _requires_baseline(mode: Any, review_policy: Mapping[str, Any]) -> bool:
    explicit = review_policy.get("requires_paired_evidence")
    if isinstance(explicit, bool):
        return explicit
    return mode in {"regression", "migration"}


def _validate_declared_paths(root: Path, paths: Sequence[str], field_name: str) -> list[str]:
    errors: list[str] = []
    for relative_path in paths:
        resolved = resolve_evidence_path(root, relative_path)
        if resolved is None:
            errors.append(f"{field_name} path escapes its evidence root: {relative_path!r}")
        elif not resolved.exists():
            errors.append(f"{field_name} path does not exist: {relative_path!r}")
    return errors


def _declared_dimension_map(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        str(dimension["id"]): dimension
        for dimension in _manifest_dimensions(manifest)
        if isinstance(dimension.get("id"), str)
    }


def _manifest_dimensions(manifest: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    dimensions = manifest.get("dimensions")
    if not isinstance(dimensions, list):
        return []
    return [dimension for dimension in dimensions if isinstance(dimension, Mapping)]


def _record_dimensions(record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    dimensions = record.get("dimensions")
    if not isinstance(dimensions, list):
        return []
    return [dimension for dimension in dimensions if isinstance(dimension, Mapping)]


def _optional_string_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        return None
    return list(value)


def _string_list(value: Any, *, field_name: str) -> list[str]:
    result = _optional_string_list(value)
    if result is None:
        raise VisualReviewValidationError([f"{field_name} must be a string list"])
    return result


def _ordered_human_reasons(reasons: set[str]) -> list[str]:
    return [reason for reason in HUMAN_REASON_VALUES if reason in reasons]


def _dedupe(values: Sequence[str]) -> list[str]:
    deduped: list[str] = []
    for value in values:
        if value not in deduped:
            deduped.append(value)
    return deduped
