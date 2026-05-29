from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from roboharness.approval.visual_review import (
    MANIFEST_SCHEMA_VERSION,
    RECORD_SCHEMA_VERSION,
    VISUAL_REVIEW_SUMMARY_SCHEMA_VERSION,
    VisualReviewValidationError,
    build_visual_review_summary,
    ingest_visual_review_record,
    validate_visual_review_manifest,
    write_visual_review_package,
    write_visual_review_summary,
)


def _write_evidence(root: Path, *paths: str) -> None:
    for relative_path in paths:
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("evidence")


def _manifest(*, mode: str = "regression") -> dict[str, Any]:
    requires_paired = mode != "current_only"
    dimension: dict[str, Any] = {
        "id": "hand_pose",
        "required": True,
        "phase": "lift",
        "evidence_type": "paired_keyframe" if requires_paired else "current_static_keyframe",
        "views": ["front"],
        "current": ["current/lift/front_rgb.png"],
        "metric_fallback": ["grip_center_error_mm"],
        "why_not_metricized": "Scalar metrics do not fully capture hand orientation.",
    }
    if requires_paired:
        dimension["baseline"] = ["baseline/lift/front_rgb.png"]
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "case_id": "case-1",
        "mode": mode,
        "task_intent": "Lift the cube with a plausible grasp.",
        "dimensions": [dimension],
        "metric_summary": {"grip_center_error_mm": 3.0},
        "review_policy": {
            "requires_paired_evidence": requires_paired,
            "allow_automatic_visual_pass": mode == "regression",
        },
    }


def _record(
    *,
    verdict: str = "PASS",
    confidence: str = "medium",
    evidence: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": RECORD_SCHEMA_VERSION,
        "case_id": "case-1",
        "reviewer_context": "agent_visual_review",
        "overall_visual_verdict": verdict,
        "dimensions": [
            {
                "id": "hand_pose",
                "verdict": verdict,
                "confidence": confidence,
                "evidence": evidence
                if evidence is not None
                else ["current/lift/front_rgb.png", "baseline/lift/front_rgb.png"],
                "rationale": "The gripper appears plausibly aligned with the cube.",
            }
        ],
        "needs_human_reasons": [],
    }


def test_write_visual_review_package_emits_manifest_prompt_and_schema(tmp_path: Path) -> None:
    _write_evidence(
        tmp_path,
        "current/lift/front_rgb.png",
        "baseline/lift/front_rgb.png",
    )

    package = write_visual_review_package(tmp_path, _manifest())

    assert package.manifest_path.exists()
    assert package.prompt_path.exists()
    assert package.schema_path.exists()
    assert "Do not infer unseen motion" in package.prompt_path.read_text()
    schema = json.loads(package.schema_path.read_text())
    assert schema["properties"]["schema_version"]["const"] == RECORD_SCHEMA_VERSION


def test_manifest_validation_rejects_missing_metricization_and_path_escape(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    manifest["dimensions"][0].pop("metric_fallback")
    manifest["dimensions"][0].pop("why_not_metricized")
    manifest["dimensions"][0]["current"] = ["../escape.png"]

    with pytest.raises(VisualReviewValidationError) as excinfo:
        validate_visual_review_manifest(manifest, current_root=tmp_path, baseline_root=tmp_path)

    message = str(excinfo.value)
    assert "metric_fallback or why_not_metricized" in message
    assert "escapes its evidence root" in message


def test_ingest_visual_review_record_accepts_paired_pass() -> None:
    result = ingest_visual_review_record(_manifest(), _record())

    assert result.is_valid is True
    assert result.effective_visual_verdict == "PASS"
    assert result.summary["blocking_dimensions"] == []
    assert result.summary["metric_findings"][0]["id"] == "visual.hand_pose"


def test_build_visual_review_summary_persists_effective_verdict(tmp_path: Path) -> None:
    summary = build_visual_review_summary(
        _manifest(),
        _record(),
        manifest_path="case/visual_review_manifest.json",
        record_path="case/visual_review.json",
    )

    assert summary["schema_version"] == VISUAL_REVIEW_SUMMARY_SCHEMA_VERSION
    assert summary["case_id"] == "case-1"
    assert summary["is_valid"] is True
    assert summary["effective_visual_verdict"] == "PASS"
    assert summary["summary"]["manifest_path"] == "case/visual_review_manifest.json"
    assert summary["summary"]["record_path"] == "case/visual_review.json"

    path = write_visual_review_summary(
        _manifest(mode="current_only"),
        _record(evidence=["current/lift/front_rgb.png"]),
        tmp_path / "visual_review_summary.json",
    )
    written = json.loads(path.read_text(encoding="utf-8"))
    assert written["schema_version"] == VISUAL_REVIEW_SUMMARY_SCHEMA_VERSION
    assert written["effective_visual_verdict"] == "NEEDS_HUMAN"
    assert written["summary"]["needs_human_reasons"] == [
        "current_only_review_cannot_auto_pass"
    ]


def test_ingest_visual_review_record_fails_on_declared_dimension_failure() -> None:
    result = ingest_visual_review_record(_manifest(), _record(verdict="FAIL"))

    assert result.effective_visual_verdict == "FAIL"
    assert result.summary["blocking_dimensions"] == ["hand_pose"]


def test_insufficient_required_visual_evidence_escalates_to_human_review() -> None:
    result = ingest_visual_review_record(
        _manifest(),
        _record(verdict="INSUFFICIENT_EVIDENCE", evidence=[]),
    )

    assert result.effective_visual_verdict == "NEEDS_HUMAN"
    assert result.summary["blocking_dimensions"] == ["hand_pose"]
    assert result.summary["needs_human_reasons"] == ["missing_required_evidence"]


def test_explicit_needs_human_dimension_verdict_is_preserved() -> None:
    record = _record(verdict="NEEDS_HUMAN", evidence=[])
    record["needs_human_reasons"] = ["view_conflict"]

    result = ingest_visual_review_record(_manifest(), record)

    assert result.effective_visual_verdict == "NEEDS_HUMAN"
    assert result.summary["blocking_dimensions"] == ["hand_pose"]
    assert result.summary["needs_human_reasons"] == ["view_conflict"]


def test_low_confidence_required_pass_escalates_to_human_review() -> None:
    result = ingest_visual_review_record(_manifest(), _record(confidence="low"))

    assert result.effective_visual_verdict == "NEEDS_HUMAN"
    assert result.summary["blocking_dimensions"] == ["hand_pose"]
    assert result.summary["needs_human_reasons"] == ["low_confidence_high_risk"]


def test_current_only_pass_cannot_auto_approve() -> None:
    record = _record(evidence=["current/lift/front_rgb.png"])

    result = ingest_visual_review_record(_manifest(mode="current_only"), record)

    assert result.effective_visual_verdict == "NEEDS_HUMAN"
    assert result.summary["needs_human_reasons"] == ["current_only_review_cannot_auto_pass"]


def test_invalid_review_record_returns_review_invalid() -> None:
    record = _record(evidence=["current/lift/front_rgb.png", "undeclared/path.png"])

    result = ingest_visual_review_record(_manifest(), record)

    assert result.is_valid is False
    assert result.effective_visual_verdict == "REVIEW_INVALID"
    assert "undeclared paths" in result.summary["validation_errors"][0]
