"""Tests for Python-authored harness contract generation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from roboharness.contract import (
    ApprovalPolicy,
    EvidenceBoundary,
    EvidenceReference,
    HarnessContract,
    HarnessWorkflow,
    MetricGate,
    SemanticPhase,
    ValidationCommand,
    VisualReviewDimension,
    check_project_harness_skill,
    generate_project_harness_skill,
    load_contract_from_file,
    normalize_contract,
)


def _sample_contract() -> HarnessContract:
    return HarnessContract(
        project_slug="sample-bot",
        name="Sample Bot",
        version="1.0",
        description="Review sample robot behavior against approved evidence boundaries.",
        phases=(
            SemanticPhase(
                id="approach",
                label="Approach",
                description="Move toward the target.",
                cameras=("front",),
            ),
            SemanticPhase(
                id="lift",
                label="Lift",
                description="Lift the object.",
                cameras=("front", "side"),
            ),
        ),
        metric_gates=(
            MetricGate(
                id="lift_height",
                metric="cube_height_mm",
                operator="gt",
                threshold=5.0,
                phase="lift",
                evidence=(
                    EvidenceReference(
                        phase="lift",
                        view="front",
                        boundary="trial",
                        path="lift/front_rgb.png",
                    ),
                ),
            ),
        ),
        visual_review_dimensions=(
            VisualReviewDimension(
                id="object_pose",
                label="Object Pose",
                phase="lift",
                views=("front", "side"),
                description="Object should be lifted like the baseline.",
                metric_fallback=("cube_height_mm",),
                evidence_boundary="trial",
            ),
        ),
        evidence_boundaries=(
            EvidenceBoundary(
                id="trial",
                root="outputs",
                description="Generated trial outputs.",
                allowed_patterns=("**/*.json", "**/*_rgb.png"),
            ),
        ),
        approval_policy=ApprovalPolicy(out_of_scope_request="draft_scope_brief"),
        validation_commands=(
            ValidationCommand(
                id="focused_tests",
                command="python -m pytest tests/unit/contract",
                description="Run focused contract tests.",
            ),
        ),
        workflows=(
            HarnessWorkflow(
                id="regression_review",
                label="Regression Review",
                description="Review the normal regression path.",
                phases=("approach", "lift"),
                metric_gates=("lift_height",),
                visual_dimensions=("object_pose",),
                validation_commands=("focused_tests",),
            ),
            HarnessWorkflow(
                id="release_review",
                label="Release Review",
                description="Review release-facing checks in the same project skill.",
                validation_commands=("focused_tests",),
            ),
        ),
    )


def test_normalize_contract_preserves_authoritative_contract_surface() -> None:
    snapshot = normalize_contract(_sample_contract())

    assert snapshot["schema_version"] == "roboharness_harness_contract/v1"
    assert snapshot["project_slug"] == "sample-bot"
    assert [workflow["id"] for workflow in snapshot["workflows"]] == [
        "regression_review",
        "release_review",
    ]
    assert snapshot["approval_policy"]["out_of_scope_request"] == "draft_scope_brief"


def test_generate_project_harness_skill_writes_drift_checked_artifacts(tmp_path: Path) -> None:
    contract = _sample_contract()

    result = generate_project_harness_skill(contract, tmp_path)

    assert (tmp_path / "SKILL.md").exists()
    assert (tmp_path / "contract.snapshot.json").exists()
    assert (tmp_path / "schemas" / "harness-contract.schema.json").exists()
    assert (tmp_path / "scope-brief-template.md").exists()
    assert (tmp_path / "stubs" / "run-validation.py").exists()
    assert result.snapshot_sha256

    skill_text = (tmp_path / "SKILL.md").read_text()
    assert "`contract.py` is the authority" in skill_text
    assert "Regression Review" in skill_text
    assert "Release Review" in skill_text

    manifest = json.loads((tmp_path / ".generated-manifest.json").read_text())
    assert manifest["source"] == "contract.py"
    assert {entry["path"] for entry in manifest["files"]} == {
        "README.md",
        "SKILL.md",
        "contract.snapshot.json",
        "schemas/generated-artifacts.schema.json",
        "schemas/harness-contract.schema.json",
        "scope-brief-template.md",
        "stubs/run-validation.py",
    }
    assert check_project_harness_skill(contract, tmp_path).ok


def test_check_project_harness_skill_detects_manual_edits(tmp_path: Path) -> None:
    contract = _sample_contract()
    generate_project_harness_skill(contract, tmp_path)
    (tmp_path / "SKILL.md").write_text("manual edit\n")

    report = check_project_harness_skill(contract, tmp_path)

    assert not report.ok
    assert report.changed == ("SKILL.md",)


def test_load_contract_from_file_accepts_contract_variable_without_pycache(tmp_path: Path) -> None:
    contract_path = tmp_path / "contract.py"
    contract_path.write_text(
        """
from roboharness.contract import HarnessContract, HarnessWorkflow, SemanticPhase

CONTRACT = HarnessContract(
    project_slug="tmp-bot",
    name="Tmp Bot",
    version="0.1",
    description="Temporary contract.",
    phases=(SemanticPhase(id="phase", label="Phase", description="Phase."),),
    workflows=(HarnessWorkflow(id="workflow", label="Workflow", description="Workflow."),),
)
"""
    )

    contract = load_contract_from_file(contract_path)

    assert contract.project_slug == "tmp-bot"
    assert not (tmp_path / "__pycache__").exists()


def test_contract_validation_rejects_unknown_workflow_reference() -> None:
    contract = HarnessContract(
        project_slug="sample-bot",
        name="Sample Bot",
        version="1.0",
        description="Broken contract.",
        phases=(SemanticPhase(id="phase", label="Phase", description="Phase."),),
        workflows=(
            HarnessWorkflow(
                id="workflow",
                label="Workflow",
                description="Workflow.",
                phases=("missing",),
            ),
        ),
    )

    with pytest.raises(ValueError, match="unknown id 'missing'"):
        normalize_contract(contract)
