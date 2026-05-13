"""Tests for the MuJoCo alarmed grasp-loop helper pipeline."""

from __future__ import annotations

import base64
import copy
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from examples._mujoco_grasp_fixture import (
    MUJOCO_GRASP_CAMERAS,
    MUJOCO_GRASP_PHASE_ORDER,
    MUJOCO_GRASP_PRIMARY_VIEWS,
    build_grasp_phases,
    build_grasp_protocol,
)
from examples._mujoco_grasp_wedge import (
    BASELINE_VISUAL_ROOT,
    KNOWN_BAD_VISUAL_ROOT,
    ContractCompileError,
    available_contract_presets,
    build_alarms,
    build_approval_report,
    build_autonomous_report,
    build_contract_from_preset,
    build_default_contract,
    build_phase_manifest,
    build_summary_html,
    compile_contract,
    evaluate_autonomous_report,
    load_blessed_baseline,
    resolve_evidence_pairs,
    validate_contract,
    write_artifact_pack,
)
from roboharness.evaluate.result import Verdict

_ONE_PIXEL_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+yv3sAAAAASUVORK5CYII="
)


def _build_known_bad_contract() -> tuple[dict[str, Any], Any, list[Any], Any]:
    baseline = load_blessed_baseline()
    current = copy.deepcopy(baseline["snapshot_metrics"])
    current["approach"]["grip_center_error_mm"] += 25.0

    report = build_autonomous_report(
        snapshot_metrics=current,
        baseline_report=baseline,
        baseline_source="fixture",
    )
    result = evaluate_autonomous_report(report)
    alarms = build_alarms(report, result)
    manifest = build_phase_manifest(report, result, alarms)
    return report, result, alarms, manifest


def _build_ambiguous_contract() -> tuple[dict[str, Any], Any, list[Any], Any]:
    baseline = load_blessed_baseline()
    current = copy.deepcopy(baseline["snapshot_metrics"])
    for phase_metrics in current.values():
        phase_metrics["max_abs_qvel"] = 60.0

    report = build_autonomous_report(
        snapshot_metrics=current,
        baseline_report=baseline,
        baseline_source="fixture",
    )
    result = evaluate_autonomous_report(report)
    alarms = build_alarms(report, result)
    manifest = build_phase_manifest(report, result, alarms)
    return report, result, alarms, manifest


def _build_review_bundle() -> tuple[dict[str, Any], dict[str, Any], Any, list[Any], Any, list[Any]]:
    report, result, alarms, manifest = _build_known_bad_contract()
    contract = build_default_contract(baseline_source="fixture")
    evidence_pairs = resolve_evidence_pairs(
        trial_dir=KNOWN_BAD_VISUAL_ROOT,
        baseline_visual_root=BASELINE_VISUAL_ROOT,
        manifest=manifest,
        report=report,
    )
    approval_report = build_approval_report(
        contract=contract,
        report=report,
        evaluation_result=result,
        manifest=manifest,
        evidence_pairs=evidence_pairs,
    )
    return contract, approval_report, report, result, alarms, evidence_pairs


def _write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_ONE_PIXEL_PNG)


def test_build_grasp_fixture_keeps_locked_phase_order_and_cameras() -> None:
    phases = build_grasp_phases()
    protocol = build_grasp_protocol()

    assert list(phases) == MUJOCO_GRASP_PHASE_ORDER
    assert phases["plan"] == []
    assert protocol.phase_names() == MUJOCO_GRASP_PHASE_ORDER
    for phase in protocol.phases:
        assert phase.cameras == MUJOCO_GRASP_CAMERAS


def test_load_blessed_baseline_has_expected_contract() -> None:
    baseline = load_blessed_baseline()

    assert baseline["task"] == "mujoco_grasp"
    assert baseline["phase_order"] == MUJOCO_GRASP_PHASE_ORDER
    assert baseline["summary_metrics"]["loop_runtime_s"] > 0.0
    assert baseline["snapshot_metrics"]["approach"]["grip_center_error_mm"] > 0.0


def test_default_contract_is_valid_and_grounded() -> None:
    contract = build_default_contract(baseline_source="fixture")

    validate_contract(contract)

    assert contract["schema_version"] == "roboharness_contract/v1"
    assert contract["contract_id"] == "mujoco-grasp-regression-v1"
    assert contract["mode"] == "regression"
    assert contract["cases"]["source"] == "deterministic_mujoco_grasp"
    assert contract["approval_policy"]["surface_changed_cases_only"] is True
    assert contract["rules"]
    assert all(rule["judge"] == "metric" for rule in contract["rules"])
    assert all(rule["evidence_at"] for rule in contract["rules"])


def test_available_contract_presets_include_regression_and_migration() -> None:
    assert available_contract_presets() == (
        "mujoco_regression_v1",
        "mujoco_migration_guarded_v1",
    )


def test_build_contract_from_migration_preset_is_grounded() -> None:
    contract = build_contract_from_preset(
        baseline_source="fixture",
        contract_preset="mujoco_migration_guarded_v1",
    )

    validate_contract(contract)

    assert contract["contract_id"] == "mujoco-grasp-migration-guarded-v1"
    assert contract["contract_preset"] == "mujoco_migration_guarded_v1"
    assert contract["mode"] == "migration"


def test_compile_contract_prompt_selects_regression_preset() -> None:
    contract = compile_contract(
        baseline_source="fixture",
        contract_prompt="keep regression mode with no behavior change against the baseline",
    )

    assert contract["contract_preset"] == "mujoco_regression_v1"
    assert contract["mode"] == "regression"
    assert contract["source_prompt"].startswith("keep regression mode")


def test_compile_contract_prompt_selects_migration_preset() -> None:
    contract = compile_contract(
        baseline_source="fixture",
        contract_prompt="treat this as migration mode and require manual blessing after review",
    )

    assert contract["contract_preset"] == "mujoco_migration_guarded_v1"
    assert contract["mode"] == "migration"


def test_compile_contract_prompt_fails_closed_for_unsupported_visual_authoring() -> None:
    with pytest.raises(ContractCompileError) as excinfo:
        compile_contract(
            baseline_source="fixture",
            contract_prompt="switch to a top-down ball grasp with palm-down visual goals",
        )

    assert "does not ground safely" in excinfo.value.envelope.cause


def test_validate_contract_rejects_missing_grounding() -> None:
    contract = build_default_contract(baseline_source="fixture")
    broken_rule = copy.deepcopy(contract["rules"][0])
    broken_rule["evidence_at"] = []
    contract["rules"][0] = broken_rule

    with pytest.raises(ContractCompileError) as excinfo:
        validate_contract(contract)

    envelope = excinfo.value.envelope.to_dict()
    assert envelope["problem"] == "Contract blocked."
    assert "evidence_at" in envelope["cause"]
    assert envelope["recoverable"] is True
    assert envelope["next_action"] == "Fix contract"


def test_validate_contract_rejects_unknown_phase_grounding() -> None:
    contract = build_default_contract(baseline_source="fixture")
    contract["rules"][0]["evidence_at"][0]["phase"] = "bogus_phase"

    with pytest.raises(ContractCompileError) as excinfo:
        validate_contract(contract)

    assert "unsupported phase" in excinfo.value.envelope.cause


def test_evaluation_passes_when_current_run_matches_blessed_baseline() -> None:
    baseline = load_blessed_baseline()

    report = build_autonomous_report(
        snapshot_metrics=copy.deepcopy(baseline["snapshot_metrics"]),
        baseline_report=baseline,
        baseline_source="fixture",
    )
    result = evaluate_autonomous_report(report)

    assert result.verdict is Verdict.PASS
    assert result.failed == []


def test_custom_contract_threshold_drives_evaluator_and_review_output() -> None:
    baseline = load_blessed_baseline()
    contract = build_default_contract(baseline_source="fixture")
    for rule in contract["rules"]:
        if rule["id"] == "all:loop_runtime_s":
            rule["pass_if"]["value"] = 0.0
            break
    validate_contract(contract)

    report = build_autonomous_report(
        snapshot_metrics=copy.deepcopy(baseline["snapshot_metrics"]),
        baseline_report=baseline,
        baseline_source="fixture",
    )
    result = evaluate_autonomous_report(report, contract=contract)
    alarms = build_alarms(report, result)
    manifest = build_phase_manifest(report, result, alarms)
    approval_report = build_approval_report(
        contract=contract,
        report=report,
        evaluation_result=result,
        manifest=manifest,
        evidence_pairs=[],
    )

    assert result.verdict is Verdict.DEGRADED
    assert approval_report["overall_verdict"] == "FAIL"
    assert approval_report["summary"]["cases_surfaced"] == 1
    assert approval_report["summary"]["cases_suppressed"] == 0
    assert approval_report["surfaced_cases"][0]["rules"]["failed"] == ["all:loop_runtime_s"]
    assert approval_report["user_action"]["needs_review"] is True


def test_manifest_localizes_first_regression_to_approach() -> None:
    _report, result, alarms, manifest = _build_known_bad_contract()

    assert result.verdict is Verdict.FAIL
    assert manifest.failed_phase_id == "approach"
    assert manifest.failed_phase == "approach"
    assert manifest.primary_views == ["side", "top"]
    assert manifest.rerun_hint == "restore:pre_grasp"
    assert "approach/side_rgb.png" in manifest.evidence_paths
    assert any(
        alarm.metric == "grip_center_error_mm_abs_delta" and alarm.status == "alarm"
        for alarm in alarms
    )


def test_write_artifact_pack_skips_report_html_when_report_not_generated(tmp_path: Path) -> None:
    baseline = load_blessed_baseline()
    contract = build_default_contract(baseline_source="fixture")
    report = build_autonomous_report(
        snapshot_metrics=copy.deepcopy(baseline["snapshot_metrics"]),
        baseline_report=baseline,
        baseline_source="fixture",
    )
    result = evaluate_autonomous_report(report, report_path="trial/autonomous_report.json")
    alarms = build_alarms(report, result)
    manifest = build_phase_manifest(report, result, alarms)
    approval_report = build_approval_report(
        contract=contract,
        report=report,
        evaluation_result=result,
        manifest=manifest,
        evidence_pairs=[],
    )
    trial_dir = tmp_path / "trial_001"
    trial_dir.mkdir()

    write_artifact_pack(
        trial_dir=trial_dir,
        contract=contract,
        approval_report=approval_report,
        report=report,
        evaluation_result=result,
        alarms=alarms,
        manifest=manifest,
        report_generated=False,
    )

    autonomous_report = json.loads((trial_dir / "autonomous_report.json").read_text())
    compiled_contract = json.loads((trial_dir / "contract.json").read_text())
    approval_json = json.loads((trial_dir / "approval_report.json").read_text())
    alarms_report = json.loads((trial_dir / "alarms.json").read_text())
    phase_manifest = json.loads((trial_dir / "phase_manifest.json").read_text())

    assert "report_html" not in autonomous_report["artifacts"]
    assert autonomous_report["artifacts"]["contract"] == "contract.json"
    assert autonomous_report["artifacts"]["approval_report"] == "approval_report.json"
    assert compiled_contract["contract_id"] == "mujoco-grasp-regression-v1"
    assert approval_json["overall_verdict"] == "PASS"
    assert approval_json["summary"]["cases_surfaced"] == 0
    assert alarms_report["verdict"] == "pass"
    assert phase_manifest["failed_phase_id"] is None


def test_approval_report_surfaces_the_first_regression_case() -> None:
    contract, approval_report, _report, result, _alarms, evidence_pairs = _build_review_bundle()

    assert contract["mode"] == "regression"
    assert result.verdict is Verdict.FAIL
    assert approval_report["overall_verdict"] == "FAIL"
    assert approval_report["run_state"] == "review_ready_surfaced"
    assert approval_report["summary"] == {
        "cases_total": 1,
        "cases_surfaced": 1,
        "cases_suppressed": 0,
        "cases_unchanged": 0,
        "reruns": 1,
    }
    assert approval_report["surfaced_cases"][0]["case_id"] == "deterministic_mujoco_grasp"
    assert approval_report["surfaced_cases"][0]["status"] == "REGRESSION"
    assert approval_report["surfaced_cases"][0]["material_reason"] == ["hard_metric_failed"]
    assert approval_report["surfaced_cases"][0]["rules"]["failed"]
    assert approval_report["user_action"] == {
        "needs_review": True,
        "needs_baseline_blessing": False,
        "review_case_ids": ["deterministic_mujoco_grasp"],
    }
    assert [pair.status for pair in evidence_pairs] == ["full", "full"]


def test_approval_report_suppresses_clean_case_when_run_matches_baseline() -> None:
    baseline = load_blessed_baseline()
    contract = build_default_contract(baseline_source="fixture")
    report = build_autonomous_report(
        snapshot_metrics=copy.deepcopy(baseline["snapshot_metrics"]),
        baseline_report=baseline,
        baseline_source="fixture",
    )
    result = evaluate_autonomous_report(report)
    alarms = build_alarms(report, result)
    manifest = build_phase_manifest(report, result, alarms)

    approval_report = build_approval_report(
        contract=contract,
        report=report,
        evaluation_result=result,
        manifest=manifest,
        evidence_pairs=[],
    )

    assert approval_report["overall_verdict"] == "PASS"
    assert approval_report["run_state"] == "review_ready_success"
    assert approval_report["surfaced_cases"] == []
    assert approval_report["suppressed_cases"] == [
        {
            "case_id": "deterministic_mujoco_grasp",
            "status": "UNCHANGED",
            "reason": "no_material_change",
        }
    ]
    assert approval_report["summary"] == {
        "cases_total": 1,
        "cases_surfaced": 0,
        "cases_suppressed": 1,
        "cases_unchanged": 1,
        "reruns": 0,
    }
    assert approval_report["user_action"] == {
        "needs_review": False,
        "needs_baseline_blessing": False,
        "review_case_ids": [],
    }


def test_migration_pass_surfaces_intended_change_for_review() -> None:
    baseline = load_blessed_baseline()
    contract = build_default_contract(baseline_source="fixture")
    contract["mode"] = "migration"
    report = build_autonomous_report(
        snapshot_metrics=copy.deepcopy(baseline["snapshot_metrics"]),
        baseline_report=baseline,
        baseline_source="fixture",
    )
    result = evaluate_autonomous_report(report, contract=contract)
    alarms = build_alarms(report, result)
    manifest = build_phase_manifest(report, result, alarms)

    approval_report = build_approval_report(
        contract=contract,
        report=report,
        evaluation_result=result,
        manifest=manifest,
        evidence_pairs=[],
    )

    assert result.verdict is Verdict.PASS
    assert approval_report["overall_verdict"] == "PASS"
    assert approval_report["run_state"] == "review_ready_migration"
    assert approval_report["stop_reason"] == "awaiting_user_blessing"
    assert approval_report["summary"] == {
        "cases_total": 1,
        "cases_surfaced": 1,
        "cases_suppressed": 0,
        "cases_unchanged": 0,
        "reruns": 0,
    }
    assert approval_report["surfaced_cases"][0]["status"] == "INTENDED_CHANGE_CONFIRMED"
    assert approval_report["surfaced_cases"][0]["material_reason"] == [
        "intended_change_requires_review"
    ]
    assert approval_report["user_action"] == {
        "needs_review": True,
        "needs_baseline_blessing": True,
        "review_case_ids": ["deterministic_mujoco_grasp"],
    }


def test_known_bad_fixture_resolves_two_ordered_evidence_pairs() -> None:
    report, result, _alarms, manifest = _build_known_bad_contract()

    evidence_pairs = resolve_evidence_pairs(
        trial_dir=KNOWN_BAD_VISUAL_ROOT,
        baseline_visual_root=BASELINE_VISUAL_ROOT,
        manifest=manifest,
        report=report,
    )

    assert result.verdict is Verdict.FAIL
    assert manifest.failed_phase_id == "approach"
    assert manifest.primary_views == ["side", "top"]
    assert manifest.rerun_hint == "restore:pre_grasp"
    assert [pair.view_name for pair in evidence_pairs] == ["side", "top"]
    assert [pair.status for pair in evidence_pairs] == ["full", "full"]


def test_ambiguous_fixture_resolves_temporal_evidence_state() -> None:
    report, result, _alarms, manifest = _build_ambiguous_contract()

    evidence_pairs = resolve_evidence_pairs(
        trial_dir=KNOWN_BAD_VISUAL_ROOT,
        baseline_visual_root=BASELINE_VISUAL_ROOT,
        manifest=manifest,
        report=report,
    )

    assert result.verdict is Verdict.DEGRADED
    assert manifest.failed_phase_id == "lift"
    assert manifest.primary_views == ["side"]
    assert [pair.view_name for pair in evidence_pairs] == ["side"]
    assert [pair.status for pair in evidence_pairs] == ["ambiguous"]


def test_summary_html_renders_current_vs_baseline_evidence_with_metric_copy() -> None:
    contract, approval_report, report, _result, alarms, evidence_pairs = _build_review_bundle()
    manifest = build_phase_manifest(report, evaluate_autonomous_report(report), alarms)

    html = build_summary_html(
        report,
        alarms,
        manifest,
        evidence_pairs,
        contract=contract,
        approval_report=approval_report,
    )

    assert "Run Decision" in html
    assert "Review surfaced cases" in html
    assert "Approval Queue" in html
    assert "Compiled Contract" in html
    assert "Baseline Promotion" in html
    assert "Current vs Baseline" in html
    assert "FAIL/full evidence" in html
    assert "Surfaced" in html
    assert ">1<" in html
    assert "Current" in html
    assert "Baseline" in html
    assert "grip_center_error_mm:" in html
    assert "threshold 12.0" in html
    assert "data:image/png;base64," in html
    assert "Regression mode. Old baseline remains authoritative." in html
    assert "Agent Next Action" in html
    assert "Hard Metric Results" in html
    assert "Artifact Pack" in html
    assert "Phase Timeline" in html


def test_summary_html_renders_temporal_evidence_and_lightbox_for_ambiguous_runs() -> None:
    report, result, alarms, manifest = _build_ambiguous_contract()
    contract = build_default_contract(baseline_source="fixture")
    evidence_pairs = resolve_evidence_pairs(
        trial_dir=KNOWN_BAD_VISUAL_ROOT,
        baseline_visual_root=BASELINE_VISUAL_ROOT,
        manifest=manifest,
        report=report,
    )
    approval_report = build_approval_report(
        contract=contract,
        report=report,
        evaluation_result=result,
        manifest=manifest,
        evidence_pairs=evidence_pairs,
    )

    html = build_summary_html(
        report,
        alarms,
        manifest,
        evidence_pairs,
        contract=contract,
        approval_report=approval_report,
    )

    assert "FAIL/ambiguous still-image evidence" in html
    assert "Temporal Evidence" in html
    assert "Checkpoint order adds motion context for this view." in html
    assert "Current checkpoints" in html
    assert "Baseline checkpoints" in html
    assert "Expand Current lift / side" in html
    assert "image-lightbox" in html
    assert "Close" in html


@pytest.mark.parametrize(
    ("missing_role", "expected_copy"),
    [
        (
            "baseline",
            (
                "Baseline image missing for approach/top. Rebuild or restore "
                "the blessed baseline pack."
            ),
        ),
        (
            "current",
            (
                "Current capture missing for approach/top. Re-run from "
                "restore:pre_grasp to rebuild evidence."
            ),
        ),
    ],
)
def test_summary_html_degrades_cleanly_when_one_side_of_evidence_is_missing(
    tmp_path: Path,
    missing_role: str,
    expected_copy: str,
) -> None:
    report, result, alarms, manifest = _build_known_bad_contract()
    contract = build_default_contract(baseline_source="fixture")
    current_root = tmp_path / "current"
    baseline_root = tmp_path / "baseline"

    _write_png(current_root / "approach" / "side_rgb.png")
    _write_png(baseline_root / "approach" / "side_rgb.png")
    if missing_role != "current":
        _write_png(current_root / "approach" / "top_rgb.png")
    if missing_role != "baseline":
        _write_png(baseline_root / "approach" / "top_rgb.png")

    evidence_pairs = resolve_evidence_pairs(
        trial_dir=current_root,
        baseline_visual_root=baseline_root,
        manifest=manifest,
        report=report,
    )
    approval_report = build_approval_report(
        contract=contract,
        report=report,
        evaluation_result=result,
        manifest=manifest,
        evidence_pairs=evidence_pairs,
    )
    html = build_summary_html(
        report,
        alarms,
        manifest,
        evidence_pairs,
        contract=contract,
        approval_report=approval_report,
    )

    assert [pair.view_name for pair in evidence_pairs] == ["side", "top"]
    assert evidence_pairs[0].status == "full"
    assert evidence_pairs[1].status == "partial"
    assert "FAIL/partial evidence" in html
    assert expected_copy in html
    assert "Current" in html
    assert "Baseline" in html


def test_summary_html_renders_explicit_success_state_for_pass_runs() -> None:
    baseline = load_blessed_baseline()
    contract = build_default_contract(baseline_source="fixture")
    report = build_autonomous_report(
        snapshot_metrics=copy.deepcopy(baseline["snapshot_metrics"]),
        baseline_report=baseline,
        baseline_source="fixture",
    )
    result = evaluate_autonomous_report(report)
    alarms = build_alarms(report, result)
    manifest = build_phase_manifest(report, result, alarms)
    approval_report = build_approval_report(
        contract=contract,
        report=report,
        evaluation_result=result,
        manifest=manifest,
        evidence_pairs=[],
    )

    evidence_pairs = resolve_evidence_pairs(
        trial_dir=KNOWN_BAD_VISUAL_ROOT,
        baseline_visual_root=BASELINE_VISUAL_ROOT,
        manifest=manifest,
        report=report,
    )
    html = build_summary_html(
        report,
        alarms,
        manifest,
        evidence_pairs,
        contract=contract,
        approval_report=approval_report,
    )

    assert result.verdict is Verdict.PASS
    assert evidence_pairs == []
    assert approval_report["surfaced_cases"] == []
    assert (
        manifest.agent_next_action
        == "No rerun required. The canonical primary views match baseline; inspect the final "
        "lift captures only if you are chasing a visual false negative."
    )
    assert "No material changes surfaced." in html
    assert "Old baseline remains authoritative." in html
    assert "PASS/no failed phase" in html
    assert "No visual regression detected for the canonical primary views." in html
    assert manifest.agent_next_action in html
    assert "Inspect the final lift captures first" not in html
    assert "Why this proof:" not in html
    assert "Rerun hint:" not in html
    assert "first failing phase <code>none</code>" not in html


def test_summary_html_renders_manifest_mismatch_banner_instead_of_crashing() -> None:
    report, result, alarms, manifest = _build_known_bad_contract()
    contract = build_default_contract(baseline_source="fixture")
    mismatch_manifest = replace(
        manifest,
        primary_views=["front"],
        evidence_paths=["approach/front_rgb.png"],
    )

    evidence_pairs = resolve_evidence_pairs(
        trial_dir=KNOWN_BAD_VISUAL_ROOT,
        baseline_visual_root=BASELINE_VISUAL_ROOT,
        manifest=mismatch_manifest,
        report=report,
    )
    approval_report = build_approval_report(
        contract=contract,
        report=report,
        evaluation_result=result,
        manifest=mismatch_manifest,
        evidence_pairs=evidence_pairs,
    )
    html = build_summary_html(
        report,
        alarms,
        mismatch_manifest,
        evidence_pairs,
        contract=contract,
        approval_report=approval_report,
    )

    assert [pair.status for pair in evidence_pairs] == ["mismatch"]
    assert "FAIL/manifest mismatch" in html
    assert "Manifest/view contract mismatch." in html


def test_checked_in_visual_fixtures_cover_approach_primary_views() -> None:
    for view in MUJOCO_GRASP_PRIMARY_VIEWS["approach"]:
        assert (BASELINE_VISUAL_ROOT / "approach" / f"{view}_rgb.png").exists()
        assert (KNOWN_BAD_VISUAL_ROOT / "approach" / f"{view}_rgb.png").exists()
