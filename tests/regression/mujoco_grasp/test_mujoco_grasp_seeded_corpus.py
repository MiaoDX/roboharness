"""Seeded evaluator corpus for MuJoCo approval-queue trust checks."""

from __future__ import annotations

import copy
from typing import Any

from examples._mujoco_grasp_wedge import (
    BASELINE_VISUAL_ROOT,
    KNOWN_BAD_VISUAL_ROOT,
    build_alarms,
    build_approval_report,
    build_autonomous_report,
    build_default_contract,
    build_phase_manifest,
    evaluate_autonomous_report,
    load_blessed_baseline,
    resolve_evidence_pairs,
)


def _evaluate_seeded_case(case_name: str) -> dict[str, Any]:
    baseline = load_blessed_baseline()
    current = copy.deepcopy(baseline["snapshot_metrics"])

    if case_name == "bad":
        current["approach"]["grip_center_error_mm"] += 25.0
    elif case_name == "ambiguous":
        for phase_metrics in current.values():
            phase_metrics["max_abs_qvel"] = 60.0
    elif case_name != "good":
        raise ValueError(f"Unknown seeded corpus case: {case_name}")

    report = build_autonomous_report(
        snapshot_metrics=current,
        baseline_report=baseline,
        baseline_source="fixture",
    )
    evaluation_result = evaluate_autonomous_report(report)
    alarms = build_alarms(report, evaluation_result)
    manifest = build_phase_manifest(report, evaluation_result, alarms)
    evidence_pairs = resolve_evidence_pairs(
        trial_dir=KNOWN_BAD_VISUAL_ROOT,
        baseline_visual_root=BASELINE_VISUAL_ROOT,
        manifest=manifest,
        report=report,
    )
    approval_report = build_approval_report(
        contract=build_default_contract(baseline_source="fixture"),
        report=report,
        evaluation_result=evaluation_result,
        manifest=manifest,
        evidence_pairs=evidence_pairs,
    )
    return {
        "case_name": case_name,
        "overall_verdict": approval_report["overall_verdict"],
        "run_state": approval_report["run_state"],
        "surfaced": approval_report["summary"]["cases_surfaced"] > 0,
        "review_case_ids": approval_report["user_action"]["review_case_ids"],
        "needs_review": approval_report["user_action"]["needs_review"],
        "surfaced_statuses": [case["status"] for case in approval_report["surfaced_cases"]],
        "material_reasons": [case["material_reason"] for case in approval_report["surfaced_cases"]],
        "ambiguous_rules": [
            case["rules"]["ambiguous"] for case in approval_report["surfaced_cases"]
        ],
        "evidence_statuses": [pair.status for pair in evidence_pairs],
    }


def test_seeded_corpus_cases_cover_good_bad_and_ambiguous_review_outcomes() -> None:
    good = _evaluate_seeded_case("good")
    bad = _evaluate_seeded_case("bad")
    ambiguous = _evaluate_seeded_case("ambiguous")

    assert good["overall_verdict"] == "PASS"
    assert good["run_state"] == "review_ready_success"
    assert good["surfaced"] is False
    assert good["needs_review"] is False
    assert good["review_case_ids"] == []
    assert good["evidence_statuses"] == []

    assert bad["overall_verdict"] == "FAIL"
    assert bad["run_state"] == "review_ready_surfaced"
    assert bad["surfaced"] is True
    assert bad["needs_review"] is True
    assert bad["review_case_ids"] == ["deterministic_mujoco_grasp"]
    assert bad["surfaced_statuses"] == ["REGRESSION"]
    assert bad["material_reasons"] == [["hard_metric_failed"]]
    assert bad["evidence_statuses"] == ["full", "full"]

    assert ambiguous["overall_verdict"] == "AMBIGUOUS"
    assert ambiguous["run_state"] == "evidence_degraded"
    assert ambiguous["surfaced"] is True
    assert ambiguous["needs_review"] is True
    assert ambiguous["review_case_ids"] == ["deterministic_mujoco_grasp"]
    assert ambiguous["surfaced_statuses"] == ["AMBIGUOUS"]
    assert ambiguous["material_reasons"] == [["visual_intent_unclear"]]
    assert ambiguous["ambiguous_rules"] == [["still_image_review_required"]]
    assert ambiguous["evidence_statuses"] == ["ambiguous"]


def test_seeded_corpus_surface_precision_and_recall_are_perfect_for_v1_seeds() -> None:
    expected_surface = {
        "good": False,
        "bad": True,
        "ambiguous": True,
    }
    actual_results = {case_name: _evaluate_seeded_case(case_name) for case_name in expected_surface}

    true_positives = sum(
        1
        for case_name, should_surface in expected_surface.items()
        if should_surface and actual_results[case_name]["surfaced"]
    )
    false_positives = sum(
        1
        for case_name, should_surface in expected_surface.items()
        if not should_surface and actual_results[case_name]["surfaced"]
    )
    false_negatives = sum(
        1
        for case_name, should_surface in expected_surface.items()
        if should_surface and not actual_results[case_name]["surfaced"]
    )

    surfaced_predictions = true_positives + false_positives
    expected_surfaces = true_positives + false_negatives
    precision = true_positives / surfaced_predictions if surfaced_predictions else 1.0
    recall = true_positives / expected_surfaces if expected_surfaces else 1.0

    assert precision == 1.0
    assert recall == 1.0
    assert false_positives == 0
    assert false_negatives == 0
