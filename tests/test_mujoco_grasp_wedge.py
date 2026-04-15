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
    build_alarms,
    build_autonomous_report,
    build_phase_manifest,
    build_summary_html,
    evaluate_autonomous_report,
    load_blessed_baseline,
    resolve_evidence_pairs,
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
    report = build_autonomous_report(
        snapshot_metrics=copy.deepcopy(baseline["snapshot_metrics"]),
        baseline_report=baseline,
        baseline_source="fixture",
    )
    result = evaluate_autonomous_report(report, report_path="trial/autonomous_report.json")
    alarms = build_alarms(report, result)
    manifest = build_phase_manifest(report, result, alarms)
    trial_dir = tmp_path / "trial_001"
    trial_dir.mkdir()

    write_artifact_pack(
        trial_dir=trial_dir,
        report=report,
        evaluation_result=result,
        alarms=alarms,
        manifest=manifest,
        report_generated=False,
    )

    autonomous_report = json.loads((trial_dir / "autonomous_report.json").read_text())
    alarms_report = json.loads((trial_dir / "alarms.json").read_text())
    phase_manifest = json.loads((trial_dir / "phase_manifest.json").read_text())

    assert "report_html" not in autonomous_report["artifacts"]
    assert alarms_report["verdict"] == "pass"
    assert phase_manifest["failed_phase_id"] is None


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


def test_summary_html_renders_current_vs_baseline_evidence_with_metric_copy() -> None:
    report, _result, alarms, manifest = _build_known_bad_contract()
    evidence_pairs = resolve_evidence_pairs(
        trial_dir=KNOWN_BAD_VISUAL_ROOT,
        baseline_visual_root=BASELINE_VISUAL_ROOT,
        manifest=manifest,
        report=report,
    )

    html = build_summary_html(report, alarms, manifest, evidence_pairs)

    assert "Current vs Baseline" in html
    assert "FAIL/full evidence" in html
    assert "Current" in html
    assert "Baseline" in html
    assert "grip_center_error_mm:" in html
    assert "threshold 12.0" in html
    assert "data:image/png;base64," in html
    assert "Agent Next Action" in html
    assert "Artifact Pack" in html
    assert "Phase Timeline" in html


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
    report, _result, alarms, manifest = _build_known_bad_contract()
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
    html = build_summary_html(report, alarms, manifest, evidence_pairs)

    assert [pair.view_name for pair in evidence_pairs] == ["side", "top"]
    assert evidence_pairs[0].status == "full"
    assert evidence_pairs[1].status == "partial"
    assert "FAIL/partial evidence" in html
    assert expected_copy in html
    assert "Current" in html
    assert "Baseline" in html


def test_summary_html_renders_explicit_success_state_for_pass_runs() -> None:
    baseline = load_blessed_baseline()
    report = build_autonomous_report(
        snapshot_metrics=copy.deepcopy(baseline["snapshot_metrics"]),
        baseline_report=baseline,
        baseline_source="fixture",
    )
    result = evaluate_autonomous_report(report)
    alarms = build_alarms(report, result)
    manifest = build_phase_manifest(report, result, alarms)

    evidence_pairs = resolve_evidence_pairs(
        trial_dir=KNOWN_BAD_VISUAL_ROOT,
        baseline_visual_root=BASELINE_VISUAL_ROOT,
        manifest=manifest,
        report=report,
    )
    html = build_summary_html(report, alarms, manifest, evidence_pairs)

    assert result.verdict is Verdict.PASS
    assert evidence_pairs == []
    assert (
        manifest.agent_next_action
        == "No rerun required. The canonical primary views match baseline; inspect the final "
        "lift captures only if you are chasing a visual false negative."
    )
    assert "PASS/no failed phase" in html
    assert "No visual regression detected for the canonical primary views." in html
    assert manifest.agent_next_action in html
    assert "Inspect the final lift captures first" not in html
    assert "Why this proof:" not in html
    assert "Rerun hint:" not in html
    assert "first failing phase <code>none</code>" not in html


def test_summary_html_renders_manifest_mismatch_banner_instead_of_crashing() -> None:
    report, _result, alarms, manifest = _build_known_bad_contract()
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
    html = build_summary_html(report, alarms, mismatch_manifest, evidence_pairs)

    assert [pair.status for pair in evidence_pairs] == ["mismatch"]
    assert "FAIL/manifest mismatch" in html
    assert "Manifest/view contract mismatch." in html


def test_checked_in_visual_fixtures_cover_approach_primary_views() -> None:
    for view in MUJOCO_GRASP_PRIMARY_VIEWS["approach"]:
        assert (BASELINE_VISUAL_ROOT / "approach" / f"{view}_rgb.png").exists()
        assert (KNOWN_BAD_VISUAL_ROOT / "approach" / f"{view}_rgb.png").exists()
