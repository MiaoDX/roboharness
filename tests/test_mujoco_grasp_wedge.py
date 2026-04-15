"""Tests for the MuJoCo alarmed grasp-loop helper pipeline."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from examples._mujoco_grasp_fixture import (
    MUJOCO_GRASP_CAMERAS,
    MUJOCO_GRASP_PHASE_ORDER,
    build_grasp_phases,
    build_grasp_protocol,
)
from examples._mujoco_grasp_wedge import (
    build_alarms,
    build_autonomous_report,
    build_phase_manifest,
    build_summary_html,
    evaluate_autonomous_report,
    load_blessed_baseline,
    write_artifact_pack,
)
from roboharness.evaluate.result import Verdict


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


def test_summary_html_surfaces_alarm_first_sections() -> None:
    baseline = load_blessed_baseline()
    current = copy.deepcopy(baseline["snapshot_metrics"])
    current["lift"]["cube_height_mm"] -= 20.0

    report = build_autonomous_report(
        snapshot_metrics=current,
        baseline_report=baseline,
        baseline_source="fixture",
    )
    result = evaluate_autonomous_report(report)
    alarms = build_alarms(report, result)
    manifest = build_phase_manifest(report, result, alarms)
    html = build_summary_html(report, alarms, manifest)

    assert "Agent Next Action" in html
    assert "Artifact Pack" in html
    assert "Phase Timeline" in html
    assert "lift" in html
