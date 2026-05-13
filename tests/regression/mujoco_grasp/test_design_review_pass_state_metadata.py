from __future__ import annotations

import copy

from examples._mujoco_grasp_wedge import (
    BASELINE_VISUAL_ROOT,
    build_alarms,
    build_approval_report,
    build_autonomous_report,
    build_default_contract,
    build_phase_manifest,
    build_summary_html,
    evaluate_autonomous_report,
    load_blessed_baseline,
    resolve_evidence_pairs,
)
from roboharness.evaluate.result import Verdict


def test_pass_manifest_and_summary_do_not_emit_failure_only_metadata() -> None:
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
    evidence_pairs = resolve_evidence_pairs(
        trial_dir=BASELINE_VISUAL_ROOT,
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

    assert result.verdict is Verdict.PASS
    assert manifest.failed_phase_id is None
    assert manifest.suspected_root_cause == "none"
    assert manifest.rerun_hint == "not_required"
    assert manifest.evidence_paths == []
    assert "Root cause</th><td><code>none</code></td>" in html
    assert "Rerun hint</th><td>No rerun required</td>" in html
    assert "trajectory_regression" not in html
    assert "restore:plan" not in html
