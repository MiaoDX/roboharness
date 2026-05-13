from __future__ import annotations

import base64
from pathlib import Path

from examples._mujoco_grasp_wedge import (
    PhaseManifest,
    build_approval_report,
    build_default_contract,
    build_summary_html,
)
from roboharness.evaluate.assertions import AssertionEngine, MetricAssertion
from roboharness.evaluate.result import Operator, Severity
from roboharness.reporting import generate_html_report

_ONE_PIXEL_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+yv3sAAAAASUVORK5CYII="
)


def _write_tiny_png(path: Path) -> None:
    path.write_bytes(_ONE_PIXEL_PNG)


def test_generate_html_report_contains_responsive_table_and_meshcat_guards(tmp_path: Path) -> None:
    trial_dir = tmp_path / "task" / "trial_001"
    cp_dir = trial_dir / "plan"
    cp_dir.mkdir(parents=True)
    (cp_dir / "metadata.json").write_text('{"step": 0}')
    _write_tiny_png(cp_dir / "front_rgb.png")
    (cp_dir / "meshcat_scene.html").write_text("<html>meshcat</html>")

    evaluation_result = AssertionEngine(
        [MetricAssertion("loop_runtime_s", Operator.LT, 10.0, Severity.MAJOR)]
    ).evaluate({"summary_metrics": {"loop_runtime_s": 5.0}})

    report_path = generate_html_report(
        tmp_path,
        "task",
        meshcat_mode="iframe",
        evaluation_result=evaluation_result,
    )

    html = report_path.read_text()
    assert '<div class="table-scroll"><table class="eval-table">' in html
    assert "min-width: 640px" in html
    assert "max-width: 480px" in html
    assert "flex-basis: 100%" in html
    assert ".evidence-zoom-button" in html
    assert ".image-lightbox" in html


def test_build_summary_html_wraps_artifact_pack_table_for_small_screens() -> None:
    contract = build_default_contract(baseline_source="baseline_autonomous_report.json")
    manifest = PhaseManifest(
        task="mujoco_grasp",
        verdict="pass",
        failed_phase_id=None,
        failed_phase=None,
        suspected_root_cause="trajectory_regression",
        primary_views=["side", "top"],
        regressions=[],
        rerun_hint="restore:plan",
        agent_next_action="No rerun required.",
        evidence_paths=[],
        phase_aliases={},
        phase_statuses=[],
    )
    approval_report = build_approval_report(
        contract=contract,
        report={
            "case_id": "deterministic_mujoco_grasp",
            "baseline_source": "baseline_autonomous_report.json",
            "summary_metrics": {},
            "baseline_summary_metrics": {},
        },
        evaluation_result=AssertionEngine(
            [MetricAssertion("loop_runtime_s", Operator.LT, 10.0, Severity.MAJOR)]
        ).evaluate({"summary_metrics": {"loop_runtime_s": 5.0}}),
        manifest=manifest,
        evidence_pairs=[],
    )

    html = build_summary_html(
        {
            "case_id": "deterministic_mujoco_grasp",
            "baseline_source": "baseline_autonomous_report.json",
            "summary_metrics": {},
            "baseline_summary_metrics": {},
            "snapshot_metrics": {},
        },
        [],
        manifest,
        [],
        contract=contract,
        approval_report=approval_report,
    )

    assert '<div class="table-scroll"><table class="meta-table">' in html
    assert "image-lightbox" in html
