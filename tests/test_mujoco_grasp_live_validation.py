"""Optional live validation for the MuJoCo grasp phase-2 evidence contract."""

from __future__ import annotations

import copy
import os
from pathlib import Path

import pytest

from examples._mujoco_grasp_fixture import (
    GRASP_MJCF,
    MUJOCO_GRASP_CAMERAS,
    build_grasp_phases,
    build_grasp_protocol,
)
from examples._mujoco_grasp_wedge import (
    BASELINE_VISUAL_ROOT,
    build_alarms,
    build_autonomous_report,
    build_phase_manifest,
    collect_phase_metrics,
    evaluate_autonomous_report,
    load_blessed_baseline,
    resolve_evidence_pairs,
)
from roboharness.evaluate.result import Verdict


def _require_mujoco_rendering() -> None:
    """Skip when MuJoCo rendering is unavailable in the current environment."""
    mujoco = pytest.importorskip("mujoco")
    gl_backend = os.environ.get("MUJOCO_GL", "").lower()
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    if not gl_backend and not has_display:
        pytest.skip("MuJoCo rendering not available (no MUJOCO_GL or DISPLAY set)")
    try:
        mujoco.Renderer  # noqa: B018
    except AttributeError:
        pytest.skip("MuJoCo version does not support Renderer")


def _close_backend(backend: object) -> None:
    visualizer = getattr(backend, "visualizer", None)
    renderer = getattr(visualizer, "_renderer", None)
    if renderer is not None and hasattr(renderer, "close"):
        renderer.close()
        visualizer._renderer = None


def test_live_run_matches_known_bad_fixture_contract(tmp_path: Path) -> None:
    _require_mujoco_rendering()

    from roboharness.backends.mujoco_meshcat import MuJoCoMeshcatBackend
    from roboharness.core.harness import Harness

    backend = MuJoCoMeshcatBackend(
        xml_string=GRASP_MJCF,
        cameras=MUJOCO_GRASP_CAMERAS,
        render_width=320,
        render_height=240,
    )
    harness = Harness(backend, output_dir=str(tmp_path), task_name="mujoco_grasp")
    harness.load_protocol(build_grasp_protocol())
    harness.reset()

    phases = copy.deepcopy(build_grasp_phases())
    bad_approach = []
    for action in phases["approach"]:
        bad_action = action.copy()
        bad_action[0] = -0.10
        bad_action[2] = 0.02
        bad_approach.append(bad_action)
    phases["approach"] = bad_approach

    checkpoint_results: dict[str, object] = {}
    try:
        for phase_name, actions in phases.items():
            result = harness.run_to_next_checkpoint(actions)
            assert result is not None
            checkpoint_results[phase_name] = result

        baseline = load_blessed_baseline()
        report = build_autonomous_report(
            snapshot_metrics=collect_phase_metrics(harness, backend, checkpoint_results),
            baseline_report=baseline,
            baseline_source="live-validation",
        )
        evaluation_result = evaluate_autonomous_report(report)
        alarms = build_alarms(report, evaluation_result)
        manifest = build_phase_manifest(report, evaluation_result, alarms)
        evidence_pairs = resolve_evidence_pairs(
            trial_dir=tmp_path / "mujoco_grasp" / "trial_001",
            baseline_visual_root=BASELINE_VISUAL_ROOT,
            manifest=manifest,
            report=report,
        )
    finally:
        _close_backend(backend)

    assert evaluation_result.verdict is Verdict.FAIL
    assert manifest.failed_phase_id == "approach"
    assert manifest.primary_views == ["side", "top"]
    assert manifest.rerun_hint == "restore:pre_grasp"
    assert [pair.view_name for pair in evidence_pairs] == ["side", "top"]
    assert [pair.status for pair in evidence_pairs] == ["full", "full"]
