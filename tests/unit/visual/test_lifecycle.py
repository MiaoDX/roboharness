from __future__ import annotations

import json
from pathlib import Path

from roboharness.evidence import RendererReport, SemanticSnapshotBundle
from roboharness.visual import (
    VisualCaseRun,
    VisualCaseSpec,
    VisualSuiteOptions,
    VisualSuiteRun,
    VisualSuiteSpec,
    run_visual_suite,
)


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"png")


def _snapshot_bundle(case_id: str) -> SemanticSnapshotBundle:
    return SemanticSnapshotBundle.from_dict(
        {
            "schema_version": 2,
            "snapshot_order": ["00_planned", "03_close_done", "09_home_final_done"],
            "metadata": {
                "robot_type": "g1",
                "case_id": case_id,
                "control_backend": "decoupled_wbc",
                "runtime_surface": "in_process",
            },
            "snapshots": [
                {
                    "name": "00_planned",
                    "q": [0.0],
                    "metrics": {"semantic_milestone": "planned"},
                },
                {
                    "name": "03_close_done",
                    "q": [1.0],
                    "metrics": {"semantic_milestone": "close_done"},
                },
                {
                    "name": "09_home_final_done",
                    "q": [2.0],
                    "metrics": {"semantic_milestone": "home_final_done"},
                },
            ],
        }
    )


def _renderer_report(case_dir: Path, renderer: str) -> RendererReport:
    snapshots = []
    for snapshot_name in ("00_planned", "03_close_done", "09_home_final_done"):
        images = []
        for camera in ("front2back", "left2right", "top2down"):
            relative_path = f"{renderer}/{snapshot_name}_{camera}.png"
            _write_image(case_dir / relative_path)
            images.append(
                {
                    "camera": camera,
                    "path": (case_dir / relative_path).as_posix(),
                    "unique_colors": 128,
                    "workspace_visible": True,
                }
            )
        snapshots.append(
            {
                "name": snapshot_name,
                "capture_ok": True,
                "motion_ok": True,
                "metrics": {"semantic_milestone": snapshot_name},
                "images": images,
            }
        )
    return RendererReport.from_dict(
        {
            "output_dir": (case_dir / renderer).as_posix(),
            "renderer": renderer,
            "capture_ok": True,
            "motion_ok": True,
            "flags": [],
            "trustworthiness_flags": [],
            "metadata": {},
            "snapshots": snapshots,
        }
    )


def _visual_case(case_dir: Path, case_id: str = "X36_Y28_Z13") -> VisualCaseRun:
    case_run = VisualCaseRun(
        case_id=case_id,
        output_dir=case_dir,
        robot_type="g1",
        runner={"runner_type": "g1_visual_harness"},
        runtime={"control_backend": "decoupled_wbc"},
        plan={"intent_level": "task_intent", "control_level": "planned_control"},
        extra={"semantic_visual_ok": True},
    )
    case_run.set_snapshot_bundle(_snapshot_bundle(case_id))
    case_run.add_renderer_report("meshcat", _renderer_report(case_dir, "meshcat"))
    case_run.add_renderer_report("mujoco", _renderer_report(case_dir, "mujoco"))
    case_run.set_metrics(
        {
            "final_snapshot_name": "09_home_final_done",
            "grasp_accuracy_snapshot_name": "03_close_done",
            "semantic_visual_ok": True,
            "workspace_framing_ok": True,
            "grip_center_error_mm": 12.0,
            "pinch_gap_error_mm": 4.0,
            "render_mujoco_enabled": True,
            "render_total_s": 3.25,
        },
        snapshot_metrics={
            "03_close_done": {
                "grip_center_error_mm": 12.0,
                "semantic_milestone": "close_done",
            }
        },
    )
    case_run.set_verdict("pass")
    return case_run


def test_visual_case_run_writes_groot_style_review_artifacts(tmp_path: Path) -> None:
    case_dir = tmp_path / "X36_Y28_Z13"
    case_run = _visual_case(case_dir)

    artifacts = case_run.write_artifacts(task_intent="Review GR00T case through lifecycle API.")

    assert artifacts.autonomous_report_path == case_dir / "autonomous_report.json"
    assert artifacts.proof_pack_path == case_dir / "proof_pack.json"
    assert artifacts.visual_review_manifest_path == case_dir / "visual_review_manifest.json"
    proof_pack = json.loads(artifacts.proof_pack_path.read_text(encoding="utf-8"))
    manifest = json.loads(artifacts.visual_review_manifest_path.read_text(encoding="utf-8"))
    assert proof_pack["case_id"] == "X36_Y28_Z13"
    assert proof_pack["selected_phase"] == "09_home_final_done"
    assert {ref["renderer"] for ref in proof_pack["renderer_evidence"]} == {
        "meshcat",
        "mujoco",
    }
    assert manifest["review_policy"]["allow_automatic_visual_pass"] is False


def test_visual_suite_run_writes_suite_artifacts(tmp_path: Path) -> None:
    case_dir = tmp_path / "X36_Y28_Z13"
    case_run = _visual_case(case_dir)
    case_run.write_artifacts()
    suite = VisualSuiteRun(suite_name="representative", output_root=tmp_path)
    suite.add_case(case_run)

    artifacts = suite.write_artifacts(task_intent="Review GR00T suite.")

    suite_report = json.loads(artifacts.suite_report_path.read_text(encoding="utf-8"))
    suite_proof_pack = json.loads(artifacts.suite_proof_pack_path.read_text(encoding="utf-8"))
    queue = json.loads(artifacts.visual_review_queue_path.read_text(encoding="utf-8"))
    assert suite_report["suite_proof_pack_path"] == artifacts.suite_proof_pack_path.as_posix()
    assert suite_report["visual_review_queue_path"] == artifacts.visual_review_queue_path.as_posix()
    assert suite_proof_pack["reviewable_count"] == 1
    assert suite_proof_pack["skipped_count"] == 0
    assert queue["total_items"] == 1


def test_run_visual_suite_records_execution_errors_without_hiding_them(tmp_path: Path) -> None:
    suite_spec = VisualSuiteSpec(
        suite_name="representative",
        cases=[
            VisualCaseSpec("X36_Y28_Z13"),
            VisualCaseSpec("BROKEN"),
        ],
    )

    def _run_case(case_spec: VisualCaseSpec, case_dir: Path) -> VisualCaseRun:
        if case_spec.case_id == "BROKEN":
            raise RuntimeError("boom")
        return _visual_case(case_dir, case_spec.case_id)

    artifacts = run_visual_suite(
        suite_spec,
        case_runner=_run_case,
        output_root=tmp_path,
        options=VisualSuiteOptions(task_intent="Review suite."),
    )

    suite_report = json.loads(artifacts.suite_report_path.read_text(encoding="utf-8"))
    suite_proof_pack = json.loads(artifacts.suite_proof_pack_path.read_text(encoding="utf-8"))
    queue = json.loads(artifacts.visual_review_queue_path.read_text(encoding="utf-8"))
    assert suite_report["total_cases"] == 2
    assert suite_report["execution_error_count"] == 1
    assert suite_proof_pack["reviewable_count"] == 1
    assert suite_proof_pack["skipped_count"] == 1
    assert suite_proof_pack["cases"][1]["error"] == "case directory does not exist"
    assert queue["total_items"] == 1
