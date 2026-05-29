from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from roboharness.approval.visual_review import (
    VisualReviewValidationError,
    ingest_visual_review_record,
    validate_visual_review_manifest,
)
from roboharness.evidence import (
    CASE_PROOF_PACK_SCHEMA_VERSION,
    STATIC_VISUAL_DIMENSIONS,
    SUITE_PROOF_PACK_SCHEMA_VERSION,
    VISUAL_REVIEW_QUEUE_SCHEMA_VERSION,
    build_case_proof_pack,
    build_paired_visual_review_manifest,
    build_static_visual_review_manifest,
    build_suite_proof_pack,
    build_visual_review_queue,
    load_case_proof_pack,
    load_suite_proof_pack,
    write_visual_review_queue,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"png")


def _renderer_report(case_dir: Path, renderer: str) -> dict[str, object]:
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
    return {
        "output_dir": (case_dir / renderer).as_posix(),
        "renderer": renderer,
        "capture_ok": True,
        "motion_ok": True,
        "flags": [],
        "trustworthiness_flags": [],
        "metadata": {},
        "snapshots": snapshots,
    }


def _write_groot_case(case_dir: Path) -> None:
    meshcat_report = _renderer_report(case_dir, "meshcat")
    mujoco_report = _renderer_report(case_dir, "mujoco")
    _write_json(case_dir / "meshcat" / "report.json", meshcat_report)
    _write_json(case_dir / "mujoco" / "report.json", mujoco_report)
    _write_json(
        case_dir / "snapshot_bundle.json",
        {
            "schema_version": 2,
            "snapshot_order": ["00_planned", "03_close_done", "09_home_final_done"],
            "metadata": {
                "robot_type": "g1",
                "case_id": "X36_Y28_Z13",
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
        },
    )
    _write_json(
        case_dir / "autonomous_report.json",
        {
            "case_id": "X36_Y28_Z13",
            "output_dir": case_dir.as_posix(),
            "robot_type": "g1",
            "verdict": "pass",
            "verdict_reasons": [],
            "failure_taxonomy": [],
            "runtime": {"control_backend": "decoupled_wbc"},
            "plan": {"intent_level": "task_intent"},
            "snapshot_order": ["00_planned", "03_close_done", "09_home_final_done"],
            "snapshot_metrics": {
                "03_close_done": {
                    "grip_center_error_mm": 12.0,
                    "semantic_milestone": "close_done",
                }
            },
            "summary_metrics": {
                "final_snapshot_name": "09_home_final_done",
                "grasp_accuracy_snapshot_name": "03_close_done",
                "semantic_visual_ok": True,
                "workspace_framing_ok": True,
                "grip_center_error_mm": 12.0,
                "pinch_gap_error_mm": 4.0,
                "render_mujoco_enabled": True,
                "render_total_s": 3.25,
            },
            "renderer_reports": {
                "meshcat": meshcat_report,
                "mujoco": mujoco_report,
            },
        },
    )


def _write_suite_report(path: Path, case_dirs: list[Path]) -> None:
    _write_json(
        path,
        {
            "suite_name": "representative",
            "output_root": path.parent.as_posix(),
            "suite_verdict": "pass",
            "total_cases": len(case_dirs),
            "pass_count": len(case_dirs),
            "fail_count": 0,
            "execution_error_count": 0,
            "results": [
                {
                    "case_id": case_dir.name,
                    "output_dir": case_dir.as_posix(),
                    "status": "pass",
                    "report_json": (case_dir / "autonomous_report.json").as_posix(),
                }
                for case_dir in case_dirs
            ],
        },
    )


def test_build_case_proof_pack_from_groot_style_case(tmp_path: Path) -> None:
    case_dir = tmp_path / "X36_Y28_Z13"
    _write_groot_case(case_dir)

    proof_pack = build_case_proof_pack(case_dir)

    assert proof_pack.schema_version == CASE_PROOF_PACK_SCHEMA_VERSION
    assert proof_pack.case_id == "X36_Y28_Z13"
    assert proof_pack.verdict == "pass"
    assert proof_pack.selected_phase == "09_home_final_done"
    assert proof_pack.snapshot_order == ("00_planned", "03_close_done", "09_home_final_done")
    assert proof_pack.metric_summary["grip_center_error_mm"] == 12.0
    assert {ref.renderer for ref in proof_pack.renderer_evidence} == {"meshcat", "mujoco"}
    assert {ref.path for ref in proof_pack.renderer_evidence if ref.renderer == "mujoco"} == {
        "mujoco/09_home_final_done_front2back.png",
        "mujoco/09_home_final_done_left2right.png",
        "mujoco/09_home_final_done_top2down.png",
    }
    assert [artifact.path for artifact in proof_pack.artifacts] == [
        "autonomous_report.json",
        "snapshot_bundle.json",
        "meshcat/report.json",
        "mujoco/report.json",
    ]

    path = proof_pack.write_json(case_dir / "proof_pack.json")
    assert load_case_proof_pack(path).to_dict() == proof_pack.to_dict()


def test_build_suite_proof_pack_and_visual_review_queue_from_case_artifacts(
    tmp_path: Path,
) -> None:
    case_dir = tmp_path / "X36_Y28_Z13"
    _write_groot_case(case_dir)
    suite_report_path = tmp_path / "suite_report_representative.json"
    _write_suite_report(suite_report_path, [case_dir])

    suite_proof_pack = build_suite_proof_pack(suite_report_path)

    assert suite_proof_pack.schema_version == SUITE_PROOF_PACK_SCHEMA_VERSION
    assert suite_proof_pack.suite_name == "representative"
    assert suite_proof_pack.reviewable_count == 1
    assert suite_proof_pack.skipped_count == 0
    assert suite_proof_pack.cases[0].case_id == "X36_Y28_Z13"
    assert suite_proof_pack.cases[0].status == "reviewable"
    assert suite_proof_pack.cases[0].proof_pack_path == "X36_Y28_Z13/proof_pack.json"
    assert (
        suite_proof_pack.cases[0].visual_review_manifest_path
        == "X36_Y28_Z13/visual_review_manifest.json"
    )
    assert (case_dir / "proof_pack.json").exists()
    assert (case_dir / "visual_review_manifest.json").exists()

    suite_path = suite_proof_pack.write_json(tmp_path / "suite_proof_pack.json")
    assert load_suite_proof_pack(suite_path).to_dict() == suite_proof_pack.to_dict()

    queue = build_visual_review_queue(suite_proof_pack)
    assert queue.schema_version == VISUAL_REVIEW_QUEUE_SCHEMA_VERSION
    assert queue.suite_name == "representative"
    assert len(queue.items) == 1
    assert queue.items[0].visual_review_manifest_path == ("X36_Y28_Z13/visual_review_manifest.json")
    queue_path = write_visual_review_queue(queue, tmp_path / "visual_review_queue.json")
    queue_payload = json.loads(queue_path.read_text(encoding="utf-8"))
    assert queue_payload["total_items"] == 1


def test_suite_proof_pack_accepts_repo_relative_case_output_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    suite_root = repo_root / "tmp" / "visual_harness" / "suite"
    case_dir = suite_root / "X36_Y28_Z13"
    _write_groot_case(case_dir)
    suite_report_path = suite_root / "suite_report_representative.json"
    _write_json(
        suite_report_path,
        {
            "suite_name": "representative",
            "output_root": suite_root.as_posix(),
            "results": [
                {
                    "case_id": "X36_Y28_Z13",
                    "output_dir": "tmp/visual_harness/suite/X36_Y28_Z13",
                    "status": "pass",
                }
            ],
        },
    )
    monkeypatch.chdir(repo_root)

    suite_proof_pack = build_suite_proof_pack(suite_report_path)
    queue = build_visual_review_queue(suite_proof_pack)

    assert suite_proof_pack.reviewable_count == 1
    assert suite_proof_pack.skipped_count == 0
    assert suite_proof_pack.cases[0].case_dir == "X36_Y28_Z13"
    assert suite_proof_pack.cases[0].proof_pack_path == "X36_Y28_Z13/proof_pack.json"
    assert len(queue.items) == 1


def test_suite_proof_pack_skips_execution_errors(tmp_path: Path) -> None:
    suite_report_path = tmp_path / "suite_report_representative.json"
    _write_json(
        suite_report_path,
        {
            "suite_name": "representative",
            "output_root": tmp_path.as_posix(),
            "results": [
                {
                    "case_id": "BROKEN",
                    "output_dir": (tmp_path / "BROKEN").as_posix(),
                    "status": "execution_error",
                    "error_type": "RuntimeError",
                    "error": "boom",
                }
            ],
        },
    )

    suite_proof_pack = build_suite_proof_pack(suite_report_path)
    queue = build_visual_review_queue(suite_proof_pack)

    assert suite_proof_pack.reviewable_count == 0
    assert suite_proof_pack.skipped_count == 1
    assert suite_proof_pack.cases[0].status == "skipped"
    assert suite_proof_pack.cases[0].error == "case directory does not exist"
    assert queue.items == ()


def test_static_visual_review_manifest_uses_mujoco_keyframes_and_current_only_policy(
    tmp_path: Path,
) -> None:
    case_dir = tmp_path / "X36_Y28_Z13"
    _write_groot_case(case_dir)
    proof_pack = build_case_proof_pack(case_dir)

    manifest = build_static_visual_review_manifest(
        proof_pack,
        task_intent="Confirm the G1 final home pose and task success evidence.",
    )

    assert manifest["mode"] == "current_only"
    assert manifest["review_policy"] == {
        "requires_paired_evidence": False,
        "allow_automatic_visual_pass": False,
        "human_escalation_reasons": ["current_only_review_cannot_auto_pass"],
    }
    assert [dimension["id"] for dimension in manifest["dimensions"]] == list(
        STATIC_VISUAL_DIMENSIONS
    )
    first_dimension = manifest["dimensions"][0]
    assert first_dimension["phase"] == "09_home_final_done"
    assert first_dimension["evidence_type"] == "current_static_keyframe"
    assert first_dimension["current"] == [
        "mujoco/09_home_final_done_front2back.png",
        "mujoco/09_home_final_done_left2right.png",
        "mujoco/09_home_final_done_top2down.png",
    ]
    validate_visual_review_manifest(manifest, current_root=case_dir)

    record = {
        "schema_version": "roboharness_visual_review/v1",
        "case_id": "X36_Y28_Z13",
        "reviewer_context": "unit_test",
        "overall_visual_verdict": "PASS",
        "dimensions": [
            {
                "id": dimension["id"],
                "verdict": "PASS",
                "confidence": "medium",
                "evidence": list(dimension["current"]),
                "rationale": "Static keyframes show no obvious issue.",
            }
            for dimension in manifest["dimensions"]
        ],
        "needs_human_reasons": [],
    }
    result = ingest_visual_review_record(manifest, record)
    assert result.effective_visual_verdict == "NEEDS_HUMAN"
    assert result.summary["needs_human_reasons"] == ["current_only_review_cannot_auto_pass"]


def test_paired_visual_review_manifest_requires_explicit_baseline_evidence(
    tmp_path: Path,
) -> None:
    current_dir = tmp_path / "current" / "X36_Y28_Z13"
    baseline_dir = tmp_path / "baseline" / "X36_Y28_Z13"
    _write_groot_case(current_dir)
    _write_groot_case(baseline_dir)
    current_proof_pack = build_case_proof_pack(current_dir)
    baseline_proof_pack = build_case_proof_pack(baseline_dir)

    manifest = build_paired_visual_review_manifest(
        current_proof_pack,
        baseline_proof_pack,
        task_intent="Confirm the current run does not regress against the blessed baseline.",
    )

    assert manifest["mode"] == "regression"
    assert manifest["review_policy"] == {
        "requires_paired_evidence": True,
        "allow_automatic_visual_pass": True,
        "human_escalation_reasons": [],
    }
    first_dimension = manifest["dimensions"][0]
    assert first_dimension["evidence_type"] == "paired_keyframe"
    assert first_dimension["current"] == [
        "mujoco/09_home_final_done_front2back.png",
        "mujoco/09_home_final_done_left2right.png",
        "mujoco/09_home_final_done_top2down.png",
    ]
    assert first_dimension["baseline"] == first_dimension["current"]
    validate_visual_review_manifest(
        manifest,
        current_root=current_dir,
        baseline_root=baseline_dir,
    )

    record = {
        "schema_version": "roboharness_visual_review/v1",
        "case_id": "X36_Y28_Z13",
        "reviewer_context": "unit_test",
        "overall_visual_verdict": "PASS",
        "dimensions": [
            {
                "id": dimension["id"],
                "verdict": "PASS",
                "confidence": "medium",
                "evidence": list(dimension["current"]) + list(dimension["baseline"]),
                "rationale": "Current and baseline keyframes match within visual tolerance.",
            }
            for dimension in manifest["dimensions"]
        ],
        "needs_human_reasons": [],
    }
    result = ingest_visual_review_record(manifest, record)
    assert result.effective_visual_verdict == "PASS"


def test_paired_visual_review_manifest_migration_requires_baseline_blessing(
    tmp_path: Path,
) -> None:
    current_dir = tmp_path / "current" / "X36_Y28_Z13"
    baseline_dir = tmp_path / "baseline" / "X36_Y28_Z13"
    _write_groot_case(current_dir)
    _write_groot_case(baseline_dir)
    manifest = build_paired_visual_review_manifest(
        build_case_proof_pack(current_dir),
        build_case_proof_pack(baseline_dir),
        task_intent="Confirm migration evidence before human baseline blessing.",
        mode="migration",
    )
    record = {
        "schema_version": "roboharness_visual_review/v1",
        "case_id": "X36_Y28_Z13",
        "reviewer_context": "unit_test",
        "overall_visual_verdict": "PASS",
        "dimensions": [
            {
                "id": dimension["id"],
                "verdict": "PASS",
                "confidence": "medium",
                "evidence": list(dimension["current"]) + list(dimension["baseline"]),
                "rationale": "Current and baseline keyframes match within visual tolerance.",
            }
            for dimension in manifest["dimensions"]
        ],
        "needs_human_reasons": [],
    }

    result = ingest_visual_review_record(manifest, record)

    assert manifest["review_policy"]["allow_automatic_visual_pass"] is False
    assert result.effective_visual_verdict == "NEEDS_HUMAN"
    assert result.summary["needs_human_reasons"] == ["baseline_blessing_required"]


def test_paired_visual_review_manifest_rejects_case_mismatch(tmp_path: Path) -> None:
    current_dir = tmp_path / "current" / "X36_Y28_Z13"
    baseline_dir = tmp_path / "baseline" / "X36_Y28_Z13"
    _write_groot_case(current_dir)
    _write_groot_case(baseline_dir)
    baseline_proof_pack = build_case_proof_pack(baseline_dir)
    baseline_proof_pack = replace(baseline_proof_pack, case_id="OTHER_CASE")

    with pytest.raises(ValueError, match="matching case_id"):
        build_paired_visual_review_manifest(
            build_case_proof_pack(current_dir),
            baseline_proof_pack,
            task_intent="Confirm explicit paired evidence.",
        )


def test_static_visual_review_manifest_path_boundary_rejects_escape(tmp_path: Path) -> None:
    case_dir = tmp_path / "X36_Y28_Z13"
    _write_groot_case(case_dir)
    proof_pack = build_case_proof_pack(case_dir)
    manifest = build_static_visual_review_manifest(
        proof_pack,
        task_intent="Confirm static visual evidence.",
    )
    manifest["dimensions"][0]["current"] = ["../escape.png"]

    with pytest.raises(VisualReviewValidationError, match="escapes its evidence root"):
        validate_visual_review_manifest(manifest, current_root=case_dir)


def test_static_visual_review_manifest_rejects_temporal_dimension_for_v1(
    tmp_path: Path,
) -> None:
    case_dir = tmp_path / "X36_Y28_Z13"
    _write_groot_case(case_dir)
    proof_pack = build_case_proof_pack(case_dir)
    manifest = build_static_visual_review_manifest(
        proof_pack,
        task_intent="Confirm static visual evidence.",
        dimensions=("trajectory_naturalness",),
    )

    with pytest.raises(VisualReviewValidationError, match="temporal"):
        validate_visual_review_manifest(manifest, current_root=case_dir)
