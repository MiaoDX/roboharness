from __future__ import annotations

import json
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
    build_case_proof_pack,
    build_static_visual_review_manifest,
    load_case_proof_pack,
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
