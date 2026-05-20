from __future__ import annotations

import json
from pathlib import Path

from examples.demos.g1.wbc_reach import (
    G1_VISUAL_REVIEW_PHASE,
    G1_VISUAL_REVIEW_VIEWS,
    build_g1_wbc_visual_review_manifest,
    prepare_g1_wbc_visual_review_package,
)


def test_g1_wbc_visual_review_manifest_is_current_only() -> None:
    manifest = build_g1_wbc_visual_review_manifest()

    assert manifest["case_id"] == "g1_wbc_reach"
    assert manifest["mode"] == "current_only"
    assert manifest["review_policy"]["requires_paired_evidence"] is False
    assert manifest["review_policy"]["allow_automatic_visual_pass"] is False
    assert {dimension["id"] for dimension in manifest["dimensions"]} == {
        "robot_posture",
        "hand_pose",
        "object_relative_position",
        "obvious_collision_or_penetration",
        "task_success_visual_check",
    }


def test_prepare_g1_wbc_visual_review_package_writes_static_pose_files(
    tmp_path: Path,
) -> None:
    trial_dir = tmp_path / "g1_wbc_reach" / "trial_001"
    for view in G1_VISUAL_REVIEW_VIEWS:
        image_path = trial_dir / G1_VISUAL_REVIEW_PHASE / f"{view}_rgb.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_text("evidence")

    package = prepare_g1_wbc_visual_review_package(trial_dir=trial_dir)

    manifest = json.loads(package.manifest_path.read_text())
    assert manifest["mode"] == "current_only"
    assert manifest["metric_summary"]["phase"] == G1_VISUAL_REVIEW_PHASE
    assert package.prompt_path.exists()
    assert package.schema_path.exists()
