"""Contract tests for the dogfooded roboharness project harness skill."""

from __future__ import annotations

import json
from pathlib import Path

from roboharness.contract import check_project_harness_skill, load_contract_from_file


def test_dogfood_project_harness_skill_is_generated_from_contract() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    skill_dir = repo_root / "agent-skill" / "roboharness-harness"
    contract = load_contract_from_file(skill_dir / "contract.py")

    report = check_project_harness_skill(contract, skill_dir)
    snapshot = json.loads((skill_dir / "contract.snapshot.json").read_text())

    assert report.ok, report.to_dict()
    assert not (repo_root / "SKILL.md").exists()
    assert snapshot["project_slug"] == "roboharness"
    assert {workflow["id"] for workflow in snapshot["workflows"]} == {
        "mujoco_contract_trust_loop",
        "release_truth_alignment",
    }
    assert snapshot["approval_policy"]["human_scope_approval_required"] is True
