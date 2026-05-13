from __future__ import annotations

from pathlib import Path

from roboharness.approval.evidence import EvidenceTarget, resolve_evidence_pairs


def test_resolve_evidence_pairs_handles_full_partial_and_mismatch(tmp_path: Path) -> None:
    current_root = tmp_path / "current"
    baseline_root = tmp_path / "baseline"
    (current_root / "phase").mkdir(parents=True)
    (baseline_root / "phase").mkdir(parents=True)

    (current_root / "phase" / "front.png").write_text("current-front")
    (baseline_root / "phase" / "front.png").write_text("baseline-front")
    (current_root / "phase" / "side.png").write_text("current-side")

    pairs = resolve_evidence_pairs(
        current_root=current_root,
        baseline_root=baseline_root,
        current_label="Meshcat",
        baseline_label="MuJoCo",
        targets=[
            EvidenceTarget(
                phase_id="phase",
                phase_label="Phase",
                view_name="front",
                current_relative_path="phase/front.png",
                baseline_relative_path="phase/front.png",
            ),
            EvidenceTarget(
                phase_id="phase",
                phase_label="Phase",
                view_name="side",
                current_relative_path="phase/side.png",
                baseline_relative_path="phase/side.png",
                missing_baseline_message="MuJoCo evidence missing for phase/side.",
            ),
            EvidenceTarget(
                phase_id="phase",
                phase_label="Phase",
                view_name="top",
                current_relative_path="phase/top.png",
                baseline_relative_path="phase/top.png",
                forced_mismatch_message="Frame ordering drifted across frameworks.",
            ),
        ],
    )

    assert [pair.status for pair in pairs] == ["full", "partial", "mismatch"]
    assert pairs[0].current_label == "Meshcat"
    assert pairs[0].baseline_label == "MuJoCo"
    assert pairs[1].diagnostic_message == "MuJoCo evidence missing for phase/side."
    assert pairs[2].diagnostic_message == "Frame ordering drifted across frameworks."


def test_resolve_evidence_pairs_rejects_path_escape(tmp_path: Path) -> None:
    current_root = tmp_path / "current"
    baseline_root = tmp_path / "baseline"
    current_root.mkdir()
    baseline_root.mkdir()

    [pair] = resolve_evidence_pairs(
        current_root=current_root,
        baseline_root=baseline_root,
        targets=[
            EvidenceTarget(
                phase_id="phase",
                phase_label="Phase",
                view_name="front",
                current_relative_path="../escape.png",
                baseline_relative_path="front.png",
            )
        ],
    )

    assert pair.status == "mismatch"
    assert "escaped the allowed roots" in pair.diagnostic_message
