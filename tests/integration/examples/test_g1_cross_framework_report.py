from __future__ import annotations

from pathlib import Path

from examples.g1_cross_framework_report import (
    ASSET_ROOT,
    DEFAULT_BUNDLE_ID,
    build_cross_framework_pairs,
    build_cross_framework_summary_html,
    generate_cross_framework_report,
)


def test_build_cross_framework_pairs_for_committed_bundle_are_full() -> None:
    bundle_dir = ASSET_ROOT / DEFAULT_BUNDLE_ID

    pairs = build_cross_framework_pairs(bundle_dir)

    assert len(pairs) == 13
    assert all(pair.status == "full" for pair in pairs)
    assert pairs[0].current_label == "Meshcat"
    assert pairs[0].baseline_label == "MuJoCo"
    assert pairs[0].phase_label == "Planned"


def test_cross_framework_summary_html_includes_lightbox_and_bundle_copy() -> None:
    bundle_dir = ASSET_ROOT / DEFAULT_BUNDLE_ID

    html = build_cross_framework_summary_html(bundle_dir)

    assert "Cross-Framework Proof" in html
    assert "Meshcat vs MuJoCo" in html
    assert "image-lightbox" in html
    assert "data:image/gif;base64" in html


def test_generate_cross_framework_report_writes_html(tmp_path: Path) -> None:
    report_path = generate_cross_framework_report(tmp_path)

    assert report_path.exists()
    html = report_path.read_text()
    assert "G1 Cross-Framework Proof Surface" in html
    assert DEFAULT_BUNDLE_ID in html
