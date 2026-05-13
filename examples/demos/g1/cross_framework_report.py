#!/usr/bin/env python3
"""Render a static cross-framework G1 proof surface from committed assets."""

from __future__ import annotations

import argparse
import html
import re
from dataclasses import dataclass
from pathlib import Path

from roboharness.approval.evidence import (
    EvidencePair,
    EvidenceTarget,
    render_lightbox_shell,
    render_zoomable_image,
    resolve_evidence_pairs,
)
from roboharness.reporting import generate_html_report

ASSET_ROOT = Path(__file__).resolve().parents[3] / "assets" / "g1"
DEFAULT_BUNDLE_ID = "X36_Y28_Z13"
TASK_NAME = "g1_cross_framework"
CURRENT_LABEL = "Meshcat"
BASELINE_LABEL = "MuJoCo"
_PHASE_IMAGE_RE = re.compile(r"(?P<index>\d+)_(?P<phase>.+)_(?P<view>[^_]+)\.png$")


@dataclass(frozen=True)
class PhaseAsset:
    """One committed phase image belonging to a renderer-specific stack."""

    index: int
    phase_slug: str
    view_name: str
    relative_path: str


def build_cross_framework_pairs(bundle_dir: Path) -> list[EvidencePair]:
    """Resolve phase-ordered Meshcat vs MuJoCo evidence pairs for one asset bundle."""
    meshcat_assets = _load_phase_assets(bundle_dir, "meshcat")
    mujoco_assets = _load_phase_assets(bundle_dir, "mujoco")
    frame_ids = sorted(set(meshcat_assets) | set(mujoco_assets))
    if not frame_ids:
        raise ValueError(f"no cross-framework PNG frames found under {bundle_dir}")

    targets: list[EvidenceTarget] = []
    for frame_id in frame_ids:
        meshcat = meshcat_assets.get(frame_id)
        mujoco = mujoco_assets.get(frame_id)
        reference = meshcat or mujoco
        if reference is None:
            continue
        phase_slug = reference.phase_slug
        mismatch = _phase_mismatch_message(frame_id, meshcat, mujoco)
        meshcat_rel = (
            meshcat.relative_path
            if meshcat is not None
            else f"meshcat/{frame_id:02d}_{phase_slug}_front2back.png"
        )
        mujoco_rel = (
            mujoco.relative_path
            if mujoco is not None
            else f"mujoco/{frame_id:02d}_{phase_slug}_top2down.png"
        )
        targets.append(
            EvidenceTarget(
                phase_id=f"{frame_id:02d}_{phase_slug}",
                phase_label=_phase_label(phase_slug),
                view_name=_paired_view_name(meshcat, mujoco),
                current_relative_path=meshcat_rel,
                baseline_relative_path=mujoco_rel,
                forced_mismatch_message=mismatch,
                missing_current_message=(
                    f"{CURRENT_LABEL} evidence missing for frame {frame_id:02d} "
                    f"({_phase_label(phase_slug)})."
                ),
                missing_baseline_message=(
                    f"{BASELINE_LABEL} evidence missing for frame {frame_id:02d} "
                    f"({_phase_label(phase_slug)})."
                ),
                empty_message=(
                    f"No cross-framework evidence available for frame {frame_id:02d} "
                    f"({_phase_label(phase_slug)})."
                ),
            )
        )

    return resolve_evidence_pairs(
        current_root=bundle_dir,
        baseline_root=bundle_dir,
        targets=targets,
        current_label=CURRENT_LABEL,
        baseline_label=BASELINE_LABEL,
        caption_builder=_build_pair_caption,
    )


def build_cross_framework_summary_html(bundle_dir: Path) -> str:
    """Build the summary block inserted into the shared HTML report shell."""
    pairs = build_cross_framework_pairs(bundle_dir)
    preview = _render_preview(bundle_dir)
    cards = "".join(_render_pair_card(pair) for pair in pairs)
    summary_counts = _render_counts_strip(
        comparable=sum(pair.status == "full" for pair in pairs),
        partial=sum(pair.status == "partial" for pair in pairs),
        mismatched=sum(pair.status == "mismatch" for pair in pairs),
        total=len(pairs),
    )
    return (
        '<section class="approval-queue">'
        "<div>"
        "<h3>Cross-Framework Proof</h3>"
        "<p>The same committed G1 bundle shown through kept Meshcat and MuJoCo review angles.</p>"
        "</div>"
        f'<div class="queue-list">{preview}</div>'
        "</section>"
        f"{summary_counts}"
        '<section class="evidence-section">'
        '<div class="evidence-section-head">'
        "<div>"
        "<h3>Meshcat vs MuJoCo</h3>"
        f"<p>Bundle <code>{html.escape(bundle_dir.name)}</code> in committed phase order.</p>"
        "</div>"
        '<p class="evidence-state-label">cross-framework/full review</p>'
        "</div>"
        '<div class="evidence-banner evidence-banner-full">'
        "<p>Every card compares the same G1 phase across the kept review angles. "
        "This is a proof surface, not a simulator rerun.</p>"
        "</div>"
        f'<div class="evidence-grid">{cards}</div>'
        "</section>"
        f"{render_lightbox_shell()}"
    )


def generate_cross_framework_report(
    output_dir: Path,
    *,
    bundle_id: str = DEFAULT_BUNDLE_ID,
) -> Path:
    """Generate the self-contained HTML report for one committed G1 asset bundle."""
    bundle_dir = ASSET_ROOT / bundle_id
    if not bundle_dir.is_dir():
        raise FileNotFoundError(f"G1 asset bundle not found: {bundle_dir}")

    (output_dir / TASK_NAME / "trial_001").mkdir(parents=True, exist_ok=True)
    return generate_html_report(
        output_dir,
        TASK_NAME,
        title="Roboharness: G1 Cross-Framework Proof Surface",
        subtitle=(
            "Committed Unitree G1 evidence aligned across Meshcat and MuJoCo. "
            "Useful for verifying that the same phase progression stays legible across frameworks."
        ),
        accent_color="#0f766e",
        summary_html=build_cross_framework_summary_html(bundle_dir),
        footer_text=(
            "Generated by "
            "<code>examples/demos/g1/cross_framework_report.py "
            f"--bundle-id {html.escape(bundle_id)}</code>"
        ),
        meshcat_mode="none",
    )


def _load_phase_assets(bundle_dir: Path, stack_name: str) -> dict[int, PhaseAsset]:
    stack_dir = bundle_dir / stack_name
    if not stack_dir.is_dir():
        raise FileNotFoundError(f"missing stack directory: {stack_dir}")
    assets: dict[int, PhaseAsset] = {}
    for image_path in sorted(stack_dir.glob("*.png")):
        match = _PHASE_IMAGE_RE.fullmatch(image_path.name)
        if match is None:
            raise ValueError(f"unexpected phase image name: {image_path.name}")
        frame_id = int(match.group("index"))
        assets[frame_id] = PhaseAsset(
            index=frame_id,
            phase_slug=match.group("phase"),
            view_name=match.group("view"),
            relative_path=f"{stack_name}/{image_path.name}",
        )
    return assets


def _phase_label(phase_slug: str) -> str:
    return phase_slug.replace("_", " ").title()


def _paired_view_name(meshcat: PhaseAsset | None, mujoco: PhaseAsset | None) -> str:
    left = meshcat.view_name if meshcat is not None else "front2back"
    right = mujoco.view_name if mujoco is not None else "top2down"
    return f"{left} vs {right}"


def _phase_mismatch_message(
    frame_id: int,
    meshcat: PhaseAsset | None,
    mujoco: PhaseAsset | None,
) -> str | None:
    if meshcat is None or mujoco is None:
        return None
    if meshcat.phase_slug == mujoco.phase_slug:
        return None
    return (
        f"Frame {frame_id:02d} is not phase-aligned across frameworks: "
        f"{CURRENT_LABEL} uses {meshcat.phase_slug!r}, "
        f"{BASELINE_LABEL} uses {mujoco.phase_slug!r}."
    )


def _build_pair_caption(target: EvidenceTarget, status: str) -> str:
    if status == "partial":
        return (
            f"{target.phase_label} is only partially comparable across {CURRENT_LABEL} and "
            f"{BASELINE_LABEL} because one side of the pair is missing."
        )
    if status == "empty":
        return f"{target.phase_label} is currently missing on both framework surfaces."
    if status == "mismatch":
        return (
            f"{target.phase_label} could not be paired cleanly because the committed "
            "frame ordering drifted across frameworks."
        )
    return (
        f"{target.phase_label} stays phase-aligned across the kept {CURRENT_LABEL} and "
        f"{BASELINE_LABEL} review angles."
    )


def _render_preview(bundle_dir: Path) -> str:
    preview_gif = bundle_dir / "g1_meshcat_mujoco_comparison.gif"
    preview_body = ""
    if preview_gif.exists():
        preview_image = render_zoomable_image(
            preview_gif,
            alt="Phase-ordered G1 comparison across Meshcat and MuJoCo",
            caption="Phase-ordered G1 comparison across Meshcat and MuJoCo",
        )
        preview_body = (
            '<figure class="evidence-figure">'
            f"{preview_image}"
            "<figcaption>Phase-ordered G1 preview across both frameworks.</figcaption>"
            "</figure>"
        )
    return (
        '<article class="queue-card queue-card-surfaced">'
        '<div class="queue-card-head">'
        '<span class="queue-badge">Proof Pack</span>'
        f'<span class="queue-badge queue-badge-case">{html.escape(bundle_dir.name)}</span>'
        "</div>"
        "<p><strong>Why this exists:</strong> the shared paired-evidence contract is now used "
        "outside the MuJoCo grasp wedge, on a committed G1 cross-framework bundle.</p>"
        f"{preview_body}"
        "</article>"
    )


def _render_counts_strip(
    *,
    comparable: int,
    partial: int,
    mismatched: int,
    total: int,
) -> str:
    counts = (
        ("Comparable", comparable),
        ("Partial", partial),
        ("Mismatched", mismatched),
        ("Total", total),
    )
    pills = "".join(
        (f'<div class="count-pill"><span>{html.escape(label)}</span><strong>{count}</strong></div>')
        for label, count in counts
    )
    return f'<section class="counts-strip">{pills}</section>'


def _render_pair_card(pair: EvidencePair) -> str:
    diagnostic = (
        f'<p class="evidence-diagnostic">{html.escape(pair.diagnostic_message)}</p>'
        if pair.diagnostic_message
        else ""
    )
    return (
        f'<article class="evidence-card evidence-card-{html.escape(pair.status)}">'
        '<div class="evidence-card-head">'
        f'<span class="evidence-badge">Phase {html.escape(pair.phase_label)}</span>'
        f'<span class="evidence-badge">View {html.escape(pair.view_name)}</span>'
        f'<span class="evidence-badge evidence-badge-status">{html.escape(pair.status)}</span>'
        "</div>"
        '<div class="evidence-compare-grid">'
        f"{_render_pair_media(pair, side='current')}"
        f"{_render_pair_media(pair, side='baseline')}"
        "</div>"
        f'<p class="evidence-caption">{html.escape(pair.interpretation_caption)}</p>'
        f"{diagnostic}"
        "</article>"
    )


def _render_pair_media(pair: EvidencePair, *, side: str) -> str:
    if side == "current":
        image_path = pair.current_image_path
        label = pair.current_label
    else:
        image_path = pair.baseline_image_path
        label = pair.baseline_label

    fallback = (
        pair.diagnostic_message or f"{label} image missing for {pair.phase_id}/{pair.view_name}."
    )
    if image_path is not None and image_path.exists():
        caption = f"{label} {pair.phase_label} / {pair.view_name}"
        return (
            '<figure class="evidence-figure">'
            f'<div class="evidence-role">{html.escape(label)}</div>'
            f"{render_zoomable_image(image_path, alt=caption, caption=caption)}"
            f"<figcaption>{html.escape(caption)}</figcaption>"
            "</figure>"
        )
    return (
        '<figure class="evidence-figure">'
        f'<div class="evidence-role">{html.escape(label)}</div>'
        f'<div class="evidence-placeholder" role="img" aria-label="{html.escape(fallback)}">'
        f"{html.escape(label)}"
        "</div>"
        f"<figcaption>{html.escape(fallback)}</figcaption>"
        "</figure>"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a static G1 cross-framework proof surface from committed assets"
    )
    parser.add_argument(
        "--bundle-id",
        default=DEFAULT_BUNDLE_ID,
        help=f"G1 asset bundle id under assets/g1/ (default: {DEFAULT_BUNDLE_ID})",
    )
    parser.add_argument(
        "--output-dir",
        default="./harness_output",
        help="Output directory for the generated HTML report (default: ./harness_output)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    report_path = generate_cross_framework_report(output_dir, bundle_id=args.bundle_id)
    print(f"[g1-cross-framework] wrote report: {report_path}")


if __name__ == "__main__":
    main()
