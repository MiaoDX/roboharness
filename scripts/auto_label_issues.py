#!/usr/bin/env python3
"""Auto-label open issues based on codebase analysis.

Analyzes all open issues against the current codebase state and applies
appropriate priority labels. Creates missing labels, updates stale labels,
closes fully-addressed issues, and adds context comments.

Usage:
    GITHUB_TOKEN=ghp_xxx python scripts/auto_label_issues.py [--dry-run]

Requires: requests (pip install requests)

Analysis date: 2026-04-10
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field

try:
    import requests
except ImportError:
    print("ERROR: 'requests' is required. Install with: pip install requests")
    sys.exit(1)

import os

REPO = "MiaoDX/roboharness"
API_BASE = "https://api.github.com"


# ---------------------------------------------------------------------------
# Label definitions (to be created if missing)
# ---------------------------------------------------------------------------
LABELS_TO_CREATE = {
    "priority:high": {
        "color": "d73a4a",
        "description": "Clear bug, broken functionality, or security concern",
    },
    "priority:medium": {
        "color": "fbca04",
        "description": "Useful feature or improvement with clear scope, doable in one PR",
    },
    "priority:low": {
        "color": "0e8a16",
        "description": "Nice-to-have, exploratory, or vague scope",
    },
}

# Labels from the old scheme that should be removed when replaced
OLD_PRIORITY_LABELS = {"P0", "P1", "P2", "P3"}


# ---------------------------------------------------------------------------
# Issue action definitions — the core analysis
# ---------------------------------------------------------------------------
@dataclass
class IssueAction:
    """Describes the labeling action for a single issue."""

    number: int
    title: str
    new_label: str
    remove_labels: list[str] = field(default_factory=list)
    close: bool = False
    comment: str | None = None
    rationale: str = ""


ACTIONS: list[IssueAction] = [
    # ── #5 — ManiSkill Gymnasium Wrapper Validation ──────────────────────
    IssueAction(
        number=5,
        title="ManiSkill Gymnasium Wrapper Validation",
        new_label="priority:medium",
        remove_labels=["P1"],
        close=True,
        comment=(
            "Closing — fully addressed by PR #121 (ManiSkill compatibility tests + "
            "reward scalar fix) and commit `9d8e9df` (upstream contribution examples "
            "for ManiSkill).\n\n"
            "Evidence:\n"
            "- `tests/test_maniskill_compat.py` — comprehensive mock-based validation\n"
            "- `examples/contrib_maniskill_visual_debug.py` — example integration\n"
            "- README lists ManiSkill as '✅ Implemented'"
        ),
        rationale=(
            "PR #121 implemented tests, contrib example exists, README confirms done. "
            "All acceptance criteria met."
        ),
    ),
    # ── #10 — English Blog Post ──────────────────────────────────────────
    IssueAction(
        number=10,
        title="English Blog Post",
        new_label="priority:low",
        remove_labels=["P3"],
        comment=(
            "Label updated: P3 → priority:low.\n\n"
            "Current state: PR #125 merged the blog draft at "
            "`blog/why-ai-coding-agents-dont-need-a-separate-vlm-for-robot-debugging.md`. "
            "Remaining work is manual (publish to Medium/dev.to, share on social media)."
        ),
        rationale=(
            "Draft written and merged via PR #125. Only manual publishing remains — "
            "nice-to-have community promotion."
        ),
    ),
    # ── #12 — Academic Citations and Collaboration ───────────────────────
    IssueAction(
        number=12,
        title="Academic Citations and Collaboration",
        new_label="priority:low",
        remove_labels=["P3"],
        comment=(
            "Label updated: P3 → priority:low.\n\n"
            "Exploratory/community task with no codebase dependency. "
            "No changes to project state since filing."
        ),
        rationale=(
            "Vague scope, exploratory. No codebase changes needed. "
            "Existing P3 maps to priority:low."
        ),
    ),
    # ── #67 — Re-integrate roboharness into GR00T WBC grasp project ─────
    IssueAction(
        number=67,
        title="Re-integrate roboharness into GR00T WBC grasp project",
        new_label="priority:medium",
        comment=(
            "Label: priority:medium.\n\n"
            "This is the original validation target (strategic Action 3). The codebase "
            "now has mature SimulatorBackend protocol, TaskProtocol, and examples — "
            "making this integration easier than when filed. Clear scope, doable in one PR."
        ),
        rationale=(
            "Real-world API validation target with clear acceptance criteria. "
            "Codebase maturity (TaskProtocol, refactored backends) makes this easier now."
        ),
    ),
    # ── #69 — Validate SimulatorBackend on unitree_mujoco ────────────────
    IssueAction(
        number=69,
        title="Validate SimulatorBackend on unitree_mujoco",
        new_label="priority:medium",
        comment=(
            "Label: priority:medium.\n\n"
            "Was informally blocked on LeRobot G1 validation (#74), which is now 3/4 "
            "complete. The blocker is effectively resolved — this can proceed. "
            "Well-scoped: one example file + backend validation."
        ),
        rationale=(
            "Blocker (#74) largely resolved (3/4 sub-issues done). "
            "Clear scope: one example + protocol validation. Doable in one PR."
        ),
    ),
    # ── #70 — MCP tool interface for roboharness ─────────────────────────
    IssueAction(
        number=70,
        title="MCP tool interface for roboharness",
        new_label="priority:low",
        comment=(
            "Label: priority:low.\n\n"
            "Foundation implemented via PR #123 (server.py + tools.py, 16 tests, "
            "94.97% coverage). Remaining work depends on external prerequisites: "
            "SKILL.md validation, Constraint Evaluator testing, mujoco-mcp maturity. "
            "Keeping open to track prerequisite completion."
        ),
        rationale=(
            "Core implementation done (PR #123 merged). Remaining prerequisites are "
            "external/exploratory. Low urgency."
        ),
    ),
    # ── #74 — Full LeRobot G1 integration via EnvHub ─────────────────────
    IssueAction(
        number=74,
        title="Full LeRobot G1 integration via EnvHub (with DDS/SDK layer)",
        new_label="priority:low",
        comment=(
            "Label: priority:low.\n\n"
            "3 of 4 sub-issues completed (#81 ✅, #82 ✅, #83 ✅). Only #84 "
            "(upstream SDK import contribution) remains — that's tracked separately. "
            "Core integration is functional: `examples/lerobot_g1_native.py` works, "
            "ONNX controllers integrated, CI runs the example."
        ),
        rationale=(
            "Nearly complete — 3/4 sub-issues done. Remaining sub-issue (#84) is "
            "upstream contribution work tracked in its own issue."
        ),
    ),
    # ── #84 — Contribute conditional SDK imports upstream ─────────────────
    IssueAction(
        number=84,
        title="Contribute conditional SDK imports to lerobot/unitree-g1-mujoco",
        new_label="priority:low",
        comment=(
            "Label: priority:low.\n\n"
            "Upstream contribution task. The roboharness codebase already works around "
            "the SDK import issue (PR #109). This is a nice-to-have improvement for "
            "the upstream HuggingFace repo, not a blocker for roboharness."
        ),
        rationale=(
            "Upstream contribution, not blocking internal work. "
            "Workaround already in place via PR #109."
        ),
    ),
    # ── #91 — Reverse integration into upstream WBC repos ────────────────
    IssueAction(
        number=91,
        title="Reverse integration into upstream WBC repositories",
        new_label="priority:medium",
        comment=(
            "Label: priority:medium.\n\n"
            "Some groundwork done (PR #130 for Rerun/ManiSkill contrib examples, "
            "roboharness org created). Phase 1 targets are well-defined (3 repos) "
            "with clear acceptance criteria (diffs < 100 lines, tests pass). "
            "Good strategic value for project adoption."
        ),
        rationale=(
            "Clear Phase 1 scope with 3 target repos. Strategic value for adoption. "
            "Groundwork started (contrib examples merged, org created)."
        ),
    ),
    # ── #110 — upstream obs-space shape mismatch ─────────────────────────
    IssueAction(
        number=110,
        title="upstream: fix obs-space shape mismatch in lerobot/unitree-g1-mujoco",
        new_label="priority:medium",
        comment=(
            "Label: priority:medium.\n\n"
            "Real bug (shape 97 vs 100 mismatch breaks SyncVectorEnv). Workaround "
            "exists in roboharness (PR #109 patches obs space post-reset). "
            "Upstream fix needed — clear proposed solutions in the issue body."
        ),
        rationale=(
            "Confirmed bug with clear reproduction and proposed fix. "
            "Workaround in place but upstream PR still needed. "
            "Users would hit this when using make_env()."
        ),
    ),
]


# ---------------------------------------------------------------------------
# GitHub API helpers
# ---------------------------------------------------------------------------
def gh_headers(token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def ensure_label_exists(token: str, label: str, props: dict[str, str], *, dry_run: bool) -> None:
    """Create a label if it doesn't already exist."""
    url = f"{API_BASE}/repos/{REPO}/labels/{label}"
    resp = requests.get(url, headers=gh_headers(token), timeout=10)
    if resp.status_code == 200:
        print(f"  ✓ Label '{label}' already exists")
        return
    if dry_run:
        print(f"  [DRY RUN] Would create label '{label}'")
        return
    create_url = f"{API_BASE}/repos/{REPO}/labels"
    payload = {"name": label, "color": props["color"], "description": props["description"]}
    resp = requests.post(create_url, json=payload, headers=gh_headers(token), timeout=10)
    if resp.status_code == 201:
        print(f"  ✓ Created label '{label}'")
    else:
        print(f"  ✗ Failed to create '{label}': {resp.status_code} {resp.text[:200]}")


def add_label(token: str, issue_number: int, label: str, *, dry_run: bool) -> None:
    if dry_run:
        print(f"  [DRY RUN] Would add label '{label}' to #{issue_number}")
        return
    url = f"{API_BASE}/repos/{REPO}/issues/{issue_number}/labels"
    resp = requests.post(url, json={"labels": [label]}, headers=gh_headers(token), timeout=10)
    if resp.status_code == 200:
        print(f"  ✓ Added '{label}' to #{issue_number}")
    else:
        print(f"  ✗ Failed to add '{label}' to #{issue_number}: {resp.status_code}")


def remove_label(token: str, issue_number: int, label: str, *, dry_run: bool) -> None:
    if dry_run:
        print(f"  [DRY RUN] Would remove label '{label}' from #{issue_number}")
        return
    url = f"{API_BASE}/repos/{REPO}/issues/{issue_number}/labels/{label}"
    resp = requests.delete(url, headers=gh_headers(token), timeout=10)
    if resp.status_code in (200, 404):
        print(f"  ✓ Removed '{label}' from #{issue_number}")
    else:
        print(f"  ✗ Failed to remove '{label}' from #{issue_number}: {resp.status_code}")


def add_comment(token: str, issue_number: int, body: str, *, dry_run: bool) -> None:
    if dry_run:
        print(f"  [DRY RUN] Would comment on #{issue_number}")
        return
    url = f"{API_BASE}/repos/{REPO}/issues/{issue_number}/comments"
    resp = requests.post(url, json={"body": body}, headers=gh_headers(token), timeout=10)
    if resp.status_code == 201:
        print(f"  ✓ Commented on #{issue_number}")
    else:
        print(f"  ✗ Failed to comment on #{issue_number}: {resp.status_code}")


def close_issue(token: str, issue_number: int, *, dry_run: bool) -> None:
    if dry_run:
        print(f"  [DRY RUN] Would close #{issue_number}")
        return
    url = f"{API_BASE}/repos/{REPO}/issues/{issue_number}"
    resp = requests.patch(url, json={"state": "closed"}, headers=gh_headers(token), timeout=10)
    if resp.status_code == 200:
        print(f"  ✓ Closed #{issue_number}")
    else:
        print(f"  ✗ Failed to close #{issue_number}: {resp.status_code}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def print_analysis() -> None:
    """Print the full analysis summary."""
    print("\n" + "=" * 72)
    print("ISSUE ANALYSIS SUMMARY — 2026-04-10")
    print("=" * 72)

    for a in ACTIONS:
        status = "CLOSE" if a.close else "LABEL"
        print(f"\n#{a.number} — {a.title}")
        print(f"  Action: {status} → {a.new_label}")
        if a.remove_labels:
            print(f"  Remove: {', '.join(a.remove_labels)}")
        print(f"  Rationale: {a.rationale}")

    print("\n" + "=" * 72)
    print("LABEL SUMMARY")
    print("=" * 72)
    print("  priority:high   (0): — (no issues currently meet this threshold)")
    print("  priority:medium (5): #5 (closing), #67, #69, #91, #110")
    print("  priority:low    (4): #10, #12, #70, #74, #84")
    print("  close           (1): #5 (fully addressed)")
    print("=" * 72 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-label roboharness issues")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without making changes",
    )
    parser.add_argument(
        "--analysis-only",
        action="store_true",
        help="Print analysis summary only, no API calls",
    )
    args = parser.parse_args()

    print_analysis()

    if args.analysis_only:
        return

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("ERROR: Set GITHUB_TOKEN environment variable to apply labels.")
        print("Run with --analysis-only to see the plan without a token.")
        sys.exit(1)

    dry_run = args.dry_run
    if dry_run:
        print("[DRY RUN MODE — no changes will be made]\n")

    # Step 1: Create missing labels
    print("Step 1: Ensuring labels exist...")
    for label, props in LABELS_TO_CREATE.items():
        ensure_label_exists(token, label, props, dry_run=dry_run)
        time.sleep(0.5)

    # Step 2: Process each issue
    print("\nStep 2: Processing issues...")
    for action in ACTIONS:
        print(f"\n--- #{action.number}: {action.title} ---")

        # Add comment first (explains the label change)
        if action.comment:
            add_comment(token, action.number, action.comment, dry_run=dry_run)
            time.sleep(0.5)

        # Remove old labels
        for old_label in action.remove_labels:
            remove_label(token, action.number, old_label, dry_run=dry_run)
            time.sleep(0.3)

        # Add new label
        add_label(token, action.number, action.new_label, dry_run=dry_run)
        time.sleep(0.3)

        # Close if needed
        if action.close:
            close_issue(token, action.number, dry_run=dry_run)
            time.sleep(0.3)

    print("\n✅ Done! All issues processed.")


if __name__ == "__main__":
    main()
