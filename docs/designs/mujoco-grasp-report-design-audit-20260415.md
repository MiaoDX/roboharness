# MuJoCo Grasp Report Design Audit

Date: 2026-04-15

Reviewed artifact:
- `tmp/design_review_report/report.html`

Review branch:
- `codex/design-review-mujoco-20260415`

Runtime used to generate the audited artifact:
- `MUJOCO_GL=egl python examples/mujoco_grasp.py --output-dir tmp/design_review_report --report`

## Outcome

This report was already structurally solid. The real defects were trust breakers in a
diagnostic surface that users rely on during debugging:

1. Embedded Meshcat scenes rendered as dead iframes because the report pointed at the
   wrong relative paths.
2. Mobile and tablet layouts overflowed horizontally because wide tables and the 3D
   viewer were not properly contained.
3. Pass-state metadata contradicted the page's own action copy by still surfacing
   `trajectory_regression` and `restore:plan` hints on successful runs.

The review fixed all three issues.

## Score

- Design score: `C` to `B+`
- AI slop score: `A` to `A`

The report did not need a visual rewrite. It needed the interactions and diagnostics to
be trustworthy.

## Findings

### FINDING-001: Broken embedded 3D scenes

Impact: High

Problem:
- The report shipped iframe `src` values like `plan/meshcat_scene.html`, but the actual
  files lived under `mujoco_grasp/trial_001/...`.
- The page loaded five 404s inside sections labeled `Interactive 3D Scene`.

Why it mattered:
- A dead viewer makes the whole artifact feel unreliable, even when the rest of the
  report is correct.

Fix:
- Compute Meshcat paths relative to the report output root.
- Add iframe titles while touching the embed markup.

Commits:
- `4598c97` `style(design): FINDING-001 - fix embedded meshcat scene links`
- `393b202` `test(design): regression test for FINDING-001`

Files:
- `src/roboharness/reporting.py`
- `tests/test_design_review_meshcat_paths.py`

### FINDING-002: Horizontal overflow on small screens

Impact: High

Problem:
- On a `375px` viewport, the report expanded to `761px`.
- The evaluation table, artifact-pack table, and fixed-width Meshcat viewer forced
  sideways panning.

Why it mattered:
- The report stopped being readable on phones and cramped laptop splits, exactly when a
  debugging artifact should remain easy to scan.

Fix:
- Wrap wide tables in `.table-scroll`.
- Allow code-like tokens to wrap instead of forcing width growth.
- Remove the viewer's hard minimum width and make the Meshcat iframe responsive.

Commits:
- `645313e` `style(design): FINDING-002 - contain report overflow on small screens`
- `b7d36b8` `test(design): regression test for FINDING-002`
- `f99e897` `chore: format design review follow-up`

Files:
- `src/roboharness/reporting.py`
- `tests/test_design_review_report_responsive.py`

### FINDING-003: Pass-state metadata contradicted the UI

Impact: Medium

Problem:
- The main page correctly said `No rerun required`, while the Artifact Pack still showed
  a fake root cause and rerun hint for successful runs.

Why it mattered:
- Pass-state reports should reduce ambiguity, not add it. The bad metadata encouraged
  unnecessary second-guessing by both humans and agents.

Fix:
- Successful manifests now emit `suspected_root_cause="none"`,
  `rerun_hint="not_required"`, and empty evidence paths.
- The rendered summary card now aligns with the pass-state copy.

Commits:
- `54e986e` `style(design): FINDING-003 - clean pass-state artifact metadata`
- `8c7068a` `test(design): regression test for FINDING-003`
- `f99e897` `chore: format design review follow-up`

Files:
- `examples/_mujoco_grasp_wedge.py`
- `tests/test_design_review_pass_state_metadata.py`

## Verification

Browser verification on the regenerated local report confirmed:

- Embedded Meshcat 404s removed.
- Mobile `scrollWidth` reduced from `761` to `375`.
- Pass-state text now shows `Root cause: none` and `Rerun hint: No rerun required`.

Branch verification commands:

```bash
export PYTHONPATH="$PWD/src:$PWD"
ruff check .
ruff format --check .
mypy src/
pytest -q
```

Expected verified result for this branch:
- `466 passed, 9 skipped`
- Coverage: `95.44%`

## Supporting Artifacts

External gstack audit report:
- `~/.gstack/projects/MiaoDX-roboharness/designs/design-audit-20260415/design-audit-mujoco-grasp-report.md`

External screenshot bundle:
- `~/.gstack/projects/MiaoDX-roboharness/designs/design-audit-20260415/screenshots/`

These external artifacts are preserved for gstack cross-session discovery. This file is
the repo-local mirror required by project policy.
