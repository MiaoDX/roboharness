# Engineering Review Snapshot: MuJoCo Alarmed Grasp Loop

Generated during `/plan-eng-review` on 2026-04-14  
Review target: [`docs/designs/mujoco-alarmed-grasp-loop.md`](./mujoco-alarmed-grasp-loop.md)  
Status: IN PROGRESS  
Design branch: `dongxu/dev-0412-02`  
Current review branch when mirrored: `dongxu/dev-0414-1`

## Purpose

This file is the durable engineering-review record for the approved MuJoCo alarmed
grasp-loop wedge. It exists so implementation can resume from repo-local artifacts
instead of depending on transient chat context.

## Locked Decisions

### 1A. Scope

Reduce the first implementation to one complete MuJoCo wedge only.

- Emit `autonomous_report.json`, `alarms.json`, and `phase_manifest.json`.
- Reuse the existing evaluator, report renderer, and CI distribution path.
- Defer bigger report-platform redesign and cross-stack rollout.

### 2A. Baseline Strategy

Use one blessed deterministic baseline fixture, not rolling history.

- The first slice optimizes for reproducibility and debuggability.
- Trend/history-based baselines can come later if the wedge proves useful.

### 3A. Source of Truth

Treat `autonomous_report.json` as the single source of truth.

- Evaluator verdicts derive from it.
- `alarms.json` derives from evaluator output.
- HTML consumes the same evaluation result instead of inventing a separate path.

### 4A. Phase IDs

Keep canonical internal phase IDs aligned with the existing protocol.

- Use built-in IDs such as `plan` and `grasp` internally.
- Aliases such as `plan_start` or `contact` are display-only copy for UI or manifest
  presentation when needed.
- Do not rename internal protocol identifiers for this wedge.

### 5A. Where Artifact-Pack Assembly Lives

Keep artifact-pack assembly local to the MuJoCo example for now.

- Only extract strictly necessary renderer hooks into shared `src/` code.
- Do not generalize the full packaging pipeline before the MuJoCo wedge lands.

### 6A. Verdict Path

Use one verdict path only.

- Replace `assert_grasp_success()` as the primary gate with evaluator-backed validation.
- `--assert-success`, alarms, and HTML all consume the same `EvaluationResult`.

### 7A. DRY Boundary

Extract the duplicated MuJoCo grasp fixture and phase script into one small example-side
helper module.

- Shared by `examples/mujoco_grasp.py`.
- Shared by `examples/mujoco_rerun.py`.
- Shared by `examples/contrib_rerun_robotics_viz.py`.

### 8A. Machine-Readable Artifact Types

Add small local dataclasses with `to_dict()` for new machine-readable artifacts.

- Keep them example-side only for now.
- Do not promote them into the public `src/` API yet.
- Target artifacts are `alarms.json` and `phase_manifest.json`.

## Key Findings Already Established

- The design doc uses semantic phase names like `plan_start` and `contact`, but the
  current protocol already has canonical internal names like `plan` and `grasp`.
  Resolution: aliases are allowed in presentation only; internals stay unchanged.
- The current CI artifact path and example output path are mismatched:
  `.github/workflows/ci.yml` uploads `harness_output/report.html`, while
  `examples/mujoco_grasp.py` currently writes `harness_output/mujoco_grasp_report.html`.
  The eventual implementation should align those paths.
- The distribution plan should stay boring:
  existing GitHub Actions artifacts plus the existing Pages flow, with no new
  publishing mechanism introduced for this wedge.

## Code Anchors

These files were identified as the main implementation and test anchors during review:

- `docs/designs/mujoco-alarmed-grasp-loop.md`
- `src/roboharness/reporting.py`
- `src/roboharness/evaluate/result.py`
- `src/roboharness/evaluate/assertions.py`
- `src/roboharness/storage/history.py`
- `src/roboharness/mcp/tools.py`
- `examples/mujoco_grasp.py`
- `examples/mujoco_rerun.py`
- `examples/contrib_rerun_robotics_viz.py`
- `examples/sonic_tracking.py`
- `tests/test_assertions.py`
- `tests/test_reporting.py`
- `tests/test_mcp_tools.py`
- `tests/test_sonic_tracking_example.py`

## Remaining Review Work

The engineering review is not fully closed yet. The remaining work items are:

1. Finish the Test Review section.
2. Produce the required ASCII coverage diagram.
3. Write the QA test-plan artifact required by the review workflow.
4. Ask and resolve issue `9` if a real testing tradeoff remains.
5. Finish Performance Review.
6. Close out the review with `NOT in scope`, `What already exists`, failure modes,
   worktree parallelization strategy, review log, dashboard, and completion summary.

## Implementation Start Rules

If implementation starts from a fresh session:

1. Read the design doc at `docs/designs/mujoco-alarmed-grasp-loop.md`.
2. Read this engineering review snapshot.
3. Resume the latest checkpoint under `~/.gstack/projects/MiaoDX-roboharness/checkpoints/`.
4. Treat the decisions above as locked unless a new review decision explicitly changes them.

