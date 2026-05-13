# TODOS

This file captures deferred work from approved planning and review artifacts.

## Completed

### Sync README and docs to the new evidence contract

- **Completed:** PR #201 (2026-04-15)
- Updated `README.md`, `CONTRIBUTING.md`, and `ARCHITECTURE.md` so the MuJoCo grasp
  example explains the paired-evidence summary, explicit evidence states, the
  `failed phase -> proof -> rerun` loop, and the current contributor verification flow.

### Run post-implementation boomerang reviews

- **Completed:** 2026-04-20
- Repo-local boomerang review is mirrored in
  `docs/designs/unattended-refactor-harness-boomerang-review-20260420.md`.
- Outcome: the report design contract held, but the README front door needed an honest
  split between package-first integration and the repo-only MuJoCo wedge demo.

### Build a seeded evaluator corpus before treating the queue as trustworthy

- **Completed:** 2026-04-20
- Added `tests/regression/mujoco_grasp/test_mujoco_grasp_seeded_corpus.py` with seeded `good`, `bad`, and
  `ambiguous` cases for the MuJoCo wedge.
- The corpus locks surfaced-case precision `1.0`, surfaced-case recall `1.0`, and the
  trust boundary that ambiguous still-image evidence must stay review-required.

### Add temporal evidence when still images are ambiguous

- **Completed:** 2026-04-20
- Ambiguous MuJoCo wedge cases now render checkpoint-ordered temporal evidence strips
  in the approval report.
- The first implementation stays wedge-tight: it adds phase-ordered visual context
  without introducing a full clip or video pipeline.

### Add click-to-zoom or lightbox support for evidence cards

- **Completed:** 2026-04-20
- Evidence cards and temporal thumbnails now expand in-place with a keyboard-closeable
  lightbox instead of forcing users into the deeper checkpoint gallery.

### Split the canonical spec from the review log once implementation starts

- **Completed:** 2026-04-20
- `docs/designs/unattended-refactor-harness-v1.md` is the canonical product/design
  contract.
- `showcase-repo-plan.md` remains the reviewed plan and rationale log, and both files
  now state that split explicitly.

### Extract the shared evidence contract once a second stack needs it

- **Completed:** 2026-04-20
- Moved the paired-evidence data model, bounded path resolver, and reusable lightbox
  helpers into `src/roboharness/approval/evidence.py`.
- The extraction is now justified by a second concrete consumer:
  `examples/g1_cross_framework_report.py`, which renders the committed
  `assets/g1/X36_Y28_Z13/` Meshcat vs MuJoCo proof bundle through the same paired-
  evidence contract.

### Revisit prompt-to-contract compilation after presets prove out

- **Completed:** 2026-04-20
- The MuJoCo wedge now supports reviewed preset selection via
  `--contract-preset mujoco_regression_v1|mujoco_migration_guarded_v1`.
- It also supports constrained prompt-assisted authoring via `--contract-prompt`,
  but only to choose among those grounded presets.
- Open-ended visual or freeform NL rule authoring still fails closed and must use
  explicit JSON until a broader compiler is genuinely warranted.
