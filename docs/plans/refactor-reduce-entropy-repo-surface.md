---
refactor_scope: reduce-entropy-repo-surface
status: DONE
accepted_severities:
  - P0
  - P1
  - P2
last_verified: 2026-05-20
---

# Refactor Scope: Reduce Entropy Repo Surface

## Status

DONE

## Target

Reduce current repo entropy across the root documentation boundary, bootstrap
truth, stale import paths, and stale CI status notes.

## Accepted Severities

- P1: current guidance that can send agents or contributors down a stale setup,
  test, or source-of-truth path.
- P2: root clutter, stale import paths inside the repo, and historical docs
  that look current from the root.

## Accepted Cleanup Checklist

- [x] Move root planning/history artifacts into `docs/plans/` and update current
  pointers.
- [x] Align bootstrap guidance with the repo's untracked `uv.lock` policy and
  actual CI install commands.
- [x] Migrate G1 locomotion callers to the canonical
  `roboharness.robots.unitree_g1` import surface and remove the old
  `roboharness.controllers.locomotion` module.
- [x] Refresh stale CI strategy status claims.

## Parked Cross-Seam / Future Ideas

- Historical design artifacts may still mention old root paths as part of their
  original record; only current pointers are updated in this pass.

## Evidence Ladder

- L0: stale path/import searches and docs checks.
- L1: targeted unit/integration tests for changed Python imports.

## Stop Condition

The accepted checklist is complete, current path/import searches show only the
cleanup gate's record of removed paths, and targeted tests for the G1 import
migration pass.

## Execution Log

- 2026-05-20: Started reduce-entropy cleanup from `$intuitive-reduce-entropy`
  audit.
- 2026-05-20: Moved root planning/history files into `docs/plans/`, aligned
  bootstrap guidance, migrated in-repo G1 imports to
  `roboharness.robots.unitree_g1`, and refreshed CI strategy status.
- 2026-05-20: Verified with stale-path/import searches, `pytest -q`,
  `ruff check .`, `ruff format --check .`, and `mypy src/`.
- 2026-05-20: Reopened the G1 import cleanup after user confirmation that the
  old import path should not be preserved for this repo. Removed
  `roboharness.controllers.locomotion`, its coverage omit, and
  old-path tests/invalidation.
