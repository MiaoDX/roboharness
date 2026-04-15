# Changelog

All notable changes to this project will be documented in this file.

## [0.2.2] - 2026-04-15

### Fixed

- Fixed generated MuJoCo grasp reports so embedded Meshcat scenes resolve from the
  report output root instead of shipping dead iframes.
- Contained wide report tables and the Meshcat viewer on small screens so the report
  no longer forces horizontal scrolling on mobile or split-screen layouts.
- Cleared pass-state artifact metadata so successful runs show `Root cause: none` and
  `Rerun hint: No rerun required` instead of failure-only hints.

### Added

- Added regression tests covering Meshcat embed paths, responsive report containment,
  and pass-state metadata consistency.
- Added a repo-local mirror of the MuJoCo grasp report design audit under
  `docs/designs/`.
