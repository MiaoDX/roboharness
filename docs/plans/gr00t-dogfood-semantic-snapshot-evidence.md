# GR00T Dogfood: Semantic Snapshot Evidence

Status: active

## Context

GR00T whole-body control built a mature visual harness around semantic phase
snapshots, deterministic replay rendering, renderer trustworthiness reports,
and autonomous visual reports. That success should move the reusable evidence
language back into Roboharness instead of leaving each downstream repository to
rebuild the same layer.

This is dogfooding in the strict sense: GR00T should be able to use
Roboharness for the visual evidence lifecycle without Roboharness taking over
GR00T's robot runtime, task planner, control backends, safety startup, or
renderer implementation.

## Accepted Direction

- Roboharness owns public artifact concepts for semantic snapshot bundles,
  renderer reports, and autonomous evidence reports.
- GR00T visual harness dogfooding treats `roboharness.evidence` as a hard
  artifact-layer dependency, not an optional fallback. During local development,
  GR00T installs the sibling Roboharness checkout into its active `.venv`;
  after the API supports the repo's visual harness, GR00T can switch to a `uv`
  git-main dependency.
- `SimulatorBackend.step()` remains one useful ingestion path, not the only
  Roboharness model.
- Downstream projects own evidence producers: runtime sessions, robot-specific
  execution, semantic snapshot recording, and renderer implementations.
- Roboharness owns reusable artifact validation, JSON round-tripping,
  proof-pack assembly inputs, baseline/approval surfaces, and bounded visual
  review preparation.
- GR00T compatibility is a first acceptance gate, but the first Roboharness
  slice uses small fixtures rather than importing or depending on the GR00T
  repository.

## Non-Goals

- Do not migrate the GR00T runner in this slice.
- Do not add a new simulator backend.
- Do not change MuJoCo wedge approval semantics.
- Do not make `HarnessContract` a runtime or renderer configuration system.
- Do not bless new baselines or alter human escalation policy.

## First Slice

1. Add a public `roboharness.evidence` package with typed artifact models and
   JSON helpers.
2. Support GR00T-style `snapshot_bundle.json` and renderer report shapes through
   round-trip tests.
3. Export the new artifact API from package boundaries without importing
   optional simulator dependencies.
4. Record the architecture direction in `CONTEXT.md` and ADR 0002.

## Acceptance Gates

- A core-package import still works without optional simulator extras.
- Unit tests prove semantic snapshot bundles, renderer reports, and autonomous
  evidence reports can be parsed and re-emitted without losing unknown
  downstream fields.
- Focused lint passes for touched Python files.

## Intuitive Flow Precheck

- Scope: keep this slice to public artifact models, JSON helpers, exports, and
  fixture tests. Do not migrate downstream runners.
- Risks: preserve unknown downstream fields so GR00T can adopt the API without
  losing project-specific metrics, runtime, plan, or renderer metadata.
- Tests: add focused unit tests for GR00T-style snapshot bundles, renderer
  reports, autonomous evidence reports, and core package import without
  optional simulator extras.
- DX: expose the API from `roboharness.evidence` and top-level package exports
  so downstream contracts can import it without depending on examples.
- Execution: implement in one bounded slice, then run focused tests and lint on
  touched Python files.

## Later Slices

- [x] Add a GR00T dogfood gate that installs Roboharness explicitly and fails
  when `roboharness.evidence` is missing or stale.
- [x] Generate visual review manifests from evidence bundles and contract
  dimensions.
- [x] Add proof-pack assembly helpers that consume autonomous evidence reports.
- [x] Teach a GR00T project harness skill to use the new Roboharness evidence
  API.
- Migrate GR00T visual harness code only after fixture compatibility and public
  boundaries are stable.

## Follow-Up Slice: Proof Pack And Static Review

Implemented after the first artifact-model slice:

- `roboharness.evidence` now exposes case-level proof-pack assembly for
  downstream visual harness case directories.
- The first proof-pack assembler consumes `autonomous_report.json`,
  `snapshot_bundle.json`, and renderer `report.json` files without importing
  GR00T or owning downstream runtime execution.
- Static visual-review manifest generation is current-only and selects
  case-local keyframes from the proof pack. It sets
  `allow_automatic_visual_pass=false`, so current-only review can veto or
  escalate but cannot bless an automatic visual pass.
- The `roboharness proof-pack` CLI writes `proof_pack.json` and optionally a
  validated `visual_review_manifest.json`.
- GR00T owns the project-specific dogfood gate and generated project harness
  contract under its `skills/visual-harness/roboharness/` directory.
