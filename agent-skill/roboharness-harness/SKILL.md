<!-- Generated from contract.py by roboharness.contract. Do not edit. -->
# Roboharness Harness

## When To Use

Use this harness to review roboharness trust-loop changes where metric-backed proof packs, bounded visual evidence, approval decisions, and release truth must stay aligned with the accepted project contract.

Use this project harness only for checks named in `contract.snapshot.json`. Prompt text in this skill is guidance; `contract.py` is the authority.

## Authority Rules

- Read `contract.snapshot.json` before selecting a workflow.
- Run `roboharness contract check contract.py --output-dir .` before trusting generated instructions.
- Do not invent new review gates, visual dimensions, evidence roots, or approval paths from chat context.
- If a request is outside the approved contract, draft a Harness Scope Brief from `scope-brief-template.md` and ask for user approval before treating new checks as authoritative.
- Ambiguous results: `never_self_promote_to_pass`.
- Out-of-scope requests: `draft_scope_brief_before_contract_change`.

## Workflows

### MuJoCo Contract Trust Loop (`mujoco_contract_trust_loop`)
Review changes to the deterministic MuJoCo grasp proof pack, including metric gates, visual review dimensions, evidence pairing, and approval semantics.

- Phases: Plan (`plan`), Pre-Grasp (`pre_grasp`), Approach (`approach`), Grasp (`grasp`), Lift (`lift`)
- Hard metric gates:
  - `loop_runtime_drift`: `loop_runtime_s_abs_delta lt 1.0` at `all`
  - `approach_center_drift`: `grip_center_error_mm_abs_delta lt 12.0` at `approach`
  - `grasp_gap_drift`: `pinch_gap_error_mm_abs_delta lt 10.0` at `grasp`
  - `lift_contact`: `contact_count ge 1.0` at `lift`
  - `lift_height`: `cube_height_mm gt 5.0` at `lift`
- Visual review dimensions:
  - `hand_pose`: Hand Pose at `grasp` views `front, side`
  - `object_relative_position`: Object Relative Position at `lift` views `front, side`
  - `proof_pack_integrity`: Proof Pack Integrity at `plan` views `front, top`
- Validation commands:
  - `python -m roboharness.cli contract check agent-skill/roboharness-harness/contract.py --output-dir agent-skill/roboharness-harness`
  - `python -m pytest --no-cov tests/regression/mujoco_grasp/test_mujoco_grasp_wedge.py tests/contract/test_approval_evidence.py`
  - `ruff check src/roboharness examples/demos/mujoco tests/contract`

### Release Truth Alignment (`release_truth_alignment`)
Review release-facing changes where pyproject metadata, package version, status, and release documentation must tell the same truth.

- Phases: Release Truth (`release`)
- Validation commands:
  - `python -m roboharness.cli contract check agent-skill/roboharness-harness/contract.py --output-dir agent-skill/roboharness-harness`
  - `python -m pytest --no-cov tests/contract/test_release_truth.py`
  - `ruff check src/roboharness examples/demos/mujoco tests/contract`

## Evidence Boundaries

- `mujoco_trial`: `examples/demos/mujoco/output` (Generated MuJoCo grasp trial proof packs.)
- `mujoco_baseline`: `assets/example_mujoco_grasp` (Blessed baseline and known-bad visual fixtures for the maintained wedge.)
- `release_truth`: `.` (Package metadata, version files, release docs, and status dashboard.)

## Baseline And Approval Policy

- Surface changed cases only: `True`
- Require user blessing for a new baseline: `True`
- Human scope approval required before a proposed contract becomes authoritative: `True`
