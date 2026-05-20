# Agent Visual Review v1

Date: 2026-05-20
Status: Design contract for implementation planning

This document records the v1 design for using the visual capability of top-tier
coding agents to review robot simulation proof packs. The goal is not to attach
a separate VLM service. The goal is to make roboharness artifacts easy for a
multimodal coding agent to evaluate in a bounded, reproducible, metric-first
way.

## Goal

Roboharness should reduce human involvement in robot simulation testing by
letting a coding agent visually review the proof pack when metrics alone are
not enough.

The review strategy is metric-first:

- use deterministic metrics wherever reliable metrics exist
- reserve agent visual review for necessary judgments that are hard to
  metricize
- keep human escalation explicit and narrow
- fail closed when evidence, schema, or review output is not trustworthy

## Terminology

- **Visual Harness**: the overall roboharness system for producing visual and
  metric evidence from robot simulation runs.
- **Proof Pack**: the compact evidence bundle containing visuals, metrics,
  baseline comparison, and decision metadata.
- **Agent Visual Review**: structured visual review performed by a multimodal
  coding agent against a bounded proof pack and manifest.
- **Visual Reviewer Invocation**: one isolated call/session that reads the
  review package and writes a structured review record.
- **Visual Review Manifest**: the minimal structured input for the reviewer.
- **Visual Review Record**: the structured JSON output from the reviewer.

## Non-Goals

v1 does not include:

- a built-in model provider integration
- a separate VLM service
- full trajectory or motion-window recording
- automatic baseline blessing
- visual pass for locomotion, SONIC tracking, smoothness, or temporal behavior
- free-form success judgment over an entire HTML report

## First Integrations

v1 supports two first integrations with different approval authority.

### MuJoCo Grasp

MuJoCo grasp is the canonical paired approval path.

It already has:

- deterministic metric gates
- blessed baseline evidence
- paired current-vs-baseline proof panels
- approval report semantics
- seeded good/bad/ambiguous corpus coverage

When hard metrics pass and all required visual dimensions pass with acceptable
confidence, MuJoCo grasp may produce automatic approval unless another approval
policy requires human escalation.

### G1 WBC Reach

G1 WBC reach is the first humanoid visual review target.

It starts as a static-pose review path only. It may evaluate robot posture,
hand pose, object-relative position, obvious collision, and task-success visual
agreement. It should be more conservative than MuJoCo grasp unless paired
baseline evidence and safe dimensions are available.

G1 locomotion and SONIC-style motion tracking remain outside automatic visual
pass until motion-window evidence exists.

## Review Flow

v1 uses a two-step review flow.

### 1. Prepare Review Package

Roboharness prepares:

- `visual_review_manifest.json`
- `visual_review_prompt.md`
- `visual_review_schema.json`
- selected evidence paths or copied evidence inputs

The prepare step validates the manifest before review. If a required visual
dimension lacks metricization rationale, evidence, or policy grounding, package
preparation fails.

### 2. Agent Writes Review Record

An agent invocation reads the prepared package and writes:

- `visual_review.json`

This invocation may be run by a subagent, a new CLI/tmux session, CI, or an MCP
workflow. The core roboharness package does not bind to a provider in v1.

### 3. Ingest Review Record

Roboharness validates `visual_review.json` against the schema and manifest,
then adds a summary to `approval_report.json`.

The raw review record remains separate for auditability.

## Manifest Requirements

The visual review manifest is the reviewer boundary. The reviewer must not
browse arbitrary files or infer task criteria outside the manifest.

Required manifest fields:

```json
{
  "schema_version": "roboharness_visual_review_manifest/v1",
  "case_id": "deterministic_mujoco_grasp",
  "mode": "regression",
  "task_intent": "The cube should be grasped and lifted with a plausible gripper pose.",
  "dimensions": [],
  "metric_summary": {},
  "review_policy": {}
}
```

Each visual dimension must declare:

- `id`
- `required`
- `phase`
- `evidence_type`
- `views`
- `current`
- `baseline` when paired review is required
- `metric_fallback` or `why_not_metricized`

Example dimension:

```json
{
  "id": "hand_pose",
  "required": true,
  "phase": "lift",
  "evidence_type": "paired_keyframe",
  "views": ["side", "top"],
  "current": ["lift/side_rgb.png", "lift/top_rgb.png"],
  "baseline": ["lift/side_rgb.png", "lift/top_rgb.png"],
  "metric_fallback": ["grip_center_error_mm", "pinch_gap_error_mm"],
  "why_not_metricized": "The exact visual hand orientation is not fully captured by the current numeric grasp metrics."
}
```

## Metricization Gate

Every visual review dimension must justify why the model is being asked to look
at images.

Prepare-time validation fails when:

- a required dimension has neither `metric_fallback` nor `why_not_metricized`
- an optional dimension lacks rationale but participates in verdict aggregation
- a dimension references evidence that does not exist
- a regression dimension lacks baseline evidence
- a temporal dimension requests automatic pass without motion-window evidence

This keeps visual review from becoming the default path for checks that should
be metrics.

## Supported v1 Dimensions

v1 supports static-pose dimensions only:

- `robot_posture`
- `hand_pose`
- `object_relative_position`
- `obvious_collision_or_penetration`
- `task_success_visual_check`

Out of scope for automatic visual pass:

- `trajectory_naturalness`
- `smoothness`
- `contact_sequence`
- `late_sharp_motion`
- `locomotion_quality`
- general "looks good" scoring

Temporal dimensions must return `INSUFFICIENT_EVIDENCE` or `NEEDS_HUMAN` until
motion-window evidence exists.

## Review Record Schema

The reviewer writes structured JSON.

```json
{
  "schema_version": "roboharness_visual_review/v1",
  "case_id": "deterministic_mujoco_grasp",
  "reviewer_context": "agent_visual_review",
  "overall_visual_verdict": "PASS",
  "dimensions": [
    {
      "id": "hand_pose",
      "verdict": "PASS",
      "confidence": "medium",
      "evidence": ["lift/side_rgb.png", "lift/top_rgb.png"],
      "rationale": "The gripper appears closed around the cube with no obvious side-slip posture."
    }
  ],
  "needs_human_reasons": []
}
```

Dimension verdict values:

- `PASS`
- `FAIL`
- `INSUFFICIENT_EVIDENCE`
- `NEEDS_HUMAN`
- `NOT_APPLICABLE`

Confidence values:

- `high`
- `medium`
- `low`

Numeric confidence is not used.

Human escalation reasons must come from a fixed taxonomy:

- `missing_required_evidence`
- `view_conflict`
- `low_confidence_high_risk`
- `baseline_blessing_required`
- `migration_intent_confirmation_required`
- `unsupported_temporal_dimension`
- `current_only_review_cannot_auto_pass`

## Evidence Boundary

The review record may reference only evidence paths declared in the manifest.

Invalid evidence references make the review record invalid. The reviewer must
not browse the artifact directory and choose its own screenshots. If desired
evidence is missing, the reviewer must output `INSUFFICIENT_EVIDENCE` with a
human escalation reason.

## Aggregation Policy

Hard metrics remain the safety floor.

Rules:

- metric `FAIL` + visual `PASS` -> approval `FAIL`
- metric `PASS` + required visual `FAIL` -> approval `FAIL`
- metric `PASS` + required visual `INSUFFICIENT_EVIDENCE` -> `NEEDS_HUMAN`
- metric `PASS` + required visual `NEEDS_HUMAN` -> `NEEDS_HUMAN`
- metric `PASS` + required visual `PASS` with low confidence -> `NEEDS_HUMAN`
- metric `PASS` + all required visual dimensions `PASS` with medium/high
  confidence -> visual approval passes
- migration visual `PASS` never blesses a baseline automatically

Visual review can veto or escalate a metric-passing run. It cannot rescue a
hard metric failure.

## Approval Report Integration

`visual_review.json` and `visual_review_manifest.json` remain separate files.

`approval_report.json` includes only a summary:

```json
{
  "visual_review": {
    "manifest_path": "visual_review_manifest.json",
    "record_path": "visual_review.json",
    "overall_visual_verdict": "PASS",
    "blocking_dimensions": [],
    "needs_human_reasons": []
  }
}
```

The approval report should not duplicate full reviewer rationale.

## Invalid Review Handling

If `visual_review.json` cannot be trusted, the approval-level verdict is
`REVIEW_INVALID`.

Use `REVIEW_INVALID` for:

- schema-version mismatch
- missing required fields
- illegal enum values
- evidence references outside the manifest
- `case_id` mismatch
- dimensions not declared in the manifest
- invalid confidence values

`REVIEW_INVALID` is distinct from `CONTRACT_INVALID`. The contract may be valid
while the review output is unusable.

## Current-Only Review

When baseline evidence is unavailable, v1 may run a limited current-only visual
check.

Current-only review may find obvious failures:

- collapsed robot
- hand nowhere near object
- obvious collision or penetration
- visibly impossible final pose

Current-only review cannot produce final automatic pass. If hard metrics pass
and no visual failure is found, final approval still escalates with
`current_only_review_cannot_auto_pass`.

## Prompt Rules

The visual reviewer prompt must instruct the agent to:

- answer only manifest-declared dimensions
- avoid inventing task criteria
- avoid inferring unseen motion from still frames
- avoid using implementation intent to fill evidence gaps
- return `INSUFFICIENT_EVIDENCE` when evidence is inadequate
- return `NEEDS_HUMAN` when semantic approval is outside the model's authority
- keep each rationale to one sentence

## Acceptance Criteria

The v1 implementation is complete when:

- MuJoCo grasp can prepare a visual review manifest, prompt, and schema
- G1 WBC reach can prepare a static-pose visual review manifest
- review records are schema-validated
- review evidence references are checked against the manifest
- invalid review records produce `REVIEW_INVALID`
- approval reports include a visual review summary
- metric failures cannot be rescued by visual review
- low-confidence required passes escalate
- current-only review cannot auto-pass
- tests cover pass, fail, insufficient evidence, needs-human, invalid review,
  and current-only cases

## Later Work

Later phases may add:

- motion-window evidence
- provider-specific invocation helpers
- MCP tools for prepare/record/apply review
- visual review benchmark corpus across model families
- locomotion and SONIC visual review once temporal evidence exists
