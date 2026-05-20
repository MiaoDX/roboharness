# Unattended Refactor Harness v1

Date: 2026-04-17
Status: Canonical product/design contract. Approved direction from CEO-review follow-up discussion.

`docs/plans/unattended-refactor-harness-plan.md` remains the reviewed strategy
and rationale log for this direction.

This document records the accepted product contract for unattended Claude/Codex
refactor and migration runs in roboharness.

## North Star

Roboharness should compress:

`2-3 hours unattended agent work -> 3 minutes final approval review`

The user should not need to monitor the run live. The system should return a small,
trustworthy proof pack that makes the final decision legible.

## Accepted Decisions

1. Natural-language prompts are not the source of truth.
2. Before a run starts, the prompt is compiled into a tidy JSON contract.
3. If the contract compiler cannot ground a rule safely, the run must stop and ask
   the user before execution.
4. Two explicit run modes exist: `regression` and `migration`.
5. Metrics provide the hard safety floor.
6. Visual evidence provides intent proof and the human trust surface.
7. `AMBIGUOUS` may trigger more evidence gathering, but it can never self-pass.
8. The first screen is a changed-cases approval queue, not a full dashboard.
9. Cases with no material change should not appear on the first screen.
10. Migration runs may propose a new baseline, but the user blesses it once at the
    final approval boundary.

## Compile-Time Behavior

The contract compiler should:

- parse the user prompt into a small JSON contract
- keep progressive disclosure explicit:
  - preset-first for the default path
  - prompt-assisted preset selection for convenience
  - raw JSON only for advanced users
- require every rule to declare:
  - `judge`: `metric`, `visual`, or `hybrid`
  - `evidence_at`: phase, view, and optionally motion window
- refuse open-ended prompt authoring if the request cannot be grounded to a reviewed
  preset or explicit rule set
- ask the user before execution if any rule cannot be grounded safely
- fail closed if the ambiguity is not resolved

When the user is running through a skill flow, the compiler should use the
existing AskUser tooling rather than guessing.

## Runtime Verdict Model

Required v1 verdicts:

- `PASS`
- `FAIL`
- `AMBIGUOUS`
- `CONTRACT_INVALID`

Rules:

- `FAIL` means one or more hard contract rules were violated
- `AMBIGUOUS` means the system could not establish intent safely
- `AMBIGUOUS` may trigger more evidence gathering
- `AMBIGUOUS` may never self-promote to `PASS`
- `CONTRACT_INVALID` means the run should not have started or the evidence model
  broke in a way that invalidates the approval logic

## Stop Policy

The run should combine model judgment with a boring hard guard.

### Soft Stop

Claude/Codex may stop early when it believes the goal is not reachable.

### Hard Stop

Recommended v1 defaults:

- repeat failure signature limit: `2`
- optional rerun ceiling: `12`

Failure signature should be:

- `case_id`
- `phase_id`
- `violated_rule_id`

Views should not be part of the stop signature. They belong in the evidence pack.

## Material Change Surfacing

The first screen should surface only:

- materially changed cases
- ambiguous cases

The report should still show:

- total case count
- surfaced case count
- unchanged case count

### A case is surfaced when:

- a hard metric fails
- a visual judgment is ambiguous
- the run claims an intended migration success that needs review
- an anti-goal is hit at any point

### A case stays off the first screen when:

- no contract rule marked it as changed
- no hard gate failed
- no ambiguity remains

## Contract Schema

This is the minimal v1 contract shape:

```json
{
  "schema_version": "roboharness_contract/v1",
  "contract_id": "string",
  "mode": "regression | migration",
  "source_prompt": "string",
  "cases": {
    "source": "string",
    "immutable": true
  },
  "compile_policy": {
    "on_ambiguity": "ask_user_if_skill_else_fail",
    "require_grounded_rules": true
  },
  "runtime_policy": {
    "on_ambiguous_verdict": "gather_more_evidence_but_never_self_pass",
    "soft_stop": "agent_may_stop_when_goal_unreachable",
    "hard_stop": {
      "repeat_failure_signature_limit": 2,
      "max_reruns": 12
    },
    "failure_signature": [
      "case_id",
      "phase_id",
      "violated_rule_id"
    ]
  },
  "approval_policy": {
    "surface_changed_cases_only": true,
    "show_unchanged_case_count": true,
    "require_user_blessing_for_new_baseline": true
  },
  "rules": []
}
```

## Rule Types

Three rule types are enough for v1.

### 1. `metric_gate`

Hard boolean or threshold rule.

Example:

- target object must be grasped at `lift`
- failed alarm count must remain zero

### 2. `visual_goal`

What the user wants the new behavior to look like.

Example:

- final top-down view must show palm-down over the ball

### 3. `anti_goal`

What must never be accepted, even if the endpoint looks plausible.

Accepted examples from discussion:

- still side-grasping, just with the arm approaching from above
- palm faces down, but the fingers still touch the bottle
- all earlier waypoints still reflect a side grasp, then one quick sharp motion
  fakes the final pose

## Example Contract

```json
{
  "schema_version": "roboharness_contract/v1",
  "contract_id": "grasp-bottle-to-ball-topdown-v1",
  "mode": "migration",
  "source_prompt": "Reuse the existing 8 grasp harness cases, switch from side-grasp bottle to top-down ball grasp, and make the final grasp show the palm in the top2down view.",
  "cases": {
    "source": "existing_8_grasp_cases",
    "immutable": true
  },
  "compile_policy": {
    "on_ambiguity": "ask_user_if_skill_else_fail",
    "require_grounded_rules": true
  },
  "runtime_policy": {
    "on_ambiguous_verdict": "gather_more_evidence_but_never_self_pass",
    "soft_stop": "agent_may_stop_when_goal_unreachable",
    "hard_stop": {
      "repeat_failure_signature_limit": 2,
      "max_reruns": 12
    },
    "failure_signature": [
      "case_id",
      "phase_id",
      "violated_rule_id"
    ]
  },
  "approval_policy": {
    "surface_changed_cases_only": true,
    "show_unchanged_case_count": true,
    "require_user_blessing_for_new_baseline": true
  },
  "rules": [
    {
      "id": "ball_is_grasped",
      "type": "metric_gate",
      "judge": "metric",
      "evidence_at": [
        { "phase": "lift" }
      ],
      "pass_if": {
        "metric": "target_object_grasped",
        "op": "eq",
        "value": true
      },
      "severity": "fail"
    },
    {
      "id": "no_failed_alarms",
      "type": "metric_gate",
      "judge": "metric",
      "evidence_at": [
        { "phase": "all" }
      ],
      "pass_if": {
        "metric": "failed_alarm_count",
        "op": "eq",
        "value": 0
      },
      "severity": "fail"
    },
    {
      "id": "final_palm_top2down",
      "type": "visual_goal",
      "judge": "visual",
      "evidence_at": [
        { "phase": "final_grasp", "view": "top2down" }
      ],
      "must_show": "palm faces down over the ball in the final grasp pose",
      "severity": "surface"
    },
    {
      "id": "no_side_grasp_disguised_as_topdown",
      "type": "anti_goal",
      "judge": "hybrid",
      "evidence_at": [
        { "phase": "approach", "view": "top2down" },
        { "phase": "contact", "view": "side" }
      ],
      "must_not_show": "side-grasp strategy masquerading as top-down grasp",
      "severity": "fail"
    },
    {
      "id": "no_bottle_contact",
      "type": "anti_goal",
      "judge": "hybrid",
      "evidence_at": [
        { "phase": "contact", "view": "top2down" },
        { "phase": "contact", "view": "side" }
      ],
      "must_not_show": "fingers contacting the bottle instead of the ball",
      "severity": "fail"
    },
    {
      "id": "no_terminal_sharp_snap",
      "type": "anti_goal",
      "judge": "hybrid",
      "evidence_at": [
        { "phase": "final_motion_window", "view": "top2down" },
        { "phase": "final_motion_window", "view": "side" }
      ],
      "must_not_show": "late sharp movement in the last waypoints to fake the final pose",
      "severity": "fail"
    }
  ]
}
```

## Final Report Schema

This is the minimal v1 final-report shape:

```json
{
  "schema_version": "roboharness_report/v1",
  "contract_id": "string",
  "overall_verdict": "PASS | FAIL | AMBIGUOUS | CONTRACT_INVALID",
  "stop_reason": "string",
  "summary": {
    "cases_total": 8,
    "cases_surfaced": 2,
    "cases_unchanged": 6,
    "reruns": 7
  },
  "surfaced_cases": [],
  "unchanged": {
    "count": 6
  },
  "user_action": {
    "needs_review": true,
    "needs_baseline_blessing": true,
    "review_case_ids": []
  }
}
```

For each surfaced case, the report should include:

- `case_id`
- surfaced status
- why it was surfaced
- one canonical proof panel
- rule outcomes
- hard metric results
- one short explanatory caption

## Example Final Report

```json
{
  "schema_version": "roboharness_report/v1",
  "contract_id": "grasp-bottle-to-ball-topdown-v1",
  "overall_verdict": "PASS",
  "stop_reason": "all_rules_satisfied",
  "summary": {
    "cases_total": 8,
    "cases_surfaced": 2,
    "cases_unchanged": 6,
    "reruns": 7
  },
  "surfaced_cases": [
    {
      "case_id": "scene_03",
      "status": "INTENDED_CHANGE_CONFIRMED",
      "material_reason": [
        "visual_goal_changed",
        "proposed_new_baseline"
      ],
      "proof_panel": {
        "phase_id": "final_grasp",
        "view": "top2down",
        "current_image": "artifacts/scene_03/final_grasp/top2down_current.png",
        "baseline_image": "artifacts/scene_03/final_grasp/top2down_baseline.png"
      },
      "rules": {
        "passed": [
          "ball_is_grasped",
          "no_failed_alarms",
          "final_palm_top2down"
        ],
        "failed": [],
        "ambiguous": []
      },
      "metrics": [
        {
          "id": "target_object_grasped",
          "verdict": "PASS",
          "observed": true
        },
        {
          "id": "failed_alarm_count",
          "verdict": "PASS",
          "observed": 0
        }
      ],
      "caption": "Top-down ball grasp reached the requested final palm pose without violating anti-goals."
    },
    {
      "case_id": "scene_05",
      "status": "AMBIGUOUS",
      "material_reason": [
        "visual_intent_unclear"
      ],
      "proof_panel": {
        "phase_id": "final_motion_window",
        "view": "side",
        "current_image": "artifacts/scene_05/final_motion_window/side_current.gif",
        "baseline_image": "artifacts/scene_05/final_motion_window/side_baseline.gif"
      },
      "rules": {
        "passed": [
          "ball_is_grasped",
          "no_failed_alarms"
        ],
        "failed": [],
        "ambiguous": [
          "no_terminal_sharp_snap"
        ]
      },
      "caption": "Hard gates passed, but the final motion window still looks too abrupt to self-approve."
    }
  ],
  "unchanged": {
    "count": 6
  },
  "user_action": {
    "needs_review": true,
    "needs_baseline_blessing": true,
    "review_case_ids": [
      "scene_03",
      "scene_05"
    ]
  }
}
```

## Known Sharp Edge

The anti-goal "one quick sharp movement at the end" is not safely expressible as a
loose visual sentence.

For v1, the compiler must force one of:

- an explicit metric
- an explicit motion-window artifact
- a user clarification before the run starts

Do not let this stay as vague prose. That is exactly the kind of rule an agent can
reinterpret if the contract is sloppy.

## What This Means Strategically

This direction reduces the importance of the earlier repo-split debate.

The primary job is:

- contract compilation
- ambiguity handling
- approval queue design
- baseline blessing for migration mode

The showcase repo can still matter later as an external proof surface, but it is not
the central product move.
