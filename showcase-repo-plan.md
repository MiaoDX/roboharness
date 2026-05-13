<!-- /autoplan restore point: /home/mi/.gstack/projects/MiaoDX-roboharness/main-autoplan-restore-20260417-175048.md -->
# Roboharness Unattended Refactor Harness Plan

Status: APPROVED via `/autoplan` on 2026-04-18. Draft rewritten on 2026-04-17 from the earlier showcase-split plan.

Canonical product/design contract: `docs/designs/unattended-refactor-harness-v1.md`

This file remains the reviewed plan, rationale log, and deferred-work record for the
approved direction.

This document replaces the previous strategy of aggressively moving advanced demos
out of the core repo. The CEO review and follow-up discussion found that repo
cleanliness is not the real bottleneck. Trust is.

## Product Statement

Roboharness is not selling a clean repo.

It is selling **trust compression**:

`2-3 hours unattended Claude/Codex refactor work -> 3 minutes final approval queue`

The user should be able to:

1. `uv add roboharness` into an existing robot codebase
2. wire a small setup layer into the current grasp/debug pipeline
3. give the agent a task plus approval contract
4. walk away
5. come back to a small proof pack showing only materially changed or ambiguous cases

If the system works, the user does not watch the run. They only inspect the final
evidence surface and decide whether to bless the result.

## Why The Earlier Plan Was Wrong

The previous plan asked the wrong question: "Should we move `robots/` and
`controllers/` to a showcase repo?"

That question optimizes for conceptual purity, not user trust.

What the CEO review found:

- `pip install roboharness` is already lightweight at the package layer
- heavy integrations are already behind optional extras
- top-level exports already keep the core API relatively clean
- the most impressive demos are not just noise, they are proof that adoption tax is low

So the product problem is not "too many files in core." The product problem is:

**Can an agent refactor robot code unattended and return a proof surface that a human
can trust quickly?**

## Core User Outcome

The target user outcome is:

> "When I give roboharness to Claude Code or Codex, my grasp pipeline gains a visual
> and metric harness so the agent can iterate through a hard refactor without me in
> the loop. When it finishes, I inspect the final visual harness results and know
> whether we are in good condition."

That means roboharness is not just visual testing. It is an **unattended refactor
gate for robot behavior**.

## Product Modes

Roboharness needs two explicit approval modes.

### 1. Regression Mode

Use when behavior should stay materially the same.

- old baseline remains authoritative
- visible differences are suspicious by default
- hard metric regressions fail immediately
- surfaced cases are regressions and ambiguities

### 2. Migration Mode

Use when the user wants new behavior, not just code cleanup.

Example:
- reuse the existing 8 scenes
- change bottle side-grasp to ball top-down grasp
- require the final top-down view to show the palm-down pose

In migration mode:

- scenes may remain fixed while expected behavior changes
- the user prompt is compiled into an explicit contract before the run starts
- the agent may propose a new baseline
- the user blesses that baseline once at the end

This is not optional. Without explicit migration mode, an agent can launder a wrong
behavior change as "intended."

## Contract-First Model

Natural language is not the source of truth.

Before an unattended run starts, roboharness compiles the user prompt into a small
JSON contract. If the prompt is ambiguous, the skill must stop and ask the user
before execution.

Minimum contract fields:

- `mode`: `regression` or `migration`
- `cases`: which harness cases are in scope, and whether they are immutable
- `rules`: the approval rules for this run
- `runtime_policy`: ambiguity handling and stop policy
- `approval_policy`: what gets surfaced to the user at the end

Every rule must include:

- `judge`: `metric`, `visual`, or `hybrid`
- `evidence_at`: phase, view, or motion window used to verify the rule

If a rule cannot be grounded to `judge + evidence_at`, the run must not start.

### Rule Types

Three rule types are enough for v1:

1. `metric_gate`
   - hard pass/fail condition
   - example: object must be grasped at lift, failed alarm count must be zero

2. `visual_goal`
   - intended behavior to show the user and the agent
   - example: final grasp must show palm-down over the ball in the top-down view

3. `anti_goal`
   - bad-but-plausible behavior that should never be accepted
   - examples:
     - still side-grasping while the arm approaches from above
     - fingers touching the bottle instead of the ball
     - final sharp snap motion to fake the target pose

## Runtime Verdict Semantics

RoboHarness needs explicit verdicts, not fuzzy agent prose.

Required v1 verdicts:

- `PASS`
- `FAIL`
- `AMBIGUOUS`
- `CONTRACT_INVALID`

Key rule:

**`AMBIGUOUS` may trigger more evidence gathering, but it can never self-promote to
`PASS`.**

This is the trust boundary. The agent can keep working, but if ambiguity remains,
the run hands that ambiguity back to the user honestly.

## Stop Policy

The stop policy should combine smart agent behavior with a boring hard guard.

### Soft Stop

Claude/Codex may stop early when the goal is clearly unreachable.

### Hard Stop

V1 should stop when the same failure signature repeats, with a coarse rerun cap
behind it.

Recommended default:

- repeat failure signature limit: 2
- failure signature:
  - `case_id`
  - `phase_id`
  - `violated_rule_id`
- optional backstop: `max_reruns`

Views belong in the evidence pack, not the failure signature. The important
question is whether the agent is stuck on the same contract violation, not whether
the screenshots differ slightly.

## Final User Surface: Approval Queue

The first screen should not be a giant dashboard.

It should be an **approval queue**.

The first screen should surface only:

- materially changed cases
- ambiguous cases

It should hide unchanged cases by default, while still reporting the count:

- `8 total cases`
- `2 surfaced`
- `6 unchanged by contract`

For each surfaced case, the first screen should show:

- case id
- intended change / regression / ambiguity status
- one canonical proof panel
- the rules that passed, failed, or stayed ambiguous
- the hard metric verdicts
- one short sentence explaining why this case is surfaced

Unchanged cases can remain in the folder tree and deeper report views. The user
explicitly said that is enough.

## Material Change Policy

V1 should surface a case when any of these happen:

- a hard metric fails
- a visual judgment is ambiguous
- the run claims an intended migration success and needs review
- an anti-goal is hit at any point

Cases with no material change should stay off the first screen.

## Existing Code Leverage

This direction should build on what already exists instead of starting over.

Relevant existing assets:

- `autonomous_report.json` as machine-readable verdict source
- `phase_manifest.json` for failing phase, primary views, and rerun hints
- `alarms.json` for evaluator-backed failures
- `report.html` and the current `Current vs Baseline` evidence surface
- `examples/_mujoco_grasp_wedge.py` and its phase-local evidence model
- the existing `RobotHarnessWrapper`, `Harness`, and report-generation flow

This is important: the current MuJoCo wedge already looks like a seed of the right
product. The job is to generalize the trust contract, not to throw away the wedge
and reorganize folders for aesthetic reasons.

## Showcase Repo Role

The showcase repo is still useful, but its role changes.

It is not the primary strategic move anymore.

Recommended role:

- external proof that roboharness works as a pip-installed dependency
- secondary distribution surface for framework-specific integrations
- not the main answer to the core product question

That means:

- keep the showcase repo
- do not make "move everything out of core" the main plan
- do not expand showcase aggressively until the contract-first product loop lands
- keep load-bearing integration proof in the core repo for now

## In Scope

### Product Scope

- define a compiled JSON contract for unattended regression and migration runs
- make rule grounding mandatory with `judge + evidence_at`
- formalize `PASS / FAIL / AMBIGUOUS / CONTRACT_INVALID`
- build the changed-cases approval queue as the main return surface
- keep unchanged cases out of the first screen
- preserve human blessing as the final approval boundary

### Implementation Scope

- prototype the contract-first loop in the existing MuJoCo grasp wedge first
- use skill-driven AskUser behavior when a contract clause cannot be compiled safely
- keep leveraging current report artifacts instead of inventing a second reporting stack

## Not In Scope

- deleting `robots/` or `controllers/` in the first pass
- moving all advanced demos out of the core repo
- making the showcase repo the primary product bet
- letting natural-language prompts execute without compilation
- allowing ambiguous runs to self-pass
- showing every unchanged case on the first approval screen
- full multi-project orchestration across repos before the core approval loop works

## Phased Delivery

### Phase 1: Contract + Report Spec

- finalize contract schema
- finalize final-report schema
- document rule grounding and ambiguity semantics

### Phase 2: MuJoCo Grasp Wedge Prototype

- compile migration/regression contracts for the grasp example
- add surfaced-case approval queue
- keep folder-level deep drill-down for unchanged cases

### Phase 3: Skill Flow

- compile user prompts into JSON contracts
- use AskUser when required clauses are missing or ambiguous
- fail closed when the user does not resolve ambiguity

### Phase 4: Second Integration

- extend from the MuJoCo wedge into one higher-value integration path
- ideal candidate: LeRobot evaluation or another already-validated harness case

## Success Metrics

The right metrics are product-trust metrics, not repo-tidiness metrics.

Success looks like:

- a user can describe a run in natural language and get a compiled contract in minutes
- every contract clause either compiles cleanly or asks for clarification before run start
- an unattended run returns `PASS`, `FAIL`, or `AMBIGUOUS` with machine-readable reasons
- the first screen shows only surfaced cases and unchanged-case count
- a user can review the surfaced approval queue quickly without replaying the whole run
- migration runs produce a proposed new baseline that the user can bless once

## Open Questions

These are real and should stay explicit:

1. How should motion-window evidence be represented for anti-goals like "late sharp
   movement" when still images are insufficient?
2. What is the exact baseline blessing flow for accepted migration runs?
3. Which parts of the current MuJoCo wedge should be promoted into shared code, and
   which should stay example-local until a second use case proves the abstraction?
4. How small can the approval queue stay while still feeling honest to the user?

## Immediate Recommendation

Do not spend the next cycle deleting demos.

Spend it making unattended runs trustworthy:

- contract compilation
- ambiguity handling
- changed-case approval queue
- baseline blessing for migration mode

That is much closer to the product the user actually described.

## /autoplan Phase 1 CEO Review

Mode selected: `SELECTIVE_EXPANSION`

### 0A. Premise Challenge

- Premise: trust compression matters more than repo tidiness. This is valid and the
  repo evidence supports it: optional extras already keep installation lightweight,
  while the stronger existing leverage is in evaluator artifacts, report packs, CLI
  inspection, and MCP surfaces rather than in moving files around.
- Premise: natural-language prompts should not be the source of truth. This is also
  valid, but the current plan overreaches by making freeform prompt compilation feel
  like the v1 product. For the MuJoCo wedge, the safer first move is constrained,
  template-shaped contract compilation with boring defaults, not an open-ended NL
  compiler that pretends to understand arbitrary intent.
- Premise: the changed-cases approval queue is the correct front-door surface.
  Partial. The queue is the right reviewer UX shape, but it only earns trust if the
  evaluator and surfacing logic have measured precision. A smaller queue with noisy
  or missing cases is worse than a bigger but honest proof pack.
- Premise: migration-mode baseline blessing can happen once at the end. This is not
  solid enough as written. If a weak run can relabel a regression as an intended
  migration and ask for one final bless, the trust boundary collapses. The old
  baseline must remain authoritative until the reviewer has inspected surfaced cases
  against it and explicitly accepted a new baseline as a separate approval step.
- Premise: the showcase repo should be strategically secondary until the approval
  loop works. This is directionally right, but it currently conflicts with the Q2
  roadmap and the README/front door. That mismatch is not cosmetic. It guarantees
  team drift unless the docs are rewritten around one top-level objective.

### 0B. Existing Code Leverage

| Sub-problem | Existing code | Reuse decision |
|---|---|---|
| Single-case verdict artifact pack | `examples/_mujoco_grasp_wedge.py::build_autonomous_report()` plus `write_artifact_pack()` | Reuse directly as the seed contract/evidence pack |
| First failing phase + evidence hints | `examples/_mujoco_grasp_wedge.py::build_phase_manifest()` and `resolve_evidence_pairs()` | Reuse directly; extend toward surfaced-case proof |
| Reviewer-facing HTML evidence | `examples/_mujoco_grasp_wedge.py::build_summary_html()` and `src/roboharness/reporting.py` | Extend in place; do not invent a second renderer stack |
| Multi-case aggregation | `src/roboharness/evaluate/batch.py` | Reuse as the substrate for a changed-case queue |
| CLI front door | `src/roboharness/cli.py` inspect/report/evaluate paths | Reuse; front-door rewrite should align these to the new story |
| Agent/tooling integration | `src/roboharness/mcp/tools.py` (`evaluate_constraints`, `evaluate_batch_trials`, `compare_baselines`) | Reuse; keep library/CLI first, agent adapters second |
| Strategic wedge already prioritized | `docs/product/roadmap-2026-q2.md` LeRobot Evaluation + Constraint Evaluator | Reuse as proof that evaluator-first work already has roadmap support |

### 0C. Dream State Mapping

```text
CURRENT STATE                    THIS PLAN                         12-MONTH IDEAL
Repo proves a strong             MuJoCo wedge becomes a           Roboharness is a shared
single-case evidence pack,       contract-governed approval       approval/evidence substrate
batch helpers, CLI, MCP,         wedge: constrained templates,    across multiple robot eval
and multiple demos, but          changed-case surfacing,          flows, with benchmarked judge
the front door still tells       baseline-governance, and         precision, template-driven
a scattered visual-demo story.   README/front-door alignment.     contracts, and trusted queues.
```

### 0C-bis. Implementation Alternatives

APPROACH A: Full Contract-First Unattended Refactor Product Now
  Summary: treat `showcase-repo-plan.md` literally as v1 product scope: freeform
  prompt compilation, unattended regression and migration runs, approval queue, and
  baseline blessing flow all in one push.
  Effort:  L
  Risk:    High
  Pros:
  - matches the most ambitious story
  - creates a strong north-star narrative
  - could differentiate if it actually lands
  Cons:
  - overpromises far beyond the current shipped wedge
  - forces the team to solve contract authoring, queue UX, and evaluator trust at once
  - makes README/front door claims fragile immediately
  Reuses:
  - existing MuJoCo wedge artifacts, batch helpers, CLI, and MCP tools

APPROACH B: Metric-First CI Gate
  Summary: narrow the next cycle to evaluator reliability, success-rate gating, and
  fixed report artifacts. Defer the approval queue and migration semantics.
  Effort:  M
  Risk:    Low
  Pros:
  - matches the current shipped surface most honestly
  - aligns tightly with the roadmap's LeRobot Evaluation + Constraint Evaluator work
  - reduces product overclaim risk
  Cons:
  - undershoots the user's unattended-review goal
  - keeps the front door stuck in a utilitarian CI-tool story
  - does not answer how a human approves intended migrations
  Reuses:
  - `evaluate/batch.py`, existing assertions, CLI, and current report artifacts

APPROACH C: Approval/Evidence Substrate with a MuJoCo Wedge
  Summary: keep the new direction, but ground v1 in one constrained MuJoCo wedge:
  template-shaped contract compilation, explicit baseline governance, evaluator
  reliability work, and a changed-case approval queue derived from existing machine
  artifacts. Rewrite the README/front door around this wedge.
  Effort:  M
  Risk:    Medium
  Pros:
  - preserves the user's actual trust-compression goal
  - fits the repo's real leverage instead of pretending a control plane already exists
  - creates a credible bridge from the current wedge to LeRobot or a second integration
  Cons:
  - less dramatic than the full unattended-refactor story
  - still requires discipline to avoid turning into showcase sprawl
  - defers the more magical freeform-NL compiler pitch
  Reuses:
  - current MuJoCo wedge, batch aggregation, CLI, MCP surfaces, and evaluator contracts

**RECOMMENDATION:** Choose Approach C. It is the smallest path that preserves the
new direction while staying honest about what the repo can credibly prove next.

### 0D. Selective Expansion Analysis

- Complexity check: the next work should stay inside a narrow band: the MuJoCo grasp
  wedge, its machine-readable artifacts, the approval-surface/report path, and the
  repo front door. That is a manageable feature wedge, not a platform rewrite.
- Accepted into scope:
  - rewrite the product framing from "unattended refactor gate" to "approval/evidence
    substrate," with unattended refactor as the first buyer workflow rather than the
    whole product identity
  - constrain v1 contract compilation to templates or presets backed by existing
    evaluators and fixture semantics
  - make evaluator reliability a gating milestone before the queue story is treated
    as trustworthy
  - include the README/front-door rewrite in the same strategic track, because the
    repo currently tells the wrong story to new users
- Deferred:
  - fully freeform NL-to-grounded-contract compilation, because it adds the exact
    ambiguity risk that this wedge is supposed to control
  - multi-repo orchestration or a showcase-led distribution push, because the roadmap
    and README are already too diffuse for the current trust loop
  - second-stack extraction before the MuJoCo wedge proves the contract and queue model

### 0E. Temporal Interrogation

- Hour 1 foundations: the most important early task is not schema syntax. It is
  settling the product claim. If the README, roadmap, and implementation plan are
  still saying different things, every later decision will fork.
- Hour 2-4 core logic: the next hidden trap is contract authoring overhead. If the
  first wedge asks the user to specify arbitrary `judge + evidence_at + anti_goal`
  combinations from scratch, the "walk away" promise dies before the run starts.
- Hour 4-6 integration: baseline governance becomes the sharpest trust boundary. The
  system must prove that old baseline, intended-change manifest, surfaced cases, and
  final blessing remain distinct artifacts.
- Day 2 and beyond: the likely follow-on pain is surfacing precision. The queue will
  look compelling long before it is statistically trustworthy, so evaluator quality
  needs a seeded-good/seeded-bad/ambiguous corpus before the UX is frozen.
- 6 months out: the failure mode is strategic diffusion. If the repo keeps chasing
  showcase demos, extra engines, and more front-door demos before the approval loop
  is validated, the product story will fragment again.

### 0F. Mode Selection Confirmation

`SELECTIVE_EXPANSION` is the right mode here. The direction is promising, but it
needs selective tightening rather than blind scope growth: one stronger framing
choice, one narrower compiler model, one stricter baseline policy, and one clearer
front door.

### CEO DUAL VOICES

CODEX SAYS (CEO — strategy challenge)
- The repo is strategically incoherent today: the new plan demotes showcase as the
  main bet, but the roadmap still makes it `Do Now` and the README still sells a
  broad visual-demo gallery. One top-level objective needs to win.
- The product promise is ahead of the shipped surface. The repo proves a deterministic
  MuJoCo wedge, batch evaluation, reporting, and thin CLI/MCP tooling, not yet a
  complete unattended-refactor approval system.
- Contract-first is overdesigned if v1 expects freeform rule authoring. Template-led
  presets are more credible for the first wedge.
- Final-only baseline blessing is a trust-laundering hole. Old baseline authority and
  per-case review need to remain explicit.
- Approval queue UX should not harden before evaluator precision is benchmarked.

CLAUDE SUBAGENT (CEO — strategic independence)
- The core reframing is right: trust is the product, not repo cleanliness.
- The current framing is still too narrow around an "unattended refactor gate." The
  stronger product identity is a shared approval/evidence substrate, with unattended
  refactor as one wedge.
- NL-to-grounded-contract compilation is under-proven as a core v1 bet. A constrained,
  template-authored assistive layer is safer.
- The MuJoCo wedge should stay the proving ground, but the moat is not the queue by
  itself. It is evaluator depth, benchmark tasks, integrations, and artifact trust.
- Library/CLI-first product logic should stay primary; agent adapters and AskUser
  tooling should remain secondary wrappers around that substrate.

CEO DUAL VOICES — CONSENSUS TABLE:

| Dimension | Claude | Codex | Consensus |
|---|---|---|---|
| Premises valid? | Partial | Partial | CONFIRMED: trust is the right axis, but trust cannot come from contract/UI shape alone |
| Right problem to solve? | Partial | Partial | CONFIRMED: approval-loop trust is promising, but the wedge buyer and product identity need sharper framing |
| Scope calibration correct? | Partial | No | DISAGREE: both want a narrower first wedge, Codex wants a harder cut now |
| Alternatives sufficiently explored? | No | No | CONFIRMED: simpler wedges were not seriously compared before this review |
| Competitive/market risks covered? | No | No | CONFIRMED: incumbent workflow and moat are still underdefined |
| 6-month trajectory sound? | Partial | No | DISAGREE: both flag roadmap drift, Codex rates the current trajectory more severely |

### Section 1. Architecture Review

The right architecture is not a brand-new control plane. The plan should bind a
small contract layer to the existing artifact path: MuJoCo wedge generates the
machine-readable truth, batch aggregation computes surfaced cases, and the front-door
report/README explain that flow clearly. The architecture to avoid is a second,
marketing-shaped product surface that cannot be derived from the actual artifacts.

### Section 2. Error & Rescue Map

The new error surface is mostly semantic, not simulator-level. The dangerous failures
are compile-time ambiguity, mislabeled intended changes, hidden changed cases, and
baseline drift. Those need explicit recovery states rather than optimistic prose.

| Failure case | User sees | Rescue action | Plan fix |
|---|---|---|---|
| Prompt cannot ground a rule to `judge + evidence_at` | contract stops before execution | ask the user in skill flow or fail closed outside it | template-shaped contract compiler with explicit `CONTRACT_INVALID` |
| Agent labels a regression as an intended migration | surfaced case looks like a success candidate | compare against the old baseline before any blessing step | separate intended-change manifest from final baseline blessing |
| Surfacing logic hides a materially changed case | quiet queue with false confidence | run evaluator corpus and inspect queue precision | make evaluator reliability a gating milestone before queue claims |
| Same failure repeats across reruns | unattended loop spins without progress | stop after repeated failure signature and surface the stuck case | retain boring hard-stop defaults |
| README/front door promises more than the wedge can prove | new users misunderstand the product | align front door with the actual MuJoCo wedge and substrate story | rewrite README/front door in the same phase |

### Section 3. Security & Threat Model

There is no major new network or auth surface in this plan. The real threat model is
trust-boundary integrity inside the artifact pack: old baseline, intended-change
manifest, evaluator outputs, surfaced cases, and final blessing must remain distinct
and auditable. If those collapse into one loose approval story, the product becomes
easy to game even without a classic security exploit.

### Section 4. Data Flow & Interaction Edge Cases

The plan is correct to separate `regression` and `migration`, but it still needs to
state how mixed realities are handled: one case regresses, one migrates successfully,
one stays ambiguous, and five remain unchanged. The queue should carry explicit counts
and per-case reasons, while unchanged cases remain off the first screen but not hidden
from the artifact tree. Motion-window anti-goals are the sharpest edge case: if still
images cannot prove or disprove them, the contract must require a motion artifact, a
metric proxy, or user clarification before the run starts.

### Section 5. Code Quality Review

The quality risk is strategic duplication. Contract schema, approval semantics, and
baseline policy must not be separately redefined in the plan, README, and example
code. One machine-readable schema and one surfaced-case policy should drive the docs,
not the other way around. The repo already has enough primitives; it does not need a
clever abstraction pass to prove this wedge.

### Section 6. Test Review

Current tests prove a deterministic MuJoCo failure path well, but they do not yet
justify the bigger promise in this plan. The missing proof is a seeded evaluation
corpus across regression, migration, ambiguous, and unchanged outcomes, plus queue
precision checks showing what gets surfaced and what stays quiet. Until that exists,
the approval queue is a design direction, not a trust claim.

### Section 7. Performance Review

Simulation runtime is not the dominant risk here. Reviewer throughput is. The plan's
`3 minutes final approval review` target is only plausible if the queue renders a
small number of surfaced cases quickly and if each case is grounded in a compact proof
panel with no manual replay. That means keeping the first screen narrow and deriving
it from existing artifacts rather than doing heavy recomputation at read time.

### Section 8. Observability & Debuggability Review

The current repo already has good raw evidence artifacts. What it lacks is provenance
clarity at the approval boundary. The reviewed plan should require explicit links
between contract clauses, surfaced-case reasons, baseline lineage, and evidence files
so a reviewer can answer "why is this case here?" in one hop. Extra logging is less
important than artifact lineage.

### Section 9. Deployment & Rollout Review

The rollout risk is narrative mismatch, not deploy machinery. If the implementation
moves toward the substrate story but the README and roadmap still headline multi-demo
showcase breadth, users will pull the repo for the wrong reason. The README/front-door
rewrite is therefore load-bearing product work, not optional polish. The showcase repo
can remain useful, but only as external proof after the wedge story is fixed.

### Section 10. Long-Term Trajectory Review

The long-term direction becomes sound only if one objective dominates the next cycle:
prove the approval/evidence substrate on one wedge. If the repo keeps equal priority
on showcase growth, extra simulator breadth, and autonomous-review product claims, the
result will be another diluted roadmap. Extraction into shared code should happen only
after a second use case genuinely needs the same contract and approval semantics.

### Section 11. Design & UX Review

At the CEO stage, the good design call is the changed-cases approval queue instead of
a giant dashboard. The missing design discipline is specificity about what the first
screen says and what it refuses to say. The first screen should explain why each case
is surfaced, what rule or ambiguity triggered it, and whether the baseline is still
authoritative. It should not imply global product maturity that the wedge has not yet
earned.

## NOT in scope

- Freeform NL-to-contract compilation for arbitrary robot tasks: too ambiguous for the
  first wedge; use constrained templates and presets first.
- Treating the showcase repo as the primary product bet: conflicts with the proof goal
  and would keep the roadmap/front door split alive.
- One-click final baseline blessing with no per-case review against the old baseline:
  this weakens the trust boundary instead of strengthening it.
- Shared cross-simulator extraction before the MuJoCo wedge proves the contract and
  surfaced-case model on a second use case.

## What already exists

- A deterministic MuJoCo evidence pack already exists in `examples/mujoco_grasp.py`
  and `examples/_mujoco_grasp_wedge.py`, including `autonomous_report.json`,
  `phase_manifest.json`, `alarms.json`, and a current-vs-baseline report surface.
- Multi-case aggregation already exists in `src/roboharness/evaluate/batch.py`, which
  is the natural substrate for changed-case surfacing rather than a net-new queue engine.
- CLI pathways already exist in `src/roboharness/cli.py` to inspect, report, evaluate,
  and trend harness output. The front door should make those feel like product surfaces.
- MCP tool surfaces already exist in `src/roboharness/mcp/tools.py`; they should stay
  adapters around the library/CLI substrate, not become the primary product identity.
- The roadmap already prioritizes LeRobot Evaluation and the Constraint Evaluator, so
  evaluator-first work is not a strategic detour. It is already part of the repo's plan.

### CEO Failure Modes Registry

| Failure mode | Why it matters | Mitigation in reviewed plan |
|---|---|---|
| README, roadmap, and plan keep different top-level stories | team execution diffuses immediately | rewrite the front door and realign future planning around one wedge objective |
| Freeform contract authoring creates too much setup friction | the unattended promise dies before run start | use constrained templates and boring defaults first |
| Final-only baseline blessing launders regressions as intended changes | reviewer trust collapses | keep old baseline authoritative and make new-baseline blessing a separate explicit step |
| Queue UX hardens before evaluator precision is known | the product looks clean but lies | add a seeded evaluation corpus and surfacing-precision checks before strong claims |
| Approval surface becomes a second truth source | docs/UI drift from machine artifacts | derive queue states from the same machine-readable artifacts already used by the wedge |
| Motion-window anti-goals stay under-specified | ambiguous behavior slips through or blocks runs late | require motion artifacts, metric proxies, or compile-time clarification before execution |

## Dream State Delta

If this direction lands as reviewed, roboharness still will not be a general
unattended-refactor platform. What it will become is more valuable and more honest:
an approval/evidence substrate proved on one MuJoCo wedge, with constrained contract
compilation, explicit baseline governance, surfaced-case review, and a front door that
matches the product reality. The gap to the 12-month ideal remains meaningful:
benchmarked evaluator quality, a second integration that proves the abstraction,
sharper incumbent comparison, and a stronger ecosystem story built on top of the wedge
rather than in competition with it.

### CEO COMPLETION SUMMARY

```text
+=====================================================================+
|              CEO PLAN REVIEW — COMPLETION SUMMARY                    |
+=====================================================================+
| Mode selected        | SELECTIVE_EXPANSION                           |
| Premise challenge    | accepted, with substrate-first reframing      |
| Alternatives         | 3 examined, Approach C chosen                 |
| Scope additions      | 4 accepted, 3 deferred                        |
| Outside voices       | codex + recovered subagent, 4/6 confirmed     |
| Not in scope         | written                                       |
| What already exists  | written                                       |
| Error & Rescue Map   | written                                       |
| Failure modes        | written                                       |
| Dream state delta    | written                                       |
| Overall CEO verdict  | strong direction, but narrow and de-risk v1   |
+=====================================================================+
```

### Premise Confirmation Gate

The following Phase 1 premise updates require explicit confirmation before Design,
Eng, and DX review continue:

1. Product framing shifts from **"unattended refactor gate"** to **"approval/evidence
   substrate"**, with unattended refactor as the first buyer workflow.
2. V1 contract compilation is **template-first**, not open-ended freeform NL
   compilation.
3. **Old baseline remains authoritative during the run**; new baseline blessing is a
   separate explicit approval step after surfaced-case review.
4. The approval queue is treated as **conditionally trustworthy** until evaluator
   precision is benchmarked on a seeded good/bad/ambiguous corpus.

> **Phase 1 complete.** Codex: 6 concerns. Claude subagent: 6 issues.
> Consensus: 4/6 confirmed, 2 disagreements -> surfaced at gate.
> Passing to Phase 2 after premise confirmation.

## /autoplan Phase 2 Design Review

### Step 0. Design Scope

- UI scope: **yes**. This plan changes two user-facing surfaces:
  - the approval queue / proof surface returned by a run
  - the README / front-door narrative that tells a new user what roboharness is for
- Design tooling status during review: **`DESIGN_NOT_AVAILABLE`**. No mockups were
  generated. This was a text-only design review, and no approved mockups exist.
- `DESIGN.md`: not present. The correct design baseline is the repo's existing report
  vocabulary in `src/roboharness/reporting.py`, plus the canonical product contract in
  `docs/designs/unattended-refactor-harness-v1.md`.
- Existing patterns worth reusing:
  - top-of-report summary block before the checkpoint gallery
  - compact status badges and severity colors in the current report shell
  - one-command README quick start examples
  - deeper drill-down living below the first-screen summary instead of inside it

Initial design completeness: **3/10**. The plan says what machine artifacts should
exist, but it still leaves too much freedom about what the reviewer sees first, what
state the run is in, and when baseline authority changes.

### Design Dual Voices

CODEX SAYS (design — UX challenge)
- The report needs a run-level decision layer before the queue. Without that, a user
  sees surfaced cards before they understand whether the run is blocked, reviewable,
  or already in a safe success state.
- The state model is under-specified. Empty, loading/compiling, partial evidence,
  contract-blocked, and degraded-trust states cannot be left to implementer taste.
- The first screen is overloaded. Triage-first, details second. The user should not
  parse proof panels, queue counts, and baseline promotion options all at once.
- Responsive behavior is undefined.
- Accessibility requirements are absent.
- The canonical proof panel is still too generic to implement consistently.

CLAUDE SUBAGENT (design — independent review)
- The approval queue is missing a run-level decision banner that explains whether the
  reviewer is in `review surfaced cases`, `fix contract`, `inspect degraded evidence`,
  or `no review required`.
- The baseline blessing flow is still visually ambiguous. The user needs a separate,
  delayed baseline-promotion step that cannot appear like part of normal queue triage.
- Zero-surfaced success is unspecified. A blank queue is not a success state.
- Partial, degraded, and incomplete review states are unspecified.
- The canonical proof panel is too generic and could drift into several different UIs.
- Mixed-outcome queue ordering and CTA grammar are unspecified.
- Over time, the canonical product spec and this review log should become separate
  artifacts so the implementation does not pull product truth from commentary.

### Design Litmus Scorecard

| Litmus check | Claude | Codex | Consensus |
|---|---|---|---|
| 1. Is the run decision legible before any queue item? | No | No | CONFIRMED gap -> fix |
| 2. Is zero-surfaced success explicitly designed? | No | No | CONFIRMED gap -> fix |
| 3. Are `compiling`, `contract_invalid`, `degraded`, and `partial` states locked? | No | No | CONFIRMED gap -> fix |
| 4. Are queue order and CTA grammar explicit for mixed outcomes? | No | Partial | CONFIRMED gap -> fix |
| 5. Is the proof-panel anatomy specific enough to build once? | No | No | CONFIRMED gap -> fix |
| 6. Is the responsive strategy intentional? | Partial | No | single-voice flag -> fix |
| 7. Are accessibility requirements explicit? | Partial | No | single-voice flag -> fix |

### Pass 1. Information Architecture

**Rating:** 3/10 -> 9/10 after fixes.

What was wrong:
- the plan jumped straight from "approval queue" to per-case proof without saying what
  the run's top-level decision is
- baseline authority could look like just another queue card
- the current README/front door still teaches "many demos" before it teaches "one
  proof surface that saves you review time"

Locked information hierarchy:

```text
APPROVAL REPORT FIRST SCREEN
============================
[ Run decision banner ]
[ Counts row: surfaced / suppressed / unchanged / total ]
[ Surfaced queue ordered by severity ]
[ Suppressed summary with reasons, not first-class cards ]
[ Baseline promotion panel, only if migration run is review-complete ]
[ Deep links to unchanged archive / full artifact tree ]
```

Run-level decision banner states:
- `Contract blocked`
- `Review surfaced cases`
- `Evidence degraded, review cautiously`
- `Run incomplete`
- `No surfaced cases, baseline remains authoritative`

README / front-door hierarchy:

```text
README FIRST SCREEN
===================
[ one-sentence product promise ]
[ 10-minute MuJoCo wedge quickstart ]
[ what artifact pack is returned ]
[ why old baseline still rules during review ]
[ advanced demos / showcase repo / other integrations ]
```

Constraint worship for this plan: if only three things survive the first screen, they
are the run decision, the surfaced proof queue, and the exact next approval action.

### Pass 2. Interaction State Coverage

**Rating:** 2/10 -> 9/10 after fixes.

The draft named success and ambiguity in prose, but not as a real state model. The
reviewed plan now locks the following matrix.

| Feature | compiling | contract_invalid | review-ready success | review-ready surfaced | run incomplete | evidence degraded | partial reviewable |
|---|---|---|---|---|---|---|---|
| Run decision banner | `Compiling contract from preset/prompt... no run has started.` | `Contract blocked. Fix the highlighted clauses before execution.` | `No cases surfaced. Old baseline remains authoritative.` | `Review surfaced cases against the old baseline.` | `Run stopped before full review coverage was collected.` | `Review is possible, but trust is degraded because evidence is incomplete or weak.` | `Some surfaced cases are reviewable, others are blocked by partial evidence.` |
| Counts row | hidden until compile finishes | hidden | shows total, surfaced `0`, unchanged, suppressed | shows total, surfaced, suppressed, unchanged | shows total collected so far plus `run incomplete` label | shows surfaced plus degraded count | shows surfaced plus partial-reviewable count |
| Surfaced queue | no queue yet | replaced by fix list | replaced by explicit success state, no empty list chrome | ordered cards with reason, proof panel, CTA | ordered cards plus stuck/incomplete copy | cards include degraded-evidence badge and warning text | cards include what is missing and whether user action is still possible |
| Canonical proof panel | skeleton not needed in static artifact | not shown | not shown | paired proof panel or rule evidence panel shown | shown only for completed surfaced items | shown with visible degraded banner | shown with placeholder/missing side clearly labeled |
| Baseline promotion panel | not shown | not shown | not shown | not shown until all surfaced cases reviewed | blocked | blocked | shown only for fully reviewable intended-change successes after surfaced review is complete |

Zero-surfaced success must render explicit copy:
- `No material changes surfaced.`
- `Old baseline remains authoritative.`
- `No new baseline is available to bless.`

### Pass 3. User Journey & Emotional Arc

**Rating:** 4/10 -> 9/10 after fixes.

| Step | User does | User feels | Plan now specifies? |
|---|---|---|---|
| 1 | Lands on README/front door | `What is this actually for?` | Yes, wedge-first promise and quickstart |
| 2 | Runs the MuJoCo wedge quickstart | `Can I get a real artifact fast?` | Yes, one wedge and one returned proof pack |
| 3 | Opens the report | `Tell me the run state before I read details.` | Yes, run decision banner comes first |
| 4 | Scans surfaced queue | `Why is this case here?` | Yes, one surfaced reason and one CTA per card |
| 5 | Reads the proof panel | `Do I trust this evidence?` | Yes, panel anatomy and degraded states are locked |
| 6 | Reaches baseline promotion | `Am I blessing a new expectation or just approving review?` | Yes, promotion is delayed and visually separate |
| 7 | Drops to unchanged archive only if needed | `I can inspect deeper without losing the triage path.` | Yes |

Time-horizon design:
- 5-second visceral: the run state is obvious
- 5-minute behavioral: the user can review surfaced cases without replaying the run
- 5-day reflective: the product feels like a trustworthy approval substrate, not a
  pile of screenshots

### Pass 4. AI Slop Risk

**Rating:** 4/10 -> 8/10 after fixes.

Classifier: **HYBRID**.
- The README/front door is a landing surface.
- The approval report is an app-like review surface.

Hard anti-slop rules for the README/front door:
- do not lead with the current demo-gallery sprawl
- do not use a generic "AI agents can see, judge, iterate" hero without immediately
  showing the MuJoCo wedge and its returned proof pack
- one primary quickstart above the fold, not a seven-row demo table

Hard anti-slop rules for the approval surface:
- each surfaced card has one job: explain why the case is surfaced and what to do next
- no generic dashboard tiles with duplicated stats
- no "proof panel" label without a locked anatomy

Canonical proof-panel anatomy:
- case id and mode badge
- surfaced reason badge (`regression`, `ambiguous`, `intended change needs review`,
  `contract invalid`, `degraded evidence`)
- old baseline authority sentence
- evidence pair or typed rule evidence
- passed / failed / ambiguous rule chips
- one CTA line in imperative grammar

Queue ordering for mixed outcomes:
1. `CONTRACT_INVALID`
2. `FAIL`
3. `AMBIGUOUS`
4. `REVIEW_SUCCESS`

CTA grammar:
- `Fix contract`
- `Inspect regression`
- `Review ambiguity`
- `Bless or reject intended change`

### Pass 5. Design System Alignment

**Rating:** 5/10 -> 8/10 after fixes.

No `DESIGN.md` exists, so the right move is narrow alignment rather than style
invention:
- reuse the existing report-shell badge, card, and severity vocabulary
- extend `src/roboharness/reporting.py` with the smallest CSS surface necessary
- keep README copy technical and concrete, not brand-forward
- keep the design truth anchored in `docs/designs/unattended-refactor-harness-v1.md`

Design-system rule for this phase: no new visual identity work until the front door
and approval queue both tell the same story.

### Pass 6. Responsive & Accessibility

**Rating:** 2/10 -> 9/10 after fixes.

Responsive requirements:
- report desktop: counts row and queue summary may sit side by side, but proof panels
  stay vertically stacked by case to preserve reading order
- report narrow widths: run banner remains first, queue cards stack, proof pairs stack
  `Current` above `Baseline`
- README narrow widths: the wedge quickstart remains above additional demos and
  showcase links, with no giant table as the first scroll experience
- no horizontal scrolling for the critical review path

Accessibility requirements:
- semantic heading order for README and report
- all status states must be readable without color
- proof images need alt text with case, phase/view, and evidence role
- baseline-promotion control must not appear active while review is blocked
- queue cards must be keyboard navigable if any expand/collapse affordance is added
- partial or degraded states must say exactly what is missing or weak

### Pass 7. Unresolved Design Decisions

**Resolved now:**
- run-level decision banner appears above the queue
- zero-surfaced success has explicit copy
- the full review state matrix is now part of the plan
- queue ordering and CTA grammar are locked
- the canonical proof-panel anatomy is locked
- responsive and accessibility requirements are now explicit
- README/front door becomes wedge-first, with the demo gallery and showcase repo moved lower

**Deferred:**

| Decision deferred | If deferred, what happens |
|---|---|
| Split canonical spec from review log | this file remains noisy until the implementation phase creates a cleaner product spec artifact |
| Add motion-window proof UI | motion-rooted anti-goals remain typed as future work or compile-time clarification paths |

### Design NOT in Scope

- pixel-level visual exploration or mockup generation in this review pass
- a full custom web app for the approval queue
- brand refresh or new design language work
- motion scrubbers or temporal playback UI in v1

### Design What Already Exists

- `docs/designs/unattended-refactor-harness-v1.md` already captures the accepted
  product contract
- `src/roboharness/reporting.py` already owns the report shell and status vocabulary
- `examples/_mujoco_grasp_wedge.py` already proves a compact summary block can drive
  the review flow
- `README.md` already contains a MuJoCo quick start, even though the story above it is
  currently too diffuse

### DESIGN PLAN REVIEW — COMPLETION SUMMARY

```text
+=====================================================================+
|          DESIGN PLAN REVIEW — COMPLETION SUMMARY                    |
+=====================================================================+
| System Audit         | DESIGN_NOT_AVAILABLE, UI scope yes           |
| Step 0               | 3/10 initial, hierarchy + states focus       |
| Pass 1  (Info Arch)  | 3/10 -> 9/10 after fixes                     |
| Pass 2  (States)     | 2/10 -> 9/10 after fixes                     |
| Pass 3  (Journey)    | 4/10 -> 9/10 after fixes                     |
| Pass 4  (AI Slop)    | 4/10 -> 8/10 after fixes                     |
| Pass 5  (Design Sys) | 5/10 -> 8/10 after fixes                     |
| Pass 6  (Responsive) | 2/10 -> 9/10 after fixes                     |
| Pass 7  (Decisions)  | 7 resolved, 2 deferred                       |
+---------------------------------------------------------------------+
| NOT in scope         | written                                      |
| What already exists  | written                                      |
| Approved Mockups     | none generated, text-only review             |
| Decisions made       | 7 added to plan                              |
| Decisions deferred   | 2 listed above                               |
| Overall design score | 3/10 -> 8/10                                 |
+=====================================================================+
```

> **Phase 2 complete.** Codex: 6 concerns. Claude subagent: 7 issues.
> Consensus: 5/7 confirmed, 0 disagreements, 2 single-voice gaps captured in fixes.
> Passing to Phase 3.

## /autoplan Phase 3 Engineering Review

### Step 0. Scope Challenge With Actual Code

The actual codebase is narrower than the plan's current product language, which is
good for implementation but dangerous for overclaiming:

- `src/roboharness/evaluate/result.py` currently models verdicts as only
  `PASS / DEGRADED / FAIL`. That is enough for today's constraint evaluator, but it is
  not the same thing as the reviewed contract-first run states
  (`PASS / FAIL / AMBIGUOUS / CONTRACT_INVALID` plus degraded/partial review states).
- `src/roboharness/evaluate/batch.py` is already the right substrate for multi-case
  aggregation, but it only returns verdict counts, failure-code counts, and trial
  summaries. It does not yet know `surfaced_cases`, `suppressed_cases`, or reasons.
- `src/roboharness/cli.py` already exposes `inspect`, `report`, `evaluate`, and
  `trend`, but there is no contract lifecycle API or explicit baseline-promotion step.
- `src/roboharness/reporting.py` already provides a shared HTML shell and
  `summary_html` insertion point. That is a presentation seam, not a product-truth seam.
- The MuJoCo wedge and its tests already exist and should stay the first proving ground.

Complexity check:
- acceptable if the work is done as a deterministic library contract with thin adapters
- not acceptable if the implementation binds product logic directly to interactive
  Claude/Codex behavior

### Engineering Dual Voices

CODEX SAYS (eng — architecture challenge)
- No final external Codex verdict was returned in this resumed session. The partial
  repo scan confirmed the factual substrate, though:
  - current shared evaluator semantics stop at `PASS / DEGRADED / FAIL`
  - the current CLI/report layers are thin adapters, not contract-governance logic
  - there is no existing immutable baseline lineage model in shared code

CLAUDE SUBAGENT (eng — independent review)
- Baseline promotion breaks the trust boundary unless `blessed_baseline`,
  `current_run`, and `proposed_baseline` are separate immutable artifacts.
- The core architecture is too coupled to interactive agent behavior. It needs a
  deterministic `compile_contract` library API plus thin adapters.
- Existing reuse is too optimistic because the current shared evaluator model only has
  `PASS / DEGRADED / FAIL`.
- The approval queue needs both `surfaced_cases` and `suppressed_cases`, with reasons.
- The contract layer needs strict versioned JSON Schema and resource limits.
- Motion-window anti-goals are a hidden second product unless they are cut or typed by
  evidence kind.
- The stop policy is still too coarse for repeated ambiguous or partial-evidence loops.
- The verification plan is too thin for real 2am failure modes.

ENG DUAL VOICES — CONSENSUS TABLE:

| Dimension | Claude | Codex | Consensus |
|---|---|---|---|
| 1. Architecture sound? | No | N/A | single-voice critical flag |
| 2. Test coverage sufficient? | No | N/A | single-voice critical flag |
| 3. Performance risks addressed? | Partial | N/A | single-voice concern |
| 4. Security threats covered? | Partial | N/A | single-voice concern |
| 5. Error paths handled? | No | N/A | single-voice critical flag |
| 6. Deployment risk manageable? | Partial | N/A | single-voice concern |

### Section 1. Architecture Review

Approved architecture:

```text
README / FRONT DOOR
    |
    +--> 10-minute MuJoCo wedge quickstart
            |
            v
    PRESET / PROMPT INPUT
            |
            v
    compile_contract(...)
            |
            +--> contract.json
            +--> compile_diagnostics.json
            |
            v
    run adapter (example CLI / skill / MCP)
            |
            +--> current_run/
            |      +--> autonomous_report.json
            |      +--> alarms.json
            |      +--> phase_manifest.json
            |      +--> evidence/
            |
            +--> evaluate/batch substrate
            |      +--> surfaced_cases.json
            |      +--> suppressed_cases.json
            |      +--> run_summary.json
            |
            +--> approval report
                   +--> run decision banner
                   +--> surfaced queue
                   +--> baseline-promotion gate

BASELINE GOVERNANCE
    blessed_baseline/   current_run/   proposed_baseline/
    immutable           immutable      immutable until explicit bless step
```

Locked architecture decisions:
- create a deterministic `compile_contract(...)` library layer first
- keep CLI, skill, and MCP as thin adapters over that library
- do not overload `src/roboharness/evaluate/result.py::Verdict` with all new product
  semantics until a second consumer proves that the shared evaluator truly needs it
- extend batch aggregation with surfaced/suppressed reasoning instead of inventing a
  second queue-only classification path
- keep baseline lineage auditable and immutable until an explicit bless step

### Section 2. Code Quality Review

The main engineering quality risks are semantic duplication and boundary collapse:

- `DEGRADED` today is not a safe stand-in for `AMBIGUOUS`, `CONTRACT_INVALID`,
  `evidence degraded`, or `partial reviewable`. The reviewed plan should introduce a
  separate run-level contract/result model instead of stretching old enums until they
  lie.
- The contract compiler must be schema-first, not prompt-first. One versioned JSON
  Schema needs to drive compile validation, adapter behavior, and docs examples.
- Queue policy belongs in one place. The README, report copy, and surfaced/suppressed
  machine artifacts should all derive from the same state model.
- Resource limits need to be explicit: max cases, max rules, max evidence references,
  and max compile/runtime artifact size for unattended runs.
- Motion-window anti-goals either become typed evidence kinds (`still_pair`, `clip`,
  `metric_proxy`) or they leave v1. Unbounded "the model will figure it out" language
  is not engineering.

### Section 3. Test Review

The existing MuJoCo wedge tests are a strong seed, but they do not cover the reviewed
product claim yet. The test surface needs to expand from "single-case evaluator proof"
to "contract compile, surfacing, baseline governance, and front-door clarity."

CODE PATH COVERAGE
==============================
[+] contract compiler
    |
    +-- [GAP] preset -> valid schema v1 contract
    +-- [GAP] ambiguous prompt -> clarification required or `CONTRACT_INVALID`
    +-- [GAP] unsupported evidence kind -> fail closed
    +-- [GAP] resource-limit breach -> fail closed with explicit envelope

[+] evaluator + run-level verdict mapping
    |
    +-- [GAP] evaluator pass + no surfaced cases -> review-ready success
    +-- [GAP] metric fail -> surfaced regression
    +-- [GAP] ambiguous visual result -> surfaced ambiguity, never self-pass
    +-- [GAP] compile failure -> contract-invalid path, no run starts
    +-- [GAP] missing evidence -> degraded / partial reviewable path

[+] batch surfacing
    |
    +-- [GAP] `surfaced_cases` and `suppressed_cases` both emitted with reasons
    +-- [GAP] mixed-outcome queue ordering
    +-- [GAP] zero-surfaced success state
    +-- [GAP] unchanged count remains visible while cards stay hidden

[+] baseline governance
    |
    +-- [GAP] `blessed_baseline`, `current_run`, `proposed_baseline` remain distinct
    +-- [GAP] bless step requires explicit reviewer action
    +-- [GAP] no bless action available while surfaced review is blocked

[+] public surfaces
    |
    +-- [GAP] README wedge-first quickstart
    +-- [GAP] run decision banner copy
    +-- [GAP] error envelope fields render consistently across CLI / HTML / JSON

[+] existing MuJoCo wedge
    |
    +-- [TESTED] `tests/regression/mujoco_grasp/test_mujoco_grasp_wedge.py`
    +-- [TESTED] `tests/regression/mujoco_grasp/test_mujoco_grasp_live_validation.py`
    `-- [GAP] migration-mode queue and bless-path states

USER FLOW COVERAGE
==============================
1. Regression preset -> run -> surfaced fail / ambiguous -> reviewer inspects old baseline
2. Migration preset -> run -> intended-change review -> explicit bless or reject
3. Compile failure -> no run starts, user gets actionable fix
4. Zero surfaced cases -> explicit success state, no bless affordance
5. Partial evidence -> reviewable vs blocked state made explicit
6. README quickstart -> first artifact -> first-screen decision -> next action

Current coverage quality:
- strong for the existing MuJoCo evaluator wedge
- weak for contract compile, queue surfacing, and baseline governance

Phase 3 test-plan artifact of record:
- `~/.gstack/projects/MiaoDX-roboharness/mi-main-eng-review-test-plan-20260417-224344.md`

That artifact already captures the QA-facing verification path for:
- README/front door rewrite
- run-level decision banner and surfaced queue behavior
- mixed-outcome queue ordering
- zero-surfaced success
- baseline-governance flow

### Section 4. Performance Review

There is no database or network fan-out risk in the current plan. The real performance
risks are bounded but real:

- `evaluate_batch()` currently walks all `autonomous_report.json` files recursively. A
  surfaced-case queue should not repeatedly reload every unchanged case to render the
  first screen.
- approval HTML should embed or resolve only surfaced-case proof, not the unchanged
  archive again
- compile and review artifacts need size caps so unattended runs do not quietly
  accumulate unbounded prompt-derived contracts or proof packs
- evaluator-precision corpus measurement should live in a dedicated verification path,
  not every default developer loop

Performance guardrails:
- surfaced queue derived once from existing artifacts
- unchanged cases counted, not rendered
- proof panels capped by surfaced cases only
- compile/runtime artifact sizes bounded by schema and policy

### Engineering NOT in Scope

- freeform NL-to-contract compilation in v1
- automatic baseline blessing with no explicit reviewer step
- cross-simulator extraction before a second wedge needs the abstraction
- changing or relocating `assets/g1/`
- widening the default dependency footprint just to make optional proof lanes run everywhere

### Engineering What Already Exists

| Sub-problem | Existing code | Reuse decision |
|---|---|---|
| Shared evaluator verdict substrate | `src/roboharness/evaluate/result.py` | Reuse carefully; add a separate run-level contract state model first |
| Multi-case aggregation | `src/roboharness/evaluate/batch.py` | Reuse as the substrate for surfaced/suppressed counts and reasons |
| CLI front door | `src/roboharness/cli.py` | Reuse as thin adapters, not as the home of product truth |
| HTML report shell | `src/roboharness/reporting.py` | Reuse as presentation-only shell |
| MuJoCo proof wedge | `examples/_mujoco_grasp_wedge.py` + tests | Reuse as the first implementation wedge |
| Agent adapters | `src/roboharness/mcp/tools.py` | Reuse later; keep secondary to the library contract |
| Existing quickstart surface | `README.md` | Rewrite around the wedge, do not discard it |

### Failure Modes Registry

| Codepath | Realistic failure | Test planned? | Error handling? | User-visible? | Critical gap? |
|---|---|---|---|---|---|
| contract compile | ambiguous prompt compiles into the wrong rule | Yes | Yes, `CONTRACT_INVALID` + fix path | Yes | **Yes until tested** |
| verdict mapping | `DEGRADED` is treated as `AMBIGUOUS` or `PASS` | Yes | Partial today | Yes | **Yes** |
| queue surfacing | materially changed case lands in `suppressed_cases` | Yes | No, until precision corpus exists | Yes | **Yes** |
| baseline governance | proposed baseline mutates blessed baseline in-place | Yes | No today | Yes | **Yes** |
| report rendering | first screen shows queue cards without run decision context | Yes | Yes in reviewed plan | Yes | No |
| docs/front door | README still leads with demo sprawl | Yes | No today | Yes | Medium |
| scope hygiene | MuJoCo wedge work drifts into `assets/g1/` | Yes | yes, blast-radius rule | No | No |

Critical gap resolution required before implementation is called complete:
- evaluator precision must be measured on seeded good/bad/ambiguous corpora
- baseline lineage must remain immutable and auditable

### Worktree Parallelization Strategy

| Step | Modules touched | Depends on |
|---|---|---|
| 1. Contract schema + compile library | new contract layer, schema docs, tests | — |
| 2. Run-level verdict mapping + surfaced/suppressed aggregation | `src/roboharness/evaluate/`, MuJoCo wedge, tests | Step 1 |
| 3. Approval report + README/front door rewrite | report surface, README/docs, example wedge | Step 2 |
| 4. Baseline bless flow | contract layer, CLI/skill adapters, tests | Steps 1-3 |
| 5. Precision corpus + verification lanes | fixtures/evals/tests | Steps 2-4 |

Parallel lanes:
- Lane A: schema + compile library
- Lane B: README/front door rewrite and approval-surface copy, once the state model is locked
- Lane C: precision corpus and verification plan, after surfacing semantics stabilize

Conflict flags:
- verdict mapping and approval report should not proceed independently, because the
  queue cannot invent states the compiler and batch layer do not emit
- `assets/g1/` is explicitly outside the write scope for this wedge

### Engineering Completion Summary

- Step 0: Scope Challenge completed with actual code analysis
- Architecture Review: 5 issues found, all resolved in-plan
- Code Quality Review: 5 issues found, all resolved in-plan
- Test Review: diagram produced, major gaps enumerated, QA artifact referenced
- Performance Review: 4 bounded risks identified
- NOT in scope: written
- What already exists: written
- Failure modes: written, 4 critical gaps flagged
- Outside voices: subagent-only; final Codex verdict unavailable in this resumed session
- Parallelization: 3 viable lanes after state-model lock

> **Phase 3 complete.** Codex: unavailable final verdict. Claude subagent: 8 issues.
> Consensus: source=subagent-only, no dual confirmation available.
> Passing to Phase 3.5.

## /autoplan Phase 3.5 DX Review

### Step 0. DX Scope Assessment

Developer-facing scope: **yes**. This plan changes the primary getting-started story,
the public quickstart, the approval surface, and the baseline-review flow.

Product type: **developer-facing CLI + static proof artifact + README/front door**.
Mode selected: **DX POLISH**.

TARGET DEVELOPER PERSONA
========================
Who:       robotics / embodied-AI engineer using Claude/Codex to refactor or migrate one MuJoCo task
Context:   they want one proof-oriented wedge that can be added to an existing codebase without adopting a new platform
Tolerance: about 10 minutes to first meaningful run on a cold setup, under 5 minutes to understand the result once artifacts exist
Expects:   one canonical quickstart, preset-first defaults, headless compatibility, explicit error messages, and no hidden approval magic

### Developer Empathy Narrative

I open the README and immediately feel the current product-story split. There is a lot
of proof that roboharness does many things, but not yet one obvious answer to the
question I actually have: "How do I use this to trust an unattended robot refactor?"
I can find the MuJoCo example, but I have to infer that it is the main wedge rather
than just one demo among many. If I am trying this under time pressure, that matters.
I want one install line, one command, one returned proof pack, and one clear rule
about baseline authority. I do not want to decide between seven demos before I have
seen the product pay off once. If the first screen tells me `review surfaced cases`
and shows current vs old baseline with the exact next action, I trust the tool. If the
front door keeps talking about a showcase repo and broad simulator breadth before that
moment lands, I start treating roboharness like a cool demo gallery instead of a tool
I can adopt.

### Reference Benchmark

This plan does not need a fake apples-to-apples robotics competitor table. The useful
benchmark is the developer-expectation tier for a CLI-onboarding flow:

| Tier | Time to hello world | What it feels like |
|---|---|---|
| Champion | < 2 min | one command, obvious payoff, almost no reading |
| Competitive | 2-5 min | one quickstart path, one artifact, one clear next step |
| Acceptable niche tool | 5-10 min | install plus a short doc scan, but payoff is still obvious |
| Current repo feel | ~10 min | real quickstart exists, but story is split across too many surfaces |
| Reviewed target | 5 min to first meaningful artifact, 30 seconds to understand the run state | wedge-first quickstart + explicit proof surface |

### Magical Moment Specification

Magical moment: the developer opens the returned artifact pack and instantly sees:
- whether the run is blocked, reviewable, or clean
- which cases surfaced and why
- what the old baseline says
- what action to take next

Delivery vehicle: **copy-paste wedge quickstart**, not a hosted playground.

### DX Dual Voices

CODEX SAYS (DX — developer experience challenge)
- No final external Codex DX verdict was returned in this resumed session.

CLAUDE SUBAGENT (DX — independent review)
- There is no real zero-to-hello-world path yet. The README still makes the user
  choose between too many entrypoints.
- Progressive disclosure is unresolved between prose, preset/template, and raw JSON.
- There is no universal user-facing error contract.
- Public CLI/API ergonomics are still conceptual.
- Docs IA is underspecified, and the plan file still mixes plan and review log.
- There is no override / escape-hatch matrix telling the developer what is safe to change.

DX DUAL VOICES — CONSENSUS TABLE:

| Dimension | Claude | Codex | Consensus |
|---|---|---|---|
| 1. Getting started < 5 min? | No | N/A | single-voice critical flag |
| 2. API/CLI naming guessable? | Partial | N/A | single-voice concern |
| 3. Error messages actionable? | No | N/A | single-voice critical flag |
| 4. Docs findable & complete? | No | N/A | single-voice critical flag |
| 5. Upgrade path safe? | Partial | N/A | single-voice concern |
| 6. Dev environment friction-free? | Partial | N/A | single-voice concern |

### Developer Journey Map

| Stage | Developer does | Friction points | Status |
|---|---|---|---|
| 1. Discover | lands on README/front door | too many demos compete with the main wedge | fixed in plan |
| 2. Decide | asks "is this for unattended review or just visual testing?" | current story is split | fixed in plan |
| 3. Install | chooses one dependency path | install matrix is broad | fixed by wedge-first quickstart |
| 4. Start | uses preset-first command path | progressive disclosure not yet defined | fixed in plan |
| 5. Compile | turns prompt/preset into contract | compile errors not standardized | fixed in plan |
| 6. Run | gets artifact pack from MuJoCo wedge | success vs blocked vs degraded states need clarity | fixed in plan |
| 7. Review | opens approval surface | queue order, proof anatomy, baseline authority previously unclear | fixed in plan |
| 8. Decide | blesses or rejects intended change | safe override / bless boundary not explicit | fixed in plan |
| 9. Extend | goes beyond the first wedge | current README jumps here too early | deliberately deferred |

### First-Time Developer Confusion Report

FIRST-TIME DEVELOPER REPORT
===========================
Persona: robotics-cli-dev
Attempting: contract-first MuJoCo wedge onboarding

CONFUSION LOG:
T+0:00  I hit the README and see lots of demos. I still do not know which path proves the new contract-first story.
T+1:00  I find the MuJoCo example, but I am not yet sure whether it is the main workflow or just another showcase.
T+2:00  I can imagine generating a report, but I still do not know how baseline blessing is supposed to work.
T+4:00  If errors happen during contract compile, I do not know what shape the message takes or what I am supposed to fix first.
T+6:00  If the first screen gives me run state, surfaced reason, proof, and next action, I trust the product. If it does not, I am back to manual replay.

Addressed in this review:
- yes, by moving the README/front door to a single wedge-first path
- yes, by defining three progressive-disclosure levels
- yes, by requiring a universal error envelope
- yes, by making baseline blessing explicit and delayed

### DX Pass 1. Getting Started Experience

**Rating:** 3/10 -> 8/10 after fixes.

Locked DX fix:
- the README opens with one 10-minute MuJoCo wedge path
- the first quickstart explains what the user gets back, not just what command to run
- the front door does not ask the user to choose among showcase integrations before
  the wedge has paid off once

### DX Pass 2. API / CLI / SDK Design

**Rating:** 4/10 -> 8/10 after fixes.

The plan now needs an explicit three-tier progressive-disclosure model:

1. Preset-first
   - one quickstart path for the MuJoCo wedge
   - minimal required parameters
2. Prompt-assisted draft
   - the skill or MCP layer compiles a contract draft and only asks when grounding fails
3. Raw JSON advanced
   - developers can hand-author `contract.json` directly when they need full control

Public API / CLI shape to preserve:
- library truth first: `compile_contract`, `execute_run`, `summarize_run`, `bless_baseline`
- adapters second: CLI, example entrypoints, MCP
- raw JSON is an advanced escape hatch, not the first-time onboarding path

### DX Pass 3. Error Messages & Debugging

**Rating:** 2/10 -> 9/10 after fixes.

Universal error envelope required across CLI, HTML, and machine-readable outputs:

```json
{
  "problem": "string",
  "cause": "string",
  "fix": "string",
  "docs_url": "string",
  "recoverable": true,
  "next_action": "string"
}
```

This is the reviewed contract for:
- contract compile failures
- missing or degraded evidence
- blocked baseline blessing
- unsupported evidence kinds
- stop-policy termination

### DX Pass 4. Documentation & Learning

**Rating:** 3/10 -> 8/10 after fixes.

Docs IA now needs to be explicit:

1. What roboharness is for
2. 10-minute MuJoCo wedge quickstart
3. What artifacts are returned
4. How review and baseline authority work
5. Progressive disclosure paths
6. Other demos and showcase repo

The README/front door rewrite is part of the same strategic track, not post-hoc polish.

### DX Pass 5. Upgrade & Migration Path

**Rating:** 5/10 -> 8/10 after fixes.

Required upgrade safety rules:
- schema version is explicit
- old baseline remains authoritative until explicit bless
- proposed baselines are separate artifacts
- raw artifact names stay stable where possible
- advanced JSON users get migration notes when schema evolves

### DX Pass 6. Developer Environment & Tooling

**Rating:** 4/10 -> 7/10 after fixes.

The plan should keep the environment story boring:
- one wedge-specific install path first
- optional extras matrix below it
- headless / CI path made explicit
- no hidden GUI requirement
- no assumption that all users install every optional stack

### DX Pass 7. Community & Ecosystem

**Rating:** 5/10 -> 6/10 after fixes.

The ecosystem story is intentionally demoted, not removed:
- showcase repo remains proof, but not the front door
- second integrations matter after the wedge proves the substrate story
- the README should link outward only after the main wedge is understandable

### DX Pass 8. DX Measurement & Feedback Loops

**Rating:** 3/10 -> 8/10 after fixes.

Measurement commitments added by this review:
- TTHW target: **10 min current -> 5 min target**
- first-review target: **30 seconds to understand run state**
- queue trust gated by seeded good/bad/ambiguous evaluator corpora
- post-implementation boomerang passes: `/document-release` and `/devex-review`

### DX NOT in Scope

- hosted sandbox or playground
- broad showcase-repo expansion as the main onboarding path
- new full-platform docs set before the wedge is implemented
- trying to teach every optional integration above the fold

### DX What Already Exists

- `README.md` already has a MuJoCo quickstart, even if it is not the front-door hero
- `examples/mujoco_grasp.py --report` already exists as the obvious first wedge
- `src/roboharness/cli.py` already provides supporting inspection/reporting verbs
- the current artifact pack already gives the repo a real proof substrate to build on

### DX PLAN REVIEW — SCORECARD

```text
+=====================================================================+
|               DX PLAN REVIEW — SCORECARD                            |
+=====================================================================+
| Dimension            | Score | Note                                 |
|----------------------|-------|--------------------------------------|
| Getting Started      | 8/10  | wedge-first quickstart now explicit  |
| API/CLI/SDK          | 8/10  | progressive disclosure now defined   |
| Error Messages       | 9/10  | universal error envelope locked      |
| Documentation        | 8/10  | docs IA now explicit                 |
| Upgrade Path         | 8/10  | schema + baseline lineage clarified  |
| Dev Environment      | 7/10  | install path simpler, still optional |
| Community            | 6/10  | demoted, not deleted                 |
| DX Measurement       | 8/10  | concrete TTHW + trust metrics added  |
+---------------------------------------------------------------------+
| TTHW                 | 10 min current -> 5 min target               |
| Magical Moment       | one returned proof pack, one clear next step |
| Product Type         | CLI + proof artifact + docs front door       |
| Mode                 | POLISH                                       |
| Overall DX           | 5/10 -> 8/10                                 |
+=====================================================================+
```

### DX IMPLEMENTATION CHECKLIST

```text
DX IMPLEMENTATION CHECKLIST
============================
[ ] README opens with one wedge-first quickstart
[ ] 10-minute MuJoCo quickstart is copy-paste complete
[ ] quickstart says what artifact pack is returned
[ ] progressive disclosure model is explicit: preset / prompt-assisted / raw JSON
[ ] every user-facing error uses the universal envelope
[ ] baseline blessing is explained as a separate explicit step
[ ] override matrix says what can and cannot be changed safely
[ ] CLI / example entrypoints stay thin adapters over library truth
[ ] headless / CI path is documented and not treated as an afterthought
[ ] showcase repo and other demos are moved below the primary wedge path
[ ] TTHW target is measurable after implementation
[ ] queue trust is tied to evaluator precision measurement
```

> **Phase 3.5 complete.** DX overall: 8/10. TTHW: 10 min -> 5 min target.
> Codex: unavailable final verdict. Claude subagent: 6 issues.
> Consensus: source=subagent-only, no dual confirmation available.
> Passing to Phase 4 (Final Gate).

## Cross-Phase Themes

- **Theme: one canonical truth source** — flagged in CEO, Design, and Eng. The
  contract, surfaced/suppressed reasoning, report copy, and README examples must all
  derive from the same machine-readable state model.
- **Theme: proof before polish** — flagged in CEO, Design, and DX. The user only
  trusts the system if the run state and canonical proof land before broader demos,
  metadata, or ecosystem breadth.
- **Theme: old baseline stays authoritative until explicit bless** — flagged in CEO,
  Design, Eng, and DX. Baseline governance is the trust boundary, not a UX detail.
- **Theme: queue trust must be earned** — flagged in CEO, Eng, and DX. The changed-case
  approval queue is conditional on evaluator precision being measured on seeded
  good/bad/ambiguous corpora.
- **Theme: wedge-first now, expansion later** — flagged in CEO, Eng, and DX. The
  MuJoCo wedge and README/front door rewrite are the next work. Showcase-led breadth,
  freeform NL compilation, and shared extraction stay deferred.

## Approval

Approved by user on 2026-04-18 via the final `/autoplan` approval gate.

- Final reviewed artifact: this file
- Canonical product/design contract remains `docs/designs/unattended-refactor-harness-v1.md`
- Deferred follow-ups mirrored to repo-root `TODOS.md`
- `assets/g1/` remains untouched and out of scope for this wedge
- Implementation can proceed from the locked Design, Engineering, and DX sections above

<!-- AUTONOMOUS DECISION LOG -->
## Decision Audit Trail

| # | Phase | Decision | Classification | Principle | Rationale | Rejected |
|---|---|---|---|---|---|---|
| 1 | CEO | Select `SELECTIVE_EXPANSION` mode | Mechanical | P3, P5 | The direction is promising but materially overextended; this mode lets the review keep the new goal while tightening the first wedge. | `EXPANSION`, `HOLD`, `REDUCTION` |
| 2 | CEO | Reframe the product from unattended-refactor gate to approval/evidence substrate | User Challenge | P1, P6 | Both outside voices independently flagged that the broader substrate framing is stronger and less brittle than centering the whole product on unattended refactors. | Keep unattended-refactor gate as the full product identity |
| 3 | CEO | Choose Approach C as the working strategy | Auto-decided | P1, P5, P6 | It preserves the user's trust-compression goal while staying grounded in the repo's actual leverage. | Full contract-first product now; metric-only CI gate |
| 4 | CEO | Constrain v1 contract compilation to templates/presets | User Challenge | P5, P6 | Both voices flagged freeform NL compilation as under-proven and too high-friction for the first wedge. | Open-ended freeform NL-to-grounded-contract compilation in v1 |
| 5 | CEO | Make evaluator reliability a gating milestone before queue claims | Auto-decided | P1, P5 | Queue UX is only valuable if surfaced-case precision is trustworthy. | Freeze the queue UX before measuring precision |
| 6 | CEO | Keep old baseline authoritative until explicit post-run blessing | Auto-decided | P1, P5 | Final-only blessing is a trust-laundering hole; the trust boundary needs separate artifacts and approval steps. | Single final bless with no per-case old-baseline review |
| 7 | CEO | Include README/front-door rewrite in the same strategic track | Auto-decided | P2, P4 | The repo currently tells the wrong story, and leaving that mismatch in place would blunt the implementation immediately. | Treat README rewrite as optional follow-up polish |
| 8 | CEO | Defer showcase-led expansion and cross-simulator extraction | Auto-decided | P3, P5 | Both are strategically useful, but they are not load-bearing until the wedge proves the substrate story. | Prioritize showcase breadth or shared extraction now |
| 9 | Design | Treat the UI scope as hybrid: wedge-first README plus app-like approval surface | Mechanical | P1, P4 | The plan changes both onboarding and the review UI, and the hierarchy rules differ across those surfaces. | Backend-only interpretation of the plan |
| 10 | Design | Add a run-level decision banner above the queue | Auto-decided | P1, P5 | Reviewers need the global run state before they inspect any individual surfaced case. | Start with queue cards only |
| 11 | Design | Lock the explicit review state matrix in the plan | Auto-decided | P1, P5 | The implementer should not invent `contract_invalid`, `degraded`, and `partial` behavior under pressure. | Leave fallback states implicit |
| 12 | Design | Order mixed-outcome queue items as `CONTRACT_INVALID`, `FAIL`, `AMBIGUOUS`, `REVIEW_SUCCESS` | Auto-decided | P1, P5 | Severity-first ordering preserves trust and reviewer throughput. | Arbitrary or chronological ordering |
| 13 | Design | Define a canonical proof-panel anatomy and CTA grammar | Auto-decided | P1, P5 | Each surfaced card needs one job and one vocabulary, or the UI will fragment immediately. | Generic "proof panel" language with implementer freedom |
| 14 | Design | Rewrite the README/front door around one wedge-first quickstart and move demo breadth lower | Auto-decided | P2, P4 | The current front door tells the wrong story first. | Keep the demo gallery as the first impression |
| 15 | Design | Continue with text-only review because `DESIGN_NOT_AVAILABLE` prevented mockup generation | Mechanical | P3 | No approved mockups existed, but the review still needed to lock hierarchy and states. | Block the plan until mockups exist |
| 16 | Design | Defer splitting canonical spec from review log to a later cleanup artifact | Taste | P3 | The split is valuable, but not load-bearing for this planning pass. | Force the split before finishing the review |
| 17 | Eng | Introduce a deterministic `compile_contract(...)` library API with thin adapters | Auto-decided | P1, P5 | Contract truth cannot live inside interactive agent glue. | Agent-first orchestration as the primary implementation seam |
| 18 | Eng | Keep `blessed_baseline`, `current_run`, and `proposed_baseline` as distinct immutable artifacts | Auto-decided | P1, P5 | Baseline lineage is the trust boundary. | In-place baseline mutation or ambiguous artifact roles |
| 19 | Eng | Add a separate run-level contract/result model before touching the shared evaluator verdict enum | Auto-decided | P4, P5 | The current `PASS / DEGRADED / FAIL` evaluator semantics do not cover the reviewed product states safely. | Force all new semantics into the existing evaluator enum now |
| 20 | Eng | Extend batch aggregation with `surfaced_cases` and `suppressed_cases` plus reasons | Auto-decided | P1, P5 | Safety requires both what is shown and what is intentionally hidden to be explicit. | Queue-only surfaced output with no suppressed audit trail |
| 21 | Eng | Require versioned JSON Schema and resource limits for the contract layer | Auto-decided | P1, P5 | Unattended runs need boring boundaries. | Prompt-shaped contracts with no hard limits |
| 22 | Eng | Type motion-window anti-goals by evidence kind or defer them from v1 | Auto-decided | P3, P5 | Otherwise they become a hidden second product. | Leave motion-window semantics fuzzy in v1 |
| 23 | Eng | Use the existing QA artifact at `~/.gstack/projects/MiaoDX-roboharness/mi-main-eng-review-test-plan-20260417-224344.md` as the phase-3 test-plan output | Mechanical | P4 | The artifact already captures the reviewed verification surface and avoids duplicate planning noise. | Create a parallel test-plan artifact with the same scope |
| 24 | Eng | Keep `assets/g1/` out of scope and untouched during the MuJoCo wedge work | Mechanical | P3, P4 | It was already untracked and unrelated to the reviewed wedge. | Let implementation blast radius drift into G1 assets |
| 25 | DX | Set the onboarding target to one wedge-first path: 10-minute quickstart, 5-minute TTHW target | Auto-decided | P1, P5 | The current docs are too diffuse for a first-time adopter evaluating trust. | Multi-demo onboarding as the primary path |
| 26 | DX | Define three progressive-disclosure levels: preset-first, prompt-assisted, raw JSON advanced | Auto-decided | P1, P5 | This resolves the mismatch between convenience and control without making raw JSON the first experience. | One-size-fits-all onboarding path |
| 27 | DX | Standardize a universal error envelope across CLI, HTML, and JSON outputs | Auto-decided | P1 | Developers need the same debugging grammar everywhere. | Surface-specific ad hoc error copy |
| 28 | DX | Make the README/front door part of the implementation plan, not follow-up polish | Auto-decided | P2, P4 | The product story and the wedge implementation need to land together. | Ship the wedge first and defer the front door |
| 29 | DX | Add an override matrix so developers know what is safe to customize | Auto-decided | P1, P5 | Trust breaks when users cannot tell which defaults are advisory versus load-bearing. | Leave overrides implicit |
| 30 | DX | Defer hosted playgrounds and broader ecosystem docs until after the wedge proves itself | Taste | P3 | They may be useful later, but they are not the next bottleneck. | Expand the scope into platform onboarding now |
