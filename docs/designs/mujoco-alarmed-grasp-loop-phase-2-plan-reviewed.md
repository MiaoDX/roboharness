<!-- /autoplan restore point: /home/mi/.gstack/projects/MiaoDX-roboharness/dongxu-dev-0415-1-autoplan-restore-20260415-103257.md -->
# Plan: MuJoCo Alarmed Grasp Loop Phase 2

Generated on 2026-04-15
Branch: `dongxu/dev-0415-1`
Status: APPROVED via `/autoplan` on 2026-04-15
Review source: `docs/designs/mujoco-alarmed-grasp-loop-phase-2-plan.md`

Related artifacts:
- `docs/designs/mujoco-alarmed-grasp-loop.md`
- `docs/designs/mujoco-alarmed-grasp-loop-eng-review.md`
- `examples/_mujoco_grasp_wedge.py`
- `tests/test_mujoco_grasp_wedge.py`

## Summary

Phase 1 landed the first complete MuJoCo wedge: evaluator-backed alarms,
`autonomous_report.json` as source of truth, `phase_manifest.json`, and an
alarm-first report summary. It already proves that a regression can be classified
and localized by metric, but it still leaves one sharp gap: the report tells the
agent which phase and views to inspect, yet it does not line up the failing
current captures against the blessed baseline inside the HTML itself.

Phase 2 closes that gap, but the goal is not prettier HTML. The goal is to make
the existing MuJoCo wedge more autonomous: the agent should get structured evidence,
see the paired failing-phase proof immediately, use the rerun hint, and continue
without manual replay. The HTML evidence section is supporting UX for that loop,
not the product center.

## Problem

The current report is strong on diagnosis metadata and weak on immediate visual
proof. `phase_manifest.json` already says `failed_phase_id == "approach"` with
`primary_views == ["side", "top"]`, but the artifact pack still makes the human or
agent do extra glue-work to prove the regression visually and decide whether the
loop is ready for the next rerun.

There is also no checked-in deterministic known-bad visual fixture pack for the
phase-localized failure case. The tests mutate metrics to prove the evaluator path,
but they do not prove the full report can point at a concrete bad visual example
in a repeatable way. And if phase 2 relies only on checked-in fixtures, it risks
proving renderer polish instead of loop reliability.

## Premises

1. The right phase-2 move is not to broaden the platform. It is to make the
   existing MuJoCo wedge more autonomous and less dependent on manual inspection.
2. The report should answer two questions immediately: what failed first, and
   what visual evidence plus rerun action should the agent use next.
3. A checked-in deterministic bad fixture is worth the repo weight for tests, but
   phase 2 also needs one generated-from-live-run validation path so the wedge does
   not collapse into demo theater.
4. The locked 1A-8A decisions from the phase-1 engineering review still hold:
   example-local packaging, `autonomous_report.json` as source of truth, no new
   public API promotion, no broad abstraction pass.

## User Outcome

After phase 2, the developer or coding agent should be able to inspect one artifact
pack and immediately see:
- the first failing phase and the metric that made it win alarm triage
- up to two manifest-selected primary-view comparison cards, rendered in priority
  order (`side`, then `top`), with `current` on the left and `baseline` on the right
- the top 1-2 metric deltas, formatted as `baseline -> current -> delta -> threshold`,
  that justify why each image pair matters
- whether the evidence is full, partial, ambiguous, missing, or structurally invalid
- the rerun action that preserves the diagnosis flow

No manual replay. No alt-tabbing across folders. No guessing whether the approach
view is actually worse than the baseline. Less human glue before the next rerun.

## In Scope

- Add a deterministic known-bad approach-regression fixture path for the MuJoCo
  wedge.
- Add a blessed baseline visual fixture pack that covers the views the wedge now
  considers primary, not just the sparse front-only images currently checked in.
- Extend the example-local wedge helpers so the report can resolve current and
  baseline evidence pairs for the first failing phase.
- Extend the alarm-first summary HTML to render a current-vs-baseline visual
  evidence section for the failing phase.
- Add one generated-from-live-run validation path that exercises the same report
  contract as the checked-in deterministic fixtures.
- Add focused tests around the fixture contract, phase localization, evidence
  resolution, and HTML rendering.

## Not In Scope

- New cross-simulator abstractions.
- Promoting new artifact dataclasses into `src/roboharness/` public API.
- New judge models or heuristic root-cause engines beyond the current evaluator
  and phase manifest.
- Full report renderer redesign outside the MuJoCo example wedge.
- New CLI surface area unless the implementation proves a small flag is required.
- Cross-simulator or second-task platform extraction in this phase.
- Temporal clip or scrubber UI beyond what is needed to prove the still-image
  approach is sufficient for the current wedge.

## What Already Exists

- `examples/_mujoco_grasp_wedge.py` already builds `autonomous_report.json`,
  evaluator-backed alarms, `phase_manifest.json`, and an alarm-first summary block.
- `tests/test_mujoco_grasp_wedge.py` already proves the evaluator localizes an
  `approach` regression and emits `primary_views == ["side", "top"]` plus
  `rerun_hint == "restore:pre_grasp"`.
- `examples/_mujoco_grasp_fixture.py` already defines the canonical phase order,
  phase labels, and per-phase primary camera mapping.
- `assets/example_mujoco_grasp/baseline_autonomous_report.json` already provides
  the blessed deterministic baseline metrics fixture.
- `assets/example_mujoco_grasp/` currently contains only a sparse front-view image
  set, which is not enough for the failing-phase side/top comparison the manifest
  now recommends.

## Proposed Delivery

### 1. Deterministic Visual Fixture Contract

Check in two small visual fixture packs under `assets/example_mujoco_grasp/`:

- a blessed baseline capture pack with enough views to support the current primary
  view rules, especially `approach/side` and `approach/top`
- a known-bad approach-regression capture pack with images that make the
  divergence visually obvious in those same views

The fixture packs should stay example-local. They are test/demo assets for this
wedge, not a new general asset format.

The metrics fixture remains `baseline_autonomous_report.json`. The known-bad path
does not introduce a second verdict pipeline. It is there to let the report show
concrete visual evidence in a deterministic way.

Decision for this plan: keep the visual fixtures as image packs plus the existing
metrics fixture. Do **not** check in a second full artifact pack with copied
derived manifests unless the implementation proves that image-only fixtures are
insufficient. That keeps `autonomous_report.json` as the only source-of-truth
contract while still making the evidence deterministic.

### 2. Evidence-Pair Resolver

Add one small example-local helper that:

- reads the first failing phase from the evaluator-backed manifest
- reads the manifest-selected primary views in deterministic priority order
- resolves current and baseline image pairs for those views from root-locked paths only
- returns one explicit record per selected view with image paths, render status,
  metric explanation payload, and one-line interpretation copy
- degrades explicitly, not vaguely, when evidence is partial, missing, ambiguous, or
  inconsistent with the manifest

The resolver contract should stay boring and example-local. A minimal local dataclass
such as `EvidencePair` is fine. It should include:
- `phase_id`
- `phase_label`
- `view_name`
- `current_image_path | None`
- `baseline_image_path | None`
- `status` (`full`, `partial`, `empty`, `mismatch`, `ambiguous`)
- `metric_explanations` (top 1-2 metrics only)
- `interpretation_caption`

Fallback behavior is part of the contract:
- `full`: render both images
- `partial`: render the available image plus a labeled placeholder for the missing side
- `empty`: render a single diagnostic empty panel with rerun guidance
- `mismatch`: render an inline contract-warning banner and skip the broken pair
- `ambiguous`: render the still image pair plus a warning that temporal proof is weak

No new shared renderer abstraction unless a tiny hook in `src/roboharness/reporting.py`
is strictly required.

### 3. Alarm-First HTML Upgrade

Extend `build_summary_html()` so the summary block contains, top-down:

- the existing alarm cards
- a new **Current vs Baseline** evidence section for the first failing phase
- a diagnostic action panel that says why the phase/view was selected and what rerun
  action to take next
- the existing phase timeline
- artifact metadata

The evidence section is the first proof surface, not a footer. It should follow this
exact rendering contract:
- render both manifest-selected primary views in deterministic priority order,
  `side` then `top`, with a hard cap of two cards
- each comparison card contains: `Phase badge`, `View badge`, `Current`, `Baseline`,
  `1-2 metric delta chips`, and a one-line interpretation caption
- always label `Current` and `Baseline` in visible copy, not just visually
- use metric copy in the format:
  `grip_center_error_mm: 8.4 -> 33.6 (+25.2, threshold 12.0)`
- if no failing phase exists, replace the evidence section with a success state:
  `No visual regression detected for the canonical primary views.`

The summary state model must be explicit:
- `FAIL/full evidence`
- `FAIL/partial evidence`
- `FAIL/empty evidence`
- `FAIL/manifest mismatch`
- `FAIL/ambiguous still-image evidence`
- `PASS/no failed phase`

The report must stay portable over CI, GitHub Pages, and SSH. The summary evidence
should therefore embed the baseline and current comparison images directly into the
HTML block as data URIs, or achieve the same self-contained outcome, instead of
depending on repo-relative asset paths.

Responsive and a11y requirements are part of the phase-2 contract:
- two comparison columns above a comfortable desktop breakpoint, stacked cards below it
- no horizontal scrolling for the comparison surface
- fixed image aspect ratio per card so current vs baseline stays visually comparable
- semantic headings plus `figure` / `figcaption` for evidence media
- descriptive alt text that includes phase, view, and evidence role
- non-color status text for every banner or badge
- minimum 44px targets only if any evidence affordance becomes clickable

### 4. Narrow Example Integration

Keep the integration local to the MuJoCo example path:

- `examples/mujoco_grasp.py` passes the current trial directory and blessed visual
  fixture root into the summary builder when `--report` is enabled.
- `examples/_mujoco_grasp_wedge.py` owns the evidence resolver and rendering data.
- No new package-level abstraction for generic baseline image management yet.

No new CLI flag is planned for phase 2. The reviewed default is to keep this behind
the existing `--report` flow unless the implementation proves an extra flag is
required for determinism or operator control.

### 5. Tests

Extend the phase-2 test plan in two lanes:

Default unit-test lane (`tests/test_mujoco_grasp_wedge.py`):
1. deterministic known-bad fixture reports `Verdict.FAIL`
2. the first failing phase remains `approach`
3. the chosen views remain `["side", "top"]`
4. the rerun hint remains `restore:pre_grasp`
5. the resolver returns two ordered evidence cards (`side`, then `top`) when both
   baseline and current evidence exist
6. `build_summary_html()` renders a `Current vs Baseline` section when evidence
   pairs exist
7. the rendered HTML labels `Current` and `Baseline` explicitly and includes the top
   1-2 metric explanations in the locked copy format
8. missing baseline or current images degrade cleanly without blowing up report
   generation
9. success/no-failing-phase renders the explicit success state instead of empty cards
10. manifest/view mismatch renders an inline diagnostic banner instead of crashing
11. fixture coverage stays aligned with the primary-view policy for the supported
    failing phase(s)

Optional MuJoCo-enabled live-validation lane (`tests/test_mujoco_grasp_live_validation.py`
or equivalent example smoke path):
12. one generated-from-live-run path still produces the same failing-phase contract
    as the deterministic fixture path
13. the live-run path is guarded with `pytest.importorskip("mujoco")` or bound to the
    existing MuJoCo-enabled CI/demo lane so default `pytest -q` in `[dev]` does not
    fail on missing optional deps
14. the live-run assertion checks the same machine contract as the fixture lane:
    failed phase, ordered views, rerun hint, and evidence status

## Files Expected To Change

- `docs/designs/mujoco-alarmed-grasp-loop-phase-2-plan.md`
- `examples/_mujoco_grasp_wedge.py`
- `examples/mujoco_grasp.py`
- `assets/example_mujoco_grasp/`
- `tests/test_mujoco_grasp_wedge.py`
- `tests/test_mujoco_grasp_live_validation.py` or the equivalent MuJoCo-enabled
  smoke path if the repo prefers example-driven validation over a new test file
- `src/roboharness/reporting.py` only for a tiny CSS hook that styles the new
  evidence cards, placeholders, and diagnostic banners

## Success Criteria

- A deterministic known-bad fixture makes the wedge fail in `approach`, not just
  somewhere later in the pipeline.
- One live generated run also exercises the same failing-phase evidence contract,
  so the feature is not only proving canned assets.
- The HTML report shows current-vs-baseline evidence for the first failing phase
  without requiring manual file browsing.
- The primary views in the HTML match the same evaluator-backed manifest already
  used by the machine-readable artifact path.
- The implementation stays example-local and does not expand into a cross-stack
  abstraction pass.
- Tests cover both the happy path and the missing-evidence fallback.
- The artifact pack reduces human glue in the diagnosis loop. At minimum, the plan
  must preserve the current rerun flow and make the next inspection target obvious.

## Failure Modes To Guard Against

- The HTML shows baseline images that do not correspond to the same phase/view as
  the current image.
- The summary path invents its own failure classification instead of consuming the
  existing evaluator/manifest result.
- The new fixture pack becomes sparse or inconsistent with `MUJOCO_GRASP_PRIMARY_VIEWS`,
  causing the report to suggest views that do not exist.
- The implementation quietly adds a second baseline contract in parallel with
  `autonomous_report.json`.
- The live-run validation path diverges from the deterministic fixture contract,
  making the tests pass while the real loop regresses.

## Open Questions

No blocking implementation question remains in the plan itself. The only remaining
review-level taste question is whether the summary should show rerun guidance before
or after the first paired evidence block. This review chooses evidence immediately
after alarms, then the action panel, because the proof should arrive before the
instruction.

## Initial Recommendation

Take the smallest explicit path:

- check in baseline and known-bad image fixtures
- keep the evidence-pair resolver in `examples/_mujoco_grasp_wedge.py`
- keep the report upgrade inside `build_summary_html()`
- keep the report self-contained by embedding the comparison images in the HTML
- add one live-run validation path using the same manifest/evidence contract
- avoid new CLI flags unless the implementation proves one is necessary

That is enough to make phase 2 real without turning it into a platform project.

## /autoplan Phase 1 CEO Review

Mode selected: `SELECTIVE_EXPANSION`

### 0A. Premise Challenge

- The right problem is still the MuJoCo wedge, not a broader cross-stack platform.
  Both outside voices agreed on that.
- The original phase-2 framing was slightly off. It treated HTML legibility as the
  main bottleneck. The real outcome is less human glue in the fail -> inspect ->
  rerun loop.
- Doing nothing leaves the wedge in an awkward middle state: the machine-readable
  contract points at the right phase, but the proof remains too manual and too easy
  to dismiss as a sparse demo.

### 0B. Existing Code Leverage

| Sub-problem | Existing code | Reuse decision |
|---|---|---|
| Failing-phase classification | `examples/_mujoco_grasp_wedge.py::build_phase_manifest()` | Reuse directly |
| View selection | `examples/_mujoco_grasp_fixture.py::MUJOCO_GRASP_PRIMARY_VIEWS` | Reuse directly |
| Rerun action | `examples/_mujoco_grasp_wedge.py::_build_rerun_hint()` | Reuse directly |
| Agent next action | `examples/_mujoco_grasp_wedge.py::_build_agent_next_action()` | Reuse directly |
| Alarm-first summary surface | `examples/_mujoco_grasp_wedge.py::build_summary_html()` | Extend in place |
| Standalone HTML embedding | `src/roboharness/reporting.py` image embedding path | Reuse, avoid parallel renderer |
| Deterministic evaluator path | `autonomous_report.json` + evaluator-backed assertions | Reuse, no second verdict path |

### 0C. Dream State Mapping

```text
CURRENT STATE                  THIS PLAN                      12-MONTH IDEAL
MuJoCo wedge emits            MuJoCo wedge shows             Agent consumes one
phase-local alarms and        paired current-vs-baseline     compact artifact pack,
manifest guidance, but        evidence for the first         restores the failing
the proof is still            failing phase and proves       checkpoint, reruns, and
somewhat manual.              the contract on live runs.     iterates across stacks.
```

### 0C-bis. Implementation Alternatives

APPROACH A: Minimal HTML Evidence Upgrade
  Summary: Add checked-in fixtures plus a side-by-side HTML evidence section for the
  failing phase. No extra live-run validation.
  Effort:  S
  Risk:    Medium
  Pros:
  - smallest diff
  - easiest to test
  - keeps the wedge narrow
  Cons:
  - risks demo theater
  - does not prove the real loop is healthier
  - overweights browser UX for an agent-facing feature
  Reuses:
  - existing manifest, alarms, summary HTML, and reporting path

APPROACH B: Loop-Closure Proof with Paired Evidence
  Summary: Add deterministic fixtures, add one generated-from-live-run validation
  path, and render paired current-vs-baseline evidence in the existing alarm-first
  report.
  Effort:  M
  Risk:    Low
  Pros:
  - proves the feature on both canned and live data
  - stays inside the current wedge and locked 1A-8A decisions
  - improves human and agent actionability without new platform abstraction
  Cons:
  - more assets and tests than the minimal option
  - still image evidence may later prove insufficient for temporal failures
  Reuses:
  - existing evaluator, manifest, trial outputs, and summary report path

APPROACH C: Reusable Evidence Contract Extraction
  Summary: extract a shared baseline-evidence contract now so future stacks can plug
  in without revisiting the MuJoCo design.
  Effort:  L
  Risk:    High
  Pros:
  - stronger long-term leverage
  - clearer product surface for future integrations
  - reduces later extraction pain
  Cons:
  - premature abstraction for one landed wedge
  - increases files and coupling before the wedge proves itself
  - violates the current locked example-local bias
  Reuses:
  - shared report renderer hooks only, but would require new contract surfaces

**RECOMMENDATION:** Choose Approach B. It is the smallest path that improves the
actual loop outcome instead of just polishing the report.

### 0D. Selective Expansion Analysis

- Complexity check: the reviewed phase should still stay in roughly 5 target areas
  (`examples/_mujoco_grasp_wedge.py`, `examples/mujoco_grasp.py`, assets, tests,
  and maybe a tiny reporting hook). That is acceptable for a feature enhancement.
- Minimum set: paired current-vs-baseline evidence, deterministic known-bad fixture
  support, one live-run validation path, and focused tests.
- Expansion scan:
  - accepted into scope: one live-run validation path using the same MuJoCo wedge
    contract, because it closes the biggest demo-theater risk inside the current
    blast radius
  - deferred: temporal clip/overlay UX, because it may be right later but it adds
    scope before the still-image approach is proven insufficient
  - deferred: second-stack or GR00T WBC + SONIC reapplication in this phase,
    because it is strategically valuable but outside the narrow blast radius for
    this immediate extension

### 0E. Temporal Interrogation

- Hour 1 foundations: the implementer needs a single explicit rule that the
  machine-readable manifest stays canonical and the HTML only consumes it.
- Hour 2-3 core logic: the ambiguity is whether image fixtures and live-run evidence
  resolve through the same interface, or whether the code accidentally forks.
- Hour 4-5 integration: the likely surprise is report portability. Repo-relative
  image paths will look fine locally and fail the CI/Pages/SSH story.
- Hour 6+ polish/tests: the missing plan item would have been fixture drift. The
  primary-view policy and checked-in assets must stay aligned in tests.

### 0F. Mode Selection Confirmation

`SELECTIVE_EXPANSION` is the right mode because this is not a greenfield product.
It is a landed feature enhancement with real opportunity to cherry-pick one or two
load-bearing improvements without opening a platform project.

### CEO DUAL VOICES

CODEX SAYS (CEO — strategy challenge)
- The loop-closure outcome matters more than report polish.
- HTML is not obviously the primary interface for an agent-facing feature.
- Deterministic fixtures without live-run validation risk demo theater.
- Example-local forever becomes strategy debt if there is no exit criterion.

CLAUDE SUBAGENT (CEO — strategic independence)
- Phase 2 should stay wedge-tight.
- The checked-in known-bad fixture is useful for tests but insufficient as product
  proof on its own.
- The moat is the agent-actionable failure pack and rerun path, not the HTML shell.

CEO DUAL VOICES — CONSENSUS TABLE:

| Dimension | Claude | Codex | Consensus |
|---|---|---|---|
| Premises valid? | Partial | Partial | CONFIRMED: reframe around autonomy |
| Right problem to solve? | Yes, wedge-tight | Yes, wedge-tight | CONFIRMED |
| Scope calibration correct? | Add live-run proof | Add live-run proof | CONFIRMED |
| Alternatives sufficiently explored? | Not enough | Not enough | CONFIRMED |
| Competitive/market risks covered? | Moat underspecified | Moat underspecified | CONFIRMED |
| 6-month trajectory sound? | Risky if example-local forever | Risky if example-local forever | CONFIRMED |

### Section 1. Architecture Review

No architecture rewrite is needed. The right boundary is still:
`mujoco_grasp.py` drives the run, `_mujoco_grasp_wedge.py` resolves evidence and
builds summary HTML, `reporting.py` embeds the resulting block, and the manifest
remains the canonical truth source. The architectural issue to avoid is a second
classification path living only in the HTML builder.

### Section 2. Error & Rescue Map

The main new failure path is evidence resolution, not simulation logic. Missing
baseline images, missing current images, sparse fixture coverage, or mismatched
phase/view keys must degrade to text guidance instead of exploding report
generation or silently showing the wrong comparison.

| Failure case | User sees | Rescue action | Plan fix |
|---|---|---|---|
| Baseline image missing | partial evidence card with explicit missing baseline label | rebuild or restore blessed baseline pack | explicit `partial` state + placeholder copy |
| Current image missing | partial evidence card with explicit missing current label | rerun from manifest-selected restore point | explicit `partial` state + rerun guidance |
| Manifest/view mismatch | mismatch banner, no fake comparison | inspect path contract and fixture tree | explicit `mismatch` state + root-locked resolution |
| Live run passes while fixture lane fails | divergent validation signal | compare fixture contract to generated run and fix drift | required live-validation lane in review scope |

### Section 3. Security & Threat Model

No meaningful new internet-facing attack surface is introduced if this remains a
local example/report feature. The real risk is trust-boundary drift inside the
artifact pack: if the HTML renders evidence not derived from the manifest-selected
phase/view contract, the feature becomes misleading rather than unsafe.

### Section 4. Data Flow & Interaction Edge Cases

The critical edge case is stale or sparse evidence. The plan must account for:
current image missing, baseline image missing, manifest points at a view that assets
do not cover, and a live run that passes while the deterministic fixture fails. Each
case should remain legible in the report.

### Section 5. Code Quality Review

The obvious quality risk is duplicating evidence resolution logic in multiple places.
There should be one explicit resolver for paired evidence and one explicit renderer
path in `build_summary_html()`. No clever abstraction layer is justified yet.

### Section 6. Test Review

The tests need to cover both fixture-backed and live-run-backed evidence resolution.
The most important addition is not another string-presence check, it is proof that
the same phase/view contract survives both deterministic and generated paths.

### Section 7. Performance Review

This phase should not materially change simulation runtime. The only performance risk
worth noting is report size inflation if the self-contained HTML embeds too many or
too-large comparison images. The implementation should keep the evidence narrow to
the selected failing phase and primary views.

### Section 8. Observability & Debuggability Review

The wedge is already strong on static artifacts. What it still needs is clearer proof
that the evidence section came from the same manifest-selected phase and views. That
can be done with explicit evidence-path text in the summary and tests that assert the
selection logic, not extra logging infrastructure.

### Section 9. Deployment & Rollout Review

No special rollout machinery is needed because this is example-local. The deployment
risk is distribution fit: if the new evidence only helps in local HTML and not in the
CI/Pages artifact path, then the feature ships but the user workflow does not improve.

### Section 10. Long-Term Trajectory Review

This stays on the right trajectory if phase 2 proves the loop is easier to run and
re-run, not merely easier to admire. The long-term debt item is example-local code
with no extraction trigger. That trigger should be stated now: only extract once a
second task or stack needs the same evidence contract.

### Section 11. Design & UX Review

At the CEO stage, the important call is not colors or CSS. It is that the proof surface
must move earlier and become explicit enough that the next rerun feels trustworthy.
The deferred risk is still the same: still images can under-explain temporal failures.
That is a later evidence-medium question, not a reason to broaden this phase now.


### CEO Failure Modes Registry

| Failure mode | Why it matters | Mitigation in reviewed plan |
|---|---|---|
| HTML becomes a second verdict path | the summary can drift away from the evaluator | keep `autonomous_report.json` + manifest canonical |
| Fixture-only proof passes while live loop regresses | phase 2 turns into demo theater | require one generated-from-live-run validation path |
| Example-local code calcifies into accidental platform debt | future stacks inherit a bad seam | state the extraction trigger explicitly and defer until a second stack needs it |
| Evidence arrives too late in the report | the developer loses trust and keeps hunting manually | move paired proof ahead of timeline and metadata |

### CEO COMPLETION SUMMARY

```text
+====================================================================+
|              CEO PLAN REVIEW — COMPLETION SUMMARY                   |
+====================================================================+
| Mode selected        | SELECTIVE_EXPANSION                          |
| Premise challenge    | accepted, with autonomy-first reframing      |
| Alternatives         | 3 examined, Approach B chosen                |
| Scope additions      | 1 accepted (live validation), 2 deferred     |
| Outside voices       | codex + subagent, 6/6 confirmed              |
| Not in scope         | written                                       |
| What already exists  | written                                       |
| Error & Rescue Map   | written                                       |
| Failure modes        | written                                       |
| Dream state delta    | written                                       |
| Overall CEO verdict  | phase 2 is worth building, stay wedge-tight  |
+====================================================================+
```

## Dream State Delta

If phase 2 lands as reviewed, the wedge still is not the 12-month platform. But it
does become a stronger autonomous debug packet: one failing phase, one paired proof,
one rerun hint, less manual interpretation, and one explicit path back into a rerun.
The remaining gap to the long-term ideal is portability across real workflows,
temporal evidence when stills are not enough, and eventual shared contract extraction.

## /autoplan Phase 2 Design Review

### Step 0. Design Scope

- UI scope: **yes**. This phase changes a user-facing static HTML report and the
  machine-facing artifact pack that points into it.
- `DESIGN.md`: not present. Reuse the report vocabulary that already exists in
  `src/roboharness/reporting.py` instead of inventing a new design system for one
  example-local wedge.
- Existing patterns worth preserving:
  - alarm cards
  - status-colored phase timeline cards
  - the compact summary block inserted ahead of the checkpoint gallery
  - explicit `Agent Next Action` copy in the console and HTML

Initial design completeness: **4/10**. The phase is strong on provenance and weak on
presentation. The implementer could still ship six different UIs while technically
following the current draft.

### Design Dual Voices

CODEX SAYS (design — UX challenge)
- The phase needs a real state model, not just the phrase `degrades cleanly`.
- The report hierarchy still risks putting metadata and instructions ahead of proof.
- The plan is specific about lineage and generic about layout, copy, and a11y.
- The open question about a CSS hook is smaller than the unresolved UI contract.

CLAUDE SUBAGENT (design — independent review)
- The hierarchy is backwards if proof lands after the timeline.
- The plan must explicitly say whether it renders one view or both.
- Success, empty, partial, and error states are missing.
- The comparison card anatomy and metric-copy format need to be locked before build.

### Design Litmus Scorecard

| Litmus check | Claude | Codex | Consensus |
|---|---|---|---|
| 1. Debug purpose unmistakable in first screen? | Yes | Yes | CONFIRMED |
| 2. One strong visual anchor present? | No, not yet | No, not yet | CONFIRMED gap -> fixed by evidence-first cards |
| 3. Page understandable by scanning headlines only? | No | No | CONFIRMED gap -> fix order and section names |
| 4. Each section has one job? | Partial | No | CONFIRMED gap -> split proof, action, timeline, metadata |
| 5. Are cards actually necessary? | Yes | Yes | CONFIRMED |
| 6. Does motion improve hierarchy or atmosphere? | N/A | N/A | N/A, static report artifact |
| 7. Would it still read cleanly without decorative shadows? | Yes | Yes | CONFIRMED |

### Pass 1. Information Architecture

**Rating:** 4/10 -> 10/10 after fixes.

What was wrong:
- the original plan still preserved `alarms -> next action -> phase timeline -> evidence`
- the implementer had no ASCII map of what belongs in the first viewport

Locked fix:

```text
FIRST VIEWPORT / SUMMARY BLOCK
==============================
[ Alarm summary ]
[ Evidence card: approach / side ]
[ Evidence card: approach / top ]
[ Why it failed + rerun action ]
[ Phase timeline ]
[ Artifact metadata ]
```

Constraint worship for this wedge: if only three things survive the first screen, they
are the alarm summary, the paired proof, and the rerun guidance. The timeline and
artifact metadata are supporting context.

### Pass 2. Interaction State Coverage

**Rating:** 3/10 -> 10/10 after fixes.

The draft named only the happy path and a vague degrade path. The reviewed plan now
locks the full state table below.

| Feature | Loading | Empty | Error | Success | Partial |
|---|---|---|---|---|---|
| Evidence surface | None. Static generated HTML, so no loading UI exists. | `No visual evidence available for the failing phase.` Show rerun hint. | Inline `contract mismatch` banner when phase/view mapping is inconsistent. | `No visual regression detected for the canonical primary views.` No comparison cards. | Render the available image plus a labeled placeholder and reason. |
| Diagnostic action panel | None | Keeps rerun guidance visible when no images exist. | Explains which path contract failed and what to inspect next. | States that no rerun is required unless the user is chasing a visual false negative. | Explains which side is missing and whether rerun is still trustworthy. |
| Phase timeline | None | Still renders for orientation. | Still renders, with failing phase highlighted if known. | Renders all phases as `ok`. | Renders normally; does not hide because evidence is partial. |

Implementation lock: when the success state renders, the `Agent Next Action`
panel should stop after the explicit success sentence. Do not append `Why this
proof` or `Rerun hint` copy when no failing phase exists, because that turns a
pass back into rerun-oriented guidance.

### Pass 3. User Journey & Emotional Arc

**Rating:** 4/10 -> 10/10 after fixes.

| Step | User does | User feels | Plan now specifies? |
|---|---|---|---|
| 1 | Opens `report.html` | `Tell me what broke.` | Yes, alarms lead. |
| 2 | Sees the first paired proof | `Okay, this is real.` | Yes, evidence lands before timeline. |
| 3 | Reads metric chips + caption | `I know why this card matters.` | Yes. |
| 4 | Reads rerun guidance | `I know where to restore and what to tune.` | Yes. |
| 5 | Expands to timeline/checkpoint gallery if needed | `I can inspect deeper without hunting.` | Yes. |

Time-horizon design:
- 5-second visceral: proof appears immediately
- 5-minute behavioral: rerun path is obvious
- 5-year reflective: the wedge teaches a reusable artifact contract instead of a one-off demo

### Pass 4. AI Slop Risk

**Rating:** 6/10 -> 9/10 after fixes.

Classifier: **APP UI**. This is a task-focused diagnostic workspace, not a marketing
surface. The plan now rejects generic dashboard mush and locks one explicit comparison
card schema:
- phase badge
- view badge
- `Current` figure
- `Baseline` figure
- 1-2 metric delta chips
- one-line interpretation caption

Additional anti-slop decisions:
- keep the visual language calm and utilitarian
- no decorative gradients or ornamental blobs in the evidence surface
- no three-column SaaS feature-grid treatment inside the report summary
- cards exist only because each selected view is an interaction-sized proof unit

### Pass 5. Design System Alignment

**Rating:** 5/10 -> 8/10 after fixes.

No `DESIGN.md` exists, so the right move is not `/design-consultation`. The report
already has an established visual language in `src/roboharness/reporting.py`:
neutral cards, severity colors, compact metadata tables, and image galleries.
Phase 2 should extend that language narrowly:
- reuse existing badge and card semantics
- add only the minimum CSS classes needed for evidence cards, placeholders, and
  mismatch banners
- do not introduce a new palette, type system, or layout shell

### Pass 6. Responsive & Accessibility

**Rating:** 2/10 -> 9/10 after fixes.

Responsive spec:
- desktop: each evidence card uses a two-column inner layout (`Current`, `Baseline`)
- narrow width: evidence cards stack vertically, but the card order remains `side`, then `top`
- no horizontal scroll for the evidence surface
- image aspect ratio stays fixed per card so visual comparison remains trustworthy

Accessibility spec:
- evidence section uses semantic headings and `figure` / `figcaption`
- alt text includes phase, view, and whether the image is current or baseline
- all status badges have visible text, not color only
- any future clickable affordance must preserve 44px targets and visible focus
- placeholder panels must say exactly what is missing (`baseline image missing`,
  `current capture missing`, `phase/view contract mismatch`)

### Pass 7. Unresolved Design Decisions

**Resolved now:**
- render both selected primary views, max two cards, `side` then `top`
- put paired evidence before the rerun-action copy
- keep the evidence surface passive in phase 2, no lightbox or zoom interaction
- use explicit state names instead of vague degradation language

**Deferred:**

| Decision deferred | If deferred, what happens |
|---|---|
| click-to-zoom / lightbox for evidence cards | users rely on the deeper checkpoint gallery for high-resolution inspection |
| temporal clip or scrubber evidence | still-image ambiguity remains possible for motion-rooted failures |

### Design NOT in Scope

- click-to-zoom or lightbox affordances for the new evidence cards
- temporal clip / scrubber UI
- broader report-shell redesign outside the MuJoCo example wedge

### Design What Already Exists

- `.alarm-grid` and `.alarm-card` patterns in `src/roboharness/reporting.py`
- `.phase-timeline` and `.phase-card-*` status vocabulary in the shared report CSS
- the current `Agent Next Action` copy block in `examples/_mujoco_grasp_wedge.py`
- the existing checkpoint gallery below the summary block, which remains the deep-inspection surface

### DESIGN PLAN REVIEW — COMPLETION SUMMARY

```text
+====================================================================+
|         DESIGN PLAN REVIEW — COMPLETION SUMMARY                    |
+====================================================================+
| System Audit         | no DESIGN.md, UI scope yes                  |
| Step 0               | 4/10 initial, state model + hierarchy focus |
| Pass 1  (Info Arch)  | 4/10 -> 10/10 after fixes                   |
| Pass 2  (States)     | 3/10 -> 10/10 after fixes                   |
| Pass 3  (Journey)    | 4/10 -> 10/10 after fixes                   |
| Pass 4  (AI Slop)    | 6/10 -> 9/10 after fixes                    |
| Pass 5  (Design Sys) | 5/10 -> 8/10 after fixes                    |
| Pass 6  (Responsive) | 2/10 -> 9/10 after fixes                    |
| Pass 7  (Decisions)  | 4 resolved, 2 deferred                      |
+--------------------------------------------------------------------+
| NOT in scope         | written (3 items)                           |
| What already exists  | written                                     |
| TODOS.md updates     | 2 items identified, held in this plan       |
| Approved Mockups     | none generated, text-only review            |
| Decisions made       | 6 added to plan                             |
| Decisions deferred   | 2 (listed above)                            |
| Overall design score | 4/10 -> 9/10                                |
+====================================================================+
```

Phase 2 is now design-complete enough to implement. The only remaining taste item is
whether the rerun action belongs above or below paired proof. This review keeps proof
first, then action.

## /autoplan Phase 3 Engineering Review

### Step 0. Scope Challenge With Actual Code

The current code shape is narrow and favorable:
- `examples/_mujoco_grasp_wedge.py` already owns the evaluator, manifest, and summary HTML
- `examples/mujoco_grasp.py` already has the current trial directory and the blessed
  baseline report path in hand when `--report` runs
- `src/roboharness/reporting.py` already owns the report CSS and the summary insertion point
- `tests/test_mujoco_grasp_wedge.py` already asserts phase localization and the
  current alarm-first sections

Real hidden complexity lives in three seams, not everywhere:
1. `build_summary_html()` currently has no access to baseline image roots or current
   trial directories, so the evidence resolver contract has to be added explicitly
2. the baseline asset tree does not mirror the manifest's phase/view contract today
3. the optional live-run validation cannot silently land in the default `[dev]` test
   lane because `pyproject.toml` does not install `mujoco` in `[dev]`

Complexity check: still acceptable. This remains an example-local enhancement with one
small optional expansion into a MuJoCo-enabled validation path.

### Engineering Dual Voices

CODEX SAYS (eng — architecture challenge)
- unavailable in this resumed session; the direct Codex CLI review stalled while
  traversing the repo and did not return a final verdict

CLAUDE SUBAGENT (eng — independent review)
- unavailable in this turn; no fresh engineering subagent run was started

ENG DUAL VOICES — CONSENSUS TABLE:

| Dimension | Claude | Codex | Consensus |
|---|---|---|---|
| Architecture sound? | N/A | N/A | outside voices unavailable |
| Test coverage sufficient? | N/A | N/A | outside voices unavailable |
| Performance risks addressed? | N/A | N/A | outside voices unavailable |
| Security threats covered? | N/A | N/A | outside voices unavailable |
| Error paths handled? | N/A | N/A | outside voices unavailable |
| Deployment risk manageable? | N/A | N/A | outside voices unavailable |

### Section 1. Architecture Review

Approved architecture:

```text
examples/mujoco_grasp.py
    |
    +--> collect_phase_metrics()
    +--> build_autonomous_report()
    +--> evaluate_autonomous_report()
    +--> build_alarms()
    +--> build_phase_manifest()
    +--> resolve_evidence_pairs(trial_dir, baseline_visual_root, manifest, report)
    +--> build_summary_html(report, alarms, manifest, evidence_pairs)
    +--> src/roboharness/reporting.generate_html_report(summary_html=...)

assets/example_mujoco_grasp/
    +--> baseline_autonomous_report.json
    +--> baseline_visual/<phase>/<view>_rgb.png
    +--> known_bad_visual/<phase>/<view>_rgb.png
```

Architecture decisions locked here:
- keep the resolver and any `EvidencePair` dataclass in `examples/_mujoco_grasp_wedge.py`
- pass explicit roots into the resolver; do not make `src/roboharness/reporting.py`
  know anything about baseline asset locations
- allow one tiny shared CSS hook in `src/roboharness/reporting.py` for evidence cards
  and diagnostic banners because the summary block already depends on shared CSS classes

### Section 2. Code Quality Review

The main code-quality risk is duplication:
- path selection logic must not live separately in `build_phase_manifest()` and the
  new resolver
- base64/data-URI logic should not be duplicated across summary-card branches
- missing-evidence fallback text should come from one explicit status mapping, not
  be stitched together ad hoc in multiple HTML templates

Explicit-over-clever recommendation:
- use one boring example-local resolver
- use one boring render function per comparison card or status panel
- reject any attempt to invent a generic baseline-image manager in `src/`

Path-safety requirement:
- only resolve evidence paths inside the current trial root and blessed baseline
  visual root
- reject `..`, absolute escapes, or unknown phase/view paths as `mismatch`

### Section 3. Test Review

Current test surface is too thin for phase 2. The existing suite proves manifest and
section names, but not the new evidence contract.

CODE PATH COVERAGE
===========================
[+] examples/_mujoco_grasp_wedge.py
    |
    +-- resolve_evidence_pairs(...)
    |   +-- [GAP] full evidence for `approach/side` and `approach/top`
    |   +-- [GAP] baseline image missing -> labeled placeholder
    |   +-- [GAP] current image missing -> labeled placeholder
    |   +-- [GAP] no failing phase -> success state, no comparison cards
    |   +-- [GAP] manifest/view mismatch -> diagnostic banner, no crash
    |   `-- [GAP] ambiguous still-image state -> warning banner + rerun guidance
    |
    `-- build_summary_html(...)
        +-- [★  TESTED] alarm-first sections exist today
        +-- [GAP] evidence cards render in locked order: `side`, then `top`
        +-- [GAP] metric chips render in `baseline -> current -> delta -> threshold` format
        +-- [GAP] partial evidence cards retain visible `Current` / `Baseline` labels
        `-- [GAP] pass state omits empty comparison chrome and shows explicit success copy

[+] examples/mujoco_grasp.py
    |
    +-- report wiring under `--report`
    |   +-- [GAP] passes trial root + blessed baseline visual root into resolver
    |   +-- [GAP] keeps console summary useful in headless mode when evidence is partial
    |   `-- [GAP] preserves existing `--report` flow with no new required flags
    |
    `-- live validation path
        `-- [GAP] [->E2E/optional] MuJoCo-enabled run reproduces the same contract as fixtures

USER FLOW COVERAGE
===========================
[+] Developer opens `report.html` after a failing run
    +-- [GAP] sees proof before supporting metadata
    +-- [GAP] can tell which side is missing when evidence is partial
    `-- [GAP] can tell when still images are suggestive but not definitive

[+] Headless / CI operator reads terminal output first
    `-- [GAP] console summary still points to failed phase, selected views, and rerun hint

[+] Demo-enabled validation lane
    `-- [GAP] [->E2E] real MuJoCo run executes under optional dependency guard / CI lane

---------------------------------
COVERAGE: 1/14 paths meaningfully tested today (7%)
  Code paths: 1/11 (9%)
  User flows: 0/3 (0%)
QUALITY: ★★★: 0  ★★: 0  ★: 1
GAPS: 13 paths need tests (1 optional live-validation path)
---------------------------------

Regression rule triggered:
- `build_summary_html()` already exists and phase 2 changes its behavior materially.
  That means a regression test for the new evidence section is **required**, not optional.

### Test Plan Artifact

A QA-facing artifact should be written under `~/.gstack/projects/MiaoDX-roboharness/`.
It should tell a tester exactly what to verify:
- failing run -> report summary shows evidence cards
- pass/no-failure run -> report shows success copy instead of empty evidence chrome
- partial evidence -> placeholder + explanation
- manifest mismatch -> diagnostic banner
- MuJoCo-enabled run -> same failing-phase contract as fixture lane

### Section 4. Performance Review

No new database or network path exists. The only meaningful performance risk is report
size and duplication:
- the main report body already embeds checkpoint images for every phase
- the summary evidence section will duplicate up to four images if it embeds `current`
  and `baseline` for two views

Performance guardrails:
- cap the summary evidence surface at two views and four images total
- do not embed any non-selected views in the summary
- keep baseline visual fixtures small and deterministic
- avoid introducing a second full image gallery in the summary block

### Engineering NOT in Scope

- a shared baseline-image abstraction in `src/roboharness/`
- a non-MuJoCo live-validation path in this phase
- report-shell virtualization, pagination, or image lazy-loading redesign
- widening the default `[dev]` extra just to make the optional MuJoCo live test run everywhere

### Engineering What Already Exists

| Sub-problem | Existing code | Reuse decision |
|---|---|---|
| Summary insertion point | `src/roboharness/reporting.generate_html_report()` | Reuse; tiny CSS hook only |
| Manifest-selected views | `build_phase_manifest()` + `MUJOCO_GRASP_PRIMARY_VIEWS` | Reuse directly |
| Headless console summary | `examples/mujoco_grasp.py::main()` | Extend in place |
| Trial image layout | existing per-phase trial directories | Reuse directly |
| Default unit tests | `tests/test_mujoco_grasp_wedge.py` | Extend in place |
| Optional MuJoCo CI lane | existing demo / MuJoCo-enabled environments | Reuse; do not force into default `[dev]` lane |

### Failure Modes Registry

| Codepath | Realistic failure | Test planned? | Error handling? | User-visible? | Critical gap? |
|---|---|---|---|---|---|
| Resolver -> baseline asset lookup | baseline `approach/top` missing | Yes | Yes, placeholder panel | Yes | No |
| Resolver -> current trial lookup | current selected capture missing | Yes | Yes, placeholder panel | Yes | No |
| Resolver -> path normalization | path escapes baseline root | Yes | Yes, mismatch banner | Yes | No |
| Summary render -> wrong phase/view pairing | baseline image does not match manifest-selected view | Yes | Partially, via explicit badges | Yes | **Yes until tested** |
| Live validation lane | optional MuJoCo test never runs in CI | Yes | No | No, failure would be silent | **Yes** |
| Pass-state render | no failing phase still renders empty evidence chrome | Yes | Yes | Yes | No |

Critical gap resolution required before implementation is considered complete:
- bind the live-validation path to a MuJoCo-enabled CI/demo lane or a guarded optional
  test file so it cannot silently disappear

### Worktree Parallelization Strategy

| Step | Modules touched | Depends on |
|---|---|---|
| 1. Baseline + known-bad visual fixture tree | `assets/example_mujoco_grasp/` | — |
| 2. Evidence resolver contract | `examples/` | Step 1 path contract |
| 3. HTML evidence cards + CSS hook | `examples/`, `src/roboharness/` | Step 2 |
| 4. CLI/report wiring | `examples/` | Step 2 |
| 5. Unit tests + optional live validation | `tests/` | Steps 2-4 |

Parallel lanes:
- Lane A: Step 1 (assets)
- Lane B: Step 2 -> Step 3 -> Step 4 (shared `examples/` modules, sequential)
- Lane C: Step 5 after Lane B stabilizes

Execution order:
- launch Lane A first
- once the asset contract is agreed, Lane B owns the implementation seam
- Lane C starts after the resolver and HTML contract are stable

Conflict flags:
- Lane B is intentionally sequential because `examples/_mujoco_grasp_wedge.py` is the
  shared hotspot
- Lane C is low-conflict, but test expectations will churn if Lane B changes the card contract late

### Engineering Completion Summary

- Step 0: Scope Challenge — scope accepted with one explicit optional-test split for MuJoCo live validation
- Architecture Review: 4 issues found, all resolved in-plan
- Code Quality Review: 3 issues found, all resolved in-plan
- Test Review: diagram produced, 13 gaps identified
- Performance Review: 1 issue found, bounded by card/image caps
- NOT in scope: written
- What already exists: written
- TODOS.md updates: 2 items identified, held in this plan
- Failure modes: 2 critical gaps flagged, both tied to test/CI discipline
- Outside voice: unavailable in this resumed session
- Parallelization: 3 lanes, 1 light parallel / 2 sequential
- Lake Score: 5/5 recommendations chose the complete option

## /autoplan Phase 3.5 DX Review

### Step 0. DX Scope Assessment

Product type: **developer-facing CLI example + static debug artifact pack**.
Mode: **DX POLISH**.

TARGET DEVELOPER PERSONA
========================
Who:       robotics / AI-agent developer debugging one MuJoCo task from a terminal or CI artifact
Context:   they already have the repo or package, they want one command to produce a report that CC/Codex can act on
Tolerance: about 5 minutes to first successful artifact on a cold machine, about 30 seconds to understand a failure once the artifact exists
Expects:   one copy-paste command, clear output paths, headless compatibility, no hidden GUI requirement

### Developer Empathy Narrative

I open the README and the MuJoCo Grasp section is easy to find. Good. The command is
short: install the `demo` extra, then run `python examples/mujoco_grasp.py --report`.
That part feels modern enough. But once I imagine the run failing, the docs still hand
me more of a generic report than a deterministic debug loop. I can infer that
`phase_manifest.json` and `alarms.json` matter, but the README does not tell me what I
should inspect first when the run regresses. If the HTML makes me hunt through a
summary table, then a timeline, then a gallery, the promise of `one artifact pack, no
human glue` weakens immediately. I do not want more files. I want the first screen to
say: `approach`, `side/top`, `baseline vs current`, `rerun from pre_grasp`. If the
report can do that, I trust it. If not, I am back to replaying the task or poking at
folders by hand.

### Reference DX Benchmark (no tight robotics equivalent)

This wedge has no direct public competitor with the same robotics + artifact-pack loop,
so this review uses reference DX tiers instead of pretending there is a perfect analog.

| Tier | TTHW | Notable DX choice |
|---|---|---|
| Champion | < 2 min | zero-friction magic in one command or sandbox |
| Competitive | 2-5 min | one command plus a visible, meaningful artifact |
| Needs work | > 5 min | extra setup or unclear artifact interpretation |
| This phase target | 2-5 min after `demo` deps are present | one command, one artifact pack, first failure obvious |

### Magical Moment Specification

Magical moment: the developer opens the generated artifact pack and immediately sees
`which phase failed, what the current vs baseline proof looks like, and where to rerun`.

Delivery vehicle: **copy-paste demo command**.

Implementation requirements:
- preserve the existing `python examples/mujoco_grasp.py --report` entrypoint
- do not require a new CLI flag to see the improved proof surface
- keep console output useful for headless users by printing verdict, failed phase,
  selected views or evidence status, and rerun hint
- make the first HTML viewport earn trust without requiring the developer to open
  other files first

### DX Dual Voices

CODEX SAYS (DX — developer experience challenge)
- unavailable in this resumed session; no final outside DX review was returned

CLAUDE SUBAGENT (DX — independent review)
- unavailable in this turn; no fresh DX subagent run was started

DX DUAL VOICES — CONSENSUS TABLE:

| Dimension | Claude | Codex | Consensus |
|---|---|---|---|
| 1. Getting started < 5 min? | N/A | N/A | outside voices unavailable |
| 2. API/CLI naming guessable? | N/A | N/A | outside voices unavailable |
| 3. Error messages actionable? | N/A | N/A | outside voices unavailable |
| 4. Docs findable & complete? | N/A | N/A | outside voices unavailable |
| 5. Upgrade path safe? | N/A | N/A | outside voices unavailable |
| 6. Dev environment friction-free? | N/A | N/A | outside voices unavailable |

### Developer Journey Map

| Stage | Developer does | Friction points | Status |
|---|---|---|---|
| 1. Discover | finds the MuJoCo Grasp section in `README.md` | low | ok |
| 2. Install | installs `roboharness[demo]` | optional dep cost exists, but expected | ok |
| 3. Run | executes `python examples/mujoco_grasp.py --report` | none if deps exist | ok |
| 4. Locate artifact pack | reads printed output paths | output already prints paths, phase 2 must keep it | fixed |
| 5. Open summary report | opens `report.html` | current summary still too metadata-heavy | fixed in plan |
| 6. Understand why it failed | reads evidence cards and metric chips | previously underspecified | fixed in plan |
| 7. Choose rerun target | reads rerun hint and action panel | previously too detached from proof | fixed in plan |
| 8. Validate deeper | drops into timeline and checkpoint gallery | no issue, existing surface already works | ok |
| 9. Repeat in CI/demo lane | relies on optional MuJoCo validation | needed explicit env split | fixed in plan |

### First-Time Developer Confusion Report

FIRST-TIME DEVELOPER REPORT
===========================
Persona: robotics-cli-dev
Attempting: MuJoCo grasp report debugging

CONFUSION LOG:
T+0:00  I find the MuJoCo Grasp quick start quickly. Good sign.
T+1:00  I can run one command, but I do not yet know what file will actually matter if the run fails.
T+2:00  The console prints verdict and rerun hint, which helps, but the report still needs to earn trust immediately.
T+3:00  If I open the HTML and proof is buried under metadata or a timeline, I stop trusting the artifact pack.
T+4:00  If the first screen shows `approach`, `side/top`, `baseline vs current`, and `restore:pre_grasp`, I know what to do next.

Addressed in this review:
- yes, by moving paired proof ahead of supporting metadata
- yes, by locking explicit state copy for partial / missing / mismatch evidence
- yes, by preserving the one-command `--report` flow

### DX Pass 1. Getting Started Experience

**Rating:** 6/10 -> 8/10 after fixes.

Strengths:
- one copy-paste command already exists
- the artifact pack is generated locally, no extra service required

Gap to close:
- the plan needs to guarantee that the first report screen explains the failure with
  almost no extra hunting

### DX Pass 2. API / CLI Design

**Rating:** 7/10 -> 9/10 after fixes.

Phase 2 correctly keeps the current CLI shape:
- no required new flag
- `--report` remains the single switch for the improved proof surface
- baseline configuration stays optional and explicit via the existing path argument

The main DX fix is not a new option. It is making the output of the existing command
more legible and more deterministic.

### DX Pass 3. Error Messages & Debugging

**Rating:** 5/10 -> 8/10 after fixes.

New error-copy requirement for the phase-2 surface:
- every evidence failure state should say **problem + cause + fix**
- examples:
  - `Baseline image missing for approach/top. Rebuild or restore the blessed baseline pack.`
  - `Current capture missing for approach/side. Re-run from restore:pre_grasp to rebuild evidence.`
  - `Manifest/view contract mismatch. The report requested approach/top, but no matching asset exists under the blessed baseline root.`

### DX Pass 4. Documentation & Learning

**Rating:** 5/10 -> 7/10 after fixes.

This plan still does not include a README or docs sync. That is acceptable for phase 2
only because the existing command path is already discoverable and the bigger win is the
artifact itself. A post-ship `/document-release` pass should mirror the new evidence
contract into user-facing docs.

### DX Pass 5. Upgrade & Migration Path

**Rating:** 9/10 -> 9/10.

No public API migration is introduced. The example-local wedge keeps the same entrypoint
and the same artifact names. That means upgrade risk is low as long as phase 2 does not
rename existing machine-readable files.

### DX Pass 6. Developer Environment & Tooling

**Rating:** 4/10 -> 8/10 after fixes.

This is the hidden DX trap. The plan wants one live generated validation path, but the
repo's default `[dev]` extra does not install `mujoco`. The fix is explicit separation:
- default unit tests stay dependency-light and deterministic
- optional live validation runs only in a MuJoCo-enabled environment or guarded CI lane

That keeps `pytest -q` honest while still making the phase real.

### DX Pass 7. Community & Ecosystem

**Rating:** 6/10 -> 6/10.

No change in this phase. The wedge remains a strong example, but ecosystem leverage is
intentionally deferred until the evidence contract proves itself on a second task or stack.

### DX Pass 8. Measurement & Feedback Loops

**Rating:** 5/10 -> 7/10 after fixes.

The phase now includes one measurable boomerang:
- the live validation lane must reproduce the same failing-phase contract as the deterministic fixtures

Post-implementation recommendation:
- run `/devex-review` after shipping so reality can be compared to this plan's TTHW and friction assumptions

### DX NOT in Scope

- a broader README / docs rewrite in this phase
- hosted playgrounds or browser sandboxes
- new SDK / CLI surfaces for selecting baseline visual packs
- cross-platform environment automation beyond the existing example command

### DX What Already Exists

- a prominent MuJoCo Grasp quick start in `README.md`
- the existing `--report` entrypoint in `examples/mujoco_grasp.py`
- console summary lines that already print verdict, failed phase, and rerun hint
- machine-readable artifacts already written alongside the report

### DX PLAN REVIEW — SCORECARD

```text
+====================================================================+
|              DX PLAN REVIEW — SCORECARD                             |
+====================================================================+
| Dimension            | Score  | Prior  | Trend  |
|----------------------|--------|--------|--------|
| Getting Started      | 8/10   |   —    |   ↑    |
| API/CLI/SDK          | 9/10   |   —    |   ↑    |
| Error Messages       | 8/10   |   —    |   ↑    |
| Documentation        | 7/10   |   —    |   ↑    |
| Upgrade Path         | 9/10   |   —    |   =    |
| Dev Environment      | 8/10   |   —    |   ↑    |
| Community            | 6/10   |   —    |   =    |
| DX Measurement       | 7/10   |   —    |   ↑    |
+--------------------------------------------------------------------+
| TTHW                 | 5 min  | 3 min  |   ↑    |
| Competitive Rank     | Competitive                                  |
| Magical Moment       | designed via copy-paste demo command         |
| Product Type         | CLI example + static debug artifact          |
| Mode                 | POLISH                                       |
| Overall DX           | 6/10   | 8/10   |   ↑    |
+====================================================================+
| DX PRINCIPLE COVERAGE                                               |
| Zero Friction      | covered with one-command path, still doc-light  |
| Learn by Doing     | covered via report artifact and checkpoint flow  |
| Fight Uncertainty  | covered by explicit evidence-state copy          |
| Opinionated + Escape Hatches | covered, no new flag required         |
| Code in Context    | covered by manifest + evidence pairing          |
| Magical Moments    | covered by first-viewport proof                |
+====================================================================+
```

### DX IMPLEMENTATION CHECKLIST

```text
DX IMPLEMENTATION CHECKLIST
============================
[ ] Time to hello world < 5 min after demo deps are present
[ ] Installation is one command at the README layer
[ ] First run produces meaningful output
[ ] Magical moment delivered via `python examples/mujoco_grasp.py --report`
[ ] Every evidence error message has: problem + cause + fix
[ ] API/CLI naming remains guessable without docs
[ ] Existing flags keep sensible defaults
[ ] Report summary explains the failure without requiring extra file hunting
[ ] Examples show real use cases, not just pass-only happy paths
[ ] Upgrade path keeps artifact names stable
[ ] Optional MuJoCo validation is guarded and wired into a real lane
[ ] Works in CI/CD without special handling in the default `[dev]` unit-test lane
[ ] Headless users can act from console output alone
[ ] Post-ship `/document-release` updates README/docs with the new evidence contract
[ ] Post-ship `/devex-review` measures reality against this plan
```

DX overall: the plan is now good enough for implementation. The biggest remaining DX
tradeoff is deliberate: docs polish is deferred so the phase can stay focused on the
artifact pack itself.

## Cross-Phase Themes

- **Theme: proof before polish** — flagged in CEO, Design, and DX. High-confidence
  signal that the phase succeeds only if paired evidence becomes the first-class output.
- **Theme: one canonical truth source** — flagged in CEO, Design, and Eng. The HTML,
  manifest, and tests must all consume the same evaluator-backed contract.
- **Theme: no fixture theater** — flagged in CEO, Eng, and DX. Deterministic assets are
  necessary, but the phase is incomplete unless one live MuJoCo-enabled validation path
  proves the same contract.
- **Theme: example-local now, extraction later** — flagged in CEO and Eng. The
  extraction trigger should remain explicit: only widen the abstraction once a second
  task or stack genuinely needs it.

## Approval

Approved by user on 2026-04-15.

- Final reviewed artifact: this file
- Deferred follow-ups mirrored to repo-root `TODOS.md`
- Implementation can proceed directly from the locked design, engineering, and DX
  sections above

## Decision Audit Trail

| # | Phase | Decision | Classification | Principle | Rationale | Rejected |
|---|---|---|---|---|---|---|
| 1 | CEO | Select `SELECTIVE_EXPANSION` mode | Mechanical | P3, P5 | This is a landed feature enhancement, not a greenfield bet or a hotfix. | `EXPANSION`, `HOLD`, `REDUCTION` |
| 2 | CEO | Keep the work wedge-tight, no cross-stack expansion in phase 2 | Mechanical | P2, P4, P5 | Broad platform work increases files and abstractions before the wedge proves itself. | Cross-simulator extraction now |
| 3 | CEO | Reframe the phase outcome around loop closure, not report polish | Auto-decided | P1, P6 | The user outcome is less human glue in the rerun loop, not nicer chrome. | HTML-first framing |
| 4 | CEO | Add one live-run validation path alongside deterministic fixtures | Auto-decided | P1, P2 | It fixes the largest completeness gap without leaving the current blast radius. | Fixture-only proof |
| 5 | CEO | Keep manifest-driven evidence selection canonical | Mechanical | P4, P5 | One truth source prevents the HTML from inventing its own verdict path. | Separate HTML-only classification |
| 6 | CEO | Defer temporal evidence UX and second-stack validation | Auto-decided | P3, P5 | Both are valuable, but not load-bearing for this narrow phase if the core loop still lacks paired proof. | Broadening this phase |
| 7 | Design | Move paired evidence ahead of the action panel and timeline | Taste | P1, P5 | Proof should arrive before instruction so the rerun advice is trusted. | `alarms -> next action -> evidence -> timeline` |
| 8 | Design | Render both selected primary views, max two cards, in `side` then `top` order | Auto-decided | P1, P5 | The manifest already says both views matter, and hiding one would reintroduce guesswork. | First-view-only rendering |
| 9 | Design | Lock explicit summary states: `full`, `partial`, `empty`, `mismatch`, `ambiguous`, `pass` | Auto-decided | P1, P5 | The implementer should not invent fallback behavior under pressure. | Vague `degrades cleanly` language |
| 10 | Design | Lock metric copy to `baseline -> current -> delta -> threshold`, top 1-2 metrics only | Auto-decided | P5 | This is specific enough to guide implementation without turning into a metric dump. | Free-form prose or raw metric tables |
| 11 | Design | Keep evidence cards passive in phase 2, no lightbox or zoom | Taste | P3, P5 | The deeper checkpoint gallery already exists, so click chrome is not the next bottleneck. | New zoom / lightbox interaction |
| 12 | Design | Add responsive + accessibility requirements to the plan, not as TODO theater | Auto-decided | P1 | Static HTML still needs explicit layout and assistive semantics. | Defer responsive/a11y to implementation taste |
| 13 | Eng | Keep the new resolver and any `EvidencePair` dataclass example-local | Auto-decided | P4, P5 | The wedge has not earned a shared abstraction yet. | New `src/`-level evidence package |
| 14 | Eng | Mirror baseline and known-bad visual fixtures to the trial path contract under asset roots | Auto-decided | P5 | Matching the trial naming/layout keeps pairing logic explicit and testable. | Ad hoc asset filenames |
| 15 | Eng | Split live validation into an optional MuJoCo-enabled lane instead of the default `[dev]` unit-test lane | Auto-decided | P1, P3 | Completeness requires the live path, but default `pytest -q` cannot start depending on optional MuJoCo installs. | Fixture-only testing or forcing `mujoco` into `[dev]` |
| 16 | Eng | Allow a tiny shared CSS hook in `src/roboharness/reporting.py` only | Mechanical | P3, P5 | The summary block already depends on shared report CSS, and a tiny hook is lower-risk than inline style sprawl. | No styling hook or broader renderer refactor |
| 17 | Eng | Cap the summary evidence surface at two views / four images total | Auto-decided | P3 | It bounds HTML size and duplication without weakening the diagnostic loop. | Show every phase or every camera in the summary |
| 18 | Eng | Enforce root-locked evidence path resolution and treat escapes as mismatches | Auto-decided | P1, P5 | This keeps the new file-reading path explicit and safe. | Trust arbitrary relative paths from data |
| 19 | DX | Preserve the existing `--report` flow and avoid new required CLI flags | Auto-decided | P5 | The magical moment is already one command; phase 2 should make that command better, not more complicated. | New flag for visual baseline selection |
| 20 | DX | Preserve and slightly enrich headless console output rather than pushing all meaning into HTML | Auto-decided | P1 | CI and SSH users need to act before opening a browser. | HTML-only explanation |
| 21 | DX | Defer README / docs polish to a post-ship `/document-release` pass | Taste | P3 | The artifact pack is the load-bearing phase-2 change; docs polish matters but should not crowd out the implementation seam. | Expand this phase into broader docs sync |
| 22 | DX | Require every evidence failure state to say problem + cause + fix | Auto-decided | P1 | Error empathy is part of the developer contract, not optional polish. | Bare placeholders or silent fallback |
