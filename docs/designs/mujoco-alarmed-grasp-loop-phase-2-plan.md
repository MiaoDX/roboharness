<!-- /autoplan restore point: /home/mi/.gstack/projects/MiaoDX-roboharness/dongxu-dev-0415-1-autoplan-restore-20260415-103257.md -->
# Plan: MuJoCo Alarmed Grasp Loop Phase 2

Generated on 2026-04-15
Branch: `dongxu/dev-0415-1`
Status: Draft for `/autoplan`

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

Phase 2 closes that gap. The goal is to add one deterministic known-bad approach
regression fixture and use it to make the HTML show direct current-vs-baseline
evidence for the first failing phase. This is not a platform rewrite. It is a
small wedge extension that makes the failure legible without replaying MuJoCo.

## Problem

The current report is strong on diagnosis metadata and weak on immediate visual
proof. `phase_manifest.json` already says `failed_phase_id == "approach"` with
`primary_views == ["side", "top"]`, but a developer or coding agent still has to
open separate image files and mentally compare them to the blessed baseline. That
is exactly the glue-work the wedge is supposed to remove.

There is also no checked-in deterministic known-bad visual fixture pack for the
phase-localized failure case. The tests mutate metrics to prove the evaluator path,
but they do not prove the full report can point at a concrete bad visual example
in a repeatable way.

## Premises

1. The right phase-2 move is not to broaden the platform. It is to make the
   existing MuJoCo wedge more self-explanatory.
2. The report should answer two questions immediately: what failed first, and
   what does the current view look like next to the blessed baseline.
3. A checked-in deterministic bad fixture is worth the repo weight if it turns
   the phase-localized failure mode into something tests and humans can see.
4. The locked 1A-8A decisions from the phase-1 engineering review still hold:
   example-local packaging, `autonomous_report.json` as source of truth, no new
   public API promotion, no broad abstraction pass.

## User Outcome

After phase 2, the developer or coding agent should be able to open one HTML
report and immediately see:
- the first failing phase
- the primary views worth inspecting
- the current failing frame beside the blessed baseline frame
- the metric deltas that justify why that image pair matters

No manual replay. No alt-tabbing across folders. No guessing whether the approach
view is actually worse than the baseline.

## In Scope

- Add a deterministic known-bad approach-regression fixture path for the MuJoCo
  wedge.
- Add a blessed baseline visual fixture pack that covers the views the wedge now
  considers primary, not just the sparse front-only images currently checked in.
- Extend the example-local wedge helpers so the report can resolve current and
  baseline evidence pairs for the first failing phase.
- Extend the alarm-first summary HTML to render a current-vs-baseline visual
  evidence section for the failing phase.
- Add focused tests around the fixture contract, phase localization, evidence
  resolution, and HTML rendering.

## Not In Scope

- New cross-simulator abstractions.
- Promoting new artifact dataclasses into `src/roboharness/` public API.
- New judge models or heuristic root-cause engines beyond the current evaluator
  and phase manifest.
- Full report renderer redesign outside the MuJoCo example wedge.
- New CLI surface area unless the implementation proves a small flag is required.

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
- reads the manifest-selected primary views
- resolves current and baseline image pairs for those views
- degrades cleanly when one side is missing

The helper should return a boring explicit structure the summary HTML can render.
No new shared renderer abstraction unless a tiny hook in `src/roboharness/reporting.py`
is strictly required.

### 3. Alarm-First HTML Upgrade

Extend `build_summary_html()` so the summary block contains:

- the existing alarm cards
- the existing agent next action panel
- the existing phase timeline
- a new **Current vs Baseline** section for the first failing phase

That section should:

- highlight the failed phase and view name
- show current and baseline images side-by-side
- include the metric regression text that explains why the view was chosen
- fall back to text-only guidance when the baseline or current image is unavailable

The report must stay portable over CI, GitHub Pages, and SSH. The summary evidence
should therefore embed the baseline and current comparison images directly into the
HTML block as data URIs, or achieve the same self-contained outcome, instead of
depending on repo-relative asset paths.

The report should still read top-down in the same order:
alarms -> next action -> phase timeline -> current-vs-baseline evidence.

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

Extend `tests/test_mujoco_grasp_wedge.py` with focused assertions for:

1. deterministic known-bad fixture reports `Verdict.FAIL`
2. the first failing phase remains `approach`
3. the chosen views remain `["side", "top"]`
4. the rerun hint remains `restore:pre_grasp`
5. `build_summary_html()` renders a `Current vs Baseline` section when evidence
   pairs exist
6. the rendered HTML references both current and baseline evidence paths for the
   failing phase
7. missing baseline or current images degrade cleanly without blowing up report
   generation
8. fixture coverage stays aligned with the primary-view policy for the supported
   failing phase(s)

## Files Expected To Change

- `docs/designs/mujoco-alarmed-grasp-loop-phase-2-plan.md`
- `examples/_mujoco_grasp_wedge.py`
- `examples/mujoco_grasp.py`
- `assets/example_mujoco_grasp/`
- `tests/test_mujoco_grasp_wedge.py`
- `src/roboharness/reporting.py` only if a tiny renderer hook is truly necessary

## Success Criteria

- A deterministic known-bad fixture makes the wedge fail in `approach`, not just
  somewhere later in the pipeline.
- The HTML report shows current-vs-baseline evidence for the first failing phase
  without requiring manual file browsing.
- The primary views in the HTML match the same evaluator-backed manifest already
  used by the machine-readable artifact path.
- The implementation stays example-local and does not expand into a cross-stack
  abstraction pass.
- Tests cover both the happy path and the missing-evidence fallback.

## Failure Modes To Guard Against

- The HTML shows baseline images that do not correspond to the same phase/view as
  the current image.
- The summary path invents its own failure classification instead of consuming the
  existing evaluator/manifest result.
- The new fixture pack becomes sparse or inconsistent with `MUJOCO_GRASP_PRIMARY_VIEWS`,
  causing the report to suggest views that do not exist.
- The implementation quietly adds a second baseline contract in parallel with
  `autonomous_report.json`.

## Open Questions

1. Is a tiny report-renderer CSS hook needed to keep the current-vs-baseline block
   readable, or can `summary_html` own the whole thing cleanly?

## Initial Recommendation

Take the smallest explicit path:

- check in baseline and known-bad image fixtures
- keep the evidence-pair resolver in `examples/_mujoco_grasp_wedge.py`
- keep the report upgrade inside `build_summary_html()`
- keep the report self-contained by embedding the comparison images in the HTML
- avoid new CLI flags unless the implementation proves one is necessary

That is enough to make phase 2 real without turning it into a platform project.
