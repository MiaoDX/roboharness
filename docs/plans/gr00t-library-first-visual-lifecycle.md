# GR00T Library-First Visual Lifecycle

Status: proposed

## Context

GR00T whole-body control now dogfoods RoboHarness at the evidence/artifact
layer. GR00T imports RoboHarness, normalizes semantic snapshot evidence through
`roboharness.evidence`, and emits proof packs, visual review manifests, suite
proof packs, review queues, paired manifests, and review summaries.

That is useful, but it does not yet reduce downstream visual-harness code much.
GR00T still owns duplicate case report dataclasses, suite report dataclasses,
output layout glue, suite aggregation, and artifact write orchestration around
G1 decoupled WBC, G1 SONIC/native, and Realman visual harnesses.

The next step is not a RoboHarness-owned command such as
`roboharness run gr00t:g1`. GR00T should keep its own operator commands and
import RoboHarness as a Python library. RoboHarness should own more reusable
visual lifecycle and suite mechanics while downstream projects keep robot
runtime and domain semantics.

## Direction

Build a library-first visual harness API that downstream repos can embed.

GR00T remains responsible for:

- G1, G1 SONIC/native, and Realman command entrypoints.
- ROS, deploy-live, hardware-live, safety startup, and environment sourcing.
- Robot-specific planner/runtime execution.
- Robot-specific case registries and profile/backend selection.
- Semantic snapshot production and renderer execution.
- Robot-specific metrics, verdict reasons, and failure taxonomy.

RoboHarness should own:

- Common visual case lifecycle objects.
- Common visual suite lifecycle objects.
- Output directory and artifact naming conventions.
- Case and suite JSON schema normalization.
- Proof-pack, visual-review manifest, suite-proof-pack, and review-queue writes.
- Paired regression manifest and visual-review summary helpers.
- Resume/missing-case indexing once the basic lifecycle is stable.

## L3: Embedded Visual Lifecycle API

L3 introduces thin visual lifecycle objects that GR00T can use inside existing
runner entrypoints. These objects should not require downstream projects to
pretend deploy-live or hardware-live flows are `SimulatorBackend.step()` loops.

Candidate shape:

```python
from roboharness.visual import VisualCaseRun, VisualSuiteRun

suite = VisualSuiteRun(
    suite_name="representative",
    output_root=output_root,
    metadata={...},
)

case_run = VisualCaseRun(
    case_id=case.case_id,
    robot_type="g1",
    output_dir=case_output_dir,
    metadata={...},
)

# Downstream-owned runtime, planner, capture, and metric code.
case_run.set_snapshot_bundle(snapshot_bundle)
case_run.add_renderer_report("meshcat", meshcat_report)
case_run.add_renderer_report("mujoco", mujoco_report)
case_run.set_metrics(summary_metrics)
case_run.set_verdict(verdict, reasons=verdict_reasons, taxonomy=failure_taxonomy)
case_run.write_artifacts()

suite.add_case(case_run)
suite.write_artifacts()
```

L3 acceptance gates:

- GR00T can generate the same case and suite artifacts for G1 decoupled WBC,
  G1 SONIC/native static evidence, and Realman through the new library API.
- GR00T keeps existing operator commands and artifact filenames compatible.
- GR00T deletes or thins at least one duplicated generic report/suite glue path
  after migration.
- RoboHarness core imports remain free of optional simulator dependencies.
- Existing proof-pack, paired-review, and contract-check behavior remains
  compatible.

## L4: Embedded Suite Executor API

L4 is still library-first. GR00T keeps its command surface, but delegates more
suite orchestration to RoboHarness.

Candidate shape:

```python
from roboharness.visual import run_visual_suite

run_visual_suite(
    suite_spec=suite_spec,
    case_runner=run_one_gr00t_case,
    output_root=output_root,
    options=VisualSuiteOptions(...),
)
```

The downstream `case_runner` owns robot execution and returns a visual case
result or raises a structured execution error. RoboHarness owns the case loop,
output layout, status accounting, suite aggregation, artifact writes, and later
resume/missing-case behavior.

L4 acceptance gates:

- GR00T G1 and Realman suite entrypoints call the embedded suite executor
  without changing user-facing commands.
- G1 SONIC/native deploy-live remains a downstream-owned runtime surface; the
  suite executor records it but does not launch or certify live hardware safety.
- Suite reports do not hide execution errors, skipped cases, or missing
  artifacts.
- The migrated GR00T code removes more generic loop/report code than it adds in
  adapter code.

## Non-Goals

- Do not make `roboharness run gr00t:*` the primary integration path.
- Do not move GR00T ROS, SONIC, Realman, deploy-live, or hardware-live startup
  ownership into RoboHarness.
- Do not force GR00T visual harnesses into `SimulatorBackend`.
- Do not turn `HarnessContract` into a runtime or renderer configuration
  language.
- Do not alter baseline blessing or human escalation policy.

## First Slice

1. Add `roboharness.visual` with typed embedded case/suite lifecycle objects.
2. Use fixtures to prove the API can reproduce the current GR00T-style
   autonomous report, proof pack, manifest, suite proof pack, and review queue.
3. Keep the API independent of optional MuJoCo, Meshcat, Rerun, ROS, and GR00T
   imports.
4. Migrate one GR00T case-level path to the embedded lifecycle API.
5. Migrate one GR00T suite-level path and verify the current dogfood surfaces:
   G1 decoupled WBC, G1 SONIC/native static evidence, and Realman.

## Open Implementation Defaults

- Use conservative dataclasses or typed dictionaries first; avoid a framework
  hierarchy until duplicated behavior is visible.
- Preserve unknown downstream fields round-trip, as `roboharness.evidence`
  already does.
- Keep CLI additions limited to generic artifact inspection/checking unless a
  later plan explicitly scopes a command surface.
- Prefer additive API introduction, then downstream migration, then cleanup of
  deprecated bridge code.
