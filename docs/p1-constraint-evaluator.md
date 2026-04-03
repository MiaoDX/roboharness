# P1: Constraint Evaluator — Design Specification

> **Status**: Draft — awaiting review  
> **Author**: WLB  
> **Date**: 2026-04-03  
> **Target**: v0.3.0  

## Background

This spec comes from a 2026-04-03 discussion about applying Harness Engineering principles to roboharness.

The referenced Harness Engineering article identifies three scaling dimensions:

### Time Scaling
- **Problem**: Running an agent more times in simulation is only useful if it can "see" how well each run went — and correct itself
- **P0** (PR #47, report module): gives *humans* a 3-second glance at trial results (fail-first reporting)
- **P1** (this spec): gives *agents* a deterministic verdict — no human review needed
- **Key insight**: scaling without feedback loops = blind iteration. Hard constraints turn raw metrics into actionable outcomes

### Space Scaling
- Running across multiple environments, scenarios, and cases in parallel
- Requires structured per-case pass/fail so results can be aggregated
- P1's evaluation result provides the foundation for this aggregation

### Interaction Scaling
- Collaboration between agents: coding agent writes → harness agent tests → results feed back
- P0 + P1 form the minimal interaction loop: the harness becomes the bridge between "code changed" and "behavior verified"
- CI + PR comment = the shortest feedback loop

### What we are borrowing from the article

1. **A harness is not just a test runner** — it is the core of an evaluation feedback loop
2. **Deterministic assertions are the prerequisite for time scaling** — without them you cannot tell if more runs are producing better or worse outcomes
3. **Machine-readable + human-readable, two-track** — PR comments for people (P0), structured verdicts for CI/agents (P1)
4. **Configurable constraints = reusability** — different grasp tasks have different thresholds; they should not be hardcoded

---

## What to Build

### 1. Metric Assertion

The atomic unit: a single check against one metric value.

**Required fields:**
- **Metric name** — the key to look up in the report (e.g., `grip_center_error_mm`)
- **Operator** — comparison type: less-than, less-or-equal, equal, greater-than, in-range
- **Threshold** — the value (or value pair for ranges) to compare against
- **Severity** — `critical` / `major` / `minor` / `info`
- **Phase scope** — which simulation phase to check; `*` means all phases

### 2. Assertion Engine

The engine takes a list of assertions and a harness report JSON, and returns a structured evaluation result.

**Process:**
1. Load constraint definitions from a configuration file (YAML or JSON) or from Python constants
2. Extract metric values from the `autonomous_report.json` (same schema the P0 report module already reads)
3. Evaluate each assertion deterministically — same input always produces same output
4. Aggregate results with severity-aware verdict logic

**Verdict logic:**
- `pass` — all critical and major assertions pass
- `degraded` — some major assertions fail, but no critical failures
- `fail` — at least one critical assertion fails

The verdict should be machine-readable (a structured data object, not just text).

### 3. CLI Subcommand

A new `roboharness evaluate` command for standalone use.

**Usage:**
- Evaluate a report against default constraints: `roboharness evaluate <report-path>`
- Evaluate against a custom constraint file: `roboharness evaluate <report-path> --constraints <file>`
- Machine-readable output for CI: `roboharness evaluate <report-path> --format json`

**Exit codes:**
- `0` — pass
- `1` — critical failure
- `2` — degraded (major failures, no critical)

### 4. Constraint Definition Format

A human-readable configuration file (YAML recommended) where users specify:

- A list of constraint definitions (metric, operator, threshold, severity, phase scope)
- Global verdict rules (which severities cause `fail` vs `degraded` vs `pass`)

Example use case: one YAML file per task type (grasp, push, place, etc.) with different thresholds.

### 5. CI Integration (optional, future)

A CI job that runs `roboharness evaluate` against any harness report artifacts found in a PR. This closes the loop: code change → harness run → constraint check → PR gets automatic pass/fail verdict.

---

## Suggested File Structure

```
src/roboharness/
  evaluate/
    __init__.py          # package init
    __main__.py          # CLI entry point
    assertions.py        # MetricAssertion + AssertionEngine
    constraints.py       # Load/save constraint definitions
    result.py            # EvaluationResult and related types
    defaults.py          # Default constraint presets
constraints/
  grasp_default.yaml     # Example constraint file for grasp tasks
tests/
  test_assertions.py     # Unit tests for assertion engine
  test_cli.py            # CLI integration tests
```

**Design note**: The evaluator should be a sibling module to the existing `report/` package, keeping descriptive (what happened) and prescriptive (was it good enough) concerns separate. They can share data types if the coupling grows later.

---

## Acceptance Criteria

When an implementer considers this done, all of the following should be true:

- [ ] Assertion engine produces deterministic results — same input JSON, same verdict, every time
- [ ] CLI `roboharness evaluate` works on real harness report files
- [ ] Constraint definitions load from YAML or JSON
- [ ] Exit codes work correctly for CI integration (0/1/2)
- [ ] Unit tests cover the assertion engine
- [ ] Integration tests use real harness report data (not fabricated fixtures)
- [ ] No changes to the existing `report/` module behavior — backward compatible
- [ ] Documentation usage guide with examples

---

## Open Questions for the Implementer

1. **Phase-level vs trial-level assertions**: Some constraints apply per-phase (e.g., "pinch gap must close during pregrasp phase"), others apply globally (e.g., "total trial duration under 30s"). The spec supports both via a phase scope field — the implementer should decide the cleanest way to implement this.

2. **Default threshold values**: The numbers in the examples (grip center error < 10mm, etc.) are estimates. They should be validated against real harness benchmarks before being shipped as defaults.

3. **Error handling**: What happens when a metric referenced by an assertion is missing from the report? Options: treat as failure, skip with warning, or configurable. The implementer should pick one and document it.

4. **Performance**: The evaluator should be fast enough to run on every PR. For large reports with many phases, avoid unnecessary data copies or repeated parsing.

---

## What We Are NOT Building (Yet)

These are explicitly out of scope for P1:

- **Training or ML inference** — the evaluator is purely deterministic rule-checking
- **Real-time simulation feedback** — this runs post-hoc on completed report JSON, not during simulation (that would be P2+)
- **Changes to the existing JSON report schema** — the evaluator reads the same report format P0 already uses
- **A complex failure category taxonomy** — categories like GEOMETRY, CONTACT, LIFT can be deferred to v1.0+ if needed

---

## Related

- [PR #47](https://github.com/MiaoDX/roboharness/pull/47) — P0: fail-first report module
- `docs/roadmap.md` — Milestone A2: task-level summary metrics (P1 refines this milestone)
- Harness Engineering article discussion, 2026-04-03
