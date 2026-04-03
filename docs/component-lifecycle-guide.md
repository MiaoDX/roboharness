# When to Remove a Harness Component

Every harness component is a hypothesis about the current model's capability boundary. As models improve, these hypotheses expire at different rates. This guide defines the process for deciding when to retire a component.

## Core idea

Roboharness components exist because today's models have specific limitations — limited single-view 3D reasoning, noisy monocular depth, inability to diagnose failures without intermediate state, and self-rationalisation bias when evaluating their own output. Each of these limitations is temporary. The harness should get leaner over time, not accumulate indefinitely.

## Lifecycle metadata

Each component is annotated with structured lifecycle metadata (see `src/roboharness/core/lifecycle.py`):

| Field | Purpose |
|-------|---------|
| `component_name` | Unique identifier |
| `version_introduced` | When it was added |
| `assumptions` | The capability-gap hypotheses it addresses |
| `horizon` | Expected timeframe before assumptions expire |

Each assumption records:
- **description** — what limitation this assumes exists
- **removal_condition** — the observable, testable condition under which the assumption no longer holds
- **evidence** — notes from past experiments

## Current component assumptions

| Component | Assumed Limitation | Horizon | Removal Condition |
|-----------|-------------------|---------|-------------------|
| Multi-view capture | Single-view 3D inference is unreliable | Medium-term | >95% grasp success from single RGB view |
| Depth capture | RGB-only depth estimation has gaps | Near-term | <1cm error from RGB at manipulation distances |
| Intermediate checkpoints | Limited failure diagnosis from final state | Long-term | Reliable root-cause identification from final state only |
| Constraint evaluator | Self-rationalisation bias | Very long-term | >90% concordance between self-eval and independent eval |

## The "harness diet" review process

Run this process when a new model generation is released or at quarterly intervals:

### 1. Audit

```python
from roboharness import default_registry

for report in default_registry.audit():
    print(f"{report['component']} ({report['horizon']}): expired={report['expired']}")
```

### 2. Test removal conditions

For each component, design a targeted experiment:

1. **Baseline**: run the standard evaluation suite with the component enabled.
2. **Ablation**: disable or bypass the component.
3. **Compare**: measure task success rate, failure diagnosis quality, or evaluation concordance (depending on the component).

Record results in the assumption's `evidence` field.

### 3. Decide

```python
from roboharness import default_registry

# Pass in experimental evidence
evidence = {
    "RGB-only depth estimation has gaps for close-range manipulation": True,
    # ... other assumptions tested
}
reports = default_registry.audit(evidence=evidence)
for r in reports:
    if r["expired"]:
        print(f"CANDIDATE FOR REMOVAL: {r['component']}")
```

A component is a candidate for removal when **all** of its assumptions are disproven by experimental evidence. Even then, removal should go through a deprecation cycle:

1. Mark as deprecated in the next minor release.
2. Log a warning when the component is used.
3. Remove in the following minor release.

### 4. Record

After each review, update:
- The assumption's `evidence` field with experiment results and date.
- This guide's table if horizons or conditions change.
- The changelog with any deprecations or removals.

## Adding new components

When adding a new harness component, register its lifecycle metadata:

```python
from roboharness import (
    ComponentAssumption,
    ComponentLifecycle,
    ExpirationHorizon,
    default_registry,
)

default_registry.register(
    ComponentLifecycle(
        component_name="my_new_component",
        version_introduced="0.4.0",
        assumptions=[
            ComponentAssumption(
                description="Models cannot do X reliably",
                removal_condition="Model achieves Y on benchmark Z",
            ),
        ],
        horizon=ExpirationHorizon.MEDIUM_TERM,
    )
)
```

If you cannot articulate the assumption and removal condition, reconsider whether the component is necessary.

## Historical removals

_No components have been removed yet. This section will track removal decisions and their outcomes._

| Component | Removed in | Reason | Impact |
|-----------|-----------|--------|--------|
| — | — | — | — |
