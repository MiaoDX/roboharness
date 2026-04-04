"""Component lifecycle metadata: track harness assumptions and expiration.

Every harness component encodes a hypothesis about the current model's
capability boundary.  As models improve, components should be re-evaluated
and potentially retired.  This module provides lightweight metadata to make
those assumptions explicit and reviewable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ExpirationHorizon(Enum):
    """Expected timeframe before a component's underlying assumption expires."""

    NEAR_TERM = "near_term"  # likely obsolete within 1 model generation
    MEDIUM_TERM = "medium_term"  # 2-3 model generations
    LONG_TERM = "long_term"  # 4+ model generations
    VERY_LONG_TERM = "very_long_term"  # fundamental limitation, unlikely soon


@dataclass(frozen=True)
class ComponentAssumption:
    """A single testable assumption that justifies a component's existence.

    Attributes:
        description: Human-readable statement of what limitation this assumes.
        removal_condition: Observable condition under which this assumption
            no longer holds and the component can be removed.
        evidence: Optional notes on experiments or observations supporting
            the assumption (or its expiration).
    """

    description: str
    removal_condition: str
    evidence: str = ""


@dataclass
class ComponentLifecycle:
    """Lifecycle metadata for a harness component.

    Attach this to any component that encodes an implicit model-capability
    assumption.  The metadata makes it possible to conduct periodic
    "harness diet" reviews when new models are released.

    Attributes:
        component_name: Unique identifier (e.g. ``"multi_view_capture"``).
        version_introduced: Version string when the component was added.
        assumptions: The capability-gap hypotheses this component addresses.
        horizon: Expected timeframe before the assumptions expire.
        metadata: Arbitrary extra annotations.
    """

    component_name: str
    version_introduced: str
    assumptions: list[ComponentAssumption] = field(default_factory=list)
    horizon: ExpirationHorizon = ExpirationHorizon.LONG_TERM
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_expired(self, evidence: dict[str, bool] | None = None) -> bool:
        """Check whether all assumptions have been invalidated.

        Args:
            evidence: Mapping of assumption descriptions to ``True`` if the
                assumption has been empirically disproven (i.e. removal
                condition met).  Assumptions not present in the dict are
                treated as still valid.

        Returns:
            ``True`` when *every* assumption is disproven, meaning the
            component is a candidate for removal.
        """
        if not self.assumptions or evidence is None:
            return False
        return all(evidence.get(a.description, False) for a in self.assumptions)

    def summary(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary for reports and metadata files."""
        result: dict[str, Any] = {
            "component": self.component_name,
            "version_introduced": self.version_introduced,
            "horizon": self.horizon.value,
            "assumptions": [
                {
                    "description": a.description,
                    "removal_condition": a.removal_condition,
                    "evidence": a.evidence,
                }
                for a in self.assumptions
            ],
        }
        if self.metadata:
            result["metadata"] = self.metadata
        return result


class LifecycleRegistry:
    """Central registry of component lifecycle metadata.

    Typical usage::

        registry = LifecycleRegistry()
        registry.register(ComponentLifecycle(
            component_name="depth_capture",
            version_introduced="0.1.0",
            assumptions=[ComponentAssumption(...)],
            horizon=ExpirationHorizon.NEAR_TERM,
        ))

        # During a "harness diet" review:
        for report in registry.audit():
            print(report)
    """

    def __init__(self) -> None:
        self._components: dict[str, ComponentLifecycle] = {}

    def register(self, lifecycle: ComponentLifecycle) -> None:
        """Register lifecycle metadata for a component."""
        self._components[lifecycle.component_name] = lifecycle

    def get(self, component_name: str) -> ComponentLifecycle | None:
        """Look up lifecycle metadata by component name."""
        return self._components.get(component_name)

    def list_components(self) -> list[str]:
        """Return all registered component names."""
        return list(self._components.keys())

    def audit(self, evidence: dict[str, bool] | None = None) -> list[dict[str, Any]]:
        """Produce an audit report for all registered components.

        Args:
            evidence: Optional mapping of assumption descriptions to booleans.
                See :meth:`ComponentLifecycle.is_expired`.

        Returns:
            List of summary dicts, each augmented with an ``"expired"`` flag.
        """
        reports: list[dict[str, Any]] = []
        for lifecycle in self._components.values():
            report = lifecycle.summary()
            report["expired"] = lifecycle.is_expired(evidence)
            reports.append(report)
        return reports

    def by_horizon(self, horizon: ExpirationHorizon) -> list[ComponentLifecycle]:
        """Return components matching a given expiration horizon."""
        return [c for c in self._components.values() if c.horizon == horizon]

    def __len__(self) -> int:
        return len(self._components)

    def __contains__(self, component_name: str) -> bool:
        return component_name in self._components


# ---------------------------------------------------------------------------
# Default registry with roboharness built-in component assumptions
# ---------------------------------------------------------------------------

default_registry: LifecycleRegistry = LifecycleRegistry()
"""Pre-populated registry with lifecycle metadata for all core roboharness
components.  Import and query this directly for audits and reports."""

default_registry.register(
    ComponentLifecycle(
        component_name="multi_view_capture",
        version_introduced="0.1.0",
        assumptions=[
            ComponentAssumption(
                description="Single-view 3D inference is unreliable for manipulation tasks",
                removal_condition=(
                    "Model achieves >95% grasp success from a single RGB view "
                    "without auxiliary camera angles"
                ),
            ),
        ],
        horizon=ExpirationHorizon.MEDIUM_TERM,
    )
)

default_registry.register(
    ComponentLifecycle(
        component_name="depth_capture",
        version_introduced="0.1.0",
        assumptions=[
            ComponentAssumption(
                description="RGB-only depth estimation has gaps for close-range manipulation",
                removal_condition=(
                    "Model infers metric depth from RGB alone with <1cm error "
                    "at manipulation distances (<1m)"
                ),
            ),
        ],
        horizon=ExpirationHorizon.NEAR_TERM,
    )
)

default_registry.register(
    ComponentLifecycle(
        component_name="intermediate_checkpoints",
        version_introduced="0.1.0",
        assumptions=[
            ComponentAssumption(
                description=(
                    "Models have limited ability to diagnose failures from final state alone"
                ),
                removal_condition=(
                    "Model reliably identifies failure root cause and recovery "
                    "strategy from final-state observation only"
                ),
            ),
        ],
        horizon=ExpirationHorizon.LONG_TERM,
    )
)

default_registry.register(
    ComponentLifecycle(
        component_name="constraint_evaluator",
        version_introduced="0.3.0",
        assumptions=[
            ComponentAssumption(
                description=(
                    "Models exhibit self-rationalisation bias when evaluating own outputs"
                ),
                removal_condition=(
                    "Model self-evaluation matches independent evaluator agreement "
                    "rate (>90% concordance on pass/fail)"
                ),
            ),
        ],
        horizon=ExpirationHorizon.VERY_LONG_TERM,
    )
)
