"""Tests for component lifecycle metadata."""

from __future__ import annotations

import pytest

from roboharness.core.lifecycle import (
    ComponentAssumption,
    ComponentLifecycle,
    ExpirationHorizon,
    LifecycleRegistry,
    default_registry,
)

# ---------------------------------------------------------------------------
# ComponentAssumption
# ---------------------------------------------------------------------------


def test_assumption_fields():
    a = ComponentAssumption(
        description="Models can't do X",
        removal_condition="Model does X reliably",
        evidence="Tested on v2, still fails",
    )
    assert a.description == "Models can't do X"
    assert a.removal_condition == "Model does X reliably"
    assert a.evidence == "Tested on v2, still fails"


def test_assumption_default_evidence():
    a = ComponentAssumption(description="d", removal_condition="r")
    assert a.evidence == ""


def test_assumption_is_frozen():
    a = ComponentAssumption(description="d", removal_condition="r")
    with pytest.raises(AttributeError):
        a.description = "changed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ComponentLifecycle
# ---------------------------------------------------------------------------


def test_lifecycle_defaults():
    lc = ComponentLifecycle(component_name="test", version_introduced="0.1.0")
    assert lc.assumptions == []
    assert lc.horizon == ExpirationHorizon.LONG_TERM
    assert lc.metadata == {}


def test_lifecycle_is_expired_no_assumptions():
    lc = ComponentLifecycle(component_name="test", version_introduced="0.1.0")
    assert lc.is_expired() is False
    assert lc.is_expired({"anything": True}) is False


def test_lifecycle_is_expired_no_evidence():
    lc = ComponentLifecycle(
        component_name="test",
        version_introduced="0.1.0",
        assumptions=[ComponentAssumption(description="A", removal_condition="R")],
    )
    assert lc.is_expired() is False
    assert lc.is_expired({}) is False


def test_lifecycle_is_expired_partial_evidence():
    lc = ComponentLifecycle(
        component_name="test",
        version_introduced="0.1.0",
        assumptions=[
            ComponentAssumption(description="A1", removal_condition="R1"),
            ComponentAssumption(description="A2", removal_condition="R2"),
        ],
    )
    assert lc.is_expired({"A1": True}) is False


def test_lifecycle_is_expired_all_disproven():
    lc = ComponentLifecycle(
        component_name="test",
        version_introduced="0.1.0",
        assumptions=[
            ComponentAssumption(description="A1", removal_condition="R1"),
            ComponentAssumption(description="A2", removal_condition="R2"),
        ],
    )
    assert lc.is_expired({"A1": True, "A2": True}) is True


def test_lifecycle_summary():
    lc = ComponentLifecycle(
        component_name="depth",
        version_introduced="0.1.0",
        assumptions=[
            ComponentAssumption(
                description="RGB depth is bad",
                removal_condition="RGB depth is good",
                evidence="tested v3",
            ),
        ],
        horizon=ExpirationHorizon.NEAR_TERM,
        metadata={"owner": "core-team"},
    )
    s = lc.summary()
    assert s["component"] == "depth"
    assert s["version_introduced"] == "0.1.0"
    assert s["horizon"] == "near_term"
    assert len(s["assumptions"]) == 1
    assert s["assumptions"][0]["evidence"] == "tested v3"
    assert s["metadata"]["owner"] == "core-team"


# ---------------------------------------------------------------------------
# LifecycleRegistry
# ---------------------------------------------------------------------------


def test_registry_register_and_get():
    reg = LifecycleRegistry()
    lc = ComponentLifecycle(component_name="c1", version_introduced="0.1.0")
    reg.register(lc)
    assert reg.get("c1") is lc
    assert reg.get("missing") is None


def test_registry_list_components():
    reg = LifecycleRegistry()
    reg.register(ComponentLifecycle(component_name="a", version_introduced="0.1.0"))
    reg.register(ComponentLifecycle(component_name="b", version_introduced="0.1.0"))
    assert sorted(reg.list_components()) == ["a", "b"]


def test_registry_contains_and_len():
    reg = LifecycleRegistry()
    assert len(reg) == 0
    assert "x" not in reg
    reg.register(ComponentLifecycle(component_name="x", version_introduced="0.1.0"))
    assert len(reg) == 1
    assert "x" in reg


def test_registry_by_horizon():
    reg = LifecycleRegistry()
    reg.register(
        ComponentLifecycle(
            component_name="a",
            version_introduced="0.1.0",
            horizon=ExpirationHorizon.NEAR_TERM,
        )
    )
    reg.register(
        ComponentLifecycle(
            component_name="b",
            version_introduced="0.1.0",
            horizon=ExpirationHorizon.LONG_TERM,
        )
    )
    near = reg.by_horizon(ExpirationHorizon.NEAR_TERM)
    assert len(near) == 1
    assert near[0].component_name == "a"


def test_registry_audit():
    reg = LifecycleRegistry()
    reg.register(
        ComponentLifecycle(
            component_name="c",
            version_introduced="0.1.0",
            assumptions=[ComponentAssumption(description="A", removal_condition="R")],
        )
    )
    reports = reg.audit()
    assert len(reports) == 1
    assert reports[0]["expired"] is False

    reports = reg.audit(evidence={"A": True})
    assert reports[0]["expired"] is True


def test_registry_audit_empty():
    reg = LifecycleRegistry()
    assert reg.audit() == []


# ---------------------------------------------------------------------------
# Default registry (built-in component registrations)
# ---------------------------------------------------------------------------


def test_default_registry_has_four_components():
    assert len(default_registry) == 4
    expected = {
        "multi_view_capture",
        "depth_capture",
        "intermediate_checkpoints",
        "constraint_evaluator",
    }
    assert set(default_registry.list_components()) == expected


def test_default_registry_horizons():
    assert (
        default_registry.get("depth_capture").horizon  # type: ignore[union-attr]
        == ExpirationHorizon.NEAR_TERM
    )
    assert (
        default_registry.get("multi_view_capture").horizon  # type: ignore[union-attr]
        == ExpirationHorizon.MEDIUM_TERM
    )
    assert (
        default_registry.get("intermediate_checkpoints").horizon  # type: ignore[union-attr]
        == ExpirationHorizon.LONG_TERM
    )
    assert (
        default_registry.get("constraint_evaluator").horizon  # type: ignore[union-attr]
        == ExpirationHorizon.VERY_LONG_TERM
    )


def test_default_registry_none_expired_without_evidence():
    reports = default_registry.audit()
    assert all(r["expired"] is False for r in reports)


def test_default_registry_each_has_assumptions():
    for name in default_registry.list_components():
        lc = default_registry.get(name)
        assert lc is not None
        assert len(lc.assumptions) >= 1
        for a in lc.assumptions:
            assert a.description
            assert a.removal_condition


def test_public_api_exports():
    """Lifecycle types are importable from the top-level package."""
    from roboharness import ComponentAssumption as CA
    from roboharness import ComponentLifecycle as CL
    from roboharness import ExpirationHorizon as EH
    from roboharness import LifecycleRegistry as LR
    from roboharness import default_registry as dr

    assert CA is ComponentAssumption
    assert CL is ComponentLifecycle
    assert EH is ExpirationHorizon
    assert LR is LifecycleRegistry
    assert dr is default_registry
