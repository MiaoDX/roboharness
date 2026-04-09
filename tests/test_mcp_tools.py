"""Tests for roboharness.mcp.tools — MCP tool implementations."""

from __future__ import annotations

import pytest

from roboharness.core.harness import Harness
from roboharness.evaluate.constraints import _parse_assertion
from roboharness.mcp.tools import TOOL_SCHEMAS, HarnessTools
from roboharness.storage.history import EvaluationRecord

from .conftest import MockBackend

# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture()
def harness(tmp_path):
    h = Harness(MockBackend(), output_dir=tmp_path, task_name="test_task")
    h.reset()
    return h


@pytest.fixture()
def tools(harness, tmp_path):
    return HarnessTools(harness, history_dir=tmp_path / "history")


# ── TOOL_SCHEMAS ─────────────────────────────────────────────────────────


def test_tool_schemas_are_valid():
    assert len(TOOL_SCHEMAS) == 3
    names = {s["name"] for s in TOOL_SCHEMAS}
    assert names == {"capture_checkpoint", "evaluate_constraints", "compare_baselines"}
    for schema in TOOL_SCHEMAS:
        assert "description" in schema
        assert "inputSchema" in schema


# ── capture_checkpoint ───────────────────────────────────────────────────


def test_capture_checkpoint_default_cameras(tools):
    result = tools.capture_checkpoint()
    assert result["checkpoint_name"].startswith("step_")
    assert result["step"] == 0
    assert result["sim_time"] == 0.0
    assert len(result["views"]) == 1
    assert result["views"][0]["name"] == "front"
    assert result["views"][0]["rgb_shape"] == [64, 64, 3]


def test_capture_checkpoint_custom_name_and_cameras(tools, harness):
    # Advance a few steps first
    for _ in range(5):
        harness.step(None)

    result = tools.capture_checkpoint(
        checkpoint_name="mid_grasp",
        cameras=["front", "side"],
    )
    assert result["checkpoint_name"] == "mid_grasp"
    assert result["step"] == 5
    assert result["sim_time"] == pytest.approx(0.05)
    assert len(result["views"]) == 2
    assert result["views"][0]["name"] == "front"
    assert result["views"][1]["name"] == "side"


def test_capture_checkpoint_state_is_serialisable(tools):
    """State dict values must be plain Python types, not numpy arrays."""
    result = tools.capture_checkpoint()
    for v in result["state"].values():
        assert not hasattr(v, "dtype"), f"numpy value leaked into state: {v!r}"


# ── evaluate_constraints ─────────────────────────────────────────────────


def test_evaluate_constraints_pass(tools):
    report = {"summary_metrics": {"grip_error_mm": 3.5}}
    assertions = [
        {"metric": "grip_error_mm", "operator": "lt", "threshold": 10.0},
    ]
    result = tools.evaluate_constraints(report, assertions)
    assert result["verdict"] == "pass"
    assert result["passed"] == 1
    assert result["failed"] == 0


def test_evaluate_constraints_fail(tools):
    report = {"summary_metrics": {"grip_error_mm": 15.0}}
    assertions = [
        {
            "metric": "grip_error_mm",
            "operator": "lt",
            "threshold": 10.0,
            "severity": "critical",
        },
    ]
    result = tools.evaluate_constraints(report, assertions)
    assert result["verdict"] == "fail"
    assert result["critical_failures"] == 1


def test_evaluate_constraints_degraded(tools):
    report = {"summary_metrics": {"grip_error_mm": 15.0}}
    assertions = [
        {"metric": "grip_error_mm", "operator": "lt", "threshold": 10.0, "severity": "major"},
    ]
    result = tools.evaluate_constraints(report, assertions)
    assert result["verdict"] == "degraded"
    assert result["major_failures"] == 1


def test_evaluate_constraints_missing_metric(tools):
    report = {"summary_metrics": {}}
    assertions = [
        {"metric": "nonexistent", "operator": "gt", "threshold": 0.0},
    ]
    result = tools.evaluate_constraints(report, assertions)
    assert result["failed"] == 1


def test_evaluate_constraints_phase_specific(tools):
    report = {
        "snapshot_metrics": {
            "lift": {"height_m": 0.25},
        },
    }
    assertions = [
        {"metric": "height_m", "operator": "ge", "threshold": 0.2, "phase": "lift"},
    ]
    result = tools.evaluate_constraints(report, assertions)
    assert result["verdict"] == "pass"


def test_evaluate_constraints_in_range(tools):
    report = {"summary_metrics": {"force_n": 5.0}}
    assertions = [
        {"metric": "force_n", "operator": "in_range", "threshold": [2.0, 8.0]},
    ]
    result = tools.evaluate_constraints(report, assertions)
    assert result["verdict"] == "pass"


# ── compare_baselines ────────────────────────────────────────────────────


def test_compare_baselines_no_history(tools):
    result = tools.compare_baselines(task="grasp", current_rate=0.8)
    assert result["regressed"] is False
    assert result["previous_rate"] is None
    assert result["window_size"] == 0


def test_compare_baselines_stable(tools):
    # Seed history with consistent runs
    for _ in range(5):
        tools._history.append(
            EvaluationRecord(task="grasp", success_rate=0.8, total_trials=10, successes=8)
        )

    result = tools.compare_baselines(task="grasp", current_rate=0.8)
    assert result["regressed"] is False
    assert result["previous_rate"] == pytest.approx(0.8)
    assert result["delta"] == pytest.approx(0.0)


def test_compare_baselines_regression(tools):
    for _ in range(5):
        tools._history.append(
            EvaluationRecord(task="grasp", success_rate=0.9, total_trials=10, successes=9)
        )

    result = tools.compare_baselines(task="grasp", current_rate=0.5)
    assert result["regressed"] is True
    assert result["delta"] == pytest.approx(-0.4)


def test_compare_baselines_improvement(tools):
    for _ in range(5):
        tools._history.append(
            EvaluationRecord(task="grasp", success_rate=0.5, total_trials=10, successes=5)
        )

    result = tools.compare_baselines(task="grasp", current_rate=0.9)
    assert result["regressed"] is False
    assert result["delta"] == pytest.approx(0.4)


# ── _parse_assertion helper ──────────────────────────────────────────────


def test_parse_assertion_minimal():
    raw = {"metric": "x", "operator": "gt", "threshold": 1.0}
    a = _parse_assertion(raw)
    assert a.metric == "x"
    assert a.operator.value == "gt"
    assert a.threshold == 1.0
    assert a.severity.value == "major"
    assert a.phase == "*"


def test_parse_assertion_full():
    raw = {
        "metric": "y",
        "operator": "in_range",
        "threshold": [0.0, 1.0],
        "severity": "critical",
        "phase": "lift",
    }
    a = _parse_assertion(raw)
    assert a.severity.value == "critical"
    assert a.phase == "lift"
    assert a.threshold == (0.0, 1.0)
