"""Default constraint presets for common task types."""

from __future__ import annotations

from roboharness.evaluate.assertions import MetricAssertion
from roboharness.evaluate.result import Operator, Severity

# Default grasp constraints derived from real harness report thresholds.
GRASP_DEFAULTS: list[MetricAssertion] = [
    MetricAssertion(
        metric="grip_center_error_mm",
        operator=Operator.LT,
        threshold=50.0,
        severity=Severity.CRITICAL,
        phase="*",
    ),
    MetricAssertion(
        metric="pinch_gap_error_mm",
        operator=Operator.LT,
        threshold=15.0,
        severity=Severity.CRITICAL,
        phase="*",
    ),
    MetricAssertion(
        metric="pinch_elevation_deg",
        operator=Operator.LT,
        threshold=15.0,
        severity=Severity.MAJOR,
        phase="*",
    ),
    MetricAssertion(
        metric="index_middle_vertical_deg",
        operator=Operator.LT,
        threshold=20.0,
        severity=Severity.MAJOR,
        phase="*",
    ),
]
