"""Metric assertions and assertion engine for constraint evaluation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from roboharness.evaluate.result import (
    AssertionResult,
    EvaluationResult,
    Operator,
    Severity,
    Verdict,
)


@dataclass
class MetricAssertion:
    """A single constraint check against one metric value.

    Attributes:
        metric: Key to look up in the report (e.g. ``grip_center_error_mm``).
        operator: Comparison type.
        threshold: Scalar value, or ``(low, high)`` tuple for ``in_range``.
        severity: How important this assertion is for the overall verdict.
        phase: Which simulation phase to check. ``"*"`` means trial-level
            (``summary_metrics``).
    """

    metric: str
    operator: Operator
    threshold: float | tuple[float, float]
    severity: Severity = Severity.MAJOR
    phase: str = "*"

    def evaluate(self, value: float | None) -> AssertionResult:
        """Evaluate this assertion against a metric value.

        Missing metrics (``None``) are treated as failures.
        """
        if value is None:
            return AssertionResult(
                metric=self.metric,
                operator=self.operator,
                threshold=self.threshold,
                severity=self.severity,
                phase=self.phase,
                passed=False,
                actual_value=None,
                message=f"metric '{self.metric}' missing from report",
            )

        passed = self._compare(value)
        msg = f"{self.metric}={value}" if passed else self._fail_message(value)
        return AssertionResult(
            metric=self.metric,
            operator=self.operator,
            threshold=self.threshold,
            severity=self.severity,
            phase=self.phase,
            passed=passed,
            actual_value=value,
            message=msg,
        )

    def _compare(self, value: float) -> bool:
        if self.operator == Operator.LT:
            return value < self.threshold  # type: ignore[operator]
        if self.operator == Operator.LE:
            return value <= self.threshold  # type: ignore[operator]
        if self.operator == Operator.EQ:
            return value == self.threshold
        if self.operator == Operator.GT:
            return value > self.threshold  # type: ignore[operator]
        if self.operator == Operator.GE:
            return value >= self.threshold  # type: ignore[operator]
        if self.operator == Operator.IN_RANGE:
            low, high = self.threshold  # type: ignore[misc]
            return low <= value <= high
        msg = f"unknown operator: {self.operator}"
        raise ValueError(msg)

    def _fail_message(self, value: float) -> str:
        if self.operator == Operator.IN_RANGE:
            low, high = self.threshold  # type: ignore[misc]
            return f"{self.metric}={value}, expected in [{low}, {high}]"
        return f"{self.metric}={value}, expected {self.operator.value} {self.threshold}"


def _extract_metric(report: dict[str, Any], metric: str, phase: str) -> float | None:
    """Extract a metric value from a report dict.

    For ``phase="*"``, looks in ``summary_metrics``.
    For a specific phase, looks in ``snapshot_metrics[phase]``.
    """
    if phase == "*":
        metrics = report.get("summary_metrics", {})
    else:
        metrics = report.get("snapshot_metrics", {}).get(phase, {})
    value = metrics.get(metric)
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class AssertionEngine:
    """Evaluates a list of assertions against a harness report."""

    def __init__(self, assertions: Sequence[MetricAssertion]) -> None:
        self.assertions = assertions

    def evaluate(self, report: dict[str, Any], report_path: str = "") -> EvaluationResult:
        """Run all assertions against the report and return an aggregated result."""
        results: list[AssertionResult] = []
        for assertion in self.assertions:
            value = _extract_metric(report, assertion.metric, assertion.phase)
            results.append(assertion.evaluate(value))

        verdict = self._compute_verdict(results)
        return EvaluationResult(verdict=verdict, results=results, report_path=report_path)

    @staticmethod
    def _compute_verdict(results: list[AssertionResult]) -> Verdict:
        """Determine verdict from assertion results.

        - ``fail`` if any critical assertion fails
        - ``degraded`` if any major assertion fails (no critical failures)
        - ``pass`` otherwise
        """
        has_critical_failure = any(
            not r.passed and r.severity == Severity.CRITICAL for r in results
        )
        has_major_failure = any(not r.passed and r.severity == Severity.MAJOR for r in results)
        if has_critical_failure:
            return Verdict.FAIL
        if has_major_failure:
            return Verdict.DEGRADED
        return Verdict.PASS
