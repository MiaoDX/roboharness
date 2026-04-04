"""Evaluation result types for the constraint evaluator."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any


class Verdict(enum.Enum):
    """Overall evaluation verdict."""

    PASS = "pass"
    DEGRADED = "degraded"
    FAIL = "fail"


class Severity(enum.Enum):
    """Assertion severity level."""

    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"
    INFO = "info"


class Operator(enum.Enum):
    """Comparison operator for metric assertions."""

    LT = "lt"
    LE = "le"
    EQ = "eq"
    GT = "gt"
    GE = "ge"
    IN_RANGE = "in_range"


@dataclass
class AssertionResult:
    """Result of evaluating a single assertion."""

    metric: str
    operator: Operator
    threshold: float | tuple[float, float]
    severity: Severity
    phase: str
    passed: bool
    actual_value: float | None = None
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        threshold: float | list[float] = (
            list(self.threshold) if isinstance(self.threshold, tuple) else self.threshold
        )
        return {
            "metric": self.metric,
            "operator": self.operator.value,
            "threshold": threshold,
            "severity": self.severity.value,
            "phase": self.phase,
            "passed": self.passed,
            "actual_value": self.actual_value,
            "message": self.message,
        }


@dataclass
class EvaluationResult:
    """Aggregated result of all assertions against a report."""

    verdict: Verdict
    results: list[AssertionResult] = field(default_factory=list)
    report_path: str = ""

    @property
    def passed(self) -> list[AssertionResult]:
        return [r for r in self.results if r.passed]

    @property
    def failed(self) -> list[AssertionResult]:
        return [r for r in self.results if not r.passed]

    @property
    def critical_failures(self) -> list[AssertionResult]:
        return [r for r in self.results if not r.passed and r.severity == Severity.CRITICAL]

    @property
    def major_failures(self) -> list[AssertionResult]:
        return [r for r in self.results if not r.passed and r.severity == Severity.MAJOR]

    def to_dict(self) -> dict[str, Any]:
        passed_count = 0
        failed_count = 0
        critical_count = 0
        major_count = 0
        for r in self.results:
            if r.passed:
                passed_count += 1
            else:
                failed_count += 1
                if r.severity == Severity.CRITICAL:
                    critical_count += 1
                elif r.severity == Severity.MAJOR:
                    major_count += 1
        return {
            "verdict": self.verdict.value,
            "report_path": self.report_path,
            "total_assertions": len(self.results),
            "passed": passed_count,
            "failed": failed_count,
            "critical_failures": critical_count,
            "major_failures": major_count,
            "results": [r.to_dict() for r in self.results],
        }
