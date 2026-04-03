"""Evaluation result types for the constraint evaluator."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field


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

    def to_dict(self) -> dict:
        """Serialize to dict."""
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

    def to_dict(self) -> dict:
        """Serialize to dict."""
        return {
            "verdict": self.verdict.value,
            "report_path": self.report_path,
            "total_assertions": len(self.results),
            "passed": len(self.passed),
            "failed": len(self.failed),
            "critical_failures": len(self.critical_failures),
            "major_failures": len(self.major_failures),
            "results": [r.to_dict() for r in self.results],
        }
