"""Batch evaluation: aggregate results across multiple trials."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from roboharness.evaluate.assertions import AssertionEngine, MetricAssertion
from roboharness.evaluate.result import EvaluationResult, Verdict


@dataclass
class TrialSummary:
    """Summary of a single trial's evaluation."""

    report_path: str
    case_id: str
    verdict: Verdict
    original_verdict: str
    critical_failures: int
    major_failures: int
    failure_codes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "report_path": self.report_path,
            "case_id": self.case_id,
            "verdict": self.verdict.value,
            "original_verdict": self.original_verdict,
            "critical_failures": self.critical_failures,
            "major_failures": self.major_failures,
            "failure_codes": self.failure_codes,
        }


@dataclass
class BatchResult:
    """Aggregated results across multiple trials."""

    results_dir: str
    total_trials: int
    verdicts: dict[str, int] = field(default_factory=dict)
    success_rate: float = 0.0
    failure_distribution: dict[str, int] = field(default_factory=dict)
    constraint_failures: dict[str, int] = field(default_factory=dict)
    trials: list[TrialSummary] = field(default_factory=list)
    per_evaluation: list[EvaluationResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "results_dir": self.results_dir,
            "total_trials": self.total_trials,
            "verdicts": self.verdicts,
            "success_rate": self.success_rate,
            "failure_distribution": self.failure_distribution,
            "constraint_failures": self.constraint_failures,
            "trials": [t.to_dict() for t in self.trials],
        }


@dataclass
class VariantResult:
    """Results for a single variant (e.g. grasp position)."""

    variant_id: str
    batch: BatchResult

    def to_dict(self) -> dict:
        return {
            "variant_id": self.variant_id,
            **self.batch.to_dict(),
        }


@dataclass
class ComparisonResult:
    """Side-by-side comparison of multiple variants."""

    variants: list[VariantResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "variants": [v.to_dict() for v in self.variants],
            "summary": {
                v.variant_id: {
                    "total_trials": v.batch.total_trials,
                    "success_rate": v.batch.success_rate,
                    "verdicts": v.batch.verdicts,
                }
                for v in self.variants
            },
        }


def find_reports(results_dir: Path) -> list[Path]:
    """Find all autonomous_report.json files under a directory."""
    return sorted(results_dir.rglob("autonomous_report.json"))


def _load_report(path: Path) -> dict[str, Any]:
    with path.open() as f:
        result: dict[str, Any] = json.load(f)
        return result


def _failure_codes_from_report(report: dict[str, Any]) -> list[str]:
    """Extract failure taxonomy codes from a report."""
    taxonomy = report.get("failure_taxonomy", [])
    return [entry.get("code", "unknown") for entry in taxonomy if isinstance(entry, dict)]


def evaluate_batch(
    results_dir: Path,
    assertions: list[MetricAssertion],
) -> BatchResult:
    """Evaluate all trial reports in a directory against constraints.

    Finds all ``autonomous_report.json`` files under *results_dir*, evaluates
    each against the given assertions, and returns aggregate statistics.
    """
    report_paths = find_reports(results_dir)
    engine = AssertionEngine(assertions)

    verdict_counter: Counter[str] = Counter()
    failure_code_counter: Counter[str] = Counter()
    constraint_counter: Counter[str] = Counter()
    trials: list[TrialSummary] = []
    evaluations: list[EvaluationResult] = []

    for rpath in report_paths:
        report = _load_report(rpath)
        eval_result = engine.evaluate(report, report_path=str(rpath))
        evaluations.append(eval_result)

        case_id = report.get("case_id", rpath.parent.name)
        original_verdict = report.get("verdict", "unknown")
        failure_codes = _failure_codes_from_report(report)

        verdict_counter[eval_result.verdict.value] += 1
        failure_code_counter.update(failure_codes)

        # Track which constraints fail most often
        for ar in eval_result.failed:
            constraint_counter[ar.metric] += 1

        trials.append(
            TrialSummary(
                report_path=str(rpath),
                case_id=case_id,
                verdict=eval_result.verdict,
                original_verdict=original_verdict,
                critical_failures=len(eval_result.critical_failures),
                major_failures=len(eval_result.major_failures),
                failure_codes=failure_codes,
            )
        )

    total = len(report_paths)
    pass_count = verdict_counter.get("pass", 0)
    success_rate = pass_count / total if total > 0 else 0.0

    return BatchResult(
        results_dir=str(results_dir),
        total_trials=total,
        verdicts=dict(verdict_counter),
        success_rate=success_rate,
        failure_distribution=dict(failure_code_counter),
        constraint_failures=dict(constraint_counter),
        trials=trials,
        per_evaluation=evaluations,
    )


def evaluate_batch_with_comparison(
    results_dir: Path,
    assertions: list[MetricAssertion],
) -> ComparisonResult:
    """Evaluate trials grouped by variant (subdirectory) for comparison.

    Each immediate subdirectory of *results_dir* is treated as a variant.
    Reports are found recursively within each variant directory.
    """
    variants: list[VariantResult] = []

    subdirs = sorted(p for p in results_dir.iterdir() if p.is_dir())

    for subdir in subdirs:
        reports = find_reports(subdir)
        if not reports:
            continue
        batch = evaluate_batch(subdir, assertions)
        variants.append(VariantResult(variant_id=subdir.name, batch=batch))

    return ComparisonResult(variants=variants)


def format_batch_human(batch: BatchResult) -> str:
    """Format batch results for human-readable CLI output."""
    lines: list[str] = []
    lines.append(f"Batch evaluation: {batch.results_dir}")
    lines.append(f"  Total trials: {batch.total_trials}")
    lines.append(f"  Success rate: {batch.success_rate:.0%}")
    lines.append(f"  Verdicts: {_format_verdicts(batch.verdicts)}")

    if batch.failure_distribution:
        lines.append("  Failure distribution:")
        for code, count in sorted(batch.failure_distribution.items(), key=lambda x: -x[1]):
            lines.append(f"    {code}: {count}")

    if batch.constraint_failures:
        lines.append("  Constraint failures:")
        for metric, count in sorted(batch.constraint_failures.items(), key=lambda x: -x[1]):
            lines.append(f"    {metric}: {count}/{batch.total_trials} trials")

    return "\n".join(lines)


def format_comparison_human(comparison: ComparisonResult) -> str:
    """Format comparison results for human-readable CLI output."""
    lines: list[str] = []
    lines.append("Variant comparison:")
    lines.append("")

    # Header
    max_name = max(len(v.variant_id) for v in comparison.variants) if comparison.variants else 10
    header = f"  {'Variant':<{max_name}}  Trials  Rate    Verdicts"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    for v in comparison.variants:
        b = v.batch
        verdicts_str = _format_verdicts(b.verdicts)
        lines.append(
            f"  {v.variant_id:<{max_name}}  {b.total_trials:>6}"
            f"  {b.success_rate:>5.0%}   {verdicts_str}"
        )

    return "\n".join(lines)


def _format_verdicts(verdicts: dict[str, int]) -> str:
    parts = []
    for v in ["pass", "degraded", "fail"]:
        if v in verdicts:
            parts.append(f"{v}={verdicts[v]}")
    return ", ".join(parts) if parts else "none"


def check_success_rate(batch: BatchResult, min_rate: float) -> bool:
    """Check if the batch success rate meets a minimum threshold.

    Useful for CI integration: assert that >=80% of trials pass.
    """
    return batch.success_rate >= min_rate
