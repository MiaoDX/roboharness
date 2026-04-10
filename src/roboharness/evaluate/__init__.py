"""Constraint evaluation and batch trial aggregation for roboharness."""

from __future__ import annotations

from roboharness.evaluate.assertions import AssertionEngine, MetricAssertion
from roboharness.evaluate.batch import (
    ComparisonResult,
    EvalBatchResult,
    VariantResult,
    check_success_rate,
    evaluate_batch,
    evaluate_batch_with_comparison,
)
from roboharness.evaluate.constraints import load_constraints, save_constraints
from roboharness.evaluate.defaults import GRASP_DEFAULTS
from roboharness.evaluate.lerobot_plugin import (
    EpisodeResult,
    LeRobotEvalConfig,
    LeRobotEvalReport,
    check_eval_threshold,
    evaluate_policy,
)
from roboharness.evaluate.result import (
    AssertionResult,
    EvaluationResult,
    Operator,
    Severity,
    Verdict,
)

__all__ = [
    "GRASP_DEFAULTS",
    "AssertionEngine",
    "AssertionResult",
    "ComparisonResult",
    "EpisodeResult",
    "EvalBatchResult",
    "EvaluationResult",
    "LeRobotEvalConfig",
    "LeRobotEvalReport",
    "MetricAssertion",
    "Operator",
    "Severity",
    "VariantResult",
    "Verdict",
    "check_eval_threshold",
    "check_success_rate",
    "evaluate_batch",
    "evaluate_batch_with_comparison",
    "evaluate_policy",
    "load_constraints",
    "save_constraints",
]
