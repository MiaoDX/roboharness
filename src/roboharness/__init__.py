"""Roboharness: A Visual Testing Harness for AI Coding Agents in Robot Simulation."""

__version__ = "0.1.1"

from roboharness.core.capture import CaptureResult
from roboharness.core.checkpoint import Checkpoint, CheckpointStore
from roboharness.core.controller import Controller
from roboharness.core.harness import Harness
from roboharness.core.lifecycle import (
    ComponentAssumption,
    ComponentLifecycle,
    ExpirationHorizon,
    LifecycleRegistry,
    default_registry,
)
from roboharness.core.protocol import (
    BUILTIN_PROTOCOLS,
    DANCE_PROTOCOL,
    GRASP_PROTOCOL,
    LOCO_MANIPULATION_PROTOCOL,
    LOCOMOTION_PROTOCOL,
    REACH_PROTOCOL,
    TaskPhase,
    TaskProtocol,
)
from roboharness.evaluate.assertions import AssertionEngine, MetricAssertion
from roboharness.evaluate.result import EvaluationResult, Operator, Severity, Verdict
from roboharness.runner import BatchResult, ParallelTrialRunner, TrialSpec
from roboharness.storage.history import EvaluationHistory, EvaluationRecord, TrendResult

# Lazy imports for optional lerobot module — only available when gymnasium is installed.
# Import directly from roboharness.lerobot for full API access.


def __getattr__(name: str) -> object:
    if name == "LeRobotEvaluator":
        from roboharness.lerobot.evaluator import LeRobotEvaluator

        return LeRobotEvaluator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BUILTIN_PROTOCOLS",
    "DANCE_PROTOCOL",
    "GRASP_PROTOCOL",
    "LOCOMOTION_PROTOCOL",
    "LOCO_MANIPULATION_PROTOCOL",
    "REACH_PROTOCOL",
    "AssertionEngine",
    "BatchResult",
    "CaptureResult",
    "Checkpoint",
    "CheckpointStore",
    "ComponentAssumption",
    "ComponentLifecycle",
    "Controller",
    "EvaluationHistory",
    "EvaluationRecord",
    "EvaluationResult",
    "ExpirationHorizon",
    "Harness",
    "LeRobotEvaluator",
    "LifecycleRegistry",
    "MetricAssertion",
    "Operator",
    "ParallelTrialRunner",
    "Severity",
    "TaskPhase",
    "TaskProtocol",
    "TrendResult",
    "TrialSpec",
    "Verdict",
    "default_registry",
]
