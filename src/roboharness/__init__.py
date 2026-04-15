"""Roboharness: A Visual Testing Harness for AI Coding Agents in Robot Simulation."""

__version__ = "0.2.2"

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
from roboharness.evaluate.lerobot_plugin import (
    EpisodeResult,
    LeRobotEvalConfig,
    LeRobotEvalReport,
    check_eval_threshold,
    evaluate_policy,
)
from roboharness.evaluate.result import EvaluationResult, Operator, Severity, Verdict
from roboharness.runner import BatchResult, ParallelTrialRunner, TrialSpec
from roboharness.storage.history import EvaluationHistory, EvaluationRecord, TrendResult

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
    "EpisodeResult",
    "EvaluationHistory",
    "EvaluationRecord",
    "EvaluationResult",
    "ExpirationHorizon",
    "Harness",
    "LeRobotEvalConfig",
    "LeRobotEvalReport",
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
    "check_eval_threshold",
    "default_registry",
    "evaluate_policy",
]
