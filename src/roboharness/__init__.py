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
from roboharness.evaluate.assertions import AssertionEngine, MetricAssertion
from roboharness.evaluate.result import EvaluationResult, Operator, Severity, Verdict
from roboharness.runner import BatchResult, ParallelTrialRunner, TrialSpec
from roboharness.storage.history import EvaluationHistory, EvaluationRecord, TrendResult

__all__ = [
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
    "LifecycleRegistry",
    "MetricAssertion",
    "Operator",
    "ParallelTrialRunner",
    "Severity",
    "TrendResult",
    "TrialSpec",
    "Verdict",
    "default_registry",
]
