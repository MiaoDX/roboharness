"""Roboharness: A Visual Testing Harness for AI Coding Agents in Robot Simulation."""

__version__ = "0.1.0"

from roboharness.core.capture import CaptureResult
from roboharness.core.checkpoint import Checkpoint, CheckpointStore
from roboharness.core.controller import Controller
from roboharness.core.harness import Harness

__all__ = [
    "Controller",
    "Harness",
    "Checkpoint",
    "CheckpointStore",
    "CaptureResult",
]
