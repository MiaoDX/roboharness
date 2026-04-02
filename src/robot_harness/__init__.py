"""Robot-Harness: A Visual Testing Harness for AI Coding Agents in Robot Simulation."""

__version__ = "0.1.0"

from robot_harness.core.capture import CaptureResult
from robot_harness.core.checkpoint import Checkpoint, CheckpointStore
from robot_harness.core.harness import Harness

__all__ = [
    "Harness",
    "Checkpoint",
    "CheckpointStore",
    "CaptureResult",
]
