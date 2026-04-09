"""MCP tool implementations for roboharness.

Each public method on :class:`HarnessTools` corresponds to one MCP tool.
Methods accept and return plain dicts/scalars so they can be called directly
or serialised as JSON for the MCP ``tools/call`` response.

The class is intentionally decoupled from the ``mcp`` SDK -- it contains
pure business logic.  The thin MCP server wrapper lives in ``server.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from roboharness.core.capture import CameraView, CaptureResult
from roboharness.core.checkpoint import Checkpoint
from roboharness.core.harness import Harness
from roboharness.evaluate.assertions import AssertionEngine
from roboharness.evaluate.constraints import _parse_assertion
from roboharness.storage.history import EvaluationHistory

# JSON-schema descriptions exposed by the MCP server layer.
TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "capture_checkpoint",
        "description": (
            "Pause the simulation and capture multi-view screenshots plus "
            "simulator state at the current timestep."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "checkpoint_name": {
                    "type": "string",
                    "description": "Label for this capture (default: current step number).",
                },
                "cameras": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": 'Camera names to capture (default: ["front"]).',
                },
            },
        },
    },
    {
        "name": "evaluate_constraints",
        "description": (
            "Run the constraint evaluator on a report dict and return the "
            "verdict (pass / degraded / fail) with per-assertion details."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "report": {
                    "type": "object",
                    "description": "Harness report dict with summary_metrics / snapshot_metrics.",
                },
                "assertions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "metric": {"type": "string"},
                            "operator": {
                                "type": "string",
                                "enum": ["lt", "le", "eq", "gt", "ge", "in_range"],
                            },
                            "threshold": {},
                            "severity": {
                                "type": "string",
                                "enum": ["critical", "major", "minor", "info"],
                                "default": "major",
                            },
                            "phase": {"type": "string", "default": "*"},
                        },
                        "required": ["metric", "operator", "threshold"],
                    },
                    "description": "List of metric assertions to evaluate.",
                },
            },
            "required": ["report", "assertions"],
        },
    },
    {
        "name": "compare_baselines",
        "description": (
            "Compare the current success rate against recent evaluation "
            "history and flag regressions or improvements."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "task": {"type": "string", "description": "Task name to compare."},
                "current_rate": {
                    "type": "number",
                    "description": "Current success rate (0.0-1.0).",
                },
                "window": {
                    "type": "integer",
                    "description": "Number of recent runs to average (default 5).",
                    "default": 5,
                },
                "threshold": {
                    "type": "number",
                    "description": "Minimum delta to flag a change (default 0.1).",
                    "default": 0.1,
                },
            },
            "required": ["task", "current_rate"],
        },
    },
]


def _camera_view_to_dict(view: CameraView) -> dict[str, Any]:
    """Serialise a CameraView to a JSON-friendly dict."""
    result: dict[str, Any] = {
        "name": view.name,
        "rgb_shape": list(view.rgb.shape),
    }
    if view.depth is not None:
        result["depth_shape"] = list(view.depth.shape)
    if view.segmentation is not None:
        result["segmentation_shape"] = list(view.segmentation.shape)
    return result


def _capture_to_dict(capture: CaptureResult) -> dict[str, Any]:
    """Serialise a CaptureResult to a JSON-friendly dict."""
    state = {}
    for k, v in capture.state.items():
        if isinstance(v, np.ndarray):
            state[k] = v.tolist()
        else:
            state[k] = v

    return {
        "checkpoint_name": capture.checkpoint_name,
        "step": capture.step,
        "sim_time": capture.sim_time,
        "views": [_camera_view_to_dict(v) for v in capture.views],
        "state": state,
        "metadata": capture.metadata,
    }


class HarnessTools:
    """Stateful tool implementations backed by a running :class:`Harness`.

    Parameters
    ----------
    harness:
        A fully-initialised harness with a connected simulator backend.
    history_dir:
        Path to the evaluation-history directory.  If ``None``,
        ``compare_baselines`` will use the harness output dir.
    """

    def __init__(
        self,
        harness: Harness,
        history_dir: str | Path | None = None,
    ) -> None:
        self._harness = harness
        history_path = Path(history_dir) if history_dir else harness.output_dir / "history"
        self._history = EvaluationHistory(history_path)

    # ---- Tool: capture_checkpoint ----------------------------------------

    def capture_checkpoint(
        self,
        checkpoint_name: str | None = None,
        cameras: list[str] | None = None,
    ) -> dict[str, Any]:
        """Capture multi-view screenshots and simulation state.

        Returns a JSON-serialisable dict describing the captured views,
        simulation state, and metadata.
        """
        checkpoint = Checkpoint(
            name=checkpoint_name or f"step_{self._harness.step_count}",
            cameras=cameras or ["front"],
        )
        capture = self._harness.capture(checkpoint)
        return _capture_to_dict(capture)

    # ---- Tool: evaluate_constraints --------------------------------------

    def evaluate_constraints(
        self,
        report: dict[str, Any],
        assertions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Run the constraint evaluator and return a verdict dict.

        Parameters
        ----------
        report:
            A harness report dict containing ``summary_metrics`` and/or
            ``snapshot_metrics`` sections.
        assertions:
            List of assertion dicts, each with ``metric``, ``operator``,
            ``threshold``, and optional ``severity`` / ``phase``.
        """
        parsed = [_parse_assertion(a) for a in assertions]
        engine = AssertionEngine(parsed)
        result = engine.evaluate(report)
        return result.to_dict()

    # ---- Tool: compare_baselines -----------------------------------------

    def compare_baselines(
        self,
        task: str,
        current_rate: float,
        window: int = 5,
        threshold: float = 0.1,
    ) -> dict[str, Any]:
        """Compare current success rate against recent history.

        Returns a dict with trend direction, delta, and human-readable message.
        """
        trend = self._history.detect_trend(
            task=task,
            current_rate=current_rate,
            window=window,
            threshold=threshold,
        )
        return trend.to_dict()
