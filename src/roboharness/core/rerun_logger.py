"""Optional Rerun logging for capture artifacts."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import numpy as np

from roboharness.core.capture import CameraView, CaptureResult


class RerunCaptureLogger:
    """Logs captures to a per-trial `.rrd` file when rerun-sdk is installed."""

    def __init__(self, app_id: str = "roboharness") -> None:
        self._app_id = app_id
        self._rr: Any | None = None
        self._rrd_path: Path | None = None

    @property
    def active(self) -> bool:
        return self._rr is not None and self._rrd_path is not None

    @property
    def rrd_path(self) -> Path | None:
        return self._rrd_path

    def configure_trial(self, trial_dir: Path, task_name: str) -> Path | None:
        """Initialize rerun recording for a trial.

        Returns the `.rrd` output path if rerun-sdk is available, otherwise None.
        """
        rr = self._import_rerun()
        if rr is None:
            self._rr = None
            self._rrd_path = None
            return None

        trial_dir.mkdir(parents=True, exist_ok=True)
        rrd_path = trial_dir / "capture.rrd"

        rr.init(f"{self._app_id}.{task_name}", spawn=False)
        rr.save(str(rrd_path))
        self._rr = rr
        self._rrd_path = rrd_path
        self._log_default_blueprint()
        return rrd_path

    def log_capture(self, capture: CaptureResult) -> None:
        """Append one checkpoint capture frame to rerun recording."""
        if self._rr is None:
            return

        rr = self._rr
        rr.set_time_sequence("sim_step", capture.step)
        rr.set_time_seconds("sim_time", capture.sim_time)

        rr.log("harness/checkpoint", rr.TextDocument(capture.checkpoint_name))
        rr.log("harness/state", rr.TextDocument(_as_json_text(capture.state)))

        for view in capture.views:
            self._log_view(view)

    def _log_view(self, view: CameraView) -> None:
        if self._rr is None:
            return
        rr = self._rr
        base = f"camera/{view.name}"
        rr.log(f"{base}/rgb", rr.Image(view.rgb))

        if view.depth is not None:
            depth = view.depth.astype(np.float32, copy=False)
            depth_image = rr.DepthImage(depth) if hasattr(rr, "DepthImage") else rr.Image(depth)
            rr.log(f"{base}/depth", depth_image)

        if view.segmentation is not None:
            seg = view.segmentation.astype(np.uint16, copy=False)
            seg_image = (
                rr.SegmentationImage(seg) if hasattr(rr, "SegmentationImage") else rr.Image(seg)
            )
            rr.log(f"{base}/segmentation", seg_image)

    def _log_default_blueprint(self) -> None:
        """Best-effort standard debug layout.

        Uses rerun.blueprint when available. Compatible no-op if APIs change.
        """
        if self._rr is None:
            return
        rr = self._rr

        if not hasattr(rr, "blueprint") or not hasattr(rr, "send_blueprint"):
            return

        bp = rr.blueprint
        required = ("Blueprint", "Horizontal", "Spatial2D", "TextDocumentView")
        if not all(hasattr(bp, name) for name in required):
            return

        blueprint = bp.Blueprint(
            bp.Horizontal(
                bp.Spatial2D(origin="/camera"),
                bp.TextDocumentView(origin="/harness"),
            )
        )
        rr.send_blueprint(blueprint)

    @staticmethod
    def _import_rerun() -> Any | None:
        try:
            return importlib.import_module("rerun")
        except ImportError:
            return None


def _as_json_text(payload: dict[str, Any]) -> str:
    import json

    return json.dumps(payload, indent=2, default=str, sort_keys=True)
