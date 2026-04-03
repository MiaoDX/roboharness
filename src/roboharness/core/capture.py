"""Multi-view screenshot capture and storage."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class CameraView:
    """A single camera view capture."""

    name: str
    rgb: np.ndarray  # (H, W, 3) uint8
    depth: np.ndarray | None = None  # (H, W) float32, meters
    segmentation: np.ndarray | None = None  # (H, W) int32

    def save(self, directory: Path) -> dict[str, str]:
        """Save camera view to directory. Returns dict of saved file paths."""
        directory.mkdir(parents=True, exist_ok=True)
        saved = {}

        rgb_path = directory / f"{self.name}_rgb.png"
        _save_image(self.rgb, rgb_path)
        saved["rgb"] = str(rgb_path)

        if self.depth is not None:
            depth_path = directory / f"{self.name}_depth.npy"
            np.save(depth_path, self.depth)
            saved["depth"] = str(depth_path)

            # Also save a normalized visualization for agent consumption
            depth_viz_path = directory / f"{self.name}_depth_viz.png"
            _save_depth_viz(self.depth, depth_viz_path)
            saved["depth_viz"] = str(depth_viz_path)

        if self.segmentation is not None:
            seg_path = directory / f"{self.name}_segmentation.npy"
            np.save(seg_path, self.segmentation)
            saved["segmentation"] = str(seg_path)

        return saved


@dataclass
class CaptureResult:
    """Result of a multi-view capture at a checkpoint."""

    checkpoint_name: str
    step: int
    sim_time: float
    views: list[CameraView] = field(default_factory=list)
    state: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def save(self, base_dir: Path) -> Path:
        """Save capture to disk in agent-consumable format.

        Directory layout:
            base_dir/
            ├── front_rgb.png
            ├── front_depth_viz.png
            ├── side_rgb.png
            ├── state.json
            └── metadata.json
        """
        base_dir.mkdir(parents=True, exist_ok=True)

        # Save each camera view
        all_files = {}
        for view in self.views:
            files = view.save(base_dir)
            all_files[view.name] = files

        # Save state
        state_path = base_dir / "state.json"
        _save_json(self.state, state_path)

        # Save metadata
        meta = {
            "checkpoint": self.checkpoint_name,
            "step": self.step,
            "sim_time": self.sim_time,
            "timestamp": self.timestamp,
            "cameras": list(all_files.keys()),
            "files": all_files,
            **self.metadata,
        }
        meta_path = base_dir / "metadata.json"
        _save_json(meta, meta_path)

        return base_dir


def _save_image(arr: np.ndarray, path: Path) -> None:
    """Save RGB array as PNG. Uses PIL if available, falls back to raw numpy."""
    try:
        from PIL import Image

        img = Image.fromarray(arr)
        img.save(path)
    except ImportError:
        # Fallback: save as npy with .png extension note
        npy_path = path.with_suffix(".npy")
        np.save(npy_path, arr)


def _save_depth_viz(depth: np.ndarray, path: Path) -> None:
    """Save depth as a normalized grayscale visualization."""
    valid = depth[np.isfinite(depth)]
    if valid.size == 0:
        return
    d_min, d_max = valid.min(), valid.max()
    if d_max - d_min < 1e-6:
        normalized = np.zeros_like(depth, dtype=np.uint8)
    else:
        normalized = ((depth - d_min) / (d_max - d_min) * 255).astype(np.uint8)
    _save_image(np.stack([normalized] * 3, axis=-1), path)


def _save_json(data: dict, path: Path) -> None:
    """Save dict as JSON, converting numpy types."""

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj: Any) -> Any:
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            return super().default(obj)

    with path.open("w") as f:
        json.dump(data, f, indent=2, cls=NumpyEncoder)
