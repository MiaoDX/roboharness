"""Shared internal utilities — image saving and JSON I/O."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def save_image(arr: np.ndarray, path: Path) -> None:
    """Save RGB array as PNG. Uses PIL if available, falls back to raw numpy."""
    try:
        from PIL import Image

        img = Image.fromarray(arr)
        img.save(path)
    except ImportError:
        # Fallback: save as npy with .png extension note
        npy_path = path.with_suffix(".npy")
        np.save(npy_path, arr)


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)


def save_json(data: dict[str, Any], path: Path) -> None:
    """Save dict as JSON, converting numpy types."""
    with path.open("w") as f:
        json.dump(data, f, indent=2, cls=NumpyEncoder)


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON file and return as dict."""
    with path.open() as f:
        result: dict[str, Any] = json.load(f)
    return result
