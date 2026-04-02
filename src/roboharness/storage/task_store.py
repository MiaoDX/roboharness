"""Task-oriented storage for harness output.

Storage layout for a grasp task with multiple grasp positions:

    harness_output/
    └── pick_and_place/                     # task name
        ├── task_config.json                # task-level config
        ├── grasp_position_001/             # grasp position 1
        │   ├── position.json               # grasp pose (xyz + quat)
        │   ├── trial_001/                  # first attempt at this position
        │   │   ├── plan_start/
        │   │   │   ├── front_rgb.png
        │   │   │   ├── side_rgb.png
        │   │   │   ├── top_rgb.png
        │   │   │   ├── state.json
        │   │   │   └── metadata.json
        │   │   ├── pre_grasp/
        │   │   │   └── ...
        │   │   ├── contact/
        │   │   │   └── ...
        │   │   ├── lift/
        │   │   │   └── ...
        │   │   └── result.json             # trial outcome (success/fail + reason)
        │   ├── trial_002/                  # second attempt (after agent modified code)
        │   │   └── ...
        │   └── summary.json               # best trial, success rate, iterations
        ├── grasp_position_002/
        │   └── ...
        └── report.json                    # overall task report
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TrialResult:
    """Result of a single trial attempt."""

    trial_id: int
    success: bool
    reason: str = ""
    metrics: dict[str, float] = field(default_factory=dict)
    duration: float = 0.0
    checkpoints_reached: list[str] = field(default_factory=list)


class TaskStore:
    """Base class for task-oriented storage management.

    Organizes harness output by task → position/variant → trial → checkpoint.
    """

    def __init__(self, base_dir: Path | str, task_name: str):
        self.base_dir = Path(base_dir) / task_name
        self.task_name = task_name
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def get_variant_dir(self, variant_name: str) -> Path:
        """Get directory for a specific task variant (e.g., grasp position)."""
        d = self.base_dir / variant_name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def get_trial_dir(self, variant_name: str, trial_id: int) -> Path:
        """Get directory for a specific trial within a variant."""
        d = self.get_variant_dir(variant_name) / f"trial_{trial_id:03d}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def get_checkpoint_dir(
        self, variant_name: str, trial_id: int, checkpoint_name: str
    ) -> Path:
        """Get directory for a specific checkpoint within a trial."""
        d = self.get_trial_dir(variant_name, trial_id) / checkpoint_name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save_task_config(self, config: dict[str, Any]) -> Path:
        """Save task-level configuration."""
        path = self.base_dir / "task_config.json"
        _write_json(path, config)
        return path

    def save_trial_result(
        self, variant_name: str, result: TrialResult
    ) -> Path:
        """Save the result of a trial."""
        trial_dir = self.get_trial_dir(variant_name, result.trial_id)
        path = trial_dir / "result.json"
        _write_json(path, {
            "trial_id": result.trial_id,
            "success": result.success,
            "reason": result.reason,
            "metrics": result.metrics,
            "duration": result.duration,
            "checkpoints_reached": result.checkpoints_reached,
            "timestamp": time.time(),
        })
        return path

    def save_variant_summary(
        self, variant_name: str, summary: dict[str, Any]
    ) -> Path:
        """Save summary for a variant (e.g., best trial, success rate)."""
        path = self.get_variant_dir(variant_name) / "summary.json"
        _write_json(path, summary)
        return path

    def save_report(self, report: dict[str, Any]) -> Path:
        """Save overall task report."""
        path = self.base_dir / "report.json"
        _write_json(path, report)
        return path

    def list_variants(self) -> list[str]:
        """List all variants (e.g., grasp positions) for this task."""
        return [
            d.name
            for d in sorted(self.base_dir.iterdir())
            if d.is_dir()
        ]

    def list_trials(self, variant_name: str) -> list[int]:
        """List all trial IDs for a variant."""
        variant_dir = self.get_variant_dir(variant_name)
        trials = []
        for d in sorted(variant_dir.iterdir()):
            if d.is_dir() and d.name.startswith("trial_"):
                try:
                    trials.append(int(d.name.split("_")[1]))
                except (ValueError, IndexError):
                    pass
        return trials


class GraspTaskStore(TaskStore):
    """Specialized storage for grasp tasks with multiple grasp positions.

    Each grasp position is a "variant" that stores:
    - The target grasp pose (position + orientation)
    - Multiple trial attempts by the agent
    - Per-checkpoint captures (plan_start, pre_grasp, contact, lift)
    """

    CHECKPOINTS = ["plan_start", "pre_grasp", "contact", "lift"]

    def __init__(self, base_dir: Path | str, task_name: str = "grasp"):
        super().__init__(base_dir, task_name)

    def add_grasp_position(
        self,
        position_id: int,
        xyz: tuple[float, float, float],
        quaternion: tuple[float, float, float, float] | None = None,
        object_name: str = "",
        **extra: Any,
    ) -> Path:
        """Register a new grasp position."""
        variant_name = f"grasp_position_{position_id:03d}"
        variant_dir = self.get_variant_dir(variant_name)
        position_data = {
            "position_id": position_id,
            "xyz": list(xyz),
            "quaternion": list(quaternion) if quaternion else None,
            "object_name": object_name,
            **extra,
        }
        path = variant_dir / "position.json"
        _write_json(path, position_data)
        return variant_dir

    def get_grasp_checkpoint_dir(
        self,
        position_id: int,
        trial_id: int,
        checkpoint: str,
    ) -> Path:
        """Get directory for a checkpoint within a grasp trial."""
        variant_name = f"grasp_position_{position_id:03d}"
        return self.get_checkpoint_dir(variant_name, trial_id, checkpoint)

    def generate_report(self) -> dict[str, Any]:
        """Generate a summary report across all grasp positions."""
        variants = self.list_variants()
        report: dict[str, Any] = {
            "task": self.task_name,
            "total_positions": len(variants),
            "positions": {},
        }

        for variant in variants:
            summary_path = self.get_variant_dir(variant) / "summary.json"
            if summary_path.exists():
                with open(summary_path) as f:
                    report["positions"][variant] = json.load(f)

        self.save_report(report)
        return report


def _write_json(path: Path, data: dict[str, Any]) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
