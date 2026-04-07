"""Checkpoint management for simulation harness."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Checkpoint:
    """A named checkpoint in the simulation timeline.

    Checkpoints define moments where the harness pauses simulation,
    captures multi-view screenshots, and allows the agent to inspect state.
    """

    name: str
    cameras: list[str] = field(default_factory=lambda: ["front"])
    trigger_step: int | None = None
    trigger_condition: str | None = None  # e.g. "contact_detected", "plan_complete"
    metadata: dict[str, Any] = field(default_factory=dict)


class CheckpointStore:
    """Manages simulation state snapshots for checkpoint save/restore.

    This allows agents to "rewind" the simulation to a previous checkpoint
    when they decide the current approach isn't working.
    """

    def __init__(self, base_dir: Path | str = "./checkpoints"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._snapshots: dict[str, dict[str, Any]] = {}

    def save(self, name: str, state: dict[str, Any]) -> None:
        """Save a simulation state snapshot (in-memory)."""
        self._snapshots[name] = state

    def restore(self, name: str) -> dict[str, Any]:
        """Restore a simulation state snapshot."""
        if name not in self._snapshots:
            raise KeyError(
                f"Checkpoint '{name}' not found. Available: {list(self._snapshots.keys())}"
            )
        return self._snapshots[name]

    def list_checkpoints(self) -> list[str]:
        """List all saved checkpoints."""
        return list(self._snapshots.keys())

    def has(self, name: str) -> bool:
        """Check if a checkpoint exists."""
        return name in self._snapshots
