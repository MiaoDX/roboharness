"""Evaluation history store for tracking success rates across runs.

Append-only JSONL files persist evaluation results with timestamps, git commit
hashes, task names, verdicts, and metrics.  Trend detection compares current
runs against recent history to flag regressions.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EvaluationRecord:
    """A single evaluation run persisted in the history store."""

    task: str
    success_rate: float
    total_trials: int
    successes: int
    timestamp: float = field(default_factory=time.time)
    commit: str = ""
    metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvaluationRecord:
        return cls(
            task=data["task"],
            success_rate=data["success_rate"],
            total_trials=data["total_trials"],
            successes=data["successes"],
            timestamp=data.get("timestamp", 0.0),
            commit=data.get("commit", ""),
            metrics=data.get("metrics", {}),
        )


@dataclass
class TrendResult:
    """Result of comparing current evaluation against recent history."""

    task: str
    current_rate: float
    previous_rate: float | None
    delta: float | None
    window_size: int
    regressed: bool
    message: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _get_git_commit() -> str:
    """Get the current git commit hash, or empty string if unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return ""


class EvaluationHistory:
    """Append-only JSONL store for evaluation results.

    Records are stored in ``eval_history.jsonl`` inside the given directory.
    Each line is a JSON object representing one :class:`EvaluationRecord`.
    """

    FILENAME = "eval_history.jsonl"

    def __init__(self, directory: Path | str) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._path = self.directory / self.FILENAME

    @property
    def path(self) -> Path:
        return self._path

    def append(self, record: EvaluationRecord) -> None:
        """Append a single evaluation record to the history file."""
        with self._path.open("a") as f:
            f.write(json.dumps(record.to_dict(), default=str) + "\n")

    def load(self, task: str | None = None) -> list[EvaluationRecord]:
        """Load all records, optionally filtered by task name."""
        if not self._path.exists():
            return []
        records: list[EvaluationRecord] = []
        with self._path.open() as f:
            for raw_line in f:
                stripped = raw_line.strip()
                if not stripped:
                    continue
                data = json.loads(stripped)
                if task is None or data.get("task") == task:
                    records.append(EvaluationRecord.from_dict(data))
        return records

    def record_from_report(
        self, report: dict[str, Any], commit: str = ""
    ) -> list[EvaluationRecord]:
        """Create and append records from a ``report_command`` report dict.

        Returns the list of records that were appended.
        """
        if not commit:
            commit = _get_git_commit()

        records: list[EvaluationRecord] = []
        for task_name, task_data in report.get("tasks", {}).items():
            rate = task_data.get("success_rate")
            if rate is None:
                continue
            record = EvaluationRecord(
                task=task_name,
                success_rate=rate,
                total_trials=task_data.get("trials_with_results", 0),
                successes=task_data.get("successes", 0),
                commit=commit,
                metrics={},
            )
            self.append(record)
            records.append(record)
        return records

    def detect_trend(
        self,
        task: str,
        current_rate: float,
        window: int = 5,
        threshold: float = 0.1,
    ) -> TrendResult:
        """Compare *current_rate* against the last *window* runs for *task*.

        A regression is flagged when the current rate is more than *threshold*
        below the average of recent history.
        """
        records = self.load(task=task)

        if not records:
            return TrendResult(
                task=task,
                current_rate=current_rate,
                previous_rate=None,
                delta=None,
                window_size=0,
                regressed=False,
                message=f"No previous history for '{task}' — baseline recorded.",
            )

        recent = records[-window:]
        avg_rate = sum(r.success_rate for r in recent) / len(recent)
        delta = current_rate - avg_rate
        regressed = delta < -threshold

        if regressed:
            msg = (
                f"REGRESSION: '{task}' success rate dropped from "
                f"{avg_rate:.0%} to {current_rate:.0%} (Δ{delta:+.0%})"
            )
        elif delta > threshold:
            msg = (
                f"IMPROVEMENT: '{task}' success rate rose from "
                f"{avg_rate:.0%} to {current_rate:.0%} (Δ{delta:+.0%})"
            )
        else:
            msg = f"'{task}' success rate stable at {current_rate:.0%} (avg {avg_rate:.0%})"

        return TrendResult(
            task=task,
            current_rate=current_rate,
            previous_rate=avg_rate,
            delta=delta,
            window_size=len(recent),
            regressed=regressed,
            message=msg,
        )
