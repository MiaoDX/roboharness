"""Parallel trial execution for throughput scaling.

Enables concurrent trial execution with isolated simulator instances,
configurable concurrency, and cross-trial result aggregation.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from roboharness.core.harness import SimulatorBackend
from roboharness.storage.task_store import TaskStore, TrialResult

logger = logging.getLogger(__name__)


@dataclass
class TrialSpec:
    """Specification for a single trial to execute.

    Attributes:
        variant_name: Name of the task variant (e.g., "grasp_position_001").
        trial_id: Unique trial identifier within the variant.
        metadata: Arbitrary per-trial configuration passed to the trial function.
    """

    variant_name: str
    trial_id: int
    metadata: dict[str, Any] = field(default_factory=dict)


#: Callable that runs a single trial given an isolated backend and output directory.
#: Returns a TrialResult describing the outcome.
TrialFn = Callable[[SimulatorBackend, Path, TrialSpec], TrialResult]


@dataclass
class BatchResult:
    """Aggregated results across multiple parallel trials."""

    results: list[TrialResult]
    specs: list[TrialSpec]
    total_duration: float

    @property
    def total_trials(self) -> int:
        return len(self.results)

    @property
    def successful_trials(self) -> int:
        return sum(1 for r in self.results if r.success)

    @property
    def failed_trials(self) -> int:
        return self.total_trials - self.successful_trials

    @property
    def success_rate(self) -> float:
        if self.total_trials == 0:
            return 0.0
        return self.successful_trials / self.total_trials

    def per_variant_summary(self) -> dict[str, dict[str, Any]]:
        """Compute per-variant success rates and statistics."""
        variant_results: dict[str, list[TrialResult]] = defaultdict(list)
        for spec, result in zip(self.specs, self.results, strict=True):
            variant_results[spec.variant_name].append(result)

        summaries: dict[str, dict[str, Any]] = {}
        for variant, results in variant_results.items():
            successes = [r for r in results if r.success]
            durations = [r.duration for r in results if r.duration > 0]
            summaries[variant] = {
                "total_trials": len(results),
                "successful_trials": len(successes),
                "success_rate": len(successes) / len(results) if results else 0.0,
                "mean_duration": sum(durations) / len(durations) if durations else 0.0,
            }
        return summaries

    def failure_phase_distribution(self) -> dict[str, int]:
        """Count how many trials failed at each checkpoint phase.

        Uses the last checkpoint reached before failure to determine the phase.
        Trials that failed before reaching any checkpoint are counted under
        "before_first_checkpoint".
        """
        distribution: dict[str, int] = defaultdict(int)
        for result in self.results:
            if result.success:
                continue
            if result.checkpoints_reached:
                last_phase = result.checkpoints_reached[-1]
            else:
                last_phase = "before_first_checkpoint"
            distribution[last_phase] += 1
        return dict(distribution)

    def summary(self) -> dict[str, Any]:
        """Generate a complete summary dict suitable for JSON serialization."""
        return {
            "total_trials": self.total_trials,
            "successful_trials": self.successful_trials,
            "failed_trials": self.failed_trials,
            "success_rate": self.success_rate,
            "total_duration": self.total_duration,
            "per_variant": self.per_variant_summary(),
            "failure_phase_distribution": self.failure_phase_distribution(),
        }


class ParallelTrialRunner:
    """Runs trials concurrently with isolated simulator instances.

    Each trial receives its own ``SimulatorBackend`` (created by the factory)
    and its own output directory (managed by the ``TaskStore``). No mutable
    state is shared between concurrent trials.

    Usage::

        def my_trial(backend, output_dir, spec):
            harness = Harness(backend, output_dir=output_dir)
            harness.add_checkpoint("pre_grasp", cameras=["front"])
            harness.reset()
            result = harness.run_to_next_checkpoint(actions)
            return TrialResult(trial_id=spec.trial_id, success=True)

        runner = ParallelTrialRunner(
            backend_factory=lambda: MyBackend(),
            store=my_store,
            max_workers=4,
        )
        batch = runner.run(specs, trial_fn=my_trial)
        print(batch.success_rate)
    """

    def __init__(
        self,
        backend_factory: Callable[[], SimulatorBackend],
        store: TaskStore,
        max_workers: int = 4,
    ):
        self.backend_factory = backend_factory
        self.store = store
        self.max_workers = max_workers

    def run(self, specs: list[TrialSpec], trial_fn: TrialFn) -> BatchResult:
        """Execute trials in parallel.

        Args:
            specs: List of trial specifications to execute.
            trial_fn: Callable receiving ``(backend, output_dir, spec)`` that
                runs a single trial and returns a ``TrialResult``.

        Returns:
            Aggregated ``BatchResult`` with per-trial and per-variant statistics.
        """
        start = time.monotonic()
        # Maintain insertion order so results[i] corresponds to specs[i].
        results: list[TrialResult | None] = [None] * len(specs)

        def run_one(spec: TrialSpec) -> TrialResult:
            backend = self.backend_factory()
            output_dir = self.store.get_trial_dir(spec.variant_name, spec.trial_id)
            return trial_fn(backend, output_dir, spec)

        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(specs))) as executor:
            future_to_idx = {executor.submit(run_one, spec): i for i, spec in enumerate(specs)}
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                spec = specs[idx]
                try:
                    result = future.result()
                except Exception as exc:
                    logger.error("Trial %s/%d raised: %s", spec.variant_name, spec.trial_id, exc)
                    result = TrialResult(
                        trial_id=spec.trial_id,
                        success=False,
                        reason=f"Unhandled exception: {exc}",
                    )
                results[idx] = result
                self.store.save_trial_result(spec.variant_name, result)

        total_duration = time.monotonic() - start
        # Every slot must be filled — assert rather than silently dropping.
        if not all(r is not None for r in results):
            raise RuntimeError("BUG: unfilled result slot")
        return BatchResult(
            results=[r for r in results if r is not None],  # narrowing for type checker
            specs=list(specs),
            total_duration=total_duration,
        )
