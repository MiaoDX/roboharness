"""Tests for parallel trial execution."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np

from roboharness.core.harness import Harness, SimulatorBackend
from roboharness.runner import BatchResult, ParallelTrialRunner, TrialSpec
from roboharness.storage.task_store import TaskStore, TrialResult

from .conftest import MockBackend

# -- TrialSpec tests --


def test_trial_spec_defaults():
    spec = TrialSpec(variant_name="v1", trial_id=1)
    assert spec.variant_name == "v1"
    assert spec.trial_id == 1
    assert spec.metadata == {}


def test_trial_spec_with_metadata():
    spec = TrialSpec(variant_name="v1", trial_id=2, metadata={"key": "value"})
    assert spec.metadata["key"] == "value"


# -- BatchResult tests --


def test_batch_result_empty():
    batch = BatchResult(results=[], specs=[], total_duration=0.0)
    assert batch.total_trials == 0
    assert batch.success_rate == 0.0
    assert batch.failure_phase_distribution() == {}


def test_batch_result_success_rate():
    results = [
        TrialResult(trial_id=1, success=True),
        TrialResult(trial_id=2, success=False, reason="fell"),
        TrialResult(trial_id=3, success=True),
    ]
    specs = [
        TrialSpec(variant_name="v1", trial_id=1),
        TrialSpec(variant_name="v1", trial_id=2),
        TrialSpec(variant_name="v2", trial_id=3),
    ]
    batch = BatchResult(results=results, specs=specs, total_duration=1.0)
    assert batch.total_trials == 3
    assert batch.successful_trials == 2
    assert batch.failed_trials == 1
    assert batch.success_rate == 2 / 3


def test_batch_result_per_variant_summary():
    results = [
        TrialResult(trial_id=1, success=True, duration=1.0),
        TrialResult(trial_id=2, success=False, duration=0.5),
        TrialResult(trial_id=3, success=True, duration=2.0),
    ]
    specs = [
        TrialSpec(variant_name="v1", trial_id=1),
        TrialSpec(variant_name="v1", trial_id=2),
        TrialSpec(variant_name="v2", trial_id=3),
    ]
    batch = BatchResult(results=results, specs=specs, total_duration=3.0)
    summary = batch.per_variant_summary()

    assert summary["v1"]["total_trials"] == 2
    assert summary["v1"]["successful_trials"] == 1
    assert summary["v1"]["success_rate"] == 0.5
    assert summary["v1"]["mean_duration"] == 0.75

    assert summary["v2"]["total_trials"] == 1
    assert summary["v2"]["success_rate"] == 1.0


def test_batch_result_failure_phase_distribution():
    results = [
        TrialResult(trial_id=1, success=False, checkpoints_reached=["pre_grasp", "contact"]),
        TrialResult(trial_id=2, success=False, checkpoints_reached=["pre_grasp"]),
        TrialResult(trial_id=3, success=False, checkpoints_reached=[]),
        TrialResult(trial_id=4, success=True, checkpoints_reached=["pre_grasp", "contact", "lift"]),
    ]
    specs = [TrialSpec(variant_name="v1", trial_id=i) for i in range(1, 5)]
    batch = BatchResult(results=results, specs=specs, total_duration=1.0)
    dist = batch.failure_phase_distribution()

    assert dist["contact"] == 1
    assert dist["pre_grasp"] == 1
    assert dist["before_first_checkpoint"] == 1
    assert "lift" not in dist  # trial 4 succeeded


def test_batch_result_summary_keys():
    batch = BatchResult(
        results=[TrialResult(trial_id=1, success=True)],
        specs=[TrialSpec(variant_name="v1", trial_id=1)],
        total_duration=0.5,
    )
    s = batch.summary()
    assert "total_trials" in s
    assert "success_rate" in s
    assert "per_variant" in s
    assert "failure_phase_distribution" in s
    assert "total_duration" in s


# -- ParallelTrialRunner tests --


def _simple_trial_fn(backend: SimulatorBackend, output_dir: Path, spec: TrialSpec) -> TrialResult:
    """A minimal trial function that runs a few steps and succeeds."""
    backend.reset()
    for _ in range(3):
        backend.step(np.zeros(3))
    return TrialResult(
        trial_id=spec.trial_id,
        success=True,
        reason="completed",
        duration=0.1,
        checkpoints_reached=["phase_a"],
    )


def test_parallel_runner_single_trial(tmp_path: Path):
    store = TaskStore(tmp_path, "test_task")
    runner = ParallelTrialRunner(
        backend_factory=MockBackend,
        store=store,
        max_workers=1,
    )
    specs = [TrialSpec(variant_name="v1", trial_id=1)]
    batch = runner.run(specs, _simple_trial_fn)

    assert batch.total_trials == 1
    assert batch.successful_trials == 1
    assert batch.success_rate == 1.0

    # Verify result was persisted
    result_path = store.get_trial_dir("v1", 1) / "result.json"
    assert result_path.exists()
    with result_path.open() as f:
        data = json.load(f)
    assert data["success"] is True


def test_parallel_runner_multiple_trials(tmp_path: Path):
    store = TaskStore(tmp_path, "test_task")
    runner = ParallelTrialRunner(
        backend_factory=MockBackend,
        store=store,
        max_workers=4,
    )
    specs = [TrialSpec(variant_name=f"v{i}", trial_id=i) for i in range(1, 9)]
    batch = runner.run(specs, _simple_trial_fn)

    assert batch.total_trials == 8
    assert batch.successful_trials == 8


def test_parallel_runner_isolation(tmp_path: Path):
    """Each trial must get its own backend instance."""
    backend_ids: list[int] = []
    lock = threading.Lock()

    def tracking_trial_fn(
        backend: SimulatorBackend, output_dir: Path, spec: TrialSpec
    ) -> TrialResult:
        with lock:
            backend_ids.append(id(backend))
        backend.reset()
        return TrialResult(trial_id=spec.trial_id, success=True)

    store = TaskStore(tmp_path, "test_task")
    runner = ParallelTrialRunner(
        backend_factory=MockBackend,
        store=store,
        max_workers=4,
    )
    specs = [TrialSpec(variant_name="v1", trial_id=i) for i in range(1, 5)]
    runner.run(specs, tracking_trial_fn)

    # All backend instances should be unique objects
    assert len(set(backend_ids)) == 4


def test_parallel_runner_output_dirs_isolated(tmp_path: Path):
    """Each trial must write to its own output directory."""
    seen_dirs: list[Path] = []
    lock = threading.Lock()

    def dir_tracking_fn(
        backend: SimulatorBackend, output_dir: Path, spec: TrialSpec
    ) -> TrialResult:
        with lock:
            seen_dirs.append(output_dir)
        return TrialResult(trial_id=spec.trial_id, success=True)

    store = TaskStore(tmp_path, "test_task")
    runner = ParallelTrialRunner(
        backend_factory=MockBackend,
        store=store,
        max_workers=2,
    )
    specs = [
        TrialSpec(variant_name="v1", trial_id=1),
        TrialSpec(variant_name="v1", trial_id=2),
        TrialSpec(variant_name="v2", trial_id=1),
    ]
    runner.run(specs, dir_tracking_fn)

    assert len(set(seen_dirs)) == 3


def test_parallel_runner_handles_trial_failure(tmp_path: Path):
    """Runner should catch exceptions and record them as failed trials."""

    def failing_trial_fn(
        backend: SimulatorBackend, output_dir: Path, spec: TrialSpec
    ) -> TrialResult:
        if spec.trial_id == 2:
            raise RuntimeError("Simulator crashed")
        return TrialResult(trial_id=spec.trial_id, success=True)

    store = TaskStore(tmp_path, "test_task")
    runner = ParallelTrialRunner(
        backend_factory=MockBackend,
        store=store,
        max_workers=2,
    )
    specs = [
        TrialSpec(variant_name="v1", trial_id=1),
        TrialSpec(variant_name="v1", trial_id=2),
        TrialSpec(variant_name="v1", trial_id=3),
    ]
    batch = runner.run(specs, failing_trial_fn)

    assert batch.total_trials == 3
    assert batch.successful_trials == 2
    assert batch.failed_trials == 1
    # The failed trial should have reason containing the error
    failed = [r for r in batch.results if not r.success]
    assert len(failed) == 1
    assert "Simulator crashed" in failed[0].reason


def test_parallel_runner_preserves_order(tmp_path: Path):
    """Results must correspond to specs by index."""
    store = TaskStore(tmp_path, "test_task")
    runner = ParallelTrialRunner(
        backend_factory=MockBackend,
        store=store,
        max_workers=4,
    )
    specs = [TrialSpec(variant_name="v1", trial_id=i) for i in range(1, 6)]
    batch = runner.run(specs, _simple_trial_fn)

    for spec, result in zip(batch.specs, batch.results, strict=True):
        assert spec.trial_id == result.trial_id


def test_parallel_runner_with_harness(tmp_path: Path):
    """Integration test: trial_fn creates a Harness and runs checkpoints."""

    def harness_trial_fn(
        backend: SimulatorBackend, output_dir: Path, spec: TrialSpec
    ) -> TrialResult:
        harness = Harness(backend=backend, output_dir=output_dir, task_name="run")
        harness.add_checkpoint("phase_a", cameras=["front"])
        harness.reset()
        result = harness.run_to_next_checkpoint([np.zeros(3)] * 5)
        reached = [result.checkpoint_name] if result else []
        return TrialResult(
            trial_id=spec.trial_id,
            success=result is not None,
            checkpoints_reached=reached,
        )

    store = TaskStore(tmp_path, "test_task")
    runner = ParallelTrialRunner(
        backend_factory=MockBackend,
        store=store,
        max_workers=2,
    )
    specs = [
        TrialSpec(variant_name="v1", trial_id=1),
        TrialSpec(variant_name="v1", trial_id=2),
    ]
    batch = runner.run(specs, harness_trial_fn)

    assert batch.total_trials == 2
    assert batch.successful_trials == 2
    for result in batch.results:
        assert result.checkpoints_reached == ["phase_a"]
