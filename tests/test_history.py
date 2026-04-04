"""Tests for evaluation history store and trend detection."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from roboharness.cli import main, trend_command
from roboharness.storage.history import (
    EvaluationHistory,
    EvaluationRecord,
    TrendResult,
    detect_trend,
)


class TestEvaluationRecord:
    def test_round_trip(self) -> None:
        record = EvaluationRecord(
            task="grasp",
            success_rate=0.8,
            total_trials=10,
            successes=8,
            timestamp=1000.0,
            commit="abc123",
            metrics={"lift_height": 0.15},
        )
        data = record.to_dict()
        restored = EvaluationRecord.from_dict(data)
        assert restored.task == "grasp"
        assert restored.success_rate == 0.8
        assert restored.commit == "abc123"
        assert restored.metrics == {"lift_height": 0.15}

    def test_from_dict_defaults(self) -> None:
        record = EvaluationRecord.from_dict(
            {"task": "t", "success_rate": 0.5, "total_trials": 2, "successes": 1}
        )
        assert record.timestamp == 0.0
        assert record.commit == ""
        assert record.metrics == {}


class TestEvaluationHistory:
    def test_append_and_load(self, tmp_path: Path) -> None:
        history = EvaluationHistory(tmp_path)
        record = EvaluationRecord(task="grasp", success_rate=0.9, total_trials=10, successes=9)
        history.append(record)
        loaded = history.load()
        assert len(loaded) == 1
        assert loaded[0].task == "grasp"
        assert loaded[0].success_rate == 0.9

    def test_append_is_additive(self, tmp_path: Path) -> None:
        history = EvaluationHistory(tmp_path)
        for i in range(3):
            history.append(
                EvaluationRecord(
                    task="grasp", success_rate=0.5 + i * 0.1, total_trials=10, successes=5 + i
                )
            )
        assert len(history.load()) == 3

    def test_load_filter_by_task(self, tmp_path: Path) -> None:
        history = EvaluationHistory(tmp_path)
        history.append(
            EvaluationRecord(task="grasp", success_rate=0.9, total_trials=10, successes=9)
        )
        history.append(
            EvaluationRecord(task="push", success_rate=0.5, total_trials=10, successes=5)
        )
        assert len(history.load(task="grasp")) == 1
        assert len(history.load(task="push")) == 1
        assert len(history.load()) == 2

    def test_load_empty(self, tmp_path: Path) -> None:
        history = EvaluationHistory(tmp_path)
        assert history.load() == []

    def test_jsonl_format(self, tmp_path: Path) -> None:
        history = EvaluationHistory(tmp_path)
        history.append(EvaluationRecord(task="t", success_rate=0.5, total_trials=2, successes=1))
        lines = history.path.read_text().strip().split("\n")
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["task"] == "t"

    def test_record_from_report(self, tmp_path: Path) -> None:
        history = EvaluationHistory(tmp_path)
        report = {
            "tasks": {
                "grasp": {
                    "success_rate": 0.75,
                    "trials_with_results": 4,
                    "successes": 3,
                },
                "push": {
                    "success_rate": None,  # no results — should be skipped
                    "trials_with_results": 0,
                    "successes": 0,
                },
            }
        }
        records = history.record_from_report(report, commit="abc123")
        assert len(records) == 1
        assert records[0].task == "grasp"
        assert records[0].commit == "abc123"
        loaded = history.load()
        assert len(loaded) == 1


class TestDetectTrend:
    def _make_history(self, tmp_path: Path, rates: list[float]) -> EvaluationHistory:
        history = EvaluationHistory(tmp_path)
        for i, rate in enumerate(rates):
            history.append(
                EvaluationRecord(
                    task="grasp",
                    success_rate=rate,
                    total_trials=10,
                    successes=int(rate * 10),
                    timestamp=float(1000 + i),
                )
            )
        return history

    def test_no_history(self, tmp_path: Path) -> None:
        history = EvaluationHistory(tmp_path)
        result = detect_trend(history, "grasp", 0.8)
        assert not result.regressed
        assert result.previous_rate is None
        assert "baseline" in result.message.lower()

    def test_stable(self, tmp_path: Path) -> None:
        history = self._make_history(tmp_path, [0.8, 0.8, 0.8])
        result = detect_trend(history, "grasp", 0.8)
        assert not result.regressed
        assert "stable" in result.message.lower()

    def test_regression(self, tmp_path: Path) -> None:
        history = self._make_history(tmp_path, [0.9, 0.9, 0.9])
        result = detect_trend(history, "grasp", 0.6)
        assert result.regressed
        assert result.delta is not None
        assert result.delta < 0
        assert "regression" in result.message.lower()
        assert "90%" in result.message
        assert "60%" in result.message

    def test_improvement(self, tmp_path: Path) -> None:
        history = self._make_history(tmp_path, [0.5, 0.5, 0.5])
        result = detect_trend(history, "grasp", 0.8)
        assert not result.regressed
        assert "improvement" in result.message.lower()

    def test_window_limits(self, tmp_path: Path) -> None:
        # 10 records but window=3 — only last 3 count
        history = self._make_history(tmp_path, [0.2] * 7 + [0.9, 0.9, 0.9])
        result = detect_trend(history, "grasp", 0.9, window=3)
        assert not result.regressed
        assert result.window_size == 3

    def test_custom_threshold(self, tmp_path: Path) -> None:
        history = self._make_history(tmp_path, [0.8, 0.8, 0.8])
        # Delta of -0.15 is below default 0.1 threshold → regression
        result = detect_trend(history, "grasp", 0.65, threshold=0.2)
        assert not result.regressed  # delta=-0.15 > -0.2, so NOT regressed
        result2 = detect_trend(history, "grasp", 0.65, threshold=0.1)
        assert result2.regressed  # delta=-0.15 < -0.1, so regressed

    def test_trend_result_to_dict(self) -> None:
        tr = TrendResult(
            task="t",
            current_rate=0.5,
            previous_rate=0.8,
            delta=-0.3,
            window_size=3,
            regressed=True,
            message="bad",
        )
        d = tr.to_dict()
        assert d["regressed"] is True
        assert d["delta"] == -0.3


@pytest.fixture
def harness_output_with_results(tmp_path: Path) -> Path:
    """Minimal harness output with results for trend testing."""
    trial_dir = tmp_path / "pick_and_place" / "trial_001"
    cp = trial_dir / "pre_grasp"
    cp.mkdir(parents=True)
    (cp / "metadata.json").write_text(
        json.dumps({"checkpoint": "pre_grasp", "step": 50, "sim_time": 0.5})
    )
    (trial_dir / "result.json").write_text(
        json.dumps(
            {
                "trial_id": 1,
                "success": True,
                "reason": "ok",
                "metrics": {},
                "duration": 1.0,
                "checkpoints_reached": ["pre_grasp"],
                "timestamp": 1000000.0,
            }
        )
    )
    return tmp_path


class TestTrendCommand:
    def test_trend_first_run(self, harness_output_with_results: Path) -> None:
        trends = trend_command(harness_output_with_results)
        assert len(trends) == 1
        assert trends[0]["task"] == "pick_and_place"
        assert "baseline" in trends[0]["message"].lower()
        # Should have written history
        history = EvaluationHistory(harness_output_with_results)
        assert len(history.load()) == 1

    def test_trend_second_run_stable(self, harness_output_with_results: Path) -> None:
        # First run records baseline
        trend_command(harness_output_with_results)
        # Second run should compare against first
        trends = trend_command(harness_output_with_results)
        assert len(trends) == 1
        assert not trends[0]["regressed"]

    def test_trend_cli_subcommand(
        self,
        harness_output_with_results: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        ret = main(["trend", str(harness_output_with_results)])
        assert ret == 0
        captured = capsys.readouterr()
        assert "pick_and_place" in captured.out

    def test_trend_regression_exit_code(self, tmp_path: Path) -> None:
        """Trend subcommand returns exit code 2 on regression."""
        # Build history with high success rate
        history = EvaluationHistory(tmp_path)
        for _ in range(3):
            history.append(
                EvaluationRecord(
                    task="pick_and_place",
                    success_rate=0.9,
                    total_trials=10,
                    successes=9,
                )
            )

        # Create current run with low success rate
        trial_dir = tmp_path / "pick_and_place" / "trial_001"
        cp = trial_dir / "pre_grasp"
        cp.mkdir(parents=True)
        (cp / "metadata.json").write_text(
            json.dumps({"checkpoint": "pre_grasp", "step": 1, "sim_time": 0.1})
        )
        (trial_dir / "result.json").write_text(
            json.dumps(
                {
                    "trial_id": 1,
                    "success": False,
                    "reason": "failed",
                    "metrics": {},
                    "duration": 0.1,
                    "checkpoints_reached": ["pre_grasp"],
                    "timestamp": 2000000.0,
                }
            )
        )

        ret = main(["trend", str(tmp_path)])
        assert ret == 2

    def test_trend_missing_dir(self, tmp_path: Path) -> None:
        ret = main(["trend", str(tmp_path / "nope")])
        assert ret == 1
