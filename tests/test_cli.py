"""Tests for CLI inspect and report commands."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from roboharness.cli import inspect_command, main, report_command


@pytest.fixture
def harness_output(tmp_path: Path) -> Path:
    """Create a realistic harness output directory structure."""
    # task: pick_and_place, trial_001, two checkpoints
    trial_dir = tmp_path / "pick_and_place" / "trial_001"

    # Checkpoint: pre_grasp
    cp1 = trial_dir / "pre_grasp"
    cp1.mkdir(parents=True)
    (cp1 / "front_rgb.png").write_bytes(b"fake-png")
    (cp1 / "side_rgb.png").write_bytes(b"fake-png")
    (cp1 / "state.json").write_text(json.dumps({
        "qpos": [0.1, 0.2, 0.3],
        "qvel": [0.0, 0.0, 0.0],
        "time": 0.5,
    }))
    (cp1 / "metadata.json").write_text(json.dumps({
        "checkpoint": "pre_grasp",
        "step": 50,
        "sim_time": 0.5,
        "timestamp": 1000000.0,
        "cameras": ["front", "side"],
        "files": {"front": {"rgb": "front_rgb.png"}, "side": {"rgb": "side_rgb.png"}},
        "trial": 1,
        "task": "pick_and_place",
    }))

    # Checkpoint: lift
    cp2 = trial_dir / "lift"
    cp2.mkdir(parents=True)
    (cp2 / "front_rgb.png").write_bytes(b"fake-png")
    (cp2 / "state.json").write_text(json.dumps({
        "qpos": [0.5, 0.6, 0.7],
        "qvel": [0.1, 0.0, 0.0],
        "time": 1.2,
    }))
    (cp2 / "metadata.json").write_text(json.dumps({
        "checkpoint": "lift",
        "step": 120,
        "sim_time": 1.2,
        "timestamp": 1000001.0,
        "cameras": ["front"],
        "files": {"front": {"rgb": "front_rgb.png"}},
        "trial": 1,
        "task": "pick_and_place",
    }))

    return tmp_path


@pytest.fixture
def harness_output_with_results(harness_output: Path) -> Path:
    """Extend harness output with trial result files."""
    trial_dir = harness_output / "pick_and_place" / "trial_001"
    (trial_dir / "result.json").write_text(json.dumps({
        "trial_id": 1,
        "success": True,
        "reason": "object lifted above threshold",
        "metrics": {"lift_height": 0.15, "grasp_force": 2.3},
        "duration": 1.2,
        "checkpoints_reached": ["pre_grasp", "lift"],
        "timestamp": 1000001.0,
    }))

    # Add a second failed trial
    trial2 = harness_output / "pick_and_place" / "trial_002"
    cp = trial2 / "pre_grasp"
    cp.mkdir(parents=True)
    (cp / "front_rgb.png").write_bytes(b"fake-png")
    (cp / "state.json").write_text(json.dumps({"qpos": [0.0], "time": 0.3}))
    (cp / "metadata.json").write_text(json.dumps({
        "checkpoint": "pre_grasp",
        "step": 30,
        "sim_time": 0.3,
        "timestamp": 1000002.0,
        "cameras": ["front"],
        "files": {},
        "trial": 2,
        "task": "pick_and_place",
    }))
    (trial2 / "result.json").write_text(json.dumps({
        "trial_id": 2,
        "success": False,
        "reason": "gripper missed object",
        "metrics": {"lift_height": 0.0},
        "duration": 0.3,
        "checkpoints_reached": ["pre_grasp"],
        "timestamp": 1000002.0,
    }))

    return harness_output


class TestInspect:
    def test_nonexistent_dir(self, tmp_path: Path) -> None:
        result = inspect_command(tmp_path / "nope")
        assert "not found" in result

    def test_empty_dir(self, tmp_path: Path) -> None:
        result = inspect_command(tmp_path)
        assert "No captures found" in result

    def test_lists_captures(self, harness_output: Path) -> None:
        result = inspect_command(harness_output)
        assert "pick_and_place" in result
        assert "pre_grasp" in result
        assert "lift" in result
        assert "front_rgb.png" in result
        assert "step=50" in result
        assert "step=120" in result

    def test_shows_state_summary(self, harness_output: Path) -> None:
        result = inspect_command(harness_output)
        assert "qpos" in result

    def test_total_count(self, harness_output: Path) -> None:
        result = inspect_command(harness_output)
        assert "Total: 2 captures" in result


class TestReport:
    def test_nonexistent_dir(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="not found"):
            report_command(tmp_path / "nope")

    def test_empty_dir(self, tmp_path: Path) -> None:
        result = report_command(tmp_path)
        assert result == {"tasks": {}}
        assert (tmp_path / "report.json").exists()

    def test_captures_only(self, harness_output: Path) -> None:
        report = report_command(harness_output)
        task = report["tasks"]["pick_and_place"]
        assert task["total_captures"] == 2
        assert task["total_trials"] == 1
        assert "pre_grasp" in task["checkpoints"]
        assert "lift" in task["checkpoints"]
        assert (harness_output / "report.json").exists()

    def test_with_results(self, harness_output_with_results: Path) -> None:
        report = report_command(harness_output_with_results)
        task = report["tasks"]["pick_and_place"]
        assert task["total_trials"] == 2
        assert task["trials_with_results"] == 2
        assert task["successes"] == 1
        assert task["success_rate"] == 0.5

    def test_report_json_written(self, harness_output_with_results: Path) -> None:
        report_command(harness_output_with_results)
        report_path = harness_output_with_results / "report.json"
        assert report_path.exists()
        loaded = json.loads(report_path.read_text())
        assert "tasks" in loaded


class TestMain:
    def test_no_args(self) -> None:
        assert main([]) == 1

    def test_inspect(self, harness_output: Path, capsys: pytest.CaptureFixture[str]) -> None:
        ret = main(["inspect", str(harness_output)])
        assert ret == 0
        captured = capsys.readouterr()
        assert "pre_grasp" in captured.out

    def test_report(
        self, harness_output_with_results: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        ret = main(["report", str(harness_output_with_results)])
        assert ret == 0
        captured = capsys.readouterr()
        assert "report.json" in captured.out

    def test_report_missing_dir(self, tmp_path: Path) -> None:
        ret = main(["report", str(tmp_path / "nope")])
        assert ret == 1


class TestVariantLayout:
    """Test with TaskStore-style layout: task/variant/trial/checkpoint."""

    @pytest.fixture
    def variant_output(self, tmp_path: Path) -> Path:
        variant_dir = tmp_path / "grasp" / "grasp_position_001" / "trial_001" / "contact"
        variant_dir.mkdir(parents=True)
        (variant_dir / "front_rgb.png").write_bytes(b"fake-png")
        (variant_dir / "state.json").write_text(json.dumps({"qpos": [0.0], "time": 0.8}))
        (variant_dir / "metadata.json").write_text(json.dumps({
            "checkpoint": "contact",
            "step": 80,
            "sim_time": 0.8,
            "cameras": ["front"],
            "files": {},
        }))

        result_dir = tmp_path / "grasp" / "grasp_position_001" / "trial_001"
        (result_dir / "result.json").write_text(json.dumps({
            "trial_id": 1,
            "success": True,
            "reason": "contact detected",
            "metrics": {"force": 5.0},
            "duration": 0.8,
            "checkpoints_reached": ["contact"],
        }))
        return tmp_path

    def test_inspect_variant(self, variant_output: Path) -> None:
        result = inspect_command(variant_output)
        assert "grasp" in result
        assert "contact" in result
        assert "Total: 1 capture" in result

    def test_report_variant(self, variant_output: Path) -> None:
        report = report_command(variant_output)
        task = report["tasks"]["grasp"]
        assert task["successes"] == 1
        assert task["success_rate"] == 1.0
