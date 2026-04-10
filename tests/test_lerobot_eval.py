"""Tests for the LeRobot evaluation plugin.

Validates LeRobotEvaluator, EvalReport, EpisodeResult, and the lerobot-eval CLI
using mock environments — no LeRobot or MuJoCo required.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest

gym = pytest.importorskip("gymnasium", reason="gymnasium not installed")

from gymnasium import spaces  # noqa: E402

from roboharness.lerobot.evaluator import (  # noqa: E402
    LEROBOT_EVAL_DEFAULTS,
    LEROBOT_EVAL_PROTOCOL,
    EpisodeResult,
    EvalReport,
    LeRobotEvaluator,
)

# ---------------------------------------------------------------------------
# Mock environments
# ---------------------------------------------------------------------------


class MockEvalEnv(gym.Env):
    """Minimal Gymnasium env for evaluator testing.

    Reports ``success=True`` when cumulative reward exceeds a threshold.
    """

    metadata: ClassVar[dict[str, Any]] = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(
        self,
        render_mode: str = "rgb_array",
        reward_per_step: float = 1.0,
        episode_length: int = 50,
        success_threshold: float = 30.0,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self._reward_per_step = reward_per_step
        self._episode_length = episode_length
        self._success_threshold = success_threshold
        self._step_count = 0
        self._cumulative_reward = 0.0

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed, options=options)
        self._step_count = 0
        self._cumulative_reward = 0.0
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self._step_count += 1
        reward = self._reward_per_step
        self._cumulative_reward += reward
        obs = np.random.default_rng(self._step_count).standard_normal(4).astype(np.float32) * 0.1
        terminated = self._step_count >= self._episode_length
        info: dict[str, Any] = {
            "success": self._cumulative_reward >= self._success_threshold,
        }
        return obs, reward, terminated, False, info

    def render(self) -> np.ndarray:
        return np.zeros((64, 64, 3), dtype=np.uint8)


class MockFailEnv(gym.Env):
    """Env where episodes always fail (success is never True)."""

    metadata: ClassVar[dict[str, Any]] = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(self, render_mode: str = "rgb_array"):
        super().__init__()
        self.render_mode = render_mode
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self._step_count = 0

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed, options=options)
        self._step_count = 0
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self._step_count += 1
        obs = np.zeros(4, dtype=np.float32)
        terminated = self._step_count >= 10
        return obs, -1.0, terminated, False, {"success": False}

    def render(self) -> np.ndarray:
        return np.zeros((64, 64, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Policy fixtures
# ---------------------------------------------------------------------------


def random_policy(obs: Any) -> np.ndarray:
    """Random policy — always returns zeros."""
    return np.zeros(2, dtype=np.float32)


# ---------------------------------------------------------------------------
# EpisodeResult tests
# ---------------------------------------------------------------------------


class TestEpisodeResult:
    def test_to_dict(self) -> None:
        ep = EpisodeResult(
            episode_id=0,
            total_reward=42.0,
            episode_length=100,
            success=True,
            duration=1.5,
            checkpoints_captured=["early", "mid"],
            metrics={"torso_z": 0.75},
        )
        d = ep.to_dict()
        assert d["episode_id"] == 0
        assert d["total_reward"] == 42.0
        assert d["episode_length"] == 100
        assert d["success"] is True
        assert d["duration"] == 1.5
        assert d["checkpoints_captured"] == ["early", "mid"]
        assert d["metrics"] == {"torso_z": 0.75}

    def test_default_fields(self) -> None:
        ep = EpisodeResult(
            episode_id=1, total_reward=0.0, episode_length=10, success=False, duration=0.1
        )
        assert ep.checkpoints_captured == []
        assert ep.metrics == {}


# ---------------------------------------------------------------------------
# EvalReport tests
# ---------------------------------------------------------------------------


class TestEvalReport:
    def _make_report(self, successes: list[bool], rewards: list[float]) -> EvalReport:
        episodes = [
            EpisodeResult(
                episode_id=i,
                total_reward=r,
                episode_length=50,
                success=s,
                duration=0.5,
            )
            for i, (s, r) in enumerate(zip(successes, rewards, strict=True))
        ]
        return EvalReport(episodes=episodes, output_dir="/tmp/test", task_name="test_task")

    def test_empty_report(self) -> None:
        report = EvalReport(episodes=[], output_dir="/tmp/empty", task_name="empty")
        assert report.num_episodes == 0
        assert report.success_rate == 0.0
        assert report.mean_reward == 0.0
        assert report.mean_episode_length == 0.0
        assert report.exit_code(min_success_rate=0.0) == 0

    def test_success_rate(self) -> None:
        report = self._make_report([True, True, False, True], [10.0, 20.0, 5.0, 15.0])
        assert report.num_episodes == 4
        assert report.success_rate == 0.75

    def test_mean_reward(self) -> None:
        report = self._make_report([True, True], [10.0, 30.0])
        assert report.mean_reward == 20.0

    def test_mean_episode_length(self) -> None:
        episodes = [
            EpisodeResult(
                episode_id=i, total_reward=0.0, episode_length=length, success=False, duration=0.1
            )
            for i, length in enumerate([100, 200, 300])
        ]
        report = EvalReport(episodes=episodes, output_dir="/tmp/test", task_name="t")
        assert report.mean_episode_length == 200.0

    def test_exit_code_pass(self) -> None:
        report = self._make_report([True, True, True], [10.0, 10.0, 10.0])
        assert report.exit_code(min_success_rate=0.8) == 0

    def test_exit_code_fail(self) -> None:
        report = self._make_report([True, False, False], [10.0, 0.0, 0.0])
        # success_rate = 1/3 ≈ 0.333, below 0.5
        assert report.exit_code(min_success_rate=0.5) == 1

    def test_autonomous_report_format(self) -> None:
        report = self._make_report([True, False], [10.0, 5.0])
        ar = report.to_autonomous_report()

        assert "summary_metrics" in ar
        assert "snapshot_metrics" in ar
        assert "episodes" in ar
        assert ar["task_name"] == "test_task"

        sm = ar["summary_metrics"]
        assert sm["success_rate"] == 0.5
        assert sm["mean_reward"] == 7.5
        assert sm["num_episodes"] == 2
        assert sm["min_reward"] == 5.0
        assert sm["max_reward"] == 10.0
        assert sm["std_reward"] == pytest.approx(2.5, abs=0.01)

        # snapshot_metrics keys match episode IDs
        assert "episode_0" in ar["snapshot_metrics"]
        assert "episode_1" in ar["snapshot_metrics"]
        assert ar["snapshot_metrics"]["episode_0"]["success"] == 1.0
        assert ar["snapshot_metrics"]["episode_1"]["success"] == 0.0

        # episodes list
        assert len(ar["episodes"]) == 2
        assert ar["episodes"][0]["total_reward"] == 10.0


# ---------------------------------------------------------------------------
# LeRobotEvaluator tests
# ---------------------------------------------------------------------------


class TestLeRobotEvaluator:
    def test_run_successful_episodes(self, tmp_path: Path) -> None:
        env = MockEvalEnv(reward_per_step=1.0, episode_length=50, success_threshold=30.0)
        evaluator = LeRobotEvaluator(
            env=env,
            output_dir=tmp_path / "eval_out",
            task_name="test_eval",
            max_steps=100,
        )
        report = evaluator.run(policy=random_policy, num_episodes=3)

        assert report.num_episodes == 3
        # With reward=1.0/step and 50 steps, cumulative > 30 → success
        assert report.success_rate == 1.0
        assert report.mean_reward == 50.0

        # Report file should be written
        report_path = tmp_path / "eval_out" / "autonomous_report.json"
        assert report_path.exists()
        data = json.loads(report_path.read_text())
        assert data["summary_metrics"]["success_rate"] == 1.0

    def test_run_failing_episodes(self, tmp_path: Path) -> None:
        env = MockFailEnv()
        evaluator = LeRobotEvaluator(
            env=env,
            output_dir=tmp_path / "fail_out",
            task_name="fail_eval",
            max_steps=100,
        )
        report = evaluator.run(policy=random_policy, num_episodes=2)

        assert report.num_episodes == 2
        assert report.success_rate == 0.0
        assert report.mean_reward == -10.0  # -1.0 * 10 steps
        assert report.exit_code(min_success_rate=0.5) == 1

    def test_checkpoint_interval(self, tmp_path: Path) -> None:
        env = MockEvalEnv(reward_per_step=1.0, episode_length=50, success_threshold=30.0)
        evaluator = LeRobotEvaluator(
            env=env,
            output_dir=tmp_path / "cp_out",
            task_name="cp_eval",
            checkpoint_interval=20,
            max_steps=100,
        )
        report = evaluator.run(policy=random_policy, num_episodes=1)
        ep = report.episodes[0]
        # With 50 step episode and interval 20: checkpoints at step 20 and 40
        assert len(ep.checkpoints_captured) == 2
        assert "step_20" in ep.checkpoints_captured
        assert "step_40" in ep.checkpoints_captured

    def test_episode_duration_positive(self, tmp_path: Path) -> None:
        env = MockEvalEnv(episode_length=10)
        evaluator = LeRobotEvaluator(
            env=env,
            output_dir=tmp_path / "dur_out",
            task_name="dur_eval",
            max_steps=100,
        )
        report = evaluator.run(policy=random_policy, num_episodes=1)
        assert report.episodes[0].duration > 0.0

    def test_max_steps_truncation(self, tmp_path: Path) -> None:
        # Env with episode_length=1000 but max_steps=20
        env = MockEvalEnv(episode_length=1000, success_threshold=9999)
        evaluator = LeRobotEvaluator(
            env=env,
            output_dir=tmp_path / "trunc_out",
            task_name="trunc_eval",
            max_steps=20,
        )
        report = evaluator.run(policy=random_policy, num_episodes=1)
        assert report.episodes[0].episode_length == 20

    def test_report_persisted_to_disk(self, tmp_path: Path) -> None:
        env = MockEvalEnv(episode_length=10)
        evaluator = LeRobotEvaluator(
            env=env,
            output_dir=tmp_path / "persist_out",
            task_name="persist_eval",
            max_steps=100,
        )
        evaluator.run(policy=random_policy, num_episodes=2)

        report_path = tmp_path / "persist_out" / "autonomous_report.json"
        assert report_path.exists()
        data = json.loads(report_path.read_text())

        assert data["summary_metrics"]["num_episodes"] == 2
        assert len(data["episodes"]) == 2
        assert "snapshot_metrics" in data

    def test_custom_success_key(self, tmp_path: Path) -> None:
        """Env uses a non-standard key for success."""

        class CustomKeyEnv(gym.Env):
            metadata: ClassVar[dict[str, Any]] = {"render_modes": ["rgb_array"]}

            def __init__(self) -> None:
                super().__init__()
                self.observation_space = spaces.Box(-1, 1, shape=(2,), dtype=np.float32)
                self.action_space = spaces.Box(-1, 1, shape=(1,), dtype=np.float32)
                self._step_count = 0

            def reset(self, *, seed: int | None = None, options: Any = None) -> tuple[Any, dict]:
                super().reset(seed=seed)
                self._step_count = 0
                return np.zeros(2, dtype=np.float32), {}

            def step(self, action: Any) -> tuple[Any, float, bool, bool, dict]:
                self._step_count += 1
                terminated = self._step_count >= 5
                return (
                    np.zeros(2, dtype=np.float32),
                    1.0,
                    terminated,
                    False,
                    {"is_success": True},
                )

            def render(self) -> np.ndarray:
                return np.zeros((32, 32, 3), dtype=np.uint8)

        evaluator = LeRobotEvaluator(
            env=CustomKeyEnv(),
            output_dir=tmp_path / "key_out",
            task_name="key_eval",
            max_steps=100,
            success_key="is_success",
        )
        report = evaluator.run(policy=lambda _: np.zeros(1, dtype=np.float32), num_episodes=1)
        assert report.success_rate == 1.0


# ---------------------------------------------------------------------------
# Built-in protocol and defaults
# ---------------------------------------------------------------------------


class TestProtocolAndDefaults:
    def test_protocol_phases(self) -> None:
        assert LEROBOT_EVAL_PROTOCOL.name == "lerobot_eval"
        names = LEROBOT_EVAL_PROTOCOL.phase_names()
        assert names == ["early", "mid", "final"]

    def test_defaults_tuple(self) -> None:
        assert len(LEROBOT_EVAL_DEFAULTS) == 2
        metrics = {a.metric for a in LEROBOT_EVAL_DEFAULTS}
        assert "success_rate" in metrics
        assert "mean_reward" in metrics


# ---------------------------------------------------------------------------
# CLI integration tests
# ---------------------------------------------------------------------------


class TestLeRobotEvalCLI:
    def _write_report(self, path: Path, success_rate: float, mean_reward: float) -> None:
        data = {
            "task_name": "cli_test",
            "summary_metrics": {
                "success_rate": success_rate,
                "mean_reward": mean_reward,
                "std_reward": 0.0,
                "mean_episode_length": 50.0,
                "std_episode_length": 0.0,
                "num_episodes": 10,
                "min_reward": mean_reward,
                "max_reward": mean_reward,
            },
            "snapshot_metrics": {},
            "episodes": [],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))

    def test_lerobot_eval_pass(self, tmp_path: Path) -> None:
        from roboharness.cli import main

        report_path = tmp_path / "report.json"
        self._write_report(report_path, success_rate=0.8, mean_reward=10.0)

        exit_code = main(["lerobot-eval", str(report_path)])
        assert exit_code == 0

    def test_lerobot_eval_json_output(self, tmp_path: Path) -> None:
        from roboharness.cli import main

        report_path = tmp_path / "report.json"
        self._write_report(report_path, success_rate=1.0, mean_reward=20.0)

        exit_code = main(["lerobot-eval", str(report_path), "--format", "json"])
        assert exit_code == 0

    def test_lerobot_eval_min_success_rate_fail(self, tmp_path: Path) -> None:
        from roboharness.cli import main

        report_path = tmp_path / "report.json"
        self._write_report(report_path, success_rate=0.3, mean_reward=5.0)

        exit_code = main(["lerobot-eval", str(report_path), "--min-success-rate", "0.5"])
        assert exit_code == 1

    def test_lerobot_eval_missing_file(self, tmp_path: Path) -> None:
        from roboharness.cli import main

        exit_code = main(["lerobot-eval", str(tmp_path / "nonexistent.json")])
        assert exit_code == 1

    def test_lerobot_eval_command_function(self, tmp_path: Path) -> None:
        from roboharness.cli import lerobot_eval_command

        report_path = tmp_path / "report.json"
        self._write_report(report_path, success_rate=0.9, mean_reward=15.0)

        result_dict, exit_code = lerobot_eval_command(report_path)
        assert exit_code == 0
        assert result_dict["verdict"] == "pass"
