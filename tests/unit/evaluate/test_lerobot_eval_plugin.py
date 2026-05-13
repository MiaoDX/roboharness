"""Tests for the LeRobot evaluation plugin."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from roboharness.evaluate.lerobot_plugin import (
    EpisodeResult,
    LeRobotEvalConfig,
    LeRobotEvalReport,
    check_eval_threshold,
    evaluate_lerobot_policy,
    evaluate_policy,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class FakeEnv:
    """Minimal Gymnasium-like env for testing without gymnasium dependency."""

    def __init__(self, episode_length: int = 20, success_reward: float = 1.0) -> None:
        self.episode_length = episode_length
        self.success_reward = success_reward
        self._step_count = 0
        self._obs = np.zeros(4, dtype=np.float32)

    @property
    def observation_space(self) -> Any:
        return type("Space", (), {"shape": (4,)})()

    @property
    def action_space(self) -> Any:
        return type("Space", (), {"shape": (2,), "sample": lambda: np.zeros(2)})()

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        self._step_count = 0
        self._obs = np.zeros(4, dtype=np.float32)
        return self._obs.copy(), {}

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self._step_count += 1
        self._obs = np.random.default_rng(self._step_count).standard_normal(4).astype(np.float32)
        terminated = self._step_count >= self.episode_length
        reward = self.success_reward if terminated else 0.1
        return self._obs.copy(), reward, terminated, False, {}

    def render(self) -> np.ndarray:
        return np.zeros((64, 64, 3), dtype=np.uint8)

    def close(self) -> None:
        pass


def _constant_policy(obs: np.ndarray) -> np.ndarray:
    """Policy that always returns zeros."""
    return np.zeros(2, dtype=np.float32)


# ---------------------------------------------------------------------------
# EpisodeResult
# ---------------------------------------------------------------------------


class TestEpisodeResult:
    def test_to_dict_contains_required_fields(self) -> None:
        result = EpisodeResult(
            episode_id=0,
            success=True,
            total_reward=5.0,
            episode_length=100,
            checkpoint_dirs=["/tmp/cp1", "/tmp/cp2"],
            metrics={"custom_metric": 0.95},
        )
        d = result.to_dict()
        assert d["episode_id"] == 0
        assert d["success"] is True
        assert d["total_reward"] == 5.0
        assert d["episode_length"] == 100
        assert d["checkpoint_dirs"] == ["/tmp/cp1", "/tmp/cp2"]
        assert d["metrics"]["custom_metric"] == 0.95

    def test_default_values(self) -> None:
        result = EpisodeResult(episode_id=1)
        assert result.success is False
        assert result.total_reward == 0.0
        assert result.episode_length == 0
        assert result.checkpoint_dirs == []
        assert result.metrics == {}


# ---------------------------------------------------------------------------
# LeRobotEvalConfig
# ---------------------------------------------------------------------------


class TestLeRobotEvalConfig:
    def test_defaults(self) -> None:
        config = LeRobotEvalConfig()
        assert config.n_episodes == 10
        assert config.max_steps_per_episode == 1000
        assert config.checkpoint_steps == []
        assert config.success_key == "success"
        assert config.output_dir is None

    def test_custom_values(self) -> None:
        config = LeRobotEvalConfig(
            n_episodes=5,
            max_steps_per_episode=200,
            checkpoint_steps=[10, 50, 100],
            success_key="is_success",
            output_dir="/tmp/eval",
        )
        assert config.n_episodes == 5
        assert config.checkpoint_steps == [10, 50, 100]
        assert config.success_key == "is_success"


# ---------------------------------------------------------------------------
# LeRobotEvalReport
# ---------------------------------------------------------------------------


class TestLeRobotEvalReport:
    def test_success_rate_all_pass(self) -> None:
        episodes = [
            EpisodeResult(episode_id=i, success=True, total_reward=5.0, episode_length=100)
            for i in range(5)
        ]
        report = LeRobotEvalReport(episodes=episodes)
        assert report.success_rate == 1.0
        assert report.n_episodes == 5
        assert report.mean_reward == 5.0

    def test_success_rate_mixed(self) -> None:
        episodes = [
            EpisodeResult(episode_id=0, success=True, total_reward=5.0, episode_length=100),
            EpisodeResult(episode_id=1, success=False, total_reward=2.0, episode_length=50),
            EpisodeResult(episode_id=2, success=True, total_reward=4.0, episode_length=80),
            EpisodeResult(episode_id=3, success=False, total_reward=1.0, episode_length=30),
        ]
        report = LeRobotEvalReport(episodes=episodes)
        assert report.success_rate == 0.5
        assert report.n_episodes == 4
        assert report.mean_reward == 3.0
        assert report.mean_episode_length == 65.0

    def test_success_rate_empty(self) -> None:
        report = LeRobotEvalReport(episodes=[])
        assert report.success_rate == 0.0
        assert report.n_episodes == 0
        assert report.mean_reward == 0.0

    def test_to_dict_structure(self) -> None:
        episodes = [
            EpisodeResult(episode_id=0, success=True, total_reward=5.0, episode_length=100),
            EpisodeResult(episode_id=1, success=False, total_reward=2.0, episode_length=50),
        ]
        report = LeRobotEvalReport(episodes=episodes)
        d = report.to_dict()

        assert d["n_episodes"] == 2
        assert d["success_rate"] == 0.5
        assert d["mean_reward"] == 3.5
        assert d["mean_episode_length"] == 75.0
        assert len(d["episodes"]) == 2
        assert d["episodes"][0]["success"] is True
        assert d["episodes"][1]["success"] is False

    def test_save_json(self, tmp_path: Path) -> None:
        episodes = [
            EpisodeResult(episode_id=0, success=True, total_reward=5.0, episode_length=100),
        ]
        report = LeRobotEvalReport(episodes=episodes)
        out_path = tmp_path / "eval_report.json"
        report.save_json(out_path)

        assert out_path.exists()
        data = json.loads(out_path.read_text())
        assert data["n_episodes"] == 1
        assert data["success_rate"] == 1.0


# ---------------------------------------------------------------------------
# evaluate_policy
# ---------------------------------------------------------------------------


class TestEvaluatePolicy:
    def test_basic_evaluation(self, tmp_path: Path) -> None:
        env = FakeEnv(episode_length=20)
        config = LeRobotEvalConfig(
            n_episodes=3,
            max_steps_per_episode=50,
            output_dir=str(tmp_path),
        )
        report = evaluate_policy(env, _constant_policy, config)

        assert report.n_episodes == 3
        # Each episode runs to completion (terminated at step 20)
        for ep in report.episodes:
            assert ep.episode_length == 20
            assert ep.total_reward > 0

    def test_with_checkpoints(self, tmp_path: Path) -> None:
        env = FakeEnv(episode_length=30)
        config = LeRobotEvalConfig(
            n_episodes=2,
            max_steps_per_episode=50,
            checkpoint_steps=[5, 15, 25],
            output_dir=str(tmp_path),
        )
        report = evaluate_policy(env, _constant_policy, config)

        assert report.n_episodes == 2
        for ep in report.episodes:
            # Should have captured checkpoints at steps 5, 15, 25
            assert len(ep.checkpoint_dirs) == 3

    def test_max_steps_truncation(self, tmp_path: Path) -> None:
        env = FakeEnv(episode_length=1000)  # Very long episode
        config = LeRobotEvalConfig(
            n_episodes=1,
            max_steps_per_episode=15,
            output_dir=str(tmp_path),
        )
        report = evaluate_policy(env, _constant_policy, config)

        assert report.n_episodes == 1
        assert report.episodes[0].episode_length == 15

    def test_success_from_info_key(self, tmp_path: Path) -> None:
        """Success is determined from info dict when success_key is set."""

        class SuccessEnv(FakeEnv):
            def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
                obs, reward, terminated, truncated, info = super().step(action)
                if terminated:
                    info["is_success"] = True
                return obs, reward, terminated, truncated, info

        env = SuccessEnv(episode_length=10)
        config = LeRobotEvalConfig(
            n_episodes=2,
            max_steps_per_episode=20,
            success_key="is_success",
            output_dir=str(tmp_path),
        )
        report = evaluate_policy(env, _constant_policy, config)
        assert all(ep.success for ep in report.episodes)

    def test_no_output_dir(self) -> None:
        """Works without output_dir — no files saved."""
        env = FakeEnv(episode_length=10)
        config = LeRobotEvalConfig(
            n_episodes=1,
            max_steps_per_episode=20,
        )
        report = evaluate_policy(env, _constant_policy, config)
        assert report.n_episodes == 1

    def test_custom_metrics_fn(self, tmp_path: Path) -> None:
        """Custom metrics function adds per-episode metrics."""
        env = FakeEnv(episode_length=10)

        def my_metrics(episode_rewards: list[float], info: dict[str, Any]) -> dict[str, float]:
            return {"max_reward": max(episode_rewards)}

        config = LeRobotEvalConfig(
            n_episodes=1,
            max_steps_per_episode=20,
            output_dir=str(tmp_path),
        )
        report = evaluate_policy(env, _constant_policy, config, metrics_fn=my_metrics)
        assert "max_reward" in report.episodes[0].metrics

    def test_report_json_output(self, tmp_path: Path) -> None:
        """Report JSON is saved to output_dir when specified."""
        env = FakeEnv(episode_length=10)
        config = LeRobotEvalConfig(
            n_episodes=2,
            max_steps_per_episode=20,
            output_dir=str(tmp_path),
        )
        evaluate_policy(env, _constant_policy, config)

        json_path = tmp_path / "lerobot_eval_report.json"
        assert json_path.exists()
        data = json.loads(json_path.read_text())
        assert data["n_episodes"] == 2

    def test_action_shape_validation(self, tmp_path: Path) -> None:
        """Wrong action shape raises ValueError before env.step()."""
        env = FakeEnv(episode_length=10)

        def bad_policy(obs: np.ndarray) -> np.ndarray:
            return np.zeros(5, dtype=np.float32)  # FakeEnv expects shape (2,)

        config = LeRobotEvalConfig(
            n_episodes=1,
            max_steps_per_episode=20,
            output_dir=str(tmp_path),
        )
        with pytest.raises(ValueError, match="Action shape mismatch"):
            evaluate_policy(env, bad_policy, config)

    def test_dict_observation_policy(self, tmp_path: Path) -> None:
        """PolicyAdapter accepting dict observations works without hacks."""

        class DictObsEnv(FakeEnv):
            def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
                self._step_count = 0
                return {"state": self._obs.copy()}, {}

            def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
                self._step_count += 1
                terminated = self._step_count >= self.episode_length
                reward = self.success_reward if terminated else 0.1
                return {"state": self._obs.copy()}, reward, terminated, False, {}

        env = DictObsEnv(episode_length=5)

        def dict_policy(obs: Any) -> np.ndarray:
            assert isinstance(obs, dict)
            return np.zeros(2, dtype=np.float32)

        config = LeRobotEvalConfig(
            n_episodes=1,
            max_steps_per_episode=10,
            output_dir=str(tmp_path),
        )
        report = evaluate_policy(env, dict_policy, config)
        assert report.n_episodes == 1
        assert report.episodes[0].episode_length == 5


# ---------------------------------------------------------------------------
# evaluate_lerobot_policy integration
# ---------------------------------------------------------------------------


class TestEvaluateLerobotPolicy:
    @patch("roboharness.evaluate.lerobot_plugin.load_lerobot_policy")
    @patch("roboharness.evaluate.lerobot_plugin.create_native_env")
    @patch("roboharness.evaluate.lerobot_plugin.evaluate_policy")
    def test_delegates_to_evaluate_policy(
        self,
        mock_eval: Any,
        mock_create_env: Any,
        mock_load_policy: Any,
        tmp_path: Path,
    ) -> None:
        fake_policy = MagicMock()
        fake_env = MagicMock()
        fake_report = MagicMock()
        mock_load_policy.return_value = fake_policy
        mock_create_env.return_value = fake_env
        mock_eval.return_value = fake_report

        config = LeRobotEvalConfig(n_episodes=3)
        report = evaluate_lerobot_policy(
            str(tmp_path),
            repo_id="lerobot/unitree-g1-mujoco",
            config=config,
        )

        assert report is fake_report
        mock_load_policy.assert_called_once_with(str(tmp_path), device="cpu")
        mock_create_env.assert_called_once_with("lerobot/unitree-g1-mujoco")
        mock_eval.assert_called_once_with(fake_env, fake_policy, config=config, metrics_fn=None)

    @patch("roboharness.evaluate.lerobot_plugin.load_lerobot_policy")
    def test_missing_repo_id_raises(self, mock_load_policy: Any, tmp_path: Path) -> None:
        mock_load_policy.return_value = MagicMock()
        with pytest.raises(ValueError, match="Could not infer environment repo_id"):
            evaluate_lerobot_policy(str(tmp_path))


# ---------------------------------------------------------------------------
# check_eval_threshold
# ---------------------------------------------------------------------------


class TestCheckEvalThreshold:
    def test_passes_when_above_threshold(self) -> None:
        episodes = [
            EpisodeResult(episode_id=i, success=True, total_reward=5.0, episode_length=100)
            for i in range(10)
        ]
        report = LeRobotEvalReport(episodes=episodes)
        assert check_eval_threshold(report, min_success_rate=0.8) is True

    def test_fails_when_below_threshold(self) -> None:
        episodes = [
            EpisodeResult(episode_id=0, success=True, total_reward=5.0, episode_length=100),
            EpisodeResult(episode_id=1, success=False, total_reward=1.0, episode_length=50),
        ]
        report = LeRobotEvalReport(episodes=episodes)
        assert check_eval_threshold(report, min_success_rate=0.8) is False

    def test_exact_threshold(self) -> None:
        episodes = [
            EpisodeResult(episode_id=0, success=True, total_reward=5.0, episode_length=100),
            EpisodeResult(episode_id=1, success=False, total_reward=1.0, episode_length=50),
        ]
        report = LeRobotEvalReport(episodes=episodes)
        assert check_eval_threshold(report, min_success_rate=0.5) is True

    def test_min_reward_threshold(self) -> None:
        episodes = [
            EpisodeResult(episode_id=0, success=True, total_reward=5.0, episode_length=100),
            EpisodeResult(episode_id=1, success=True, total_reward=3.0, episode_length=80),
        ]
        report = LeRobotEvalReport(episodes=episodes)
        assert check_eval_threshold(report, min_mean_reward=3.5) is True
        assert check_eval_threshold(report, min_mean_reward=5.0) is False

    def test_combined_thresholds(self) -> None:
        episodes = [
            EpisodeResult(episode_id=0, success=True, total_reward=5.0, episode_length=100),
            EpisodeResult(episode_id=1, success=False, total_reward=1.0, episode_length=50),
        ]
        report = LeRobotEvalReport(episodes=episodes)
        # success_rate=0.5, mean_reward=3.0
        assert check_eval_threshold(report, min_success_rate=0.5, min_mean_reward=2.0) is True
        assert check_eval_threshold(report, min_success_rate=0.5, min_mean_reward=4.0) is False

    def test_empty_report(self) -> None:
        report = LeRobotEvalReport(episodes=[])
        assert check_eval_threshold(report, min_success_rate=0.0) is True
        assert check_eval_threshold(report, min_success_rate=0.5) is False
