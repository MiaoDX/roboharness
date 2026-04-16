"""Tests for the LeRobot evaluation harness example."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# The example is a script with a main() — import it as a module.
from examples.lerobot_eval_harness import _random_policy, main


class TestRandomPolicy:
    def test_uses_action_space_sample(self) -> None:
        action_space = MagicMock()
        action_space.sample.return_value = np.array([1.0, 2.0])
        action = _random_policy(np.zeros(4), action_space)
        assert np.array_equal(action, np.array([1.0, 2.0]))

    def test_fallback_zeros(self) -> None:
        action = _random_policy(np.zeros(4))
        assert np.array_equal(action, np.zeros(2, dtype=np.float32))


class TestMainCartPoleFallback:
    @patch("examples.lerobot_eval_harness.evaluate_policy")
    @patch("gymnasium.make")
    def test_cartpole_fallback(self, mock_gym_make: Any, mock_eval: Any, tmp_path: Path) -> None:
        fake_env = MagicMock()
        fake_env.observation_space = "obs"
        fake_env.action_space = MagicMock()
        fake_env.action_space.sample.return_value = np.zeros(2)
        mock_gym_make.return_value = fake_env

        fake_report = MagicMock()
        fake_report.n_episodes = 2
        fake_report.success_rate = 0.5
        fake_report.mean_reward = 10.0
        fake_report.mean_episode_length = 50.0
        fake_report.wall_time = 1.0
        fake_report.episodes = []
        mock_eval.return_value = fake_report

        with patch("sys.argv", ["lerobot_eval_harness.py", "--output-dir", str(tmp_path)]):
            main()

        mock_gym_make.assert_called_once()
        fake_env.close.assert_called_once()


class TestMainInvalidCheckpoint:
    def test_invalid_checkpoint_exits_one(self, tmp_path: Path) -> None:
        with patch(
            "sys.argv",
            [
                "lerobot_eval_harness.py",
                "--checkpoint-path",
                str(tmp_path / "does_not_exist"),
                "--output-dir",
                str(tmp_path),
            ],
        ):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1


class TestMainThresholdPassFail:
    @patch("examples.lerobot_eval_harness.evaluate_policy")
    @patch("gymnasium.make")
    def test_threshold_pass(self, mock_gym_make: Any, mock_eval: Any, tmp_path: Path) -> None:
        fake_env = MagicMock()
        fake_env.observation_space = "obs"
        fake_env.action_space = MagicMock()
        fake_env.action_space.sample.return_value = np.zeros(2)
        mock_gym_make.return_value = fake_env

        fake_report = MagicMock()
        fake_report.n_episodes = 2
        fake_report.success_rate = 1.0
        fake_report.mean_reward = 10.0
        fake_report.mean_episode_length = 50.0
        fake_report.wall_time = 1.0
        fake_report.episodes = []
        mock_eval.return_value = fake_report

        with patch(
            "sys.argv",
            [
                "lerobot_eval_harness.py",
                "--assert-threshold",
                "--min-success-rate",
                "0.5",
                "--output-dir",
                str(tmp_path),
            ],
        ):
            main()

    @patch("examples.lerobot_eval_harness.evaluate_policy")
    @patch("gymnasium.make")
    def test_threshold_fail_exits_one(
        self, mock_gym_make: Any, mock_eval: Any, tmp_path: Path
    ) -> None:
        fake_env = MagicMock()
        fake_env.observation_space = "obs"
        fake_env.action_space = MagicMock()
        fake_env.action_space.sample.return_value = np.zeros(2)
        mock_gym_make.return_value = fake_env

        fake_report = MagicMock()
        fake_report.n_episodes = 2
        fake_report.success_rate = 0.0
        fake_report.mean_reward = 0.0
        fake_report.mean_episode_length = 10.0
        fake_report.wall_time = 1.0
        fake_report.episodes = []
        mock_eval.return_value = fake_report

        with patch(
            "sys.argv",
            [
                "lerobot_eval_harness.py",
                "--assert-threshold",
                "--min-success-rate",
                "0.5",
                "--output-dir",
                str(tmp_path),
            ],
        ):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1


class TestMainLeRobotPolicy:
    @patch("examples.lerobot_eval_harness.evaluate_lerobot_policy")
    def test_evaluates_lerobot_checkpoint(self, mock_eval: Any, tmp_path: Path) -> None:
        fake_report = MagicMock()
        fake_report.n_episodes = 1
        fake_report.success_rate = 1.0
        fake_report.mean_reward = 5.0
        fake_report.mean_episode_length = 100.0
        fake_report.wall_time = 2.0
        fake_report.episodes = []
        mock_eval.return_value = fake_report

        checkpoint_dir = tmp_path / "checkpoint"
        checkpoint_dir.mkdir()

        with patch(
            "sys.argv",
            [
                "lerobot_eval_harness.py",
                "--checkpoint-path",
                str(checkpoint_dir),
                "--repo-id",
                "lerobot/unitree-g1-mujoco",
                "--output-dir",
                str(tmp_path / "out"),
            ],
        ):
            main()

        mock_eval.assert_called_once()
        call_kwargs = mock_eval.call_args.kwargs
        assert call_kwargs["checkpoint_path"] == str(checkpoint_dir)
        assert call_kwargs["repo_id"] == "lerobot/unitree-g1-mujoco"
