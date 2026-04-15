"""LeRobot evaluation plugin — visual regression testing for robot policies.

Wraps LeRobot's evaluation workflow with roboharness visual checkpoints,
structured JSON output, and CI-friendly pass/fail gates.

Usage::

    from roboharness.evaluate.lerobot_plugin import evaluate_policy, check_eval_threshold

    report = evaluate_policy(
        env=env,
        policy_fn=my_policy,
        n_episodes=10,
        config=LeRobotEvalConfig(checkpoint_steps=[10, 50, 100]),
    )

    # CI gate: exit 1 if success rate < 80%
    if not check_eval_threshold(report, min_success_rate=0.8):
        sys.exit(1)
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EpisodeResult:
    """Result of evaluating a single episode.

    Attributes:
        episode_id: Sequential episode index.
        success: Whether the episode was successful.
        total_reward: Cumulative reward over the episode.
        episode_length: Number of steps in the episode.
        checkpoint_dirs: Paths to checkpoint capture directories.
        metrics: Additional per-episode metrics from the metrics function.
    """

    episode_id: int
    success: bool = False
    total_reward: float = 0.0
    episode_length: int = 0
    checkpoint_dirs: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize this episode result to a JSON-compatible dict."""
        return {
            "episode_id": self.episode_id,
            "success": self.success,
            "total_reward": self.total_reward,
            "episode_length": self.episode_length,
            "checkpoint_dirs": self.checkpoint_dirs,
            "metrics": self.metrics,
        }


@dataclass
class LeRobotEvalConfig:
    """Configuration for LeRobot policy evaluation.

    Attributes:
        n_episodes: Number of episodes to run.
        max_steps_per_episode: Maximum steps before truncating an episode.
        checkpoint_steps: Steps at which to capture visual checkpoints.
        success_key: Key in the info dict that indicates episode success.
        output_dir: Directory for saving evaluation output. None to skip file output.
    """

    n_episodes: int = 10
    max_steps_per_episode: int = 1000
    checkpoint_steps: list[int] = field(default_factory=list)
    success_key: str = "success"
    output_dir: str | None = None


@dataclass
class LeRobotEvalReport:
    """Aggregated evaluation report across all episodes.

    Attributes:
        episodes: Per-episode results.
        wall_time: Total wall-clock time for evaluation in seconds.
    """

    episodes: list[EpisodeResult] = field(default_factory=list)
    wall_time: float = 0.0

    @property
    def n_episodes(self) -> int:
        """Number of evaluated episodes."""
        return len(self.episodes)

    @property
    def success_rate(self) -> float:
        """Fraction of episodes that succeeded (0.0 if no episodes)."""
        if not self.episodes:
            return 0.0
        return sum(1 for ep in self.episodes if ep.success) / len(self.episodes)

    @property
    def mean_reward(self) -> float:
        """Mean cumulative reward across all episodes (0.0 if no episodes)."""
        if not self.episodes:
            return 0.0
        return sum(ep.total_reward for ep in self.episodes) / len(self.episodes)

    @property
    def mean_episode_length(self) -> float:
        """Mean number of steps per episode (0.0 if no episodes)."""
        if not self.episodes:
            return 0.0
        return sum(ep.episode_length for ep in self.episodes) / len(self.episodes)

    def to_dict(self) -> dict[str, Any]:
        """Serialize this evaluation report to a JSON-compatible dict."""
        return {
            "n_episodes": self.n_episodes,
            "success_rate": self.success_rate,
            "mean_reward": self.mean_reward,
            "mean_episode_length": self.mean_episode_length,
            "wall_time": self.wall_time,
            "episodes": [ep.to_dict() for ep in self.episodes],
        }

    def save_json(self, path: Path) -> None:
        """Save the report as JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(self.to_dict(), f, indent=2)


#: Type alias for a policy function: obs → action.
PolicyFn = Callable[[np.ndarray], np.ndarray]

#: Type alias for a custom metrics function: (episode_rewards, last_info) → metrics dict.
MetricsFn = Callable[[list[float], dict[str, Any]], dict[str, float]]


def _save_checkpoint_screenshot(
    env: Any,
    checkpoint_name: str,
    episode_dir: Path,
) -> str | None:
    """Capture and save a screenshot at the current env state.

    Returns the checkpoint directory path, or None if capture failed.
    """
    cp_dir = episode_dir / checkpoint_name
    cp_dir.mkdir(parents=True, exist_ok=True)

    try:
        frame = env.render()
    except Exception:
        logger.debug("Failed to render frame for checkpoint '%s'", checkpoint_name, exc_info=True)
        return str(cp_dir)

    if frame is not None and isinstance(frame, np.ndarray):
        from roboharness._utils import save_image

        save_image(frame, cp_dir / "default_rgb.png")

    return str(cp_dir)


def evaluate_policy(
    env: Any,
    policy_fn: PolicyFn,
    config: LeRobotEvalConfig | None = None,
    *,
    metrics_fn: MetricsFn | None = None,
) -> LeRobotEvalReport:
    """Evaluate a policy on an environment with optional visual checkpoints.

    Runs the policy for ``config.n_episodes`` episodes, capturing checkpoint
    screenshots at the configured steps. Produces a structured evaluation report.

    Args:
        env: A Gymnasium-compatible environment.
        policy_fn: Callable that maps observation to action.
        config: Evaluation configuration. Uses defaults if None.
        metrics_fn: Optional function to compute custom per-episode metrics.

    Returns:
        Aggregated evaluation report.
    """
    if config is None:
        config = LeRobotEvalConfig()

    output_dir = Path(config.output_dir) if config.output_dir else None
    checkpoint_set = set(config.checkpoint_steps)
    episodes: list[EpisodeResult] = []

    start_time = time.monotonic()

    for ep_idx in range(config.n_episodes):
        ep_dir = output_dir / f"episode_{ep_idx:03d}" if output_dir else None
        obs, _info = env.reset()

        total_reward = 0.0
        episode_rewards: list[float] = []
        checkpoint_dirs: list[str] = []
        last_info: dict[str, Any] = {}
        step_count = 0
        success = False

        for step in range(1, config.max_steps_per_episode + 1):
            action = policy_fn(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            step_count = step

            reward_val = float(reward) if not isinstance(reward, float) else reward
            total_reward += reward_val
            episode_rewards.append(reward_val)
            last_info = info

            # Capture checkpoint screenshot
            if step in checkpoint_set and ep_dir is not None:
                cp_name = f"step_{step:04d}"
                cp_dir = _save_checkpoint_screenshot(env, cp_name, ep_dir)
                if cp_dir is not None:
                    checkpoint_dirs.append(cp_dir)

            if terminated or truncated:
                # Check for success in info dict
                if config.success_key in info:
                    success = bool(info[config.success_key])
                break

        # Compute custom metrics
        custom_metrics: dict[str, float] = {}
        if metrics_fn is not None:
            custom_metrics = metrics_fn(episode_rewards, last_info)

        episodes.append(
            EpisodeResult(
                episode_id=ep_idx,
                success=success,
                total_reward=total_reward,
                episode_length=step_count,
                checkpoint_dirs=checkpoint_dirs,
                metrics=custom_metrics,
            )
        )

        logger.info(
            "Episode %d: success=%s, reward=%.2f, length=%d",
            ep_idx,
            success,
            total_reward,
            step_count,
        )

    wall_time = time.monotonic() - start_time
    report = LeRobotEvalReport(episodes=episodes, wall_time=wall_time)

    # Save report JSON
    if output_dir is not None:
        report.save_json(output_dir / "lerobot_eval_report.json")
        logger.info("Saved evaluation report to %s", output_dir / "lerobot_eval_report.json")

    return report


def check_eval_threshold(
    report: LeRobotEvalReport,
    *,
    min_success_rate: float = 0.0,
    min_mean_reward: float | None = None,
) -> bool:
    """Check if the evaluation report meets minimum thresholds.

    CI-friendly gate: returns True if all thresholds are met, False otherwise.

    Args:
        report: Evaluation report to check.
        min_success_rate: Minimum required success rate (0.0 to 1.0).
        min_mean_reward: Minimum required mean reward (None to skip check).

    Returns:
        True if all thresholds are met.
    """
    if report.n_episodes == 0:
        # Empty report: only passes if threshold is zero
        return min_success_rate <= 0.0

    if report.success_rate < min_success_rate:
        logger.warning(
            "Success rate %.1f%% below threshold %.1f%%",
            report.success_rate * 100,
            min_success_rate * 100,
        )
        return False

    if min_mean_reward is not None and report.mean_reward < min_mean_reward:
        logger.warning(
            "Mean reward %.2f below threshold %.2f",
            report.mean_reward,
            min_mean_reward,
        )
        return False

    return True
