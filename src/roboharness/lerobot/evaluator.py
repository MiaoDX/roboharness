"""LeRobot evaluation plugin — run policy episodes with visual checkpoint capture.

Wraps any Gymnasium-compatible environment with ``RobotHarnessWrapper``, runs
evaluation episodes using a user-supplied policy, and produces a JSON report
compatible with ``roboharness evaluate``.

Usage::

    from roboharness.lerobot import LeRobotEvaluator

    evaluator = LeRobotEvaluator(
        env=my_env,
        output_dir="./eval_output",
        task_name="g1_locomotion",
    )
    report = evaluator.run(policy=my_policy, num_episodes=10)
    print(f"Success rate: {report.success_rate:.0%}")
    sys.exit(report.exit_code(min_success_rate=0.8))
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from roboharness._utils import save_json
from roboharness.core.protocol import TaskPhase, TaskProtocol
from roboharness.evaluate.assertions import MetricAssertion
from roboharness.evaluate.result import Operator, Severity

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Built-in protocol for LeRobot evaluation
# ---------------------------------------------------------------------------

LEROBOT_EVAL_PROTOCOL = TaskProtocol(
    name="lerobot_eval",
    description="LeRobot policy evaluation with early/mid/final snapshots",
    phases=[
        TaskPhase("early", "Early episode behaviour after reset"),
        TaskPhase("mid", "Mid-episode policy performance"),
        TaskPhase("final", "End-of-episode outcome"),
    ],
)

# Default constraints for LeRobot evaluation (reasonable baselines)
LEROBOT_EVAL_DEFAULTS: tuple[MetricAssertion, ...] = (
    MetricAssertion(
        metric="success_rate",
        operator=Operator.GE,
        threshold=0.0,
        severity=Severity.CRITICAL,
        phase="*",
    ),
    MetricAssertion(
        metric="mean_reward",
        operator=Operator.GE,
        threshold=0.0,
        severity=Severity.MAJOR,
        phase="*",
    ),
)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class EpisodeResult:
    """Metrics from a single evaluation episode."""

    episode_id: int
    total_reward: float
    episode_length: int
    success: bool
    duration: float
    checkpoints_captured: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "total_reward": self.total_reward,
            "episode_length": self.episode_length,
            "success": self.success,
            "duration": self.duration,
            "checkpoints_captured": self.checkpoints_captured,
            "metrics": self.metrics,
        }


@dataclass
class EvalReport:
    """Aggregate evaluation report across episodes."""

    episodes: list[EpisodeResult]
    output_dir: str
    task_name: str

    @property
    def num_episodes(self) -> int:
        return len(self.episodes)

    @property
    def success_rate(self) -> float:
        if not self.episodes:
            return 0.0
        return sum(1 for e in self.episodes if e.success) / len(self.episodes)

    @property
    def mean_reward(self) -> float:
        if not self.episodes:
            return 0.0
        return sum(e.total_reward for e in self.episodes) / len(self.episodes)

    @property
    def mean_episode_length(self) -> float:
        if not self.episodes:
            return 0.0
        return sum(e.episode_length for e in self.episodes) / len(self.episodes)

    def to_autonomous_report(self) -> dict[str, Any]:
        """Convert to the standard ``autonomous_report.json`` format.

        The returned dict is compatible with ``roboharness evaluate``:
        it contains ``summary_metrics``, ``snapshot_metrics``, and ``episodes``.
        """
        rewards = [e.total_reward for e in self.episodes]
        lengths = [float(e.episode_length) for e in self.episodes]

        return {
            "task_name": self.task_name,
            "summary_metrics": {
                "success_rate": self.success_rate,
                "mean_reward": self.mean_reward,
                "std_reward": float(np.std(rewards)) if rewards else 0.0,
                "mean_episode_length": self.mean_episode_length,
                "std_episode_length": float(np.std(lengths)) if lengths else 0.0,
                "num_episodes": self.num_episodes,
                "min_reward": float(min(rewards)) if rewards else 0.0,
                "max_reward": float(max(rewards)) if rewards else 0.0,
            },
            "snapshot_metrics": {
                f"episode_{e.episode_id}": {
                    "reward": e.total_reward,
                    "length": float(e.episode_length),
                    "success": 1.0 if e.success else 0.0,
                    **e.metrics,
                }
                for e in self.episodes
            },
            "episodes": [e.to_dict() for e in self.episodes],
        }

    def exit_code(self, min_success_rate: float = 0.0) -> int:
        """CI-compatible exit code: ``0`` = pass, ``1`` = fail."""
        return 0 if self.success_rate >= min_success_rate else 1


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

#: Policy callable signature: ``(observation) -> action``.
PolicyFn = Callable[[Any], Any]


class LeRobotEvaluator:
    """Evaluate a policy on a Gymnasium env with visual checkpoint capture.

    Wraps the environment with :class:`~roboharness.wrappers.RobotHarnessWrapper`,
    runs *N* episodes using the given policy, collects per-episode metrics, and
    writes an ``autonomous_report.json`` that the existing ``roboharness evaluate``
    CLI can consume.

    Parameters
    ----------
    env:
        A Gymnasium-compatible environment (e.g. from ``lerobot.envs.factory.make_env``).
    output_dir:
        Directory where captures and the report are written.
    task_name:
        Human-readable task identifier used in report output.
    cameras:
        Camera names to capture (passed to ``RobotHarnessWrapper``).
    checkpoint_interval:
        If set, capture a checkpoint every *N* steps (mutually exclusive with *protocol*).
    protocol:
        A ``TaskProtocol`` defining semantic capture phases.
    phase_steps:
        Step numbers for each protocol phase (required when *protocol* is set).
    max_steps:
        Maximum steps per episode before truncation.
    success_key:
        Key in ``info`` dict that indicates episode success.
    auto_fix_obs_space:
        Whether to auto-correct observation space mismatches (Isaac Lab / LeRobot compat).
    """

    def __init__(
        self,
        env: Any,
        output_dir: str | Path = "./lerobot_eval",
        task_name: str = "lerobot_eval",
        cameras: list[str] | None = None,
        checkpoint_interval: int | None = None,
        protocol: TaskProtocol | None = None,
        phase_steps: dict[str, int] | None = None,
        max_steps: int = 1000,
        success_key: str = "success",
        auto_fix_obs_space: bool = True,
    ):
        self.env = env
        self.output_dir = Path(output_dir)
        self.task_name = task_name
        self.cameras = cameras
        self.checkpoint_interval = checkpoint_interval
        self.protocol = protocol
        self.phase_steps = phase_steps
        self.max_steps = max_steps
        self.success_key = success_key
        self.auto_fix_obs_space = auto_fix_obs_space

    def _wrap_env(self) -> Any:
        """Wrap the environment with ``RobotHarnessWrapper``."""
        from roboharness.wrappers import RobotHarnessWrapper

        # Build checkpoint list from interval if no protocol is given
        checkpoints = None
        if self.protocol is None and self.checkpoint_interval:
            checkpoints = [
                {"name": f"step_{step}", "step": step}
                for step in range(
                    self.checkpoint_interval,
                    self.max_steps + 1,
                    self.checkpoint_interval,
                )
            ]

        return RobotHarnessWrapper(
            self.env,
            checkpoints=checkpoints,
            cameras=self.cameras,
            output_dir=str(self.output_dir),
            task_name=self.task_name,
            protocol=self.protocol,
            phase_steps=self.phase_steps,
            auto_fix_obs_space=self.auto_fix_obs_space,
        )

    def _run_episode(
        self,
        wrapped_env: Any,
        policy: PolicyFn,
        episode_id: int,
    ) -> EpisodeResult:
        """Run a single evaluation episode and collect metrics."""
        start_time = time.monotonic()
        obs, _info = wrapped_env.reset()

        total_reward = 0.0
        checkpoints_captured: list[str] = []
        episode_success = False
        step_count = 0

        for _step in range(self.max_steps):
            action = policy(obs)
            obs, reward, terminated, truncated, info = wrapped_env.step(action)

            # Handle numpy scalars and tensors
            if hasattr(reward, "item"):
                reward = reward.item()
            total_reward += float(reward)
            step_count += 1

            if "checkpoint" in info:
                cp_name = info["checkpoint"].get("name", f"step_{_step}")
                checkpoints_captured.append(cp_name)

            # Check for success flag in info
            if self.success_key in info:
                val = info[self.success_key]
                if hasattr(val, "item"):
                    val = val.item()
                episode_success = bool(val)

            if terminated or truncated:
                break

        duration = time.monotonic() - start_time

        return EpisodeResult(
            episode_id=episode_id,
            total_reward=total_reward,
            episode_length=step_count,
            success=episode_success,
            duration=duration,
            checkpoints_captured=checkpoints_captured,
        )

    def run(
        self,
        policy: PolicyFn,
        num_episodes: int = 10,
        min_success_rate: float = 0.0,
    ) -> EvalReport:
        """Run evaluation episodes and produce a report.

        Parameters
        ----------
        policy:
            Callable ``(obs) -> action``.
        num_episodes:
            Number of episodes to evaluate.
        min_success_rate:
            Minimum success rate for :meth:`EvalReport.exit_code` (0.0-1.0).

        Returns
        -------
        EvalReport
            Aggregate results with per-episode detail.  The report is also
            written to ``output_dir/autonomous_report.json``.
        """
        wrapped_env = self._wrap_env()
        episodes: list[EpisodeResult] = []

        for ep_id in range(num_episodes):
            logger.info("Episode %d/%d", ep_id + 1, num_episodes)
            result = self._run_episode(wrapped_env, policy, ep_id)
            episodes.append(result)
            logger.info(
                "  reward=%.2f  length=%d  success=%s",
                result.total_reward,
                result.episode_length,
                result.success,
            )

        report = EvalReport(
            episodes=episodes,
            output_dir=str(self.output_dir),
            task_name=self.task_name,
        )

        # Persist the report
        self.output_dir.mkdir(parents=True, exist_ok=True)
        report_path = self.output_dir / "autonomous_report.json"
        save_json(report.to_autonomous_report(), report_path)
        logger.info("Report saved to %s", report_path)

        return report
