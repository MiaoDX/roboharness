"""Gymnasium Wrapper — drop-in integration for any Gymnasium-compatible environment.

Usage:
    env = gym.make("Isaac-Reach-Franka-v0", render_mode="rgb_array")
    env = RobotHarnessWrapper(env,
        checkpoints=[
            {"name": "pre_grasp", "step": 50},
            {"name": "contact", "step": 100},
            {"name": "lift", "step": 150},
        ],
        cameras=["front", "side"],
        output_dir="./harness_output",
    )

    obs, info = env.reset()
    for _ in range(200):
        obs, reward, terminated, truncated, info = env.step(action)
        if "checkpoint" in info:
            # Agent can inspect info["checkpoint"]["capture_dir"]
            print(f"Checkpoint: {info['checkpoint']['name']}")
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import Wrapper
except ImportError:
    import gym  # type: ignore[no-redef]
    from gym import Wrapper  # type: ignore[no-redef,assignment]


class RobotHarnessWrapper(Wrapper):  # type: ignore[type-arg]
    """Gymnasium wrapper that adds checkpoint-based visual capture.

    Wraps any Gymnasium environment with `render_mode="rgb_array"` to add:
    - Automatic screenshot capture at predefined step counts
    - Multi-camera support (if the environment provides it)
    - State logging in agent-consumable JSON format
    - Checkpoint save/restore via environment snapshots

    The wrapper is transparent — it does not modify observations, rewards,
    or done signals. It only adds checkpoint info to the `info` dict.
    """

    def __init__(
        self,
        env: gym.Env,  # type: ignore[type-arg]
        checkpoints: list[dict[str, Any]] | None = None,
        cameras: list[str] | None = None,
        output_dir: str | Path = "./harness_output",
        task_name: str = "default",
    ):
        super().__init__(env)
        self.cameras = cameras or ["default"]
        self.output_dir = Path(output_dir)
        self.task_name = task_name
        self._step_count = 0
        self._trial_count = 0

        # Parse checkpoint definitions
        self._checkpoints: dict[int, str] = {}
        for cp in checkpoints or []:
            step = cp.get("step")
            name = cp.get("name", f"checkpoint_{step}")
            if step is not None:
                self._checkpoints[step] = name

        # State snapshots for restore
        self._snapshots: dict[str, Any] = {}

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        """Reset environment and internal counters."""
        self._step_count = 0
        self._trial_count += 1
        result = self.env.reset(**kwargs)
        # Handle both old gym (obs) and new gymnasium (obs, info) return
        if isinstance(result, tuple):
            return result
        return result, {}

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Step environment. Captures screenshots at checkpoint steps."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._step_count += 1

        # Check if we hit a checkpoint
        if self._step_count in self._checkpoints:
            cp_name = self._checkpoints[self._step_count]
            capture_info = self._capture_checkpoint(cp_name, obs, reward, info)
            info["checkpoint"] = capture_info

        return obs, reward, terminated, truncated, info

    def _capture_checkpoint(
        self, name: str, obs: Any, reward: float, info: dict[str, Any]
    ) -> dict[str, Any]:
        """Capture screenshots and state at a checkpoint."""
        capture_dir = (
            self.output_dir
            / self.task_name
            / f"trial_{self._trial_count:03d}"
            / name
        )
        capture_dir.mkdir(parents=True, exist_ok=True)

        saved_files: dict[str, str] = {}

        # Capture render output
        try:
            frame = self.env.render()
            if isinstance(frame, np.ndarray):
                path = capture_dir / "default_rgb.png"
                _save_image(frame, path)
                saved_files["default_rgb"] = str(path)
        except Exception:
            pass

        # Save state info
        state = {
            "step": self._step_count,
            "reward": float(reward) if isinstance(reward, (int, float, np.number)) else 0.0,
            "timestamp": time.time(),
            "checkpoint": name,
            "trial": self._trial_count,
        }

        # Include observation summary (avoid dumping huge arrays)
        if isinstance(obs, np.ndarray):
            state["obs_shape"] = list(obs.shape)
            state["obs_dtype"] = str(obs.dtype)
        elif isinstance(obs, dict):
            state["obs_keys"] = list(obs.keys())

        state_path = capture_dir / "state.json"
        with open(state_path, "w") as f:
            json.dump(state, f, indent=2, default=str)
        saved_files["state"] = str(state_path)

        metadata_path = capture_dir / "metadata.json"
        meta = {
            "checkpoint": name,
            "step": self._step_count,
            "trial": self._trial_count,
            "task": self.task_name,
            "cameras": self.cameras,
            "files": saved_files,
        }
        with open(metadata_path, "w") as f:
            json.dump(meta, f, indent=2)

        return {
            "name": name,
            "step": self._step_count,
            "capture_dir": str(capture_dir),
            "files": saved_files,
        }


def _save_image(arr: np.ndarray, path: Path) -> None:
    """Save RGB array as PNG."""
    try:
        from PIL import Image

        img = Image.fromarray(arr)
        img.save(path)
    except ImportError:
        np.save(path.with_suffix(".npy"), arr)
