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

Multi-camera support:
    The wrapper automatically detects multi-camera environments and captures
    from all configured cameras at each checkpoint. Detection checks for:

    1. ``render_camera(camera_name)`` method on the environment
    2. Isaac Lab ``TiledCamera`` sensors via ``env.unwrapped.scene``
    3. Falls back to ``env.render()`` for the default camera

    If cameras=["front", "wrist"] is passed but the environment only supports
    single-view rendering, the wrapper captures one frame labeled "default".
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, cast

import numpy as np

from roboharness._utils import save_image as _save_image
from roboharness.core.protocol import TaskProtocol

try:
    import gymnasium as gym
    from gymnasium import Wrapper
except ImportError:
    import gym  # type: ignore[no-redef]
    from gym import Wrapper  # type: ignore[no-redef,assignment]

logger = logging.getLogger(__name__)


class MultiCameraCapability:
    """Detected multi-camera capability of an environment."""

    NONE = "none"  # Only env.render() — single view
    RENDER_CAMERA = "render_camera"  # env.render_camera(name) method
    ISAAC_TILED = "isaac_tiled"  # Isaac Lab TiledCamera via scene


def _detect_camera_capability(env: gym.Env) -> str:  # type: ignore[type-arg]
    """Detect how the environment supports camera rendering.

    Checks (in priority order):
    1. ``render_camera(camera_name)`` method on env or env.unwrapped
    2. Isaac Lab scene with camera sensors (TiledCamera)
    3. Falls back to NONE (single env.render())
    """
    # Check for render_camera method on the env or unwrapped env
    for target in (env, getattr(env, "unwrapped", None)):
        if target is not None and callable(getattr(target, "render_camera", None)):
            return MultiCameraCapability.RENDER_CAMERA

    # Check for Isaac Lab TiledCamera via scene attribute
    unwrapped = getattr(env, "unwrapped", env)
    scene = getattr(unwrapped, "scene", None)
    if scene is not None:
        # Isaac Lab scenes expose sensors as dict-like or attribute access
        if callable(getattr(scene, "keys", None)):
            for key in scene:
                sensor = scene[key]
                type_name = type(sensor).__name__
                if "Camera" in type_name:
                    return MultiCameraCapability.ISAAC_TILED
        elif callable(getattr(scene, "__iter__", None)):
            for sensor in scene:
                type_name = type(sensor).__name__
                if "Camera" in type_name:
                    return MultiCameraCapability.ISAAC_TILED

    return MultiCameraCapability.NONE


def _capture_frame_from_env(
    env: gym.Env,  # type: ignore[type-arg]
    camera_name: str,
    capability: str,
) -> np.ndarray | None:
    """Capture a single frame from the environment for the given camera.

    Returns an RGB numpy array (H, W, 3) or None if capture failed.
    """
    try:
        if capability == MultiCameraCapability.RENDER_CAMERA:
            # Try env first, then unwrapped
            for target in (env, getattr(env, "unwrapped", None)):
                if target is not None and callable(getattr(target, "render_camera", None)):
                    frame = target.render_camera(camera_name)
                    return _to_numpy_rgb(frame)

        if capability == MultiCameraCapability.ISAAC_TILED:
            unwrapped = getattr(env, "unwrapped", env)
            scene = getattr(unwrapped, "scene", None)
            if scene is not None and camera_name in scene:
                sensor = scene[camera_name]
                # Isaac Lab cameras expose .data.output["rgb"] or similar
                data = getattr(sensor, "data", None)
                if data is not None:
                    output = getattr(data, "output", None)
                    if isinstance(output, dict) and "rgb" in output:
                        return _to_numpy_rgb(output["rgb"])

        # Fallback: use env.render() for the default camera
        frame = env.render()
        return _to_numpy_rgb(frame)

    except Exception:
        logger.debug("Failed to capture frame for camera '%s'", camera_name, exc_info=True)
        return None


def _to_numpy_rgb(frame: Any) -> np.ndarray | None:
    """Convert a frame to a numpy RGB array, handling torch tensors."""
    if frame is None:
        return None
    if isinstance(frame, np.ndarray):
        return frame
    # Handle torch tensors and similar array-like objects
    if hasattr(frame, "cpu") and hasattr(frame, "numpy"):
        arr = cast("np.ndarray", frame.detach().cpu().numpy())
        if arr.dtype != np.uint8:
            arr = (
                np.clip(arr * 255, 0, 255).astype(np.uint8)
                if arr.max() <= 1.0
                else arr.astype(np.uint8)
            )
        return arr
    return None


class RobotHarnessWrapper(Wrapper):  # type: ignore[type-arg]
    """Gymnasium wrapper that adds checkpoint-based visual capture.

    Wraps any Gymnasium environment with `render_mode="rgb_array"` to add:
    - Automatic screenshot capture at predefined step counts
    - Multi-camera support (if the environment provides it)
    - State logging in agent-consumable JSON format
    - Checkpoint save/restore via environment snapshots

    The wrapper is transparent — it does not modify observations, rewards,
    or done signals. It only adds checkpoint info to the `info` dict.

    Multi-camera detection:
        The wrapper automatically detects whether the wrapped environment
        supports named cameras. It checks for:

        1. A ``render_camera(camera_name)`` method on the env
        2. Isaac Lab ``TiledCamera`` sensors via ``env.unwrapped.scene``
        3. Falls back to ``env.render()`` for single-view capture

        The detected capability is available via the ``camera_capability``
        attribute.
    """

    def __init__(
        self,
        env: gym.Env,  # type: ignore[type-arg]
        checkpoints: list[dict[str, Any]] | None = None,
        cameras: list[str] | None = None,
        output_dir: str | Path = "./harness_output",
        task_name: str = "default",
        protocol: TaskProtocol | None = None,
        phase_steps: dict[str, int] | None = None,
    ):
        super().__init__(env)
        self.output_dir = Path(output_dir)
        self.task_name = task_name
        self._step_count = 0
        self._trial_count = 0
        self._active_protocol: TaskProtocol | None = None

        # Detect multi-camera capability
        self.camera_capability = _detect_camera_capability(env)
        if self.camera_capability != MultiCameraCapability.NONE:
            logger.info(
                "Multi-camera support detected: %s (cameras: %s)",
                self.camera_capability,
                cameras,
            )

        # Build checkpoints from protocol or raw dicts
        self._checkpoints: dict[int, str] = {}
        if protocol is not None:
            self._active_protocol = protocol
            if phase_steps is None:
                raise ValueError("phase_steps is required when protocol is provided")
            for phase in protocol.phases:
                step = phase_steps.get(phase.name)
                if step is not None:
                    self._checkpoints[step] = phase.name
            # Derive cameras from protocol phases unless explicitly overridden
            if cameras is None:
                seen: set[str] = set()
                cams: list[str] = []
                for phase in protocol.phases:
                    for cam in phase.cameras:
                        if cam not in seen:
                            seen.add(cam)
                            cams.append(cam)
                self.cameras = cams
            else:
                self.cameras = cameras
        else:
            self.cameras = cameras or ["default"]
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

    @property
    def active_protocol(self) -> TaskProtocol | None:
        """The currently loaded task protocol, or None."""
        return self._active_protocol

    @property
    def has_multi_camera(self) -> bool:
        """Whether the environment supports named multi-camera rendering."""
        return self.camera_capability != MultiCameraCapability.NONE

    def _capture_checkpoint(
        self, name: str, obs: Any, reward: float, info: dict[str, Any]
    ) -> dict[str, Any]:
        """Capture screenshots and state at a checkpoint.

        Iterates over all configured cameras and captures a frame from each.
        For environments without multi-camera support, captures a single frame
        from ``env.render()`` labeled as ``"default"``.
        """
        capture_dir = self.output_dir / self.task_name / f"trial_{self._trial_count:03d}" / name
        capture_dir.mkdir(parents=True, exist_ok=True)

        saved_files: dict[str, str] = {}
        captured_cameras: list[str] = []

        if self.camera_capability == MultiCameraCapability.NONE:
            # Single-view fallback: one call to env.render()
            frame = _capture_frame_from_env(self.env, "default", MultiCameraCapability.NONE)
            if frame is not None:
                path = capture_dir / "default_rgb.png"
                _save_image(frame, path)
                saved_files["default_rgb"] = str(path)
                captured_cameras.append("default")
        else:
            # Multi-camera: capture from each configured camera
            for camera_name in self.cameras:
                frame = _capture_frame_from_env(self.env, camera_name, self.camera_capability)
                if frame is not None:
                    path = capture_dir / f"{camera_name}_rgb.png"
                    _save_image(frame, path)
                    saved_files[f"{camera_name}_rgb"] = str(path)
                    captured_cameras.append(camera_name)

        # Save state info
        state = {
            "step": self._step_count,
            "reward": _to_float(reward),
            "timestamp": time.time(),
            "checkpoint": name,
            "trial": self._trial_count,
        }

        # Include observation summary (avoid dumping huge arrays)
        if isinstance(obs, np.ndarray):
            state["obs_shape"] = list(obs.shape)
            state["obs_dtype"] = str(obs.dtype)
        elif hasattr(obs, "shape") and hasattr(obs, "dtype"):
            # Torch tensors and similar array-like objects
            state["obs_shape"] = list(obs.shape)
            state["obs_dtype"] = str(obs.dtype)
        elif isinstance(obs, dict):
            state["obs_keys"] = list(obs.keys())

        state_path = capture_dir / "state.json"
        with state_path.open("w") as f:
            json.dump(state, f, indent=2, default=str)
        saved_files["state"] = str(state_path)

        metadata_path = capture_dir / "metadata.json"
        meta = {
            "checkpoint": name,
            "step": self._step_count,
            "trial": self._trial_count,
            "task": self.task_name,
            "cameras": captured_cameras,
            "camera_capability": self.camera_capability,
            "files": saved_files,
        }
        with metadata_path.open("w") as f:
            json.dump(meta, f, indent=2)

        return {
            "name": name,
            "step": self._step_count,
            "capture_dir": str(capture_dir),
            "files": saved_files,
        }


def _to_float(value: Any) -> float:
    """Convert a scalar value to float, handling torch tensors and numpy types."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, np.number):
        return float(value)
    # Handle torch tensors and similar array-like objects
    if hasattr(value, "item"):
        # Multi-element tensors (e.g. vectorized envs): take the mean
        if hasattr(value, "numel") and value.numel() > 1:
            return float(value.float().mean().item())
        if hasattr(value, "size") and isinstance(value.size, int) and value.size > 1:
            return float(value.mean())
        return float(value.item())
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
