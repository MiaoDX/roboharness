"""Shared LeRobot environment utilities for native integration."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

from roboharness.wrappers import VectorEnvAdapter

if TYPE_CHECKING:
    import gymnasium as gym
    import numpy as np


def _patch_config_for_headless(env_id: str) -> None:
    """Patch the HuggingFace-cached config.yaml for headless (CI) rendering.

    The lerobot/unitree-g1-mujoco env.py loads config.yaml at import time.
    The default config has ``ENABLE_ONSCREEN: true`` which requires GLFW/display.
    For headless environments (MUJOCO_GL=osmesa, no DISPLAY), we disable onscreen
    rendering so the simulator uses offscreen-only mode.
    """
    import os

    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    if has_display:
        return  # Display available, no patching needed

    try:
        from huggingface_hub import snapshot_download

        repo_dir = Path(snapshot_download(env_id, repo_type="model"))
    except Exception:
        return  # Can't patch, let make_env handle errors

    config_path = repo_dir / "config.yaml"
    if not config_path.exists():
        return

    import yaml

    config = yaml.safe_load(config_path.read_text())
    if config.get("ENABLE_ONSCREEN") is True:
        config["ENABLE_ONSCREEN"] = False
        config["ENABLE_OFFSCREEN"] = True
        config_path.write_text(yaml.dump(config, default_flow_style=False))
        print("      Patched config.yaml: ENABLE_ONSCREEN=false (headless mode)")


def create_native_env(
    env_id: str = "lerobot/unitree-g1-mujoco",
    *,
    n_envs: int = 1,
) -> gym.Env:
    """Create a LeRobot environment, preferring the official ``make_env()`` factory.

    Strategy (in order):
      1. Try LeRobot's ``make_env()`` — wraps the hub env in ``SyncVectorEnv``.
         We unwrap the batch dimension via ``VectorEnvAdapter`` so downstream
         wrappers see a standard single-env interface.
      2. Fall back to importing the hub's ``env.py`` directly (works without
         the full LeRobot install; avoids the ``SyncVectorEnv`` obs-space
         mismatch that the upstream env has).

    Args:
        env_id: HuggingFace model ID for the LeRobot environment.
        n_envs: Number of vectorized environments (usually 1 for eval).

    Returns:
        A Gymnasium-compatible environment.
    """
    try:
        from huggingface_hub import snapshot_download  # noqa: F401 — used below
    except ImportError:
        print(
            "ERROR: huggingface_hub is required for native integration.\n"
            "Install with: pip install roboharness[demo,unitree] lerobot"
        )
        sys.exit(1)

    # Patch config for headless CI environments before importing env module
    _patch_config_for_headless(env_id)

    env = _try_lerobot_make_env(env_id, n_envs=n_envs)
    if env is None:
        env = _fallback_hub_make_env(env_id, n_envs=n_envs)

    # Add MuJoCo rendering capability — the hub env has a MuJoCo model but
    # doesn't expose render_camera(), so the wrapper can't capture screenshots.
    _add_mujoco_rendering(env)

    print(f"      Env type: {type(env).__name__}")
    print(f"      Obs space (declared): {env.observation_space}")
    print(f"      Act space: {env.action_space}")

    return env


def _try_lerobot_make_env(env_id: str, *, n_envs: int = 1) -> gym.Env | None:
    """Try creating the env via LeRobot's official ``make_env()`` factory.

    Returns a ``VectorEnvAdapter``-wrapped env on success, or ``None`` if
    LeRobot is not installed or ``make_env()`` fails.
    """
    try:
        from lerobot.common.envs.factory import (  # type: ignore[import-not-found]
            make_env,
        )
    except ImportError:
        print("      LeRobot not installed — falling back to hub env import")
        return None

    try:
        vec_env = make_env(env_id, n_envs=n_envs)
    except Exception as exc:
        print(f"      LeRobot make_env() failed ({exc}) — falling back to hub env import")
        return None

    # make_env() wraps in SyncVectorEnv; adapt to standard gym.Env.
    env = VectorEnvAdapter(vec_env)
    print("      Created via LeRobot make_env() + VectorEnvAdapter")
    return env


def _fallback_hub_make_env(env_id: str, *, n_envs: int = 1) -> gym.Env:
    """Import the hub's ``env.py`` directly (no LeRobot dependency)."""
    from huggingface_hub import snapshot_download

    repo_dir = Path(snapshot_download(env_id, repo_type="model"))
    sys.path.insert(0, str(repo_dir))
    try:
        from env import make_env as hub_make_env  # type: ignore[import-not-found]
    except ImportError as e:
        print(f"ERROR: Failed to import hub env module: {e}")
        sys.exit(1)

    env = hub_make_env(n_envs=n_envs)

    # Obs-space shape mismatch (upstream declares (97,) but returns (100,) due to
    # floating_base_acc being 6-D not 3-D) is handled automatically by
    # RobotHarnessWrapper(auto_fix_obs_space=True). See issue #110.

    print("      Created via direct hub env import (fallback)")
    return env  # type: ignore[no-any-return]


def _add_mujoco_rendering(
    env: gym.Env,
    width: int = 640,
    height: int = 480,
) -> None:
    """Patch the env to support render_camera() using MuJoCo's renderer.

    The hub env has a MuJoCo model/data underneath but doesn't expose camera
    rendering. We find the model/data, create a mujoco.Renderer, and add
    render_camera() + cameras property so RobotHarnessWrapper can capture
    multi-view screenshots.
    """
    unwrapped = getattr(env, "unwrapped", env)

    # Find the MuJoCo model and data on the env (attribute names vary by env)
    # Search the unwrapped env and one level deeper (e.g. env.sim_env.mj_model
    # for the lerobot/unitree-g1-mujoco hub env).
    model = None
    data = None
    search_targets = [unwrapped]
    for nested in ("sim_env", "simulator", "sim"):
        obj = getattr(unwrapped, nested, None)
        if obj is not None:
            search_targets.append(obj)

    for target in search_targets:
        for attr in ("model", "_model", "mj_model"):
            candidate = getattr(target, attr, None)
            if candidate is not None and hasattr(candidate, "ncam"):
                model = candidate
                break
        if model is not None:
            break

    for target in search_targets:
        for attr in ("data", "_data", "mj_data"):
            candidate = getattr(target, attr, None)
            if candidate is not None and hasattr(candidate, "qpos"):
                data = candidate
                break
        if data is not None:
            break

    if model is None or data is None:
        print("      Warning: could not find MuJoCo model/data — no screenshots")
        return

    import mujoco

    renderer = mujoco.Renderer(model, height, width)
    camera_names = [model.camera(i).name for i in range(model.ncam)]

    def render_camera(camera_name: str) -> np.ndarray:
        if camera_name not in camera_names:
            raise ValueError(f"Unknown camera: {camera_name}. Available: {camera_names}")
        renderer.update_scene(data, camera=camera_name)
        return renderer.render()  # type: ignore[no-any-return]

    # Patch the unwrapped env so the wrapper detects render_camera capability
    unwrapped.render_camera = render_camera  # type: ignore[attr-defined,union-attr]
    unwrapped.cameras = camera_names  # type: ignore[attr-defined,union-attr]
    # Store model/data for controller state access
    unwrapped.mj_model = model  # type: ignore[attr-defined,union-attr]
    unwrapped.mj_data = data  # type: ignore[attr-defined,union-attr]
    print(f"      Added MuJoCo rendering: {len(camera_names)} cameras {camera_names}")
