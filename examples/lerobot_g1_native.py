#!/usr/bin/env python3
"""Native LeRobot G1 Integration — using LeRobot's official make_env() factory.

Uses LeRobot's `make_env()` to create the Unitree G1 MuJoCo environment via the
official digital-twin pipeline (DDS-ready), then wraps it with RobotHarnessWrapper
for visual checkpoint capture.

This replaces the manual asset-download approach in `lerobot_g1.py` with the
official LeRobot factory, enabling:
  - Standardized env creation via HuggingFace Hub
  - DDS communication for sim-to-real transfer (when hardware available)
  - Consistent observation/action spaces defined by the env config

Requirements:
    pip install torch --index-url https://download.pytorch.org/whl/cpu  # CPU-only
    pip install roboharness[demo] lerobot

Run:
    MUJOCO_GL=osmesa python examples/lerobot_g1_native.py

Output:
    ./harness_output/lerobot_g1_native/trial_001/
        initial/   — env after reset
        mid/       — midpoint of episode
        final/     — end of episode
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym  # noqa: TC002 — used at runtime inside create_native_env
import numpy as np

from roboharness.core.protocol import TaskPhase, TaskProtocol
from roboharness.wrappers import RobotHarnessWrapper

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LEROBOT_ENV_ID = "lerobot/unitree-g1-mujoco"
DEFAULT_N_STEPS = 500


# ---------------------------------------------------------------------------
# Environment creation via make_env()
# ---------------------------------------------------------------------------


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
    env_id: str = LEROBOT_ENV_ID,
    *,
    n_envs: int = 1,
) -> gym.Env:
    """Create a LeRobot environment by importing the hub env module directly.

    We import the hub's ``env.py`` directly rather than going through
    lerobot's ``make_env()`` factory, which wraps in ``SyncVectorEnv``
    and breaks due to an obs-space shape mismatch in the upstream env.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print(
            "ERROR: huggingface_hub is required for native integration.\n"
            "Install with: pip install roboharness[demo,unitree] lerobot"
        )
        sys.exit(1)

    # Patch config for headless CI environments before importing env module
    _patch_config_for_headless(env_id)

    # Import the hub env module directly to avoid lerobot's VectorEnv wrapping.
    # lerobot's make_env() wraps in SyncVectorEnv which breaks when the hub env's
    # observation_space shape doesn't match actual obs (upstream bug in g1 env).
    repo_dir = Path(snapshot_download(env_id, repo_type="model"))
    sys.path.insert(0, str(repo_dir))
    try:
        from env import make_env as hub_make_env  # type: ignore[import-not-found]
    except ImportError as e:
        print(f"ERROR: Failed to import hub env module: {e}")
        sys.exit(1)

    env = hub_make_env(n_envs=n_envs)

    # Fix observation_space to match actual obs shape (upstream declares (97,)
    # but _get_obs() returns (100,) due to floating_base_acc being 6-D not 3-D)
    obs, _ = env.reset()
    actual_shape = np.asarray(obs).shape
    declared_shape = env.observation_space.shape
    if actual_shape != declared_shape:
        from gymnasium import spaces

        print(f"      Fixing obs space: declared {declared_shape} -> actual {actual_shape}")
        env.observation_space = spaces.Box(-np.inf, np.inf, shape=actual_shape, dtype=np.float32)

    # Add MuJoCo rendering capability — the hub env has a MuJoCo model but
    # doesn't expose render_camera(), so the wrapper can't capture screenshots.
    _add_mujoco_rendering(env)

    print(f"      Env type: {type(env).__name__}")
    print(f"      Obs space: {env.observation_space}")
    print(f"      Act space: {env.action_space}")

    return env


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
    import mujoco

    unwrapped = getattr(env, "unwrapped", env)

    # Find the MuJoCo model and data on the env (attribute names vary by env)
    model = None
    data = None
    for attr in ("model", "_model", "mj_model"):
        model = getattr(unwrapped, attr, None)
        if model is not None and hasattr(model, "ncam"):
            break
        model = None
    for attr in ("data", "_data", "mj_data"):
        data = getattr(unwrapped, attr, None)
        if data is not None and hasattr(data, "qpos"):
            break
        data = None

    if model is None or data is None:
        print("      Warning: could not find MuJoCo model/data — no screenshots")
        return

    renderer = mujoco.Renderer(model, height, width)
    camera_names = [model.camera(i).name for i in range(model.ncam)]

    def render_camera(camera_name: str) -> np.ndarray:
        if camera_name not in camera_names:
            raise ValueError(f"Unknown camera: {camera_name}. Available: {camera_names}")
        renderer.update_scene(data, camera=camera_name)
        return renderer.render()

    # Patch the unwrapped env so the wrapper detects render_camera capability
    unwrapped.render_camera = render_camera  # type: ignore[attr-defined]
    unwrapped.cameras = camera_names  # type: ignore[attr-defined]
    print(f"      Added MuJoCo rendering: {len(camera_names)} cameras {camera_names}")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_integration(
    checkpoint_infos: list[dict[str, Any]],
    expected_count: int,
) -> list[str]:
    """Validate that the wrapper integration worked correctly."""
    failures: list[str] = []

    if len(checkpoint_infos) != expected_count:
        failures.append(f"Expected {expected_count} checkpoints, got {len(checkpoint_infos)}")

    for cp_info in checkpoint_infos:
        files = cp_info.get("files", {})
        state_path = files.get("state")
        if not state_path or not Path(state_path).exists():
            failures.append(f"Checkpoint '{cp_info['name']}': missing state.json")
            continue

        state = json.loads(Path(state_path).read_text())
        if "step" not in state or "reward" not in state:
            failures.append(f"Checkpoint '{cp_info['name']}': state.json missing required fields")

    return failures


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------


def generate_html_report(output_dir: Path) -> Path:
    """Generate a self-contained HTML report with embedded checkpoint images."""
    from roboharness.reporting import generate_html_report as _generate

    return _generate(
        output_dir,
        "lerobot_g1_native",
        title="LeRobot G1 Native Integration Report",
        accent_color="#00b894",
        summary_html=(
            f"<strong>Env:</strong> <code>{LEROBOT_ENV_ID}</code> via <code>make_env()</code>"
            "<br/><strong>Integration:</strong> Native LeRobot factory + RobotHarnessWrapper"
        ),
        footer_text="Generated by <code>examples/lerobot_g1_native.py --report</code>",
        meshcat_mode="none",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Native LeRobot G1 integration via make_env() + RobotHarnessWrapper"
    )
    parser.add_argument("--output-dir", default="./harness_output", help="Output directory")
    parser.add_argument("--report", action="store_true", help="Generate HTML report")
    parser.add_argument("--n-steps", type=int, default=DEFAULT_N_STEPS, help="Number of steps")
    parser.add_argument(
        "--assert-success", action="store_true", help="Exit non-zero on validation failure"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    n_steps = args.n_steps

    print("=" * 60)
    print("  Roboharness: Native LeRobot G1 Integration")
    print("=" * 60)

    # 1. Create environment via make_env()
    print(f"\n[1/4] Creating environment via make_env('{LEROBOT_ENV_ID}') ...")
    env = create_native_env(LEROBOT_ENV_ID)
    print(f"      Obs space: {env.observation_space}")
    print(f"      Act space: {env.action_space}")

    # 2. Wrap with RobotHarnessWrapper using semantic protocol
    print("[2/4] Wrapping with RobotHarnessWrapper ...")
    episode_protocol = TaskProtocol(
        name="random_episode",
        description="Random-action episode observation",
        phases=[
            TaskPhase("initial", "Initial state after environment reset"),
            TaskPhase("mid", "Midpoint of episode"),
            TaskPhase("final", "Final state at episode end"),
        ],
    )
    cp_steps = [1, n_steps // 2, n_steps]

    # Detect available cameras (added by _add_mujoco_rendering or native env)
    cameras = ["default"]
    unwrapped = getattr(env, "unwrapped", env)
    env_cameras = getattr(unwrapped, "cameras", None) or getattr(unwrapped, "_cameras", None)
    if env_cameras:
        cameras = list(env_cameras)

    wrapped = RobotHarnessWrapper(
        env,
        protocol=episode_protocol,
        phase_steps={"initial": cp_steps[0], "mid": cp_steps[1], "final": cp_steps[2]},
        cameras=cameras,
        output_dir=str(output_dir),
        task_name="lerobot_g1_native",
    )
    print(f"      Protocol: {wrapped.active_protocol.name}")
    print(f"      Multi-camera: {wrapped.has_multi_camera}")
    print(f"      Camera capability: {wrapped.camera_capability}")
    print(f"      Cameras: {cameras}")

    # 3. Run episode
    print(f"[3/4] Running episode ({n_steps} steps) ...")
    obs, _info = wrapped.reset()
    print(f"      Initial obs type: {type(obs).__name__}", end="")
    if hasattr(obs, "shape"):
        print(f", shape: {obs.shape}", end="")
    elif isinstance(obs, dict):
        print(f", keys: {list(obs.keys())}", end="")
    print()

    checkpoint_infos: list[dict[str, Any]] = []
    for i in range(n_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = wrapped.step(action)

        if "checkpoint" in info:
            cp = info["checkpoint"]
            checkpoint_infos.append(cp)
            print(f"      Checkpoint '{cp['name']}' at step {cp['step']} | reward={reward:.3f}")
            print(f"        -> {cp['capture_dir']}")

        if terminated or truncated:
            print(f"      Episode ended at step {i + 1} (terminated={terminated})")
            obs, _info = wrapped.reset()

    # 4. Validate
    print("[4/4] Validating integration ...")
    failures = validate_integration(checkpoint_infos, expected_count=len(episode_protocol.phases))

    if failures:
        print("      VALIDATION FAILED:")
        for msg in failures:
            print(f"        FAIL: {msg}")
    else:
        print("      All checks passed!")

    # Summary
    trial_dir = output_dir / "lerobot_g1_native" / "trial_001"
    total_images = len(list(trial_dir.rglob("*_rgb.png"))) if trial_dir.exists() else 0
    print(f"      {total_images} images saved to: {trial_dir}")

    if args.report:
        report_path = generate_html_report(output_dir)
        print(f"      HTML report: {report_path}")

    print("\n  Output structure:")
    if trial_dir.exists():
        for cp_dir in sorted(trial_dir.iterdir()):
            if cp_dir.is_dir():
                files = sorted(f.name for f in cp_dir.iterdir() if f.is_file())
                print(f"    {cp_dir.name}/")
                for fname in files:
                    print(f"      {fname}")

    print("\n" + "=" * 60)
    wrapped.close()

    if args.assert_success and failures:
        print("\n  VALIDATION: FAILED")
        sys.exit(1)
    elif args.assert_success:
        print("\n  VALIDATION: PASSED")


if __name__ == "__main__":
    main()
