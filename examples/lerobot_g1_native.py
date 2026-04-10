#!/usr/bin/env python3
"""Native LeRobot G1 Integration — controller demos on the official factory.

Uses LeRobot's ``make_env()`` to create the Unitree G1 MuJoCo environment, then
runs a selected locomotion controller via RobotHarnessWrapper to produce a visual
checkpoint report.

Supported controllers:
  - **groot**: GR00T decoupled WBC (15 lower-body joints)
  - **sonic**: SONIC kinematic planner (29 full-body joints)

Requirements:
    pip install torch --index-url https://download.pytorch.org/whl/cpu  # CPU-only
    pip install roboharness[demo] lerobot

Run:
    MUJOCO_GL=osmesa python examples/lerobot_g1_native.py --controller groot
    MUJOCO_GL=osmesa python examples/lerobot_g1_native.py --controller sonic

Output (groot):
    ./harness_output/lerobot_g1_native_groot/trial_001/
        initial/   — robot standing after reset
        walking/   — GR00T decoupled WBC walking forward
        final/     — final stopped state

Output (sonic):
    ./harness_output/lerobot_g1_native_sonic/trial_001/
        initial/   — robot standing after reset
        walking/   — SONIC planner walking forward
        final/     — final stopped state
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

CONTROLLER_CONFIGS: dict[str, dict[str, Any]] = {
    "groot": {
        "label": "GR00T decoupled WBC",
        "accent_color": "#2d8cf0",
        "n_dof": 15,  # lower body + waist
    },
    "sonic": {
        "label": "SONIC kinematic planner",
        "accent_color": "#7c4dff",
        "n_dof": 29,  # full body
    },
}


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

    # Obs-space shape mismatch (upstream declares (97,) but returns (100,) due to
    # floating_base_acc being 6-D not 3-D) is handled automatically by
    # RobotHarnessWrapper(auto_fix_obs_space=True). See issue #110.

    # Add MuJoCo rendering capability — the hub env has a MuJoCo model but
    # doesn't expose render_camera(), so the wrapper can't capture screenshots.
    _add_mujoco_rendering(env)

    print(f"      Env type: {type(env).__name__}")
    print(f"      Obs space (declared): {env.observation_space}")
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
    # Store model/data for controller state access
    unwrapped.mj_model = model  # type: ignore[attr-defined]
    unwrapped.mj_data = data  # type: ignore[attr-defined]
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


def generate_html_report(output_dir: Path, task_name: str, controller: str) -> Path:
    """Generate a self-contained HTML report with embedded checkpoint images."""
    from roboharness.reporting import generate_html_report as _generate

    cfg = CONTROLLER_CONFIGS[controller]
    return _generate(
        output_dir,
        task_name,
        title=f"LeRobot G1 Native — {cfg['label']}",
        accent_color=cfg["accent_color"],
        summary_html=(
            f"<strong>Env:</strong> <code>{LEROBOT_ENV_ID}</code> via <code>make_env()</code>"
            f"<br/><strong>Controller:</strong> {cfg['label']} ({cfg['n_dof']} DOF)"
            "<br/><strong>Integration:</strong> Native LeRobot factory + RobotHarnessWrapper"
        ),
        footer_text=(
            f"Generated by <code>examples/lerobot_g1_native.py --controller {controller}</code>"
        ),
        meshcat_mode="none",
    )


# ---------------------------------------------------------------------------
# Controller helpers
# ---------------------------------------------------------------------------


def _load_controller(name: str) -> Any:
    """Load a single locomotion controller by name. Returns None on failure."""
    try:
        if name == "groot":
            from roboharness.controllers.locomotion import GrootLocomotionController

            return GrootLocomotionController()
        elif name == "sonic":
            from roboharness.controllers.locomotion import SonicLocomotionController

            return SonicLocomotionController()
    except (ImportError, Exception) as exc:
        print(f"      Warning: could not load {name} controller ({exc})")
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Native LeRobot G1 integration — locomotion controller demo"
    )
    parser.add_argument("--output-dir", default="./harness_output", help="Output directory")
    parser.add_argument("--report", action="store_true", help="Generate HTML report")
    parser.add_argument("--n-steps", type=int, default=DEFAULT_N_STEPS, help="Number of steps")
    parser.add_argument(
        "--controller",
        choices=list(CONTROLLER_CONFIGS),
        default="groot",
        help="Locomotion controller to run (default: groot)",
    )
    parser.add_argument(
        "--assert-success", action="store_true", help="Exit non-zero on validation failure"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    n_steps = args.n_steps
    controller_name: str = args.controller
    cfg = CONTROLLER_CONFIGS[controller_name]
    task_name = f"lerobot_g1_native_{controller_name}"

    print("=" * 60)
    print(f"  Roboharness: Native LeRobot G1 — {cfg['label']}")
    print("=" * 60)

    # 1. Create environment via make_env()
    print(f"\n[1/5] Creating environment via make_env('{LEROBOT_ENV_ID}') ...")
    env = create_native_env(LEROBOT_ENV_ID)
    print(f"      Obs space: {env.observation_space}")
    print(f"      Act space: {env.action_space}")

    # 2. Load locomotion controller
    print(f"[2/5] Loading {cfg['label']} controller ...")
    ctrl = _load_controller(controller_name)

    # Access MuJoCo data for controller state (stored by _add_mujoco_rendering)
    unwrapped = getattr(env, "unwrapped", env)
    mj_data = getattr(unwrapped, "mj_data", None)

    has_controller = ctrl is not None and mj_data is not None
    if ctrl is not None and mj_data is None:
        print("      Warning: MuJoCo data not found — falling back to random actions")
    if has_controller:
        print(f"      Loaded ({cfg['n_dof']} DOF output)")
    else:
        print("      Falling back to random actions")

    # 3. Wrap with RobotHarnessWrapper
    print("[3/5] Wrapping with RobotHarnessWrapper ...")
    episode_protocol = TaskProtocol(
        name=f"{controller_name}_locomotion",
        description=f"{cfg['label']} on native LeRobot G1",
        phases=[
            TaskPhase("initial", "Standing pose after reset"),
            TaskPhase("walking", f"{cfg['label']} walking forward"),
            TaskPhase("final", "Final stopped state"),
        ],
    )
    phase_steps = {"initial": 1, "walking": n_steps * 3 // 4, "final": n_steps}

    # Detect available cameras (added by _add_mujoco_rendering or native env)
    cameras = ["default"]
    env_cameras = getattr(unwrapped, "cameras", None) or getattr(unwrapped, "_cameras", None)
    if env_cameras:
        cameras = list(env_cameras)

    wrapped = RobotHarnessWrapper(
        env,
        protocol=episode_protocol,
        phase_steps=phase_steps,
        cameras=cameras,
        output_dir=str(output_dir),
        task_name=task_name,
        auto_fix_obs_space=True,
    )
    print(f"      Protocol: {wrapped.active_protocol.name}")
    print(f"      Multi-camera: {wrapped.has_multi_camera}")
    print(f"      Camera capability: {wrapped.camera_capability}")
    print(f"      Cameras: {cameras}")

    # 4. Run episode
    print(f"[4/5] Running episode ({n_steps} steps) ...")
    obs, _info = wrapped.reset()
    if has_controller:
        ctrl.reset()

    print(f"      Initial obs type: {type(obs).__name__}", end="")
    if hasattr(obs, "shape"):
        print(f", shape: {obs.shape}", end="")
    elif isinstance(obs, dict):
        print(f", keys: {list(obs.keys())}", end="")
    print()

    action_dim = env.action_space.shape[0] if hasattr(env.action_space, "shape") else 29
    checkpoint_infos: list[dict[str, Any]] = []

    for i in range(n_steps):
        if has_controller:
            state = {"qpos": mj_data.qpos, "qvel": mj_data.qvel}
            # Stand for first 100 steps, walk, then stop for last 50
            if i < 100:
                velocity = [0.0, 0.0, 0.0]
            elif i < n_steps - 50:
                velocity = [0.3, 0.0, 0.0]
            else:
                velocity = [0.0, 0.0, 0.0]

            targets = ctrl.compute(command={"velocity": velocity}, state=state)
            action = np.zeros(action_dim, dtype=np.float64)
            action[: len(targets)] = targets
        else:
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

    # 5. Validate
    print("[5/5] Validating integration ...")
    failures = validate_integration(checkpoint_infos, expected_count=len(episode_protocol.phases))

    if failures:
        print("      VALIDATION FAILED:")
        for msg in failures:
            print(f"        FAIL: {msg}")
    else:
        print("      All checks passed!")

    # Summary
    trial_dir = output_dir / task_name / "trial_001"
    total_images = len(list(trial_dir.rglob("*_rgb.png"))) if trial_dir.exists() else 0
    print(f"      {total_images} images saved to: {trial_dir}")

    if args.report:
        report_path = generate_html_report(output_dir, task_name, controller_name)
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
