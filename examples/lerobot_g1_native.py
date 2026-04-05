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
    # CPU-only (lighter install):
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install roboharness[lerobot-native] Pillow

    # Or full GPU:
    pip install roboharness[lerobot-native] Pillow

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
from typing import Any, ClassVar

import numpy as np

try:
    import gymnasium as gym
except ImportError:
    print("ERROR: gymnasium is required. Install with: pip install gymnasium")
    sys.exit(1)

from roboharness.wrappers import RobotHarnessWrapper

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LEROBOT_ENV_ID = "lerobot/unitree-g1-mujoco"
DEFAULT_N_STEPS = 500


# ---------------------------------------------------------------------------
# Environment creation via make_env()
# ---------------------------------------------------------------------------


def create_native_env(
    env_id: str = LEROBOT_ENV_ID,
    *,
    n_envs: int = 1,
) -> gym.Env:
    """Create a LeRobot environment using the official make_env() factory.

    ``make_env()`` returns ``dict[str, dict[int, VectorEnv]]``.  We extract
    the single underlying environment for direct Gymnasium usage.
    """
    try:
        from lerobot.envs.factory import make_env
    except ImportError:
        print(
            "ERROR: lerobot is required for native integration.\n"
            "Install with:\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
            "  pip install roboharness[lerobot-native] Pillow"
        )
        sys.exit(1)

    env_dict = make_env(env_id, n_envs=n_envs, trust_remote_code=True)

    # Extract the VectorEnv from the nested dict structure
    # make_env returns {suite_name: {index: VectorEnv}}
    for suite_name, index_map in env_dict.items():
        for idx, vec_env in index_map.items():
            print(f"      Suite: {suite_name}, index: {idx}")
            print(f"      VectorEnv type: {type(vec_env).__name__}")
            print(f"      Obs space: {vec_env.single_observation_space}")
            print(f"      Act space: {vec_env.single_action_space}")

            # For n_envs=1, unwrap the VectorEnv to get a single env
            # VectorEnv wraps the env — we use it directly as it's Gymnasium-compatible
            return _VectorEnvAdapter(vec_env)

    msg = f"make_env('{env_id}') returned empty dict"
    raise RuntimeError(msg)


class _VectorEnvAdapter(gym.Env):
    """Thin adapter from VectorEnv (n=1) to standard Gymnasium Env interface.

    LeRobot's ``make_env()`` always returns a ``VectorEnv``, even for ``n_envs=1``.
    This adapter unwraps the batch dimension so ``RobotHarnessWrapper`` sees a
    standard single-env interface: scalar reward, 1-D/2-D obs (not batched), etc.
    """

    metadata: ClassVar[dict[str, Any]] = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, vec_env: gym.vector.VectorEnv) -> None:
        super().__init__()
        self._vec_env = vec_env
        self.observation_space = vec_env.single_observation_space
        self.action_space = vec_env.single_action_space
        self.render_mode = "rgb_array"

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        obs, info = self._vec_env.reset(seed=seed, options=options)
        return _unbatch(obs), _unbatch_info(info)

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        # Add batch dimension for VectorEnv
        batched_action = np.expand_dims(np.asarray(action), axis=0)
        obs, reward, terminated, truncated, info = self._vec_env.step(batched_action)
        return (
            _unbatch(obs),
            float(reward[0]) if hasattr(reward, "__getitem__") else float(reward),
            bool(terminated[0]) if hasattr(terminated, "__getitem__") else bool(terminated),
            bool(truncated[0]) if hasattr(truncated, "__getitem__") else bool(truncated),
            _unbatch_info(info),
        )

    def render(self) -> np.ndarray | None:
        frames = self._vec_env.render()
        if frames is not None and len(frames) > 0:
            return frames[0] if hasattr(frames, "__getitem__") else frames
        return None

    def close(self) -> None:
        self._vec_env.close()


def _unbatch(obs: Any) -> Any:
    """Remove batch dimension from observation."""
    if isinstance(obs, dict):
        return {k: _unbatch(v) for k, v in obs.items()}
    if isinstance(obs, np.ndarray) and obs.ndim > 0:
        return obs[0]
    if hasattr(obs, "__getitem__"):
        return obs[0]
    return obs


def _unbatch_info(info: dict[str, Any]) -> dict[str, Any]:
    """Remove batch dimension from info dict values."""
    out: dict[str, Any] = {}
    for k, v in info.items():
        if isinstance(v, np.ndarray) and v.ndim > 0 and v.shape[0] == 1:
            out[k] = v[0]
        elif isinstance(v, dict):
            out[k] = _unbatch_info(v)
        else:
            out[k] = v
    return out


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

    # 2. Wrap with RobotHarnessWrapper
    print("[2/4] Wrapping with RobotHarnessWrapper ...")
    cp_steps = [1, n_steps // 2, n_steps]
    checkpoints = [
        {"name": "initial", "step": cp_steps[0]},
        {"name": "mid", "step": cp_steps[1]},
        {"name": "final", "step": cp_steps[2]},
    ]

    # Detect available cameras — try render_camera if available
    cameras = ["default"]
    unwrapped = getattr(env, "unwrapped", env)
    if hasattr(unwrapped, "cameras"):
        cameras = list(unwrapped.cameras)
    elif hasattr(unwrapped, "_cameras"):
        cameras = list(unwrapped._cameras)

    wrapped = RobotHarnessWrapper(
        env,
        checkpoints=checkpoints,
        cameras=cameras,
        output_dir=str(output_dir),
        task_name="lerobot_g1_native",
    )
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
    failures = validate_integration(checkpoint_infos, expected_count=len(checkpoints))

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
