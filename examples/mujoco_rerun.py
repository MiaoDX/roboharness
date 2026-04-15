#!/usr/bin/env python3
"""MuJoCo + Rerun example using the shared deterministic grasp fixture."""

from __future__ import annotations

import argparse
from pathlib import Path

from roboharness.backends.mujoco_meshcat import MuJoCoMeshcatBackend
from roboharness.core.harness import Harness

try:
    from examples._mujoco_grasp_fixture import (
        GRASP_MJCF,
        MUJOCO_GRASP_CAMERAS,
        MUJOCO_GRASP_TASK,
        build_grasp_phases,
        build_grasp_protocol,
    )
except ModuleNotFoundError:  # pragma: no cover - script execution path
    from _mujoco_grasp_fixture import (  # type: ignore[no-redef]
        GRASP_MJCF,
        MUJOCO_GRASP_CAMERAS,
        MUJOCO_GRASP_TASK,
        build_grasp_phases,
        build_grasp_protocol,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Roboharness MuJoCo + Rerun Example")
    parser.add_argument("--output-dir", default="./harness_output", help="Output directory")
    parser.add_argument("--width", type=int, default=640, help="Render width")
    parser.add_argument("--height", type=int, default=480, help="Render height")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    print("=" * 60)
    print("  Roboharness: MuJoCo + Rerun Example")
    print("=" * 60)

    print("\n[1/4] Loading MuJoCo model ...")
    backend = MuJoCoMeshcatBackend(
        xml_string=GRASP_MJCF,
        cameras=MUJOCO_GRASP_CAMERAS,
        render_width=args.width,
        render_height=args.height,
    )

    print("[2/4] Setting up harness with Rerun capture logging ...")
    harness = Harness(
        backend,
        output_dir=str(output_dir),
        task_name=MUJOCO_GRASP_TASK,
        enable_rerun=True,
        rerun_app_id="roboharness",
    )
    phases = build_grasp_phases()
    harness.load_protocol(build_grasp_protocol())
    print(f"      Protocol: {harness.active_protocol.name}")
    print(f"      Checkpoints: {harness.list_checkpoints()}")

    print("[3/4] Running grasp simulation (logging to .rrd) ...")
    harness.reset()
    for phase_name, actions in phases.items():
        result = harness.run_to_next_checkpoint(actions)
        if result is None:
            print(f"      WARNING: No checkpoint for phase '{phase_name}'")
            continue

        print(
            f"      Checkpoint '{phase_name}': {len(result.views)} views"
            f" | step={result.step} | sim_time={result.sim_time:.3f}s"
        )

    rrd_path = output_dir / MUJOCO_GRASP_TASK / "trial_001" / "capture.rrd"
    print("\n[4/4] Done!")
    print(f"      Rerun recording: {rrd_path}")
    if rrd_path.exists():
        size_mb = rrd_path.stat().st_size / (1024 * 1024)
        print(f"      File size: {size_mb:.1f} MB")
    print("\n  View the recording:")
    print(f"    rerun {rrd_path}")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
