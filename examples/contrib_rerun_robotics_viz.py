#!/usr/bin/env python3
"""Rerun + Roboharness: multi-view robot visualization on the shared grasp fixture."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from examples._mujoco_grasp_fixture import (
        GRASP_MJCF,
        MUJOCO_GRASP_CAMERAS,
        build_grasp_phases,
        build_grasp_protocol,
    )
except ModuleNotFoundError:  # pragma: no cover - script execution path
    from _mujoco_grasp_fixture import (  # type: ignore[no-redef]
        GRASP_MJCF,
        MUJOCO_GRASP_CAMERAS,
        build_grasp_phases,
        build_grasp_protocol,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rerun + Roboharness: Multi-view Robot Simulation Visualization"
    )
    parser.add_argument("--output-dir", default="./harness_rerun_output", help="Output directory")
    parser.add_argument("--width", type=int, default=640, help="Render width")
    parser.add_argument("--height", type=int, default=480, help="Render height")
    args = parser.parse_args()

    try:
        import rerun as rr  # noqa: F401
    except ImportError:
        print("ERROR: rerun-sdk is required. Install with: pip install rerun-sdk>=0.18")
        sys.exit(1)

    try:
        import mujoco  # noqa: F401
    except ImportError:
        print("ERROR: mujoco is required. Install with: pip install mujoco>=3.0")
        sys.exit(1)

    from roboharness.backends.mujoco_meshcat import MuJoCoMeshcatBackend
    from roboharness.core.harness import Harness

    output_dir = Path(args.output_dir)

    print("=" * 60)
    print("  Rerun + Roboharness: Multi-view Robot Visualization")
    print("=" * 60)

    print("\n[1/5] Loading MuJoCo model with 3 cameras ...")
    backend = MuJoCoMeshcatBackend(
        xml_string=GRASP_MJCF,
        cameras=MUJOCO_GRASP_CAMERAS,
        render_width=args.width,
        render_height=args.height,
    )

    print("[2/5] Setting up harness with Rerun capture logging ...")
    harness = Harness(
        backend,
        output_dir=str(output_dir),
        task_name="grasp_debug",
        enable_rerun=True,
        rerun_app_id="roboharness_rerun_example",
    )
    harness.load_protocol(build_grasp_protocol())
    print(f"      Protocol: {harness.active_protocol.name}")
    print(f"      Checkpoints: {harness.list_checkpoints()}")

    phases = build_grasp_phases()

    print("[3/5] Running grasp simulation (logging to Rerun) ...")
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

    print("[4/5] Adding state annotations to Rerun recording ...")
    _log_state_summary(phases)

    rrd_path = output_dir / "grasp_debug" / "trial_001" / "capture.rrd"
    print("\n[5/5] Done!")
    if rrd_path.exists():
        size_mb = rrd_path.stat().st_size / (1024 * 1024)
        print(f"      Rerun recording: {rrd_path} ({size_mb:.1f} MB)")
    else:
        print(f"      Expected recording at: {rrd_path}")

    print("\n  View the recording:")
    print(f"    rerun {rrd_path}")
    print()
    print("  What you'll see in the Rerun Viewer:")
    print("    - camera/front/rgb, camera/side/rgb, camera/top/rgb — multi-view captures")
    print("    - camera/*/depth — depth maps at each checkpoint")
    print("    - harness/checkpoint — checkpoint names on the timeline")
    print("    - harness/state — full simulation state JSON at each checkpoint")
    print("\n" + "=" * 60)


def _log_state_summary(phases: dict[str, list[object]]) -> None:
    """Log a compact annotation for each grasp phase."""
    import rerun as rr

    summaries = {
        "plan": "Initial planning snapshot before motion starts",
        "pre_grasp": "Gripper open, hovering above the cube",
        "approach": "Gripper lowered onto the cube, fingers still open",
        "grasp": "Fingers closed around the cube",
        "lift": "Cube lifted off the table surface",
    }

    cumulative_step = 0
    for phase_name, actions in phases.items():
        cumulative_step += len(actions)
        if hasattr(rr, "set_time_sequence"):
            rr.set_time_sequence("sim_step", cumulative_step)
        else:
            rr.set_time("sim_step", sequence=cumulative_step)

        description = summaries.get(phase_name, phase_name)
        rr.log("harness/phase_summary", rr.TextDocument(f"**{phase_name}**: {description}"))


if __name__ == "__main__":
    main()
