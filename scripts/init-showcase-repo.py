#!/usr/bin/env python3
"""Generate the roboharness/showcase repository structure.

Creates a complete, ready-to-push directory with:
  - Root README.md with overview and instructions
  - lerobot-g1/ showcase (adapted from examples/lerobot_g1_native.py)
  - groot-n16/ showcase (adapted from examples/lerobot_g1.py)
  - sonic-locomotion/ showcase (adapted from examples/sonic_locomotion.py)
  - Placeholder directories for pi0-libero and capx-comparison
  - CI workflow (.github/workflows/showcase-ci.yml)
  - Org profile README (.github-org/profile/README.md)

Usage:
    python scripts/init-showcase-repo.py [OUTPUT_DIR]

The default output directory is ``./showcase-output``. Run from the
roboharness repo root so the script can find source examples.

See: https://github.com/MiaoDX/roboharness/issues/139
"""

from __future__ import annotations

import argparse
import shutil
import sys
import textwrap
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
LEROBOT_EXAMPLE = REPO_ROOT / "examples" / "lerobot_g1_native.py"
GROOT_EXAMPLE = REPO_ROOT / "examples" / "lerobot_g1.py"
SONIC_EXAMPLE = REPO_ROOT / "examples" / "sonic_locomotion.py"

SHOWCASE_DIRS = [
    "groot-n16",
    "pi0-libero",
    "lerobot-g1",
    "sonic-locomotion",
    "capx-comparison",
]

# ---------------------------------------------------------------------------
# File content generators
# ---------------------------------------------------------------------------


SHOWCASE_INFO: dict[str, tuple[str, str]] = {
    "groot-n16": ("GR00T N1.6 WBC locomotion on Unitree G1", "Ready"),
    "pi0-libero": ("Pi0 policy on LIBERO manipulation benchmark", "Planned"),
    "lerobot-g1": ("LeRobot G1 native locomotion demo", "Ready"),
    "sonic-locomotion": ("GEAR-SONIC planner locomotion on Unitree G1", "Ready"),
    "capx-comparison": ("CaP-X benchmark comparison with roboharness", "Planned"),
}


def _root_readme() -> str:
    rows = []
    for d in SHOWCASE_DIRS:
        desc, status = SHOWCASE_INFO[d]
        rows.append(f"| [{d}]({d}/) | {desc} | {status} |")
    showcase_table = "\n".join(rows)
    return textwrap.dedent(f"""\
        # roboharness/showcase

        Standalone demo integrations for
        [roboharness](https://github.com/MiaoDX/roboharness) --- the visual
        testing harness for AI coding agents in robot simulation.

        Each directory is a self-contained showcase that pip-installs
        `roboharness` and runs independently. No submodules, no monorepo
        coupling.

        ## Showcases

        | Directory | Description | Status |
        |-----------|-------------|--------|
        {showcase_table}

        ## Quick start

        ```bash
        cd lerobot-g1
        pip install -r requirements.txt
        bash run.sh
        ```

        ## Design principles

        - **Self-contained**: each showcase has its own `requirements.txt`,
          `run.sh`, and main script
        - **roboharness is a pip dependency**: always uses the latest PyPI release
        - **Independent CI**: one job per showcase, failures don't block others
        - **Minimal coupling**: no git submodules, no shared code between showcases

        ## Adding a new showcase

        1. Create a new directory: `my-showcase/`
        2. Add `README.md`, `requirements.txt`, `run.sh`, and your main script
        3. Add the directory to the CI matrix in `.github/workflows/showcase-ci.yml`
        4. Open a PR

        ## License

        MIT --- same as [roboharness](https://github.com/MiaoDX/roboharness).
    """)


def _ci_workflow() -> str:
    return textwrap.dedent("""\
        name: Showcase CI

        on:
          push:
            branches: [main]
          pull_request:

        jobs:
          showcase:
            strategy:
              fail-fast: false
              matrix:
                include:
                  - name: lerobot-g1
                    dir: lerobot-g1
                  - name: groot-n16
                    dir: groot-n16
                  - name: sonic-locomotion
                    dir: sonic-locomotion

            name: ${{ matrix.name }}
            runs-on: ubuntu-latest
            env:
              MUJOCO_GL: osmesa

            steps:
              - uses: actions/checkout@v4

              - uses: actions/setup-python@v5
                with:
                  python-version: "3.10"

              - name: Install system deps (OSMesa for headless MuJoCo)
                run: |
                  sudo apt-get update
                  sudo apt-get install -y libosmesa6-dev libgl1-mesa-glx

              - name: Install Python deps
                run: |
                  cd ${{ matrix.dir }}
                  pip install -r requirements.txt

              - name: Run showcase
                run: |
                  cd ${{ matrix.dir }}
                  bash run.sh
    """)


def _lerobot_readme() -> str:
    return textwrap.dedent("""\
        # LeRobot G1 Native Locomotion

        Demonstrates [roboharness](https://github.com/MiaoDX/roboharness) integration with
        LeRobot's official `make_env("lerobot/unitree-g1-mujoco")` factory.

        Runs the Unitree G1 43-DOF humanoid through stand, walk, and stop phases using
        GR00T or SONIC locomotion controllers, capturing multi-camera checkpoint screenshots
        via `RobotHarnessWrapper`.

        ## Requirements

        - Python 3.10+
        - MuJoCo with OSMesa for headless rendering (CI), or a display (local)

        ## Quick start

        ```bash
        pip install -r requirements.txt
        bash run.sh
        ```

        ## Run options

        ```bash
        # GR00T controller (default)
        MUJOCO_GL=osmesa python lerobot_g1_native.py --controller groot --report

        # SONIC controller
        MUJOCO_GL=osmesa python lerobot_g1_native.py --controller sonic --report
        ```

        ## Output

        ```
        harness_output/lerobot_g1_native_groot/trial_001/
            initial/   -- robot standing after reset
            walking/   -- controller walking forward
            final/     -- final stopped state
        ```

        Each checkpoint directory contains multi-camera PNG screenshots and a `state.json`
        with joint angles, step count, and reward.

        ## What this demonstrates

        - **Zero-change Gymnasium wrapper**: `RobotHarnessWrapper` wraps the LeRobot env
        - **Multi-camera capture**: screenshots from all MuJoCo cameras at each checkpoint
        - **Task protocol**: structured phases (initial/walking/final) with automatic triggering
        - **HTML reports**: self-contained visual report for agent consumption

        ## References

        - [roboharness](https://github.com/MiaoDX/roboharness)
        - [lerobot/unitree-g1-mujoco](https://huggingface.co/lerobot/unitree-g1-mujoco)
        - [LeRobot](https://github.com/huggingface/lerobot)
    """)


def _lerobot_requirements() -> str:
    return textwrap.dedent("""\
        roboharness[demo]
        lerobot
        torch --index-url https://download.pytorch.org/whl/cpu
        PyYAML
    """)


def _lerobot_run_sh() -> str:
    return textwrap.dedent("""\
        #!/usr/bin/env bash
        set -euo pipefail

        # Default to headless rendering if no display is available
        export MUJOCO_GL="${MUJOCO_GL:-osmesa}"

        echo "=== LeRobot G1 Native Showcase ==="
        echo "Controller: groot (default)"
        echo "MUJOCO_GL=$MUJOCO_GL"
        echo ""

        python lerobot_g1_native.py --controller groot --report --assert-success
    """)


def _groot_readme() -> str:
    return textwrap.dedent("""\
        # GR00T N1.6 WBC Locomotion

        Demonstrates [roboharness](https://github.com/MiaoDX/roboharness) integration with
        the GR00T whole-body control (WBC) locomotion controller on the Unitree G1 humanoid.

        Downloads the real 29-DOF G1 model from HuggingFace and runs it with the GR00T RL
        balance/walk controller (ONNX), capturing multi-camera checkpoint screenshots via
        `RobotHarnessWrapper`.

        ## Requirements

        - Python 3.10+
        - MuJoCo with OSMesa for headless rendering (CI), or a display (local)

        ## Quick start

        ```bash
        pip install -r requirements.txt
        bash run.sh
        ```

        ## Run options

        ```bash
        # Default: GR00T controller with HTML report
        MUJOCO_GL=osmesa python lerobot_g1.py --report

        # With validation assertions (for CI)
        MUJOCO_GL=osmesa python lerobot_g1.py --report --assert-success
        ```

        ## Output

        ```
        harness_output/lerobot_g1/trial_001/
            initial/   -- robot in default standing pose
            steady/    -- robot walking forward
            terminal/  -- robot stopped, balancing
        ```

        Each checkpoint directory contains multi-camera PNG screenshots and a `state.json`
        with joint angles, step count, and reward.

        ## What this demonstrates

        - **GR00T WBC controller**: ONNX-based balance + walk policies from NVIDIA
        - **Multi-camera capture**: screenshots from all MuJoCo cameras at each checkpoint
        - **Locomotion protocol**: structured phases (initial/steady/terminal)
        - **Validation checks**: robot-stayed-upright assertions for CI

        ## References

        - [roboharness](https://github.com/MiaoDX/roboharness)
        - [lerobot/unitree-g1-mujoco](https://huggingface.co/lerobot/unitree-g1-mujoco)
        - [GR00T WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl)
    """)


def _groot_requirements() -> str:
    return textwrap.dedent("""\
        roboharness[demo]
        huggingface_hub
    """)


def _groot_run_sh() -> str:
    return textwrap.dedent("""\
        #!/usr/bin/env bash
        set -euo pipefail

        # Default to headless rendering if no display is available
        export MUJOCO_GL="${MUJOCO_GL:-osmesa}"

        echo "=== GR00T N1.6 WBC Locomotion Showcase ==="
        echo "Controller: groot"
        echo "MUJOCO_GL=$MUJOCO_GL"
        echo ""

        python lerobot_g1.py --report --assert-success
    """)


def _sonic_readme() -> str:
    return textwrap.dedent("""\
        # SONIC Locomotion

        Demonstrates [roboharness](https://github.com/MiaoDX/roboharness) integration with
        NVIDIA's GEAR-SONIC locomotion controller in planner mode on the Unitree G1 humanoid.

        The SONIC controller uses ONNX models (downloaded from `nvidia/GEAR-SONIC` on
        HuggingFace) to generate full-body pose trajectories from velocity commands.

        ## Requirements

        - Python 3.10+
        - MuJoCo with OSMesa for headless rendering (CI), or a display (local)

        ## Quick start

        ```bash
        pip install -r requirements.txt
        bash run.sh
        ```

        ## Run options

        ```bash
        # Default: SONIC planner with HTML report
        MUJOCO_GL=osmesa python sonic_locomotion.py --report

        # Custom render resolution
        MUJOCO_GL=osmesa python sonic_locomotion.py --report --width 1280 --height 960
        ```

        ## Output

        ```
        harness_output/sonic_locomotion/trial_001/
            initial/   -- robot in default standing pose
            walking/   -- walking forward via SONIC planner
            stopping/  -- decelerating to stop
            terminal/  -- stopped, balancing
        ```

        Each checkpoint directory contains multi-camera PNG screenshots and a `state.json`
        with joint angles, step count, and reward.

        ## What this demonstrates

        - **SONIC planner mode**: velocity commands to full-body pose trajectories
        - **ONNX inference**: models downloaded from HuggingFace on first use
        - **Four-phase protocol**: stand, walk, decelerate, stop
        - **Multi-camera capture**: screenshots from all MuJoCo cameras at each checkpoint

        ## References

        - [roboharness](https://github.com/MiaoDX/roboharness)
        - [nvidia/GEAR-SONIC](https://huggingface.co/nvidia/GEAR-SONIC)
        - [NVlabs/GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl)
    """)


def _sonic_requirements() -> str:
    return textwrap.dedent("""\
        roboharness[demo]
        huggingface_hub
    """)


def _sonic_run_sh() -> str:
    return textwrap.dedent("""\
        #!/usr/bin/env bash
        set -euo pipefail

        # Default to headless rendering if no display is available
        export MUJOCO_GL="${MUJOCO_GL:-osmesa}"

        echo "=== SONIC Locomotion Showcase ==="
        echo "Controller: GEAR-SONIC planner"
        echo "MUJOCO_GL=$MUJOCO_GL"
        echo ""

        python sonic_locomotion.py --report
    """)


def _pi0_libero_readme() -> str:
    return textwrap.dedent("""\
        # Pi0 on LIBERO

        *Planned.* This showcase will demonstrate roboharness integration with the
        [Pi0](https://www.physicalintelligence.company/blog/pi0) generalist robot
        policy on the [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)
        manipulation benchmark.

        ## Planned scope

        - Run Pi0 policy on LIBERO-Spatial or LIBERO-Object tasks in MuJoCo
        - Capture checkpoint screenshots at key manipulation stages
        - Generate HTML reports comparing task success across episodes

        ## Dependencies

        - Pi0 model weights (requires access)
        - LIBERO benchmark environments
        - roboharness `RobotHarnessWrapper`

        ## Contributing

        Want to help build this showcase? Open an issue at
        [roboharness/showcase](https://github.com/roboharness/showcase/issues).

        ## References

        - [roboharness](https://github.com/MiaoDX/roboharness)
        - [Pi0 blog post](https://www.physicalintelligence.company/blog/pi0)
        - [LIBERO benchmark](https://github.com/Lifelong-Robot-Learning/LIBERO)
    """)


def _capx_comparison_readme() -> str:
    return textwrap.dedent("""\
        # CaP-X Benchmark Comparison

        *Planned.* This showcase will compare roboharness visual testing against the
        [CaP-X](https://arxiv.org/abs/2603.22435) benchmark for coding agents that
        program robot manipulation tasks.

        ## Planned scope

        - Run the same manipulation tasks from CaP-X using roboharness
        - Compare evaluation approaches: CaP-X metrics vs roboharness checkpoints
        - Demonstrate how visual checkpoint feedback improves agent iteration

        ## Dependencies

        - CaP-X benchmark environments
        - MuJoCo or Isaac Lab backend
        - roboharness `RobotHarnessWrapper`

        ## Contributing

        Want to help build this showcase? Open an issue at
        [roboharness/showcase](https://github.com/roboharness/showcase/issues).

        ## References

        - [roboharness](https://github.com/MiaoDX/roboharness)
        - [CaP-X paper](https://arxiv.org/abs/2603.22435)
        - [CaP-X — Benchmark for Coding Agents](https://github.com/Autonomous-Robotics-Lab/CaP-X)
    """)


def _placeholder_readme(name: str) -> str:
    return textwrap.dedent(f"""\
        # {name}

        *Coming soon.* This showcase is planned but not yet implemented.

        See the [main README](../README.md) for the full showcase list.

        Want to contribute? Open an issue at
        [roboharness/showcase](https://github.com/roboharness/showcase/issues).
    """)


def _org_profile_readme() -> str:
    return textwrap.dedent("""\
        ## roboharness

        **Visual testing harness for AI coding agents in robot simulation.**

        Let Claude Code and Codex *see* what the robot is doing, *judge* if it's working,
        and *iterate* autonomously.

        | Repository | Description |
        |------------|-------------|
        | [roboharness][rh] | Core: harness, wrappers, backends, evaluation |
        | [showcase][sc] | Standalone demos (GR00T, LeRobot, SONIC, Pi0) |

        [rh]: https://github.com/MiaoDX/roboharness
        [sc]: https://github.com/roboharness/showcase

        ### Links

        - [Live Visual Reports](https://miaodx.com/roboharness/)
        - [PyPI](https://pypi.org/project/roboharness/)
        - [Documentation](https://github.com/MiaoDX/roboharness/tree/main/docs)
    """)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the roboharness/showcase repository structure"
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        default="./showcase-output",
        help="Target directory (default: ./showcase-output)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output directory",
    )
    args = parser.parse_args()

    output = Path(args.output_dir).resolve()

    if output.exists() and not args.force:
        print(f"Error: {output} already exists. Use --force to overwrite.")
        sys.exit(1)

    if output.exists() and args.force:
        shutil.rmtree(output)

    # Check that the source examples exist
    for name, path in [
        ("lerobot_g1_native", LEROBOT_EXAMPLE),
        ("lerobot_g1", GROOT_EXAMPLE),
        ("sonic_locomotion", SONIC_EXAMPLE),
    ]:
        if not path.exists():
            print(f"Error: cannot find {path} ({name})")
            print("Run this script from the roboharness repo root.")
            sys.exit(1)

    print(f"Generating showcase repo in: {output}")

    # Create directory structure
    for d in SHOWCASE_DIRS:
        (output / d).mkdir(parents=True, exist_ok=True)
    (output / ".github" / "workflows").mkdir(parents=True, exist_ok=True)
    (output / ".github-org" / "profile").mkdir(parents=True, exist_ok=True)

    # Root files
    (output / "README.md").write_text(_root_readme())
    (output / ".github" / "workflows" / "showcase-ci.yml").write_text(_ci_workflow())

    # Org profile README (for roboharness/.github repo)
    (output / ".github-org" / "profile" / "README.md").write_text(_org_profile_readme())

    # lerobot-g1 showcase
    lerobot_dir = output / "lerobot-g1"
    shutil.copy2(LEROBOT_EXAMPLE, lerobot_dir / "lerobot_g1_native.py")
    (lerobot_dir / "README.md").write_text(_lerobot_readme())
    (lerobot_dir / "requirements.txt").write_text(_lerobot_requirements())
    run_sh = lerobot_dir / "run.sh"
    run_sh.write_text(_lerobot_run_sh())
    run_sh.chmod(0o755)

    # groot-n16 showcase
    groot_dir = output / "groot-n16"
    shutil.copy2(GROOT_EXAMPLE, groot_dir / "lerobot_g1.py")
    (groot_dir / "README.md").write_text(_groot_readme())
    (groot_dir / "requirements.txt").write_text(_groot_requirements())
    run_sh = groot_dir / "run.sh"
    run_sh.write_text(_groot_run_sh())
    run_sh.chmod(0o755)

    # sonic-locomotion showcase
    sonic_dir = output / "sonic-locomotion"
    shutil.copy2(SONIC_EXAMPLE, sonic_dir / "sonic_locomotion.py")
    (sonic_dir / "README.md").write_text(_sonic_readme())
    (sonic_dir / "requirements.txt").write_text(_sonic_requirements())
    run_sh = sonic_dir / "run.sh"
    run_sh.write_text(_sonic_run_sh())
    run_sh.chmod(0o755)

    # Detailed placeholder showcases
    (output / "pi0-libero" / "README.md").write_text(_pi0_libero_readme())
    (output / "capx-comparison" / "README.md").write_text(_capx_comparison_readme())

    # Summary
    print("\nGenerated structure:")
    for p in sorted(output.rglob("*")):
        if p.is_file():
            rel = p.relative_to(output)
            print(f"  {rel}")

    print("\nDone! To use:")
    print("  1. Create the repo: gh repo create roboharness/showcase --public")
    print(f"  2. cd {output}")
    print("  3. git init && git add -A && git commit -m 'Initial showcase structure'")
    print("  4. git remote add origin git@github.com:roboharness/showcase.git")
    print("  5. git push -u origin main")
    print("\n  Org profile README is in .github-org/ -- copy to roboharness/.github repo.")


if __name__ == "__main__":
    main()
