# Roboharness

Approval/evidence harness for unattended robot code changes.

[![CI](https://github.com/MiaoDX/roboharness/actions/workflows/ci.yml/badge.svg)](https://github.com/MiaoDX/roboharness/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/roboharness)](https://pypi.org/project/roboharness/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Roboharness is not just a screenshot collector.

The core wedge is:

`long unattended agent run -> one proof pack -> short human review`

The proving ground starts with the deterministic MuJoCo grasp loop, but the
same proof surface also works for humanoid runs across multiple frameworks.
From a repo checkout, one command gets back a compiled contract,
metric-backed alarms, a phase manifest, an approval report, and an HTML proof
surface that tells you what changed and what to do next.

![Unitree G1 humanoid demo rendered side by side in Meshcat and MuJoCo](assets/g1/X36_Y28_Z13/g1_meshcat_mujoco_comparison.gif)

This README preview uses the kept review angles from the same G1 humanoid run:
Meshcat front-to-back on the left and MuJoCo top-down on the right. Each frame
keeps its phase name visible so you can compare humanoid behavior across
frameworks without opening the full report first. To regenerate the same proof
surface locally from the committed bundle, run
`python examples/demos/g1/cross_framework_report.py`.

## Choose Your Start

### Package-First Integration

Use this when you are adding roboharness to an existing codebase. The published
wheel installs the library and the `roboharness` CLI, not the repo's
`examples/` directory.

For the latest code, prefer installing from Git with `uv`:

```bash
uv pip install "roboharness @ git+https://github.com/MiaoDX/roboharness.git"
roboharness --help
```

The PyPI package can briefly trail the current README because publishing is
handled separately. If you need the latest published release instead, use:

```bash
pip install roboharness
roboharness --help
```

The fastest honest package path is the zero-change Gymnasium wrapper shown in
[Gymnasium Wrapper (Zero-Change Integration)](#gymnasium-wrapper-zero-change-integration).
If you want to evaluate the maintained MuJoCo approval wedge itself, use a repo
checkout.

### Repo Demo: 10-Minute MuJoCo Wedge

This path exercises the shipped MuJoCo approval wedge from this repository.

```bash
git clone https://github.com/MiaoDX/roboharness.git
cd roboharness
python -m pip install -e ".[demo]"
python examples/demos/mujoco/grasp.py --report
```

For headless Linux or CI:

```bash
MUJOCO_GL=egl python examples/demos/mujoco/grasp.py --report
# or
MUJOCO_GL=osmesa python examples/demos/mujoco/grasp.py --report
```

What you get back:

- `contract.json` — compiled regression contract for this wedge run
- `autonomous_report.json` — canonical metrics and baseline comparison
- `alarms.json` — evaluator-backed hard failures
- `phase_manifest.json` — first failing phase, selected views, rerun hint
- `approval_report.json` — surfaced vs suppressed case decision for review
- `report.html` — first-screen proof, not a folder hunt

How to read it:

1. Open `report.html`.
2. Read the **Run Decision** banner first.
3. Review only surfaced cases against the old baseline.
4. Use `phase_manifest.json` and the rerun hint if you need to iterate again.

Baseline rule:

- Regression mode keeps the old baseline authoritative.
- No new baseline is blessed automatically.

## Why This Exists

If Claude Code or Codex spends hours refactoring a robot behavior, the hard part is
not generating more files. It is getting back one compact proof surface that answers:

- what failed
- where it failed first
- what the current evidence looks like next to the blessed baseline
- whether anything actually needs human review

That is the job of the MuJoCo wedge today.

## Installation Matrix

```bash
uv pip install "roboharness @ git+https://github.com/MiaoDX/roboharness.git"          # latest Git core
uv pip install "roboharness[demo] @ git+https://github.com/MiaoDX/roboharness.git"    # MuJoCo, Meshcat, Gymnasium, Rerun, Pillow
uv pip install "roboharness[demo,wbc] @ git+https://github.com/MiaoDX/roboharness.git" # + whole-body control (Pinocchio, Pink)
uv pip install "roboharness[lerobot] @ git+https://github.com/MiaoDX/roboharness.git" # LeRobot evaluation path
uv pip install "roboharness[dev] @ git+https://github.com/MiaoDX/roboharness.git"     # test/lint/type deps
```

For PyPI installs, replace the `uv pip install "... @ git+..."` form with the
same extra on the published package, for example `pip install roboharness[demo]`.

## Progressive Disclosure

- Package-first: wire the wrapper or `Harness` API into your existing codebase
- Repo demo: from a clone of this repo, run `python examples/demos/mujoco/grasp.py --report`
- Preset-first: pass `--contract-preset mujoco_regression_v1` or
  `--contract-preset mujoco_migration_guarded_v1`
- Prompt-assisted: pass `--contract-prompt "treat this as migration mode and require manual blessing"`
  to select one of the reviewed presets without opening JSON
- Advanced: pass `--contract-json /path/to/contract.json` to validate a pre-authored
  metric-only contract before the wedge starts

If a contract cannot be grounded safely, the run stops before execution and emits a
user-facing error envelope with `problem`, `cause`, `fix`, `docs_url`, and `next_action`.

## Project Harness Skills From Python Contracts

Projects can author a trusted Python `HarnessContract` and generate an
agent-facing harness skill from it:

```bash
roboharness contract generate agent-skill/<project-slug>-harness/contract.py \
  --output-dir agent-skill/<project-slug>-harness
roboharness contract check agent-skill/<project-slug>-harness/contract.py \
  --output-dir agent-skill/<project-slug>-harness
```

The handwritten `contract.py` is the source of truth for semantic phases, hard
metric gates, visual review dimensions, evidence boundaries, approval policy,
validation commands, and named workflows. Generated `SKILL.md`,
`contract.snapshot.json`, schemas, scope-brief template, stubs, and manifest
live beside it under `agent-skill/<project-slug>-harness/`.

This repo dogfoods that path in `agent-skill/roboharness-harness/contract.py`.
The generated skill can guide agents, but it is not authoritative; out-of-scope
checks require a Harness Scope Brief and a reviewed contract change first.

## Proof Surface

The first screen is meant to be actionable without replay:

- **Run Decision** tells you whether the run is clean, reviewable, or degraded
- **Approval Queue** shows changed or ambiguous cases only
- **Current vs Baseline** shows the first manifest-selected proof pair
- **Temporal Evidence** appears for ambiguous still-image cases as a checkpoint-ordered strip
- **Hard Metric Results** shows the evaluator-backed failures
- Evidence images support click-to-zoom for quick inspection without dropping into the gallery
- **Phase Timeline** and the deeper checkpoint gallery stay available below the fold

| pre_grasp | contact | grasp | lift |
|:-:|:-:|:-:|:-:|
| ![pre_grasp](assets/example_mujoco_grasp/pre_grasp_front.png) | ![contact](assets/example_mujoco_grasp/contact_front.png) | ![grasp](assets/example_mujoco_grasp/grasp_front.png) | ![lift](assets/example_mujoco_grasp/lift_front.png) |
| Gripper above cube | Lowered onto cube | Fingers closed | Cube lifted |

### View Interactive Reports

- MuJoCo grasp: https://miaodx.com/roboharness/grasp/
- G1 WBC reach: https://miaodx.com/roboharness/g1-reach/
- G1 locomotion: https://miaodx.com/roboharness/g1-loco/
- Native LeRobot GR00T: https://miaodx.com/roboharness/g1-native-groot/
- Native LeRobot SONIC: https://miaodx.com/roboharness/g1-native-sonic/
- SONIC planner: https://miaodx.com/roboharness/sonic-planner/
- SONIC tracking: https://miaodx.com/roboharness/sonic/

## Other Demos

These are real integrations and proof surfaces, but they are not the front-door wedge:

| Demo | Description | Report | Run |
|:-----|:------------|:------:|:----|
| **[MuJoCo Grasp](#mujoco-grasp)** | Scripted grasp with Meshcat 3D, paired baseline proof, approval report | [Live](https://miaodx.com/roboharness/grasp/) | `python examples/demos/mujoco/grasp.py --report` |
| **G1 Cross-Framework Proof** | Committed Meshcat vs MuJoCo paired-evidence report for one G1 bundle | repo-only | `python examples/demos/g1/cross_framework_report.py` |
| **[G1 WBC Reach](#g1-humanoid-wbc-reach)** | Whole-body IK reaching (Pinocchio + Pink) | [Live](https://miaodx.com/roboharness/g1-reach/) | `python examples/demos/g1/wbc_reach.py --report` |
| **[G1 Locomotion](#lerobot-g1-locomotion)** | GR00T RL stand→walk→stop, HuggingFace model | [Live](https://miaodx.com/roboharness/g1-loco/) | `python examples/demos/g1/lerobot_locomotion.py --report` |
| **[G1 Native LeRobot (GR00T)](#native-lerobot-integration)** | Official `make_env()` factory + GR00T Balance + Walk | [Live](https://miaodx.com/roboharness/g1-native-groot/) | `python examples/demos/g1/lerobot_native.py --controller groot --report` |
| **[G1 Native LeRobot (SONIC)](#native-lerobot-integration)** | Official `make_env()` factory + SONIC planner | [Live](https://miaodx.com/roboharness/g1-native-sonic/) | `python examples/demos/g1/lerobot_native.py --controller sonic --report` |
| **[SONIC Planner](#sonic-planner)** | Standalone GEAR-SONIC planner demo on G1 | [Live](https://miaodx.com/roboharness/sonic-planner/) | `python examples/demos/sonic/locomotion.py --report` |
| **[SONIC Motion Tracking](#sonic-motion-tracking)** | Real encoder+decoder tracking demo on G1 | [Live](https://miaodx.com/roboharness/sonic/) | `python examples/demos/sonic/tracking.py --report` |

## Showcase Repository

The showcase repo is for external proof that roboharness works as a pip-installed
dependency in real projects:

- **[LeRobot Evaluation](https://github.com/roboharness/showcase/tree/main/lerobot-eval)** — visual regression testing for robot policies
- **[GR00T WBC](https://github.com/roboharness/showcase/tree/main/groot-wbc)** — whole-body control integration

Each showcase is self-contained, runs with `./run.sh`, and supports smoke mode for fast CI validation.

| pre_grasp | contact | grasp | lift |
|:-:|:-:|:-:|:-:|
| ![pre_grasp](assets/example_mujoco_grasp/pre_grasp_front.png) | ![contact](assets/example_mujoco_grasp/contact_front.png) | ![grasp](assets/example_mujoco_grasp/grasp_front.png) | ![lift](assets/example_mujoco_grasp/lift_front.png) |
| Gripper above cube | Lowered onto cube | Fingers closed | Cube lifted |

<details>
<summary><b>G1 Humanoid WBC Reach</b></summary>

```bash
pip install roboharness[demo,wbc]
python examples/demos/g1/wbc_reach.py --report
```

Whole-body control (WBC) for the Unitree G1 humanoid using Pinocchio + Pink differential-IK for upper-body reaching while maintaining lower-body balance. The controller solves inverse kinematics for both arms simultaneously, letting the robot reach arbitrary 3D targets without falling over.

| stand | reach_left | reach_both | retract |
|:-:|:-:|:-:|:-:|
| ![stand](assets/example_g1_wbc_reach/stand_front.png) | ![reach_left](assets/example_g1_wbc_reach/reach_left_front.png) | ![reach_both](assets/example_g1_wbc_reach/reach_both_front.png) | ![retract](assets/example_g1_wbc_reach/retract_front.png) |

</details>

<details>
<summary><b>LeRobot G1 Locomotion</b></summary>

```bash
pip install roboharness[demo]
python examples/demos/g1/lerobot_locomotion.py --report
```

Integrates the real [Unitree G1 43-DOF model](https://huggingface.co/lerobot/unitree-g1-mujoco) from HuggingFace with GR00T WBC locomotion policies (Balance + Walk). The example downloads the model and ONNX policies automatically, runs the G1 through stand → walk → stop phases, and captures multi-camera checkpoints via `RobotHarnessWrapper`.

</details>

<details>
<summary><b>Native LeRobot Integration</b></summary>

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu  # CPU-only
pip install roboharness[demo] lerobot

MUJOCO_GL=egl python examples/demos/g1/lerobot_native.py --controller groot --report
MUJOCO_GL=egl python examples/demos/g1/lerobot_native.py --controller sonic --report
```

Uses LeRobot's official `make_env("lerobot/unitree-g1-mujoco")` factory for standardized env creation. The published native demo reports are split by controller: one report for GR00T and one for SONIC. DDS-ready for sim-to-real transfer when hardware is available. See [#83](https://github.com/MiaoDX/roboharness/issues/83) for details.

</details>

<details>
<summary><b>LeRobot Evaluation in CI</b></summary>

```bash
pip install roboharness[lerobot]

# Evaluate a real LeRobot checkpoint with visual checkpoints + JSON report
python examples/integrations/lerobot/eval_harness.py \
  --checkpoint-path /path/to/lerobot/checkpoint \
  --repo-id lerobot/unitree-g1-mujoco \
  --n-episodes 5 \
  --checkpoint-steps 10 50 100 \
  --assert-threshold \
  --min-success-rate 0.8
```

Produces:
- `episode_000/step_0010/default_rgb.png` — checkpoint screenshots
- `lerobot_eval_report.json` — structured per-episode stats
- CI exit code 1 when thresholds are not met

</details>

<a id="sonic-planner"></a>
<details>
<summary><b>SONIC Planner</b></summary>

```bash
pip install roboharness[demo]
MUJOCO_GL=egl python examples/demos/sonic/locomotion.py --report --assert-success
```

Standalone NVIDIA GEAR-SONIC planner demo on the real Unitree G1 MuJoCo model. This path uses `planner_sonic.onnx` only: velocity commands go in, full-body pose trajectories come out, and the example uses a lightweight virtual torso harness for stable visual debugging. This is the same standalone planner path published at `/sonic-planner/`.

```python
from roboharness.robots.unitree_g1 import SonicLocomotionController, SonicMode

ctrl = SonicLocomotionController()
action = ctrl.compute(
    command={"velocity": [0.3, 0.0, 0.0], "mode": SonicMode.WALK},
    state={"qpos": qpos, "qvel": qvel},
)
```

For a planner demo wired through LeRobot's official `make_env()` stack, see **G1 Native LeRobot (SONIC)** above. The planner path and the encoder+decoder tracking path are different inference stacks with different ONNX contracts; see [docs/product/sonic-inference-stacks.md](docs/product/sonic-inference-stacks.md) for the exact split, validation policy, and joint-order conventions.

</details>

<a id="sonic-motion-tracking"></a>
<details>
<summary><b>SONIC Motion Tracking</b></summary>

```bash
pip install roboharness[demo]
MUJOCO_GL=egl python examples/demos/sonic/tracking.py --report --assert-success
```

Real encoder+decoder tracking demo on the Unitree G1. This path uses `model_encoder.onnx` + `model_decoder.onnx` directly, replays a motion clip via `set_tracking_clip(...)`, and records checkpoint metrics for torso height, tracking-frame progress, and joint-tracking error. This is the same path published at `/sonic/`.

```python
from roboharness.robots.unitree_g1 import MotionClipLoader, SonicLocomotionController

ctrl = SonicLocomotionController()
clip = MotionClipLoader.load("path/to/dance_clip/")
ctrl.set_tracking_clip(clip)

action = ctrl.compute(
    command={"tracking": True},
    state={"qpos": qpos, "qvel": qvel},
)
```

Models (`planner_sonic.onnx`, `model_encoder.onnx`, `model_decoder.onnx`) are downloaded from HuggingFace (`nvidia/GEAR-SONIC`) on first use. Requires `pip install roboharness[demo]`. See [docs/product/sonic-inference-stacks.md](docs/product/sonic-inference-stacks.md) for the exact split between planner and tracking, plus the validation policy and joint-order conventions. See [#86](https://github.com/MiaoDX/roboharness/issues/86) (Phase 1) and [#92](https://github.com/MiaoDX/roboharness/issues/92) (Phase 2).

</details>

### Gymnasium Wrapper (Zero-Change Integration)

```python
import gymnasium as gym
from roboharness.wrappers import RobotHarnessWrapper

env = gym.make("CartPole-v1", render_mode="rgb_array")
env = RobotHarnessWrapper(env,
    checkpoints=[{"name": "early", "step": 10}, {"name": "mid", "step": 50}],
    output_dir="./harness_output",
)

obs, info = env.reset()
for _ in range(200):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    if "checkpoint" in info:
        print(f"Checkpoint '{info['checkpoint']['name']}' captured!")
```

### Core Harness API

```python
from roboharness import Harness
from roboharness.backends.mujoco_meshcat import MuJoCoMeshcatBackend

backend = MuJoCoMeshcatBackend(model_path="robot.xml", cameras=["front", "side"])
harness = Harness(backend, output_dir="./output", task_name="pick_and_place")

harness.add_checkpoint("pre_grasp", cameras=["front", "side"])
harness.add_checkpoint("lift", cameras=["front", "side"])
harness.reset()
result = harness.run_to_next_checkpoint(actions)
# result.views → multi-view screenshots, result.state → joint angles, poses
```

## Supported Simulators

| Simulator | Status | Integration |
|-----------|--------|-------------|
| MuJoCo + Meshcat | ✅ Implemented | Native backend adapter |
| LeRobot (G1 MuJoCo) | ✅ Implemented | Gymnasium Wrapper + Controllers |
| LeRobot Native (`make_env`) | ✅ Implemented | `make_env()` + VectorEnvAdapter |
| Isaac Lab | ✅ Implemented | Gymnasium Wrapper (GPU required for E2E) |
| ManiSkill | ✅ Implemented | Gymnasium Wrapper |
| LocoMuJoCo / MuJoCo Playground / unitree_rl_gym | 📋 Roadmap | Various |

## Design Principles

- **Harness only does "pause → capture → resume"** — agent logic stays in your code
- **Gymnasium Wrapper for zero-change integration** — works with Isaac Lab, ManiSkill, etc.
- **SimulatorBackend protocol** — implement a few methods, plug in any simulator
- **Agent-consumable output** — PNG + JSON files that any coding agent can read

See [STATUS.md](STATUS.md), [ARCHITECTURE.md](ARCHITECTURE.md),
[docs/human/README.md](docs/human/README.md), [CONTRIBUTING.md](CONTRIBUTING.md),
[CHANGELOG.md](CHANGELOG.md), [docs/development/development-workflow.md](docs/development/development-workflow.md),
and [docs/context/context.en.md](docs/context/context.en.md) for current status, architecture,
the curated human doc index, contributor workflow, release notes, and project
background.

## Related Work

Roboharness builds on ideas from several research efforts in AI-driven robot evaluation and code-as-policy:

- **FAEA** — LLM agents as embodied manipulation controllers without demonstrations or fine-tuning ([Tsui et al., 2026](https://arxiv.org/abs/2601.20334))
- **CaP-X** — Benchmark framework for coding agents that program robot manipulation tasks ([Fu et al., 2026](https://arxiv.org/abs/2603.22435))
- **StepEval** — VLM-based subgoal evaluation for scoring intermediate robot manipulation steps ([ElMallah et al., 2025](https://arxiv.org/abs/2509.19524))
- **SOLE-R1** — Video-language reasoning as the sole reward signal for on-robot RL ([Schroeder et al., 2026](https://arxiv.org/abs/2603.28730))
- **AOR** — Multimodal coding agents that iteratively rewrite control code from visual observations ([Kumar, 2026](https://arxiv.org/abs/2603.04466))

## Citing

If you use Roboharness in academic work, please cite it using the metadata in
[`CITATION.cff`](CITATION.cff) or the "Cite this repository" button on GitHub.

## Contributing

Contributions welcome — including from AI coding agents! See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT
