<div align="center">

# Roboharness

**A Visual Testing Harness for AI Coding Agents in Robot Simulation**

[![CI](https://github.com/MiaoDX/RobotHarness/actions/workflows/ci.yml/badge.svg)](https://github.com/MiaoDX/RobotHarness/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/roboharness)](https://pypi.org/project/roboharness/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

> Let Claude Code and Codex **see** what the robot is doing, **judge** if it's working, and **iterate** autonomously.

<table>
<tr>
<td align="center"><b>Front View</b><br/><img src="assets/X32_Y28_Z13_front_view.gif" width="380"/><br/><sub>Plan → Pregrasp → Approach → Close → Lift → Holding</sub></td>
<td align="center"><b>Top-Down View</b><br/><img src="assets/X26_Y22_Z13_topdown_view.gif" width="380"/><br/><sub>Top-down view: object alignment and grasp closure</sub></td>
</tr>
</table>

<p>
  <img src="assets/architecture.svg" width="800" alt="Roboharness Architecture"/>
</p>

### **[View Interactive Visual Reports →](https://miaodx.com/roboharness/)**

*Auto-generated from CI on every push to main — MuJoCo grasp, G1 WBC reach, G1 locomotion, G1 native LeRobot, SONIC.*

</div>

## Demos

| Demo | Description | Report | Run |
|:-----|:------------|:------:|:----|
| **[MuJoCo Grasp](#mujoco-grasp)** | Scripted grasp with Meshcat 3D, multi-view captures | [Live](https://miaodx.com/roboharness/grasp/) | `python examples/mujoco_grasp.py --report` |
| **[G1 WBC Reach](#g1-humanoid-wbc-reach)** | Whole-body IK reaching (Pinocchio + Pink) | [Live](https://miaodx.com/roboharness/g1-reach/) | `python examples/g1_wbc_reach.py --report` |
| **[G1 Locomotion](#lerobot-g1-locomotion)** | GR00T RL stand→walk→stop, HuggingFace model | [Live](https://miaodx.com/roboharness/g1-loco/) | `python examples/lerobot_g1.py --report` |
| **[G1 Native LeRobot](#native-lerobot-integration)** | Official `make_env()` factory + DDS-ready | [Live](https://miaodx.com/roboharness/g1-native/) | `python examples/lerobot_g1_native.py` |
| **[SONIC Motion Tracking](#sonic-locomotion)** | Encoder+decoder pipeline, motion replay from MoCap | [Live](https://miaodx.com/roboharness/sonic/) | Controller API (see below) |

## Installation

```bash
pip install roboharness                  # core (numpy only)
pip install roboharness[demo]            # demo dependencies (MuJoCo, Meshcat, Gymnasium, Rerun, etc.)
pip install roboharness[demo,wbc]        # + whole-body control (Pinocchio, Pink)
pip install roboharness[dev]             # development (pytest, ruff, mypy)
```

## Quick Start

### MuJoCo Grasp

```bash
pip install roboharness[demo]
python examples/mujoco_grasp.py --report
```

| pre_grasp | contact | grasp | lift |
|:-:|:-:|:-:|:-:|
| ![pre_grasp](assets/example_mujoco_grasp/pre_grasp_front.png) | ![contact](assets/example_mujoco_grasp/contact_front.png) | ![grasp](assets/example_mujoco_grasp/grasp_front.png) | ![lift](assets/example_mujoco_grasp/lift_front.png) |
| Gripper above cube | Lowered onto cube | Fingers closed | Cube lifted |

<details>
<summary><b>G1 Humanoid WBC Reach</b></summary>

```bash
pip install roboharness[demo,wbc]
python examples/g1_wbc_reach.py --report
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
python examples/lerobot_g1.py --report
```

Integrates the real [Unitree G1 43-DOF model](https://huggingface.co/lerobot/unitree-g1-mujoco) from HuggingFace with GR00T WBC locomotion policies (Balance + Walk). The example downloads the model and ONNX policies automatically, runs the G1 through stand → walk → stop phases, and captures multi-camera checkpoints via `RobotHarnessWrapper`.

</details>

<details>
<summary><b>Native LeRobot Integration</b></summary>

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu  # CPU-only
pip install roboharness[demo] lerobot

MUJOCO_GL=osmesa python examples/lerobot_g1_native.py --report
```

Uses LeRobot's official `make_env("lerobot/unitree-g1-mujoco")` factory for standardized env creation. DDS-ready for sim-to-real transfer when hardware is available. See [#83](https://github.com/MiaoDX/roboharness/issues/83) for details.

</details>

<details>
<summary><b>SONIC Locomotion</b></summary>

NVIDIA GEAR-SONIC locomotion controller with two modes:

**Planner mode** — velocity commands → full-body pose trajectories (10 Hz planning, 50 Hz output):

```python
from roboharness.robots.unitree_g1 import SonicLocomotionController, SonicMode

ctrl = SonicLocomotionController()
action = ctrl.compute(
    command={"velocity": [0.3, 0.0, 0.0], "mode": SonicMode.WALK},
    state={"qpos": qpos, "qvel": qvel},
)
```

**Tracking mode** — reproduce motion capture clips via encoder+decoder pipeline:

```python
from roboharness.robots.unitree_g1 import SonicLocomotionController, MotionClip

ctrl = SonicLocomotionController()
clip = MotionClip.from_csv_dir("path/to/dance_clip/")
ctrl.set_tracking_clip(clip)

action = ctrl.compute(
    command={"tracking": True},
    state={"qpos": qpos, "qvel": qvel},
)
```

Models (`planner_sonic.onnx`, `model_encoder.onnx`, `model_decoder.onnx`) are downloaded from HuggingFace (`nvidia/GEAR-SONIC`) on first use. Requires `pip install roboharness[demo]`. See [#86](https://github.com/MiaoDX/roboharness/issues/86) (Phase 1) and [#92](https://github.com/MiaoDX/roboharness/issues/92) (Phase 2).

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

See [docs/context.en.md](docs/context.en.md) for full background and motivation.

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
