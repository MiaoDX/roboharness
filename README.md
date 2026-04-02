<div align="center">

# Robot-Harness

**A Visual Testing Harness for AI Coding Agents in Robot Simulation**

[![CI](https://github.com/MiaoDX/RobotHarness/actions/workflows/ci.yml/badge.svg)](https://github.com/MiaoDX/RobotHarness/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![GitHub stars](https://img.shields.io/github/stars/MiaoDX/RobotHarness?style=social)](https://github.com/MiaoDX/RobotHarness/stargazers)

> Let Claude Code and Codex **see** what the robot is doing, **judge** if it's working, and **iterate** autonomously.

<table>
<tr>
<td align="center"><b>Grasp: X32_Y28_Z13 (Front View)</b><br/><img src="assets/X32_Y28_Z13_front_view.gif" width="380"/><br/><sub>Plan → Pregrasp → Approach → Close → Lift → Holding</sub></td>
<td align="center"><b>Grasp: X26_Y22_Z13 (Top-Down View)</b><br/><img src="assets/X26_Y22_Z13_topdown_view.gif" width="380"/><br/><sub>Top-down view: object alignment and grasp closure</sub></td>
</tr>
</table>

</div>

## What is Robot-Harness?

Robot-Harness is a framework that lets AI Coding Agents (Claude Code, OpenAI Codex, [OpenClaw](https://github.com/openclaw/openclaw), etc.) control robot simulations through a **visual feedback loop**:

<p align="center">
  <img src="assets/architecture.svg" width="800" alt="Robot-Harness Architecture"/>
</p>

**Key insight**: Modern coding agents are already multimodal — they can write code AND see images AND make decisions. We don't need a separate VLM. Robot-Harness just needs to present simulation visuals in a format agents can directly consume.

## Installation

```bash
pip install robot-harness

# With MuJoCo + Meshcat backend
pip install robot-harness[mujoco]

# Development
pip install robot-harness[dev]
```

## Quick Start

### MuJoCo Grasp Example (End-to-End)

Run a complete grasp simulation with zero external dependencies:

```bash
pip install robot-harness[mujoco] Pillow
python examples/mujoco_grasp.py --report
```

This runs a scripted grasp sequence, captures multi-view screenshots at each checkpoint, and generates an HTML report. See [`examples/mujoco_grasp.py`](examples/mujoco_grasp.py) for the full source.

> **[View the interactive visual report online](https://miaodx.github.io/RobotHarness/)** — auto-generated from CI on every push to main.

**Checkpoint captures (front view):**

| pre_grasp | contact | grasp | lift |
|:-:|:-:|:-:|:-:|
| ![pre_grasp](assets/example_mujoco_grasp/pre_grasp_front.png) | ![contact](assets/example_mujoco_grasp/contact_front.png) | ![grasp](assets/example_mujoco_grasp/grasp_front.png) | ![lift](assets/example_mujoco_grasp/lift_front.png) |
| Gripper hovering above cube | Lowered onto cube | Fingers closed | Cube lifted off table |

### Option 1: Gymnasium Wrapper (Zero-Change Integration)

Wrap any Gymnasium-compatible environment with one line:

<details>
<summary><b>Show code example</b></summary>

```python
import gymnasium as gym
from robot_harness.wrappers import RobotHarnessWrapper

env = gym.make("CartPole-v1", render_mode="rgb_array")
env = RobotHarnessWrapper(env,
    checkpoints=[
        {"name": "early", "step": 10},
        {"name": "mid", "step": 50},
        {"name": "late", "step": 100},
    ],
    output_dir="./harness_output",
)

obs, info = env.reset()
for _ in range(200):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    if "checkpoint" in info:
        print(f"Checkpoint '{info['checkpoint']['name']}' captured!")
        print(f"  → {info['checkpoint']['capture_dir']}")
```
</details>

### Option 2: Core Harness API (Full Control)

For custom simulator integrations:

<details>
<summary><b>Show code example</b></summary>

```python
from robot_harness import Harness
from robot_harness.backends.mujoco_meshcat import MuJoCoMeshcatBackend

backend = MuJoCoMeshcatBackend(
    model_path="robot.xml",
    cameras=["front", "side", "top"],
)
harness = Harness(backend, output_dir="./harness_output", task_name="pick_and_place")

harness.add_checkpoint("pre_grasp", cameras=["front", "side", "top"])
harness.add_checkpoint("contact", cameras=["front", "wrist"])
harness.add_checkpoint("lift", cameras=["front", "side", "top"])

harness.reset()
result = harness.run_to_next_checkpoint(actions)
# result.views → multi-view screenshots
# result.state → joint angles, poses, contacts
```
</details>

### Grasp Task Storage

For tasks with multiple grasp positions, each with multiple agent retry trials:

<details>
<summary><b>Show code example and directory structure</b></summary>

```python
from robot_harness.storage import GraspTaskStore

store = GraspTaskStore(base_dir="./output", task_name="pick_and_place")
store.add_grasp_position(position_id=1, xyz=(0.5, 0.0, 0.05), object_name="red_cube")
```

Output directory structure:

```
harness_output/
└── pick_and_place/
    ├── task_config.json
    ├── grasp_position_001/
    │   ├── position.json              # grasp pose (xyz + quaternion)
    │   ├── trial_001/
    │   │   ├── plan_start/
    │   │   │   ├── front_rgb.png
    │   │   │   ├── side_rgb.png
    │   │   │   ├── state.json
    │   │   │   └── metadata.json
    │   │   ├── contact/
    │   │   ├── lift/
    │   │   └── result.json
    │   ├── trial_002/                 # agent's second attempt
    │   └── summary.json
    ├── grasp_position_002/
    └── report.json
```
</details>

## Supported Simulators

| Simulator | Status | Integration |
|-----------|--------|-------------|
| MuJoCo + Meshcat | ✅ Implemented | Native backend adapter |
| Isaac Lab | 🚧 Planned | Gymnasium Wrapper (1 line) |
| ManiSkill | 🚧 Planned | Gymnasium Wrapper |
| LocoMuJoCo | 📋 Roadmap | Gymnasium Wrapper |
| MuJoCo Playground | 📋 Roadmap | JAX-native adapter |
| unitree_rl_gym | 📋 Roadmap | MuJoCo sim-to-sim wrapper |

## Architecture

<details>
<summary><b>Project structure</b></summary>

```
robot_harness/
├── core/
│   ├── harness.py         # Main Harness class + SimulatorBackend protocol
│   ├── checkpoint.py      # Checkpoint management & state snapshots
│   └── capture.py         # Multi-view screenshot capture & storage
├── backends/
│   └── mujoco_meshcat.py  # MuJoCo + Meshcat reference backend
├── wrappers/
│   └── gymnasium_wrapper.py  # Drop-in Gymnasium wrapper
└── storage/
    └── task_store.py      # Task-oriented storage (GraspTaskStore, etc.)
```
</details>

**Design principles:**
- **Harness only does "pause → capture → resume"** — agent logic stays in your code
- **Gymnasium Wrapper for zero-change integration** — works with Isaac Lab, ManiSkill, etc.
- **SimulatorBackend protocol for custom integrations** — implement 7 methods, done
- **Agent-consumable output** — PNG images + JSON state files that any agent can `ls` and read

## Background

This project is inspired by:
- **[Anthropic's Harness Engineering](https://www.anthropic.com/engineering/claude-code-best-practices)** (Nov 2025, Mar 2026) — Building effective harnesses for long-running agents
- **[OpenAI's Codex CLI](https://github.com/openai/codex)** — Using Codex in an agent-first world
- **[AOR](https://arxiv.org/abs/2504.09837)** (Act-Observe-Rewrite, 2025) — Multi-modal LLM receives RGB images + diagnostics, outputs controller code
- **[MuJoCo](https://mujoco.org/)** — Physics engine for robotics simulation
- **[Gymnasium](https://gymnasium.farama.org/)** — Standard API for reinforcement learning environments

See [docs/context.en.md](docs/context.en.md) for the full background and motivation.
See [docs/simulator-survey.en.md](docs/simulator-survey.en.md) for the simulator compatibility analysis.

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

We especially welcome:
- New simulator backend adapters
- Real-world usage examples
- Integration with popular RL libraries (SB3, CleanRL, etc.)

**AI agents are welcome contributors!** We actively encourage contributions from AI coding agents such as [Claude Code](https://docs.anthropic.com/en/docs/claude-code), [OpenAI Codex](https://github.com/openai/codex), [OpenClaw](https://github.com/openclaw/openclaw), and other autonomous coding tools. If your agent can improve Robot-Harness, send a PR!

## License

MIT
