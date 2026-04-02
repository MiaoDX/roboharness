# 🤖 Robot-Harness

**A Visual Testing Harness for AI Coding Agents in Robot Simulation**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/MiaoDX/RobotHarness)](https://github.com/MiaoDX/RobotHarness/stargazers)

> Let Claude Code and Codex **see** what the robot is doing, **judge** if it's working, and **iterate** autonomously.

## What is Robot-Harness?

Robot-Harness is a framework that lets AI Coding Agents (Claude Code, OpenAI Codex, etc.) control robot simulations through a visual feedback loop:

```
Agent writes control code
    → Harness runs simulation
    → Pauses at checkpoints
    → Captures multi-view screenshots
    → Agent observes & judges
    → Agent modifies code & retries
    → Loop until task succeeds
```

**Key insight**: Modern coding agents are already multimodal — they can write code AND see images AND make decisions. We don't need a separate VLM. Robot-Harness just needs to present simulation visuals in a format agents can directly consume.

## Features

- 🔍 **Step-by-step simulation control** — Pause at critical checkpoints
- 📸 **Multi-view screenshot capture** — RGB + depth from multiple camera angles
- 🧠 **Agent-driven judgment** — Agent observes screenshots and decides next action
- 🔄 **Autonomous iteration** — Agent modifies code and reruns based on visual feedback
- 🔌 **Multi-simulator support** — MuJoCo, Isaac Lab, ManiSkill, and more

## Supported Simulators

| Simulator | Status | Integration |
|-----------|--------|-------------|
| MuJoCo + Meshcat | ✅ Verified | Native (current practice) |
| Isaac Lab | 🚧 Planned | Gymnasium Wrapper (1 line) |
| ManiSkill | 🚧 Planned | Gymnasium Wrapper |
| LocoMuJoCo | 📋 Roadmap | Gymnasium Wrapper |
| MuJoCo Playground | 📋 Roadmap | JAX-native adapter |
| unitree_rl_gym | 📋 Roadmap | MuJoCo sim-to-sim wrapper |

## Quick Start

```python
# Coming soon
from robot_harness import Harness

harness = Harness(simulator="mujoco", task="pick_and_place")
harness.add_checkpoint("plan_start", cameras=["front", "side", "top"])
harness.add_checkpoint("grasp_contact", cameras=["front", "wrist"])
harness.add_checkpoint("lift_complete", cameras=["front", "side", "top"])

# Agent loop
while not task_complete:
    harness.run_to_next_checkpoint()
    screenshots = harness.capture()
    state = harness.get_state()
    decision = agent.judge(screenshots, state)
    if decision.needs_fix:
        agent.modify_code(decision.feedback)
        harness.reset_to_checkpoint(decision.checkpoint)
```

## Project Structure

```
RobotHarness/
├── docs/
│   ├── context.md              # Full project context & motivation
│   └── simulator-survey.md     # Open-source simulator compatibility analysis
├── src/                        # Source code (coming soon)
├── examples/                   # Example scripts (coming soon)
└── README.md
```

## Background

This project is inspired by:
- **Anthropic's Harness Engineering** (Nov 2025, Mar 2026) — Building effective harnesses for long-running agents
- **OpenAI's Harness Engineering** — Using Codex in an agent-first world
- **AOR** (Act-Observe-Rewrite, 2025) — Multi-modal LLM receives RGB images + diagnostics, outputs controller code

See [docs/context.md](docs/context.md) for the full background.

## Contributing

This project is in early development. Contributions welcome!

## License

MIT
