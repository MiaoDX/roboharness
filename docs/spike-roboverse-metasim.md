# Spike: RoboVerse MetaSim Integration Evaluation

**Issue:** #141
**Date:** 2026-04-10
**Status:** Go (conditional) — see Recommendation below

---

## Context

Roboharness currently supports MuJoCo via a native `SimulatorBackend` implementation and
other simulators (Isaac Lab, ManiSkill, LeRobot) via the Gymnasium wrapper. Each new
simulator requires either a dedicated backend adapter or a Gymnasium-compatible env.

RoboVerse is a unified robotics simulation platform that provides **MetaSim**, an
infrastructure abstracting diverse simulation environments into a universal interface. If
MetaSim's API is stable and well-designed, a single `RoboVerseBackend` could give
roboharness access to 8+ simulators without writing individual adapters.

This document evaluates whether that integration is worth pursuing.

---

## What is RoboVerse?

**Repository:** [RoboVerseOrg/RoboVerse](https://github.com/RoboVerseOrg/RoboVerse)
**Paper:** [arXiv:2504.18904](https://arxiv.org/abs/2504.18904) (RSS 2025)
**Docs:** [roboverse.wiki](https://roboverse.wiki/)
**License:** Apache 2.0
**Version:** 0.1.0
**Activity:** 1,700+ stars, 156 forks, 1,040+ commits, active development

RoboVerse comprises three components:

1. **MetaSim** — Simulation abstraction layer with a Gym-style API (`reset`, `step`,
   `render`, `close`) that works across multiple physics backends via a "Handler" pattern.
2. **RoboVerse Pack** — Standardized dataset of robot manipulation tasks.
3. **RoboVerse Learn** — Training and evaluation pipelines.

### Supported Simulators

| Simulator    | Python Versions | GPU Required | Notes                          |
|-------------|----------------|-------------|--------------------------------|
| MuJoCo      | 3.9-3.13       | No          | Best cross-platform support    |
| SAPIEN 2    | 3.7-3.11       | Yes         | Tactile sensing                |
| SAPIEN 3    | 3.8-3.12       | Yes         | Updated engine                 |
| Genesis     | 3.10-3.12      | Yes         | Differentiable physics         |
| PyBullet    | 3.6-3.11       | No          | Lightweight, CPU-friendly      |
| Newton      | 3.10-3.12      | Yes         | NVIDIA Warp-based              |
| Isaac Sim   | 3.10-3.11      | Yes         | NVIDIA Omniverse               |
| Isaac Gym   | 3.6-3.8        | Yes         | Legacy, Vulkan required        |
| CoppeliaSim | -              | No          | Via PyRep                      |

### Installation

```bash
git clone github.com/RoboVerseOrg/RoboVerse && cd RoboVerse
pip install uv
uv pip install -e ".[mujoco]"           # single simulator
uv pip install -e ".[mujoco,genesis]"   # multiple simulators
```

---

## MetaSim API Analysis

### Architecture

MetaSim uses a **Handler** pattern to abstract simulators. Each simulator implements a
handler that translates the universal API into simulator-specific calls:

```
User Code → Task (Gym-style env) → ScenarioCfg → Handler → Simulator
```

Key abstractions:

- **ScenarioCfg**: Declarative scene configuration (objects, robots, cameras, lights)
  using type-safe config classes.
- **Handler**: Simulator-specific backend (analogous to roboharness `SimulatorBackend`).
- **Task**: Gym-style environment wrapping a handler with reward/success logic.
- **State**: Unified state representation using PyTorch tensors (position, rotation).

### API Surface

```python
# Environment creation (Gym-style)
env = make_env(task_cfg, sim="mujoco")
obs = env.reset()
obs, reward, done, info = env.step(action)
frame = env.render()
env.close()

# Scene configuration (declarative)
scenario = ScenarioCfg(
    objects={"cube": PrimitiveCubeCfg(size=0.05)},
    robot=ArticulationObjCfg(path="franka.usd"),
)

# State uses PyTorch tensors
state = {"cube": {"pos": torch.tensor([0.3, -0.2, 0.05]),
                  "rot": torch.tensor([1.0, 0.0, 0.0, 0.0])}}
```

### Mapping to roboharness SimulatorBackend Protocol

| SimulatorBackend Method | MetaSim Equivalent | Feasibility |
|------------------------|-------------------|-------------|
| `step(action)`         | `env.step(action)` | Direct mapping |
| `get_state()`          | Handler state API  | Likely feasible — unified State concept exists |
| `save_state()`         | Not documented     | Uncertain — may require simulator-specific code |
| `restore_state(state)` | Not documented     | Uncertain — same concern |
| `capture_camera(name)` | `env.render()`     | Partial — render exists but named cameras unclear |
| `get_sim_time()`       | From state dict    | Likely feasible |
| `reset()`              | `env.reset()`      | Direct mapping |

**Key gap:** `save_state()` / `restore_state()` are critical for roboharness checkpoints.
MetaSim's State documentation exists but doesn't clearly expose full physics state
serialization. This would need investigation during a prototype.

---

## Integration Cost Estimate

### Approach A: RoboVerseBackend (native backend)

Write a `RoboVerseBackend` that wraps a MetaSim handler directly.

- **Effort:** ~2-3 days for MuJoCo handler, +1 day per additional simulator
- **Pros:** Direct access to MetaSim internals, full control over state management
- **Cons:** Tight coupling to MetaSim API (v0.1.0 — likely unstable)
- **Risk:** MetaSim API is pre-1.0 and may change significantly

### Approach B: Gymnasium Wrapper (existing path)

MetaSim Tasks are Gym-style environments. If they register as Gymnasium envs, the
existing `RobotHarnessWrapper` works out of the box.

- **Effort:** ~0.5 days to validate + write example
- **Pros:** Zero new code in roboharness core, uses battle-tested wrapper path
- **Cons:** Limited to what Gymnasium env API exposes (no direct state save/restore)
- **Risk:** MetaSim may not fully implement Gymnasium's `render_mode="rgb_array"`

### Approach C: Hybrid

Use Gymnasium wrapper for step/reset/render, add a thin adapter for MetaSim-specific
features (state save/restore, named cameras) when the underlying handler is available.

- **Effort:** ~1-2 days
- **Pros:** Best of both worlds — works immediately via wrapper, enhanced via adapter
- **Cons:** More complex architecture
- **Risk:** Moderate

---

## Risks and Concerns

1. **API Instability (HIGH):** Version 0.1.0 with active development. Breaking changes
   are expected. The "actively evolving" disclaimer in the README confirms this.

2. **Heavy Dependencies (MEDIUM):** RoboVerse pulls in PyTorch, USD, and
   simulator-specific packages. This conflicts with roboharness's "numpy core, optional
   heavy deps" philosophy.

3. **GPU Requirements (MEDIUM):** Most simulators except MuJoCo and PyBullet require
   GPU. This limits CI testing to MuJoCo/PyBullet paths.

4. **Python Version Fragmentation (LOW):** Different simulators support different Python
   ranges. MuJoCo (3.9-3.13) and PyBullet (3.6-3.11) have the broadest support.

5. **State Save/Restore Gap (MEDIUM):** The `save_state`/`restore_state` methods are
   central to roboharness checkpoints. MetaSim's support for this is undocumented and may
   require per-simulator workarounds.

---

## Recommendation

**Go (conditional)** — Proceed with a time-boxed prototype, but only for MuJoCo backend.

### Phase 1: Validate Gymnasium Path (0.5 day)

1. Install RoboVerse with MuJoCo only: `uv pip install -e ".[mujoco]"`
2. Create a MetaSim task env with `render_mode="rgb_array"`
3. Wrap with `RobotHarnessWrapper`
4. Verify checkpoints capture screenshots correctly
5. **Go/no-go:** If this works, proceed to Phase 2. If MetaSim doesn't expose a clean
   Gymnasium interface, stop.

### Phase 2: Prototype RoboVerseBackend (2 days)

1. Implement `RoboVerseBackend` wrapping MetaSim's MuJoCo handler
2. Focus on: `step`, `reset`, `get_state`, `capture_camera`
3. Investigate `save_state`/`restore_state` feasibility
4. Write a showcase example (`examples/roboverse_grasp.py`)
5. Add to docs and update simulator support table

### Do NOT Do

- Do not integrate GPU-only simulators yet (wait for API stability)
- Do not add RoboVerse as a core dependency (keep it optional like MuJoCo)
- Do not track RoboVerse HEAD — pin to a release tag once available
- Do not spend more than 3 days total — if blocked, document and defer

### Success Criteria

- [ ] `RobotHarnessWrapper` works with a MetaSim Gymnasium env
- [ ] At least one example produces visual checkpoint output
- [ ] No new required dependencies added to roboharness core

---

## References

- [RoboVerse GitHub](https://github.com/RoboVerseOrg/RoboVerse) — 1.7k stars, Apache 2.0
- [RoboVerse Paper](https://arxiv.org/abs/2504.18904) — RSS 2025
- [MetaSim Docs](https://roboverse.wiki/metasim/) — v0.1.0
- [roboharness SimulatorBackend Protocol](../src/roboharness/core/harness.py) — 7-method protocol
- [Roadmap §F](./roadmap-2026-q2.md) — Original spike request
