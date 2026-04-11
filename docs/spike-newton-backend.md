# Spike: Newton Physics Engine Backend

**Issue:** #143
**Date:** 2026-04-11
**Status:** Monitor — revisit Q3/Q4 2026 or when adoption criteria below are met

---

## Context

Roboharness currently supports MuJoCo via `MuJoCoMeshcatBackend` and all
Gymnasium-compatible simulators (Isaac Lab, ManiSkill, LeRobot) via
`RobotHarnessWrapper`. Neither path covers Newton directly.

NVIDIA released Newton 1.0 at GTC 2026. If Newton becomes the dominant GPU
physics engine for robot training, roboharness needs a `NewtonBackend`.

This document captures what is known today and defines the criteria for
when to begin implementation.

---

## What is Newton?

**Repository:** [newton-physics/newton](https://github.com/newton-physics/newton)
**Docs:** https://newton-physics.github.io/newton/latest/
**License:** Apache 2.0
**Version:** 1.0 GA (released March 2026, GTC 2026)
**Governance:** Linux Foundation — co-stewarded by NVIDIA, Google DeepMind, and
Disney Research

Newton is a GPU-accelerated, open-source physics simulation engine built on
[NVIDIA Warp](https://developer.nvidia.com/warp-python). It bundles multiple
physics solvers behind a unified Python API, with a focus on contact-rich
manipulation and locomotion tasks.

### Key capabilities

- **MuJoCo Warp (MJWarp)** — extends Google DeepMind's MuJoCo with GPU
  parallelisation: 252× speedup for locomotion, 475× for manipulation (vs MJX on
  RTX PRO 6000 Blackwell)
- **Differentiable simulation** — gradient-based optimisation over physics
- **OpenUSD support** — scene description via Universal Scene Description
- **Sensor integration** — contact, camera, IMU sensors
- **Multiple viewers** — `gl` (OpenGL), `usd` (USD export), `rerun` (ReRun
  time-series), `null` (headless / CI)
- **40+ built-in examples** — pendulum, humanoid locomotion, quadrupeds,
  manipulators, cloth, cable deformables

### Installation

```bash
# PyPI (stable)
pip install "newton[examples]"

# From source (dev)
git clone https://github.com/newton-physics/newton
uv pip install -e ".[examples]"
```

### Hardware requirements

| Requirement | Minimum |
|-------------|---------|
| OS | Linux x86-64 / aarch64, Windows x86-64 (macOS CPU-only) |
| GPU | NVIDIA Maxwell or newer |
| Driver | 545+ (CUDA 12) |
| Python | 3.10+ |

---

## Current Adoption (April 2026)

| Metric | Value |
|--------|-------|
| GitHub stars | ~2 000 (growing post-GTC 2026 announcement) |
| Isaac Lab integration | Yes — Newton is the "kit-less" backend in Isaac Lab v3.0 |
| RoboVerse MetaSim support | Yes — listed as a supported simulator |
| Papers citing Newton | Primarily NVIDIA announcements; no independent benchmarks yet |
| API stability | 1.0 GA but documentation is incomplete; breaking changes possible |

**Assessment:** Newton is gaining traction within the NVIDIA ecosystem (Isaac Lab
already uses it). Broader community adoption outside NVIDIA projects is still
forming. API documentation is sparse as of April 2026.

---

## Integration Analysis

There are two integration paths, from fastest to most custom.

### Option A: Via RoboVerse MetaSim (fastest, recommended if MetaSim stabilises)

RoboVerse MetaSim already wraps Newton as one of its 8+ simulator backends. If
the MetaSim spike (see `docs/spike-roboverse-metasim.md`) results in a
`RoboVerseBackend`, Newton support comes for free — no Newton-specific code needed.

**Effort:** 0 Newton-specific lines (reuses the RoboVerse adapter)
**Prerequisite:** RoboVerse MetaSim integration complete (see issue #141 spike)

### Option B: Direct `NewtonBackend` (more control, works without RoboVerse)

Implement the 7-method `SimulatorBackend` protocol directly against Newton's API.
Newton exposes a Warp-based simulation environment with state access and rendering.

**Estimated effort:** ~250–400 lines (similar to `MuJoCoMeshcatBackend`)
**Prerequisite:** Newton API must be stable enough to code against (check Q3 2026)

---

## SimulatorBackend Protocol Mapping (Option B)

| `SimulatorBackend` method | Newton equivalent | Notes |
|--------------------------|------------------|-------|
| `step(action)` | `sim.step()` + set joint targets | Action → joint position/velocity targets via Warp array |
| `get_state()` | Read Warp body/joint state arrays | `qpos`, `qvel`, contact forces; requires `.numpy()` to move off GPU |
| `save_state()` | Copy Warp state arrays | `warp.clone()` or manual numpy snapshot |
| `restore_state(state)` | Write Warp state arrays | Push saved numpy state back to GPU |
| `capture_camera(name)` | Sensor camera render (`--viewer null`) | Newton's camera sensor returns a pixel buffer; use `null` viewer for headless CI |
| `get_sim_time()` | `sim.time` | Float scalar |
| `reset()` | `sim.reset()` | Returns initial obs dict |

**Key differences from MuJoCo:**
- All arrays live on GPU (Warp arrays); GPU→CPU transfer needed for `get_state()`
- Rendering requires `CUDA` context; headless CI must use `--viewer null` (no osmesa needed)
- State save/restore involves Warp array cloning (no built-in MuJoCo-style `save_state`)

---

## CI Requirements

Newton requires a GPU for all non-macOS usage. The existing Cirun + GCP setup
(used for the Isaac Lab GPU tests) already satisfies this:

```yaml
# .github/workflows/ci.yml addition (future)
newton-test:
  runs-on: cirun-gpu-runner
  env:
    NEWTON_DEVICE: cuda:0
  steps:
    - run: pip install "newton[examples]" roboharness
    - run: pytest tests/backends/test_newton.py
```

No new CI infrastructure is needed — the GPU runner already exists.

---

## Adoption Criteria

Implement `NewtonBackend` when **two or more** of the following are true:

1. **Stars ≥ 5 000** on `newton-physics/newton` (broad community interest beyond NVIDIA)
2. **API v2.0** released with stable, documented `SimulatorBackend`-compatible interface
3. **Isaac Lab drops MuJoCo** as primary CPU backend in favour of Newton/MJWarp
4. **A community request** is filed in roboharness issues requesting Newton support
5. **RoboVerse MetaSim path blocked** (MetaSim doesn't support Newton cleanly)

If none of the criteria are met by **Q4 2026**, close this issue as "deferred
indefinitely" and rely on MuJoCo + Gymnasium wrapper coverage.

---

## Fastest Path to Newton Coverage Today

Without writing any Newton-specific code, roboharness already has partial Newton
coverage via the **Gymnasium wrapper path**:

- Isaac Lab `NewtonVisualizer` environments expose a `gymnasium.Env` interface
- `RobotHarnessWrapper` wraps any `gym.Env` — zero code changes needed

So an Isaac Lab user training with the Newton backend already gets roboharness
checkpoints and visual reports "for free" via `RobotHarnessWrapper`.

---

## Open Questions

1. **State save/restore**: Does Newton provide built-in deterministic state
   snapshots (like MuJoCo's `mj_copyData`)? Or must we snapshot Warp arrays
   manually? Answer unknown as of April 2026 — documentation does not address this.

2. **Camera sensor API**: Newton's camera sensor API is not yet publicly documented.
   The `--viewer rerun` path exists but is high-latency for CI. The `null` viewer
   should expose a pixel buffer directly but this needs verification.

3. **macOS support**: Newton runs CPU-only on macOS. Should `NewtonBackend` fall
   back to CPU mode for non-NVIDIA systems? This affects developer ergonomics.

4. **MJWarp vs native Newton solver**: Newton bundles MJWarp (MuJoCo on Warp) as a
   solver. For roboharness users who want MuJoCo compatibility with GPU speed,
   using MJWarp inside Newton is likely the right default. Needs an API check.

---

## References

- Newton GitHub: https://github.com/newton-physics/newton
- Newton blog announcement: https://developer.nvidia.com/blog/announcing-newton-an-open-source-physics-engine-for-robotics-simulation/
- Isaac Lab Newton integration: https://isaac-lab.github.io/
- RoboVerse MetaSim spike: `docs/spike-roboverse-metasim.md`
- Roadmap §E: `docs/roadmap-2026-q2.md`
