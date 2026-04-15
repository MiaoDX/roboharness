# Development Workflow

_Last updated: April 10, 2026_

This document describes how roboharness is developed across different environments, and how AI coding agents (Claude Code, Codex, etc.) fit into the workflow.

---

## Current State

Development uses three tiers, each with distinct strengths and limitations:

| Tier | Environment | GPU | Interactive | Best for |
|------|------------|-----|-------------|----------|
| **Planning** | GitHub Issues | No | Async | Scoping, design, tracking |
| **Implementation** | claude.ai/code (web) | No | Yes, multi-turn | Core logic, tests, refactors, CI config |
| **Validation** | CI (GitHub Actions + Cirun GPU) | CPU + T4 | No (pass/fail) | Automated gating |

### What works well

- **CPU-only code** (core library, storage, reporting, wrappers, tests) — the full write→test→lint→push cycle runs smoothly in claude.ai/code web sessions. Most of the codebase is here.
- **CI gating** — lint, type-check, pytest across Python 3.10–3.13, MuJoCo headless examples, and GPU smoke tests all run automatically on PRs.
- **GitHub Pages** — visual reports are auto-generated and deployed on every push to main, giving a persistent visual record of demo quality.

### The GPU gap

There is no **interactive GPU environment**. When GPU-dependent code breaks (wrong joint mapping, flipped camera, bad action scaling, ONNX model issues), the debugging loop is:

1. Write code blindly in claude.ai/code (no GPU, no MuJoCo rendering feedback)
2. Push and wait for CI (~5–10 min)
3. Download artifacts or check Pages, inspect HTML report
4. Guess at fix, repeat

This outer loop takes 15–20 minutes per iteration for problems that would take 30 seconds to diagnose with a live render.

**Example:** The SONIC ONNX filename issue (PRs #131 → #134 → #159) required three PRs across multiple sessions to resolve what was fundamentally a "try it, see what happens, fix it" problem. With an interactive GPU environment, this would have been a single session.

**Affected areas:**
- Locomotion controllers (GR00T, SONIC, Holosoma) — ONNX model loading, joint target computation
- MuJoCo rendering — camera placement, viewport sizing, depth visualization
- Isaac Lab integration — GPU-only Gymnasium environments
- Visual report quality — can only verify with mock data, not real simulation output

---

## Planned: Local Claude Code CLI on GPU Machines

The Claude Code CLI runs on any local machine with the same capabilities as the web version, plus full access to local hardware (GPU, displays, sensors).

### What it enables

- Run `MUJOCO_GL=egl python examples/sonic_locomotion.py --report` and immediately inspect output
- Multi-turn visual debugging: "the robot is falling over" → inspect report → adjust gains → re-run → verify, all in one session
- Test ONNX model loading, GPU rendering, Isaac Lab integration with real hardware
- Catch visual regressions that no unit test can detect (flipped axes, wrong camera FOV, static robot)

### Proposed four-tier workflow

| Tier | Environment | GPU | Interactive | Best for |
|------|------------|-----|-------------|----------|
| **Planning** | GitHub Issues | No | Async | Scoping, design, tracking |
| **CPU implementation** | claude.ai/code (web) | No | Yes | Core logic, tests, non-GPU code |
| **GPU implementation** | Claude Code CLI + local GPU | Yes | Yes | Demo debugging, visual QA, ONNX/rendering |
| **Validation** | CI (GitHub Actions + Cirun) | CPU + T4 | No | Automated gating, regression detection |

### When to use which tier

**Use claude.ai/code (web) when:**
- Writing or refactoring core library code (`src/roboharness/`)
- Adding tests that mock the simulator backend
- Fixing lint/type/CI issues
- Updating documentation, CI config, Pages workflow
- Any work that doesn't require GPU or real rendering output

**Use local CLI + GPU when:**
- Debugging locomotion controllers (GR00T, SONIC, Holosoma)
- Fixing visual report rendering issues
- Developing new simulator backends or camera configurations
- Working on Isaac Lab integration
- Any task where you need to see what the robot is actually doing

**Use CI for:**
- Final validation before merge (automated, not interactive)
- GPU smoke tests on PRs (via `gpu-test` label or path-filter trigger)
- Cross-Python-version compatibility

### Practical setup

Any Linux machine with an NVIDIA GPU works. A GTX 1060 is sufficient for MuJoCo rendering and ONNX inference.

```bash
# Install Claude Code CLI
npm install -g @anthropic-ai/claude-code

# Clone and install roboharness with all deps
git clone https://github.com/MiaoDX/roboharness.git
cd roboharness
pip install -e ".[demo,dev,wbc]"

# For SONIC/Isaac Lab: install PyTorch with CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Verify GPU rendering
MUJOCO_GL=egl python -c "import mujoco; print('MuJoCo OK')"
nvidia-smi
```

Then run `claude` in the repo directory. The CLI reads `CLAUDE.md` for project context automatically.

Cloud GPU instances also work (Lambda, Vast.ai, GCP spot at ~$0.30–0.80/hr for a T4). Spin up when working on GPU-dependent tasks, tear down after.

---

## Workflow Examples

### Example 1: CPU-only fix (sort order bug)

**Environment:** claude.ai/code web session

1. GitHub issue reports checkpoint ordering is wrong
2. Web session: read `reporting.py`, identify alphabetical sort, fix to step-number sort
3. Write test with mock checkpoint data, verify sort order
4. Push, CI passes, merge

No GPU needed — the fix is pure Python logic.

### Example 2: GPU-dependent fix (SONIC controller not producing robot motion)

**Ideal environment:** Local CLI + GPU

1. GitHub issue reports static robot in SONIC demo
2. Local session: run `python examples/sonic_locomotion.py --report`
3. Open report — see the robot is indeed static
4. Inspect controller output, find action mapping bug
5. Fix, re-run, visually confirm robot moves
6. Push, CI validates, merge

**What actually happened (web-only):** Took 3 PRs (#131, #134, #159) across multiple sessions because each iteration required a blind push + CI wait + artifact download cycle.

### Example 3: Mixed workflow (new locomotion controller)

1. **GitHub Issue** — define requirements, link to paper/model
2. **claude.ai/code** — implement controller class, write unit tests with mocked ONNX, add to `__init__.py` exports
3. **Local CLI + GPU** — run actual demo, verify joint targets produce expected motion, tune gains, fix visual issues
4. **Push + CI** — final validation across Python versions, merge

---

## Relationship to Other Docs

- **`docs/ci-strategy.md`** — Details on GPU CI infrastructure (Cirun + GCP), cost analysis, platform comparison
- **`docs/roadmap-2026-q2.md`** — Project technical direction and priorities
- **`docs/ecosystem-strategy.md`** — Community and distribution strategy
- **`CLAUDE.md`** — Agent instructions (read automatically by Claude Code CLI and web)
