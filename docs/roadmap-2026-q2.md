# Roboharness Project Roadmap — 2026 Q2+

_Last updated: April 10, 2026_
_This document covers the project's technical direction and priority ordering. Community and distribution strategy is in `ecosystem-strategy.md`._
_No timing estimates — priorities determine what to work on, AI coding agents determine how fast it ships._

---

## Current State

v0.1.1 on PyPI. Core capabilities:

- Checkpoint multi-view screenshots + structured JSON
- MuJoCo + Meshcat backend
- Gymnasium wrapper (zero-change integration for Isaac Lab, ManiSkill)
- CLI (inspect / report)
- Rerun integration
- LeRobot G1 locomotion (wrapper mode + native `make_env` mode)
- SONIC locomotion controller (planner + tracking modes)
- G1 WBC reach (Pinocchio + Pink)
- MCP server + tools (#70)
- CI: CPU lint/type/test + GPU smoke test (Cirun/GCP)
- GitHub Pages auto-generated HTML visual reports

---

## Priority Framework

Each direction is evaluated against two criteria:

1. **Does it help acquire users?** — Makes new users discover and use roboharness
2. **Does it help build technical influence?** — Makes roboharness cited and discussed at the AI + Robotics intersection

Directions are split into "do now" and "do later." "Do now" means the next available agent session should pick this up.

---

## Do Now

### A. LeRobot Evaluation Plugin

**Why highest priority:** LeRobot has 22,000+ stars, an ICLR 2026 paper, and is the de facto standard in the robot learning community. Its evaluation infrastructure has clear gaps:

- No headless visual evaluation (issue #538)
- Evaluation script uses fixed initial states (issue #2375)
- Multiple issues report inability to reproduce benchmark results with no diagnostic tooling (#2354, #1507, #2259)
- No CI gate, no pass/fail thresholds, no regression detection between policy versions

**What to do:**

1. Create `roboharness[lerobot]` optional dependency
2. Wrap `lerobot-eval` output, insert checkpoint screenshots at key steps
3. Produce structured JSON (per-episode success, per-step screenshot paths, key metrics)
4. Provide CI-friendly pass/fail gate (e.g., success rate < threshold → exit 1)
5. Ship a LeRobot-compatible example: `pip install roboharness[lerobot] && python examples/lerobot_eval_harness.py`

**Exit criteria:** A LeRobot user can run one command to get visual regression testing in CI.

**Status:** Complete. `examples/lerobot_eval_harness.py --checkpoint-path <path> --repo-id <repo>` loads real LeRobot policies, captures checkpoint screenshots, produces `lerobot_eval_report.json`, and supports `--assert-threshold` for CI pass/fail gates.

**Related issues:** New issue needed. Extends #83 (native LeRobot integration).

### B. Constraint Evaluator

**Why:** Upgrading from "can see" to "can judge" is the core value leap. The `evaluate/` module has code but no end-to-end demo yet.

**What to do:**

1. Complete the YAML constraint → pass/fail flow described in `docs/p1-constraint-evaluator.md`
2. Demo constraint evaluation on mujoco_grasp and g1_wbc_reach
3. Add pass/fail markers (green/red) to HTML reports
4. Provide `roboharness evaluate --constraints grasp_default.yaml` CLI path

**Exit criteria:** CI-generated HTML reports show which checkpoints passed constraints and which failed.

**Related issues:** `docs/p1-constraint-evaluator.md` has the design doc.

### C. Showcase Repository (github.com/roboharness/showcase)

**Why:** GR00T N1.6, Pi0, LeRobot, SONIC, etc. need to demonstrate roboharness integration, but embedding them in the core repo would bloat it and explode dependencies.

**Structure:**

```
roboharness/showcase/
├── README.md                     # Overview + run instructions
├── groot-n16/
│   ├── README.md
│   ├── requirements.txt          # groot deps + roboharness
│   ├── run.sh                    # one-command demo
│   ├── harness_config.yaml       # checkpoint definitions
│   └── groot_visual_test.py      # thin wrapper, not a submodule
├── pi0-libero/
│   ├── README.md
│   ├── requirements.txt
│   ├── run.sh
│   └── pi0_eval_harness.py
├── lerobot-g1/
│   └── ...
├── sonic-locomotion/
│   └── ...
├── capx-comparison/              # Comparison demo with CaP-X
│   └── ...
└── .github/
    └── workflows/
        └── showcase-ci.yml       # Each showcase is an independent CI job
```

**Key design decisions:**

- **No git submodules**: Use requirements.txt + runtime pip install to pull dependencies. Avoids submodule hell.
- **Each showcase is self-contained**: `cd showcase/groot-n16 && pip install -r requirements.txt && python groot_visual_test.py`
- **roboharness itself is a pip dependency**, not a submodule. Showcase always uses the PyPI release.
- **CI matrix**: Each showcase is an independent job — one failing doesn't block others.

**Benefits:**

- Core repo stays self-contained, no bloat
- Users see "roboharness works with GR00T / Pi0 / LeRobot" with runnable demos
- Issue #91 scope shifts from "support every framework in roboharness" to "showcase integrations in a dedicated repo"
- Version changes in large models/frameworks don't break core CI

**Prerequisites:** GitHub org `roboharness` already exists (no repos yet).

**⚡ Action items:**

- [ ] Create `.github` repo in `roboharness` org with `profile/README.md` (org landing page)
- [ ] Create `roboharness/showcase` repo
- [ ] Initialize skeleton: README + groot-n16/ + pi0-libero/ + lerobot-g1/ directories
- [ ] First runnable showcase: extract `examples/lerobot_g1_native.py` as standalone showcase
- [ ] Set up CI: GitHub Actions matrix, one job per showcase

**Exit criteria:** Showcase repo has at least 3 runnable showcases (GR00T, Pi0, LeRobot), each with CI.

---

## Do Later

### D. VLM Judge Integration

Add an optional VLM evaluation path in the evaluate module. Use open-source RoboReward 4B models or StepEval's per-subgoal pattern to auto-score checkpoint screenshots.

**Prerequisites:** Constraint Evaluator ships first. VLM judge is an enhancement layer on top.

**Open question:** Which VLM? RoboReward 4B is purpose-trained but may not generalize; general VLMs (Claude/GPT-4o) are expensive but more capable. Likely need two paths: local small model for fast CI checks, cloud model for deep analysis.

### E. Newton Physics Engine Support

NVIDIA released Newton 1.0 at GTC 2026, built on Warp, claiming 475x faster than MJX, Apache 2.0, Linux Foundation governance. If it becomes the new standard GPU physics engine, roboharness needs to support it.

**Not urgent.** Newton just hit 1.0 and the API may still be changing. Monitor community adoption first. Roboharness's MuJoCo support + Gymnasium wrapper already covers most scenarios.

**Spike complete:** `docs/spike-newton-backend.md` documents the integration plan,
SimulatorBackend protocol mapping, CI requirements, and concrete adoption criteria
for when to begin implementation. Isaac Lab users already get Newton coverage today
via `RobotHarnessWrapper` (Isaac Lab's Newton backend exposes a Gymnasium interface).

### F. RoboVerse MetaSim Evaluation

RoboVerse provides a unified API across 8+ simulators (Isaac Lab, MuJoCo, SAPIEN, Genesis, PyBullet). If the API is stable, one integration = support for 8 simulators.

**Needs a spike first:** Spend one session actually trying RoboVerse, evaluating API stability and integration cost. If viable, this could be the highest-ROI technical direction.

### G. MCP Server Enhancements

Current #70 provides a basic MCP server. Enhancement directions:

- Connect to the `evaluate` module so agents get pass/fail results via MCP
- Support returning checkpoint screenshots via MCP (base64 images)
- Consider registering in the official MCP directory

---

## Not Doing

Explicitly listed to prevent scope creep:

- **Not a training framework**: roboharness is a testing harness, not a training pipeline
- **Not real-time control**: No robot teleop or remote control
- **No heavy framework dependencies in core**: GR00T/Pi0/Isaac Lab etc. live in the showcase repo
- **No multi-agent orchestration**: Not a Cursor-style parallel agent coordination system
- **No auto-generated AGENTS.md / CLAUDE.md**: ETH Zurich research showed auto-generation performs worse

---

## Issue Status Reference

| Direction | Related Issue | Status |
|-----------|--------------|--------|
| LeRobot eval plugin | To be created | New direction |
| Constraint Evaluator | docs/p1-constraint-evaluator.md | Design complete, awaiting implementation |
| Showcase repo | Redefines #91 | New approach |
| VLM judge | To be created | Design phase |
| Newton support | None | Monitoring |
| RoboVerse evaluation | None | Needs spike |
| MCP enhancements | #70 | Basics complete |
| GPU CI | #18 | Basics complete, Cirun + GCP working |
| Isaac Lab validation | #4 | Complete |
| ManiSkill validation | #5 | Complete |
| SONIC | #86, #92 | Phase 1+2 complete |
