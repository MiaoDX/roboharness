# Roboharness Status

Current as of 2026-05-20.

## What This Repo Is Now

Roboharness is an approval/evidence harness for unattended robot-code changes.
Its core value is compressing a long agent run into a proof pack a human can
review quickly: captures, metrics, alarms, current-vs-baseline evidence, and an
approval decision.

The package currently provides:

- a checkpoint-oriented core harness with semantic task protocols
- a native MuJoCo + Meshcat backend for the maintained grasp wedge
- Gymnasium wrappers for zero-change integration with simulator environments
- LeRobot and Unitree G1 demo/evaluation paths, including SONIC planner and
  tracking examples
- metric evaluation, batch evaluation, trend history, and approval evidence
- bounded agent visual review package preparation, record validation, and
  approval-summary aggregation for the maintained MuJoCo/G1 proof surfaces
- Python-authored project harness contracts that deterministically generate
  drift-checked agent-skill artifacts under `agent-skill/<project-slug>-harness/`
- a CLI for inspecting outputs, writing `report.json`, evaluating reports, and
  tracking trends
- optional MCP tools for agent-driven checkpoint capture and evaluation

## Current Focus

The active milestone is v0.3: the MuJoCo contract-first trust loop.

Current priorities:

- keep the deterministic MuJoCo evaluator corpus grounded
- keep MuJoCo proof-pack assembly, contract grounding, and approval decisions
  localized in the maintained grasp wedge
- keep invalid contracts fail-closed
- ensure ambiguous evidence never self-promotes to pass
- require explicit human blessing before migration-mode baselines become
  authoritative
- keep release truth aligned across repo version, GitHub Releases, and PyPI

Latest validation on 2026-05-20:

- `uv run pytest -q` passed with 533 passed, 13 skipped, and total coverage 91.16%
- `uv run ruff check .` passed
- `uv run ruff format --check .` passed
- `uv run mypy src/` passed
- `uv run python -c "import pytest_cov; print('pytest-cov ok')"` passed

During this milestone, avoid adding new simulator backends, splitting new
showcase repos, expanding SONIC scope without real-model validation evidence, or
prioritizing outreach polish over trust-loop calibration.

## What Can Be Run

Package-first:

```bash
pip install roboharness
roboharness --help
```

Repo MuJoCo wedge:

```bash
python -m pip install -e ".[demo]"
MUJOCO_GL=osmesa python examples/demos/mujoco/grasp.py --report
```

Development gates:

```bash
uv --version
uv pip install -e ".[dev]"
python -c "import pytest_cov; print('pytest-cov ok')"
pytest -q
ruff check .
ruff format --check .
mypy src/
```

Useful Make targets:

```bash
make check-all
make demo-grasp
make demos
make check-gpu
```

## Validation Boundaries

CPU-only environments can validate core logic, wrappers, CLI behavior, report
generation, and mocked Isaac Lab / ManiSkill compatibility.

MuJoCo visual demos need the demo extra and headless rendering via
`MUJOCO_GL=osmesa` or `MUJOCO_GL=egl`.

Isaac Lab end-to-end validation requires NVIDIA RTX GPU hardware. The default
CPU test suite uses mock environments to validate the wrapper contract without
requiring Isaac Lab or GPU access.

## Human Documentation Surface

Humans should start with:

- `README.md` for the product entrypoint, install paths, demos, and examples
- `ARCHITECTURE.md` for subsystem boundaries and extension points
- `STATUS.md` for current state, milestone focus, and runnable surfaces
- `docs/human/README.md` for the curated human documentation index

Planning logs, design reviews, steering notes, generated reports, blog drafts,
and historical strategy docs remain useful evidence, but they are not required
reading for normal review at HEAD.
