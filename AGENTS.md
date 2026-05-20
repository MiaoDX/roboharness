# AGENTS.md

Repo-wide operating instructions for coding agents. Scope: this entire repo.

## First Read

Before running commands, read:

1. `AGENTS.md`
2. `CLAUDE.md`
3. These `pyproject.toml` sections:
   - `[project.optional-dependencies]`
   - `[tool.pytest.ini_options]`
   - `[tool.coverage.run]`

If instructions conflict, use this priority:
**system/developer/user prompt > AGENTS.md > CLAUDE.md > inferred defaults**.

Before broad changes, read `docs/steering/current.md`. Before resuming existing
work, read `STATUS.md`. For architecture changes, read `ARCHITECTURE.md`. For
public docs, examples, packaging, or Hugging Face Space changes, read the
relevant `README.md` first. Prefer existing `Makefile` targets when they fit.

## Skill Routing

When a request clearly matches an available skill, invoke that skill first.

- Product ideas, brainstorming, "worth building" -> `office-hours`
- Bugs, errors, broken behavior, 500s -> `investigate`
- Ship, deploy, push, create PR -> `ship`
- QA or site testing -> `qa`
- Code review or diff review -> `review`
- Docs after shipping -> `document-release`
- Weekly retro -> `retro`
- Design system or brand -> `design-consultation`
- Visual polish or visual audit -> `design-review`
- Architecture review -> `plan-eng-review`
- Save progress, checkpoint, resume -> `checkpoint`
- Code quality dashboard -> `health`
- Agent guidance refresh -> `$intuitive-init`
- Human doc surface refresh -> `$intuitive-doc`

## Environment And Checks

Do not run unit tests in a fresh environment before dependency preflight.

Use `uv` when available. Check with `uv --version`. This repo intentionally
does not track `uv.lock`; use the editable dev install instead of `uv sync`:

```bash
uv pip install -e ".[dev]"
```

Fallback when `uv` is unavailable:

```bash
python -m pip install -e ".[dev]"
```

Before full tests, verify coverage support:

```bash
python -c "import pytest_cov; print('pytest-cov ok')"
```

Default validation:

```bash
pytest -q
ruff check .
ruff format --check .
mypy src/
```

Run lint/type checks when Python code changes. Do not claim unit test success
unless `pytest -q` ran in the current environment. Docs-only changes do not need
the full Python gate unless they alter runnable examples or workflow commands.

## Project Rules

- Python target: 3.10+.
- Public APIs should be typed.
- New simulator backends implement `SimulatorBackend` structurally; no base
  class inheritance is required.
- Keep versions in `pyproject.toml` and `src/roboharness/__init__.py` in sync.
- Optional extras: `dev` for tests/lint/type, `demo` for MuJoCo/Meshcat/Rerun,
  `wbc` for Pinocchio/Pink controller work, `lerobot` for LeRobot integration.
- MuJoCo headless validation needs `MUJOCO_GL=osmesa` or `MUJOCO_GL=egl`.
- Isaac Lab end-to-end validation requires NVIDIA RTX hardware. CPU tests use
  mocks for compatibility coverage.
- GPU or visual validation may be impossible in CPU-only environments; report
  that limitation explicitly.

## Agent Runbooks

Long agent-only procedures live under `docs/agents/`: release, CI debugging,
testing, environments, GitHub workflow, and design/planning persistence.

## Commit Hygiene

Keep commits scoped and descriptive (`docs: ...`, `fix: ...`, `ci: ...`).
When changing workflow/docs, ensure instructions match actual repo config.

Codex commits must include:

```text
Co-authored-by: Codex <codex@users.noreply.github.com>
```
