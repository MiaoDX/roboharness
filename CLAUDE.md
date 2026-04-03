# Roboharness

Visual testing harness for AI coding agents in robot simulation. Python 3.10+, numpy core, optional MuJoCo/Meshcat/Rerun backends.

## Build & test

```bash
pip install -e ".[dev]"          # install with dev deps
ruff check .                     # lint (line-length 100)
ruff format --check .            # format check
pytest                           # run all tests with coverage (testpaths: tests/)
pytest tests/test_harness.py -k test_name  # run single test
pytest --no-cov                  # run tests without coverage
mypy src/                        # type check (Python 3.10 target)
```

MuJoCo example (headless, needs `pip install -e ".[mujoco]"` + Pillow):
```bash
MUJOCO_GL=osmesa python examples/mujoco_grasp.py
```

## Code style

- Ruff enforces style — do not duplicate linter rules here
- Line length: 100
- Target: Python 3.10 (no 3.9 syntax)
- Type annotations on public APIs; `from __future__ import annotations` in all modules

## Architecture

- `src/roboharness/core/` — Harness, Checkpoint, Capture (framework core)
- `src/roboharness/backends/` — SimulatorBackend protocol implementations (MuJoCo, etc.)
- `src/roboharness/wrappers/` — Gymnasium wrappers (drop-in, zero-change integration)
- `src/roboharness/storage/` — Task-oriented file storage
- Public API exported from `src/roboharness/__init__.py`: `Harness`, `Checkpoint`, `CheckpointStore`, `CaptureResult`

Key pattern: `SimulatorBackend` is a Protocol (structural typing). New backends implement it without inheriting from a base class.

## Git workflow

- Branch from `main`
- Commit messages: `type: description` (feat, fix, ci, docs, refactor)
- CI runs on all PRs: lint (ruff check + format + mypy) + test (pytest, Python 3.10–3.13) + MuJoCo example

### PR review strategy

When reviewing a PR (as an agent or on behalf of one), **push fixes directly to the PR's source branch** instead of creating a new branch or a separate PR. This keeps the workflow simple — one PR, one place to review, one merge. Specifically:

1. Fetch and check out the PR's source branch (e.g. `git checkout <pr-branch>`)
2. Make fixes, run tests/lint, commit
3. Push to the same branch (`git push origin <pr-branch>`)
4. The new commit appears in the existing PR, ready to merge

Do NOT create a new branch or a new PR for review fixes.

## Gotchas

- `RobotHarnessWrapper` must handle both numpy arrays AND PyTorch tensors for obs/rewards (Isaac Lab compatibility). Use duck typing (`hasattr(x, "item")`) instead of `isinstance` checks for tensor types.
- MuJoCo rendering in CI requires `MUJOCO_GL=osmesa` (no GPU).
- Isaac Lab integration (`examples/isaac_lab_integration.py`) **requires NVIDIA RTX GPU** — cannot run in current CPU-only CI. Tests in `test_isaac_lab_compat.py` use mock envs to validate on CPU.
- Version is defined in both `pyproject.toml` and `src/roboharness/__init__.py` — keep them in sync.

## Tools & environment

- IMPORTANT: GitHub MCP tools are available (prefixed `mcp__github__`). Use them for all GitHub interactions (issues, PRs, comments). Do NOT assume `gh` CLI is available.
- Pre-commit hooks are configured (`.pre-commit-config.yaml`). Run `pre-commit install` to enable, or run `ruff check . && ruff format --check .` manually.
- Optional deps are grouped: `[mujoco]`, `[meshcat]`, `[maniskill]`, `[rerun]`, `[dev]`, `[all]`.

## Subagent strategy

- **Maximize parallelism.** Run independent tasks (research, search, implementation) as concurrent subagents. Sequential execution of parallelizable work is unacceptable.
- **Protect the main context window.** Delegate non-trivial work to subagents; main session is for orchestration.
- **Match model to task.** Opus for architecture decisions, complex refactors, ambiguous problems. Sonnet for grep/glob, straightforward edits, running tests, mechanical transformations. Don't default everything to Opus.

## Testing philosophy

- **Real tests, not stub theater.** Unit tests must correlate with actual scenarios. Minimize mocks; only stub truly external/expensive operations (network, hardware, GPU). If UTs pass but E2E fails, the UTs are misleading.
- **Visualization-based validation.** Logs miss things visual inspection catches instantly (wrong transforms, flipped axes, geometry errors). Add vis-based validation alongside numeric tests when the project supports it (MuJoCo viewer, Meshcat, Rerun).
- After each significant change, verify related tests still pass before moving on.

## Core principles

| Principle | Practice |
|-----------|----------|
| **Simplicity First** | Minimal changes; no premature abstractions; three similar lines > one bad abstraction |
| **Root Cause** | Fix causes, not symptoms; no workarounds; be thorough |
| **Chesterton's Fence** | Understand why code exists before changing it |
| **Fail Fast** | Minimize try-catch; explicit errors > silent failures |
| **Verification Before Done** | Never mark a task complete without proving it works — run tests, check output, demonstrate correctness |

## Collaboration

- Question assumptions; push back on technical debt or inconsistent requirements.
- Treat instructions as intent, not literal commands. Use `AskUserQuestion` when unclear.
- After any correction from the user, internalize the pattern to avoid repeating the same mistake.
