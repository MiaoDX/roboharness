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

MuJoCo example (headless, needs `pip install -e ".[demo]"`):
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

### Release process

Releases are cut by pushing to the `release` branch, which triggers `.github/workflows/release.yml` to tag, create a GitHub Release, and publish to PyPI via OIDC trusted publishing.

**Important:** The `release` branch has **unrelated git history** (no common ancestor with `main`). It exists only as a deploy pointer. GitHub UI cannot merge unrelated histories, so **never open a PR targeting `release`**. Use this flow:

1. Bump the version in **both** `pyproject.toml` and `src/roboharness/__init__.py` on a feature branch (e.g. `claude/release-<slug>`). Use semver — minor bump for feature milestones, patch for fix-only rollups.
2. Open a PR targeting **`main`** (never `release`). Write release notes in the PR description summarizing what landed since the last tag.
3. After CI is green and the PR is merged into `main`, force-push `release` to `main`'s tip:
   ```bash
   git fetch origin main
   git push origin +origin/main:release   # or: git push --force origin main:release
   ```
   This is a force-push by design — `release` is a deploy pointer, not a history-preserving branch. Requires explicit user authorization each time.
4. The push triggers the workflow: reads version from `pyproject.toml`, tags `vX.Y.Z`, generates release notes (`gh release create --generate-notes`), builds, publishes to PyPI.
5. Verify the GitHub Release, the `vX.Y.Z` tag, and the PyPI artifact before closing out.

If a duplicate PR targeting `release` was auto-created on push, close it — the `main`-targeted PR is the canonical one.

**Environment protection:** The PyPI publish step uses the `pypi` GitHub environment. Its **Deployment branches and tags** rule must allow the `release` branch (and `v*` tag pattern if we ever switch to tag-triggered deploys). If this rule is missing, the workflow fails with `Branch 'release' is not allowed to deploy to pypi due to environment protection rules` in 1–2 seconds. Fix in Settings → Environments → pypi, then re-run the failed job.

**No git-based direct references in `pyproject.toml`:** PyPI rejects uploads that declare dependencies like `pkg @ git+https://...`, even inside optional extras, with `400 Can't have direct dependency`. The `allow-direct-references = true` hatch setting only silences the *build-time* check — PyPI still refuses on upload. Any fork/git dep must be installed via a separate step (Dockerfile, CI workflow, docs) rather than through `[project.optional-dependencies]`. See the `unitree-sdk2py` handling in `Dockerfile` and `.github/workflows/pages.yml` for the pattern.

**Validating a release end-to-end:** A successful GitHub Release + tag does **not** imply PyPI publication. The workflow creates the tag and release *before* building and uploading the wheel, so a PyPI failure still leaves a tag behind. Always confirm all three:
1. `vX.Y.Z` tag at https://github.com/MiaoDX/roboharness/tags
2. GitHub Release at https://github.com/MiaoDX/roboharness/releases/tag/vX.Y.Z
3. PyPI artifact at https://pypi.org/project/roboharness/X.Y.Z/ (or via `curl -s https://pypi.org/pypi/roboharness/json | jq -r .info.version`)

If PyPI publish fails mid-flow, do **not** reuse the broken version number — bump to the next patch (e.g. `0.2.0` broken → release as `0.2.1`). The broken version's tag and GitHub Release can stay as historical markers.

## CI failure investigation

When a CI check fails, **always read the actual logs/error messages** before diagnosing. Do not stop at the status summary (`conclusion: failure`). Specifically:

1. Use `get_check_runs` to identify which checks failed.
2. For each failed check, visit its `html_url` (via WebFetch or browser) to read the full error output. External checks (Cirun, Codecov, etc.) often have critical details only visible on their detail pages.
3. Only after reading the actual error messages, diagnose and fix.

Skipping step 2 leads to guessing — which wastes time and misses the real issue.

## Gotchas

- `RobotHarnessWrapper` must handle both numpy arrays AND PyTorch tensors for obs/rewards (Isaac Lab compatibility). Use duck typing (`hasattr(x, "item")`) instead of `isinstance` checks for tensor types.
- MuJoCo rendering in CI requires `MUJOCO_GL=osmesa` (no GPU).
- Isaac Lab integration (`examples/isaac_lab_integration.py`) **requires NVIDIA RTX GPU** — cannot run in current CPU-only CI. Tests in `test_isaac_lab_compat.py` use mock envs to validate on CPU.
- Version is defined in both `pyproject.toml` and `src/roboharness/__init__.py` — keep them in sync.

## Development environments

The project uses a tiered development workflow. Choose the right environment for the task:

| Environment | GPU | Best for |
|-------------|-----|----------|
| **claude.ai/code (web)** | No | Core logic, tests, refactors, CI config, docs |
| **Claude Code CLI + local GPU** | Yes | Demo debugging, visual QA, ONNX/rendering, locomotion controllers |
| **CI (GitHub Actions + Cirun)** | CPU + T4 | Automated gating, regression detection |

**Rule of thumb:** if you need to _see_ what the robot is doing, use local CLI + GPU. If you're writing logic and tests, use web.

Local GPU setup: run `scripts/gpu-dev-setup.sh` or see `docs/development-workflow.md` for manual steps. Use `make check-gpu` to verify the setup, `make demos` to run all demos.

## Tools & environment

- IMPORTANT: GitHub MCP tools are available (prefixed `mcp__github__`). Use them for all GitHub interactions (issues, PRs, comments). In **cloud/web environments** (Claude Code on the web), do NOT use `gh` CLI — it cannot be authorized; always use GitHub MCP tools. If an MCP tool call fails due to temporary unavailability, wait ~2 minutes and retry. In **local environments** (CLI/IDE), `gh` CLI is available and can be used normally.
- Pre-commit hooks are configured (`.pre-commit-config.yaml`). Run `pre-commit install` to enable, or run `ruff check . && ruff format --check .` manually.
- Optional deps: `[demo]` (all example dependencies), `[dev]` (testing/linting tools).

## Subagent strategy

- **Maximize parallelism.** Run independent tasks (research, search, implementation) as concurrent subagents. Sequential execution of parallelizable work is unacceptable.
- **Protect the main context window.** Delegate non-trivial work to subagents; main session is for orchestration.
- **Match model to task.** Opus for architecture decisions, complex refactors, ambiguous problems. Sonnet for grep/glob, straightforward edits, running tests, mechanical transformations. Don't default everything to Opus.

## Testing philosophy

- **Test-driven development.** Write tests first, then implement. This ensures every feature has coverage from the start and avoids the "write tests later" debt that never gets paid. When adding a new module or function, start with the test file.
- **Coverage threshold: 90%+.** CI enforces a minimum coverage threshold (see `pyproject.toml`). New code must not lower overall coverage. If you add code, add tests for it in the same PR.
- **Real tests, not stub theater.** Unit tests must correlate with actual scenarios. Minimize mocks; only stub truly external/expensive operations (network, hardware, GPU). If UTs pass but E2E fails, the UTs are misleading.
- **Zero false positives.** Every assertion must verify a specific expected value, not just "something truthy". No `assert result is not None` when you can check the actual value. No range checks (`0.0 <= x <= 1.0`) when the expected value is known. Pin expected results from known fixture data. A test that can never fail is worse than no test — it provides false confidence.
- **No `continue-on-error` in CI.** Every CI job must pass. If a job is flaky, fix the root cause — don't mask it. No `continue-on-error: true`, no `|| true` on test commands, no `try/except` that swallows test failures.
- **Guard optional deps correctly.** Use `pytest.importorskip("module")` at the top of the test, not `try/except ImportError: pytest.skip()` wrapping the test body. The latter swallows unrelated `ImportError`s from actual bugs.
- **Visualization-based validation.** Logs miss things visual inspection catches instantly (wrong transforms, flipped axes, geometry errors). Add vis-based validation alongside numeric tests when the project supports it (MuJoCo viewer, Meshcat, Rerun).
- **Coverage omit is for genuine hardware deps only.** Files in `[tool.coverage.run] omit` must require hardware or optional heavy dependencies (MuJoCo, GPU, Pinocchio) that aren't in `[dev]`. Don't omit files just because tests haven't been written — that hides debt.
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
