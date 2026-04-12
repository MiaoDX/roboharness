# AGENTS.md

This file defines the default operating playbook for coding agents working in this repository.
Its scope is the entire repo tree rooted at this directory.

## 0) First-read policy (mandatory)

Before running any command, read in this order:

1. `AGENTS.md` (this file)
2. `CLAUDE.md`
3. `pyproject.toml` sections:
   - `[project.optional-dependencies]`
   - `[tool.pytest.ini_options]`
   - `[tool.coverage.run]`

If instructions conflict, priority is:
**system/developer/user prompt > AGENTS.md > CLAUDE.md > inferred defaults**.

---

## 1) Environment preflight (mandatory before tests)

Do not run UT immediately on a fresh environment.
Always complete dependency preflight first.

### 1.1 Preferred package manager: `uv`

Use `uv` first when available:

```bash
uv --version
```

If available, install project + dev dependencies with one of:

```bash
uv sync --dev
```

If no lockfile/workspace setup is present, use:

```bash
uv pip install -e ".[dev]"
```

### 1.2 Fallback when `uv` is unavailable

Use pip:

```bash
python -m pip install -e ".[dev]"
```

### 1.3 Fast sanity check before UT

Because pytest coverage flags are configured in `pyproject.toml`, verify `pytest-cov` is usable before running full tests:

```bash
python -c "import pytest_cov; print('pytest-cov ok')"
```

If this fails, install missing dev deps first (via `uv` preferred, pip fallback), then proceed.

---

## 2) Standard test workflow

### 2.1 Full unit tests (default)

```bash
pytest -q
```

### 2.2 Focused debugging loop

```bash
pytest tests/<target_file>.py -k <pattern> -q
```

### 2.3 Optional no-cov local loop (only for faster debugging)

```bash
pytest --no-cov -q
```

Before finishing work, run full default UT (`pytest -q`) at least once.

---

## 3) Lint/type checks for code changes

Run when Python code changes are made:

```bash
ruff check .
ruff format --check .
mypy src/
```

If a check cannot run because of environment limits, report the exact blocker.

---

## 4) Operational best practices

1. **Fail fast with clear diagnostics**: prefer explicit errors over silent fallbacks.
2. **No hidden dependency assumptions**: always derive test requirements from `pyproject.toml`.
3. **Reproducible command logs**: report exact commands and outcomes.
4. **Small, verifiable steps**: install deps -> sanity check -> run UT.
5. **No dependency drift in fixes**: avoid ad-hoc single-package installs unless doing triage; converge back to `.[dev]`.

---

## 5) Quick command checklist (copy/paste)

```bash
# 1) Read project testing/dependency config
sed -n '1,220p' pyproject.toml

# 2) Install deps (uv preferred)
uv --version && uv sync --dev || python -m pip install -e ".[dev]"

# 3) Verify pytest-cov is available
python -c "import pytest_cov; print('pytest-cov ok')"

# 4) Run all unit tests
pytest -q
```

---

## 6) Commit hygiene

- Keep commits scoped and descriptive (`docs: ...`, `fix: ...`, etc.).
- When changing workflow/docs, ensure instructions match actual repo configuration.
- Do not claim UT success unless `pytest -q` has been run in the current environment.
- If a commit is created by Codex, include the Git trailer
  `Co-authored-by: Codex <codex@users.noreply.github.com>` in the commit message.
- If a commit is created by another AI coding agent, include a corresponding
  co-author trailer so agent usage can be tracked later.
- If you maintain a dedicated bot/user account, prefer that account's verified
  noreply email for the relevant agent trailer.
