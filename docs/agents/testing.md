# Testing Standards

The default test contract is defined in `pyproject.toml`:

- tests live under `tests`
- third-party pytest plugin autoload is disabled
- `pytest_cov` is loaded explicitly
- coverage fails below 90%

Before running full tests in a fresh environment, install dev dependencies and
verify `pytest-cov`:

```bash
uv --version
uv pip install -e ".[dev]"
python -c "import pytest_cov; print('pytest-cov ok')"
pytest -q
```

This repo intentionally does not track `uv.lock`, so do not treat a local
ignored `uv.lock` as shared setup state. If `uv` is unavailable, use:

```bash
python -m pip install -e ".[dev]"
```

## Quality Bar

- Prefer test-driven development for new modules or functions.
- New code should not reduce overall coverage.
- Tests should assert specific expected values, not only truthiness.
- Minimize mocks; use them for network, hardware, GPU, or genuinely expensive
  dependencies.
- Do not add `continue-on-error`, `|| true`, or swallowed exceptions to test
  commands.
- Use `pytest.importorskip("module")` at module scope for optional dependency
  tests; do not wrap test bodies in broad `try/except ImportError`.
- Add visual validation where numeric tests cannot catch the real failure mode.
- Coverage omit entries must be for genuine optional heavy or hardware
  dependencies, not missing test coverage.

## Focused Loops

Use focused pytest while debugging:

```bash
pytest tests/<target_file>.py -k <pattern> -q
pytest --no-cov -q
```

Before finishing Python changes, run the default gate at least once:

```bash
pytest -q
ruff check .
ruff format --check .
mypy src/
```

Docs-only changes do not need this full gate unless they modify runnable
examples, commands, CI behavior, or package metadata.
