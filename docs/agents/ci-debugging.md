# CI Debugging Runbook

When a CI check fails, read the actual logs and error messages before
diagnosing. Do not stop at a status summary such as `conclusion: failure`.

## Required Loop

1. Identify the failed check runs.
2. Open the failed check's detail page or log artifact.
3. Read the concrete error output.
4. Diagnose from the log evidence.
5. Fix the smallest relevant cause.
6. Re-run the matching local check when possible.

External services such as Cirun or Codecov often show the key failure detail
only on their detail pages.

## Local Commands

Use the Makefile when it maps to the task:

```bash
make check-all
make test
make test-quick
make lint
make typecheck
```

Equivalent direct gates:

```bash
pytest -q
ruff check .
ruff format --check .
mypy src/
```

MuJoCo or visual checks usually need:

```bash
MUJOCO_GL=osmesa python examples/demos/mujoco/grasp.py --report
```

## CI Shape

`.github/workflows/ci.yml` currently includes:

- lint: Ruff check, Ruff format check, and mypy.
- test: pytest across Python 3.10, 3.11, 3.12, and 3.13.
- test-mujoco: demo + dev install with OSMesa.
- mujoco-example: `examples/demos/mujoco/grasp.py --report --assert-success`.
- LeRobot native example path with direct runtime installs.
- GPU relevance filtering and paused GCP GPU runner notes.

CPU-only environments can validate core logic, wrappers, CLI behavior, report
generation, and mocked Isaac Lab / ManiSkill compatibility. They cannot fully
validate real GPU or visual behavior; report that limitation explicitly.
