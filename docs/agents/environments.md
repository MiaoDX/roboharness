# Development Environments

Roboharness work spans CPU-only code, simulator rendering, and GPU-dependent
robot behavior. Choose the environment based on what must be proven.

| Environment | GPU | Best for |
| --- | --- | --- |
| Claude/Codex web or cloud | No | Core logic, tests, refactors, CI config, docs |
| Local CLI/IDE with GPU | Yes | Demo debugging, visual QA, ONNX/rendering, locomotion controllers |
| CI | CPU + selected GPU runners | Automated gating and regression detection |

If you need to see what the robot is doing, use a local or CI environment that
can render the relevant simulator output. If you are writing core logic and
tests, CPU-only environments are usually enough.

## Optional Extras

- `.[dev]` - tests, linting, typing, common lightweight integrations.
- `.[demo]` - MuJoCo, Meshcat, Rerun, examples, and visual demos.
- `.[wbc]` - Pinocchio / Pink whole-body-control work.
- `.[lerobot]` - LeRobot integration work.

## Headless MuJoCo

Use OSMesa or EGL for headless rendering:

```bash
MUJOCO_GL=osmesa python examples/mujoco_grasp.py --report
MUJOCO_GL=egl python examples/mujoco_grasp.py --report
```

The Makefile defaults `MUJOCO_GL` to `osmesa` for demo targets:

```bash
make demo-grasp
make demos
```

## GPU Setup

Use the project helper or the detailed human workflow doc:

```bash
scripts/gpu-dev-setup.sh
make check-gpu
```

See `docs/development/development-workflow.md` for manual setup and background.

Isaac Lab end-to-end validation requires NVIDIA RTX hardware. CPU tests validate
the wrapper contract through mock environments, not real Isaac Lab execution.
