# Roboharness Architecture

Roboharness is a visual testing harness for **AI Coding Agents** in robot simulation. Its core premise: when an AI agent writes robot control code, it needs to **pause, observe, judge, and iterate** at critical moments — just like a human engineer watching a simulation replay.

## Design Philosophy

### Why Roboharness?

Traditional robot simulation testing relies on numerical assertions (joint angles within range, end-effector error below threshold). But many problems — wrong coordinate transforms, flipped axes, unnatural motion trajectories — can pass numerically while being immediately obvious visually.

Roboharness enables AI agents to automatically capture multi-view screenshots at semantically meaningful moments in simulation, combined with numerical state, forming a **"visual + numerical" dual-channel verification** loop.

### Three Core Principles

1. **Protocol-Driven**: All external dependencies (simulators, controllers, visualizers) integrate via structural typing Protocols — no base class inheritance required.
2. **Checkpoint-Oriented**: Simulation doesn't run straight through. It pauses at semantically meaningful points for capture and inspection.
3. **Agent-First**: The API is designed around the AI agent's workflow — load a task protocol, execute action sequences, receive visual feedback, decide what to do next.

## High-Level Architecture

```
                    ┌──────────────────────────────────┐
                    │         AI Coding Agent           │
                    │  (LLM writing/modifying control)  │
                    └──────────┬───────────────────────┘
                               │ command
                               ▼
                    ┌──────────────────────┐
                    │     Controller       │  high-level cmd → low-level action
                    │     (Protocol)       │  e.g. target pose → joint angles
                    └──────────┬───────────┘
                               │ action
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                          Harness                                 │
│                                                                  │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────────┐  │
│  │ TaskProtocol │  │ Checkpoint[] │  │ CheckpointStore        │  │
│  │ (semantic    │→ │ (capture     │  │ (state snapshot        │  │
│  │  phases)     │  │  points)     │  │  save/restore)         │  │
│  └─────────────┘  └──────────────┘  └────────────────────────┘  │
│                                                                  │
│  step() ──→ run_to_next_checkpoint() ──→ capture() ──→ save()   │
│                                                                  │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │  SimulatorBackend   │  simulator adapter layer
              │     (Protocol)      │  step / get_state / capture_camera / ...
              └─────────────────────┘
              Implementations:
                • MuJoCoMeshcatBackend
                • (Isaac Lab, ManiSkill, ...)
```

## Module Breakdown

### `core/` — Framework Core

| File | Responsibility |
|------|---------------|
| `harness.py` | **Harness** class + **SimulatorBackend** Protocol. Manages the simulation loop, checkpoint scheduling, and multi-view capture |
| `protocol.py` | **TaskProtocol** + **TaskPhase**. Semantic task protocols defining natural phases of a task (see below) |
| `checkpoint.py` | **Checkpoint** dataclass + **CheckpointStore** for state snapshot management |
| `capture.py` | **CameraView** (single camera frame) + **CaptureResult** (full capture at a checkpoint) |
| `controller.py` | **Controller** Protocol. Interface for high-level command → low-level action conversion |
| `lifecycle.py` | **ComponentLifecycle** metadata system. Tags components with existence assumptions and expiration horizons for periodic "harness diet" reviews |
| `rerun_logger.py` | Optional Rerun visualization logging integration |

### `core/protocol.py` — Semantic Task Protocols

This is a recently introduced core concept. The traditional approach captures at fixed simulation step counts ("screenshot every 100 steps"), which is simple but loses semantic meaning. TaskProtocol maps capture points to the natural phases of a task:

```python
# A grasping task's semantic protocol
GRASP_PROTOCOL = TaskProtocol(
    name="grasp",
    phases=[
        TaskPhase("plan",      "Plan grasp trajectory and visualize target path"),
        TaskPhase("pre_grasp", "Move to pre-grasp pose above the object"),
        TaskPhase("approach",  "Approach the object along the planned path"),
        TaskPhase("grasp",     "Close gripper on the object"),
        TaskPhase("lift",      "Lift the grasped object"),
        TaskPhase("place",     "Place the object at the target location"),
        TaskPhase("home",      "Return to home position"),
    ],
)

# Load with a single call
harness.load_protocol(GRASP_PROTOCOL)
# Or select only the phases you need
harness.load_protocol(GRASP_PROTOCOL, phases=["pre_grasp", "grasp", "lift"])
```

**Four built-in protocols:**

| Protocol | Use Case | Phases |
|----------|----------|--------|
| `GRASP_PROTOCOL` | Pick-and-place | plan → pre_grasp → approach → grasp → lift → place → home |
| `LOCOMOTION_PROTOCOL` | Legged walking | initial → accelerate → steady → decelerate → terminal |
| `LOCO_MANIPULATION_PROTOCOL` | Mobile manipulation | navigate → pre_grasp → grasp → transport → place → retreat |
| `DANCE_PROTOCOL` | Rhythmic motion | ready → sequence → finale |

**Custom protocols are straightforward:**

```python
my_protocol = TaskProtocol(
    name="assembly",
    phases=[
        TaskPhase("pick", "Pick up the part", cameras=["front", "wrist"]),
        TaskPhase("align", "Align part with target", cameras=["top", "wrist"]),
        TaskPhase("insert", "Insert part into slot", cameras=["front", "side"]),
    ],
)
```

`BUILTIN_PROTOCOLS` dictionary provides a registry of all built-in protocols for discovery and iteration.

### `backends/` — Simulator Adapter Layer

**SimulatorBackend** is a `@runtime_checkable` Protocol with 7 methods:

```python
class SimulatorBackend(Protocol):
    def step(self, action) -> dict[str, Any]: ...       # advance one step
    def get_state(self) -> dict[str, Any]: ...           # read current state
    def save_state(self) -> dict[str, Any]: ...          # save full state (for rollback)
    def restore_state(self, state) -> None: ...          # restore to a saved state
    def capture_camera(self, camera_name) -> CameraView: # capture a camera frame
    def get_sim_time(self) -> float: ...                 # simulation time
    def reset(self) -> dict[str, Any]: ...               # reset to initial state
```

New simulators only need to implement these 7 methods — **no base class inheritance required**. Current implementations:

- **MuJoCoMeshcatBackend** — MuJoCo physics + Meshcat 3D visualization export
- **MeshcatVisualizer** — Standalone Meshcat interactive scene exporter

Planned future backends (see `docs/spike-newton-backend.md` and `docs/spike-roboverse-metasim.md`):

- **NewtonBackend** — NVIDIA Newton 1.0 (Warp-based GPU physics, 475× faster than MJX for manipulation). Awaiting API stabilisation and community adoption. Fastest path to Newton coverage today: use `RobotHarnessWrapper` with Isaac Lab's Newton-backed environments.
- **RoboVerseBackend** — Single adapter for 8+ simulators via RoboVerse MetaSim (MuJoCo, Isaac Lab, SAPIEN, Genesis, Newton, …).

### `evaluate/` — Evaluation Engine

Automated constraint checking and evaluation:

```
report.json ──→ MetricAssertion[] ──→ AssertionEngine ──→ EvaluationResult
                                                              │
                                                              ├── Verdict: PASS / DEGRADED / FAIL
                                                              └── AssertionResult[]
```

- **MetricAssertion** — A single constraint (`grip_error < 5.0mm`, `lift_height > 0.02m`)
- **AssertionEngine** — Runs all constraints against a report, produces a Verdict
- **Severity** — CRITICAL (any failure → FAIL), MAJOR (failure → DEGRADED), MINOR, INFO
- **Operator** — lt, le, eq, gt, ge, in_range
- **Constraints** — Loadable from JSON/YAML files, separating configuration from code

**Batch evaluation** (`evaluate/batch.py`): Cross-trial aggregated analysis — success rates, failure phase distribution, variant comparison.

### `runner.py` — Parallel Trial Execution

```python
runner = ParallelTrialRunner(
    backend_factory=lambda: MyBackend(),   # isolated simulator per trial
    store=my_store,                         # output storage
    max_workers=4,                          # concurrency
)
batch = runner.run(specs, trial_fn=my_trial)
print(batch.success_rate)
```

- **TrialSpec** — Specification for a single trial (variant_name, trial_id, metadata)
- **ParallelTrialRunner** — ThreadPoolExecutor-based concurrent runner; each trial gets its own backend and output directory
- **BatchResult** — Aggregated results: success rate, per-variant statistics, failure_phase_distribution

### `storage/` — Storage System

Hierarchical file organization:

```
harness_output/
└── pick_and_place/                    # task name
    ├── task_config.json
    ├── grasp_position_001/            # variant (e.g. different grasp positions)
    │   ├── position.json
    │   ├── trial_001/                 # first attempt
    │   │   ├── pre_grasp/             # checkpoint
    │   │   │   ├── front_rgb.png      # multi-view screenshots
    │   │   │   ├── side_rgb.png
    │   │   │   ├── state.json         # simulation state
    │   │   │   └── metadata.json
    │   │   ├── grasp/
    │   │   ├── lift/
    │   │   └── result.json            # trial outcome
    │   └── summary.json               # variant summary
    └── report.json                    # overall report
```

- **TaskStore** — Generic task → variant → trial → checkpoint storage
- **GraspTaskStore** — Grasp-task-specific storage with predefined checkpoints `["plan_start", "pre_grasp", "contact", "lift"]`
- **EvaluationHistory** — Append-only JSONL log recording success rates and metrics per evaluation run. Supports trend detection (regression/improvement/stable)

### `wrappers/` — Gymnasium Integration

**RobotHarnessWrapper** provides zero-change integration with any Gymnasium environment:

```python
env = gym.make("CartPole-v1", render_mode="rgb_array")
env = RobotHarnessWrapper(env,
    checkpoints=[{"name": "early", "step": 10}, {"name": "late", "step": 100}],
    output_dir="./output",
)

obs, info = env.reset()
for _ in range(200):
    obs, reward, terminated, truncated, info = env.step(action)
    if "checkpoint" in info:
        print(f"Checkpoint: {info['checkpoint']['name']}")
```

Automatically detects multi-camera capabilities (`render_camera()` method, Isaac Lab TiledCamera, or fallback to `env.render()`).

### `robots/` — Robot-Specific Code

Currently supports **Unitree G1** humanoid robot:

- **GrootLocomotionController** — ONNX-based walking controller (15-DOF lower body)
- **HolosomaLocomotionController** — 29-DOF whole-body controller
- **SonicLocomotionController** — Multi-mode controller (walk/dance/track), 10Hz re-planning + interpolation

### `reporting.py` — HTML Report Generation

Generates self-contained HTML reports with multi-view screenshots at each checkpoint, state metadata, and optional embedded Meshcat 3D interactive scenes.

### `cli.py` — Command-Line Interface

```bash
roboharness inspect ./harness_output    # inspect output directory contents
roboharness report ./harness_output     # generate HTML report + JSON summary
roboharness evaluate report.json        # run constraint evaluation
roboharness evaluate-batch ./output/    # batch evaluation
roboharness trend ./output/             # trend detection (regression/improvement)
```

### `core/lifecycle.py` — Component Lifecycle

A unique metadata system: each framework component can be tagged with the assumptions justifying its existence and an expected expiration horizon. As AI model capabilities improve, some helper components may no longer be needed.

```python
ComponentLifecycle(
    name="intermediate_checkpoints",
    horizon=ExpirationHorizon.LONG_TERM,
    assumptions=[
        ComponentAssumption(
            description="Models cannot diagnose intermediate failures from final state alone",
            removal_condition="Models can accurately diagnose mid-process errors from a final screenshot",
        ),
    ],
)
```

## Data Flow

A typical agent workflow:

```python
from roboharness import Harness, GRASP_PROTOCOL

# 1. Create backend and harness
backend = MuJoCoMeshcatBackend(xml_string=model_xml, cameras=["front", "side", "top"])
harness = Harness(backend, output_dir="./output", task_name="pick_cube")

# 2. Load semantic protocol (auto-registers checkpoints)
harness.load_protocol(GRASP_PROTOCOL, phases=["pre_grasp", "grasp", "lift"])

# 3. Reset simulation
harness.reset()

# 4. Execute phase by phase
for phase_name, actions in my_action_sequences.items():
    result = harness.run_to_next_checkpoint(actions)
    # result.views — multi-view screenshots
    # result.state — simulation state (joint angles, contact forces, etc.)
    # result.sim_time — simulation time

    # Agent inspects screenshots, decides whether to adjust
    if not looks_good(result):
        harness.restore_checkpoint("pre_grasp")  # rollback to a previous checkpoint
        # retry with different approach...
```

## Dependencies

**Core (zero extra dependencies):** numpy

**Optional dependency groups:**

| Group | Purpose | Key Packages |
|-------|---------|-------------|
| `[mujoco]` | MuJoCo simulation | mujoco >= 3.0 |
| `[meshcat]` | 3D interactive visualization | meshcat >= 0.3 |
| `[rerun]` | Rerun time-series visualization | rerun-sdk >= 0.18 |
| `[wbc]` | Whole-body control (IK) | pinocchio, pink, qpsolvers |
| `[lerobot]` | LeRobot policy inference | onnxruntime, huggingface_hub |
| `[dev]` | Development tools | pytest, ruff, mypy |

## Extension Guide

### Adding a New Simulator Backend

Implement the 7 methods of `SimulatorBackend` — no inheritance needed:

```python
class MySimBackend:
    def step(self, action): ...
    def get_state(self): ...
    def save_state(self): ...
    def restore_state(self, state): ...
    def capture_camera(self, camera_name): ...
    def get_sim_time(self): ...
    def reset(self): ...
```

### Adding a New Task Protocol

```python
MY_PROTOCOL = TaskProtocol(
    name="my_task",
    description="Description of this task type",
    phases=[
        TaskPhase("phase_1", "What happens in phase 1", cameras=["front"]),
        TaskPhase("phase_2", "What happens in phase 2", cameras=["front", "top"]),
    ],
)
```

### Adding a New Controller

Implement the `Controller` Protocol's `compute()` method:

```python
class MyController:
    def compute(self, command: dict, state: dict) -> Any:
        # command → action conversion logic
        return joint_positions
```
