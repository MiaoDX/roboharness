# Roboharness: Context Document

> **Document Version**: v0.1-draft | **Date**: 2026-04-02
> **Purpose**: This is the complete context document for the roboharness project, intended as a reference for Claude Code, Codex, and other AI Agents during code review, architecture design, and feature development.

## Part 1: Project Overview and Motivation

### 1.1 What is Roboharness

Roboharness is a **visual testing framework for AI Coding Agents in robot simulation**. Its core goal is to enable Claude Code, OpenAI Codex, and similar coding agents to:

1. **Control simulation stepping** (step-by-step execution) — pause simulation at critical moments
2. **Capture multi-view screenshots** — acquire RGB/depth images from different camera positions at the same simulation moment
3. **Autonomously judge task results** — agent directly observes screenshots to determine whether motion is reasonable, grasps are successful, etc.
4. **Iteratively optimize algorithms** — based on visual judgment results, the agent autonomously modifies control code and reruns

**Fundamental difference from traditional approaches**: We don't need a separate VLM model for visual evaluation. Claude Code and Codex themselves are multimodal agents — they can write code, see images, and make decisions. Roboharness's responsibility is to **efficiently present simulation visual information in a format that agents can directly consume**.

### 1.2 Core Use Case

Taking a grasping task as an example, the complete Agent-in-the-loop workflow is:

1. Agent writes/modifies grasp control code
2. Roboharness runs the simulation, automatically pausing at predefined checkpoints (plan start, plan end, contact point, lift complete)
3. At each checkpoint, the Harness captures screenshots from multiple viewpoints and saves them as files
4. Agent examines screenshots + structured state data, judging whether the current phase is normal
5. If problems are found, the agent modifies code and reruns from the appropriate checkpoint
6. Iterates until the task succeeds

## Part 2: Current Practice (Verified Workflows)

### 2.1 Three-Tool Collaborative Architecture

The current practice is based on the division of labor among three visualization tools:

#### MuJoCo — Pure Physics Simulation Engine

- Handles only physics simulation (collisions, contact forces, gravity)
- The hand (end effector) does not perform kinematics computation in MuJoCo, keeping simulation simple
- Simulation stepping controlled via `mj_step()`
- State snapshots and restoration via `mj_getState`/`mj_setState`

#### Meshcat — Trajectory Visualization and Interactive Debugging

- Browser-based 3D visualization using three.js/WebGL
- Responsible for displaying hand motion trajectories and complete grasping animations
- Agent (Codex) has implemented autonomous Meshcat stepping control:
  - Auto-pause at various grasping stages (plan start, plan end, fixed positions during grasping)
  - Switch viewpoints at pause points for multi-angle screenshots
  - Autonomously judge grasping effectiveness based on screenshots and decide next steps
- Adding objects and trajectories is more convenient (compared to native MuJoCo rendering)

#### Rerun — Multi-modal Data Validation

- Supports depth data visualization
- Good compatibility with robot URDF, suitable for hand-eye calibration verification
- Timeline playback supports data inspection at any moment
- Synchronized display of multi-modal data (RGB, depth, point cloud, joint states)

### 2.2 Verified Agent Closed-Loop Workflow

Codex has already achieved the following closed loop in the current project:

```
Codex writes control code
    ↓
Run simulation → Meshcat renders
    ↓
Codex controls Meshcat step → pauses at key positions
    ↓
Codex switches viewpoints → screenshots (multi-view)
    ↓
Codex analyzes screenshots → judges grasping effectiveness
    ↓
If poor results → Codex modifies code → rerun
If good results → complete
```

This workflow is running and producing real results, but is currently a custom implementation for one specific project.

### 2.3 Limitations of Current Practice

- **Project coupling**: Visual testing logic is embedded in project-specific code, not reusable
- **Fixed backend**: Hard-coded MuJoCo + Meshcat + Rerun combination, cannot extend to Isaac Lab and other simulators
- **Non-standard interface**: No unified API contract between Agent and Harness
- **Ad-hoc data format**: No standardized format for screenshot and state data storage

## Part 3: Community Research — Existing Tools and Practices

### 3.1 Native Recording and Multi-View Rendering Capabilities of Simulators

#### MuJoCo

- `mujoco.Renderer`: Renders RGB/depth/segmentation to numpy arrays from any named camera via `update_scene(data, camera="cam_name")`
- `mujoco.viewer.launch_passive`: Supports pause (SPACE) and single-step (RIGHT), but no built-in video recording
- MuJoCo USD Exporter: Exports complete trajectories to USD format for multi-camera ray-traced rendering in Blender/Omniverse
- MJX-Warp (v3.3.5+): GPU-accelerated batch rendering
- `mujoco-python-viewer`: Third-party viewer supporting `read_pixels(camid=N)` off-screen rendering
- `mujoco_tools`: Supports multi-camera views, 4K video recording, trajectory rendering, CLI-driven capture — the closest existing tool to a single MuJoCo Visual Harness
- MuJoCo Playground: See arXiv:2502.08844

#### Robosuite

- Most complete multi-view infrastructure: `render_camera` parameter accepts multiple camera name lists, simultaneously rendering RGB/depth/segmentation
- Built-in `demo_video_recording.py`, `CameraMover` class, `DemoPlaybackCameraMover`
- MuJoCo states stored as `.npz` files, supporting deterministic trajectory playback — a natural foundation for "stop the world"
- Pluggable rendering backends: native MuJoCo (~60fps), NVISII ray tracing (~0.5fps), iGibson PBR (~1500fps)

#### ManiSkill v3

- `RecordEpisode` wrapper: Simultaneously records video and HDF5 trajectory data, achieving 30,000+ FPS with GPU-parallel environments (RTX 4090)
- `CameraConfig` objects define independent camera poses and resolutions

#### NVIDIA Isaac Lab

- `TiledCamera`: Batch-tiles thousands of cameras into a single GPU render
- Synthetic Data Recorder GUI + Camera Placement optimization tools
- Renderer ≠ Visualizer separation (3.0): Sensor data generation decoupled from interactive debugging
- ~57,000 FPS off-screen rendering (RTX 3090)

#### PyBullet

- `getCameraImage()` supports programmatic screenshots from arbitrary viewpoints
- `startStateLogging(STATE_LOGGING_VIDEO_MP4)` records GUI viewport
- `pybullet-blender-recorder`: Records link poses as pickle, imports to Blender for high-quality offline rendering

#### Gymnasium

- `RecordVideo` wrapper: Records MP4 from any `render_mode="rgb_array"` environment
- Single camera, no multi-view or step-level capture capability

### 3.2 Visualization Backends: Meshcat and Rerun

#### Meshcat

- Lightweight browser 3D viewer (three.js + WebGL), no GPU required, supports Jupyter
- Animation module: Keyframe recording
- Best integration with Drake: `StartRecording()`/`StopRecording()`/`PublishRecording()` captures all transforms
- `StaticHtml()` exports standalone HTML files
- `RoboMeshCat`: Context-manager style video recording
- Pinocchio `MeshcatVisualizer`: Provides `captureImage(w, h)` programmatic screenshots
- **Limitations**: No native multi-view capture, no depth/segmentation rendering, no time-synchronized multi-modal logging

#### Rerun.io

- GitHub 6,200+ stars, Rust implementation, Python/C++/Rust SDKs
- Time-aware Entity Component System: images, point clouds, 3D transforms, meshes, time series, depth maps, segmentation masks, video frames — all synchronized across multiple timelines
- Full time playback: drag, step, multiple timelines
- Programmable Blueprints (`rerun.blueprint`): Define standardized diagnostic layouts
- Simultaneous recording + visualization: Multi-sink support (live viewer + `.rrd` files)
- Dataframe API: Programmatically extract scalar metrics from recordings — can be used for automated pass/fail assertions
- URDF support (v0.24+): Drive joint animations dynamically via logged transforms
- Isaac Lab 3.0 official integration as Visualizer backend
- MCAP/ROS 2 support
- LeRobot integration: HuggingFace LeRobot uses Rerun as primary visualization backend
- `.rrd` files can be archived, shared, and reopened in the web viewer

### 3.3 Research on VLM/LLM-Based Robot Behavior Evaluation

> **Note**: Our approach does not require a separate VLM for judgment — Claude Code/Codex itself is the evaluator. However, the evaluation methodology from these studies (how to ask questions, how to structure visual input, evaluation dimensions) still provides valuable reference.

- **SuccessVQA** (Du et al., 2023): Models robot success detection as a visual question-answering task
- **AHA** (2024-2025): Fine-tunes LLaVA to detect manipulation failures and explain causes, surpassing GPT-4 by 10.3%
- **GVL** (Generative Value Learning): Shuffles video frames for VLM reordering, producing per-frame progress scores — zero-shot, no fine-tuning required
- **VLM-RMs** (ICLR 2024): CLIP cosine similarity as zero-shot reward signal
- **RL-VLM-F** (ICML 2024): VLM compares image pairs to learn reward functions
- **RoboCLIP** (NeurIPS 2023): Video-language model computes trajectory similarity
- **StepEval** (2025): Sub-goal decomposition + VLM stage-by-stage evaluation — closest to the evaluation granularity needed by roboharness
- **Robo2VLM** (2025): Generates 684,710 VQA questions from 176K real trajectories

### 3.4 AI Agent-Driven Simulation Iteration Frameworks

#### Eureka (NVIDIA, ICLR 2024)

- GPT-4 generates reward function code → GPU RL training → text statistics feedback → iterative improvement
- 83% outperforming human expert rewards across 29 tasks, with 52% average improvement
- Does not use visual feedback, purely text statistics-driven
- DrEureka extends to sim-to-real

#### AOR (Act-Observe-Rewrite, March 2025)

- The closest academic work to our practice
- Multimodal LLM (Claude) receives keyframe RGB images + structured diagnostic signals
- Each iteration outputs complete Python controller class code, dynamically compiled and loaded
- Achieves 100% success rate on robosuite Lift/PickPlaceCan — no gradient updates, no demonstrations, no reward engineering
- See arXiv:2603.04466
