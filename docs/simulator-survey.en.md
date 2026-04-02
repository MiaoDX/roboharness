# Technical Survey of Open-Source Robot Control Projects and Roboharness Integration Analysis

> The Gymnasium Wrapper approach can cover 60% of mainstream projects, but two major ecosystems (legged_gym family and JAX/Brax family) require dedicated adapters.

Among the actively community-developed open-source robot control projects from 2024–2026, Isaac Lab, ManiSkill, and LocoMuJoCo natively support the Gymnasium API and can be integrated with minimal intrusion. In contrast, unitree_rl_gym (legged_gym family) and MuJoCo Playground (JAX/Brax family) require custom bridging layers due to fundamental architectural differences.

## Overview of the Five Projects

| Project | Stars | Simulator | Gymnasium Compatible | License | Key Organization |
|---------|-------|-----------|---------------------|---------|-----------------|
| Isaac Lab | ~6,700 | PhysX + Newton/MuJoCo-Warp | Native | BSD-3-Clause | NVIDIA |
| unitree_rl_gym | ~3,100 | Isaac Gym + MuJoCo | Custom API | BSD-3-Clause | Unitree Robotics |
| ManiSkill | ~2,400 | SAPIEN (PhysX) | Via Wrapper | Apache-2.0 | Hillbot/UCSD |
| MuJoCo Playground | ~1,800 | MuJoCo MJX/Warp | Brax Functional | Apache-2.0 | Google DeepMind |
| LocoMuJoCo | ~1,400 | MuJoCo + MJX | Native | MIT | TU Darmstadt |

---

## Isaac Lab: The Standardized Framework in the NVIDIA Ecosystem

Isaac Lab is currently the most mature Gymnasium-native robot learning framework, with 376+ contributors on GitHub and three visualization backends (Kit/Newton/Rerun).

Both its `ManagerBasedRLEnv` and `DirectRLEnv` environment workflows directly inherit from `gymnasium.Env`, allowing standard environments to be created via `gym.make("Isaac-Reach-Franka-v0")`.

### Simulation Architecture

Built on the PhysX GPU-accelerated physics engine, it achieves zero-copy CUDA-to-PyTorch tensor interoperability through the OmniPhysics Tensor API. Version 3.0 introduced the Newton backend (MuJoCo-Warp), supporting a "kit-less" mode that runs without Isaac Sim.

The core of the simulation loop is the **decimation mechanism** — multiple physics substeps are executed within each RL step, with a typical configuration of `dt=1/120, decimation=2`, corresponding to an RL step duration of 1/60 second.

Inside `env.step()`, the execution flow is: preprocess actions → multiple `sim.step()` calls → render (optional) → compute rewards/terminations → reset terminated environments → compute observations.

### Configuration System

Uses a custom `@configclass` decorator (an enhanced Python dataclass) with Hydra-style CLI overrides (`env.a.b.param=value`). Environments are registered via standard `gymnasium.register()`, following the naming convention `Isaac-<Task>-<Robot>-v0`.

### Visualization

Each of the three backends has its own trade-offs:
- **Kit Visualizer**: RTX ray-traced rendering but requires the full Isaac Sim installation
- **Newton Visualizer**: Lightweight, suitable for large-scale environments
- **Rerun Visualizer**: Supports web access and time scrubbing, but suffers from performance degradation in large-scale environments

Video recording is supported via `gymnasium.wrappers.RecordVideo` (requires `--enable_cameras` and ffmpeg) as well as USD animation baking.

Known issues: headless mode rendering hangs (#324), camera-enabled environments not rendering (#3250), recordings missing debug markers (#2233), WebRTC errors in Docker (#3192).

### Roboharness Integration Feasibility: Very High

The standard Gymnasium Wrapper approach is directly applicable:

```python
env = gym.make("Isaac-Cartpole-v0", render_mode="rgb_array")
env = RobotHarnessWrapper(env)  # Single-line integration
env = Sb3VecEnvWrapper(env)     # RL library wrapper must be outermost
```

Key considerations: observations and actions are PyTorch tensors on the GPU (shape `(num_envs, ...)`). The Wrapper must efficiently handle batched GPU data and avoid unnecessary CPU transfers. Internal objects such as scene, sim, and cfg are accessible via `env.unwrapped`.

**Isaac Lab user code changes required: zero lines.**

---

## unitree_rl_gym: The Most Popular Humanoid RL Training Repository (Requires Deep Adaptation)

With 3,100+ stars and 520+ forks, this is the de facto standard for Unitree Go2/H1/G1 robot RL training. However, it inherits from ETH Zurich's legged_gym and uses a completely custom API — fully incompatible with Gymnasium.

### Simulation Architecture

Two distinct paths:
- **Training path**: Uses Isaac Gym's `self.gym.simulate(self.sim)` to drive the PhysX physics engine, combined with the `refresh_*_tensor()` family of functions to update GPU tensor state
- **Validation path** (sim-to-sim): Uses MuJoCo's `mj_step()` for single-instance CPU simulation

`step()` returns a 5-tuple `(obs_buf, privileged_obs_buf, rew_buf, reset_buf, extras)` — entirely different from Gymnasium's `(obs, reward, terminated, truncated, info)`. The environment automatically resets terminated parallel environments internally (`reset_idx()`), with no need for external `reset()` calls.

The configuration system uses nested Python classes (neither dataclasses nor YAML), with inheritance-based overrides: `LeggedRobotCfg → G1Cfg`. Environment registration uses a custom `TaskRegistry` rather than `gymnasium.register()`.

**No automated tests, no CI/CD** — this is a notable weakness of the project.

### Roboharness Integration Feasibility: Moderately Difficult

Five fundamental incompatibilities exist:
1. **Vectorized vs. single-instance**: Isaac Gym runs 4096+ parallel environments simultaneously; Gymnasium expects a single instance
2. **Return signature mismatch**: 5-tuple vs. standard 5-tuple (different field semantics)
3. **No `reset()` method**: Automatic internal reset vs. external call
4. **GPU tensors vs. NumPy arrays**: All data resides on the GPU
5. **No Space definitions**: No `observation_space` / `action_space`

Recommended path: wrap the MuJoCo sim-to-sim validation path (`deploy/deploy_mujoco/deploy_mujoco.py`), which is single-instance, CPU-based, and uses standard MuJoCo APIs. Estimated effort: ~1 week.

Alternatively, suggest that users migrate to the newer `unitree_rl_lab` (based on Isaac Lab, nearly Gymnasium-compatible).

---

## MuJoCo Playground: DeepMind's JAX Functional Architecture

Google DeepMind's GPU-accelerated robot learning framework, covering 50+ environments (DM Control Suite 25+, locomotion 19, manipulation 10), with dual backend support for MJX (JAX) and MuJoCo Warp (NVIDIA). Published at RSS 2025, it has achieved zero-shot sim-to-real transfer on five or more real robot platforms.

### Simulation Architecture

Entirely based on JAX functional programming. Environments are pure functions: `state = env.step(state, action)` rather than Gymnasium's `obs, rew, ... = env.step(action)`. All state is passed through explicit State dataclasses with no hidden internal state — a requirement for JAX JIT compilation.

Batch parallelism is achieved via `jax.vmap()`, and training loops complete full rollouts on-device using `jax.lax.scan`. Single-threaded MJX on GPU is 10x slower than CPU MuJoCo — its advantage comes entirely from massive parallelism (batch sizes of 1024–8192+).

### Roboharness Integration Feasibility: Difficult

The standard Gymnasium Wrapper cannot be used directly because:
1. **Explicit state passing vs. internal state**: The Wrapper must maintain `self._state` internally
2. **JAX arrays vs. NumPy**: Each step requires GPU-to-CPU transfer
3. **JIT incompatibility**: Gymnasium's Python control flow (checking done, calling reset) breaks JIT compilation
4. **Reset semantics conflict**: Brax's AutoResetWrapper handles resets within JIT

The existing bridging pattern `RSLRLBraxWrapper` (which achieves zero-copy JAX-to-PyTorch transfer via DLPack) is the closest reference. The recommended approach is to intercept at the JAX layer (MjxEnv level), inserting before the Brax wrapper chain is assembled, to avoid breaking JIT. However, this means the standard Gymnasium Wrapper cannot be used — a JAX-native adapter must be written instead.

---

## ManiSkill: Exemplary Gymnasium Integration for GPU-Parallel Manipulation Benchmarks

ManiSkill3, built on the SAPIEN engine, achieves GPU-parallel simulation with 200K+ state FPS and 30K+ rendering FPS on an RTX 4090, using 2–3x less GPU memory than Isaac Lab. Published at RSS 2025, it includes a rich set of tasks spanning tabletop manipulation, dexterous hands, painting/cleaning, and more.

### Simulation Architecture

All parallel environments' rigid bodies and articulations are placed in a single PhysX scene, with spatial partitioning to isolate sub-scenes. Supports heterogeneous parallel simulation — different sub-scenes can contain different objects and articulations.

### Gymnasium Compatibility (Three Modes)

```python
# Mode 1: Standard gym.Env (CPU single-instance, NumPy)
env = gym.make("PickCube-v1", num_envs=1)
env = CPUGymWrapper(env)  # Standard Gymnasium interface

# Mode 2: gymnasium.vector.VectorEnv (GPU parallel, PyTorch tensors)
env = gym.make("PickCube-v1", num_envs=N)
env = ManiSkillVectorEnv(env, auto_reset=True)

# Mode 3: Raw batch mode (non-standard, torch tensors)
env = gym.make("PickCube-v1", num_envs=N)
```

### Roboharness Integration Feasibility: Very High

The CPUGymWrapper mode is directly compatible with the standard Gymnasium Wrapper. The ManiSkillVectorEnv mode requires handling batched GPU data, similar to Isaac Lab. The RecordEpisode wrapper provides built-in recording functionality.

---

## Roboharness Integration Summary

| Project | Integration Feasibility | Integration Method | Estimated Effort |
|---------|------------------------|-------------------|-----------------|
| Isaac Lab | Very High | Standard Gymnasium Wrapper | < 1 day |
| ManiSkill | Very High | CPUGymWrapper + Standard Wrapper | < 1 day |
| LocoMuJoCo | High | Standard Gymnasium Wrapper | < 1 day |
| unitree_rl_gym | Moderately Difficult | Wrap MuJoCo sim-to-sim path | ~1 week |
| MuJoCo Playground | Difficult | JAX-native adapter | ~2 weeks |
