# Harness Engineering for Robotics: Why AI Coding Agents Need Eyes to Debug Robots

*How the three dimensions of harness engineering change when the code controls physical bodies — and what roboharness does about it.*

---

When OpenAI, Cursor, and Anthropic published their 2025–2026 harness engineering reports, they crystallized something practitioners had discovered independently: the highest-leverage work in an AI-agent development loop isn't writing prompts. It's designing the **environment** — the harness — that lets the agent observe, judge, and improve its own work.

The consensus across those reports maps to three independent scaling dimensions:

- **Time scaling** — one agent running longer without degrading
- **Space scaling** — many agents running in parallel
- **Interaction scaling** — humans managing agents without burning out

Those dimensions exist in software. The question this article asks is: **what happens to each dimension when the code controls a physical body?**

---

## The Physical Gap: Why Software Harness Engineering Breaks Down in Robotics

In software agent loops, text is the universal currency. An agent writes code, CI runs it, textual output (logs, test failures, coverage numbers) flows back. The feedback loop operates entirely in symbols that the agent already understands.

Robotics breaks this assumption. **Physical behavior cannot be fully textualized.**

Consider a humanoid robot trying to lift a cube. The control code runs. What is the result? You could write:

```
joint_7_angle: 0.847 rad
contact_force: 2.3 N
object_height: 0.002 m
```

But none of that tells you that the gripper fingers closed around the *edge* of the cube instead of its center, so the lift barely started before the object rotated out of the grasp. A developer looking at a picture catches this immediately. A text parser doesn't know what it missed.

This is the first robotics-specific harness engineering challenge: **screenshots are the lowest-cost feedback channel for physical behavior**. Not because agents lack intelligence, but because the information content of a rendered robot frame is fundamentally richer than any scalar summary the physics engine can produce.

---

## The Three Dimensions, Remapped for Robotics

### Dimension 1: Time Scaling — Making Agent Loops Run Longer

In software, Anthropic's Planner-Generator-Evaluator architecture addresses time scaling. The key insight is that the evaluator must not share context with the generator, or it will be persuaded by the generator's reasoning rather than inspecting the actual output.

For robotics, this maps naturally:

- **Planner** — decompose a task into phases: `plan → pre_grasp → approach → contact → grasp → lift`
- **Generator** — the agent writes and modifies control code; the harness runs the simulation
- **Evaluator** — reads only the harness output (PNG files + structured JSON), not the generator's reasoning

The clean boundary between generator and evaluator is the harness checkpoint: a named moment where the simulator pauses, multi-view screenshots are saved to disk, and state data is recorded. The evaluator reads files, not agent logs.

A second insight from the Anthropic report applies directly: **each harness component is a hypothesis about the current model's capability boundary**. The component that captures side-view screenshots exists because the agent can't yet detect gripper offset from front-view alone. As models improve, some components become unnecessary. A well-designed harness makes these hypotheses explicit so you know which components to try removing first.

### Dimension 2: Space Scaling — Many Parallel Trials

Cursor's recursive Planner-Worker architecture for hundreds of parallel agents identified one non-negotiable requirement: **execution isolation**. Shared state collapses under parallel load.

For robotics simulation, this translates directly: each parallel trial needs its own simulator instance, its own output directory, and no shared mutable state with other trials. A `GraspTaskStore` that serializes single trials cannot aggregate statistics across a policy sweep.

The second parallel-scaling insight from Cursor: **decentralized quality control scales better than a central gate**. Don't wait for every trial to pass before starting the next batch. Accept a statistical success rate; let subsequent runs improve it. The evaluator scores individual trials; the aggregator computes batch statistics. These are separate concerns.

For robotics, this matters because robot behavior is stochastic — a policy might succeed 80% of trials depending on initial conditions. A harness that reports "passed" or "failed" on a single trial is less useful than one that reports "82% success rate across 50 randomized trials, with failure concentrated in cases where the object starts at a lateral offset > 3cm."

### Dimension 3: Interaction Scaling — Reducing Human Review Burden

OpenAI's Symphony framing is apt here: **the goal is a loop where humans write task specifications and the harness + agents handle everything else**.

In software, this looks like: CI catches regressions automatically, so engineers review exceptions rather than every commit. In robotics, the equivalent is: task success gates catch behavior regressions automatically, so engineers don't review every simulation run.

The key requirement is mechanical: a CI job that runs the robot task and **asserts** task success against physical constraints — cube height, contact forces, gripper symmetry. Not "did the code run without crashing?" but "did the robot accomplish the goal?"

OpenAI also notes that automated entropy detection — catching when agent modifications quietly degrade behavior over time — is what makes the interaction model sustainable at scale. Without it, engineers eventually have to re-review everything because they can't trust accumulated agent changes.

---

## What the ROS World Lacks

ROS is the de facto standard for robot development. ros-mcp-server has over 900 stars, demonstrating strong community interest in LLM + robotics tooling. But ROS evaluation tooling — rostest, launch_testing, colcon — was designed for integration testing of software components, not for visual inspection of physical behavior.

The pattern in ROS is: write a launch test that checks if nodes start and topics appear. Not: run the robot for 30 seconds, capture what it did, and let an AI agent look at it.

There is no AI agent-friendly visual testing tool in the ROS ecosystem. Gazebo can publish camera topics over ROS; nothing currently takes those frames, aligns them with simulation timestamps, saves them at semantically meaningful checkpoints, and exposes them to an agent in a consumable structure. The infrastructure for this exists (ROS bags, image_transport, TF), but nobody has assembled it into a harness.

---

## The LeRobot Evaluation Gap

HuggingFace's LeRobot provides policy training, evaluation scripts, and a growing model hub for robot learning. Its `eval` command runs a policy and reports success rates. What it doesn't provide is **visual evidence for why the policy succeeded or failed in individual episodes**.

Two open LeRobot issues illustrate this gap directly:

- [lerobot#538](https://github.com/huggingface/lerobot/issues/538) — headless evaluation without a display. The issue is that `render_mode="human"` requires a GUI. The practical workaround (offscreen rendering to files) is exactly what a checkpoint harness provides, but there is no standard way to hook it into the eval loop.

- [lerobot#2375](https://github.com/huggingface/lerobot/issues/2375) — evaluation reproducibility. When a policy's success rate changes between runs, there's no artifact trail that lets you understand *what* changed in the physical behavior. Checkpoint screenshots + structured state JSON would provide that trail.

The pattern is: LeRobot measures *whether* the task succeeded; it doesn't capture *how* the robot behaved so that an agent (or human) can diagnose the gap. For scripted controllers this is acceptable. For iterative AI-agent development — where the agent is modifying the policy and needs to see the effect — it's a missing feedback channel.

---

## Roboharness: A Concrete Implementation

[Roboharness](https://github.com/MiaoDX/roboharness) is an open-source visual testing harness that implements this pattern. The core design:

```python
from roboharness import Harness, GRASP_PROTOCOL

harness = Harness(backend, output_dir="./trial_001", task_name="grasp")
harness.load_protocol(GRASP_PROTOCOL)  # plan → pre_grasp → approach → grasp → lift

harness.reset()
for phase_actions in control_phases:
    result = harness.run_to_next_checkpoint(phase_actions)
    # result.views → multi-view PNG files, already on disk
    # result.state → joint angles, contact forces, object poses
    # The agent reads both and decides what to modify
```

At each checkpoint, the harness captures multi-view screenshots (front, side, top, wrist camera) and saves structured state as JSON. The agent reads these files directly — no separate VLM, no API roundtrip, no lossy text summary of the visual result. The agent that wrote the code is the same agent that evaluates the result.

Output layout:

```
trial_001/
  pre_grasp/
    front_rgb.png     ← agent sees gripper position above object
    side_rgb.png      ← verifies approach angle
    top_rgb.png       ← checks horizontal alignment
    state.json        ← joint positions, contact forces
  approach/
    front_rgb.png
    ...
  lift/
    front_rgb.png     ← agent confirms cube left the table
    ...
```

The `SimulatorBackend` is a structural Protocol — any simulator that implements `step()`, `reset()`, and `render()` works without inheriting from a base class. MuJoCo ships as a built-in backend; Isaac Lab and ManiSkill environments integrate through the included Gymnasium wrapper.

An evaluation module lets you write constraint-based assertions against captured state:

```python
from roboharness.evaluate import Evaluator

evaluator = Evaluator(result)
evaluator.assert_object_height(min_height=0.05, checkpoint="lift")
evaluator.assert_contact_force(max_force=15.0, checkpoint="grasp")
evaluator.assert_gripper_symmetry(max_asymmetry=0.02, checkpoint="grasp")
```

This closes the loop from "agent looks at screenshots" to "CI verifies task success automatically." A PR that breaks grasp behavior gets blocked without human review — the interaction-scaling principle from OpenAI's Symphony, applied to robotics.

---

## The AOR Precedent

The closest published academic work is AOR (Act-Observe-Rewrite, arXiv:2603.04466). AOR feeds a multimodal LLM (Claude) key-frame RGB images from robosuite simulations alongside structured diagnostic signals. On each iteration, the model rewrites a complete Python controller class; the controller is dynamically compiled and run. On the robosuite Lift and PickPlaceCan tasks, AOR achieves 100% success without gradient updates, demonstrations, or reward engineering.

The AOR result validates two things:

1. Multimodal coding agents (not specialized VLMs) are sufficient evaluators for physical task behavior
2. Iterative visual feedback — screenshots at semantically meaningful moments — is enough signal for the agent to converge

Roboharness generalizes this from robosuite to any simulator via the `SimulatorBackend` protocol, adds structured multi-view capture, and packages it as reusable infrastructure. AOR is the existence proof; roboharness is the library.

---

## Open Questions

This is a nascent area. Three questions without settled answers:

**How granular should checkpoints be?** Too few and the agent misses the moment where behavior diverged. Too many and the agent gets overwhelmed with redundant frames. The current heuristic is: one checkpoint per semantically distinct phase. For grasping: plan, pre-grasp, approach, contact, lift. For locomotion: stand, walk, stop. Whether this generalizes to manipulation tasks with more continuous contact is an open question.

**Is VLM judge accuracy sufficient for automated CI?** For binary pass/fail (cube lifted? yes/no), a multimodal agent is reliable. For subtle quality metrics (gait smoothness, grasp stability under perturbation), the answer is less clear. The constraint-based evaluator (`evaluator.assert_object_height(...)`) handles cases where ground truth is numeric. The visual judgment handles cases where it isn't. The boundary between these two modes needs more empirical work.

**What's the right checkpoint for sim-to-real validation?** Simulation checkpoints capture sim state. Real robot checkpoints capture real camera feeds. The visual outputs look similar but encode different uncertainties. Whether a harness trained on sim checkpoints transfers to real robot debugging without retuning the checkpoint positions is not yet known.

---

## Try It

```bash
pip install roboharness[demo]
python examples/mujoco_grasp.py --report
```

Live visual reports, auto-generated from CI on every push:

- [MuJoCo Grasp](https://miaodx.com/roboharness/grasp/) — scripted grasp with multi-view checkpoint captures
- [G1 WBC Reach](https://miaodx.com/roboharness/g1-reach/) — whole-body IK reaching (Pinocchio + Pink)
- [G1 Locomotion](https://miaodx.com/roboharness/g1-loco/) — GR00T RL stand → walk → stop
- [SONIC Motion Tracking](https://miaodx.com/roboharness/sonic/) — encoder+decoder pipeline from MoCap

The [showcase repository](https://github.com/roboharness/showcase) has standalone integration demos for GR00T N1.6, LeRobot G1, and SONIC locomotion.

---

*[Roboharness](https://github.com/MiaoDX/roboharness) is MIT-licensed. Contributions and feedback welcome.*

*Related: [Why AI Coding Agents Don't Need a Separate VLM for Robot Debugging](./why-ai-coding-agents-dont-need-a-separate-vlm-for-robot-debugging.md) — a more technical companion piece on the single-agent evaluation pattern.*
