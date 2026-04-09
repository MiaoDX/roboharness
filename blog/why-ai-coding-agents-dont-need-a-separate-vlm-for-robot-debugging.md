# Why AI Coding Agents Don't Need a Separate VLM for Robot Debugging

*A practical argument from building [Roboharness](https://github.com/MiaoDX/roboharness), an open-source visual testing harness for robot simulation.*

---

## The Conventional Wisdom

When people talk about "AI for robotics," the architecture usually looks like this:

1. A **coding model** writes or modifies the control algorithm.
2. The simulation runs.
3. A **separate Vision-Language Model (VLM)** evaluates the result by looking at rendered frames.
4. The VLM's verdict is fed back to the coding model.
5. The coding model adjusts its approach.

This pipeline seems natural — division of labor, specialization, modularity. But after months of building and iterating on robot grasping, locomotion, and whole-body control with AI coding agents, we discovered something surprising: **the separate VLM is unnecessary overhead**.

## The Key Insight: Your Coding Agent Is Already Multimodal

Modern AI coding agents — Claude Code, OpenAI Codex, and others — are not text-only tools. They are **natively multimodal**: they read code, see images, interpret structured data, and reason across all of these simultaneously. When Claude Code looks at a PNG screenshot of a robot simulation, it doesn't need a separate model to tell it what's happening. It can see the gripper, the object, the contact points, and the spatial relationships — and immediately connect what it sees to the code it just wrote.

This changes the architecture fundamentally. Instead of a pipeline with a handoff between models:

```
Code Agent  →  Simulator  →  VLM  →  Verdict  →  Code Agent
```

You get a tight loop with one agent doing everything:

```
Code Agent  →  Simulator  →  Screenshots  →  Code Agent
                                (same agent inspects + iterates)
```

The agent writes the control code, triggers the simulation, looks at the resulting multi-view screenshots, judges whether the behavior is correct, and modifies its approach — all within a single reasoning context.

## What This Looks Like in Practice

In [Roboharness](https://github.com/MiaoDX/roboharness), we built a visual testing harness that makes this loop concrete. Here's a typical grasping workflow:

```python
from roboharness import Harness

harness = Harness(backend, output_dir="./output", task_name="grasp")
cameras = ["front", "side", "top"]
harness.add_checkpoint("pre_grasp", cameras=cameras, trigger_step=500)
harness.add_checkpoint("approach", cameras=cameras, trigger_step=1000)
harness.add_checkpoint("grasp", cameras=cameras, trigger_step=1800)
harness.add_checkpoint("lift", cameras=cameras, trigger_step=2600)

harness.reset()
for actions in grasp_phases:
    result = harness.run_to_next_checkpoint(actions)
    # result.views → list of CameraView objects (front, side, top)
    # result.state → joint angles, contact forces, object poses
    # The coding agent inspects BOTH and decides what to do next
```

At each checkpoint, the harness captures multi-view screenshots and saves them as standard PNG files alongside structured state data (JSON). The agent reads the images directly — no VLM API call, no separate inference pipeline, no translation layer between visual understanding and code modification.

The output on disk looks like this:

```
harness_output/grasp/trial_001/
  pre_grasp/
    front_rgb.png      ← agent sees the gripper is open, positioned above the cube
    side_rgb.png       ← confirms approach angle from a perpendicular view
    top_rgb.png        ← verifies horizontal alignment
    state.json         ← joint positions, contact forces
  approach/
    front_rgb.png      ← agent checks if the gripper has lowered correctly
    ...
  lift/
    front_rgb.png      ← agent confirms the cube is off the table
    ...
```

When the agent spots a problem — say, the gripper approached from too steep an angle — it modifies the control code and reruns from the relevant checkpoint. No round trip to an external service. No latency from a second model. No lossy natural-language description of what the VLM "thinks" it saw.

## Why the Separate-VLM Architecture Hurts

### 1. Information Loss at Every Handoff

When a VLM analyzes a simulation frame and produces text like *"the gripper appears to be slightly misaligned with the object"*, crucial spatial detail is lost. The coding agent now has to work from a verbal description rather than the image itself. It doesn't know *how much* misalignment, in *which direction*, or whether the depth perception is an artifact of the camera angle. The original pixel data contained all of this; the text summary doesn't.

### 2. Context Fragmentation

The coding agent wrote the control algorithm. It knows the intended trajectory, the joint limits, the expected contact geometry. When it looks at the simulation screenshot *directly*, it can connect visual observations to specific lines of code. A separate VLM has none of this context — it sees a frame in isolation and must describe what it observes without understanding *why* the robot was supposed to be in a particular configuration.

### 3. Latency and Cost

Every VLM evaluation adds an API call, network latency, and inference cost. In an iterative debugging loop where the agent might run 10-20 trials to nail a grasp strategy, this adds up quickly. With the single-agent approach, the "evaluation" is just the agent reading files it already has access to.

### 4. Error Compounding

Two models means two points of failure. The VLM might misinterpret a shadow as a gap, or describe a successful grasp as failed because of an unusual camera angle. The coding agent then "fixes" something that wasn't broken. With one agent doing both coding and visual judgment, errors in visual reasoning are immediately confronted with the agent's own understanding of what the code should produce.

## The Harness Pattern: Structured Visual Presentation

If the coding agent is doing the visual evaluation, what does the infrastructure need to provide? Not VLM wrappers or evaluation APIs — just **structured visual output in a format the agent can consume directly**.

This is what Roboharness does. The framework is built around three concepts:

**Checkpoints** — Named moments in the simulation timeline where the harness pauses, captures data, and lets the agent inspect. Instead of arbitrary frame captures, checkpoints correspond to semantically meaningful phases: `pre_grasp`, `approach`, `contact`, `lift`.

**Multi-view Captures** — At each checkpoint, the harness renders from multiple camera angles (front, side, top, wrist) and saves them as PNGs. The agent sees the same scene from different perspectives, catching issues that a single viewpoint would miss — a perfectly aligned front view might hide a lateral offset only visible from the side camera.

**State Snapshots** — Alongside images, the harness saves structured numerical data: joint angles, contact forces, object poses. The agent can cross-reference what it *sees* with what the physics engine *reports*. This dual channel — visual plus numerical — is more powerful than either alone.

The key design decision is that none of this requires the images to be sent to a separate model. The files are saved to disk in a standard layout. The coding agent — which already has file system access as part of its development workflow — simply reads them.

## What About Complex Visual Reasoning?

A fair objection: *"Sure, a coding agent can see that a gripper is above a cube. But can it detect subtle physics issues like object interpenetration, or evaluate aesthetic quality of a locomotion gait?"*

In practice, we've found the answer is yes — for the debugging use case. The agent doesn't need to be a state-of-the-art object detector. It needs to answer questions like:

- Is the cube off the table? (yes/no, visually obvious)
- Are the gripper fingers contacting the object symmetrically? (compare side views)
- Is the robot's posture stable during locomotion? (check for excessive lean or wobble)
- Did the robot reach the target position? (compare current frame to goal specification)

These are **task-grounded visual questions** where the agent already knows what to look for because it designed the behavior. This is fundamentally different from open-ended image understanding. The agent isn't asked "what's in this image?" — it's asked "did my code produce the intended physical result?"

For cases where pixel-level analysis isn't sufficient, the structured state data provides ground truth. Joint angles don't lie. Contact force magnitudes are exact. The combination of "looks right visually" plus "numbers match expectations" catches virtually everything that matters for debugging.

## When You Actually Need a Separate VLM

To be clear, there *are* scenarios where a dedicated vision model adds value:

- **Sim-to-real transfer validation**, where you're comparing real camera feeds to simulation renders and need robust visual feature matching.
- **Safety-critical deployment**, where an independent visual monitor provides redundancy.
- **Large-scale automated evaluation**, where you need to score thousands of rollouts without human-in-the-loop and want a specialized evaluator optimized for throughput.

But for the **iterative debugging loop** — where a coding agent is developing, testing, and refining robot behavior in simulation — the separate VLM is an architectural detour. The agent that writes the code is the best-positioned entity to evaluate the result.

## Try It

Roboharness is open source and designed to work with any simulator that implements a simple protocol interface:

```bash
pip install roboharness[demo]
python examples/mujoco_grasp.py --report
```

This runs a complete grasp sequence with multi-view checkpoint captures and generates an HTML report you can inspect. MuJoCo is supported as a built-in backend; Isaac Lab and ManiSkill environments work through the included Gymnasium wrapper for drop-in integration.

The [interactive visual reports](https://miaodx.com/roboharness/) are auto-generated from CI on every push — MuJoCo grasping, G1 humanoid locomotion, whole-body reaching, and more.

---

*Roboharness is MIT-licensed and available at [github.com/MiaoDX/roboharness](https://github.com/MiaoDX/roboharness). Contributions and feedback welcome.*
