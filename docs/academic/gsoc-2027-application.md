# GSoC 2027 — Organization Application Draft

_Status: Draft — finalize and submit when org application window opens (~January 2027)_  
_Reference timeline: GSoC 2026 org window was Jan 22 – Feb 6; accepted orgs announced ~Feb 21_

---

## Organization Profile

**Organization name:** Roboharness

**Short description (≤ 180 chars):**
Visual testing harness for AI coding agents in robot simulation. Helps Claude Code and Codex see what robots are doing and iterate autonomously.

**Long description:**

Roboharness is the first harness engineering tool built specifically for robotics. When AI coding agents (Claude Code, Codex) write or debug robot control code, they are blind — they can read logs and numbers but cannot see what the robot is actually doing. Roboharness solves this by capturing multi-view screenshots and structured state JSON at user-defined checkpoints, giving agents the visual context they need to iterate autonomously.

The project supports MuJoCo, Gymnasium (zero-change `RobotHarnessWrapper`), Isaac Lab, LeRobot G1 locomotion, and SONIC motion tracking. It produces auto-generated HTML visual reports that CI can publish, and includes an MCP server so agents can query checkpoint results directly. The project is MIT-licensed, Python 3.10+, and follows a SimulatorBackend protocol for easy extension to new simulators.

Roboharness is where robotics meets harness engineering — a category that until now existed only for software, not physical systems.

**Website:** https://github.com/MiaoDX/roboharness  
**License:** MIT  
**Primary language:** Python  
**Tags:** robotics, simulation, testing, AI agents, MuJoCo, Gymnasium, evaluation

---

## Why Should Google Accept Roboharness?

1. **Category-defining project.** Roboharness occupies a unique niche: visual regression testing for robot simulation. No other open-source tool targets this exact intersection of AI coding agents + robot simulation + visual feedback.

2. **Growing relevance.** AI coding agents (Claude Code, Codex) are now writing real robot code. The tooling to support them — especially visual testing and evaluation — doesn't exist yet. Roboharness fills that gap.

3. **Student-friendly architecture.** The `SimulatorBackend` protocol (structural typing, no inheritance) makes adding a new simulator backend an ideal GSoC project: well-scoped, testable, standalone, and directly useful to the community.

4. **Active development.** Regular commits, CI with real robot simulation tests, live HTML reports auto-generated on every push.

5. **Mentor capacity.** Two core mentors available (see Mentors section), with domain expertise in robotics simulation, Python, and AI agent tooling.

---

## Project Ideas

### 1. Gazebo / ROS 2 Backend (Medium, 350 hours)

**Goal:** Implement a `SimulatorBackend` for ROS 2 / Gazebo, letting roboharness capture checkpoint screenshots from a running Gazebo simulation via ROS topics and TF.

**Why it matters:** Gazebo is the most widely used simulator in the ROS ecosystem. A ROS 2 backend would make roboharness accessible to thousands of ROS developers.

**Skills:** Python, ROS 2 (rclpy), Gazebo, basic image processing  
**Difficulty:** Medium  
**Deliverables:**
- `roboharness/backends/ros2_gazebo.py` — implements `SimulatorBackend` protocol
- Subscribes to `/camera/image_raw` (or similar) ROS topic for screenshot capture
- Gets robot state via `tf2_ros`
- Integration tests using `roboharness[dev]` (mock ROS publisher)
- Example: `examples/ros2_gazebo_example.py`
- Documentation

**Mentor:** TBD

---

### 2. VLM Evaluation Judge (Hard, 350 hours)

**Goal:** Add an optional `VLMJudge` evaluator that scores checkpoint screenshots using an open-source vision-language model (e.g., a 4B-parameter robot-specific VLM).

**Why it matters:** Currently, pass/fail evaluation requires hand-written constraint functions. A VLM judge would enable natural-language task specifications like "is the robot holding the cube?" without writing code.

**Skills:** Python, HuggingFace Transformers, vision-language models, evaluation methodology  
**Difficulty:** Hard  
**Deliverables:**
- `roboharness/evaluate/vlm_judge.py` — pluggable VLM backend with a common interface
- Support for at least two backends: local model (HuggingFace) + API model (Claude via anthropic SDK)
- Prompt templates for common robot tasks (grasp success, reach target, avoid collision)
- Benchmarking script comparing VLM scores vs. ground-truth constraint evaluators
- Tests using mocked VLM responses
- Documentation

**Mentor:** TBD

---

### 3. Isaac Lab Checkpoint Integration (Medium, 175 hours)

**Goal:** Complete the Isaac Lab integration with full round-trip checkpoint capture and auto-generated HTML visual reports, matching the quality of the MuJoCo backend.

**Why it matters:** NVIDIA Isaac Lab is the dominant GPU-based robot learning platform. Full roboharness support would make it accessible to Isaac Lab users without changing their training/eval code.

**Skills:** Python, NVIDIA Isaac Lab, PyTorch (basic), understanding of `SimulatorBackend` protocol  
**Difficulty:** Medium  
**Deliverables:**
- `roboharness/backends/isaac_lab.py` — full implementation (currently partial)
- Round-trip: observation → checkpoint → screenshot → structured JSON → HTML report
- `examples/isaac_lab_harness_example.py` using a standard Isaac Lab environment
- CI smoke test (mocked; real GPU test in separate CI tier)
- Documentation

**Mentor:** TBD

---

### 4. Web Dashboard for Live Monitoring (Medium, 350 hours)

**Goal:** Build a lightweight web dashboard that shows live checkpoint screenshots and state data during a running simulation, using Rerun or a simple WebSocket server.

**Why it matters:** Currently, roboharness produces static HTML reports after the fact. A live dashboard would let agents (and humans) monitor robot behavior in real time.

**Skills:** Python, WebSocket, HTML/CSS/JS (basic), Rerun SDK or similar  
**Difficulty:** Medium  
**Deliverables:**
- `roboharness/server/` — lightweight async WebSocket server (Python)
- Live screenshot streaming (JPEG/PNG over WebSocket)
- Static HTML frontend with auto-refreshing image grid + state JSON panel
- `harness serve` CLI command
- Tests (mock WebSocket client)
- Documentation

**Mentor:** TBD

---

### 5. Benchmark Suite for Whole-Body Controllers (Medium, 350 hours)

**Goal:** Create a standardized benchmark suite that uses roboharness checkpoints to evaluate WBC (whole-body control) policies, producing comparable metrics across different controllers.

**Why it matters:** There is no standard benchmark for WBC evaluation. roboharness checkpoint data (screenshots + joint state + task success) is the right primitive for building one.

**Skills:** Python, MuJoCo, robotics (basic kinematics), statistics  
**Difficulty:** Medium  
**Deliverables:**
- `roboharness/benchmark/` — benchmark runner + metric collection
- At least 3 task scenarios: reach, grasp, balance
- Per-scenario pass/fail + summary report
- Comparison across at least 2 controllers (e.g., Pinocchio+Pink WBC vs. scripted baseline)
- HTML report with per-task visual breakdown
- Documentation

**Mentor:** TBD

---

## Mentors

_Note: Finalize mentor list before submitting. Each mentor must have a GitHub account and agree to the GSoC mentoring commitment (~5 hours/week during coding period)._

| Name | GitHub | Expertise | Availability |
|------|--------|-----------|--------------|
| TBD | @TBD | Robotics simulation, Python | TBD |
| TBD | @TBD | AI agents, ML tooling | TBD |

**Minimum required:** 2 mentors (1 primary + 1 backup per project)

---

## Communication

- **GitHub Issues/Discussions:** https://github.com/MiaoDX/roboharness/discussions (primary channel)
- **Discord:** TBD (set up before org application)
- **Mailing list:** TBD (Google Groups or similar)

---

## Application Checklist

Before submitting the org application:

- [ ] At least 2 confirmed mentors with GSoC accounts
- [ ] Discord server set up with `#gsoc-2027` channel
- [ ] GitHub Discussions enabled and active
- [ ] Project ideas page published on GitHub Wiki or docs site
- [ ] CONTRIBUTING.md updated with GSoC-specific onboarding section
- [ ] At least 3 merged contributor PRs from non-core team (shows welcoming community)
- [ ] README has "Contributing" and "Community" sections prominently visible

---

## Submission Steps (when window opens ~Jan 2027)

1. Go to https://summerofcode.withgoogle.com/
2. Sign in with a Google account associated with a GitHub account that has org admin access to MiaoDX/roboharness (or the `roboharness` org)
3. Click "Apply as a Mentoring Organization"
4. Fill in the profile using the sections above
5. Submit before the deadline (~February 6, 2027 based on 2026 timeline)
6. Monitor for acceptance announcement (~February 21, 2027)

---

_See also: `docs/academic/conference-opportunities-2026.md` for conference strategy_  
_Issue: miaodx/roboharness#157_
