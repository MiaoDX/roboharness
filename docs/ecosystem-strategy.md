# Ecosystem & Community Strategy

_Last updated: April 10, 2026_
_This document covers everything outside the project itself: making roboharness discoverable, building technical influence, entering key communities._
_Technical roadmap is in `roadmap-2026-q2.md`._

---

## Core Thesis

Roboharness's biggest bottleneck is not technical capability — it's **discoverability**. The project occupies a position no one else holds (AI coding agent + visual regression testing + robot simulation), but if no one knows that position exists, larger players' tools will cover it within 6–12 months.

This strategy's goal: **make roboharness synonymous with "harness engineering for robotics"** before larger companies react.

---

## Content Strategy: Define the Category

### Article 1: Harness Engineering for Robotics (Category-Defining Piece)

**Why this must be written:** Searched every harness engineering resource — 92+ articles, projects, and papers. All target software development. Not a single one discusses robotics. Writing the first one = defining the category.

**Angle:** Not "here's roboharness," but "when AI agents start writing robot code, what kind of harness do we need?" Roboharness is one answer, but the article's value is in raising the question itself.

**Publication plan:**

- Chinese version first on WeChat (riding the Harness Engineering hype wave)
- English version on Medium / dev.to / HuggingFace blog, simultaneously submit to Hacker News
- English title suggestion: "Harness Engineering for Robotics: Why AI Coding Agents Need Eyes to Debug Robots"

**Key content points:**

1. How the three dimensions of software harness engineering (time/space/interaction) change in robotics
2. Robotics-specific challenge: physical behavior can't be textualized — screenshots are the lowest-cost feedback channel
3. Academic validation: FAEA / CaP-X
4. Concrete demo: how roboharness lets Claude Code see what the G1 is doing
5. Open questions: Is VLM judge accuracy sufficient? How to choose checkpoint granularity?

**⚡ Action items:**

- [ ] Refine Chinese draft (already exists in project files), add ROS perspective and LeRobot evaluation gap evidence
- [ ] Include showcase Live Report links + GitHub link at the end
- [ ] After publishing, tag relevant authors on Twitter/X to spark discussion

### Article 2: Technical Blog Post (already in blog/ directory)

Draft exists: "Why AI Coding Agents Don't Need a Separate VLM for Robot Debugging." More technical, published as a companion to the category piece.

**⚡ Action items:**

- [ ] Finalize `blog/why-ai-coding-agents-dont-need-a-separate-vlm-for-robot-debugging.md`
- [ ] Publish English version on Medium, tags: robotics, ai-agents, simulation, harness-engineering

### Ongoing Content

For each new showcase integration (GR00T, Pi0, etc.), publish a short post + GIF + Live Report link. Fixed format:

```
Problem → How roboharness solves it → One command to reproduce → Live Report screenshot
```

---

## ROS Community Penetration

### Why the ROS Community Matters

ROS is the de facto standard for robot development. ros-mcp-server already has 931 stars, showing strong ROS community interest in "LLM + robotics." But ROS tooling is C++/Python mixed, CI uses colcon + Industrial CI, evaluation uses rostest/launch_testing. **There is no AI agent-friendly visual testing tool in the ROS ecosystem.**

### ⚡ Action Items

**1. Post on ROS Discourse**

- [ ] Go to https://discourse.ros.org/, post in the "General" category
- Title: `Visual testing harness for AI coding agents in robot simulation — looking for feedback`
- Content outline:
  - Opening: How do you debug simulation behavior when using Claude Code / Codex for robotics development?
  - Middle: Show MuJoCo grasp demo GIF + Live Report link (https://miaodx.com/roboharness/grasp/)
  - Closing: GitHub link, explicitly state "looking for feedback, not promoting"
- This is a genuine question + an existing attempt, not a sales pitch

**2. Submit to awesome-ros2**

- [ ] Fork https://github.com/fkromer/awesome-ros2
- [ ] Add one line under `Packages` > `Testing` or `Simulation`:
  ```
  - [roboharness](https://github.com/MiaoDX/roboharness) - Visual testing harness for AI coding agents in robot simulation. Checkpoint screenshots + structured JSON for Claude Code / Codex to see and judge robot behavior.
  ```
- [ ] Submit PR titled: `Add roboharness — visual testing for AI coding agents in robot simulation`

**3. ros2_mcp Collaboration**

- [ ] Open a discussion or issue at https://github.com/LCAS/ros2_mcp
- Title: `Integration idea: roboharness for checkpoint-based visual testing alongside ros2_mcp real-time interaction`
- Content: ros2_mcp handles real-time ROS topic interaction, roboharness MCP handles checkpoint-based visual testing. Complementary, not competing. Link to roboharness MCP server docs.
- Goal: Establish connection, not request merging

**4. Gazebo Showcase (later)**

- [ ] Create `ros2-gazebo/` directory in `roboharness/showcase`
- Capture Gazebo rendered images via ROS topic + get state via TF
- This is the easiest entry point for ROS users

---

## HuggingFace Ecosystem

### ⚡ Action Items

**1. LeRobot Upstream Discussion Contributions**

- [ ] Comment on https://github.com/huggingface/lerobot/issues/538 (headless eval) — explain that roboharness already solves headless screenshot capture, include link and code example
- [ ] Comment on https://github.com/huggingface/lerobot/issues/2375 (eval reproducibility) — propose checkpoint screenshots + structured JSON as a diagnostic approach
- [ ] Do not sell — provide concrete technical solutions and code snippets, let others judge the value

**2. HuggingFace Space Demo**

- [ ] Create HuggingFace Space: `MiaoDX/roboharness-demo`
- [ ] Type: "Static HTML" — no GPU needed, just host CI-generated HTML reports
- [ ] Content: All 5 demo Live Reports (grasp, g1-reach, g1-loco, g1-native, sonic)
- [ ] Space description: project overview + GitHub link

**3. LeRobot Plugin (after eval plugin ships)**

- [ ] If LeRobot plugin system supports `pip install lerobot-plugin-roboharness`, publish one
- [ ] Otherwise, ship as standalone `roboharness[lerobot]`

---

## Academic Community

### ⚡ Action Items

**1. Conferences**

- [ ] Check ICRA 2026 (June 19-25, Vienna) workshop list for "AI for Robot Development" or "Simulation Testing" topics. If relevant, submit a demo paper
- [ ] Check CoRL 2026 submission deadline, evaluate whether to submit a system paper
- [ ] Monitor NeurIPS 2026 workshop CFPs (typically announced September)

**2. Citation Network**

- [ ] Add a "Related Work" or "See Also" section to README citing: FAEA (arXiv:2601.20334), CaP-X (arXiv:2603.22435), StepEval (arXiv:2509.19524), SOLE-R1 (arXiv:2603.28730), AOR (arXiv:2603.04466)
- [ ] Cite these works in the category-defining article — cited authors will notice
- [ ] If the showcase includes a CaP-X comparison demo, email the CaP-X first author to discuss complementarity

---

## GitHub Strategy

### Org Structure

`github.com/roboharness` org exists, currently empty. Planned structure:

```
roboharness/              (GitHub org, exists)
├── showcase              (integration demo repo, to be created)
├── .github               (org-level profile README, to be created)
└── roboharness.github.io (later, if a standalone site is needed)
```

MiaoDX/roboharness remains the core repo's primary location. Reason: existing stars, forks, PyPI links, CI config are all there. Consider transfer to org only when the project scales (5+ external contributors).

### ⚡ Action Items

**1. Org Profile README**

- [ ] Create `.github` repo under `roboharness` org
- [ ] Write `profile/README.md`:

```markdown
# roboharness

**Harness Engineering for Robotics** — visual testing infrastructure for AI coding agents in robot simulation.

| Repo | Description |
|------|-------------|
| [MiaoDX/roboharness](https://github.com/MiaoDX/roboharness) | Core library — checkpoint screenshots + structured JSON for Claude Code / Codex |
| [roboharness/showcase](https://github.com/roboharness/showcase) | Integration demos — GR00T, Pi0, LeRobot, SONIC, Isaac Lab |

📄 [Live Visual Reports](https://miaodx.com/roboharness/) · 📦 [PyPI](https://pypi.org/project/roboharness/)
```

**2. Showcase Repo Initialization**

- [ ] Create `roboharness/showcase` repo
- [ ] Initialize skeleton: README + directory structure (see `roadmap-2026-q2.md`)
- [ ] First showcase: extract `examples/lerobot_g1_native.py` as a standalone runnable showcase
- [ ] Each showcase directory contains: README.md, requirements.txt, run.sh, main script
- [ ] CI matrix: one independent job per showcase

**3. Submit to awesome-harness-engineering**

- [ ] Fork https://github.com/walkinglabs/awesome-harness-engineering
- [ ] Find appropriate category (likely "Tools" or create a new "Robotics" category), add:
  ```
  - [roboharness](https://github.com/MiaoDX/roboharness) - Visual testing harness for AI coding agents in robot simulation. The first harness engineering tool for robotics.
  ```
- [ ] Submit PR

**4. Submit to harn.app**

- [ ] Go to https://harn.app, find submission method (usually GitHub issue or form)
- [ ] Submit roboharness, categorize as "Testing / Evaluation"
- [ ] Emphasize: the only harness engineering tool targeting robotics

---

## Funding & Accelerators

### ⚡ Action Items

**Anthropic Claude for Open Source (deadline June 30, 2026)**

- [ ] Check application at https://www.anthropic.com/open-source
- [ ] Application talking points:
  - Positioning: AI agent testing infrastructure for robotics
  - Core narrative: roboharness has 96% commits by Claude Code — the project is itself proof of Claude's capability
  - Apply via critical infrastructure exception path (star count insufficient but qualifies as critical infrastructure)
- [ ] Apply immediately, don't wait for star growth

**Robotics Factory Accelerate (deadline July 18, 2026)**

- [ ] Check requirements at https://www.roboticsfactory.org/accelerate
- [ ] Evaluate fit (leans toward company-building, not pure open source)
- [ ] If aligned, prepare application

**Google DeepMind Accelerator: Robotics**

- [ ] Monitor https://deepmind.google/models/gemini-robotics/accelerator/ for future cohort announcements
- [ ] If it opens to Asia or globally, apply immediately

**Long-term**

- [ ] Q4 2026: Prepare GSoC 2027 org application materials (project ideas list, contributor guide)
- [ ] End of Q2 2026: Evaluate NumFOCUS affiliated project application feasibility

---

## Collaborations & Positioning

### Relationship Map

| Project | Relationship | Concrete Action |
|---------|-------------|-----------------|
| LeRobot | Complementary: they train & deploy, we test | Comment on #538 and #2375; ship eval plugin; HF Space demo |
| CaP-X | Complementary: they synthesize policies, we test them | Showcase comparison demo; contact first author |
| ros-mcp-server / ros2_mcp | Complementary: real-time control vs checkpoint testing | Open discussion on ros2_mcp |
| Isaac Lab-Arena | Complementary: large-scale benchmarks vs visual debugging | Showcase integration |
| rl_sar | Complementary: sim-to-real deployment vs simulation testing | Potential robowbc collaboration |
| mujoco-mcp | Highest overlap | Differentiate: testing vs interaction |

### Relationship with Large Companies

- **NVIDIA**: roboharness supports GR00T + SONIC + Isaac Lab full stack. After showcase ships, contact NVIDIA DevRel for official documentation reference
- **HuggingFace**: LeRobot plugin + Space demo. After shipping, tag LeRobot team members
- **Anthropic**: Claude for Open Source application + roboharness as Claude Code best-practice showcase

---

## Metrics

How to know if the strategy is working:

**Short-term signals (weeks):**
- ROS Discourse post views and replies
- awesome-harness-engineering PR merged or not
- harn.app listing status
- HuggingFace Space visit count

**Medium-term signals (months):**
- GitHub star growth rate
- Whether LeRobot community members mention roboharness in discussions
- External contributors submitting PRs
- Showcase repo usage

**Long-term signals:**
- Cited in academic papers
- Referenced in NVIDIA / HuggingFace official documentation
- Mentioned as a default recommendation in "AI + Robotics" discussions

---

## Action Summary

Ordered by immediate executability. Tagged by executor:

| # | Action | Executor | Dependencies |
|---|--------|----------|-------------|
| 1 | Create `.github` repo in `roboharness` org + org profile README | Claude Code | Org exists ✅ |
| 2 | Create `roboharness/showcase` repo + skeleton | Claude Code | #1 |
| 3 | Add Related Work section to README (FAEA/CaP-X/StepEval/SOLE-R1/AOR) | Claude Code | None |
| 4 | Submit PR to awesome-harness-engineering | Claude Code | None |
| 5 | Submit PR to awesome-ros2 | Claude Code | None |
| 6 | Create HuggingFace Space (Static HTML, host 5 demo reports) | Claude Code | None |
| 7 | Apply for Anthropic Claude for Open Source | Human | None |
| 8 | Comment on LeRobot #538 and #2375 with roboharness solutions + code examples | Human | None |
| 9 | Submit to harn.app | Human | Check submission method |
| 10 | Finalize and publish category-defining article (Chinese, WeChat) | Human + Claude | Draft exists |
| 11 | Post on ROS Discourse seeking feedback | Human | None |
| 12 | Open discussion on ros2_mcp about collaboration | Human | None |
| 13 | Publish English article on Medium + submit to Hacker News | Human | After #10 |
| 14 | Check ICRA 2026 workshop list, evaluate demo paper feasibility | Human | None |
