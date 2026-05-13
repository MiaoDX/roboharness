# Strategic Direction Review — April 2026

_Review date: April 4, 2026_
_Participants: Project lead (MiaoDX) + Claude_
_Purpose: Periodic strategic review to keep roboharness aligned with its goals. This document will be updated at each review cycle._

---

## Project goals (unchanged)

1. **Build real value for the robotics open-source community** — a testing harness that people actually use
2. **Establish the project lead's technical influence** at the intersection of AI Agents + Robotics

Every decision below is evaluated against these two goals.

---

## Part 1: Current situation assessment

### What roboharness is today

Roboharness is a visual testing harness for AI coding agents in robot simulation. It lets agents like Claude Code and Codex **see** what a robot is doing (multi-view screenshots at checkpoints), **judge** whether it's working (structured state JSON), and **iterate** autonomously.

Core capabilities already implemented:
- Checkpoint-based simulation pausing with multi-view capture
- MuJoCo + Meshcat backend
- Gymnasium wrapper for zero-change integration
- CLI tools (`roboharness inspect`, `roboharness report`)
- Rerun integration for multi-modal logging
- CI with lint + type check + test + MuJoCo example
- G1 humanoid WBC reach example
- MuJoCo grasp example with visual report

### The origin story matters

The project was born from a real workflow: the lead developed a visual debugging and agent-feedback loop while working on G1 and Realman robot grasping tasks using GR00T WBC. The implementation was done first (directly in the grasp project, with Claude Code and Codex), then abstracted into this library. This is the correct open-source origin pattern — extract from working code, not design in a vacuum.

However, the library has not yet been re-integrated back into the original grasp project, nor validated against other popular community projects. This is the most important gap to close.

### Competitive landscape (as of April 2026)

**The key strategic insight: no tool combines visual regression testing + robot simulation + AI coding agent evaluation.** Every existing tool evaluates policies through success metrics, none through visual behavior correctness.

| Tool | What it tests | Visual | CI/CD | AI agent-aware |
|------|--------------|--------|-------|----------------|
| Isaac Lab-Arena | VLA policies | No | No | No |
| CaP-X / CaP-Bench (NVIDIA/Stanford) | LLM code generation | Partial | No | Yes |
| SimplerEnv | Policy sim-to-real | Partial | No | No |
| Robot Testing Framework (IIT) | Robot software (iCub) | No | Yes | No |
| ManiSkill3 | RL/IL policy eval | No | No | No |
| **roboharness** | **AI agents' robot code** | **Yes** | **Yes** | **Yes** |

CaP-X (March 2026) is the most relevant comparison. Its Visual Differencing Module compares before/after images, but lacks interactive visualization, human annotation, side-by-side multi-solution comparison, and CI/CD integration. CaP-X benchmarks models; roboharness tests development workflows. They are complementary, not competitive.

### Market forces creating the window

Five developments define the current moment:

1. **MCP servers for robotics are taking off.** ros-mcp-server (~1,100 GitHub stars) connects LLMs to robots via MCP and ROS. mujoco-mcp lets LLMs create and step simulations. Standardized protocols are replacing custom integrations.

2. **Agent-driven code-as-policy is proven.** AOR (Act-Observe-Rewrite, March 2026) achieves 100% success on robosuite tasks with zero demonstrations. Eureka/DrEureka automate reward functions and sim-to-real. GenSim generates entire simulation tasks.

3. **LeRobot is the dominant community platform.** ~22,700 GitHub stars, v0.5.0 at ICLR 2026. Supports 10+ hardware platforms including Unitree G1. Any new robotics tool must integrate with LeRobot or risk irrelevance.

4. **AWS entered with Strands Labs Robots/Robots Sim.** First major cloud provider building an agentic AI framework for physical robots.

5. **Open-source robot hardware explosion.** SO100 arms at $100, Unitree Go2 at $1,600 — massive new user base needing affordable testing tools.

**The window is 12–18 months.** Testing infrastructure for AI-driven robotics is urgently needed and currently unoccupied.

---

## Part 2: Key decisions from this review

### Decision 1: Single repo, not a product matrix

We considered splitting into multiple repos (roboharness, robosentry, roboguard, robopercept). **Decision: stay with one repo.**

Rationale:
- Evaluate (constraint verification) and regression testing are natural downstream consumers of the same checkpoint data (PNG + JSON). Splitting them means users pip-install three packages for one workflow. In robotics, tolerance for dependency friction is extremely low.
- LeRobot succeeds as a single repo covering training, evaluation, datasets, and hardware — 22K+ stars. Fragmented small tools get fragmented small audiences.
- The project lead is one person. Maintaining multiple repos is unsustainable and dilutes focus.
- Time-to-value matters in a window period. Concentrating effort on one repo is more effective than spreading across four.

What stays out: perception pipeline testing (robopercept) is genuinely different — different inputs (point clouds, detection boxes, ROS bags), different users, different dependency stack. If we do it later, it should be a separate repo. But not now.

### Decision 2: Interaction scaling is the priority dimension

The Harness Engineering article (手工川, April 2026) identifies three scaling dimensions: time (longer agent runs), space (parallel trials), interaction (human management).

**For roboharness, interaction scaling is the primary focus.** The bottleneck in AI-agent-driven robotics development is not running more simulations — MuJoCo MJX and Isaac Lab already parallelize at 30,000+ FPS. The bottleneck is enabling humans to efficiently review, annotate, and direct what AI agents produce. Roboharness should be the "eyes and hands" for engineers supervising AI agents writing robot code.

This means: the Constraint Evaluator (P1) and visual reporting are higher priority than parallel trial execution.

### Decision 3: SKILL.md before MCP

We discussed two "meta" distribution models: MCP server (AI agent calls roboharness as a service) vs. SKILL.md (AI agent reads instructions on how to use roboharness as a library).

**Decision: SKILL.md first, MCP later.**

Rationale:
- Roboharness's core operation is "add a few lines of Python to your simulation script" — pause, capture, resume. This is in-process work, not a request-response service. Claude Code can just `import roboharness` directly; MCP adds unnecessary indirection for this use case.
- mujoco-mcp already handles "control simulation via MCP." Roboharness's value is observation and judgment, which fits better as a library embedded in the simulation loop.
- SKILL.md costs almost nothing to write and delivers immediate value. MCP requires designing a tool interface, running a server process, and handling state management.

**Exception where MCP makes sense (future P3/P4):** If roboharness serves as an "evaluation plugin" alongside mujoco-mcp — mujoco-mcp controls the sim, roboharness-mcp captures and judges results — that creates an end-to-end AI agent loop. But this is a later optimization, not the current priority.

### Decision 4: Don't add a third abstraction layer

Roboharness already has two integration paths:
1. **Gymnasium wrapper** — for any `render_mode="rgb_array"` environment (covers LeRobot, ManiSkill, many MuJoCo tasks)
2. **SimulatorBackend protocol** — for custom simulation loops (covers GR00T WBC, raw MuJoCo scripts)

These two layers cover the vast majority of use cases. Adding a third abstraction "to solve all integration problems" would increase the conceptual overhead for new users without proportional benefit. Robotics developers have low patience for learning new abstractions — they want to see results fast.

### Decision 5: User acquisition is the biggest bottleneck, not features

The most important gap is not missing functionality — it's missing users. Before adding major new features, the priority should be:
1. Validate that roboharness works with popular community projects (LeRobot G1)
2. Make the value proposition visible (SKILL.md, blog post, upstream PRs)
3. Reduce friction to first use (5-minute quickstart with compelling visual output)

This doesn't mean feature work stops — it means feature work should be evaluated by its adoption impact, not its technical elegance.

---

## Part 3: Action items

### Immediate (this week)

**Action 1: Validate RobotHarnessWrapper on LeRobot G1 MuJoCo simulation.**
- Create a GitHub issue for this
- Goal: wrap LeRobot's G1 MuJoCo env with `RobotHarnessWrapper`, capture checkpoints, produce visual output
- This is a validation task, not a feature task — we want to know if it works or what breaks
- Can be delegated to Claude Code / Codex
- Success: a working `examples/demos/g1/lerobot_locomotion.py` with screenshots; or a concrete list of incompatibilities to fix
- References:
  - LeRobot G1 docs: https://huggingface.co/docs/lerobot/unitree_g1
  - LeRobot repo: https://github.com/huggingface/lerobot
  - Unitree G1 LeRobot: https://github.com/unitreerobotics/unitree_IL_lerobot

**Action 2: Write a SKILL.md for AI coding agents.**
- Location: project root or `skills/roboharness/SKILL.md`
- Content: tell Claude Code / Codex how to use roboharness when debugging robot simulation
- Should include: when to use it, pip install command, minimal code snippet for Gymnasium wrapper and for raw MuJoCo, how to interpret output
- Keep it under 60 lines (per ETH Zurich research: auto-generated long AGENTS.md hurts performance by 20%+, hand-written concise files work best)
- Reference: HumanLayer's recommendation in "Skill Issue: Harness Engineering for Coding Agents"

**Action 3: Re-integrate roboharness into the original GR00T WBC grasp project.**
- Non-blocking — do this in parallel with Actions 1-2
- Purpose: discover friction points between the library API and a real production workflow
- Record friction as GitHub issues
- This feeds directly into API improvements

### Short-term (2 weeks)

**Action 4: Continue P1 Constraint Evaluator implementation.**
- Spec exists at `docs/product/p1-constraint-evaluator.md`
- This is the feature that turns roboharness from "can see" to "can judge"
- Closes the agent feedback loop: code change → harness run → constraint check → pass/fail verdict
- After this, CI can block PRs that break robot behavior — "constraints beat instructions"

**Action 5: Find one more integration target beyond LeRobot.**
- Top candidate: `unitree_mujoco` (Unitree's official MuJoCo simulation, many G1 developers start here)
- This one doesn't use Gymnasium, so it tests the SimulatorBackend path
- Lower priority than LeRobot G1 — only start after Action 1 succeeds or produces useful findings

### Medium-term (1–2 months)

**Action 6: English technical blog post (issue #10).**
- Not a "what is roboharness" marketing piece — a "how I use AI agents to debug robot simulation" technical walkthrough
- Include real before/after: what the agent saw, what it changed, what improved
- Target audience: robotics developers who use Claude Code / Codex / Cursor
- Publish on personal blog, cross-post to relevant communities

**Action 7: Upstream contribution to LeRobot or unitree_mujoco (issue #11).**
- If the LeRobot G1 integration works, submit it as an example or recipe
- This is the highest-leverage adoption driver: appearing in a 22K-star repo

**Action 8: MCP tool interface (if validated by usage patterns).**
- Only after we have real usage data on how agents interact with roboharness
- Likely tools: `capture_checkpoint`, `evaluate_constraints`, `compare_baselines`
- Depends on mujoco-mcp ecosystem maturity

---

## Part 4: What we are NOT doing (and why)

### Not building robosentry/roboguard/robopercept as separate repos
Regression testing and constraint evaluation belong inside roboharness as modules. Perception testing is a different domain — defer indefinitely until roboharness has traction.

### Not building an MCP server right now
The core value is an in-process library, not a service. MCP comes later when we understand how agents actually use roboharness in practice.

### Not adding more abstraction layers
Two integration paths (Gymnasium wrapper + SimulatorBackend protocol) are sufficient. More abstraction = more concepts to learn = higher barrier to adoption.

### Not optimizing for parallel trial execution (space scaling)
This is genuinely useful but depends on the Constraint Evaluator being done first (you need structured pass/fail to aggregate parallel results). Sequence matters.

### Not writing a Chinese-language blog post first
English-first for international reach. Chinese content (Bilibili, Zhihu, WeChat) should be parallel but separate — different framing, not translation. International credibility matters more for the stated influence goal.

---

## Part 5: How to use this document

### For the project lead
- Review this document before each planning session
- Update the "Action items" section as items are completed or priorities shift
- If a new direction is being considered, check it against the two project goals at the top

### For Claude Code / Codex working on issues
- Check this document for context on why a feature exists and what it's trying to achieve
- The "What we are NOT doing" section is as important as the action items — it prevents scope creep
- If an issue seems to conflict with this document, flag it rather than proceeding silently

### For future strategic reviews
- Schedule a review when: a major feature ships, a new model generation releases, or the competitive landscape shifts significantly
- Each review should re-evaluate: Are we still solving the right problem? Has the window changed? What did we learn from recent integrations?
- Update this document in-place — it should always reflect current thinking, not accumulate historical layers

---

## Part 6: Reference materials

### Primary sources discussed in this review

- **Harness Engineering article**: 手工川, "Harness Engineering 的四大问题" (April 3, 2026) — three scaling dimensions (time, space, interaction) and four consensus principles
  - OpenAI report: https://openai.com/index/harness-engineering/
  - Cursor report: https://cursor.com/blog/self-driving-codebases + https://cursor.com/blog/scaling-agents
  - Anthropic report: https://www.anthropic.com/engineering/harness-design-long-running-apps
  - OpenAI Symphony: https://github.com/openai/symphony

- **CaP-X** (NVIDIA/Stanford/Berkeley/CMU, March 2026): https://capgym.github.io/ — harness framework for embodied intelligence, closest conceptual competitor
- **AOR** (Act-Observe-Rewrite, March 2026): https://arxiv.org/abs/2603.04466 — multimodal coding agents as in-context policy learners, 100% success on robosuite tasks
- **LeRobot**: https://github.com/huggingface/lerobot — 22K+ stars, dominant community platform
- **LeRobot Unitree G1 docs**: https://huggingface.co/docs/lerobot/unitree_g1
- **GR00T WBC**: https://github.com/NVlabs/GR00T-WholeBodyControl — NVIDIA's humanoid WBC platform, project lead's working context
- **ros-mcp-server**: https://github.com/robotmcp/ros-mcp-server — MCP for ROS robots (~1,100 stars)
- **mujoco-mcp**: MuJoCo simulation control via MCP
- **robotics-agent-skills**: https://github.com/arpitg1304/robotics-agent-skills — SKILL.md files for ROS coding agents

### Harness engineering general references

- Martin Fowler: https://martinfowler.com/articles/exploring-gen-ai/harness-engineering.html
- LangChain "Anatomy of an Agent Harness": https://blog.langchain.com/the-anatomy-of-an-agent-harness/
- LangChain "Improving Deep Agents with Harness Engineering": https://blog.langchain.com/improving-deep-agents-with-harness-engineering/
- HumanLayer "Skill Issue": https://www.humanlayer.dev/blog/skill-issue-harness-engineering-for-coding-agents
- ETH Zurich research: LLM-generated AGENTS.md hurts performance ~20%+; hand-written <60 lines recommended

### Unitree ecosystem

- Unitree GitHub org: https://github.com/unitreerobotics
- unitree_mujoco: MuJoCo simulation with sim-to-real
- unitree_sim_isaaclab: Isaac Lab simulation for G1/H1
- unitree_IL_lerobot: LeRobot integration for G1
- awesome-unitree-robots: https://github.com/shaoxiang/awesome-unitree-robots

### Chinese robotics open-source influence patterns

- **Unitree model**: affordable hardware + comprehensive GitHub SDK + LeRobot integration + Isaac Lab presence
- **AgiBot model**: massive dataset releases (1M+ trajectories) on HuggingFace Hub in LeRobot format, IROS Best Paper Finalist
- **Galbot model**: academic papers with open code (DexGraspNet, CoRL 2024) → NVIDIA Technical Blog feature
- **Key pattern**: English-first GitHub, HuggingFace Hub hosting, conference papers with open code, parallel Chinese community content (Bilibili/Zhihu/WeChat)

---

## Appendix: Key principles from Harness Engineering mapped to roboharness

These four consensus principles (from the article) should guide all feature decisions:

| Principle | What it means for roboharness |
|-----------|------------------------------|
| **Design the environment, not the code** | Our core thesis. The harness defines checkpoints, cameras, output format — agents write control code, the harness provides the feedback loop. |
| **Knowledge must be versioned and in the repo** | CLAUDE.md, constraint YAML, structured docs — all in-repo. Task success criteria should never live only in someone's head. |
| **Constraints beat instructions** | CI task success gates > "remember to check if the grasp works." The P1 Evaluator is the direct application of this principle. |
| **Fixing errors is cheaper than preventing them** | Accept that agent-generated code won't be perfect. Merge, then clean up. The harness exists to catch regressions, not to prevent all errors upfront. |

Additional insight worth remembering:

> "Each harness component is a hypothesis about the current model's capability boundary. These hypotheses have different expiration speeds. The evolution direction of a harness is not getting thicker, but getting thinner as models improve."

This is codified in `docs/development/component-lifecycle-guide.md`. When a new model generation releases, the first thing to do is remove components and test if quality drops — not add more layers.
