# Harness Engineering Insights for Roboharness

_Created: 2026-04-03_
_Source: [Harness Engineering 的四大问题](https://mp.weixin.qq.com/s/9o8-jCzWDgj4GCa-gk4Zqg) (手工川, 2026-04-03)_

## Context

This document distills actionable insights from a Harness Engineering analysis article that synthesizes first-party reports from OpenAI, Cursor, and Anthropic. The article frames Harness Engineering as three independent scaling dimensions — **time**, **space**, and **interaction** — and four consensus principles. We map each to concrete work items for roboharness.

The original reports referenced:
- **OpenAI**: [Harness engineering: leveraging Codex in an agent-first world](https://openai.com/index/harness-engineering/) + [Symphony](https://github.com/openai/symphony)
- **Cursor**: [Towards self-driving codebases](https://cursor.com/blog/self-driving-codebases) + [Scaling long-running autonomous coding](https://cursor.com/blog/scaling-agents)
- **Anthropic**: [Harness design for long-running application development](https://www.anthropic.com/engineering/harness-design-long-running-apps)

---

## Four consensus principles and where roboharness stands

### 1. Design the environment, not the code

> "The highest-leverage work is designing the agent's working environment — not writing prompts."

**Roboharness alignment**: This is our core thesis. The harness defines checkpoints, cameras, output format — creating a structured environment where agents observe, judge, and iterate on robot behavior. Agents write control code; the harness provides the feedback loop.

**Gap**: We don't yet provide guidance on *how* to design a harness config for a new task. A "harness design guide" would lower the barrier for new users.

### 2. Knowledge must be versioned, discoverable, and in the repo

> "What the agent can't see doesn't exist."

**Roboharness alignment**: CLAUDE.md, structured docs, and agent-consumable output (PNG + JSON) are already in-repo. Constraint definitions (P1 evaluator) will also live in-repo as YAML.

**Gap**: Task-specific knowledge (e.g., "a successful grasp means the object is lifted > 5mm") currently lives in the user's head. The constraint evaluator (P1) will make this explicit and version-controlled.

### 3. Constraints beat instructions

> "Telling an agent 'remember to write tests' doesn't work. A CI coverage gate does."

**Roboharness alignment**: We enforce code quality via ruff + mypy + pytest CI gates. The SimulatorBackend protocol is a structural constraint (you can't skip required methods).

**Gap**: We don't yet have **task-level constraints** in CI. The MuJoCo grasp example runs and uploads screenshots, but doesn't assert success. An agent could break grasp behavior and CI would still pass. This is our most important gap.

### 4. Fixing errors is cheaper than preventing them

> "Waiting for perfection kills throughput. Merge, then clean up."

**Roboharness alignment**: Our PR review strategy already says "push fixes to the same branch, don't create new PRs." But we could go further with automated quality sweeps.

**Gap**: No automated entropy detection — when agent modifications degrade task success over time, nothing flags it.

---

## Three scaling dimensions mapped to roboharness

### Dimension 1: Time Scaling — one agent running longer without degradation

**Source**: Anthropic's Planner-Generator-Evaluator architecture.

**Key insights for us**:

1. **Independent evaluator with no shared state**: The evaluator must not see the generator's reasoning, or it will be persuaded. In roboharness terms: the agent that writes control code should NOT be the same context that judges the result. The harness output (PNG + JSON) is the clean boundary — an evaluator reads only these artifacts.

2. **Harness components are hypotheses about model limitations, with different expiration speeds**: Each component we build (depth capture, intermediate checkpoints, multi-view rendering) is a bet that the model can't do something on its own. As models improve, some components become unnecessary. We should design for removal, not just addition.

3. **Planner-Generator-Evaluator maps naturally to robotics**:
   - **Planner**: decompose a manipulation task into phases (approach, contact, grasp, lift)
   - **Generator**: agent writes/modifies control code, harness runs simulation
   - **Evaluator**: independent constraint check on harness output (P1 spec)

**Actionable**:
- **P1 Constraint Evaluator** (spec exists: `docs/p1-constraint-evaluator.md`) — the most direct application of this dimension. Makes the harness a closed-loop verification system, not just an observation tool.
- **Component lifecycle metadata** — annotate harness components with the assumption they encode and conditions under which they can be removed.

### Dimension 2: Space Scaling — many agents/trials running in parallel

**Source**: Cursor's recursive Planner-Worker architecture for hundreds of parallel agents.

**Key insights for us**:

1. **Execution isolation is non-negotiable**: Cursor found that shared-state coordination collapses under load. Each worker needs its own copy. For roboharness: each parallel trial needs its own simulator instance and output directory. No shared mutable state between trials.

2. **Decentralized quality control scales better than a central gate**: Cursor removed their centralized Integrator because it became a bottleneck. Accept a stable error rate; let subsequent runs fix errors. For us: don't require every trial to pass before starting the next batch. Aggregate results across trials and let statistical patterns emerge.

3. **Repo/build structure matters for parallel agents**: Cursor saw massive speedups from splitting a monolith into independent crates. For roboharness: our modular backend architecture (each backend is independent) already supports this. But we should ensure examples and test configs are self-contained enough to run in parallel without conflicts.

4. **Model selection matters — "smart" and "persistent" are different traits**: Cursor found GPT-5.2 outperformed Opus 4.5 in long autonomous runs because Opus "takes shortcuts when convenient." For robotics harness work, persistence (trying many variations) may matter more than brilliance (elegant code on the first try).

**Actionable**:
- **Parallel trial runner** — extend `GraspTaskStore` to support concurrent trial execution across multiple simulator instances. Each trial gets an isolated workspace.
- **Batch evaluation with aggregation** — the P1 evaluator should support evaluating a batch of trials and producing aggregate statistics (success rate, failure distribution by phase, etc.).

### Dimension 3: Interaction Scaling — humans managing agents without burnout

**Source**: OpenAI's Symphony — ticket-driven agent orchestration.

**Key insights for us**:

1. **Agent self-verification reduces human review burden**: OpenAI uses Chrome DevTools Protocol to let agents see what they built. Roboharness already does this for robotics (screenshots + state JSON). The P1 evaluator completes this by providing a deterministic verdict — humans review exceptions, not every trial.

2. **Mechanical constraints replace human review**: Custom linters enforce architectural invariants even at 3am. For roboharness: CI task success gates mean a PR that breaks grasp behavior gets blocked automatically, without human review.

3. **Automated entropy management**: OpenAI runs background agents to detect and fix codebase drift. For roboharness: track task success rate over time. If a code change causes regression across the benchmark suite, flag it automatically.

4. **The interaction model becomes: humans write tickets and maintain the harness; agents do the rest**: This is the end state. For roboharness users: write a task spec (constraint YAML + checkpoint config), let agents iterate on control code until the evaluator passes. The human's job is improving the harness, not reviewing every trial.

**Actionable**:
- **CI task success gate** — `mujoco_grasp.py --assert-success` that checks physical state against constraints and fails CI if the task didn't succeed. Small change, high leverage.
- **Success rate trend tracking** — store evaluation results over time, detect regressions across commits.
- **Issue-driven agent workflow** — document how to use roboharness in a ticket-driven loop (write issue → agent picks up → runs harness → evaluator judges → PR with proof-of-work).

---

## Implementation priority

| Priority | Item | Source dimension | Tracks to |
|----------|------|-----------------|-----------|
| **P0** | CI task success gate (`--assert-success`) | Interaction | #48 |
| **P1** | Constraint Evaluator | Time | `docs/p1-constraint-evaluator.md` |
| **P2** | Parallel trial execution | Space | #49 |
| **P3** | Component lifecycle metadata | Time | #50 |
| **P4** | Batch evaluation + aggregation | Space | #51 |
| **P5** | Success rate trend tracking | Interaction | #52 |

### Why this order

1. **CI task success gate** is the smallest change with the biggest impact — it turns CI from "did it crash?" into "did it work?" This is the "constraints beat instructions" principle applied directly.
2. **Constraint Evaluator** is the foundation for everything else — parallel trials and trend tracking both need a structured pass/fail verdict.
3. **Parallel trials** unlock space scaling but depend on the evaluator for aggregation.
4. **Component lifecycle** and **trend tracking** are meta-improvements that pay off over longer timeframes.

---

## Key quotes from the article worth remembering

> "Each harness component is essentially a hypothesis about the current model's capability boundary. These hypotheses have different expiration speeds."

> "The evolution direction of a harness is not getting thicker, but getting thinner as models improve. Every time a new model releases, the first thing you should do is remove some components and see if things break."

> "When you find an agent exhibiting bizarre behavior, first check whether you've stuffed too many conflicting objectives into it. Humans slack off in this situation; agents glitch out."

> "Whether a model is smart and whether a model is willing to grind are two different things."

> "Next time someone says 'harness engineering' to you, ask: which dimension of scaling are you solving? If they can't answer, they're probably still talking about reins and horses."

---

## Related

- `docs/p1-constraint-evaluator.md` — P1 spec (evaluator design)
- `docs/roadmap.md` — project roadmap
- `drafts/harness-engineering-article.md` (in claw-agents-shared) — full article text
