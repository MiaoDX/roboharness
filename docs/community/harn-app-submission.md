# harn.app Submission — roboharness

## What is harn.app?

[harn.app](https://harn.app) is a knowledge base for harness engineering — infrastructure built around AI agents. It is auto-generated from the curated GitHub list [`walkinglabs/awesome-harness-engineering`](https://github.com/walkinglabs/awesome-harness-engineering).

**Submission path:** Submit a PR to `walkinglabs/awesome-harness-engineering`. The harn.app website will pick up new entries automatically on the next build.

## Investigation results (April 2026)

- harn.app website: 8 categories, "Tools & Runtimes (9 entries)" and "Evals & Observability (11 entries)" are the most relevant for roboharness
- awesome-harness-engineering README: "Runtimes, Harnesses & Reference Implementations" section lists 10 projects (SWE-agent, Harbor, deepagents, Claude Agent SDK, AgentKit, etc.)
- **No robotics entries exist** in the current list — roboharness would be the first

## Entry text (ready to copy-paste)

Add to the **"Runtimes, Harnesses & Reference Implementations"** section in the README:

```markdown
* [roboharness](https://github.com/MiaoDX/roboharness) - Visual testing harness for AI coding agents in robot simulation. The only robotics-specific harness engineering tool: captures multi-view screenshots and structured state JSON at checkpoints, letting AI coding agents (Claude Code, Codex) see and judge robot behavior directly. Supports MuJoCo, Gymnasium, Isaac Lab, and LeRobot via a lightweight protocol.
```

## PR instructions

1. Fork https://github.com/walkinglabs/awesome-harness-engineering
2. Add the entry above to `README.md` under **"Runtimes, Harnesses & Reference Implementations"**, after existing entries
3. Submit PR titled:

   ```
   Add roboharness — visual testing harness for AI coding agents in robot simulation
   ```

4. PR body suggestion:

   ```markdown
   ## What is roboharness?

   roboharness is a visual testing harness for AI coding agents in robot simulation.
   It's the only harness engineering tool targeting robotics specifically.

   Key capabilities:
   - Captures multi-view screenshots and structured state JSON at checkpoints
   - Lets AI coding agents (Claude Code, Codex) see and iterate on robot behavior
   - Supports MuJoCo, Gymnasium, Isaac Lab, and LeRobot
   - Drop-in `RobotHarnessWrapper` (3-line integration)

   GitHub: https://github.com/MiaoDX/roboharness
   Live demo: https://miaodx.com/roboharness/grasp/
   ```

## harn.app category context

On harn.app, this entry would appear under **"Tools & Runtimes"**, which maps to the "Runtimes, Harnesses & Reference Implementations" section in the GitHub source. The emphasis should be:
- It is infrastructure *for* AI coding agents (not a framework *of* agents)
- Robotics domain — distinct from all existing entries which target software coding tasks
- Structured output (JSON + screenshots) that agents can parse programmatically
