# Anthropic Claude for Open Source — Application Materials

_Prepared: April 2026_
_Deadline: June 30, 2026_
_Application URL: https://claude.com/contact-sales/claude-for-oss_

---

## Program Overview

Anthropic's **Claude for Open Source** program grants qualifying open-source maintainers
**six months of Claude Max 20x** (valued at $1,200) to use for development.

### Which track applies to roboharness

| Track | Requirement | roboharness |
|---|---|---|
| **Maintainer Track** | 5,000+ GitHub stars OR 1M+ monthly NPM downloads | Below threshold (early-stage project) |
| **Ecosystem Impact Track** | Written explanation (≤500 words) of project's ecosystem importance | ✓ **Use this track** |

---

## Eligibility Checklist

- [x] Primary maintainer of a public GitHub repository — https://github.com/MiaoDX/roboharness
- [x] Merge access to the repository
- [x] Active commits within the last 3 months — 50+ commits since January 2026
- [x] Natural person, age 18+
- [x] Legal resident in a country where Claude.ai is accessible
- [x] Not subject to U.S. export controls or sanctions
- [x] GitHub account in good standing
- [x] Not an Anthropic employee or contractor

---

## Application Content

### Field: GitHub Repository URL

```
https://github.com/MiaoDX/roboharness
```

### Field: Ecosystem Importance Statement (≤500 words)

> Copy-paste this into the application form's ecosystem explanation field.

---

roboharness is the first visual testing harness built specifically for AI coding agents
working in robot simulation. It fills a gap that no other tool addresses: when an AI
coding agent (Claude Code, Codex) writes robot control logic, it cannot directly observe
whether the robot is behaving correctly. Unlike web or API development — where an agent
can inspect HTML output or JSON responses — robot behavior is visual and physical.
Without a structured visual feedback channel, the agent is coding blind.

roboharness solves this by capturing multi-view screenshots, structured state JSON
(joint positions, end-effector poses, object positions), and temporal sequences at
user-defined checkpoints during robot simulation. This output is structured precisely
for LLM consumption: compact enough to fit in context, information-dense enough to
diagnose failures without a human in the loop. A single harness call gives Claude Code
everything it needs to see whether a grasp succeeded, a locomotion policy stabilized,
or a planned trajectory was executed correctly.

The project's ecosystem importance is concrete:

**1. Foundational infrastructure for AI-assisted robotics development.**
As AI coding agents become primary authors of robot control code, the tools that give
them perception become critical infrastructure — the equivalent of logging and
assertions in traditional software. roboharness is the only open-source project in this
role. The position is currently unoccupied; if left unfilled, each robotics team will
build bespoke, incompatible versions of the same primitive.

**2. Direct integrations with four major robotics ecosystems.**
roboharness ships with production-validated integrations for MuJoCo (the dominant
physics simulator for research), Gymnasium (the standard RL environment API),
Isaac Lab (NVIDIA's GPU-accelerated sim), and LeRobot (HuggingFace's robotics
learning library). These integrations are drop-in: three lines of code wrap any
existing environment.

**3. Proof-of-concept for Claude Code as a robotics development tool.**
roboharness has 96%+ of its commits authored by Claude Code. The project is itself
a live demonstration that Claude Code can own a complex robotics software project
end-to-end — architecture, implementation, CI, documentation, and community outreach.
This makes roboharness a natural candidate for Anthropic's portfolio: it is both
infrastructure that enables Claude Code in robotics, and a flagship example of what
Claude Code can build.

**4. Active development and growing ecosystem.**
50+ commits in the last 3 months. Multi-framework demo suite (MuJoCo grasp, G1
humanoid WBC reach, G1 locomotion, LeRobot native integration, SONIC motion tracking).
MCP server for Claude Code integration. HuggingFace Space for live demo reports.

Claude Max credits would directly accelerate the project's core mission: building the
evaluation and testing infrastructure that makes AI-assisted robotics development
reliable. Specifically, credits would fund the VLM judge integration (using Claude's
vision to automatically score checkpoint screenshots) — a capability that closes the
perception loop for AI coding agents in robotics.

---

### Supporting URLs to include

- **Repository:** https://github.com/MiaoDX/roboharness
- **Live demo reports:** https://miaodx.com/roboharness/
- **MuJoCo grasp demo:** https://miaodx.com/roboharness/grasp/
- **PyPI:** https://pypi.org/project/roboharness/

---

## Submission Instructions

1. Go to https://claude.com/contact-sales/claude-for-oss
2. Select the **Ecosystem Impact Track** (not the maintainer star-count track)
3. Fill in:
   - Repository URL: `https://github.com/MiaoDX/roboharness`
   - Ecosystem explanation: copy-paste the 500-word statement above
   - Supporting URLs: see list above
4. Submit before **June 30, 2026**

---

## Notes

- The program is capped at 10,000 recipients; apply immediately, don't wait for star growth
- The "critical infrastructure exception" framing in the ecosystem explanation is deliberate —
  roboharness qualifies under infrastructure that the AI-robotics ecosystem will depend on,
  not just an end-user tool
- If accepted, credits should be directed toward VLM judge development (issue #140)
  and CI/CD pipeline for GPU-accelerated demos
