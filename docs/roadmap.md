# Roboharness Public Roadmap

_Last updated: April 3, 2026 (UTC)_

This roadmap is public by design. It is intended for maintainers, contributors, and community users who want clear visibility into what is done, what is next, and why.

## Why this roadmap exists

Roboharness has two goals:

1. **Primary goal (product):** build a reliable, reusable robot testing harness for the community.
2. **Secondary goal (ecosystem):** grow adoption and project visibility (including GitHub stars) as an outcome of real usefulness.

We use an issue-driven workflow, but roadmap planning needs structure beyond a raw issue list. This document aligns our planning with open-source roadmap best practices: clear milestones, measurable outcomes, transparent status, and explicit ownership of follow-up.

---

## Current status snapshot (GitHub Issues)

### Completed (closed issues)

- #2 — MuJoCo + Meshcat end-to-end example
- #3 — PyPI publishing setup
- #6 — Rerun integration
- #7 — GitHub Actions CI
- #8 — Multi-camera support enhancement
- #9 — CLI tools

### In progress / open

- #4 — Isaac Lab Gymnasium wrapper validation
- #5 — ManiSkill Gymnasium wrapper validation
- #10 — English blog post
- #11 — Upstream example contributions
- #12 — Academic citation/collaboration
- #18 — GPU CI strategy (cuRobo / policy / WBC)
- #34 — Phased GR00T WBC integration

---

## Roadmap principles (best-practice aligned)

1. **Outcome over output:** each milestone must define success criteria, not just tasks.
2. **One source of truth:** roadmap milestones map to issues and PRs.
3. **Short feedback loops:** weekly triage and status updates.
4. **Public by default:** decisions, scope changes, and progress are visible in this file and linked issues.
5. **Stability first:** reliability and reproducibility are prioritized before expansion.

---

## 2026 plan: dual-track execution

## Track A — Product & Reliability (Community Harness)

### Milestone A1 (1–2 weeks): Make core usage robust

- Close integration-validation gaps for #4 and #5 with:
  - reproducible docs,
  - minimal runnable demos,
  - optional CI smoke checks where feasible.
- Clean issue hygiene:
  - close already-delivered issues,
  - split oversized issues into mergeable sub-issues.
- Strengthen regression coverage for:
  - checkpoint capture,
  - multi-camera behavior,
  - CLI inspect/report paths.

**Exit criteria**
- Repro commands for Isaac Lab and ManiSkill are documented and verified.
- Main CI remains green with no regressions on core harness paths.

### Milestone A2 (2–4 weeks): Improve day-2 usability

- Upgrade report output with task-level summary metrics (success rate, retries, key checkpoint thumbnails).
- Publish a concise "build your own backend/visualizer" guide.
- Define v0.x compatibility scope for:
  - CLI flags,
  - output directory schema,
  - JSON fields.

**Exit criteria**
- New users can run examples and interpret results without maintainer intervention.
- Integration contributors can implement adapters with documented contracts.

### Milestone A3 (4–8 weeks): Scale capabilities safely

- Deliver #34 Phase 1 on a CPU-first path.
- Advance #18 with a non-blocking nightly GPU workflow.
- Prepare v0.2.0 with migration notes and changelog.

**Exit criteria**
- Advanced control path is documented and reproducible.
- GPU validation exists without slowing default CI.

---

## Track B — Ecosystem Growth (Adoption & Visibility)

### Milestone B1 (1–3 weeks): Content quality and repeatability

- Deliver #10 with a technical long-form post:
  - practical workflow,
  - demo artifacts,
  - comparisons and lessons learned.
- For each release, publish:
  - one reproducible quick command,
  - one visual before/after artifact,
  - one short agent-debug loop demo.

### Milestone B2 (2–6 weeks): Distribution via upstream leverage

- Deliver #11 with at least 1–2 upstream example/integration PRs.
- Standardize example entry points across MuJoCo/Isaac/ManiSkill for easier cross-sharing.

### Milestone B3 (continuous): Credibility accumulation

- Progress #12 (citations, collaboration, technical references).
- Build a public case list of reproducible community use cases.

---

## Operating model

- **Weekly roadmap triage:** review issue status, blockers, and milestone drift.
- **PR policy:** every PR links an issue (`Closes #xx` or `Part of #xx`) and includes acceptance notes.
- **Documentation gate:** behavior changes must update at least one of README/docs/changelog.
- **Metrics board:**
  - Product metrics: reproducible demos, test pass rate, integration coverage.
  - Ecosystem metrics: monthly star growth, external references, upstream merges.

---

## Next sprint (immediate actions)

1. Reconcile issue status against merged PRs (especially #4/#5/#34 related work).
2. Add/verify minimal validation scripts for Isaac Lab and ManiSkill paths.
3. Draft and publish backend/visualizer adapter guide.
4. Prepare #10 article outline and one upstream contribution candidate for #11.

---

## Scope note

This file is the roadmap and should remain strategic.

If we need temporary notes or ideas that do not fit an issue yet, they should be captured in issue drafts/discussions first, then promoted into tracked issues before entering this roadmap.
