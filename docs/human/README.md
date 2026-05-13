# Human Documentation Index

This index defines the small documentation surface humans should read at HEAD.
The repo contains many planning notes, design reviews, generated reports, and
strategy drafts; those are evidence, not the default review path.

## Start Here

1. `README.md` — product entrypoint, install paths, demos, examples, and links
   to live proof surfaces.
2. `STATUS.md` — current state, active milestone, runnable commands, and known
   validation boundaries.
3. `ARCHITECTURE.md` — subsystem map, data flow, public contracts, and extension
   points.

## Supporting Human Docs

- `CONTRIBUTING.md` — contribution workflow.
- `CHANGELOG.md` — release history.
- `docs/development/development-workflow.md` — development environment guidance.
- `docs/context/context.en.md` and `docs/context/context.zh-CN.md` — project background.
- `docs/product/sonic-inference-stacks.md` — SONIC planner/tracking split and
  validation policy.

## Evidence and History

The following buckets are intentionally not part of the default human review
surface:

- `docs/designs/**` — design reviews, plans, and audit artifacts.
- `docs/steering/**` — milestone steering notes used as source evidence for
  `STATUS.md`.
- `.planning/**` — agent planning state.
- `assets/**` and `harness_output/**` — proof bundles, screenshots, and
  generated run output.
- `blog/**`, `docs/community/**`, `docs/funding/**`, and `docs/academic/**` —
  outreach, strategy, and application drafts.

Promote durable conclusions from those buckets into `README.md`,
`STATUS.md`, or `ARCHITECTURE.md` when they become current project truth.
