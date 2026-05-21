# Research checkpoints

This directory holds periodic ecosystem-research checkpoints for `roboharness` —
structured snapshots of the surrounding stack (harness engineering upstream,
robot simulators, multimodal models, VLAs, humanoid hardware, MCP / agent
skills) on a monthly cadence.

Each file is named `YYYY-MM.md` and follows the Layer 1 universal template from
[`MiaoDX/mithaq`](https://github.com/MiaoDX/mithaq). The research directions,
hidden assumptions, source whitelist/blacklist, and cadence are defined in the
Layer 2 vectors card at
[`mithaq/vectors/roboharness.md`](https://github.com/MiaoDX/mithaq/blob/main/vectors/roboharness.md).

## How to read

- For the latest understanding, read the most recent `YYYY-MM.md` — start with
  §1 (executive summary) and §6 (recommendations).
- To trace how a judgment evolved, scan §8 (changelog) entries across multiple
  checkpoints. Prior judgments are not deleted; revisions are annotated with
  `[revised YYYY-MM]` blocks below them.
- Open questions accumulate in §7 across checkpoints; the next checkpoint
  starts by answering or carrying them forward.

## How to produce a new checkpoint

Follow Mode A in
[`mithaq/skills/mithaq/SKILL.md`](https://github.com/MiaoDX/mithaq/blob/main/skills/mithaq/SKILL.md):

1. Read this repo's previous checkpoint (the comparison anchor).
2. Read `mithaq/vectors/roboharness.md` for the directions to cover.
3. Read `mithaq/templates/checkpoint.md` for the skeleton.
4. Run deep research per vector; verify each hidden assumption H1..HN.
5. Write `YYYY-MM.md` here. Checkpoint *instances* live in this repo; the
   template and vectors card stay in mithaq.

## Status

- 2026-05: baseline established. First checkpoint produced through the mithaq
  Mode A workflow.
