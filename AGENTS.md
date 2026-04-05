# AGENTS.md

Repository-scoped operating notes for Codex agents.

## Patch/edit workflow

- When editing files, use the dedicated `apply_patch` tool directly.
- Do **not** run `apply_patch` through shell commands (for example via `exec_command`).
- Keep commits focused and small; avoid mixing unrelated changes.

## Python test invocation in this repo

- If local test runs fail to import `roboharness`, rerun with `PYTHONPATH=src`.
- If `pytest` picks up unavailable coverage addopts from config in constrained environments, use `-o addopts=''` for focused local validation.

