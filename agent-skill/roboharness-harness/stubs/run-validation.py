"""Run validation commands from contract.snapshot.json."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _find_repo_root(start: Path) -> Path:
    for path in (start, *start.parents):
        if (path / "pyproject.toml").exists():
            return path
    return start


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workflow", default=None)
    args = parser.parse_args(argv)

    skill_dir = Path(__file__).resolve().parents[1]
    snapshot = json.loads((skill_dir / "contract.snapshot.json").read_text())
    commands_by_id = {command["id"]: command for command in snapshot["validation_commands"]}
    command_ids: list[str] = []
    if args.workflow is None:
        command_ids = list(commands_by_id)
    else:
        workflows = {workflow["id"]: workflow for workflow in snapshot["workflows"]}
        if args.workflow not in workflows:
            print(f"Unknown workflow: {args.workflow}", file=sys.stderr)
            return 2
        command_ids = list(workflows[args.workflow]["validation_commands"])

    repo_root = _find_repo_root(skill_dir)
    for command_id in command_ids:
        command = commands_by_id[command_id]["command"]
        print(f"$ {command}")
        completed = subprocess.run(command, cwd=repo_root, shell=True)
        if completed.returncode != 0:
            return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
