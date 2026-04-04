"""CLI tools for inspecting harness output, generating reports, and evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from roboharness.evaluate.assertions import AssertionEngine
from roboharness.evaluate.batch import (
    check_success_rate,
    evaluate_batch,
    evaluate_batch_with_comparison,
    format_batch_human,
    format_comparison_human,
)
from roboharness.evaluate.constraints import load_constraints
from roboharness.evaluate.defaults import GRASP_DEFAULTS


def _find_metadata_files(output_dir: Path) -> list[Path]:
    """Find all metadata.json files under the output directory."""
    return sorted(output_dir.rglob("metadata.json"))


def _find_result_files(output_dir: Path) -> list[Path]:
    """Find all result.json files under the output directory."""
    return sorted(output_dir.rglob("result.json"))


def _find_image_files(directory: Path) -> list[Path]:
    """Find all image files (PNG) in a directory."""
    return sorted(directory.glob("*_rgb.png"))


def _load_json(path: Path) -> Any:
    """Load a JSON file."""
    with path.open() as f:
        return json.load(f)


def _format_state_summary(state: dict[str, Any], max_items: int = 5) -> str:
    """Format a compact summary of simulator state."""
    parts = []
    for key, value in list(state.items())[:max_items]:
        if isinstance(value, list):
            parts.append(f"{key}: [{len(value)} values]")
        elif isinstance(value, (int, float)):
            parts.append(f"{key}: {value:.4g}" if isinstance(value, float) else f"{key}: {value}")
        else:
            parts.append(f"{key}: {value}")
    if len(state) > max_items:
        parts.append(f"... +{len(state) - max_items} more")
    return ", ".join(parts)


def inspect_command(output_dir: Path) -> str:
    """Browse screenshots and states from harness output.

    Returns a formatted string summarizing the output directory contents.
    """
    if not output_dir.exists():
        return f"Error: directory not found: {output_dir}"

    metadata_files = _find_metadata_files(output_dir)
    if not metadata_files:
        return f"No captures found in {output_dir}"

    lines: list[str] = []
    lines.append(f"Harness output: {output_dir}")
    lines.append("")

    # Group by relative path structure
    current_task = ""
    current_trial = ""

    for meta_path in metadata_files:
        meta = _load_json(meta_path)
        checkpoint_dir = meta_path.parent
        rel = checkpoint_dir.relative_to(output_dir)
        parts = rel.parts

        # Detect task/trial/checkpoint from path
        # Patterns: task/trial/checkpoint OR task/variant/trial/checkpoint
        task = parts[0] if len(parts) >= 1 else "unknown"
        if task != current_task:
            current_task = task
            current_trial = ""
            lines.append(f"Task: {task}")

        if len(parts) > 2:
            trial_label = "/".join(parts[1:-1])
        elif len(parts) > 1:
            trial_label = parts[1]
        else:
            trial_label = ""
        if trial_label != current_trial:
            current_trial = trial_label
            lines.append(f"  {trial_label}/")

        checkpoint_name = meta.get("checkpoint", parts[-1] if parts else "unknown")
        step = meta.get("step", "?")
        sim_time = meta.get("sim_time", "?")
        sim_time_str = f"{sim_time:.3f}s" if isinstance(sim_time, (int, float)) else str(sim_time)
        lines.append(f"    {checkpoint_name}  (step={step}, t={sim_time_str})")

        # List images
        images = _find_image_files(checkpoint_dir)
        if images:
            img_names = [img.name for img in images]
            lines.append(f"      images: {', '.join(img_names)}")

        # State summary
        state_path = checkpoint_dir / "state.json"
        if state_path.exists():
            state = _load_json(state_path)
            lines.append(f"      state:  {_format_state_summary(state)}")

    lines.append("")
    total = len(metadata_files)
    lines.append(f"Total: {total} capture{'s' if total != 1 else ''}")
    return "\n".join(lines)


def report_command(output_dir: Path) -> dict[str, Any]:
    """Generate a summary report from harness output.

    Collects all trial results and capture metadata, then writes report.json
    to the output directory root. Returns the report dict.
    """
    if not output_dir.exists():
        raise FileNotFoundError(f"directory not found: {output_dir}")

    result_files = _find_result_files(output_dir)
    metadata_files = _find_metadata_files(output_dir)

    # Collect per-task info
    tasks: dict[str, dict[str, Any]] = {}

    for meta_path in metadata_files:
        meta = _load_json(meta_path)
        checkpoint_dir = meta_path.parent
        rel = checkpoint_dir.relative_to(output_dir)
        task = rel.parts[0] if rel.parts else "unknown"

        if task not in tasks:
            tasks[task] = {
                "trials": {},
                "total_captures": 0,
                "checkpoints": [],
            }

        tasks[task]["total_captures"] += 1
        cp = meta.get("checkpoint", "unknown")
        if cp not in tasks[task]["checkpoints"]:
            tasks[task]["checkpoints"].append(cp)

        # Find trial part
        trial_part = None
        for p in rel.parts:
            if p.startswith("trial_"):
                trial_part = p
                break
        if trial_part and trial_part not in tasks[task]["trials"]:
            tasks[task]["trials"][trial_part] = {"checkpoints_captured": []}
        if trial_part:
            tasks[task]["trials"][trial_part]["checkpoints_captured"].append(cp)

    # Merge trial results
    for result_path in result_files:
        result = _load_json(result_path)
        rel = result_path.parent.relative_to(output_dir)
        task = rel.parts[0] if rel.parts else "unknown"

        if task not in tasks:
            tasks[task] = {"trials": {}, "total_captures": 0, "checkpoints": []}

        trial_part = None
        for p in rel.parts:
            if p.startswith("trial_"):
                trial_part = p
                break

        if trial_part:
            if trial_part not in tasks[task]["trials"]:
                tasks[task]["trials"][trial_part] = {"checkpoints_captured": []}
            tasks[task]["trials"][trial_part]["result"] = {
                "success": result.get("success"),
                "reason": result.get("reason", ""),
                "duration": result.get("duration", 0),
                "metrics": result.get("metrics", {}),
            }

    # Build summary
    report: dict[str, Any] = {"tasks": {}}
    for task_name, task_data in tasks.items():
        trials = task_data["trials"]
        results_with_outcome = [t for t in trials.values() if "result" in t]
        successes = sum(1 for t in results_with_outcome if t["result"]["success"])

        report["tasks"][task_name] = {
            "total_trials": len(trials),
            "total_captures": task_data["total_captures"],
            "checkpoints": task_data["checkpoints"],
            "trials_with_results": len(results_with_outcome),
            "successes": successes,
            "success_rate": (
                successes / len(results_with_outcome) if results_with_outcome else None
            ),
            "trials": trials,
        }

    # Write report
    report_path = output_dir / "report.json"
    with report_path.open("w") as f:
        json.dump(report, f, indent=2, default=str)

    return report


def evaluate_command(
    report_path: Path,
    constraints_path: Path | None = None,
) -> tuple[dict[str, Any], int]:
    """Evaluate a single report against constraints.

    Returns ``(result_dict, exit_code)``.  Exit codes: 0=pass, 1=fail, 2=degraded.
    """
    if not report_path.exists():
        raise FileNotFoundError(f"report not found: {report_path}")

    assertions = load_constraints(constraints_path) if constraints_path else GRASP_DEFAULTS
    engine = AssertionEngine(assertions)
    report = _load_json(report_path)
    result = engine.evaluate(report, report_path=str(report_path))

    exit_code = {"pass": 0, "fail": 1, "degraded": 2}[result.verdict.value]
    return result.to_dict(), exit_code


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="roboharness",
        description="CLI tools for roboharness output inspection and reporting.",
    )
    subparsers = parser.add_subparsers(dest="command")

    # inspect
    inspect_parser = subparsers.add_parser(
        "inspect",
        help="Browse screenshots and states from harness output.",
    )
    inspect_parser.add_argument(
        "output_dir",
        type=Path,
        help="Path to harness output directory.",
    )

    # report
    report_parser = subparsers.add_parser(
        "report",
        help="Generate summary report from harness output.",
    )
    report_parser.add_argument(
        "output_dir",
        type=Path,
        help="Path to harness output directory.",
    )

    # evaluate
    eval_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate a harness report against constraints.",
    )
    eval_parser.add_argument(
        "report_path",
        type=Path,
        help="Path to autonomous_report.json.",
    )
    eval_parser.add_argument(
        "--constraints",
        type=Path,
        default=None,
        help="Path to constraint definition file (YAML or JSON).",
    )
    eval_parser.add_argument(
        "--format",
        dest="output_format",
        choices=["human", "json"],
        default="human",
        help="Output format (default: human).",
    )

    # evaluate-batch
    batch_parser = subparsers.add_parser(
        "evaluate-batch",
        help="Aggregate evaluation results across multiple trials.",
    )
    batch_parser.add_argument(
        "results_dir",
        type=Path,
        help="Directory containing trial result files.",
    )
    batch_parser.add_argument(
        "--constraints",
        type=Path,
        default=None,
        help="Path to constraint definition file (YAML or JSON).",
    )
    batch_parser.add_argument(
        "--format",
        dest="output_format",
        choices=["human", "json"],
        default="human",
        help="Output format (default: human).",
    )
    batch_parser.add_argument(
        "--compare",
        action="store_true",
        default=False,
        help="Compare variants side-by-side (each subdirectory = one variant).",
    )
    batch_parser.add_argument(
        "--min-success-rate",
        type=float,
        default=None,
        help="Minimum success rate (0.0-1.0) for CI pass/fail. Exit 1 if below.",
    )

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 1

    if args.command == "inspect":
        output = inspect_command(args.output_dir)
        print(output)
        return 0

    if args.command == "report":
        try:
            report = report_command(args.output_dir)
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        print(f"Report written to {args.output_dir / 'report.json'}")
        for task_name, task_data in report.get("tasks", {}).items():
            total = task_data["total_trials"]
            captures = task_data["total_captures"]
            rate = task_data.get("success_rate")
            rate_str = f"{rate:.0%}" if rate is not None else "N/A"
            print(f"  {task_name}: {total} trials, {captures} captures, success={rate_str}")
        return 0

    if args.command == "evaluate":
        try:
            result_dict, exit_code = evaluate_command(args.report_path, args.constraints)
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        if args.output_format == "json":
            print(json.dumps(result_dict, indent=2))
        else:
            verdict = result_dict["verdict"]
            print(f"Verdict: {verdict}")
            print(f"  Assertions: {result_dict['passed']}/{result_dict['total_assertions']} passed")
            if result_dict["critical_failures"]:
                print(f"  Critical failures: {result_dict['critical_failures']}")
            if result_dict["major_failures"]:
                print(f"  Major failures: {result_dict['major_failures']}")
            for r in result_dict["results"]:
                if not r["passed"]:
                    print(f"  FAIL [{r['severity']}] {r['message']}")
        return exit_code

    if args.command == "evaluate-batch":
        if not args.results_dir.exists():
            print(f"Error: directory not found: {args.results_dir}", file=sys.stderr)
            return 1
        assertions = load_constraints(args.constraints) if args.constraints else GRASP_DEFAULTS
        if args.compare:
            comparison = evaluate_batch_with_comparison(args.results_dir, assertions)
            if args.output_format == "json":
                print(json.dumps(comparison.to_dict(), indent=2))
            else:
                print(format_comparison_human(comparison))
            exit_code = 0
            for v in comparison.variants:
                if v.batch.total_trials > 0 and v.batch.success_rate == 0.0:
                    exit_code = 1
                    break
        else:
            batch = evaluate_batch(args.results_dir, assertions)
            if args.output_format == "json":
                print(json.dumps(batch.to_dict(), indent=2))
            else:
                print(format_batch_human(batch))
            exit_code = 0
            if args.min_success_rate is not None:
                if not check_success_rate(batch, args.min_success_rate):
                    exit_code = 1
            elif batch.total_trials > 0 and batch.success_rate == 0.0:
                exit_code = 1
        return exit_code

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
