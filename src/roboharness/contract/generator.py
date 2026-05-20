"""Deterministic compiler for project harness skills."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from roboharness.contract.model import (
    ApprovalPolicy,
    EvidenceBoundary,
    EvidenceReference,
    HarnessContract,
    HarnessWorkflow,
    MetricGate,
    SemanticPhase,
    ValidationCommand,
    VisualReviewDimension,
)

CONTRACT_SCHEMA_VERSION = "roboharness_harness_contract/v1"
GENERATOR_VERSION = "roboharness.contract/v1"
GENERATED_MANIFEST = ".generated-manifest.json"
SUPPORTED_OPERATORS = frozenset({"lt", "le", "eq", "gt", "ge", "in_range"})
SUPPORTED_SEVERITIES = frozenset({"fail", "warn", "info"})
_SLUG_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_ID_RE = re.compile(r"^[a-z][a-z0-9_:-]*$")


class ContractDriftError(RuntimeError):
    """Raised when generated project harness artifacts drift from contract.py."""


@dataclass(frozen=True)
class GenerationResult:
    """Files written by one generator run."""

    output_dir: Path
    files: tuple[Path, ...]
    snapshot_sha256: str


@dataclass(frozen=True)
class DriftReport:
    """Drift-check result for a generated project harness skill."""

    output_dir: Path
    missing: tuple[str, ...] = ()
    changed: tuple[str, ...] = ()
    stale: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.missing and not self.changed and not self.stale

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "output_dir": str(self.output_dir),
            "missing": list(self.missing),
            "changed": list(self.changed),
            "stale": list(self.stale),
        }


def load_contract_from_file(path: str | Path) -> HarnessContract:
    """Load a Python-authored contract from ``CONTRACT`` or ``build_contract()``."""

    contract_path = Path(path)
    spec = importlib.util.spec_from_file_location(
        f"_roboharness_contract_{hashlib.sha256(str(contract_path).encode()).hexdigest()[:12]}",
        contract_path,
    )
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load contract module: {contract_path}")
    module = importlib.util.module_from_spec(spec)
    _exec_module(spec.loader, module)

    raw_contract = getattr(module, "CONTRACT", None)
    if raw_contract is None and hasattr(module, "build_contract"):
        raw_contract = module.build_contract()
    if not isinstance(raw_contract, HarnessContract):
        raise TypeError(
            f"{contract_path} must define CONTRACT or build_contract() returning HarnessContract"
        )
    validate_contract(raw_contract)
    return raw_contract


def normalize_contract(contract: HarnessContract) -> dict[str, Any]:
    """Return the deterministic machine snapshot for a project harness contract."""

    validate_contract(contract)
    return {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "project_slug": contract.project_slug,
        "name": contract.name,
        "version": contract.version,
        "description": contract.description,
        "phases": [_phase_to_dict(phase) for phase in contract.phases],
        "metric_gates": [_metric_gate_to_dict(gate) for gate in contract.metric_gates],
        "visual_review_dimensions": [
            _visual_dimension_to_dict(dimension) for dimension in contract.visual_review_dimensions
        ],
        "evidence_boundaries": [
            _evidence_boundary_to_dict(boundary) for boundary in contract.evidence_boundaries
        ],
        "approval_policy": _approval_policy_to_dict(contract.approval_policy),
        "validation_commands": [
            _validation_command_to_dict(command) for command in contract.validation_commands
        ],
        "workflows": [_workflow_to_dict(workflow) for workflow in contract.workflows],
    }


def render_project_harness_skill(
    contract: HarnessContract,
    *,
    include_stubs: bool = True,
) -> dict[str, str]:
    """Render all generated files for a project harness skill."""

    managed = _render_managed_files(contract, include_stubs=include_stubs)
    managed[GENERATED_MANIFEST] = _render_manifest_content(contract, managed)
    return managed


def generate_project_harness_skill(
    contract: HarnessContract,
    output_dir: str | Path,
    *,
    include_stubs: bool = True,
) -> GenerationResult:
    """Write deterministic generated artifacts beside a trusted ``contract.py``."""

    target = Path(output_dir)
    files = render_project_harness_skill(contract, include_stubs=include_stubs)
    written: list[Path] = []
    for relative_path, content in files.items():
        path = target / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        written.append(path)
    snapshot_sha = _sha256_text(files["contract.snapshot.json"])
    return GenerationResult(
        output_dir=target,
        files=tuple(sorted(written)),
        snapshot_sha256=snapshot_sha,
    )


def check_project_harness_skill(
    contract: HarnessContract,
    output_dir: str | Path,
    *,
    include_stubs: bool = True,
) -> DriftReport:
    """Check whether generated artifacts still match ``contract.py``."""

    target = Path(output_dir)
    expected = render_project_harness_skill(contract, include_stubs=include_stubs)
    missing: list[str] = []
    changed: list[str] = []
    for relative_path, content in expected.items():
        path = target / relative_path
        if not path.exists():
            missing.append(relative_path)
            continue
        if path.read_text() != content:
            changed.append(relative_path)

    stale: list[str] = []
    manifest_path = target / GENERATED_MANIFEST
    if manifest_path.exists():
        try:
            actual_manifest = json.loads(manifest_path.read_text())
        except json.JSONDecodeError:
            actual_manifest = {}
        actual_files = {
            str(file_entry.get("path"))
            for file_entry in actual_manifest.get("files", [])
            if isinstance(file_entry, dict) and file_entry.get("path")
        }
        stale = sorted(path for path in actual_files - set(expected) if (target / path).exists())

    return DriftReport(
        output_dir=target,
        missing=tuple(sorted(missing)),
        changed=tuple(sorted(changed)),
        stale=tuple(stale),
    )


def validate_contract(contract: HarnessContract) -> None:
    """Validate cross-references and fail before generated text becomes authority."""

    _require_slug(contract.project_slug, "project_slug")
    _require_non_empty(contract.name, "name")
    _require_non_empty(contract.version, "version")
    if not contract.phases:
        raise ValueError("HarnessContract.phases must not be empty.")

    phase_ids = _unique_ids((phase.id for phase in contract.phases), "phase")
    gate_ids = _unique_ids((gate.id for gate in contract.metric_gates), "metric gate")
    dimension_ids = _unique_ids(
        (dimension.id for dimension in contract.visual_review_dimensions),
        "visual review dimension",
    )
    boundary_ids = _unique_ids(
        (boundary.id for boundary in contract.evidence_boundaries),
        "evidence boundary",
    )
    command_ids = _unique_ids(
        (command.id for command in contract.validation_commands),
        "validation command",
    )
    workflow_ids = _unique_ids((workflow.id for workflow in contract.workflows), "workflow")
    if not workflow_ids:
        raise ValueError("HarnessContract.workflows must not be empty.")

    for phase in contract.phases:
        _require_non_empty(phase.label, f"phase {phase.id}.label")
        _require_non_empty(phase.description, f"phase {phase.id}.description")

    for gate in contract.metric_gates:
        if gate.operator not in SUPPORTED_OPERATORS:
            raise ValueError(f"MetricGate {gate.id} uses unsupported operator {gate.operator!r}.")
        if gate.severity not in SUPPORTED_SEVERITIES:
            raise ValueError(f"MetricGate {gate.id} uses unsupported severity {gate.severity!r}.")
        if gate.phase is not None and gate.phase not in phase_ids:
            raise ValueError(f"MetricGate {gate.id} references unknown phase {gate.phase!r}.")
        for reference in gate.evidence:
            _validate_evidence_reference(reference, phase_ids, boundary_ids, owner=gate.id)

    for dimension in contract.visual_review_dimensions:
        if dimension.phase not in phase_ids:
            raise ValueError(
                f"VisualReviewDimension {dimension.id} references unknown phase "
                f"{dimension.phase!r}."
            )
        if not dimension.views:
            raise ValueError(f"VisualReviewDimension {dimension.id}.views must not be empty.")
        if (
            dimension.evidence_boundary is not None
            and dimension.evidence_boundary not in boundary_ids
        ):
            raise ValueError(
                f"VisualReviewDimension {dimension.id} references unknown evidence boundary "
                f"{dimension.evidence_boundary!r}."
            )

    for boundary in contract.evidence_boundaries:
        _require_non_empty(boundary.root, f"evidence boundary {boundary.id}.root")
        if not boundary.allowed_patterns:
            raise ValueError(f"EvidenceBoundary {boundary.id}.allowed_patterns must not be empty.")

    for command in contract.validation_commands:
        _require_non_empty(command.command, f"validation command {command.id}.command")

    for workflow in contract.workflows:
        _require_non_empty(workflow.label, f"workflow {workflow.id}.label")
        _require_non_empty(workflow.description, f"workflow {workflow.id}.description")
        _require_refs(workflow.phases, phase_ids, f"workflow {workflow.id}.phases")
        _require_refs(workflow.metric_gates, gate_ids, f"workflow {workflow.id}.metric_gates")
        _require_refs(
            workflow.visual_dimensions,
            dimension_ids,
            f"workflow {workflow.id}.visual_dimensions",
        )
        _require_refs(
            workflow.validation_commands,
            command_ids,
            f"workflow {workflow.id}.validation_commands",
        )


def _exec_module(loader: Any, module: ModuleType) -> None:
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous


def _render_managed_files(
    contract: HarnessContract,
    *,
    include_stubs: bool,
) -> dict[str, str]:
    snapshot = normalize_contract(contract)
    files = {
        "SKILL.md": _render_skill_md(snapshot),
        "contract.snapshot.json": _json_dumps(snapshot),
        "schemas/harness-contract.schema.json": _json_dumps(_contract_json_schema()),
        "schemas/generated-artifacts.schema.json": _json_dumps(_generated_artifacts_schema()),
        "scope-brief-template.md": _render_scope_brief_template(snapshot),
        "README.md": _render_generated_readme(snapshot),
    }
    if include_stubs:
        files["stubs/run-validation.py"] = _render_validation_stub()
    return files


def _render_manifest_content(contract: HarnessContract, files: dict[str, str]) -> str:
    entries = [
        {
            "path": relative_path,
            "sha256": _sha256_text(content),
        }
        for relative_path, content in sorted(files.items())
        if relative_path != GENERATED_MANIFEST
    ]
    manifest = {
        "schema_version": "roboharness_generated_harness_skill/v1",
        "generator": GENERATOR_VERSION,
        "source": "contract.py",
        "project_slug": contract.project_slug,
        "snapshot_sha256": _sha256_text(files["contract.snapshot.json"]),
        "files": entries,
    }
    return _json_dumps(manifest)


def _render_skill_md(snapshot: dict[str, Any]) -> str:
    workflow_lines = []
    phases_by_id = {phase["id"]: phase for phase in snapshot["phases"]}
    gates_by_id = {gate["id"]: gate for gate in snapshot["metric_gates"]}
    dimensions_by_id = {
        dimension["id"]: dimension for dimension in snapshot["visual_review_dimensions"]
    }
    commands_by_id = {command["id"]: command for command in snapshot["validation_commands"]}

    for workflow in snapshot["workflows"]:
        workflow_lines.append(f"### {workflow['label']} (`{workflow['id']}`)")
        workflow_lines.append(workflow["description"])
        workflow_lines.append("")
        if workflow["phases"]:
            phase_labels = [
                f"{phases_by_id[phase_id]['label']} (`{phase_id}`)"
                for phase_id in workflow["phases"]
            ]
            workflow_lines.append(f"- Phases: {', '.join(phase_labels)}")
        if workflow["metric_gates"]:
            workflow_lines.append("- Hard metric gates:")
            for gate_id in workflow["metric_gates"]:
                gate = gates_by_id[gate_id]
                workflow_lines.append(
                    "  - "
                    f"`{gate_id}`: `{gate['metric']} {gate['operator']} "
                    f"{gate['threshold']}` at `{gate['phase'] or 'all'}`"
                )
        if workflow["visual_dimensions"]:
            workflow_lines.append("- Visual review dimensions:")
            for dimension_id in workflow["visual_dimensions"]:
                dimension = dimensions_by_id[dimension_id]
                workflow_lines.append(
                    "  - "
                    f"`{dimension_id}`: {dimension['label']} at "
                    f"`{dimension['phase']}` views `{', '.join(dimension['views'])}`"
                )
        if workflow["validation_commands"]:
            workflow_lines.append("- Validation commands:")
            for command_id in workflow["validation_commands"]:
                command = commands_by_id[command_id]
                workflow_lines.append(f"  - `{command['command']}`")
        workflow_lines.append("")

    boundary_lines = []
    for boundary in snapshot["evidence_boundaries"]:
        boundary_lines.append(
            f"- `{boundary['id']}`: `{boundary['root']}` ({boundary['description']})"
        )
    if not boundary_lines:
        boundary_lines.append("- none declared")

    policy = snapshot["approval_policy"]
    return "\n".join(
        [
            "<!-- Generated from contract.py by roboharness.contract. Do not edit. -->",
            f"# {snapshot['name']} Harness",
            "",
            "## When To Use",
            "",
            snapshot["description"],
            "",
            "Use this project harness only for checks named in `contract.snapshot.json`. "
            "Prompt text in this skill is guidance; `contract.py` is the authority.",
            "",
            "## Authority Rules",
            "",
            "- Read `contract.snapshot.json` before selecting a workflow.",
            "- Run `roboharness contract check contract.py --output-dir .` before trusting "
            "generated instructions.",
            "- Do not invent new review gates, visual dimensions, evidence roots, or "
            "approval paths from chat context.",
            "- If a request is outside the approved contract, draft a Harness Scope Brief "
            "from `scope-brief-template.md` and ask for user approval before treating new "
            "checks as authoritative.",
            f"- Ambiguous results: `{policy['ambiguous_result']}`.",
            f"- Out-of-scope requests: `{policy['out_of_scope_request']}`.",
            "",
            "## Workflows",
            "",
            *workflow_lines,
            "## Evidence Boundaries",
            "",
            *boundary_lines,
            "",
            "## Baseline And Approval Policy",
            "",
            f"- Surface changed cases only: `{policy['surface_changed_cases_only']}`",
            "- Require user blessing for a new baseline: "
            f"`{policy['require_user_blessing_for_new_baseline']}`",
            "- Human scope approval required before a proposed contract becomes "
            f"authoritative: `{policy['human_scope_approval_required']}`",
            "",
        ]
    )


def _render_scope_brief_template(snapshot: dict[str, Any]) -> str:
    return "\n".join(
        [
            "<!-- Generated template. Copy before editing a project-specific brief. -->",
            f"# Harness Scope Brief: {snapshot['name']}",
            "",
            "## Proposed Scope Change",
            "",
            "- Request:",
            "- Why the existing contract is insufficient:",
            "- Proposed semantic phases:",
            "- Proposed hard metric gates:",
            "- Proposed visual review dimensions:",
            "- Proposed evidence boundaries:",
            "- Proposed validation commands:",
            "",
            "## User Approval Gate",
            "",
            "Do not update `contract.py` or generated artifacts until the user approves the "
            "scope change. After approval, edit `contract.py`, regenerate the skill, and run "
            "`roboharness contract check contract.py --output-dir .`.",
            "",
        ]
    )


def _render_generated_readme(snapshot: dict[str, Any]) -> str:
    return "\n".join(
        [
            "<!-- Generated from contract.py by roboharness.contract. Do not edit. -->",
            f"# {snapshot['name']} Project Harness Skill",
            "",
            "Source of truth: `contract.py`.",
            "",
            "Generated artifacts:",
            "",
            "- `SKILL.md` - agent-facing workflow guidance",
            "- `contract.snapshot.json` - normalized machine snapshot",
            "- `schemas/` - generated JSON schemas for snapshots and manifests",
            "- `scope-brief-template.md` - template for proposing contract changes",
            "- `stubs/run-validation.py` - optional validation-command runner",
            "- `.generated-manifest.json` - drift-check manifest",
            "",
            "Regenerate:",
            "",
            "```bash",
            "roboharness contract generate contract.py --output-dir .",
            "```",
            "",
            "Check drift:",
            "",
            "```bash",
            "roboharness contract check contract.py --output-dir .",
            "```",
            "",
        ]
    )


def _render_validation_stub() -> str:
    return '''"""Run validation commands from contract.snapshot.json."""

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
'''


def _contract_json_schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Roboharness Harness Contract Snapshot",
        "type": "object",
        "required": [
            "schema_version",
            "project_slug",
            "name",
            "version",
            "phases",
            "approval_policy",
            "workflows",
        ],
        "properties": {
            "schema_version": {"const": CONTRACT_SCHEMA_VERSION},
            "project_slug": {"type": "string", "pattern": _SLUG_RE.pattern},
            "name": {"type": "string"},
            "version": {"type": "string"},
            "description": {"type": "string"},
            "phases": {"type": "array", "minItems": 1},
            "metric_gates": {"type": "array"},
            "visual_review_dimensions": {"type": "array"},
            "evidence_boundaries": {"type": "array"},
            "approval_policy": {"type": "object"},
            "validation_commands": {"type": "array"},
            "workflows": {"type": "array", "minItems": 1},
        },
        "additionalProperties": False,
    }


def _generated_artifacts_schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Roboharness Generated Harness Skill Manifest",
        "type": "object",
        "required": [
            "schema_version",
            "generator",
            "source",
            "project_slug",
            "snapshot_sha256",
            "files",
        ],
        "properties": {
            "schema_version": {"const": "roboharness_generated_harness_skill/v1"},
            "generator": {"const": GENERATOR_VERSION},
            "source": {"const": "contract.py"},
            "project_slug": {"type": "string"},
            "snapshot_sha256": {"type": "string"},
            "files": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["path", "sha256"],
                    "properties": {
                        "path": {"type": "string"},
                        "sha256": {"type": "string"},
                    },
                    "additionalProperties": False,
                },
            },
        },
        "additionalProperties": False,
    }


def _phase_to_dict(phase: SemanticPhase) -> dict[str, Any]:
    return {
        "id": phase.id,
        "label": phase.label,
        "description": phase.description,
        "cameras": list(phase.cameras),
    }


def _evidence_reference_to_dict(reference: EvidenceReference) -> dict[str, Any]:
    return {
        "phase": reference.phase,
        "view": reference.view,
        "boundary": reference.boundary,
        "path": reference.path,
    }


def _metric_gate_to_dict(gate: MetricGate) -> dict[str, Any]:
    return {
        "id": gate.id,
        "metric": gate.metric,
        "operator": gate.operator,
        "threshold": _threshold_to_json(gate.threshold),
        "phase": gate.phase,
        "severity": gate.severity,
        "description": gate.description,
        "evidence": [_evidence_reference_to_dict(reference) for reference in gate.evidence],
    }


def _visual_dimension_to_dict(dimension: VisualReviewDimension) -> dict[str, Any]:
    return {
        "id": dimension.id,
        "label": dimension.label,
        "phase": dimension.phase,
        "views": list(dimension.views),
        "description": dimension.description,
        "required": dimension.required,
        "metric_fallback": list(dimension.metric_fallback),
        "evidence_boundary": dimension.evidence_boundary,
    }


def _evidence_boundary_to_dict(boundary: EvidenceBoundary) -> dict[str, Any]:
    return {
        "id": boundary.id,
        "root": boundary.root,
        "description": boundary.description,
        "allowed_patterns": list(boundary.allowed_patterns),
        "max_files": boundary.max_files,
    }


def _approval_policy_to_dict(policy: ApprovalPolicy) -> dict[str, Any]:
    return {
        "surface_changed_cases_only": policy.surface_changed_cases_only,
        "require_user_blessing_for_new_baseline": policy.require_user_blessing_for_new_baseline,
        "ambiguous_result": policy.ambiguous_result,
        "out_of_scope_request": policy.out_of_scope_request,
        "human_scope_approval_required": policy.human_scope_approval_required,
    }


def _validation_command_to_dict(command: ValidationCommand) -> dict[str, Any]:
    return {
        "id": command.id,
        "command": command.command,
        "description": command.description,
        "required": command.required,
    }


def _workflow_to_dict(workflow: HarnessWorkflow) -> dict[str, Any]:
    return {
        "id": workflow.id,
        "label": workflow.label,
        "description": workflow.description,
        "phases": list(workflow.phases),
        "metric_gates": list(workflow.metric_gates),
        "visual_dimensions": list(workflow.visual_dimensions),
        "validation_commands": list(workflow.validation_commands),
    }


def _threshold_to_json(threshold: float | tuple[float, float]) -> float | list[float]:
    if isinstance(threshold, tuple):
        return [float(threshold[0]), float(threshold[1])]
    return float(threshold)


def _json_dumps(data: dict[str, Any]) -> str:
    return json.dumps(data, indent=2, sort_keys=True) + "\n"


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _unique_ids(ids: Any, noun: str) -> frozenset[str]:
    seen: set[str] = set()
    for item_id in ids:
        if not isinstance(item_id, str) or not _ID_RE.match(item_id):
            raise ValueError(f"Invalid {noun} id: {item_id!r}.")
        if item_id in seen:
            raise ValueError(f"Duplicate {noun} id: {item_id!r}.")
        seen.add(item_id)
    return frozenset(seen)


def _require_slug(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not _SLUG_RE.match(value):
        raise ValueError(f"{field_name} must be a lower-case hyphenated slug.")


def _require_non_empty(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must not be empty.")


def _require_refs(values: tuple[str, ...], allowed: frozenset[str], field_name: str) -> None:
    for value in values:
        if value not in allowed:
            raise ValueError(f"{field_name} references unknown id {value!r}.")


def _validate_evidence_reference(
    reference: EvidenceReference,
    phase_ids: frozenset[str],
    boundary_ids: frozenset[str],
    *,
    owner: str,
) -> None:
    if reference.phase not in phase_ids:
        raise ValueError(f"{owner} evidence references unknown phase {reference.phase!r}.")
    if reference.boundary is not None and reference.boundary not in boundary_ids:
        raise ValueError(f"{owner} evidence references unknown boundary {reference.boundary!r}.")
