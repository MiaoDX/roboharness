"""Typed source objects for project harness contracts.

These dataclasses are intentionally small and boring. A project-owned
``contract.py`` imports them, declares the approved review surface, and the
generator turns that Python source into deterministic agent-facing artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SemanticPhase:
    """A named project phase where evidence can be captured or reviewed."""

    id: str
    label: str
    description: str
    cameras: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvidenceReference:
    """Where a gate or visual dimension expects review evidence."""

    phase: str
    view: str | None = None
    boundary: str | None = None
    path: str | None = None


@dataclass(frozen=True)
class MetricGate:
    """A hard metric gate authorized by the project contract."""

    id: str
    metric: str
    operator: str
    threshold: float | tuple[float, float]
    phase: str | None = None
    severity: str = "fail"
    description: str = ""
    evidence: tuple[EvidenceReference, ...] = ()


@dataclass(frozen=True)
class VisualReviewDimension:
    """A bounded visual judgment an agent may perform for this project."""

    id: str
    label: str
    phase: str
    views: tuple[str, ...]
    description: str
    required: bool = True
    metric_fallback: tuple[str, ...] = ()
    evidence_boundary: str | None = None


@dataclass(frozen=True)
class EvidenceBoundary:
    """A file boundary where generated skills may look for evidence."""

    id: str
    root: str
    description: str
    allowed_patterns: tuple[str, ...] = ("**/*",)
    max_files: int | None = None


@dataclass(frozen=True)
class ApprovalPolicy:
    """Human-review and baseline-authority rules for the harness."""

    surface_changed_cases_only: bool = True
    require_user_blessing_for_new_baseline: bool = True
    ambiguous_result: str = "never_self_promote_to_pass"
    out_of_scope_request: str = "route_to_contract_improvement"
    human_scope_approval_required: bool = True


@dataclass(frozen=True)
class ValidationCommand:
    """A project command the generated skill can run as evidence."""

    id: str
    command: str
    description: str
    required: bool = True


@dataclass(frozen=True)
class HarnessWorkflow:
    """One named review workflow inside a project harness skill."""

    id: str
    label: str
    description: str
    phases: tuple[str, ...] = ()
    metric_gates: tuple[str, ...] = ()
    visual_dimensions: tuple[str, ...] = ()
    validation_commands: tuple[str, ...] = ()


@dataclass(frozen=True)
class HarnessContract:
    """Trusted Python-authored source of truth for a generated harness skill."""

    project_slug: str
    name: str
    version: str
    description: str
    phases: tuple[SemanticPhase, ...]
    metric_gates: tuple[MetricGate, ...] = ()
    visual_review_dimensions: tuple[VisualReviewDimension, ...] = ()
    evidence_boundaries: tuple[EvidenceBoundary, ...] = ()
    approval_policy: ApprovalPolicy = field(default_factory=ApprovalPolicy)
    validation_commands: tuple[ValidationCommand, ...] = ()
    workflows: tuple[HarnessWorkflow, ...] = ()
