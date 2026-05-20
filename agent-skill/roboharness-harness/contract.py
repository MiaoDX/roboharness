"""Approved roboharness project harness contract.

This file is the handwritten source of truth. Files generated beside it are
derived artifacts and should be regenerated, not edited directly.
"""

from __future__ import annotations

from roboharness.contract import (
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

CONTRACT = HarnessContract(
    project_slug="roboharness",
    name="Roboharness",
    version="0.3.1",
    description=(
        "Use this harness to review roboharness trust-loop changes where metric-backed "
        "proof packs, bounded visual evidence, approval decisions, and release truth must "
        "stay aligned with the accepted project contract."
    ),
    phases=(
        SemanticPhase(
            id="plan",
            label="Plan",
            description="Confirm the deterministic MuJoCo grasp trajectory and target path.",
            cameras=("front", "side", "top"),
        ),
        SemanticPhase(
            id="pre_grasp",
            label="Pre-Grasp",
            description="Move the gripper above the cube before final approach.",
            cameras=("front", "side", "top"),
        ),
        SemanticPhase(
            id="approach",
            label="Approach",
            description="Approach the cube while preserving baseline alignment.",
            cameras=("front", "side", "top"),
        ),
        SemanticPhase(
            id="grasp",
            label="Grasp",
            description="Close the gripper on the cube with plausible contact geometry.",
            cameras=("front", "side", "top"),
        ),
        SemanticPhase(
            id="lift",
            label="Lift",
            description="Lift the cube and keep the proof pack grounded in paired evidence.",
            cameras=("front", "side", "top"),
        ),
        SemanticPhase(
            id="release",
            label="Release Truth",
            description="Keep package metadata, release tags, and public version claims aligned.",
        ),
    ),
    metric_gates=(
        MetricGate(
            id="loop_runtime_drift",
            metric="loop_runtime_s_abs_delta",
            operator="lt",
            threshold=1.0,
            severity="warn",
            description="The maintained MuJoCo loop should not drift materially in runtime.",
            evidence=(EvidenceReference(phase="plan", boundary="mujoco_trial"),),
        ),
        MetricGate(
            id="approach_center_drift",
            metric="grip_center_error_mm_abs_delta",
            operator="lt",
            threshold=12.0,
            phase="approach",
            description="Approach alignment must stay close to the blessed baseline.",
            evidence=(
                EvidenceReference(
                    phase="approach",
                    view="side",
                    boundary="mujoco_trial",
                    path="approach/side_rgb.png",
                ),
            ),
        ),
        MetricGate(
            id="grasp_gap_drift",
            metric="pinch_gap_error_mm_abs_delta",
            operator="lt",
            threshold=10.0,
            phase="grasp",
            description="The gripper fingers should bracket the cube without large gap drift.",
            evidence=(
                EvidenceReference(
                    phase="grasp",
                    view="front",
                    boundary="mujoco_trial",
                    path="grasp/front_rgb.png",
                ),
            ),
        ),
        MetricGate(
            id="lift_contact",
            metric="contact_count",
            operator="ge",
            threshold=1.0,
            phase="lift",
            description="The lift phase requires at least one cube-finger contact.",
            evidence=(
                EvidenceReference(
                    phase="lift",
                    view="front",
                    boundary="mujoco_trial",
                    path="lift/front_rgb.png",
                ),
            ),
        ),
        MetricGate(
            id="lift_height",
            metric="cube_height_mm",
            operator="gt",
            threshold=5.0,
            phase="lift",
            description="The cube must visibly and metrically leave the table.",
            evidence=(
                EvidenceReference(
                    phase="lift",
                    view="side",
                    boundary="mujoco_trial",
                    path="lift/side_rgb.png",
                ),
            ),
        ),
    ),
    visual_review_dimensions=(
        VisualReviewDimension(
            id="hand_pose",
            label="Hand Pose",
            phase="grasp",
            views=("front", "side"),
            description="Check that the fingers visually bracket the cube like the baseline.",
            metric_fallback=(
                "grip_center_error_mm_abs_delta",
                "pinch_gap_error_mm_abs_delta",
            ),
            evidence_boundary="mujoco_trial",
        ),
        VisualReviewDimension(
            id="object_relative_position",
            label="Object Relative Position",
            phase="lift",
            views=("front", "side"),
            description="Check that the cube is lifted in the expected relation to the gripper.",
            metric_fallback=("cube_height_mm", "contact_count"),
            evidence_boundary="mujoco_trial",
        ),
        VisualReviewDimension(
            id="proof_pack_integrity",
            label="Proof Pack Integrity",
            phase="plan",
            views=("front", "top"),
            description=(
                "Check that selected current-vs-baseline evidence paths match the manifest "
                "instead of silently falling back to unrelated images."
            ),
            metric_fallback=("loop_runtime_s_abs_delta",),
            evidence_boundary="mujoco_baseline",
        ),
    ),
    evidence_boundaries=(
        EvidenceBoundary(
            id="mujoco_trial",
            root="examples/demos/mujoco/output",
            description="Generated MuJoCo grasp trial proof packs.",
            allowed_patterns=("**/*.json", "**/*_rgb.png", "**/*.html"),
            max_files=500,
        ),
        EvidenceBoundary(
            id="mujoco_baseline",
            root="assets/example_mujoco_grasp",
            description="Blessed baseline and known-bad visual fixtures for the maintained wedge.",
            allowed_patterns=("**/*.json", "**/*_rgb.png", "**/*.png"),
            max_files=200,
        ),
        EvidenceBoundary(
            id="release_truth",
            root=".",
            description="Package metadata, version files, release docs, and status dashboard.",
            allowed_patterns=(
                "pyproject.toml",
                "src/roboharness/__init__.py",
                "CHANGELOG.md",
                "STATUS.md",
                "docs/agents/release.md",
            ),
            max_files=20,
        ),
    ),
    approval_policy=ApprovalPolicy(
        surface_changed_cases_only=True,
        require_user_blessing_for_new_baseline=True,
        ambiguous_result="never_self_promote_to_pass",
        out_of_scope_request="draft_scope_brief_before_contract_change",
        human_scope_approval_required=True,
    ),
    validation_commands=(
        ValidationCommand(
            id="contract_drift",
            command=(
                "python -m roboharness.cli contract check "
                "agent-skill/roboharness-harness/contract.py "
                "--output-dir agent-skill/roboharness-harness"
            ),
            description="Generated project harness skill artifacts match contract.py.",
        ),
        ValidationCommand(
            id="mujoco_wedge_tests",
            command=(
                "python -m pytest --no-cov "
                "tests/regression/mujoco_grasp/test_mujoco_grasp_wedge.py "
                "tests/contract/test_approval_evidence.py"
            ),
            description="Maintained MuJoCo trust-loop contract tests pass.",
        ),
        ValidationCommand(
            id="release_truth_tests",
            command="python -m pytest --no-cov tests/contract/test_release_truth.py",
            description="Version and release-truth contract tests pass.",
        ),
        ValidationCommand(
            id="static_python_gate",
            command="ruff check src/roboharness examples/demos/mujoco tests/contract",
            description="Changed Python surfaces satisfy the repo lint gate.",
        ),
    ),
    workflows=(
        HarnessWorkflow(
            id="mujoco_contract_trust_loop",
            label="MuJoCo Contract Trust Loop",
            description=(
                "Review changes to the deterministic MuJoCo grasp proof pack, including "
                "metric gates, visual review dimensions, evidence pairing, and approval "
                "semantics."
            ),
            phases=("plan", "pre_grasp", "approach", "grasp", "lift"),
            metric_gates=(
                "loop_runtime_drift",
                "approach_center_drift",
                "grasp_gap_drift",
                "lift_contact",
                "lift_height",
            ),
            visual_dimensions=(
                "hand_pose",
                "object_relative_position",
                "proof_pack_integrity",
            ),
            validation_commands=(
                "contract_drift",
                "mujoco_wedge_tests",
                "static_python_gate",
            ),
        ),
        HarnessWorkflow(
            id="release_truth_alignment",
            label="Release Truth Alignment",
            description=(
                "Review release-facing changes where pyproject metadata, package version, "
                "status, and release documentation must tell the same truth."
            ),
            phases=("release",),
            validation_commands=("contract_drift", "release_truth_tests", "static_python_gate"),
        ),
    ),
)
