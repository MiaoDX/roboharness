"""Embedded visual lifecycle APIs for downstream robot harnesses."""

from roboharness.visual.lifecycle import (
    VisualCaseArtifacts,
    VisualCaseResult,
    VisualCaseRun,
    VisualCaseSpec,
    VisualSuiteArtifacts,
    VisualSuiteOptions,
    VisualSuiteReportArtifacts,
    VisualSuiteRun,
    VisualSuiteSpec,
    VisualSuiteSummary,
    collect_visual_suite,
    run_visual_suite,
    summarize_visual_suite_results,
    write_case_visual_artifacts,
    write_suite_visual_artifacts,
    write_visual_suite_report,
)

__all__ = [
    "VisualCaseArtifacts",
    "VisualCaseResult",
    "VisualCaseRun",
    "VisualCaseSpec",
    "VisualSuiteArtifacts",
    "VisualSuiteOptions",
    "VisualSuiteReportArtifacts",
    "VisualSuiteRun",
    "VisualSuiteSpec",
    "VisualSuiteSummary",
    "collect_visual_suite",
    "run_visual_suite",
    "summarize_visual_suite_results",
    "write_case_visual_artifacts",
    "write_suite_visual_artifacts",
    "write_visual_suite_report",
]
