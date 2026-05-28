"""Evidence artifact compatibility tests."""

from __future__ import annotations

from roboharness.evidence import (
    AutonomousEvidenceReport,
    RendererReport,
    SemanticSnapshotBundle,
    load_autonomous_evidence_report,
    load_renderer_report,
    load_semantic_snapshot_bundle,
)


def _groot_style_renderer_report() -> dict[str, object]:
    return {
        "output_dir": "case_001/meshcat",
        "renderer": "meshcat",
        "capture_ok": True,
        "motion_ok": True,
        "flags": ["workspace_framing_warning"],
        "trustworthiness_flags": [
            {
                "code": "workspace_framing_failure",
                "phase": "approach",
                "severity": "warning",
            }
        ],
        "metadata": {
            "capture_wall_s": 0.42,
            "replay_enabled": True,
            "render_entrypoint": "snapshot_bundle_renderer",
        },
        "snapshots": [
            {
                "name": "pregrasp",
                "capture_ok": True,
                "motion_ok": True,
                "metrics": {
                    "semantic_milestone": "pregrasp",
                    "control_backend": "decoupled_wbc",
                    "runtime_surface": "in_process",
                },
                "images": [
                    {
                        "camera": "front",
                        "path": "pregrasp/front.png",
                        "unique_colors": 128,
                        "green_pixels": 42,
                        "workspace_visible": True,
                    }
                ],
            }
        ],
    }


def test_semantic_snapshot_bundle_round_trips_groot_style_payload(tmp_path) -> None:
    payload = {
        "schema_version": 2,
        "snapshot_order": ["pregrasp", "lift"],
        "metadata": {
            "robot_type": "g1",
            "case_id": "X36_Y28_Z13",
            "control_backend": "decoupled_wbc",
        },
        "snapshots": [
            {
                "name": "pregrasp",
                "q": [0.0, 0.1, 0.2],
                "viz_q": [0.0, 0.1, 0.2],
                "bottle_xyz": [0.36, 0.28, 0.13],
                "camera_focus_xyz": [0.4, 0.2, 0.5],
                "grasp_markers": {"visible_grasp_trajectory_keys": ["approach_waypoints"]},
                "metrics": {
                    "semantic_milestone": "pregrasp",
                    "state_source": "env_state_act",
                    "snapshot_provenance": {"runtime_surface": "in_process"},
                },
            },
            {
                "name": "lift",
                "q": [0.3, 0.4, 0.5],
                "metrics": {
                    "semantic_milestone": "lift",
                    "grip_center_error_mm": 12.5,
                },
            },
        ],
    }

    bundle = SemanticSnapshotBundle.from_dict(payload)
    assert bundle.snapshot_order == ("pregrasp", "lift")
    assert bundle.snapshots[0].extra["q"] == [0.0, 0.1, 0.2]
    assert bundle.to_dict() == payload

    path = bundle.write_json(tmp_path / "snapshot_bundle.json")
    assert load_semantic_snapshot_bundle(path).to_dict() == payload


def test_renderer_report_round_trips_groot_style_report(tmp_path) -> None:
    payload = _groot_style_renderer_report()

    report = RendererReport.from_dict(payload)
    assert report.renderer == "meshcat"
    assert report.snapshots[0].images[0].extra["unique_colors"] == 128
    assert report.to_dict() == payload

    path = report.write_json(tmp_path / "meshcat_report.json")
    assert load_renderer_report(path).to_dict() == payload


def test_autonomous_evidence_report_preserves_downstream_fields(tmp_path) -> None:
    payload = {
        "robot_type": "g1",
        "case_id": "X36_Y28_Z13",
        "output_dir": "case_001",
        "runner": {
            "runner_type": "g1_visual_harness",
            "runtime_surface": "in_process",
            "replay_source": "saved_visual_packet",
        },
        "verdict": "pass",
        "verdict_reasons": [],
        "semantic_visual_ok": True,
        "workspace_framing_ok": True,
        "snapshot_state_progress_ok": True,
        "failure_taxonomy": [],
        "runtime": {"control_backend": "decoupled_wbc"},
        "plan": {"intent_level": "task_intent", "control_level": "planned_control"},
        "summary_metrics": {
            "render_replay_enabled": True,
            "render_total_s": 1.25,
        },
        "thresholds": {"max_grip_center_error_mm": 50.0},
        "snapshot_order": ["pregrasp"],
        "snapshot_metrics": {
            "pregrasp": {
                "semantic_milestone": "pregrasp",
                "control_backend": "decoupled_wbc",
            }
        },
        "renderer_reports": {"meshcat": _groot_style_renderer_report()},
    }

    report = AutonomousEvidenceReport.from_dict(payload)
    assert report.extra["robot_type"] == "g1"
    assert report.renderer_reports["meshcat"].renderer == "meshcat"
    assert report.to_dict() == payload

    path = report.write_json(tmp_path / "autonomous_report.json")
    assert load_autonomous_evidence_report(path).to_dict() == payload
