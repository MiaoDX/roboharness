"""Tests for the constraint evaluator: assertions, constraints, and batch evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from roboharness.evaluate.assertions import AssertionEngine, MetricAssertion, _extract_metric
from roboharness.evaluate.batch import (
    EvalBatchResult,
    check_success_rate,
    evaluate_batch,
    evaluate_batch_with_comparison,
    find_reports,
    format_batch_human,
    format_comparison_human,
)
from roboharness.evaluate.constraints import load_constraints, save_constraints
from roboharness.evaluate.defaults import GRASP_DEFAULTS
from roboharness.evaluate.result import (
    AssertionResult,
    Operator,
    Severity,
    Verdict,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_REPORT: dict = {
    "verdict": "fail",
    "verdict_reasons": ["bad_grasp_geometry"],
    "failure_taxonomy": [
        {"category": "geometry_failure", "code": "bad_grasp_geometry", "detail": "pinch_gap"}
    ],
    "summary_metrics": {
        "grip_center_error_mm": 60.0,  # Above critical threshold (50.0)
        "pinch_gap_error_mm": 20.0,  # Above critical threshold (15.0)
        "pinch_elevation_deg": 5.0,
        "index_middle_vertical_deg": 10.0,
        "planning_time_s": 1.2,
    },
    "snapshot_metrics": {
        "05_holding": {
            "grip_center_error_mm": 9.0,
            "pinch_gap_error_mm": 13.0,
        },
    },
}

SAMPLE_REPORT_PASS: dict = {
    "verdict": "pass",
    "failure_taxonomy": [],
    "summary_metrics": {
        "grip_center_error_mm": 5.0,
        "pinch_gap_error_mm": 8.0,
        "pinch_elevation_deg": 3.0,
        "index_middle_vertical_deg": 10.0,
    },
    "snapshot_metrics": {},
}


# ---------------------------------------------------------------------------
# MetricAssertion tests
# ---------------------------------------------------------------------------


class TestMetricAssertion:
    def test_lt_pass(self) -> None:
        a = MetricAssertion("x", Operator.LT, 10.0)
        r = a.evaluate(5.0)
        assert r.passed
        assert r.actual_value == 5.0

    def test_lt_fail(self) -> None:
        a = MetricAssertion("x", Operator.LT, 10.0)
        r = a.evaluate(15.0)
        assert not r.passed
        assert "expected lt 10.0" in r.message

    def test_le_boundary(self) -> None:
        a = MetricAssertion("x", Operator.LE, 10.0)
        assert a.evaluate(10.0).passed
        assert not a.evaluate(10.1).passed

    def test_eq(self) -> None:
        a = MetricAssertion("x", Operator.EQ, 5.0)
        assert a.evaluate(5.0).passed
        assert not a.evaluate(5.1).passed

    def test_gt(self) -> None:
        a = MetricAssertion("x", Operator.GT, 0.0)
        assert a.evaluate(1.0).passed
        assert not a.evaluate(0.0).passed

    def test_ge(self) -> None:
        a = MetricAssertion("x", Operator.GE, 0.0)
        assert a.evaluate(0.0).passed
        assert not a.evaluate(-0.1).passed

    def test_in_range_pass(self) -> None:
        a = MetricAssertion("x", Operator.IN_RANGE, (0.0, 10.0))
        assert a.evaluate(5.0).passed
        assert a.evaluate(0.0).passed
        assert a.evaluate(10.0).passed

    def test_in_range_fail(self) -> None:
        a = MetricAssertion("x", Operator.IN_RANGE, (0.0, 10.0))
        r = a.evaluate(11.0)
        assert not r.passed
        assert "expected in [0.0, 10.0]" in r.message

    def test_missing_value(self) -> None:
        a = MetricAssertion("x", Operator.LT, 10.0, severity=Severity.CRITICAL)
        r = a.evaluate(None)
        assert not r.passed
        assert r.actual_value is None
        assert "missing" in r.message

    def test_severity_default(self) -> None:
        a = MetricAssertion("x", Operator.LT, 10.0)
        assert a.severity == Severity.MAJOR

    def test_phase_default(self) -> None:
        a = MetricAssertion("x", Operator.LT, 10.0)
        assert a.phase == "*"


# ---------------------------------------------------------------------------
# _extract_metric tests
# ---------------------------------------------------------------------------


class TestExtractMetric:
    def test_summary_metric(self) -> None:
        val = _extract_metric(SAMPLE_REPORT, "grip_center_error_mm", "*")
        assert val == 60.0

    def test_phase_metric(self) -> None:
        val = _extract_metric(SAMPLE_REPORT, "grip_center_error_mm", "05_holding")
        assert val == 9.0

    def test_missing_metric(self) -> None:
        val = _extract_metric(SAMPLE_REPORT, "nonexistent", "*")
        assert val is None

    def test_missing_phase(self) -> None:
        val = _extract_metric(SAMPLE_REPORT, "grip_center_error_mm", "99_nonexistent")
        assert val is None

    def test_boolean_metric_returns_none(self) -> None:
        report = {"summary_metrics": {"flag": True}}
        assert _extract_metric(report, "flag", "*") is None


# ---------------------------------------------------------------------------
# AssertionEngine tests
# ---------------------------------------------------------------------------


class TestAssertionEngine:
    def test_all_pass(self) -> None:
        assertions = [
            MetricAssertion("grip_center_error_mm", Operator.LT, 100.0, Severity.CRITICAL),
            MetricAssertion("pinch_gap_error_mm", Operator.LT, 100.0, Severity.CRITICAL),
        ]
        engine = AssertionEngine(assertions)
        result = engine.evaluate(SAMPLE_REPORT)
        assert result.verdict == Verdict.PASS
        assert len(result.passed) == 2

    def test_critical_failure(self) -> None:
        assertions = [
            MetricAssertion("grip_center_error_mm", Operator.LT, 5.0, Severity.CRITICAL),
        ]
        engine = AssertionEngine(assertions)
        result = engine.evaluate(SAMPLE_REPORT)
        assert result.verdict == Verdict.FAIL
        assert len(result.critical_failures) == 1

    def test_major_failure_degraded(self) -> None:
        assertions = [
            MetricAssertion("grip_center_error_mm", Operator.LT, 5.0, Severity.MAJOR),
        ]
        engine = AssertionEngine(assertions)
        result = engine.evaluate(SAMPLE_REPORT)
        assert result.verdict == Verdict.DEGRADED

    def test_phase_scoped_assertion(self) -> None:
        assertions = [
            MetricAssertion(
                "grip_center_error_mm", Operator.LT, 10.0, Severity.CRITICAL, phase="05_holding"
            ),
        ]
        engine = AssertionEngine(assertions)
        result = engine.evaluate(SAMPLE_REPORT)
        # 9.0 < 10.0 → pass
        assert result.verdict == Verdict.PASS

    def test_deterministic(self) -> None:
        engine = AssertionEngine(GRASP_DEFAULTS)
        r1 = engine.evaluate(SAMPLE_REPORT)
        r2 = engine.evaluate(SAMPLE_REPORT)
        assert r1.verdict == r2.verdict
        assert len(r1.results) == len(r2.results)
        for a, b in zip(r1.results, r2.results, strict=True):
            assert a.passed == b.passed
            assert a.actual_value == b.actual_value


# ---------------------------------------------------------------------------
# EvaluationResult serialization tests
# ---------------------------------------------------------------------------


class TestEvaluationResult:
    def test_to_dict(self) -> None:
        engine = AssertionEngine(GRASP_DEFAULTS)
        result = engine.evaluate(SAMPLE_REPORT, report_path="test.json")
        d = result.to_dict()
        assert d["verdict"] == "fail"  # SAMPLE_REPORT exceeds critical thresholds
        assert d["report_path"] == "test.json"
        assert d["total_assertions"] == len(GRASP_DEFAULTS)
        assert isinstance(d["results"], list)

    def test_assertion_result_to_dict_range(self) -> None:
        r = AssertionResult(
            metric="x",
            operator=Operator.IN_RANGE,
            threshold=(0.0, 10.0),
            severity=Severity.MINOR,
            phase="*",
            passed=True,
            actual_value=5.0,
        )
        d = r.to_dict()
        assert d["threshold"] == [0.0, 10.0]
        assert d["operator"] == "in_range"


# ---------------------------------------------------------------------------
# Constraint loading/saving tests
# ---------------------------------------------------------------------------


class TestConstraints:
    def test_load_json(self, tmp_path: Path) -> None:
        data = {
            "constraints": [
                {"metric": "x", "operator": "lt", "threshold": 10.0, "severity": "critical"},
                {"metric": "y", "operator": "in_range", "threshold": [0.0, 5.0]},
            ]
        }
        p = tmp_path / "constraints.json"
        p.write_text(json.dumps(data))
        assertions = load_constraints(p)
        assert len(assertions) == 2
        assert assertions[0].metric == "x"
        assert assertions[0].severity == Severity.CRITICAL
        assert assertions[1].operator == Operator.IN_RANGE
        assert assertions[1].threshold == (0.0, 5.0)
        # Default severity and phase
        assert assertions[1].severity == Severity.MAJOR
        assert assertions[1].phase == "*"

    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        original = GRASP_DEFAULTS
        p = tmp_path / "saved.json"
        save_constraints(original, p)
        loaded = load_constraints(p)
        assert len(loaded) == len(original)
        for a, b in zip(original, loaded, strict=True):
            assert a.metric == b.metric
            assert a.operator == b.operator
            assert a.threshold == b.threshold
            assert a.severity == b.severity

    def test_load_yaml(self, tmp_path: Path) -> None:
        yaml_content = """
constraints:
  - metric: grip_center_error_mm
    operator: lt
    threshold: 50.0
    severity: critical
  - metric: pinch_gap_error_mm
    operator: lt
    threshold: 15.0
"""
        p = tmp_path / "test.yaml"
        p.write_text(yaml_content)
        pytest.importorskip("yaml", reason="PyYAML not installed")
        assertions = load_constraints(p)
        assert len(assertions) == 2
        assert assertions[0].severity == Severity.CRITICAL


# ---------------------------------------------------------------------------
# Batch evaluation tests
# ---------------------------------------------------------------------------


def _write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report))


class TestBatchEvaluation:
    @pytest.fixture
    def batch_dir(self, tmp_path: Path) -> Path:
        """Create a directory with multiple trial reports."""
        _write_report(
            tmp_path / "trial_001" / "autonomous_report.json",
            SAMPLE_REPORT_PASS,
        )
        _write_report(
            tmp_path / "trial_002" / "autonomous_report.json",
            SAMPLE_REPORT,
        )
        return tmp_path

    def test_find_reports(self, batch_dir: Path) -> None:
        reports = find_reports(batch_dir)
        assert len(reports) == 2

    def test_evaluate_batch(self, batch_dir: Path) -> None:
        result = evaluate_batch(batch_dir, GRASP_DEFAULTS)
        assert result.total_trials == 2
        assert result.success_rate == 0.5
        assert result.verdicts.get("pass") == 1
        assert result.verdicts.get("fail") == 1

    def test_evaluate_batch_success_rate(self, batch_dir: Path) -> None:
        result = evaluate_batch(batch_dir, GRASP_DEFAULTS)
        # SAMPLE_REPORT_PASS should pass, SAMPLE_REPORT should fail on defaults
        assert result.success_rate == 0.5

    def test_failure_distribution(self, batch_dir: Path) -> None:
        result = evaluate_batch(batch_dir, GRASP_DEFAULTS)
        # SAMPLE_REPORT has failure_taxonomy with bad_grasp_geometry
        assert result.failure_distribution["bad_grasp_geometry"] == 1

    def test_constraint_failures_tracked(self, batch_dir: Path) -> None:
        result = evaluate_batch(batch_dir, GRASP_DEFAULTS)
        # SAMPLE_REPORT exceeds grip_center_error_mm and pinch_gap_error_mm
        assert "grip_center_error_mm" in result.constraint_failures
        assert "pinch_gap_error_mm" in result.constraint_failures

    def test_batch_to_dict(self, batch_dir: Path) -> None:
        result = evaluate_batch(batch_dir, GRASP_DEFAULTS)
        d = result.to_dict()
        assert d["total_trials"] == 2
        assert "verdicts" in d
        assert "success_rate" in d
        assert isinstance(d["trials"], list)

    def test_check_success_rate_pass(self) -> None:
        batch = EvalBatchResult(results_dir=".", total_trials=10, success_rate=0.85)
        assert check_success_rate(batch, 0.8)

    def test_check_success_rate_fail(self) -> None:
        batch = EvalBatchResult(results_dir=".", total_trials=10, success_rate=0.75)
        assert not check_success_rate(batch, 0.8)

    def test_empty_dir(self, tmp_path: Path) -> None:
        result = evaluate_batch(tmp_path, GRASP_DEFAULTS)
        assert result.total_trials == 0
        assert result.success_rate == 0.0

    def test_format_batch_human(self, batch_dir: Path) -> None:
        result = evaluate_batch(batch_dir, GRASP_DEFAULTS)
        text = format_batch_human(result)
        assert "Total trials: 2" in text
        assert "Success rate:" in text


class TestVariantComparison:
    @pytest.fixture
    def variant_dir(self, tmp_path: Path) -> Path:
        """Create a directory with two variant subdirs."""
        _write_report(
            tmp_path / "variant_A" / "trial_001" / "autonomous_report.json",
            SAMPLE_REPORT_PASS,
        )
        _write_report(
            tmp_path / "variant_A" / "trial_002" / "autonomous_report.json",
            SAMPLE_REPORT_PASS,
        )
        _write_report(
            tmp_path / "variant_B" / "trial_001" / "autonomous_report.json",
            SAMPLE_REPORT,
        )
        return tmp_path

    def test_comparison(self, variant_dir: Path) -> None:
        result = evaluate_batch_with_comparison(variant_dir, GRASP_DEFAULTS)
        assert len(result.variants) == 2
        a = next(v for v in result.variants if v.variant_id == "variant_A")
        b = next(v for v in result.variants if v.variant_id == "variant_B")
        assert a.batch.success_rate == 1.0
        assert b.batch.success_rate == 0.0

    def test_comparison_to_dict(self, variant_dir: Path) -> None:
        result = evaluate_batch_with_comparison(variant_dir, GRASP_DEFAULTS)
        d = result.to_dict()
        assert "variants" in d
        assert "summary" in d
        assert "variant_A" in d["summary"]

    def test_format_comparison_human(self, variant_dir: Path) -> None:
        result = evaluate_batch_with_comparison(variant_dir, GRASP_DEFAULTS)
        text = format_comparison_human(result)
        assert "variant_A" in text
        assert "variant_B" in text
        assert "100%" in text


# ---------------------------------------------------------------------------
# Integration tests with real harness data
# ---------------------------------------------------------------------------


ASSETS_DIR = Path(__file__).parent.parent / "assets"


@pytest.mark.skipif(
    not (ASSETS_DIR / "00").exists(),
    reason="Real harness report data not available",
)
class TestIntegrationRealData:
    """Integration tests using real autonomous_report.json files from assets/."""

    def test_evaluate_single_report(self) -> None:
        report_path = ASSETS_DIR / "00" / "X26_Y22_Z13" / "autonomous_report.json"
        report = json.loads(report_path.read_text())
        engine = AssertionEngine(GRASP_DEFAULTS)
        result = engine.evaluate(report, report_path=str(report_path))
        # Should produce a deterministic verdict
        assert result.verdict in (Verdict.PASS, Verdict.DEGRADED, Verdict.FAIL)
        assert len(result.results) == len(GRASP_DEFAULTS)

    def test_batch_evaluate_assets(self) -> None:
        result = evaluate_batch(ASSETS_DIR / "00", GRASP_DEFAULTS)
        assert result.total_trials == 2
        assert 0.0 <= result.success_rate <= 1.0

    def test_variant_comparison_assets(self) -> None:
        result = evaluate_batch_with_comparison(ASSETS_DIR / "00", GRASP_DEFAULTS)
        assert len(result.variants) == 2
        for v in result.variants:
            assert v.batch.total_trials == 1


# ---------------------------------------------------------------------------
# CLI integration tests
# ---------------------------------------------------------------------------


class TestEvaluateCLI:
    @pytest.fixture
    def report_file(self, tmp_path: Path) -> Path:
        p = tmp_path / "autonomous_report.json"
        p.write_text(json.dumps(SAMPLE_REPORT_PASS))
        return p

    @pytest.fixture
    def failing_report_file(self, tmp_path: Path) -> Path:
        failing = {
            **SAMPLE_REPORT,
            "summary_metrics": {
                **SAMPLE_REPORT["summary_metrics"],
                "grip_center_error_mm": 100.0,  # Above critical threshold
            },
        }
        p = tmp_path / "autonomous_report.json"
        p.write_text(json.dumps(failing))
        return p

    def test_evaluate_pass(self, report_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
        from roboharness.cli import main

        ret = main(["evaluate", str(report_file)])
        assert ret == 0
        out = capsys.readouterr().out
        assert "pass" in out.lower()

    def test_evaluate_fail(
        self, failing_report_file: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from roboharness.cli import main

        ret = main(["evaluate", str(failing_report_file)])
        assert ret == 1
        out = capsys.readouterr().out
        assert "fail" in out.lower()

    def test_evaluate_json_format(
        self, report_file: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from roboharness.cli import main

        ret = main(["evaluate", str(report_file), "--format", "json"])
        assert ret == 0
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["verdict"] == "pass"

    def test_evaluate_missing_file(self) -> None:
        from roboharness.cli import main

        ret = main(["evaluate", "/nonexistent/report.json"])
        assert ret == 1

    def test_evaluate_with_constraints(
        self, report_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from roboharness.cli import main

        constraints = {
            "constraints": [
                {"metric": "grip_center_error_mm", "operator": "lt", "threshold": 100.0}
            ]
        }
        cp = tmp_path / "custom.json"
        cp.write_text(json.dumps(constraints))
        ret = main(["evaluate", str(report_file), "--constraints", str(cp)])
        assert ret == 0


class TestEvaluateBatchCLI:
    @pytest.fixture
    def batch_dir(self, tmp_path: Path) -> Path:
        _write_report(tmp_path / "t1" / "autonomous_report.json", SAMPLE_REPORT_PASS)
        _write_report(tmp_path / "t2" / "autonomous_report.json", SAMPLE_REPORT_PASS)
        return tmp_path

    def test_batch_human(self, batch_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
        from roboharness.cli import main

        ret = main(["evaluate-batch", str(batch_dir)])
        assert ret == 0
        out = capsys.readouterr().out
        assert "Total trials: 2" in out

    def test_batch_json(self, batch_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
        from roboharness.cli import main

        ret = main(["evaluate-batch", str(batch_dir), "--format", "json"])
        assert ret == 0
        data = json.loads(capsys.readouterr().out)
        assert data["total_trials"] == 2

    def test_batch_min_success_rate_pass(
        self, batch_dir: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from roboharness.cli import main

        ret = main(["evaluate-batch", str(batch_dir), "--min-success-rate", "0.5"])
        assert ret == 0

    def test_batch_min_success_rate_fail(
        self, batch_dir: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from roboharness.cli import main

        # Add a failing trial
        _write_report(
            batch_dir / "t3" / "autonomous_report.json",
            {
                **SAMPLE_REPORT,
                "summary_metrics": {
                    **SAMPLE_REPORT["summary_metrics"],
                    "grip_center_error_mm": 100.0,
                },
            },
        )
        ret = main(["evaluate-batch", str(batch_dir), "--min-success-rate", "0.99"])
        assert ret == 1

    def test_batch_compare(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        from roboharness.cli import main

        _write_report(tmp_path / "A" / "t1" / "autonomous_report.json", SAMPLE_REPORT_PASS)
        _write_report(tmp_path / "B" / "t1" / "autonomous_report.json", SAMPLE_REPORT)
        main(["evaluate-batch", str(tmp_path), "--compare"])
        out = capsys.readouterr().out
        assert "A" in out
        assert "B" in out

    def test_batch_missing_dir(self) -> None:
        from roboharness.cli import main

        ret = main(["evaluate-batch", "/nonexistent/dir"])
        assert ret == 1


# ---------------------------------------------------------------------------
# End-to-end: YAML constraints -> evaluate -> HTML report with verdict
# ---------------------------------------------------------------------------


class TestEvaluateToReportEndToEnd:
    """Full workflow: load YAML constraints, evaluate report, generate HTML with verdict."""

    @pytest.fixture
    def grasp_output(self, tmp_path: Path) -> Path:
        """Create a realistic harness output with checkpoints and a report."""
        import base64

        tiny_png = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4"
            "nGP4z8BQDwAEgAF/pooBPQAAAABJRU5ErkJggg=="
        )
        trial_dir = tmp_path / "grasp" / "trial_001"

        for cp_name, step, sim_time in [
            ("01_pregrasp", 0, 0.0),
            ("02_contact", 50, 0.5),
            ("03_grasp", 100, 1.0),
            ("04_lift", 150, 1.5),
        ]:
            cp_dir = trial_dir / cp_name
            cp_dir.mkdir(parents=True)
            (cp_dir / "metadata.json").write_text(
                json.dumps({"step": step, "sim_time": sim_time, "cameras": ["front"]})
            )
            (cp_dir / "front_rgb.png").write_bytes(tiny_png)

        # Write autonomous_report.json (what the evaluator reads)
        report = {
            "summary_metrics": {
                "grip_center_error_mm": 8.2,
                "pinch_gap_error_mm": 6.1,
                "pinch_elevation_deg": 12.0,
                "index_middle_vertical_deg": 18.5,
            },
            "snapshot_metrics": {
                "03_grasp": {
                    "grip_center_error_mm": 7.5,
                    "pinch_gap_error_mm": 5.8,
                },
            },
        }
        (trial_dir / "autonomous_report.json").write_text(json.dumps(report))

        return tmp_path

    def test_yaml_to_evaluate_to_html_pass(self, grasp_output: Path) -> None:
        """End-to-end: load grasp_default.yaml, evaluate passing report, HTML shows PASS."""
        from roboharness.evaluate.assertions import AssertionEngine
        from roboharness.evaluate.constraints import load_constraints
        from roboharness.reporting import generate_html_report

        # Step 1: Load constraints from the project's actual YAML file
        constraints_path = Path(__file__).parent.parent / "constraints" / "grasp_default.yaml"
        pytest.importorskip("yaml", reason="PyYAML not installed")
        assertions = load_constraints(constraints_path)
        assert len(assertions) == 4

        # Step 2: Evaluate the report
        report_path = grasp_output / "grasp" / "trial_001" / "autonomous_report.json"
        report_data = json.loads(report_path.read_text())
        engine = AssertionEngine(assertions)
        result = engine.evaluate(report_data, report_path=str(report_path))

        # All metrics are within thresholds
        assert result.verdict == Verdict.PASS

        # Step 3: Generate HTML report with evaluation
        html_path = generate_html_report(
            grasp_output,
            "grasp",
            title="Grasp Evaluation",
            evaluation_result=result,
        )
        html = html_path.read_text()

        # Verify verdict banner
        assert "verdict-pass" in html
        assert "PASS" in html
        assert "4/4 constraints satisfied" in html

        # Verify constraint summary table
        assert "grip_center_error_mm" in html
        assert "pinch_gap_error_mm" in html
        assert "Constraint Evaluation" in html

        # Verify checkpoints are still rendered
        assert "01_pregrasp" in html
        assert "04_lift" in html

    def test_yaml_to_evaluate_to_html_fail(self, grasp_output: Path) -> None:
        """End-to-end: evaluate failing report, HTML shows FAIL with red indicators."""
        from roboharness.evaluate.assertions import AssertionEngine
        from roboharness.evaluate.constraints import load_constraints
        from roboharness.reporting import generate_html_report

        constraints_path = Path(__file__).parent.parent / "constraints" / "grasp_default.yaml"
        pytest.importorskip("yaml", reason="PyYAML not installed")
        assertions = load_constraints(constraints_path)

        # Override report with failing metrics
        failing_report = {
            "summary_metrics": {
                "grip_center_error_mm": 60.0,  # > 50.0 critical threshold
                "pinch_gap_error_mm": 20.0,  # > 15.0 critical threshold
                "pinch_elevation_deg": 5.0,
                "index_middle_vertical_deg": 10.0,
            },
            "snapshot_metrics": {},
        }
        report_path = grasp_output / "grasp" / "trial_001" / "autonomous_report.json"
        report_path.write_text(json.dumps(failing_report))

        engine = AssertionEngine(assertions)
        result = engine.evaluate(failing_report, report_path=str(report_path))
        assert result.verdict == Verdict.FAIL

        html_path = generate_html_report(
            grasp_output,
            "grasp",
            title="Grasp Evaluation - Failure",
            evaluation_result=result,
        )
        html = html_path.read_text()

        # Verify fail verdict
        assert "verdict-fail" in html
        assert "FAIL" in html
        assert "2/4 constraints satisfied" in html

        # Verify failing metrics shown
        body = html.split("</style>")[1]
        assert "badge-fail" in body

    def test_cli_evaluate_with_yaml_constraints(self, grasp_output: Path) -> None:
        """End-to-end: CLI evaluate command with real YAML constraints."""
        from roboharness.cli import main

        constraints_path = Path(__file__).parent.parent / "constraints" / "grasp_default.yaml"
        pytest.importorskip("yaml", reason="PyYAML not installed")
        report_path = grasp_output / "grasp" / "trial_001" / "autonomous_report.json"

        ret = main(["evaluate", str(report_path), "--constraints", str(constraints_path)])
        assert ret == 0  # All pass
