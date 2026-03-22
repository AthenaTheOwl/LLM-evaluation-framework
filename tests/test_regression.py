"""Tests for baseline comparison and regression detection."""


import pytest

from llm_evals.models import EvalSuiteResult
from llm_evals.reporting.regression import compare_baseline, save_baseline


@pytest.fixture
def suite_result():
    return EvalSuiteResult(
        suite_name="regression-test",
        model="mock",
        run_id="abc123",
        aggregate_scores={"deterministic": 0.95, "persona": 0.80, "judge": 0.85},
        pass_rate=0.9,
        passed=True,
    )


class TestBaseline:
    def test_save_and_compare_no_regression(self, suite_result, tmp_path):
        save_baseline(suite_result, baselines_dir=tmp_path)
        report = compare_baseline(suite_result, baselines_dir=tmp_path)

        assert report is not None
        assert report.passed
        assert len(report.regressions) == 0
        # All deltas should be 0 (same scores)
        for delta in report.score_deltas.values():
            assert delta == pytest.approx(0.0)

    def test_detect_regression(self, suite_result, tmp_path):
        save_baseline(suite_result, baselines_dir=tmp_path)

        # Worse results
        worse = EvalSuiteResult(
            suite_name="regression-test",
            model="mock",
            run_id="def456",
            aggregate_scores={"deterministic": 0.80, "persona": 0.75, "judge": 0.85},
            pass_rate=0.7,
            passed=False,
        )
        report = compare_baseline(worse, tolerance=0.05, baselines_dir=tmp_path)

        assert report is not None
        assert not report.passed
        assert "deterministic" in report.regressions
        assert "persona" in report.regressions
        assert "judge" not in report.regressions  # same score

    def test_no_baseline_returns_none(self, suite_result, tmp_path):
        report = compare_baseline(suite_result, baselines_dir=tmp_path)
        assert report is None

    def test_tolerance_threshold(self, suite_result, tmp_path):
        save_baseline(suite_result, baselines_dir=tmp_path)

        # Slightly worse but within tolerance
        slightly_worse = EvalSuiteResult(
            suite_name="regression-test",
            model="mock",
            run_id="ghi789",
            aggregate_scores={"deterministic": 0.92, "persona": 0.78, "judge": 0.84},
            pass_rate=0.85,
            passed=True,
        )
        report = compare_baseline(slightly_worse, tolerance=0.05, baselines_dir=tmp_path)

        assert report is not None
        assert report.passed  # All within 5% tolerance
