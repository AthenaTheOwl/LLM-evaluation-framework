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


def _repeated(scores_per_attempt, suite_name="regression-test"):
    return EvalSuiteResult(
        suite_name=suite_name,
        model="mock",
        run_id="rep",
        aggregate_scores=scores_per_attempt[0],
        attempt_scores=scores_per_attempt,
        pass_rate=1.0,
        passed=True,
    )


class TestNoiseAwareCompare:
    def _baseline(self, tmp_path):
        base = _repeated([{"deterministic": 1.0}, {"deterministic": 0.5}, {"deterministic": 1.0}])
        save_baseline(base, baselines_dir=tmp_path)
        return base

    def test_baseline_stores_attempts_and_spread(self, tmp_path):
        import json

        path = save_baseline(self._baseline(tmp_path), baselines_dir=tmp_path)
        data = json.loads(path.read_text(encoding="utf-8"))
        assert len(data["attempt_scores"]) == 3
        assert data["spread"] == {"deterministic": 0.5}

    def test_drop_inside_spread_is_within_noise(self, tmp_path):
        self._baseline(tmp_path)
        current = _repeated(
            [{"deterministic": 0.5}, {"deterministic": 1.0}, {"deterministic": 0.5}]
        )
        report = compare_baseline(current, baselines_dir=tmp_path)
        assert report.passed
        assert report.verdicts == {"deterministic": "within_noise"}
        assert report.baseline_spread == {"deterministic": 0.5}

    def test_drop_outside_spread_is_regression(self, tmp_path):
        self._baseline(tmp_path)
        current = _repeated([{"deterministic": 0.0}] * 3)
        report = compare_baseline(current, baselines_dir=tmp_path)
        assert not report.passed
        assert report.regressions == ["deterministic"]
        assert report.verdicts == {"deterministic": "regression"}

    def test_gain_is_improvement(self, tmp_path):
        low = _repeated([{"deterministic": 0.5}, {"deterministic": 0.4}])
        save_baseline(low, baselines_dir=tmp_path)
        report = compare_baseline(_repeated([{"deterministic": 0.9}] * 2), baselines_dir=tmp_path)
        assert report.verdicts == {"deterministic": "improvement"}

    def test_tolerance_is_the_floor(self, tmp_path):
        save_baseline(_repeated([{"deterministic": 1.0}] * 2), baselines_dir=tmp_path)  # spread 0
        report = compare_baseline(_repeated([{"deterministic": 0.9}] * 2), baselines_dir=tmp_path)
        assert report.regressions == ["deterministic"]

    def test_baseline_without_spread_behaves_as_before(self, suite_result, tmp_path):
        import json

        path = tmp_path / "regression-test.json"
        path.write_text(
            json.dumps({"run_id": "old", "aggregate_scores": {"deterministic": 0.95}}),
            encoding="utf-8",
        )
        worse = suite_result.model_copy(update={"aggregate_scores": {"deterministic": 0.80}})
        report = compare_baseline(worse, tolerance=0.05, baselines_dir=tmp_path)
        assert report.regressions == ["deterministic"]
        assert report.verdicts == {} and report.baseline_spread == {}

    def test_single_run_baseline_has_no_spread(self, suite_result, tmp_path):
        import json

        path = save_baseline(suite_result, baselines_dir=tmp_path)
        assert "spread" not in json.loads(path.read_text(encoding="utf-8"))
