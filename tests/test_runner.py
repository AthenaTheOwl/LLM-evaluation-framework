"""Tests for the pipeline runner."""

import json

import pytest

from llm_evals.models import (
    Assertion,
    AssertionType,
    EvalCase,
    EvalSuite,
    JudgeConfig,
    QualityThresholds,
    RubricDimension,
)
from llm_evals.providers.mock import MockProvider
from llm_evals.runner import EvalRunner


@pytest.fixture
def mock_provider():
    provider = MockProvider(
        default_response="I can help you with that. The refund is 3-5 business days."
    )
    return provider


@pytest.fixture
def simple_suite():
    return EvalSuite(
        name="test-runner",
        model="mock",
        provider="mock",
        stages=["deterministic"],
        cases=[
            EvalCase(
                id="case-1",
                prompt="Help me",
                assertions=[
                    Assertion(type=AssertionType.CONTAINS, value="help"),
                    Assertion(type=AssertionType.MAX_LENGTH, value=500),
                ],
            ),
            EvalCase(
                id="case-2",
                prompt="Refund please",
                assertions=[
                    Assertion(type=AssertionType.CONTAINS, value="refund"),
                ],
            ),
        ],
    )


class TestEvalRunner:
    def test_basic_run(self, simple_suite, mock_provider):
        runner = EvalRunner(
            suite=simple_suite,
            model_provider=mock_provider,
            concurrency=1,
            stages_filter=["deterministic"],
        )
        result = runner.run()
        assert result.suite_name == "test-runner"
        assert len(result.case_results) == 2

    def test_all_pass(self, simple_suite, mock_provider):
        runner = EvalRunner(
            suite=simple_suite,
            model_provider=mock_provider,
            concurrency=1,
            stages_filter=["deterministic"],
        )
        result = runner.run()
        assert result.pass_rate == 1.0
        assert result.passed

    def test_case_ordering_preserved(self, simple_suite, mock_provider):
        runner = EvalRunner(
            suite=simple_suite,
            model_provider=mock_provider,
            concurrency=1,
            stages_filter=["deterministic"],
        )
        result = runner.run()
        assert result.case_results[0].case_id == "case-1"
        assert result.case_results[1].case_id == "case-2"

    def test_fail_fast(self, mock_provider):
        suite = EvalSuite(
            name="fail-fast-test",
            model="mock",
            provider="mock",
            stages=["deterministic", "persona"],
            fail_fast=True,
            cases=[
                EvalCase(
                    id="will-fail",
                    prompt="test",
                    assertions=[
                        Assertion(type=AssertionType.EXACT_MATCH, value="impossible match xyz"),
                    ],
                ),
            ],
        )
        runner = EvalRunner(
            suite=suite,
            model_provider=mock_provider,
            concurrency=1,
            stages_filter=["deterministic"],
        )
        result = runner.run()
        # Should have deterministic result but no persona result (skipped due to fail_fast)
        case_result = result.case_results[0]
        assert not case_result.passed
        assert len(case_result.stage_results) == 1

    def test_concurrent_execution(self, simple_suite, mock_provider):
        runner = EvalRunner(
            suite=simple_suite,
            model_provider=mock_provider,
            concurrency=2,
            stages_filter=["deterministic"],
        )
        result = runner.run()
        assert len(result.case_results) == 2
        assert result.passed

    def test_judge_thresholds_use_native_scale(self):
        model_provider = MockProvider(default_response="model output")
        judge_provider = MockProvider(default_response=json.dumps({
            "dimension_scores": {"quality": 1.0},
            "overall_score": 1.0,
            "confidence": 0.9,
        }))

        suite = EvalSuite(
            name="threshold-check",
            model="mock",
            provider="mock",
            stages=["judge"],
            judge=JudgeConfig(
                model="mock",
                provider="mock",
                rubric=[RubricDimension(name="quality", description="Overall quality")],
            ),
            thresholds=QualityThresholds(judge_min_score=4.0),
            cases=[EvalCase(id="case-1", prompt="Prompt")],
        )

        runner = EvalRunner(
            suite=suite,
            model_provider=model_provider,
            judge_provider=judge_provider,
            concurrency=1,
            stages_filter=["judge"],
        )
        result = runner.run()

        assert result.aggregate_scores["judge"] == pytest.approx(1.0)
        assert not result.stage_passed["judge"]
        assert not result.passed
