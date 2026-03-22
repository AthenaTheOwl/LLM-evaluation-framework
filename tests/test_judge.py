"""Tests for Stage 3: LLM-as-a-Judge evaluation."""

import json

import pytest

from llm_evals.models import (
    EvalCase,
    EvalSuite,
    JudgeConfig,
    JudgeResult,
    QualityThresholds,
    RubricDimension,
)
from llm_evals.providers.mock import MockProvider
from llm_evals.stages.judge import JudgeStage


@pytest.fixture
def mock_provider():
    provider = MockProvider()
    provider.default_response = json.dumps({
        "chain_of_thought": "The response addresses the question directly and accurately.",
        "dimension_scores": {"quality": 4.5, "faithfulness": 4.0},
        "overall_score": 4.25,
        "confidence": 0.85,
    })
    return provider


@pytest.fixture
def suite_with_judge():
    return EvalSuite(
        name="judge-test",
        model="mock",
        stages=["judge"],
        judge=JudgeConfig(
            model="mock",
            rubric=[
                RubricDimension(name="quality", description="Overall quality"),
                RubricDimension(name="faithfulness", description="Factual accuracy", weight=2.0),
            ],
            human_baseline={"quality": 4.0, "faithfulness": 4.2},
        ),
        thresholds=QualityThresholds(judge_min_score=3.5),
        cases=[EvalCase(id="t1", prompt="test")],
    )


class TestJudgeStage:
    def test_basic_evaluation(self, mock_provider, suite_with_judge):
        stage = JudgeStage(judge_provider=mock_provider)
        case = suite_with_judge.cases[0]
        result = stage.evaluate(case, "Test response", suite_with_judge)

        assert isinstance(result, JudgeResult)
        assert result.overall_score == pytest.approx(4.25)
        assert result.confidence == pytest.approx(0.85)
        assert result.passed  # 4.25 >= 3.5

    def test_calibration_delta(self, mock_provider, suite_with_judge):
        stage = JudgeStage(judge_provider=mock_provider)
        case = suite_with_judge.cases[0]
        result = stage.evaluate(case, "Test response", suite_with_judge)

        # Baseline: quality=4.0, faithfulness=4.2
        # Scores: quality=4.5, faithfulness=4.0
        # Deltas: +0.5, -0.2 → avg = +0.15
        assert result.calibration_delta is not None
        assert result.calibration_delta == pytest.approx(0.15)

    def test_no_judge_config(self, mock_provider):
        suite = EvalSuite(name="no-judge", model="mock", stages=["judge"])
        stage = JudgeStage(judge_provider=mock_provider)
        result = stage.evaluate(EvalCase(id="t", prompt="t"), "output", suite)
        assert not result.passed

    def test_handles_parse_error(self, suite_with_judge):
        bad_provider = MockProvider(default_response="invalid json response")
        stage = JudgeStage(judge_provider=bad_provider)
        case = suite_with_judge.cases[0]
        result = stage.evaluate(case, "Test response", suite_with_judge)
        assert not result.passed
        assert result.score == 0.0

    def test_chain_of_thought_in_reasoning(self, mock_provider, suite_with_judge):
        stage = JudgeStage(judge_provider=mock_provider)
        case = suite_with_judge.cases[0]
        result = stage.evaluate(case, "Test response", suite_with_judge)
        assert "addresses the question" in result.reasoning
