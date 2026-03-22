"""Tests for Stage 2: Persona-based evaluation."""

import json

import pytest

from llm_evals.models import (
    EvalCase,
    EvalSuite,
    Persona,
    PersonaResult,
    QualityThresholds,
    RubricDimension,
)
from llm_evals.providers.mock import MockProvider
from llm_evals.stages.persona import PersonaStage


@pytest.fixture
def mock_provider():
    provider = MockProvider()
    provider.default_response = json.dumps({
        "dimension_scores": {"accuracy": 4.5, "helpfulness": 4.0},
        "overall_score": 4.25,
        "reasoning": "The response is accurate and helpful.",
    })
    return provider


@pytest.fixture
def suite_with_personas():
    return EvalSuite(
        name="persona-test",
        model="mock",
        stages=["persona"],
        personas=[
            Persona(
                name="expert",
                description="You are a domain expert.",
                weight=2.0,
                rubric=[
                    RubricDimension(name="accuracy", description="Is it correct?", weight=2.0),
                    RubricDimension(name="helpfulness", description="Is it helpful?", weight=1.0),
                ],
            ),
            Persona(
                name="novice",
                description="You are a beginner.",
                weight=1.0,
                rubric=[
                    RubricDimension(name="accuracy", description="Is it correct?"),
                    RubricDimension(name="helpfulness", description="Is it helpful?"),
                ],
            ),
        ],
        thresholds=QualityThresholds(persona_min_score=3.5),
        cases=[EvalCase(id="t1", prompt="test")],
    )


class TestPersonaStage:
    def test_basic_evaluation(self, mock_provider, suite_with_personas):
        stage = PersonaStage(judge_provider=mock_provider)
        case = suite_with_personas.cases[0]
        result = stage.evaluate(case, "Test response", suite_with_personas)

        assert isinstance(result, PersonaResult)
        assert "expert" in result.persona_scores
        assert "novice" in result.persona_scores

    def test_scoring(self, mock_provider, suite_with_personas):
        stage = PersonaStage(judge_provider=mock_provider)
        case = suite_with_personas.cases[0]
        result = stage.evaluate(case, "Test response", suite_with_personas)

        # Both personas return 4.25 (from mock), weighted: (4.25*2 + 4.25*1) / 3 = 4.25
        assert result.score == pytest.approx(4.25)
        assert result.passed  # 4.25 >= 3.5

    def test_no_personas(self, mock_provider):
        suite = EvalSuite(name="empty", model="mock", stages=["persona"])
        stage = PersonaStage(judge_provider=mock_provider)
        result = stage.evaluate(EvalCase(id="t", prompt="t"), "output", suite)
        assert not result.passed

    def test_handles_json_parse_error(self, suite_with_personas):
        bad_provider = MockProvider(default_response="not valid json")
        stage = PersonaStage(judge_provider=bad_provider)
        case = suite_with_personas.cases[0]
        result = stage.evaluate(case, "Test response", suite_with_personas)
        # Should not crash, scores default to 0
        assert result.score == 0.0
