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

    def test_weighted_across_personas(self, suite_with_personas):
        # The two personas return different overall_scores, so the persona weights
        # (expert 2.0, novice 1.0) actually move the aggregate: an unweighted mean
        # would give 3.5, the weighted mean gives (5*2 + 2*1) / 3 = 4.0.
        provider = MockProvider()
        provider.responses = {
            # The persona name appears in the prompt ("... as a {name}").
            "as a expert": json.dumps({
                "dimension_scores": {"accuracy": 5.0, "helpfulness": 5.0},
                "overall_score": 5.0,
                "reasoning": "expert view",
            }),
            "as a novice": json.dumps({
                "dimension_scores": {"accuracy": 2.0, "helpfulness": 2.0},
                "overall_score": 2.0,
                "reasoning": "novice view",
            }),
        }
        stage = PersonaStage(judge_provider=provider)
        case = suite_with_personas.cases[0]
        result = stage.evaluate(case, "Test response", suite_with_personas)

        assert result.persona_scores["expert"].overall_score == pytest.approx(5.0)
        assert result.persona_scores["novice"].overall_score == pytest.approx(2.0)
        assert result.score == pytest.approx(4.0)

    def test_pass_threshold_boundary(self, suite_with_personas):
        # Aggregate exactly equals persona_min_score (3.5); the gate is >=.
        provider = MockProvider(default_response=json.dumps({
            "dimension_scores": {"accuracy": 3.5, "helpfulness": 3.5},
            "overall_score": 3.5,
            "reasoning": "borderline",
        }))
        stage = PersonaStage(judge_provider=provider)
        case = suite_with_personas.cases[0]
        result = stage.evaluate(case, "Test response", suite_with_personas)

        assert result.score == pytest.approx(3.5)
        assert result.passed

    def test_dimension_scores_clamped_to_scale(self, suite_with_personas):
        # A provider returning out-of-scale dimension scores gets clamped to the
        # rubric's [scale_min, scale_max] = [1.0, 5.0].
        provider = MockProvider(default_response=json.dumps({
            "dimension_scores": {"accuracy": 9.0, "helpfulness": -2.0},
            "overall_score": 4.0,
            "reasoning": "out of range",
        }))
        stage = PersonaStage(judge_provider=provider)
        case = suite_with_personas.cases[0]
        result = stage.evaluate(case, "Test response", suite_with_personas)

        expert = result.persona_scores["expert"]
        assert expert.dimension_scores["accuracy"] == pytest.approx(5.0)
        assert expert.dimension_scores["helpfulness"] == pytest.approx(1.0)
