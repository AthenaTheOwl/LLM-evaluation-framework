"""Tests for Stage 1: Deterministic evaluation."""


import pytest

from llm_evals.models import Assertion, AssertionType, EvalCase, EvalSuite
from llm_evals.stages.deterministic import DeterministicStage


@pytest.fixture
def stage():
    return DeterministicStage()


@pytest.fixture
def suite():
    return EvalSuite(name="test", model="mock")


def make_case(**kwargs) -> EvalCase:
    defaults = {"id": "test-case", "prompt": "test"}
    defaults.update(kwargs)
    return EvalCase(**defaults)


class TestExactMatch:
    def test_pass(self, stage, suite):
        case = make_case(
            assertions=[Assertion(type=AssertionType.EXACT_MATCH, value="hello world")]
        )
        result = stage.evaluate(case, "hello world", suite)
        assert result.passed
        assert result.score == 1.0

    def test_fail(self, stage, suite):
        case = make_case(assertions=[Assertion(type=AssertionType.EXACT_MATCH, value="hello")])
        result = stage.evaluate(case, "hello world", suite)
        assert not result.passed

    def test_whitespace_trimmed(self, stage, suite):
        case = make_case(assertions=[Assertion(type=AssertionType.EXACT_MATCH, value="hello")])
        result = stage.evaluate(case, "  hello  ", suite)
        assert result.passed


class TestContains:
    def test_pass(self, stage, suite):
        case = make_case(assertions=[Assertion(type=AssertionType.CONTAINS, value="world")])
        result = stage.evaluate(case, "hello world", suite)
        assert result.passed

    def test_case_insensitive(self, stage, suite):
        case = make_case(assertions=[Assertion(type=AssertionType.CONTAINS, value="HELLO")])
        result = stage.evaluate(case, "hello world", suite)
        assert result.passed

    def test_fail(self, stage, suite):
        case = make_case(assertions=[Assertion(type=AssertionType.CONTAINS, value="xyz")])
        result = stage.evaluate(case, "hello world", suite)
        assert not result.passed


class TestNotContains:
    def test_pass(self, stage, suite):
        case = make_case(assertions=[Assertion(type=AssertionType.NOT_CONTAINS, value="xyz")])
        result = stage.evaluate(case, "hello world", suite)
        assert result.passed

    def test_fail(self, stage, suite):
        case = make_case(assertions=[Assertion(type=AssertionType.NOT_CONTAINS, value="hello")])
        result = stage.evaluate(case, "hello world", suite)
        assert not result.passed


class TestRegex:
    def test_pass(self, stage, suite):
        case = make_case(assertions=[Assertion(type=AssertionType.REGEX, value=r"\d+-\d+ days")])
        result = stage.evaluate(case, "You'll receive it in 3-5 days", suite)
        assert result.passed

    def test_fail(self, stage, suite):
        case = make_case(assertions=[Assertion(type=AssertionType.REGEX, value=r"^\d+$")])
        result = stage.evaluate(case, "not a number", suite)
        assert not result.passed

    def test_invalid_regex(self, stage, suite):
        case = make_case(assertions=[Assertion(type=AssertionType.REGEX, value="[invalid")])
        result = stage.evaluate(case, "test", suite)
        assert not result.passed
        assert "Invalid regex" in result.assertion_results[0].message


class TestJsonSchema:
    def test_pass(self, stage, suite):
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        case = make_case(assertions=[Assertion(type=AssertionType.JSON_SCHEMA, value=schema)])
        result = stage.evaluate(case, '{"name": "test"}', suite)
        assert result.passed

    def test_fail_invalid_json(self, stage, suite):
        schema = {"type": "object"}
        case = make_case(assertions=[Assertion(type=AssertionType.JSON_SCHEMA, value=schema)])
        result = stage.evaluate(case, "not json", suite)
        assert not result.passed

    def test_fail_schema_mismatch(self, stage, suite):
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        case = make_case(assertions=[Assertion(type=AssertionType.JSON_SCHEMA, value=schema)])
        result = stage.evaluate(case, '{"age": 25}', suite)
        assert not result.passed


class TestLength:
    def test_max_pass(self, stage, suite):
        case = make_case(assertions=[Assertion(type=AssertionType.MAX_LENGTH, value=10)])
        result = stage.evaluate(case, "short", suite)
        assert result.passed

    def test_max_fail(self, stage, suite):
        case = make_case(assertions=[Assertion(type=AssertionType.MAX_LENGTH, value=5)])
        result = stage.evaluate(case, "too long text", suite)
        assert not result.passed

    def test_min_pass(self, stage, suite):
        case = make_case(assertions=[Assertion(type=AssertionType.MIN_LENGTH, value=3)])
        result = stage.evaluate(case, "hello", suite)
        assert result.passed

    def test_min_fail(self, stage, suite):
        case = make_case(assertions=[Assertion(type=AssertionType.MIN_LENGTH, value=100)])
        result = stage.evaluate(case, "short", suite)
        assert not result.passed


class TestStartsWith:
    def test_pass(self, stage, suite):
        case = make_case(assertions=[Assertion(type=AssertionType.STARTS_WITH, value="Hello")])
        result = stage.evaluate(case, "Hello, world!", suite)
        assert result.passed

    def test_fail(self, stage, suite):
        case = make_case(assertions=[Assertion(type=AssertionType.STARTS_WITH, value="World")])
        result = stage.evaluate(case, "Hello, world!", suite)
        assert not result.passed


class TestScoring:
    def test_weighted_scoring(self, stage, suite):
        case = make_case(assertions=[
            Assertion(type=AssertionType.CONTAINS, value="yes", weight=3.0),
            Assertion(type=AssertionType.CONTAINS, value="no", weight=1.0),
        ])
        result = stage.evaluate(case, "yes indeed", suite)
        # "yes" passes (weight 3), "no" fails (weight 1) → score = 3/4 = 0.75
        assert result.score == pytest.approx(0.75)
        assert not result.passed  # not ALL passed

    def test_no_assertions(self, stage, suite):
        case = make_case(assertions=[])
        result = stage.evaluate(case, "anything", suite)
        assert result.passed
        assert result.score == 1.0

    def test_all_pass(self, stage, suite):
        case = make_case(assertions=[
            Assertion(type=AssertionType.CONTAINS, value="hello"),
            Assertion(type=AssertionType.MAX_LENGTH, value=100),
        ])
        result = stage.evaluate(case, "hello world", suite)
        assert result.passed
        assert result.score == 1.0
