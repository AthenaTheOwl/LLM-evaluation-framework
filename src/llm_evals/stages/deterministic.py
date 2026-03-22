"""Stage 1: Deterministic evaluation — exact match, regex, JSON schema, format checks."""

from __future__ import annotations

import json
import re

import jsonschema

from llm_evals.models import (
    Assertion,
    AssertionResult,
    AssertionType,
    DeterministicResult,
    EvalCase,
    EvalSuite,
)
from llm_evals.stages.base import EvalStage


class DeterministicStage(EvalStage):
    @property
    def name(self) -> str:
        return "deterministic"

    def evaluate(self, case: EvalCase, model_output: str, suite: EvalSuite) -> DeterministicResult:
        if not case.assertions:
            return DeterministicResult(score=1.0, passed=True, assertion_results=[])

        results = [_check_assertion(a, model_output) for a in case.assertions]

        total_weight = sum(r.weight for r in results)
        if total_weight == 0:
            score = 1.0
        else:
            score = sum(r.weight for r in results if r.passed) / total_weight

        return DeterministicResult(
            score=score,
            passed=all(r.passed for r in results),
            assertion_results=results,
        )


def _check_assertion(assertion: Assertion, output: str) -> AssertionResult:
    """Run a single assertion against the model output."""
    checkers = {
        AssertionType.EXACT_MATCH: _check_exact_match,
        AssertionType.CONTAINS: _check_contains,
        AssertionType.NOT_CONTAINS: _check_not_contains,
        AssertionType.REGEX: _check_regex,
        AssertionType.JSON_SCHEMA: _check_json_schema,
        AssertionType.MAX_LENGTH: _check_max_length,
        AssertionType.MIN_LENGTH: _check_min_length,
        AssertionType.STARTS_WITH: _check_starts_with,
    }

    checker = checkers.get(assertion.type)
    if checker is None:
        return AssertionResult(
            type=assertion.type,
            value=assertion.value,
            weight=assertion.weight,
            passed=False,
            message=f"Unknown assertion type: {assertion.type}",
        )

    return checker(assertion, output)


def _result(
    assertion: Assertion,
    passed: bool,
    message: str = "",
    actual: str | None = None,
) -> AssertionResult:
    return AssertionResult(
        type=assertion.type,
        value=assertion.value,
        weight=assertion.weight,
        passed=passed,
        actual_value=actual,
        message=message,
    )


def _check_exact_match(assertion: Assertion, output: str) -> AssertionResult:
    expected = str(assertion.value).strip()
    actual = output.strip()
    passed = actual == expected
    msg = "" if passed else f"Expected exact match: '{expected}'"
    return _result(assertion, passed, msg, actual)


def _check_contains(assertion: Assertion, output: str) -> AssertionResult:
    target = str(assertion.value)
    passed = target.lower() in output.lower()
    msg = "" if passed else f"Output does not contain: '{target}'"
    return _result(assertion, passed, msg)


def _check_not_contains(assertion: Assertion, output: str) -> AssertionResult:
    target = str(assertion.value)
    passed = target.lower() not in output.lower()
    msg = "" if passed else f"Output should not contain: '{target}'"
    return _result(assertion, passed, msg)


def _check_regex(assertion: Assertion, output: str) -> AssertionResult:
    pattern = str(assertion.value)
    try:
        passed = bool(re.search(pattern, output))
        msg = "" if passed else f"No match for pattern: '{pattern}'"
    except re.error as e:
        passed = False
        msg = f"Invalid regex pattern: {e}"
    return _result(assertion, passed, msg)


def _check_json_schema(assertion: Assertion, output: str) -> AssertionResult:
    schema = assertion.value
    if not isinstance(schema, dict):
        return _result(assertion, False, "JSON schema value must be a dict")

    try:
        parsed = json.loads(output)
    except json.JSONDecodeError as e:
        return _result(assertion, False, f"Output is not valid JSON: {e}")

    try:
        jsonschema.validate(parsed, schema)
        return _result(assertion, True)
    except jsonschema.ValidationError as e:
        return _result(assertion, False, f"JSON schema validation failed: {e.message}")


def _check_max_length(assertion: Assertion, output: str) -> AssertionResult:
    max_len = int(assertion.value)
    actual_len = len(output)
    passed = actual_len <= max_len
    msg = "" if passed else f"Output length {actual_len} exceeds max {max_len}"
    return _result(assertion, passed, msg, str(actual_len))


def _check_min_length(assertion: Assertion, output: str) -> AssertionResult:
    min_len = int(assertion.value)
    actual_len = len(output)
    passed = actual_len >= min_len
    msg = "" if passed else f"Output length {actual_len} below min {min_len}"
    return _result(assertion, passed, msg, str(actual_len))


def _check_starts_with(assertion: Assertion, output: str) -> AssertionResult:
    prefix = str(assertion.value)
    passed = output.strip().startswith(prefix)
    msg = "" if passed else f"Output does not start with: '{prefix}'"
    return _result(assertion, passed, msg)
