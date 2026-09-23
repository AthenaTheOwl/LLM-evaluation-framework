"""pass@1 / pass@k / pass^k over repeated runs."""

import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from llm_evals.cli import app
from llm_evals.config import load_suite
from llm_evals.models import Assertion, AssertionType, EvalCase, EvalSuite
from llm_evals.providers.mock import MockProvider
from llm_evals.runner import EvalRunner

FIXTURE = Path(__file__).resolve().parents[1] / "examples" / "reliability_eval"


def _suite(*cases: EvalCase) -> EvalSuite:
    return EvalSuite(name="r", model="mock", provider="mock", stages=["deterministic"], cases=list(cases))


def _case(case_id: str, prompt: str) -> EvalCase:
    return EvalCase(id=case_id, prompt=prompt,
                    assertions=[Assertion(type=AssertionType.CONTAINS, value="ok")])


def test_single_run_has_no_reliability_block():
    provider = MockProvider(default_response="ok")
    result = EvalRunner(_suite(_case("a", "p")), model_provider=provider, concurrency=1).run()
    assert result.reliability is None


def test_flaky_case_passes_at_1_and_fails_pass_hat_k():
    provider = MockProvider(default_response="ok")
    provider.sequences["flaky"] = ["ok", "no", "ok"]
    suite = _suite(_case("steady", "steady"), _case("flaky", "flaky"))
    result = EvalRunner(suite, model_provider=provider, concurrency=2, repeats=3).run()
    rel = result.reliability
    assert rel.k == 3
    assert [c.attempts for c in rel.cases] == [[True, True, True], [True, False, True]]
    assert (rel.pass_at_1, rel.pass_at_k, rel.pass_hat_k) == (1.0, 1.0, 0.5)
    # the headline pass rate still reports attempt 1 only
    assert result.pass_rate == 1.0


def test_repeats_must_be_positive():
    with pytest.raises(ValueError):
        EvalRunner(_suite(_case("a", "p")), model_provider=MockProvider(), repeats=0)


def test_checked_in_fixture_and_gate():
    suite = load_suite(str(FIXTURE))
    assert {c.id for c in suite.cases} == {"steady-refund-answer", "flaky-refund-window"}
    runner = CliRunner()
    args = ["run", str(FIXTURE), "--stage", "deterministic", "--provider", "mock", "--repeat", "3"]
    ok = runner.invoke(app, args)
    assert ok.exit_code == 0, ok.output
    plain = re.sub(r"\x1b\[[0-9;]*m", "", ok.output)
    assert "pass^3 50%" in plain
    assert "PFP" in plain
    gated = runner.invoke(app, args + ["--min-pass-hat-k", "1.0"])
    assert gated.exit_code == 1
