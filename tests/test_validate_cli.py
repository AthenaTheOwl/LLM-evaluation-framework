"""Tests for the validate CLI command."""

from pathlib import Path

from typer.testing import CliRunner

from llm_evals.cli import app


def test_validate_default_discovers_real_fixture_suites():
    result = CliRunner().invoke(app, ["validate"])

    expected_paths = [
        Path("eval_suites/ad_campaign/suite.yaml"),
        Path("eval_suites/code_reviewer/suite.yaml"),
        Path("eval_suites/content_recommendation/suite.yaml"),
        Path("eval_suites/customer_support/suite.yaml"),
        Path("eval_suites/food_delivery/suite.yaml"),
        Path("eval_suites/healthcare/suite.yaml"),
    ]

    assert result.exit_code == 0
    assert result.output == "".join(f"OK {path}\n" for path in expected_paths)


def test_validate_real_fixture_suite_passes():
    suite_path = Path("eval_suites/customer_support/suite.yaml")

    result = CliRunner().invoke(app, ["validate", "--suite", str(suite_path)])

    assert result.exit_code == 0
    assert result.output == f"OK {suite_path}\n"


def test_validate_broken_yaml_fails_with_reason(tmp_path):
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text("name: [bad\n  broken: :\n", encoding="utf-8")

    result = CliRunner().invoke(app, ["validate", "--suite", str(suite_path)])

    assert result.exit_code == 1
    assert result.output.startswith(f"FAIL {suite_path}: while parsing a flow sequence")
    assert "expected ',' or ']'" in result.output
    assert "Traceback" not in result.output


def test_validate_missing_suite_exits_two_without_traceback(tmp_path):
    suite_path = tmp_path / "missing-suite.yaml"

    result = CliRunner().invoke(app, ["validate", "--suite", str(suite_path)])

    assert result.exit_code == 2
    assert result.output == f"ERROR {suite_path}: path does not exist\n"
    assert "Traceback" not in result.output
