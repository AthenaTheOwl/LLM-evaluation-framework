"""Tests for CLI behavior."""

from pathlib import Path

import yaml
from typer.testing import CliRunner

from llm_evals.cli import app
from llm_evals.models import EvalSuiteResult
from llm_evals.providers.mock import MockProvider


def test_provider_override_applies_to_judge(tmp_path, monkeypatch):
    cases_dir = tmp_path / "cases"
    cases_dir.mkdir()

    with open(cases_dir / "basic.yaml", "w", encoding="utf-8") as f:
        yaml.dump({"id": "case-1", "prompt": "Hello"}, f)

    suite = {
        "name": "cli-provider-override",
        "model": "claude-sonnet-4-20250514",
        "provider": "anthropic",
        "stages": ["persona"],
        "personas": [
            {
                "name": "reviewer",
                "description": "You are a reviewer.",
                "rubric": [
                    {"name": "quality", "description": "Is it good?"},
                ],
            }
        ],
        "judge": {
            "model": "claude-sonnet-4-20250514",
            "provider": "anthropic",
            "rubric": [
                {"name": "quality", "description": "Is it good?"},
            ],
        },
        "cases": ["cases/basic.yaml"],
    }
    with open(tmp_path / "suite.yaml", "w", encoding="utf-8") as f:
        yaml.dump(suite, f)

    called_providers: list[str] = []

    def fake_get_provider(name: str):
        called_providers.append(name)
        return MockProvider(
            default_response='{"dimension_scores": {"quality": 4}, "overall_score": 4}'
        )

    def fake_run(self):
        return EvalSuiteResult(
            suite_name=self.suite.name,
            model=self.suite.model,
            run_id="run12345",
            aggregate_scores={"persona": 4.0},
            stage_thresholds={"persona": 3.5},
            stage_passed={"persona": True},
            pass_rate=1.0,
            passed=True,
        )

    monkeypatch.setattr("llm_evals.providers.base.get_provider", fake_get_provider)
    monkeypatch.setattr("llm_evals.runner.get_provider", fake_get_provider)
    monkeypatch.setattr("llm_evals.runner.EvalRunner.run", fake_run)

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["run", str(Path(tmp_path)), "--stage", "persona", "--provider", "mock"],
    )

    assert result.exit_code == 0
    assert called_providers == ["mock", "mock"]


def test_run_nonexistent_path_reports_clean_error(tmp_path):
    runner = CliRunner()
    result = runner.invoke(app, ["run", str(tmp_path / "nope")])
    assert result.exit_code == 1
    assert "FAIL" in result.output
    assert "Traceback" not in result.output


def test_run_directory_without_suite_reports_clean_error(tmp_path):
    runner = CliRunner()
    result = runner.invoke(app, ["run", str(tmp_path)])
    assert result.exit_code == 1
    assert "No suite.yaml" in result.output
    assert "Traceback" not in result.output


def test_run_malformed_yaml_reports_clean_error(tmp_path):
    with open(tmp_path / "suite.yaml", "w", encoding="utf-8") as f:
        f.write("name: [bad\n  broken: :\n")
    runner = CliRunner()
    result = runner.invoke(app, ["run", str(tmp_path)])
    assert result.exit_code == 1
    assert "FAIL" in result.output
    assert "Traceback" not in result.output
