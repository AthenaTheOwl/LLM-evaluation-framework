"""Quickstart fixture tests."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from llm_evals.cli import app


def test_quickstart_example_validates():
    result = CliRunner().invoke(app, ["validate", "examples/quickstart_eval"])

    assert result.exit_code == 0
    assert result.output == f"OK {Path('examples/quickstart_eval/suite.yaml')}\n"


def test_quickstart_example_writes_report(tmp_path):
    result = CliRunner().invoke(
        app,
        [
            "run",
            "examples/quickstart_eval",
            "--stage",
            "deterministic",
            "--provider",
            "mock",
            "--output",
            "json",
            "--output-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    report = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    assert report["suite_name"] == "quickstart-smoke"
    assert report["passed"] is True
    assert report["case_results"][0]["case_id"] == "mock-response-smoke"


def test_checked_in_quickstart_report_matches_fixture():
    report_path = Path("examples/reports/quickstart-smoke.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["run_id"] == "quickstart"
    assert report["aggregate_scores"] == {"deterministic": 1.0}
    assert report["stage_passed"] == {"deterministic": True}
