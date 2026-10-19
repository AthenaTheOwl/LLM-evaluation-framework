"""Handoff evidence fixture tests."""

from __future__ import annotations

import re
from pathlib import Path

from typer.testing import CliRunner

from llm_evals.cli import app


def test_handoff_example_validates():
    result = CliRunner().invoke(app, ["validate", "examples/handoff_eval"])

    assert result.exit_code == 0
    assert result.output == f"OK {Path('examples/handoff_eval/suite.yaml')}\n"


def test_handoff_example_fails_reliability_gate_as_documented():
    result = CliRunner().invoke(
        app,
        [
            "run",
            "examples/handoff_eval",
            "--provider",
            "mock",
            "--stage",
            "deterministic",
            "--repeat",
            "3",
            "--min-pass-hat-k",
            "1.0",
        ],
    )

    assert result.exit_code == 1, result.output
    assert isinstance(result.exception, SystemExit), result.exception
    plain = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
    assert "pass^3 75% is below 100% -- exiting with code 1" in plain
    assert "pass@1 100%" in plain
    assert "pass@3 100%" in plain
    assert "pass^3 75%" in plain
    assert "failed-check-with-confident-author PFP" in plain
