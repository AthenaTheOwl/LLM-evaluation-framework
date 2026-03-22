"""JSON report generation and loading."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Union

from llm_evals.models import EvalSuiteResult


def save_json_report(result: EvalSuiteResult, output_dir: Union[str, Path]) -> Path:
    """Save eval results as a JSON file. Returns the output path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{result.suite_name}_{result.run_id}.json"
    output_path = output_dir / filename

    # Also save as results.json for CI consumption
    canonical_path = output_dir / "results.json"

    data = result.model_dump(mode="json")

    for path in [output_path, canonical_path]:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    return output_path


def load_json_report(path: Union[str, Path]) -> EvalSuiteResult:
    """Load eval results from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return EvalSuiteResult(**data)
