"""YAML configuration loading and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import yaml

from llm_evals.models import EvalSuite


def load_suite(path: Union[str, Path]) -> EvalSuite:
    """Load an eval suite from a YAML file or directory.

    If path is a directory, looks for suite.yaml inside it.
    Case entries that are strings are treated as file paths relative to the suite file.
    """
    path = Path(path)

    if path.is_dir():
        suite_file = path / "suite.yaml"
        if not suite_file.exists():
            suite_file = path / "suite.yml"
        if not suite_file.exists():
            raise FileNotFoundError(f"No suite.yaml found in {path}")
        path = suite_file

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raise ValueError(f"Empty config file: {path}")

    raw_cases = raw.pop("cases", [])
    resolved_cases = _resolve_cases(raw_cases, path.parent)
    raw["cases"] = resolved_cases

    return EvalSuite(**raw)


def _resolve_cases(raw_cases: list, base_dir: Path) -> list[dict]:
    """Resolve case entries — inline dicts pass through, strings load from file."""
    resolved = []
    for entry in raw_cases:
        if isinstance(entry, str):
            case_path = base_dir / entry
            if not case_path.exists():
                raise FileNotFoundError(f"Case file not found: {case_path}")
            with open(case_path, "r", encoding="utf-8") as f:
                case_data = yaml.safe_load(f)
            if case_data is None:
                raise ValueError(f"Empty case file: {case_path}")
            resolved.append(case_data)
        elif isinstance(entry, dict):
            resolved.append(entry)
        else:
            raise ValueError(f"Invalid case entry type: {type(entry)}")
    return resolved


def validate_suite(path: Union[str, Path]) -> list[str]:
    """Validate a suite config and return a list of issues (empty = valid)."""
    issues = []
    try:
        suite = load_suite(path)
    except Exception as e:
        return [str(e)]

    if not suite.cases:
        issues.append("Suite has no test cases")

    for case in suite.cases:
        if not case.assertions and "deterministic" in suite.stages:
            issues.append(f"Case '{case.id}' has no assertions but deterministic stage is enabled")

    if "persona" in suite.stages and not suite.personas:
        issues.append("Persona stage enabled but no personas defined")

    if "judge" in suite.stages and suite.judge is None:
        issues.append("Judge stage enabled but no judge config defined")

    return issues


def discover_suites(directory: Union[str, Path]) -> list[Path]:
    """Find all suite.yaml files in a directory tree."""
    directory = Path(directory)
    suites = []
    for suite_file in sorted(directory.rglob("suite.yaml")):
        suites.append(suite_file)
    for suite_file in sorted(directory.rglob("suite.yml")):
        if suite_file not in suites:
            suites.append(suite_file)
    return suites
