"""Baseline comparison and regression detection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from llm_evals.models import EvalSuiteResult, RegressionReport

BASELINES_DIR = Path("baselines")


def save_baseline(result: EvalSuiteResult, baselines_dir: Optional[Path] = None) -> Path:
    """Save current results as the baseline for future comparison."""
    base_dir = baselines_dir or BASELINES_DIR
    base_dir.mkdir(parents=True, exist_ok=True)

    path = base_dir / f"{result.suite_name}.json"
    data = {
        "run_id": result.run_id,
        "model": result.model,
        "aggregate_scores": result.aggregate_scores,
        "pass_rate": result.pass_rate,
        "timestamp": result.timestamp.isoformat(),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return path


def compare_baseline(
    result: EvalSuiteResult,
    tolerance: float = 0.05,
    baselines_dir: Optional[Path] = None,
) -> Optional[RegressionReport]:
    """Compare results against stored baseline. Returns None if no baseline exists."""
    base_dir = baselines_dir or BASELINES_DIR
    path = base_dir / f"{result.suite_name}.json"

    if not path.exists():
        return None

    with open(path, "r", encoding="utf-8") as f:
        baseline = json.load(f)

    baseline_scores = baseline.get("aggregate_scores", {})
    deltas: dict[str, float] = {}
    regressions: list[str] = []

    for metric, current_score in result.aggregate_scores.items():
        baseline_score = baseline_scores.get(metric)
        if baseline_score is not None:
            delta = current_score - baseline_score
            deltas[metric] = delta
            if delta < -tolerance:
                regressions.append(metric)

    return RegressionReport(
        baseline_run_id=baseline.get("run_id", "unknown"),
        score_deltas=deltas,
        regressions=regressions,
        passed=len(regressions) == 0,
    )
