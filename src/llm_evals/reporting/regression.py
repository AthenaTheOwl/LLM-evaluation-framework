"""Baseline comparison and regression detection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from llm_evals.models import EvalSuiteResult, RegressionReport

BASELINES_DIR = Path("baselines")


def score_spread(attempt_scores: list[dict[str, float]]) -> dict[str, float]:
    """Per-metric max minus min across attempts. A crude, outlier-sensitive measure of noise."""
    metrics = {m for scores in attempt_scores for m in scores}
    spread = {}
    for m in sorted(metrics):
        values = [scores[m] for scores in attempt_scores if m in scores]
        spread[m] = max(values) - min(values)
    return spread


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
        **(
            {"attempt_scores": result.attempt_scores, "spread": score_spread(result.attempt_scores)}
            if len(result.attempt_scores) > 1
            else {}
        ),
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
    baseline_spread = baseline.get("spread")
    noise_aware = bool(baseline_spread)
    current_spread = score_spread(result.attempt_scores) if len(result.attempt_scores) > 1 else {}
    deltas: dict[str, float] = {}
    regressions: list[str] = []
    verdicts: dict[str, str] = {}
    base_spreads: dict[str, float] = {}
    cur_spreads: dict[str, float] = {}

    for metric, current_score in result.aggregate_scores.items():
        baseline_score = baseline_scores.get(metric)
        if baseline_score is None:
            continue
        delta = current_score - baseline_score
        deltas[metric] = delta
        if not noise_aware:
            if delta < -tolerance:
                regressions.append(metric)
            continue
        b = baseline_spread.get(metric, 0.0)
        c = current_spread.get(metric, 0.0)
        base_spreads[metric], cur_spreads[metric] = b, c
        if delta < -(tolerance + max(b, c)):
            regressions.append(metric)
            verdicts[metric] = "regression"
        elif delta < 0:
            verdicts[metric] = "within_noise"
        else:
            verdicts[metric] = "improvement" if delta > 0 else "unchanged"

    return RegressionReport(
        baseline_run_id=baseline.get("run_id", "unknown"),
        score_deltas=deltas,
        regressions=regressions,
        passed=len(regressions) == 0,
        baseline_spread=base_spreads,
        current_spread=cur_spreads,
        verdicts=verdicts,
    )
