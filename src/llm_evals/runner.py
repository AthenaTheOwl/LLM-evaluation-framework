"""Pipeline runner — orchestrates eval stages, collects results, checks thresholds."""

from __future__ import annotations

import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from llm_evals.models import (
    EvalCase,
    EvalCaseResult,
    EvalSuite,
    EvalSuiteResult,
    StageResult,
)
from llm_evals.providers.base import LLMProvider, get_provider
from llm_evals.stages.base import EvalStage
from llm_evals.stages.deterministic import DeterministicStage
from llm_evals.stages.judge import JudgeStage
from llm_evals.stages.persona import PersonaStage


class EvalRunner:
    """Runs an eval suite through the 3-stage pipeline."""

    def __init__(
        self,
        suite: EvalSuite,
        model_provider: Optional[LLMProvider] = None,
        judge_provider: Optional[LLMProvider] = None,
        concurrency: int = 5,
        stages_filter: Optional[list[str]] = None,
    ):
        self.suite = suite
        self.model_provider = model_provider or get_provider(suite.provider)
        self.concurrency = concurrency

        # Determine which stages to run
        requested = stages_filter or suite.stages
        self.stages: list[EvalStage] = []

        if "deterministic" in requested:
            self.stages.append(DeterministicStage())

        # For persona and judge, set up a judge provider
        if "persona" in requested or "judge" in requested:
            if judge_provider is None:
                judge_provider_name = suite.judge.provider if suite.judge else suite.provider
                judge_provider = get_provider(judge_provider_name)
            judge_model = suite.judge.model if suite.judge else None

        if "persona" in requested:
            self.stages.append(PersonaStage(judge_provider, judge_model))

        if "judge" in requested:
            self.stages.append(JudgeStage(judge_provider, judge_model))

    def run(self) -> EvalSuiteResult:
        """Run all cases through the pipeline."""
        case_results: list[EvalCaseResult] = []

        if self.concurrency > 1:
            with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
                futures = {
                    pool.submit(self._run_case, case): case for case in self.suite.cases
                }
                for future in as_completed(futures):
                    case_results.append(future.result())
        else:
            for case in self.suite.cases:
                case_results.append(self._run_case(case))

        # Sort by original case order
        case_order = {c.id: i for i, c in enumerate(self.suite.cases)}
        case_results.sort(key=lambda r: case_order.get(r.case_id, 0))

        return self._build_suite_result(case_results)

    def _run_case(self, case: EvalCase) -> EvalCaseResult:
        """Run a single case through all applicable stages."""
        start = time.time()

        # Get model output
        model_output = self.model_provider.complete(
            prompt=case.prompt,
            system_prompt=case.system_prompt,
            model=self.suite.model,
        )

        stage_results: list[StageResult] = []
        for stage in self.stages:
            result = stage.evaluate(case, model_output, self.suite)
            stage_results.append(result)

            # Fail fast: skip remaining stages if deterministic fails
            if self.suite.fail_fast and stage.name == "deterministic" and not result.passed:
                break

        latency_ms = (time.time() - start) * 1000

        # Overall score: average of stage scores
        if stage_results:
            overall = sum(r.score for r in stage_results) / len(stage_results)
        else:
            overall = 0.0

        return EvalCaseResult(
            case_id=case.id,
            model_output=model_output,
            stage_results=stage_results,
            overall_score=overall,
            passed=all(r.passed for r in stage_results),
            latency_ms=latency_ms,
        )

    def _build_suite_result(self, case_results: list[EvalCaseResult]) -> EvalSuiteResult:
        """Aggregate case results into a suite result with threshold checks."""
        # Compute per-stage aggregate scores
        aggregate: dict[str, list[float]] = {}
        for cr in case_results:
            for sr in cr.stage_results:
                aggregate.setdefault(sr.stage, []).append(sr.score)

        aggregate_scores = {
            stage: sum(scores) / len(scores) if scores else 0.0
            for stage, scores in aggregate.items()
        }

        pass_rate = (
            sum(1 for cr in case_results if cr.passed) / len(case_results)
            if case_results
            else 0.0
        )

        # Check thresholds using the native score scale for each stage.
        t = self.suite.thresholds
        stage_thresholds: dict[str, float] = {}
        stage_passed: dict[str, bool] = {}
        if "deterministic" in aggregate_scores:
            stage_thresholds["deterministic"] = t.deterministic_pass_rate
            stage_passed["deterministic"] = (
                aggregate_scores["deterministic"] >= t.deterministic_pass_rate
            )
        if "persona" in aggregate_scores:
            stage_thresholds["persona"] = t.persona_min_score
            stage_passed["persona"] = aggregate_scores["persona"] >= t.persona_min_score
        if "judge" in aggregate_scores:
            stage_thresholds["judge"] = t.judge_min_score
            stage_passed["judge"] = aggregate_scores["judge"] >= t.judge_min_score

        passed = all(stage_passed.values()) if stage_passed else pass_rate > 0.0

        return EvalSuiteResult(
            suite_name=self.suite.name,
            model=self.suite.model,
            run_id=str(uuid.uuid4())[:8],
            case_results=case_results,
            aggregate_scores=aggregate_scores,
            stage_thresholds=stage_thresholds,
            stage_passed=stage_passed,
            pass_rate=pass_rate,
            passed=passed,
        )
