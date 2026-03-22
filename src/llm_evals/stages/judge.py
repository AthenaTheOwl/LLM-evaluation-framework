"""Stage 3: LLM-as-a-Judge — rubric-based evaluation with chain-of-thought and calibration."""

from __future__ import annotations

import json
import logging
from typing import Optional

from llm_evals.models import (
    EvalCase,
    EvalSuite,
    JudgeConfig,
    JudgeResult,
)
from llm_evals.providers.base import LLMProvider
from llm_evals.stages.base import EvalStage

logger = logging.getLogger(__name__)


class JudgeStage(EvalStage):
    def __init__(self, judge_provider: LLMProvider, judge_model: Optional[str] = None):
        self._provider = judge_provider
        self._judge_model = judge_model

    @property
    def name(self) -> str:
        return "judge"

    def evaluate(self, case: EvalCase, model_output: str, suite: EvalSuite) -> JudgeResult:
        judge_config = suite.judge
        if judge_config is None:
            return JudgeResult(
                score=0.0, passed=False, details={"error": "No judge config defined"}
            )

        model = self._judge_model or judge_config.model
        prompt = _build_judge_prompt(judge_config, case, model_output)
        system = _build_judge_system(judge_config)

        try:
            raw = self._provider.complete(prompt, system_prompt=system, model=model)
            result = _parse_judge_response(raw, judge_config)
        except Exception as e:
            logger.warning(f"Judge evaluation failed: {e}")
            return JudgeResult(
                score=0.0,
                passed=False,
                reasoning=f"Evaluation failed: {e}",
            )

        # Compute calibration delta if human baseline exists
        if judge_config.human_baseline:
            deltas = []
            for dim_name, human_score in judge_config.human_baseline.items():
                if dim_name in result.dimension_scores:
                    deltas.append(result.dimension_scores[dim_name] - human_score)
            if deltas:
                result.calibration_delta = sum(deltas) / len(deltas)

        threshold = suite.thresholds.judge_min_score
        result.passed = result.overall_score >= threshold
        result.score = result.overall_score

        return result


def _build_judge_system(config: JudgeConfig) -> str:
    dimensions = "\n".join(
        f"- {d.name}: {d.description} (score {d.scale_min}-{d.scale_max}, weight {d.weight})"
        for d in config.rubric
    )

    cot_instruction = ""
    if config.require_chain_of_thought:
        cot_instruction = """
First, think step-by-step about the response quality for each dimension.
Then provide your scores."""

    return f"""You are an expert AI evaluator. Your task is to assess
the quality of an AI model's response using the rubric below.
{cot_instruction}

Evaluation rubric:
{dimensions}

Respond in JSON format:
{{
  "chain_of_thought": "<your step-by-step reasoning>",
  "dimension_scores": {{"dimension_name": score, ...}},
  "overall_score": <weighted average across dimensions>,
  "confidence": <0.0 to 1.0, how confident you are in this assessment>
}}

Respond ONLY with the JSON object, no other text."""


def _build_judge_prompt(config: JudgeConfig, case: EvalCase, model_output: str) -> str:
    parts = ["## Task\nEvaluate the following AI model response.\n"]

    if case.description:
        parts.append(f"## Scenario\n{case.description}\n")

    parts.append(f"## User Prompt\n{case.prompt}\n")

    if case.reference_output:
        parts.append(f"## Reference (ideal) Response\n{case.reference_output}\n")

    parts.append(f"## Model Response to Evaluate\n{model_output}\n")
    parts.append("## Your Evaluation\nProvide your assessment in JSON format.")

    return "\n".join(parts)


def _parse_judge_response(raw: str, config: JudgeConfig) -> JudgeResult:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    data = json.loads(text)

    dimension_scores = {}
    for dim in config.rubric:
        score = data.get("dimension_scores", {}).get(dim.name, 0.0)
        score = max(dim.scale_min, min(dim.scale_max, float(score)))
        dimension_scores[dim.name] = score

    overall = data.get("overall_score")
    if overall is None:
        total_w = sum(d.weight for d in config.rubric)
        if total_w > 0:
            overall = sum(dimension_scores[d.name] * d.weight for d in config.rubric) / total_w
        else:
            overall = 0.0
    overall = float(overall)

    confidence = float(data.get("confidence", 0.5))
    confidence = max(0.0, min(1.0, confidence))

    reasoning = data.get("chain_of_thought", data.get("reasoning", ""))

    return JudgeResult(
        score=overall,
        passed=False,  # Set by caller based on threshold
        dimension_scores=dimension_scores,
        overall_score=overall,
        reasoning=reasoning,
        confidence=confidence,
    )
