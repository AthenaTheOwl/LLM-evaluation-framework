"""Stage 2: Persona-based evaluation — role-specific rubric assessment."""

from __future__ import annotations

import json
import logging
from typing import Optional

from llm_evals.models import (
    EvalCase,
    EvalSuite,
    Persona,
    PersonaResult,
    PersonaScore,
)
from llm_evals.providers.base import LLMProvider
from llm_evals.stages.base import EvalStage

logger = logging.getLogger(__name__)


class PersonaStage(EvalStage):
    def __init__(self, judge_provider: LLMProvider, judge_model: Optional[str] = None):
        self._provider = judge_provider
        self._judge_model = judge_model

    @property
    def name(self) -> str:
        return "persona"

    def evaluate(self, case: EvalCase, model_output: str, suite: EvalSuite) -> PersonaResult:
        if not suite.personas:
            return PersonaResult(score=0.0, passed=False, details={"error": "No personas defined"})

        persona_scores: dict[str, PersonaScore] = {}
        for persona in suite.personas:
            score = self._evaluate_persona(persona, case, model_output)
            persona_scores[persona.name] = score

        # Weighted average across personas
        total_weight = sum(p.weight for p in suite.personas)
        if total_weight == 0:
            aggregate = 0.0
        else:
            aggregate = sum(
                persona_scores[p.name].overall_score * p.weight for p in suite.personas
            ) / total_weight

        threshold = suite.thresholds.persona_min_score
        return PersonaResult(
            score=aggregate,
            passed=aggregate >= threshold,
            persona_scores=persona_scores,
        )

    def _evaluate_persona(
        self, persona: Persona, case: EvalCase, model_output: str
    ) -> PersonaScore:
        prompt = _build_persona_prompt(persona, case, model_output)
        system = _build_persona_system(persona)

        try:
            raw = self._provider.complete(prompt, system_prompt=system, model=self._judge_model)
            return _parse_persona_response(raw, persona)
        except Exception as e:
            logger.warning(f"Persona '{persona.name}' evaluation failed: {e}")
            return PersonaScore(
                persona_name=persona.name,
                overall_score=0.0,
                reasoning=f"Evaluation failed: {e}",
            )


def _build_persona_system(persona: Persona) -> str:
    dimensions = "\n".join(
        f"- {d.name}: {d.description} (score {d.scale_min}-{d.scale_max}, weight {d.weight})"
        for d in persona.rubric
    )
    return f"""{persona.description}

You are evaluating an AI model's response. Score each dimension below on the specified scale.

Evaluation dimensions:
{dimensions}

Respond in JSON format:
{{
  "dimension_scores": {{"dimension_name": score, ...}},
  "overall_score": <weighted average>,
  "reasoning": "<brief explanation>"
}}

Respond ONLY with the JSON object, no other text."""


def _build_persona_prompt(persona: Persona, case: EvalCase, model_output: str) -> str:
    parts = [f"## Task\nEvaluate the following AI response as a {persona.name}.\n"]

    if case.description:
        parts.append(f"## Scenario\n{case.description}\n")

    parts.append(f"## User Prompt\n{case.prompt}\n")

    if case.reference_output:
        parts.append(f"## Reference (ideal) Response\n{case.reference_output}\n")

    parts.append(f"## Actual Model Response\n{model_output}\n")
    parts.append("## Your Evaluation\nProvide your scores and reasoning in JSON format.")

    return "\n".join(parts)


def _parse_persona_response(raw: str, persona: Persona) -> PersonaScore:
    # Extract JSON from response (handle markdown code blocks)
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    data = json.loads(text)

    dimension_scores = {}
    for dim in persona.rubric:
        score = data.get("dimension_scores", {}).get(dim.name, 0.0)
        score = max(dim.scale_min, min(dim.scale_max, float(score)))
        dimension_scores[dim.name] = score

    # Compute weighted average if not provided
    overall = data.get("overall_score")
    if overall is None:
        total_w = sum(d.weight for d in persona.rubric)
        if total_w > 0:
            overall = sum(dimension_scores[d.name] * d.weight for d in persona.rubric) / total_w
        else:
            overall = 0.0
    overall = float(overall)

    return PersonaScore(
        persona_name=persona.name,
        dimension_scores=dimension_scores,
        overall_score=overall,
        reasoning=data.get("reasoning", ""),
    )
