"""Core data models for the llm-evals framework."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

# ── Config Models (loaded from YAML) ──────────────────────────────────────────


class AssertionType(str, Enum):
    EXACT_MATCH = "exact_match"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    REGEX = "regex"
    JSON_SCHEMA = "json_schema"
    MAX_LENGTH = "max_length"
    MIN_LENGTH = "min_length"
    STARTS_WITH = "starts_with"


class Assertion(BaseModel):
    type: AssertionType
    value: Any
    weight: float = 1.0


class RubricDimension(BaseModel):
    name: str
    description: str
    weight: float = 1.0
    scale_min: int = 1
    scale_max: int = 5


class Persona(BaseModel):
    name: str
    description: str
    rubric: list[RubricDimension]
    weight: float = 1.0


class JudgeConfig(BaseModel):
    model: str = "claude-sonnet-4-20250514"
    provider: str = "anthropic"
    rubric: list[RubricDimension] = Field(default_factory=list)
    require_chain_of_thought: bool = True
    confidence_threshold: float = 0.7
    human_baseline: Optional[dict[str, float]] = None


class QualityThresholds(BaseModel):
    deterministic_pass_rate: float = 0.9
    persona_min_score: float = 3.5
    judge_min_score: float = 3.5
    regression_tolerance: float = 0.05


class EvalCase(BaseModel):
    id: str
    description: str = ""
    prompt: str
    system_prompt: Optional[str] = None
    reference_output: Optional[str] = None
    variables: dict[str, str] = Field(default_factory=dict)
    assertions: list[Assertion] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvalSuite(BaseModel):
    name: str
    description: str = ""
    version: str = "1.0"
    model: str = "claude-sonnet-4-20250514"
    provider: str = "anthropic"
    stages: list[str] = Field(default_factory=lambda: ["deterministic", "persona", "judge"])
    fail_fast: bool = False
    personas: list[Persona] = Field(default_factory=list)
    judge: Optional[JudgeConfig] = None
    cases: list[EvalCase] = Field(default_factory=list)
    thresholds: QualityThresholds = Field(default_factory=QualityThresholds)


# ── Result Models ─────────────────────────────────────────────────────────────


class AssertionResult(BaseModel):
    type: AssertionType
    value: Any
    weight: float = 1.0
    passed: bool
    actual_value: Optional[str] = None
    message: str = ""


class StageResult(BaseModel):
    stage: str
    score: float
    passed: bool
    details: dict[str, Any] = Field(default_factory=dict)


class DeterministicResult(StageResult):
    stage: str = "deterministic"
    assertion_results: list[AssertionResult] = Field(default_factory=list)


class PersonaScore(BaseModel):
    persona_name: str
    dimension_scores: dict[str, float] = Field(default_factory=dict)
    overall_score: float = 0.0
    reasoning: str = ""


class PersonaResult(StageResult):
    stage: str = "persona"
    persona_scores: dict[str, PersonaScore] = Field(default_factory=dict)


class JudgeResult(StageResult):
    stage: str = "judge"
    dimension_scores: dict[str, float] = Field(default_factory=dict)
    overall_score: float = 0.0
    reasoning: str = ""
    confidence: float = 0.0
    calibration_delta: Optional[float] = None


class EvalCaseResult(BaseModel):
    case_id: str
    model_output: str
    stage_results: list[StageResult] = Field(default_factory=list)
    overall_score: float = 0.0
    passed: bool = False
    latency_ms: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.now)


class RegressionReport(BaseModel):
    baseline_run_id: str
    score_deltas: dict[str, float] = Field(default_factory=dict)
    regressions: list[str] = Field(default_factory=list)
    passed: bool = True


class EvalSuiteResult(BaseModel):
    suite_name: str
    model: str
    run_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    case_results: list[EvalCaseResult] = Field(default_factory=list)
    aggregate_scores: dict[str, float] = Field(default_factory=dict)
    stage_thresholds: dict[str, float] = Field(default_factory=dict)
    stage_passed: dict[str, bool] = Field(default_factory=dict)
    pass_rate: float = 0.0
    passed: bool = False
    regression: Optional[RegressionReport] = None
