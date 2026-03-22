"""Abstract base class for evaluation stages."""

from __future__ import annotations

from abc import ABC, abstractmethod

from llm_evals.models import EvalCase, EvalSuite, StageResult


class EvalStage(ABC):
    """Base class for all evaluation stages."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Stage identifier (e.g., 'deterministic', 'persona', 'judge')."""
        ...

    @abstractmethod
    def evaluate(self, case: EvalCase, model_output: str, suite: EvalSuite) -> StageResult:
        """Evaluate a single case's model output. Returns a StageResult."""
        ...
