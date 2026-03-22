"""Abstract base class for LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional


class LLMProvider(ABC):
    """Interface for LLM API providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider identifier (e.g., 'anthropic', 'openai', 'mock')."""
        ...

    @abstractmethod
    def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
    ) -> str:
        """Send a prompt and return the model's text response."""
        ...


def get_provider(name: str) -> LLMProvider:
    """Factory to instantiate a provider by name."""
    if name == "anthropic":
        from llm_evals.providers.anthropic import AnthropicProvider
        return AnthropicProvider()
    elif name == "openai":
        from llm_evals.providers.openai import OpenAIProvider
        return OpenAIProvider()
    elif name == "mock":
        from llm_evals.providers.mock import MockProvider
        return MockProvider()
    else:
        raise ValueError(f"Unknown provider: {name}. Choose from: anthropic, openai, mock")
