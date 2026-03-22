"""Mock LLM provider for testing without API calls."""

from __future__ import annotations

from typing import Optional

from llm_evals.providers.base import LLMProvider


class MockProvider(LLMProvider):
    """Returns canned responses for testing.

    Responses can be set via the `responses` dict keyed by prompt substring,
    or a default response is returned.
    """

    def __init__(self, default_response: str = "This is a mock response."):
        self.default_response = default_response
        self.responses: dict[str, str] = {}
        self.call_log: list[dict] = []

    @property
    def name(self) -> str:
        return "mock"

    def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
    ) -> str:
        self.call_log.append({
            "prompt": prompt,
            "system_prompt": system_prompt,
            "model": model,
        })

        for key, response in self.responses.items():
            if key in prompt:
                return response

        return self.default_response
