"""Mock LLM provider for testing without API calls."""

from __future__ import annotations

import threading
from typing import Optional

from llm_evals.providers.base import LLMProvider


class MockProvider(LLMProvider):
    """Returns canned responses for testing.

    Responses can be set via the `responses` dict keyed by prompt substring,
    or a default response is returned. `sequences` maps an exact prompt to a
    list of responses returned in turn, one per call and cycling, so a repeated
    run can script a case that passes on some attempts and fails on others.
    """

    def __init__(self, default_response: str = "This is a mock response."):
        self.default_response = default_response
        self.responses: dict[str, str] = {}
        self.sequences: dict[str, list[str]] = {}
        self.call_log: list[dict] = []
        self._sequence_calls: dict[str, int] = {}
        self._lock = threading.Lock()

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

        if prompt in self.sequences and self.sequences[prompt]:
            with self._lock:
                call = self._sequence_calls.get(prompt, 0)
                self._sequence_calls[prompt] = call + 1
            sequence = self.sequences[prompt]
            return sequence[call % len(sequence)]

        for key, response in self.responses.items():
            if key in prompt:
                return response

        return self.default_response
