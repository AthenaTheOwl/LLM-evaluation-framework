"""Anthropic (Claude) LLM provider."""

from __future__ import annotations

import os
from typing import Optional

from llm_evals.providers.base import LLMProvider

DEFAULT_MODEL = "claude-sonnet-4-20250514"


class AnthropicProvider(LLMProvider):
    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self._api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not set. Pass api_key or set the environment variable."
            )
        import anthropic

        self._client = anthropic.Anthropic(api_key=self._api_key)

    @property
    def name(self) -> str:
        return "anthropic"

    def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
    ) -> str:
        model = model or DEFAULT_MODEL
        messages = [{"role": "user", "content": prompt}]

        kwargs: dict = {
            "model": model,
            "max_tokens": 4096,
            "messages": messages,
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        response = self._client.messages.create(**kwargs)
        return response.content[0].text
