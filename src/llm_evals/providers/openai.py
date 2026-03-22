"""OpenAI LLM provider."""

from __future__ import annotations

import os
from typing import Optional

from llm_evals.providers.base import LLMProvider

DEFAULT_MODEL = "gpt-4o"


class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self._api_key:
            raise ValueError(
                "OPENAI_API_KEY not set. Pass api_key or set the environment variable."
            )
        import openai

        self._client = openai.OpenAI(api_key=self._api_key)

    @property
    def name(self) -> str:
        return "openai"

    def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
    ) -> str:
        model = model or DEFAULT_MODEL
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self._client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=4096,
        )
        return response.choices[0].message.content or ""
