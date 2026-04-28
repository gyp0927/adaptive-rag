"""Unified LLM client supporting OpenAI and Anthropic formats."""

import json
from typing import Any

import httpx

from adaptive_rag.core.config import get_settings
from adaptive_rag.core.logging import get_logger

logger = get_logger(__name__)


class LLMClient:
    """Unified LLM client.

    Supports:
    - OpenAI format (api.openai.com, api.deepseek.com, etc.)
    - Anthropic format (api.kimi.com/coding, etc.)
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self._openai_client: Any = None
        self._anthropic_client: Any = None

    def is_anthropic_format(self) -> bool:
        """Detect if the endpoint uses Anthropic format."""
        base = self.settings.LLM_BASE_URL.lower()
        # Kimi Code uses Anthropic format
        if "kimi.com" in base or "kimi.ai" in base:
            return True
        return False

    def _get_openai_client(self) -> Any:
        """Lazy init OpenAI client."""
        if self._openai_client is None:
            import openai
            self._openai_client = openai.AsyncOpenAI(
                api_key=self.settings.LLM_API_KEY,
                base_url=self.settings.LLM_BASE_URL,
                timeout=self.settings.LLM_TIMEOUT_SECONDS,
            )
        return self._openai_client

    def _get_anthropic_client(self) -> Any:
        """Lazy init Anthropic client."""
        if self._anthropic_client is None:
            import anthropic
            # Anthropic SDK auto-appends /v1, so remove trailing /v1 from base_url
            base_url = self.settings.LLM_BASE_URL.rstrip("/")
            if base_url.endswith("/v1"):
                base_url = base_url[:-3]
            self._anthropic_client = anthropic.AsyncAnthropic(
                api_key=self.settings.LLM_API_KEY,
                base_url=base_url,
                timeout=self.settings.LLM_TIMEOUT_SECONDS,
            )
        return self._anthropic_client

    async def complete(
        self,
        prompt: str,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        """Send a completion request.

        Args:
            prompt: The prompt text.
            model: Model name (default from config).
            max_tokens: Max output tokens.
            temperature: Sampling temperature.
            response_format: JSON schema for structured output (OpenAI only).

        Returns:
            Generated text response.
        """
        model = model or self.settings.COMPRESSION_MODEL
        max_tokens = max_tokens or self.settings.COMPRESSION_MAX_TOKENS
        temperature = temperature if temperature is not None else self.settings.LLM_TEMPERATURE

        if self.is_anthropic_format():
            return await self._complete_anthropic(prompt, model, max_tokens, temperature)
        else:
            return await self._complete_openai(prompt, model, max_tokens, temperature, response_format)

    async def _complete_openai(
        self,
        prompt: str,
        model: str,
        max_tokens: int,
        temperature: float,
        response_format: dict[str, Any] | None,
    ) -> str:
        """OpenAI-format completion."""
        client = self._get_openai_client()

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if response_format:
            kwargs["response_format"] = response_format

        response = await client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

    async def _complete_anthropic(
        self,
        prompt: str,
        model: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Anthropic-format completion (Kimi Code, etc.)."""
        client = self._get_anthropic_client()

        response = await client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        # Extract text from content blocks
        texts = []
        for block in response.content:
            if hasattr(block, "text"):
                texts.append(block.text)
        return "\n".join(texts)
