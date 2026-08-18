"""Thin, injectable wrapper around the Anthropic SDK.

Every agent function receives an `LLMClient` instance instead of creating one,
so tests can pass a fake object exposing the same `structured` / `text` surface
with no network access and no credentials. Credentials are resolved by the SDK
itself (environment variable or `ant auth` profile) — never hardcode a key.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class LLMClient:
    """Minimal Anthropic facade: structured (schema-constrained) and plain text.

    `client` may be an `anthropic.Anthropic` instance or any object exposing the
    same `messages.parse` / `messages.create` surface. When omitted, a real
    client is created lazily on first use so importing this module never
    requires credentials.
    """

    def __init__(self, client: Any | None = None) -> None:
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            import anthropic  # imported lazily: keeps import-time side effects out

            self._client = anthropic.Anthropic()
        return self._client

    def structured(
        self,
        *,
        model: str,
        system: str,
        user: str,
        schema: type[BaseModel],
        max_tokens: int = 4096,
        cache_system: bool = False,
    ) -> BaseModel:
        """Return a validated `schema` instance via the structured-outputs API.

        `cache_system=True` marks the system prompt as a prompt-cache breakpoint,
        which is worth it for the per-trial criteria prompts reused across
        patients (design.md §4).
        """
        system_param: Any = system
        if cache_system:
            system_param = [
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        response = self.client.messages.parse(
            model=model,
            max_tokens=max_tokens,
            system=system_param,
            messages=[{"role": "user", "content": user}],
            output_format=schema,
        )
        parsed = getattr(response, "parsed_output", None)
        if parsed is None:
            raise RuntimeError(
                f"structured call to {model} returned no parsed_output for "
                f"schema {schema.__name__}; the response did not satisfy the schema"
            )
        return parsed

    def text(
        self,
        *,
        model: str,
        system: str,
        user: str,
        max_tokens: int = 2048,
    ) -> str:
        """Return the concatenated text blocks of a plain completion."""
        response = self.client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )
