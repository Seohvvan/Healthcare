"""Thin, injectable wrappers around the provider SDKs (Anthropic, Gemini).

Every agent function receives a client instance instead of creating one, so
tests can pass a fake object exposing the same `structured` / `text` surface
with no network access and no credentials. Credentials are resolved by each SDK
itself (environment variable or `ant auth` profile) — never hardcode a key.

`LLMClient` (Anthropic) is the reference implementation and the default path;
`GeminiClient` implements the identical surface so a provider swap needs no
change in any agent or pipeline code. Pick one with `create_llm(models)`.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

from trialmatch.config import ModelConfig

__all__ = ["GeminiClient", "LLMClient", "SupportsLLM", "create_llm"]


@runtime_checkable
class SupportsLLM(Protocol):
    """Structural type every provider client (and every test fake) satisfies."""

    def structured(
        self,
        *,
        model: str,
        system: str,
        user: str,
        schema: type[BaseModel],
        max_tokens: int = 4096,
        cache_system: bool = False,
    ) -> BaseModel: ...

    def text(self, *, model: str, system: str, user: str, max_tokens: int = 2048) -> str: ...


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


class GeminiClient:
    """Google Gemini facade with the exact same surface as `LLMClient`.

    `client` may be a `google.genai.Client` instance or any object exposing the
    same `models.generate_content` surface. When omitted, a real client is
    created lazily on first use, so importing this module needs neither the SDK
    nor credentials; the SDK reads `GOOGLE_API_KEY` / `GEMINI_API_KEY` from the
    environment — never hardcode a key.
    """

    def __init__(self, client: Any | None = None) -> None:
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = self._genai().Client()
        return self._client

    @staticmethod
    def _genai() -> Any:
        """Import `google.genai` lazily with an actionable error message."""
        try:
            from google import genai  # imported lazily: no import-time SDK requirement
        except ImportError as exc:  # pragma: no cover - dependency is a core one
            raise RuntimeError(
                "the Gemini provider needs the `google-genai` package; run `uv sync`"
            ) from exc
        return genai

    @staticmethod
    def _types() -> Any:
        try:
            from google.genai import types
        except ImportError as exc:  # pragma: no cover - dependency is a core one
            raise RuntimeError(
                "the Gemini provider needs the `google-genai` package; run `uv sync`"
            ) from exc
        return types

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
        """Return a validated `schema` instance via Gemini's JSON response schema.

        `cache_system` is accepted for interface parity and ignored: Gemini 2.5
        caches long repeated prefixes implicitly, so there is no explicit cache
        breakpoint to set (contrast `LLMClient.structured`, design.md §4).
        """
        del cache_system  # interface parity: Gemini 2.5 caches implicitly
        types = self._types()
        response = self.client.models.generate_content(
            model=model,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system,
                response_mime_type="application/json",
                response_schema=schema,
                max_output_tokens=max_tokens,
                # Greedy decoding: criterion verdicts must not flip between
                # otherwise-identical runs (the masked-field A/B measures label
                # agreement, and sampling noise reads as disagreement).
                temperature=0.0,
            ),
        )

        # The SDK usually validates for us; fall back to the raw JSON body when
        # it hands back a dict (or nothing) instead of the schema instance.
        parsed = getattr(response, "parsed", None)
        if isinstance(parsed, schema):
            return parsed
        raw = getattr(response, "text", None)
        if raw:
            return schema.model_validate_json(raw)
        raise RuntimeError(
            f"structured call to {model} returned neither a parsed {schema.__name__} "
            "nor any text; the response did not satisfy the schema"
        )

    def text(
        self,
        *,
        model: str,
        system: str,
        user: str,
        max_tokens: int = 2048,
    ) -> str:
        """Return the text of a plain completion (empty string when blocked)."""
        types = self._types()
        response = self.client.models.generate_content(
            model=model,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system,
                max_output_tokens=max_tokens,
                temperature=0.0,
            ),
        )
        return response.text or ""


def create_llm(models: ModelConfig) -> SupportsLLM:
    """Build the client matching `models.provider`.

    Both clients create their SDK object lazily, so this stays free of network
    and credential requirements until an agent actually issues a call.
    """
    if models.provider == "anthropic":
        return LLMClient()
    if models.provider == "gemini":
        return GeminiClient()
    raise ValueError(
        f"unknown LLM provider {models.provider!r}; expected 'anthropic' or 'gemini'"
    )
