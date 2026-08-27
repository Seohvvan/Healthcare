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

import logging
import os
import time
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

from trialmatch.config import ModelConfig

__all__ = ["GeminiClient", "LLMClient", "SupportsLLM", "create_llm", "format_usage"]

logger = logging.getLogger(__name__)

# Free-tier defaults. The Gemini free tier enforces a per-minute request quota
# and answers excess traffic with 429 RESOURCE_EXHAUSTED, so `GeminiClient`
# paces itself to `GEMINI_RPM` requests per minute and backs off on a 429
# instead of relying on the SDK's own retry (which did not save a real run).
DEFAULT_GEMINI_RPM = 10
GEMINI_RETRY_WAITS: tuple[float, ...] = (30.0, 60.0, 120.0)
_RATE_LIMIT_STATUS = 429

# usage_summary() key -> the `usage_metadata` attribute it accumulates.
_USAGE_ATTRIBUTES = {
    "prompt_tokens": "prompt_token_count",
    "output_tokens": "candidates_token_count",
    "thinking_tokens": "thoughts_token_count",
}
_USAGE_KEYS = ("calls", *_USAGE_ATTRIBUTES)


def _gemini_rpm() -> int:
    """Requests-per-minute budget from `GEMINI_RPM` (default `DEFAULT_GEMINI_RPM`).

    Unset, non-numeric and non-positive values all fall back to the default: a
    typo must not disable pacing (0 or -1 would mean "no interval") on a tier
    whose quota is the thing that ends runs.
    """
    raw = os.environ.get("GEMINI_RPM")
    try:
        rpm = int(str(raw).strip())
    except (TypeError, ValueError):
        return DEFAULT_GEMINI_RPM
    return rpm if rpm > 0 else DEFAULT_GEMINI_RPM


def format_usage(llm: Any) -> str | None:
    """One-line token report for a client that tracks usage, else `None`.

    Every runner prints the line through this one function so the format lives
    in a single place; `LLMClient` and the test fakes expose no `usage_summary`
    and simply get `None`.
    """
    summary = getattr(llm, "usage_summary", None)
    if summary is None:
        return None
    usage = summary()
    return (
        f"llm usage: {usage['calls']} calls, "
        f"{usage['prompt_tokens']} prompt + {usage['output_tokens']} output "
        f"+ {usage['thinking_tokens']} thinking tokens"
    )


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

    Every call goes through `_generate`, which makes the client free-tier
    friendly: it paces requests to `GEMINI_RPM` per minute (read once here),
    retries a 429 with growing waits, and accumulates token usage so a run can
    report what it spent. Single-threaded use is assumed — the pipeline issues
    one call at a time, so the pacing state needs no lock.
    """

    def __init__(self, client: Any | None = None) -> None:
        self._client = client
        self._min_interval = 60.0 / _gemini_rpm()
        self._last_call_at: float | None = None
        self._usage = dict.fromkeys(_USAGE_KEYS, 0)

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

    @staticmethod
    def _client_error() -> type[Exception] | None:
        """`google.genai.errors.ClientError`, or `None` when it is unavailable.

        Detection must survive an SDK-less import and an SDK that moved the
        class, so a failed import is not an error here: `_is_rate_limited`
        falls back to matching the exception's type name.
        """
        try:
            from google.genai.errors import ClientError
        except Exception:  # noqa: BLE001 - any import failure degrades to the name check
            return None
        return ClientError

    # ----------------------------------------------------------------- #
    # free-tier plumbing: pacing, 429 backoff, usage accounting
    # ----------------------------------------------------------------- #

    def _pace(self) -> None:
        """Sleep until at least `60 / GEMINI_RPM` seconds since the last call."""
        if self._last_call_at is not None:
            remaining = self._min_interval - (time.monotonic() - self._last_call_at)
            if remaining > 0:
                time.sleep(remaining)
        self._last_call_at = time.monotonic()

    @classmethod
    def _is_rate_limited(cls, exc: BaseException) -> bool:
        """True for a Gemini 429 RESOURCE_EXHAUSTED, whatever shape it arrives in.

        The SDK's error surface has varied across versions (`status_code`,
        `code`, or only a "429 RESOURCE_EXHAUSTED ..." message), so detection is
        deliberately liberal — the cost of a false positive is one extra wait,
        the cost of a false negative is a dead run.
        """
        error_type = cls._client_error()
        if type(exc).__name__ != "ClientError" and not (
            error_type is not None and isinstance(exc, error_type)
        ):
            return False
        for attribute in ("status_code", "code"):
            value = getattr(exc, attribute, None)
            try:
                if value is not None and int(value) == _RATE_LIMIT_STATUS:
                    return True
            except (TypeError, ValueError):
                continue
        return str(exc).lstrip().startswith(str(_RATE_LIMIT_STATUS))

    def _record_usage(self, response: Any) -> None:
        """Accumulate this response's token counts; a missing block still counts.

        `usage_metadata` and each of its counters may be `None` (blocked or
        cached responses); that may lower the token totals but never the call
        count — an under-reported call count would understate the quota spend.
        """
        self._usage["calls"] += 1
        metadata = getattr(response, "usage_metadata", None)
        if metadata is None:
            return
        for key, attribute in _USAGE_ATTRIBUTES.items():
            value = getattr(metadata, attribute, None)
            if isinstance(value, int):
                self._usage[key] += value

    def usage_summary(self) -> dict[str, int]:
        """Calls and tokens spent by this client so far (a copy, safe to keep)."""
        return dict(self._usage)

    def _generate(self, **kwargs: Any) -> Any:
        """One paced, 429-retried, usage-accounted `generate_content` call."""
        attempt = 0
        while True:
            self._pace()
            try:
                response = self.client.models.generate_content(**kwargs)
            except Exception as exc:  # re-raised unless it is a retryable 429
                if attempt >= len(GEMINI_RETRY_WAITS) or not self._is_rate_limited(exc):
                    raise
                wait = GEMINI_RETRY_WAITS[attempt]
                attempt += 1
                logger.warning(
                    "Gemini rate limit (429), retry %d/%d in %.0fs: %s",
                    attempt,
                    len(GEMINI_RETRY_WAITS),
                    wait,
                    exc,
                )
                time.sleep(wait)
                continue
            self._record_usage(response)
            return response

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
        response = self._generate(
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
        response = self._generate(
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
