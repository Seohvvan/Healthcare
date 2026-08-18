"""Shared test doubles. No network, no API key: every LLM call is faked here.

`FakeLLM` and `ScriptedLLM` live in `conftest.py` so every test module imports
the same implementation (`from conftest import FakeLLM`) regardless of the
pytest import mode, instead of importing across sibling test modules.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class FakeLLM:
    """Stand-in for `LLMClient`, keyed by the requested output schema."""

    def __init__(
        self,
        responses: dict[type[BaseModel], BaseModel] | None = None,
        text_response: str = "",
    ) -> None:
        self.responses = responses or {}
        self.text_response = text_response
        self.calls: list[dict[str, Any]] = []

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
        self.calls.append(
            {
                "model": model,
                "system": system,
                "user": user,
                "schema": schema,
                "max_tokens": max_tokens,
                "cache_system": cache_system,
            }
        )
        if schema not in self.responses:
            raise AssertionError(f"FakeLLM has no canned response for {schema.__name__}")
        return self.responses[schema]

    def text(self, *, model: str, system: str, user: str, max_tokens: int = 2048) -> str:
        self.calls.append({"model": model, "system": system, "user": user})
        return self.text_response


class ScriptedLLM(FakeLLM):
    """`FakeLLM` whose canned response for a schema can change between calls.

    Each schema maps to a queue; the last entry is reused once exhausted, so a
    two-round run only needs the rounds that actually differ.
    """

    def __init__(self, scripts: dict[type[BaseModel], list[BaseModel]]) -> None:
        super().__init__({})
        self.scripts = {schema: list(items) for schema, items in scripts.items()}

    def structured(self, **kwargs: Any) -> BaseModel:  # type: ignore[override]
        schema = kwargs["schema"]
        queue = self.scripts.get(schema)
        if not queue:
            raise AssertionError(f"ScriptedLLM has no canned response for {schema.__name__}")
        self.responses[schema] = queue.pop(0) if len(queue) > 1 else queue[0]
        return super().structured(**kwargs)
