"""Gemini provider tests. No network, no SDK call, no API key: the client is faked.

`FakeGenAI` mimics the `google.genai.Client` surface used by `GeminiClient`
(`client.models.generate_content(...)`), so these tests pin the request we build
and the three response shapes we must survive.
"""

from __future__ import annotations

from typing import Any

import pytest
from conftest import FakeLLM  # shared test doubles, see tests/conftest.py

from trialmatch import cli
from trialmatch.agents.matcher import AssessmentBatch
from trialmatch.agents.profiler import ProfileExtraction
from trialmatch.config import GEMINI_MODELS, ModelConfig, Settings
from trialmatch.data import write_jsonl
from trialmatch.llm import GeminiClient, LLMClient, create_llm
from trialmatch.retrieval import BM25Index
from trialmatch.schemas import TrialRecord

NOTE = "A 62-year-old man with stage IV lung adenocarcinoma on no active treatment."

TRIAL = TrialRecord(
    nct_id="NCT0001",
    title="Osimertinib in advanced non-small cell lung cancer",
    conditions=["Non-Small Cell Lung Cancer"],
    brief_summary="A study of osimertinib in adults with advanced lung adenocarcinoma.",
    eligibility_text=(
        "Inclusion Criteria:\n\n* Adults aged 18 years or older\n\n"
        "Exclusion Criteria:\n\n* Untreated active brain metastases\n"
    ),
)


class _Response:
    """Stand-in for `google.genai.types.GenerateContentResponse`."""

    def __init__(self, parsed: Any = None, text: str | None = None) -> None:
        self.parsed = parsed
        self.text = text


class FakeGenAI:
    """Minimal stand-in for `google.genai.Client` with a recording models API."""

    def __init__(self, response: Any = None) -> None:
        self.response = response
        self.kwargs: dict[str, Any] = {}
        self.models = self

    def generate_content(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        return self.response


# --------------------------------------------------------------------------- #
# GeminiClient
# --------------------------------------------------------------------------- #


def test_gemini_structured_returns_the_parsed_object_and_builds_the_request() -> None:
    expected = ProfileExtraction(age_years=40.0)
    fake = FakeGenAI(_Response(parsed=expected))

    result = GeminiClient(fake).structured(
        model="m", system="sys", user="usr", schema=ProfileExtraction, max_tokens=123
    )

    assert result is expected
    assert fake.kwargs["model"] == "m"
    assert fake.kwargs["contents"] == "usr"
    config = fake.kwargs["config"]
    assert config.system_instruction == "sys"
    assert config.response_mime_type == "application/json"
    assert config.response_schema is ProfileExtraction
    assert config.max_output_tokens == 123


def test_gemini_structured_accepts_and_ignores_cache_system() -> None:
    """`cache_system` exists only for interface parity: Gemini caches implicitly."""
    fake = FakeGenAI(_Response(parsed=ProfileExtraction()))

    GeminiClient(fake).structured(
        model="m", system="sys", user="usr", schema=ProfileExtraction, cache_system=True
    )

    assert fake.kwargs["config"].system_instruction == "sys"


def test_gemini_structured_falls_back_to_validating_the_raw_json_text() -> None:
    fake = FakeGenAI(_Response(parsed=None, text='{"age_years": 40.0, "sex": "male"}'))

    result = GeminiClient(fake).structured(
        model="m", system="sys", user="usr", schema=ProfileExtraction
    )

    assert isinstance(result, ProfileExtraction)
    assert result.age_years == 40.0
    assert result.sex == "male"


def test_gemini_structured_falls_back_when_parsed_is_a_plain_dict() -> None:
    """The SDK hands back a dict for some schemas; the JSON body still decides."""
    fake = FakeGenAI(_Response(parsed={"age_years": 7.0}, text='{"age_years": 7.0}'))

    result = GeminiClient(fake).structured(
        model="m", system="sys", user="usr", schema=ProfileExtraction
    )

    assert isinstance(result, ProfileExtraction)
    assert result.age_years == 7.0


def test_gemini_structured_raises_when_neither_parsed_nor_text_is_usable() -> None:
    fake = FakeGenAI(_Response(parsed=None, text=None))

    with pytest.raises(RuntimeError, match="ProfileExtraction"):
        GeminiClient(fake).structured(
            model="m", system="sys", user="usr", schema=ProfileExtraction
        )


def test_gemini_text_returns_the_response_text() -> None:
    fake = FakeGenAI(_Response(text="Hello world"))

    assert GeminiClient(fake).text(model="m", system="sys", user="usr") == "Hello world"
    config = fake.kwargs["config"]
    assert config.system_instruction == "sys"
    assert config.max_output_tokens == 2048
    assert config.response_schema is None  # plain completion: no schema constraint


def test_gemini_text_returns_an_empty_string_when_the_response_has_no_text() -> None:
    """A blocked or empty candidate must not crash the caller."""
    assert GeminiClient(FakeGenAI(_Response(text=None))).text(model="m", system="s", user="u") == ""


# --------------------------------------------------------------------------- #
# provider selection
# --------------------------------------------------------------------------- #


def test_create_llm_dispatches_on_the_configured_provider() -> None:
    assert isinstance(create_llm(ModelConfig()), LLMClient)
    assert isinstance(create_llm(GEMINI_MODELS), GeminiClient)


def test_create_llm_rejects_an_unknown_provider() -> None:
    models = ModelConfig.model_construct(provider="mistral")  # bypasses Literal validation
    with pytest.raises(ValueError, match="mistral"):
        create_llm(models)


def test_settings_with_provider_swaps_the_model_preset() -> None:
    settings = Settings()
    gemini = settings.with_provider("gemini")

    assert gemini.models.provider == "gemini"
    assert gemini.models.extract_model == "gemini-2.5-flash"
    assert gemini.models.reason_model == "gemini-2.5-pro"
    assert settings.models.provider == "anthropic"  # original untouched
    assert gemini.retrieval == settings.retrieval  # only the models change
    assert gemini.with_provider("anthropic").models == ModelConfig()


def test_settings_with_provider_rejects_an_unknown_provider() -> None:
    with pytest.raises(ValueError, match="mistral"):
        Settings().with_provider("mistral")


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #


def test_cli_run_passes_the_provider_flag_to_the_factory(tmp_path, monkeypatch) -> None:
    trials_path = tmp_path / "trials.jsonl"
    write_jsonl([TRIAL], trials_path)
    index_path = tmp_path / "bm25.pkl"
    BM25Index.build([TRIAL]).save(index_path)

    llm = FakeLLM(
        {
            ProfileExtraction: ProfileExtraction(
                age_years=62.0,
                conditions=["stage IV lung adenocarcinoma"],
                search_queries=["advanced non-small cell lung cancer"],
            ),
            AssessmentBatch: AssessmentBatch(verdicts=[]),
        }
    )
    seen: list[ModelConfig] = []

    def fake_create_llm(models: ModelConfig) -> FakeLLM:
        seen.append(models)
        return llm

    monkeypatch.setattr(cli, "create_llm", fake_create_llm)

    exit_code = cli.main(
        [
            "run",
            "--note",
            NOTE,
            "--patient-id",
            "S001",
            "--trials",
            str(trials_path),
            "--index",
            str(index_path),
            "--no-questions",
            "--provider",
            "gemini",
        ]
    )

    assert exit_code == 0
    assert [models.provider for models in seen] == ["gemini"]
    assert seen[0].extract_model == "gemini-2.5-flash"
    # the agents were driven with the Gemini tier, not the Claude default
    assert {call["model"] for call in llm.calls} <= {
        GEMINI_MODELS.extract_model,
        GEMINI_MODELS.reason_model,
        GEMINI_MODELS.report_model,
    }


def test_cli_run_defaults_to_anthropic(tmp_path, monkeypatch) -> None:
    trials_path = tmp_path / "trials.jsonl"
    write_jsonl([TRIAL], trials_path)
    index_path = tmp_path / "bm25.pkl"
    BM25Index.build([TRIAL]).save(index_path)

    llm = FakeLLM(
        {
            ProfileExtraction: ProfileExtraction(age_years=62.0, search_queries=["lung cancer"]),
            AssessmentBatch: AssessmentBatch(verdicts=[]),
        }
    )
    seen: list[ModelConfig] = []
    monkeypatch.setattr(cli, "create_llm", lambda models: (seen.append(models), llm)[1])

    assert (
        cli.main(
            [
                "run",
                "--note",
                NOTE,
                "--patient-id",
                "S001",
                "--trials",
                str(trials_path),
                "--index",
                str(index_path),
                "--no-questions",
            ]
        )
        == 0
    )
    assert [models.provider for models in seen] == ["anthropic"]
