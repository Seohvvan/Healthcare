"""Gemini provider tests. No network, no SDK call, no API key: the client is faked.

`FakeGenAI` mimics the `google.genai.Client` surface used by `GeminiClient`
(`client.models.generate_content(...)`), so these tests pin the request we build
and the three response shapes we must survive, plus the free-tier plumbing
around every call: request pacing, 429 backoff and token accounting.

Time is faked module-wide by the autouse `clock` fixture, so the free-tier waits
(6s pacing, 30/60/120s backoff) are asserted without the suite ever blocking.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest
from conftest import FakeLLM  # shared test doubles, see tests/conftest.py

from trialmatch import cli
from trialmatch.agents.matcher import AssessmentBatch
from trialmatch.agents.profiler import ProfileExtraction
from trialmatch.config import GEMINI_MODELS, ModelConfig, Settings
from trialmatch.data import write_jsonl
from trialmatch.llm import GeminiClient, LLMClient, create_llm, format_usage
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


class _Usage:
    """Stand-in for `usage_metadata`; every counter may be absent (None)."""

    def __init__(
        self,
        prompt: int | None = None,
        output: int | None = None,
        thinking: int | None = None,
    ) -> None:
        self.prompt_token_count = prompt
        self.candidates_token_count = output
        self.thoughts_token_count = thinking


class _Response:
    """Stand-in for `google.genai.types.GenerateContentResponse`."""

    def __init__(
        self, parsed: Any = None, text: str | None = None, usage: Any = None
    ) -> None:
        self.parsed = parsed
        self.text = text
        self.usage_metadata = usage


class FakeGenAI:
    """Minimal stand-in for `google.genai.Client` with a recording models API."""

    def __init__(self, response: Any = None) -> None:
        self.response = response
        self.kwargs: dict[str, Any] = {}
        self.attempts = 0
        self.models = self

    def generate_content(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        self.attempts += 1
        return self.response


class ClientError(Exception):
    """Stand-in for `google.genai.errors.ClientError`, the SDK's 429 carrier.

    Across SDK versions the status has arrived on `status_code`, on `code`, or
    only as the message prefix, so `attribute` selects which shape to mimic.
    The class is deliberately named `ClientError`: detection matches the type
    name, which is what keeps it working when the SDK moves the class.
    """

    def __init__(
        self,
        message: str = "429 RESOURCE_EXHAUSTED: quota exceeded",
        *,
        status: int = 429,
        attribute: str | None = "status_code",
    ) -> None:
        super().__init__(message)
        if attribute is not None:
            setattr(self, attribute, status)


class FlakyGenAI(FakeGenAI):
    """`FakeGenAI` that raises for its first `failures` calls, then answers."""

    def __init__(self, response: Any, failures: int, error: Any = ClientError) -> None:
        super().__init__(response)
        self.failures = failures
        self.error = error

    def generate_content(self, **kwargs: Any) -> Any:
        if self.attempts < self.failures:
            self.attempts += 1
            raise self.error()
        return super().generate_content(**kwargs)


class FakeClock:
    """Deterministic `time.monotonic` / `time.sleep` pair.

    A sleep advances the clock instead of blocking, exactly as a real one does,
    so pacing sees the time its own backoff consumed.
    """

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture(autouse=True)
def clock(monkeypatch) -> FakeClock:
    """Fake the clock for every test here: pacing must never slow the suite."""
    fake = FakeClock()
    monkeypatch.setattr("time.monotonic", fake.monotonic)
    monkeypatch.setattr("time.sleep", fake.sleep)
    return fake


def _text_call(client: GeminiClient) -> str:
    return client.text(model="m", system="sys", user="usr")


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
# free tier: request pacing
# --------------------------------------------------------------------------- #


def test_gemini_paces_consecutive_calls_to_the_rpm_budget(clock) -> None:
    """The first call goes out at once; the second waits out the rest of 60/rpm."""
    client = GeminiClient(FakeGenAI(_Response(text="ok")))

    _text_call(client)
    assert clock.sleeps == []

    clock.advance(2.0)  # two seconds of work happened between the calls
    _text_call(client)

    assert clock.sleeps == pytest.approx([4.0])  # 60/10 - 2.0


def test_gemini_does_not_pace_when_the_interval_already_elapsed(clock) -> None:
    client = GeminiClient(FakeGenAI(_Response(text="ok")))

    _text_call(client)
    clock.advance(30.0)
    _text_call(client)

    assert clock.sleeps == []


def test_gemini_paces_structured_calls_too(clock) -> None:
    client = GeminiClient(FakeGenAI(_Response(parsed=ProfileExtraction())))

    for _ in range(2):
        client.structured(model="m", system="sys", user="usr", schema=ProfileExtraction)

    assert clock.sleeps == pytest.approx([6.0])


def test_gemini_rpm_env_var_sets_the_pacing_interval(clock, monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_RPM", "2")
    client = GeminiClient(FakeGenAI(_Response(text="ok")))

    monkeypatch.delenv("GEMINI_RPM")  # read once, at construction
    _text_call(client)
    _text_call(client)

    assert clock.sleeps == pytest.approx([30.0])  # 60/2


@pytest.mark.parametrize("raw", ["0", "-4", "abc", "", "2.5"])
def test_gemini_rpm_falls_back_to_the_default_when_invalid(raw, clock, monkeypatch) -> None:
    """A typo must not disable pacing — 0 or -1 would mean "no interval"."""
    monkeypatch.setenv("GEMINI_RPM", raw)
    client = GeminiClient(FakeGenAI(_Response(text="ok")))

    _text_call(client)
    _text_call(client)

    assert clock.sleeps == pytest.approx([6.0])  # 60/10, the default


# --------------------------------------------------------------------------- #
# free tier: 429 backoff
# --------------------------------------------------------------------------- #


def test_gemini_retries_a_429_with_growing_waits(clock, caplog) -> None:
    fake = FlakyGenAI(_Response(text="recovered"), failures=2)

    with caplog.at_level(logging.WARNING, logger="trialmatch.llm"):
        result = _text_call(GeminiClient(fake))

    assert result == "recovered"
    assert fake.attempts == 3  # two rejections plus the successful call
    assert clock.sleeps == pytest.approx([30.0, 60.0])  # each wait covers the pacing gap
    warnings = [record.getMessage() for record in caplog.records]
    assert len(warnings) == 2
    assert all("429" in message for message in warnings)
    assert "30" in warnings[0] and "60" in warnings[1]


def test_gemini_reraises_a_429_after_the_last_retry(clock) -> None:
    fake = FlakyGenAI(_Response(text="never reached"), failures=99)

    with pytest.raises(ClientError):
        _text_call(GeminiClient(fake))

    assert fake.attempts == 4  # the first call plus three retries
    assert clock.sleeps == pytest.approx([30.0, 60.0, 120.0])


@pytest.mark.parametrize("attribute", ["status_code", "code", None])
def test_gemini_recognises_a_429_in_every_shape_the_sdk_has_used(attribute, clock) -> None:
    """`status_code`, `code`, or only the message prefix — all must back off."""
    fake = FlakyGenAI(
        _Response(text="recovered"),
        failures=1,
        error=lambda: ClientError(attribute=attribute),
    )

    assert _text_call(GeminiClient(fake)) == "recovered"
    assert fake.attempts == 2


@pytest.mark.parametrize(
    ("status", "expected_attempts"), [(429, 2), (400, 1)]
)
def test_gemini_backoff_matches_the_installed_sdk_error(status, expected_attempts, clock) -> None:
    """Pin detection against the real `ClientError`, not only against our fake.

    The installed SDK carries the status on `code` and in the message prefix,
    and carries no `status_code` at all — a guard tuned to the fake alone would
    quietly never fire in production.
    """
    from google.genai.errors import ClientError as SdkClientError

    payload = {"error": {"code": status, "message": "RESOURCE_EXHAUSTED"}}
    fake = FlakyGenAI(
        _Response(text="recovered"),
        failures=1,
        error=lambda: SdkClientError(status, payload),
    )

    if expected_attempts == 1:
        with pytest.raises(SdkClientError):
            _text_call(GeminiClient(fake))
    else:
        assert _text_call(GeminiClient(fake)) == "recovered"
    assert fake.attempts == expected_attempts


def test_gemini_does_not_retry_a_non_429_client_error(clock) -> None:
    fake = FlakyGenAI(
        _Response(text="never reached"),
        failures=99,
        error=lambda: ClientError("400 INVALID_ARGUMENT", status=400),
    )

    with pytest.raises(ClientError, match="400"):
        _text_call(GeminiClient(fake))

    assert fake.attempts == 1
    assert clock.sleeps == []


# --------------------------------------------------------------------------- #
# free tier: usage accounting
# --------------------------------------------------------------------------- #


def test_gemini_usage_summary_starts_at_zero() -> None:
    assert GeminiClient(FakeGenAI()).usage_summary() == {
        "calls": 0,
        "prompt_tokens": 0,
        "output_tokens": 0,
        "thinking_tokens": 0,
    }


def test_gemini_accumulates_usage_over_structured_and_text_calls() -> None:
    response = _Response(
        parsed=ProfileExtraction(age_years=1.0),
        text='{"age_years": 1.0}',
        usage=_Usage(prompt=100, output=20, thinking=5),
    )
    client = GeminiClient(FakeGenAI(response))

    client.structured(model="m", system="sys", user="usr", schema=ProfileExtraction)
    _text_call(client)

    assert client.usage_summary() == {
        "calls": 2,
        "prompt_tokens": 200,
        "output_tokens": 40,
        "thinking_tokens": 10,
    }


def test_gemini_counts_a_call_whose_usage_metadata_is_missing() -> None:
    """A blocked or cached response carries no counters; the call still spent quota."""
    client = GeminiClient(FakeGenAI(_Response(text="ok", usage=None)))

    _text_call(client)

    assert client.usage_summary() == {
        "calls": 1,
        "prompt_tokens": 0,
        "output_tokens": 0,
        "thinking_tokens": 0,
    }


def test_gemini_tolerates_partial_usage_metadata() -> None:
    client = GeminiClient(FakeGenAI(_Response(text="ok", usage=_Usage(prompt=42))))

    _text_call(client)

    assert client.usage_summary() == {
        "calls": 1,
        "prompt_tokens": 42,
        "output_tokens": 0,
        "thinking_tokens": 0,
    }


def test_gemini_does_not_count_a_failed_call(clock) -> None:
    fake = FlakyGenAI(_Response(text="never reached"), failures=99)
    client = GeminiClient(fake)

    with pytest.raises(ClientError):
        _text_call(client)

    assert client.usage_summary()["calls"] == 0


def test_format_usage_is_none_for_a_client_without_usage_tracking() -> None:
    assert format_usage(FakeLLM()) is None
    assert format_usage(LLMClient(object())) is None


def test_format_usage_reports_calls_and_every_token_bucket() -> None:
    client = GeminiClient(FakeGenAI(_Response(text="ok", usage=_Usage(100, 20, 5))))
    _text_call(client)

    assert format_usage(client) == "llm usage: 1 calls, 100 prompt + 20 output + 5 thinking tokens"


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
    assert gemini.models.extract_model == "gemini-3.5-flash"
    assert gemini.models.reason_model == "gemini-3.1-pro-preview"
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
    assert seen[0].extract_model == "gemini-3.5-flash"
    # the agents were driven with the Gemini tier, not the Claude default
    assert {call["model"] for call in llm.calls} <= {
        GEMINI_MODELS.extract_model,
        GEMINI_MODELS.reason_model,
        GEMINI_MODELS.report_model,
    }


class UsageTrackingLLM(FakeLLM):
    """`FakeLLM` that also exposes `GeminiClient`'s usage surface."""

    def usage_summary(self) -> dict[str, int]:
        return {
            "calls": 3,
            "prompt_tokens": 1200,
            "output_tokens": 340,
            "thinking_tokens": 56,
        }


def test_cli_run_prints_the_usage_line_for_a_client_that_tracks_it(
    tmp_path, monkeypatch, capsys
) -> None:
    trials_path = tmp_path / "trials.jsonl"
    write_jsonl([TRIAL], trials_path)
    index_path = tmp_path / "bm25.pkl"
    BM25Index.build([TRIAL]).save(index_path)

    llm = UsageTrackingLLM(
        {
            ProfileExtraction: ProfileExtraction(age_years=62.0, search_queries=["lung cancer"]),
            AssessmentBatch: AssessmentBatch(verdicts=[]),
        }
    )
    monkeypatch.setattr(cli, "create_llm", lambda models: llm)

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
                "--provider",
                "gemini",
            ]
        )
        == 0
    )
    assert (
        "llm usage: 3 calls, 1200 prompt + 340 output + 56 thinking tokens"
        in capsys.readouterr().out
    )


def test_cli_run_defaults_to_anthropic(tmp_path, monkeypatch, capsys) -> None:
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
    # the Anthropic client tracks no usage, so the report must end as it always did
    assert "llm usage:" not in capsys.readouterr().out
