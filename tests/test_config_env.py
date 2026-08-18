"""Tests for .env loading and Gemini model overrides from the environment."""

from __future__ import annotations

import os

from trialmatch.config import GEMINI_MODELS, Settings, gemini_models_from_env, load_dotenv

_ENV_KEYS = (
    "GEMINI_MODEL",
    "GEMINI_EXTRACT_MODEL",
    "GEMINI_REASON_MODEL",
    "GEMINI_REPORT_MODEL",
)


def test_defaults_without_env(monkeypatch):
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    models = gemini_models_from_env()
    assert models == GEMINI_MODELS


def test_single_override_applies_to_all_tiers(monkeypatch):
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
    models = Settings().with_provider("gemini").models
    assert models.extract_model == "gemini-3.1-flash-lite"
    assert models.reason_model == "gemini-3.1-flash-lite"
    assert models.report_model == "gemini-3.1-flash-lite"
    assert models.provider == "gemini"


def test_tier_override_wins_over_single(monkeypatch):
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
    monkeypatch.setenv("GEMINI_REASON_MODEL", "gemini-3.1-pro-preview")
    models = gemini_models_from_env()
    assert models.extract_model == "gemini-3.1-flash-lite"
    assert models.reason_model == "gemini-3.1-pro-preview"


def test_anthropic_path_ignores_gemini_env(monkeypatch):
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
    models = Settings().with_provider("anthropic").models
    assert models.provider == "anthropic"
    assert "gemini" not in models.reason_model


def test_load_dotenv_sets_defaults_without_overriding(tmp_path, monkeypatch):
    monkeypatch.delenv("SOME_NEW_VAR", raising=False)
    monkeypatch.setenv("ALREADY_SET", "keep-me")
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# comment line\n"
        "SOME_NEW_VAR='quoted value'\n"
        "ALREADY_SET=do-not-apply\n"
        "MALFORMED LINE\n",
        encoding="utf-8",
    )
    load_dotenv(env_file)
    assert os.environ["SOME_NEW_VAR"] == "quoted value"
    assert os.environ["ALREADY_SET"] == "keep-me"
    monkeypatch.delenv("SOME_NEW_VAR", raising=False)


def test_load_dotenv_missing_file_is_noop(tmp_path):
    load_dotenv(tmp_path / "missing.env")  # must not raise
