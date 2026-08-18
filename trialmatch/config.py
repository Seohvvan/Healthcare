"""Central configuration: model tiers, paths, retrieval parameters."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

Provider = Literal["anthropic", "gemini"]


class ModelConfig(BaseModel):
    """Model tiering per design.md §4 (cost control)."""

    provider: Provider = "anthropic"  # which SDK `trialmatch.llm.create_llm` builds
    extract_model: str = "claude-haiku-4-5"  # parsing / extraction / simulator
    reason_model: str = "claude-sonnet-5"  # criterion matching / question generation
    report_model: str = "claude-sonnet-5"  # final report (bump to opus if needed)
    max_tokens_extract: int = 2048
    max_tokens_reason: int = 4096


# Gemini counterpart of the default (Claude) tiering, used by `--provider gemini`.
# Change the model ids here — they are the only place a Gemini version is pinned.
# gemini-2.5-pro is closed to new API accounts (404 at generate time as of
# 2026-08-18); 3.1-pro-preview is the replacement Google's error message names.
# Gemini 3.x models spend "thinking" tokens inside max_output_tokens, so the
# caps are far higher than the Anthropic defaults or JSON gets truncated.
GEMINI_MODELS = ModelConfig(
    provider="gemini",
    extract_model="gemini-3.5-flash",
    reason_model="gemini-3.1-pro-preview",
    report_model="gemini-3.5-flash",
    max_tokens_extract=8192,
    max_tokens_reason=16384,
)

def gemini_models_from_env() -> ModelConfig:
    """GEMINI_MODELS with .env / environment overrides applied.

    `GEMINI_MODEL` overrides all three tiers at once; the finer-grained
    `GEMINI_EXTRACT_MODEL` / `GEMINI_REASON_MODEL` / `GEMINI_REPORT_MODEL`
    win over it per tier. Unset variables fall back to `GEMINI_MODELS`.
    """
    base = GEMINI_MODELS
    single = os.environ.get("GEMINI_MODEL")
    return base.model_copy(
        update={
            "extract_model": os.environ.get(
                "GEMINI_EXTRACT_MODEL", single or base.extract_model
            ),
            "reason_model": os.environ.get("GEMINI_REASON_MODEL", single or base.reason_model),
            "report_model": os.environ.get("GEMINI_REPORT_MODEL", single or base.report_model),
        }
    )


_PROVIDER_MODEL_FACTORIES = {
    "anthropic": ModelConfig,
    "gemini": gemini_models_from_env,
}


def load_dotenv(path: Path | None = None) -> None:
    """Populate os.environ from a `.env` file without overriding existing values.

    Minimal KEY=VALUE parser (no export statements, no interpolation); values
    may be wrapped in single or double quotes. Missing file is a no-op.
    """
    target = path if path is not None else PROJECT_ROOT / ".env"
    if not target.exists():
        return
    for raw_line in target.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


class RetrievalConfig(BaseModel):
    candidates_k: int = 50  # trials passed from retrieval to matching
    bm25_k: int = 100
    dense_k: int = 100
    rrf_k: int = 60  # RRF constant


class RankingConfig(BaseModel):
    """Deterministic aggregation rules (design.md §3)."""

    unknown_penalty: float = 0.2  # per-unknown fraction penalty on the score
    excluded_score: float = -1.0


class Settings(BaseModel):
    models: ModelConfig = Field(default_factory=ModelConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    ranking: RankingConfig = Field(default_factory=RankingConfig)
    data_dir: Path = DATA_DIR
    trials_snapshot: Path = DATA_DIR / "trials.jsonl"
    log_dir: Path = PROJECT_ROOT / "runs"  # per-run JSON logs for reproducibility

    def with_provider(self, provider: str) -> Settings:
        """Copy of these settings with the model preset of `provider`.

        Only the model tiering changes; retrieval, ranking and paths are shared
        so a provider comparison varies exactly one thing.
        """
        try:
            factory = _PROVIDER_MODEL_FACTORIES[provider]
        except KeyError:
            expected = ", ".join(sorted(_PROVIDER_MODEL_FACTORIES))
            raise ValueError(
                f"unknown LLM provider {provider!r}; expected one of: {expected}"
            ) from None
        return self.model_copy(update={"models": factory()})


settings = Settings()
