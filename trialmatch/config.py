"""Central configuration: model tiers, paths, retrieval parameters."""

from __future__ import annotations

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
GEMINI_MODELS = ModelConfig(
    provider="gemini",
    extract_model="gemini-2.5-flash",
    reason_model="gemini-2.5-pro",
    report_model="gemini-2.5-flash",
)

_PROVIDER_MODELS: dict[str, ModelConfig] = {
    "anthropic": ModelConfig(),
    "gemini": GEMINI_MODELS,
}


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
            models = _PROVIDER_MODELS[provider]
        except KeyError:
            expected = ", ".join(sorted(_PROVIDER_MODELS))
            raise ValueError(
                f"unknown LLM provider {provider!r}; expected one of: {expected}"
            ) from None
        return self.model_copy(update={"models": models.model_copy()})


settings = Settings()
