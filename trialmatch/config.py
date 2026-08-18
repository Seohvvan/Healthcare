"""Central configuration: model tiers, paths, retrieval parameters."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


class ModelConfig(BaseModel):
    """Model tiering per design.md §4 (cost control)."""

    extract_model: str = "claude-haiku-4-5"  # parsing / extraction / simulator
    reason_model: str = "claude-sonnet-5"  # criterion matching / question generation
    report_model: str = "claude-sonnet-5"  # final report (bump to opus if needed)
    max_tokens_extract: int = 2048
    max_tokens_reason: int = 4096


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


settings = Settings()
