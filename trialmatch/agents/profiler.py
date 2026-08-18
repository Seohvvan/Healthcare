"""Agent 1: free-text patient description -> structured `PatientProfile`."""

from __future__ import annotations

from typing import cast

from pydantic import BaseModel, Field

from trialmatch.agents import prompts
from trialmatch.config import ModelConfig
from trialmatch.llm import LLMClient
from trialmatch.schemas import PatientProfile


class ProfileExtraction(BaseModel):
    """LLM output schema: the clinical fields only.

    `patient_id` and `raw_text` are owned by the caller, so the model never sees
    them and can never corrupt them.
    """

    age_years: float | None = None
    sex: str | None = None
    conditions: list[str] = Field(default_factory=list)
    history: list[str] = Field(default_factory=list)
    medications: list[str] = Field(default_factory=list)
    findings: list[str] = Field(default_factory=list)
    unknown_fields: list[str] = Field(default_factory=list)
    search_queries: list[str] = Field(default_factory=list)


def profile_patient(
    llm: LLMClient,
    patient_id: str,
    note: str,
    cfg: ModelConfig,
) -> PatientProfile:
    """Extract a structured profile, preserving `patient_id` and `note` verbatim."""
    extracted = cast(
        ProfileExtraction,
        llm.structured(
            model=cfg.extract_model,
            system=prompts.PROFILER_SYSTEM,
            user=prompts.profiler_user(note),
            schema=ProfileExtraction,
            max_tokens=cfg.max_tokens_extract,
        ),
    )
    return PatientProfile(
        patient_id=patient_id,
        raw_text=note,
        age_years=extracted.age_years,
        sex=_normalize_sex(extracted.sex),
        conditions=extracted.conditions,
        history=extracted.history,
        medications=extracted.medications,
        findings=extracted.findings,
        unknown_fields=extracted.unknown_fields,
        search_queries=extracted.search_queries,
    )


def _normalize_sex(sex: str | None) -> str | None:
    """Map free-form sex strings onto the "male" / "female" contract."""
    if not sex:
        return None
    lowered = sex.strip().lower()
    if lowered in {"m", "male", "man", "boy"}:
        return "male"
    if lowered in {"f", "female", "woman", "girl"}:
        return "female"
    return None
