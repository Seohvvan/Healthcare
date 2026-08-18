"""Agent 3: raw eligibility text -> `ParsedCriteria`.

A deterministic rule-based split handles the standard ClinicalTrials.gov layout
("Inclusion Criteria:" / "Exclusion Criteria:" plus bullets) with no token cost.
The LLM is only consulted when that split produces nothing usable — typically a
single unstructured blob. Set `use_llm=False` to stay fully offline.
"""

from __future__ import annotations

import re
from typing import cast

from pydantic import BaseModel, Field

from trialmatch.agents import prompts
from trialmatch.config import ModelConfig
from trialmatch.llm import LLMClient
from trialmatch.schemas import Criterion, CriterionType, ParsedCriteria, TrialRecord

_INCLUSION_HEADER_RE = re.compile(r"(?:key\s+)?inclusion\s*(?:criteria)?\s*:?", re.IGNORECASE)
_EXCLUSION_HEADER_RE = re.compile(r"(?:key\s+)?exclusion\s*(?:criteria)?\s*:?", re.IGNORECASE)

# Leading bullet glyph, "1.", "1)", "(1)" or "a)" style enumerations.
_BULLET_PREFIX_RE = re.compile(r"^\s*(?:[-*•·–—]+|\(?\d{1,2}[.)]|[a-z][.)])\s*")
_INLINE_BULLET_RE = re.compile(r"(?<=\S)\s+([*•])\s+")

# A lone item longer than this is an unsplit blob, not a criterion.
_BLOB_CHARS = 300
_MIN_CRITERION_CHARS = 3


class CriteriaSplit(BaseModel):
    """LLM output schema for the fallback path."""

    inclusion: list[str] = Field(default_factory=list)
    exclusion: list[str] = Field(default_factory=list)


def parse_criteria(
    llm: LLMClient,
    trial: TrialRecord,
    cfg: ModelConfig,
    use_llm: bool = True,
) -> ParsedCriteria:
    """Split `trial.eligibility_text` into atomic criteria."""
    inclusion, exclusion = split_eligibility_text(trial.eligibility_text)

    if not _looks_usable(inclusion, exclusion) and use_llm and trial.eligibility_text.strip():
        refined = cast(
            CriteriaSplit,
            llm.structured(
                model=cfg.extract_model,
                system=prompts.CRITERIA_PARSER_SYSTEM,
                user=prompts.criteria_parser_user(trial.nct_id, trial.eligibility_text),
                schema=CriteriaSplit,
                max_tokens=cfg.max_tokens_extract,
                cache_system=True,
            ),
        )
        inclusion = [t.strip() for t in refined.inclusion if t.strip()]
        exclusion = [t.strip() for t in refined.exclusion if t.strip()]

    return ParsedCriteria(
        nct_id=trial.nct_id,
        inclusion=_build(trial.nct_id, CriterionType.INCLUSION, inclusion),
        exclusion=_build(trial.nct_id, CriterionType.EXCLUSION, exclusion),
    )


def split_eligibility_text(text: str) -> tuple[list[str], list[str]]:
    """Rule-based split; returns (inclusion texts, exclusion texts)."""
    if not text or not text.strip():
        return [], []

    inc_header = _INCLUSION_HEADER_RE.search(text)
    exc_header = _EXCLUSION_HEADER_RE.search(text)

    if inc_header is None and exc_header is None:
        return [], []

    if inc_header is not None and exc_header is not None:
        if inc_header.start() < exc_header.start():
            inclusion_text = text[inc_header.end() : exc_header.start()]
            exclusion_text = text[exc_header.end() :]
        else:
            exclusion_text = text[exc_header.end() : inc_header.start()]
            inclusion_text = text[inc_header.end() :]
    elif inc_header is not None:
        inclusion_text = text[inc_header.end() :]
        exclusion_text = ""
    else:
        assert exc_header is not None
        inclusion_text = text[: exc_header.start()]
        exclusion_text = text[exc_header.end() :]

    return _split_bullets(inclusion_text), _split_bullets(exclusion_text)


def _split_bullets(block: str) -> list[str]:
    """Split one section into bullet items."""
    if not block.strip():
        return []
    # Give inline bullet markers their own line so both layouts split alike.
    normalized = _INLINE_BULLET_RE.sub(r"\n\1 ", block)
    items: list[str] = []
    for raw_line in normalized.splitlines():
        line = _BULLET_PREFIX_RE.sub("", raw_line).strip()
        if len(line) < _MIN_CRITERION_CHARS:
            continue
        items.append(line)
    return items


def _looks_usable(inclusion: list[str], exclusion: list[str]) -> bool:
    """Reject empty splits and single unsplit blobs."""
    items = [*inclusion, *exclusion]
    if not items:
        return False
    return not (len(items) == 1 and len(items[0]) > _BLOB_CHARS)


def _build(nct_id: str, ctype: CriterionType, texts: list[str]) -> list[Criterion]:
    return [
        Criterion(
            criterion_id=f"{nct_id}:{ctype.value}:{index}",
            ctype=ctype,
            text=text,
        )
        for index, text in enumerate(texts)
    ]
