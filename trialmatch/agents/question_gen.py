"""Agent 5: turn UNKNOWN criteria into a short list of clarifying questions.

Priority is assigned in code, not by the LLM: a question touching any exclusion
criterion is priority 0 (asked first), inclusion-only questions are priority 1.
The `max_questions` cap is also enforced in code.
"""

from __future__ import annotations

from typing import cast

from pydantic import BaseModel, Field

from trialmatch.agents import prompts
from trialmatch.config import ModelConfig
from trialmatch.llm import LLMClient
from trialmatch.schemas import (
    ClarifyingQuestion,
    Criterion,
    CriterionType,
    ParsedCriteria,
    PatientProfile,
    TrialAssessment,
)

EXCLUSION_PRIORITY = 0
INCLUSION_PRIORITY = 1


class GeneratedQuestion(BaseModel):
    """LLM output schema: one question."""

    text: str
    criterion_ids: list[str] = Field(default_factory=list)


class QuestionBatch(BaseModel):
    """LLM output schema: the deduplicated question set."""

    questions: list[GeneratedQuestion] = Field(default_factory=list)


def generate_questions(
    llm: LLMClient,
    profile: PatientProfile,
    assessments: list[TrialAssessment],
    parsed_by_nct: dict[str, ParsedCriteria],
    cfg: ModelConfig,
    max_questions: int = 5,
) -> list[ClarifyingQuestion]:
    """Generate at most `max_questions` patient-friendly clarifying questions."""
    if max_questions <= 0:
        return []

    unknown = _collect_unknown_criteria(assessments, parsed_by_nct)
    if not unknown:
        return []

    batch = cast(
        QuestionBatch,
        llm.structured(
            model=cfg.reason_model,
            system=prompts.QUESTION_SYSTEM,
            user=prompts.question_user(
                prompts.format_profile(profile),
                prompts.format_criteria(list(unknown.values())),
                max_questions,
            ),
            schema=QuestionBatch,
            max_tokens=cfg.max_tokens_reason,
        ),
    )

    candidates: list[tuple[int, int, str, list[str]]] = []
    for order, generated in enumerate(batch.questions):
        text = generated.text.strip()
        if not text:
            continue
        # Drop identifiers the model invented or that are already resolved.
        criterion_ids = [cid for cid in generated.criterion_ids if cid in unknown]
        candidates.append((_priority(criterion_ids, unknown), order, text, criterion_ids))

    # Exclusion-related questions first; original order breaks ties.
    candidates.sort(key=lambda item: (item[0], item[1]))
    return [
        ClarifyingQuestion(
            question_id=f"q{index + 1}",
            text=text,
            criterion_ids=criterion_ids,
            priority=priority,
        )
        for index, (priority, _order, text, criterion_ids) in enumerate(
            candidates[:max_questions]
        )
    ]


def _collect_unknown_criteria(
    assessments: list[TrialAssessment],
    parsed_by_nct: dict[str, ParsedCriteria],
) -> dict[str, Criterion]:
    """Map every undecided criterion id across the assessments to its criterion."""
    unknown: dict[str, Criterion] = {}
    for assessment in assessments:
        parsed = parsed_by_nct.get(assessment.nct_id)
        if parsed is None:
            continue
        by_id = {c.criterion_id: c for c in parsed.all_criteria}
        for criterion_id in assessment.unknown_criterion_ids:
            criterion = by_id.get(criterion_id)
            if criterion is not None:
                unknown[criterion_id] = criterion
    return unknown


def _priority(criterion_ids: list[str], unknown: dict[str, Criterion]) -> int:
    for criterion_id in criterion_ids:
        criterion = unknown.get(criterion_id)
        if criterion is not None and criterion.ctype is CriterionType.EXCLUSION:
            return EXCLUSION_PRIORITY
    return INCLUSION_PRIORITY
