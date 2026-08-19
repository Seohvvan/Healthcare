"""Agent 5: turn UNKNOWN criteria into a short list of clarifying questions.

Ordering ranks two things at once: whether the patient can answer the question
at all, then whether it touches an exclusion criterion. The answerability tier
comes from the model's own `patient_answerable` flag (an unvalidated
self-report steered by the prompt); the exclusion-before-inclusion slot inside
each tier and the `max_questions` cap are enforced in code. A question the
patient cannot answer from lived experience (it needs a laboratory value, an
imaging finding, or another clinician-held record) drops a whole tier, so it
never outranks a question the patient can actually answer.
"""

from __future__ import annotations

import logging
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
# Added to the priority of a question the patient cannot answer, which demotes it
# below every answerable question while keeping exclusion-before-inclusion order
# inside each tier. A model that omits the flag is trusted to have asked
# something answerable, so the penalty is opt-in.
UNANSWERABLE_PENALTY = INCLUSION_PRIORITY + 1

logger = logging.getLogger(__name__)


class GeneratedQuestion(BaseModel):
    """LLM output schema: one question."""

    text: str
    criterion_ids: list[str] = Field(default_factory=list)
    patient_answerable: bool = True


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

    if batch.questions and all(q.patient_answerable for q in batch.questions):
        # Either every question really is answerable, or the model never sets
        # the flag and the tiering is a silent no-op — visible only here.
        logger.debug("no question was flagged patient-unanswerable in this batch")

    candidates: list[tuple[int, int, str, list[str]]] = []
    for order, generated in enumerate(batch.questions):
        text = generated.text.strip()
        if not text:
            continue
        # Drop identifiers the model invented or that are already resolved.
        criterion_ids = [cid for cid in generated.criterion_ids if cid in unknown]
        priority = _priority(criterion_ids, unknown, generated.patient_answerable)
        candidates.append((priority, order, text, criterion_ids))

    # Answerable questions first, exclusion-related first within each tier;
    # original order breaks remaining ties.
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


def _priority(
    criterion_ids: list[str],
    unknown: dict[str, Criterion],
    patient_answerable: bool,
) -> int:
    """Rank a question: answerability decides the tier, criterion type the slot."""
    penalty = 0 if patient_answerable else UNANSWERABLE_PENALTY
    for criterion_id in criterion_ids:
        criterion = unknown.get(criterion_id)
        if criterion is not None and criterion.ctype is CriterionType.EXCLUSION:
            return penalty + EXCLUSION_PRIORITY
    return penalty + INCLUSION_PRIORITY
