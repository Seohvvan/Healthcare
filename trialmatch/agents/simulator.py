"""Patient Simulator: answers clarifying questions from ground-truth text only.

Used to run and evaluate the question -> re-assessment loop automatically
(design.md §3, §5.3). The simulator must never invent facts: when the
ground-truth text is silent it answers `SIMULATOR_UNKNOWN_ANSWER` verbatim, which
is what makes masked-field recovery measurable.
"""

from __future__ import annotations

from typing import cast

from pydantic import BaseModel, Field

from trialmatch.agents import prompts
from trialmatch.agents.prompts import SIMULATOR_UNKNOWN_ANSWER
from trialmatch.config import ModelConfig
from trialmatch.llm import LLMClient
from trialmatch.schemas import ClarifyingQuestion, QuestionAnswer


class SimulatedAnswer(BaseModel):
    """LLM output schema: one answer."""

    question_id: str
    answer_text: str


class AnswerBatch(BaseModel):
    """LLM output schema: one answer per question."""

    answers: list[SimulatedAnswer] = Field(default_factory=list)


def answer_questions(
    llm: LLMClient,
    full_profile_text: str,
    questions: list[ClarifyingQuestion],
    cfg: ModelConfig,
) -> list[QuestionAnswer]:
    """Answer every question, in the order asked, from `full_profile_text` alone."""
    if not questions:
        return []

    batch = cast(
        AnswerBatch,
        llm.structured(
            model=cfg.extract_model,
            system=prompts.SIMULATOR_SYSTEM,
            user=prompts.simulator_user(
                full_profile_text, prompts.format_questions(questions)
            ),
            schema=AnswerBatch,
            max_tokens=cfg.max_tokens_extract,
        ),
    )
    returned = {answer.question_id: answer.answer_text for answer in batch.answers}

    results: list[QuestionAnswer] = []
    for question in questions:
        text = (returned.get(question.question_id) or "").strip()
        results.append(
            QuestionAnswer(
                question_id=question.question_id,
                answer_text=text or SIMULATOR_UNKNOWN_ANSWER,
            )
        )
    return results
