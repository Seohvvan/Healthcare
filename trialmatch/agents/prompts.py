"""Prompt templates for every agent.

Prompts describe the task and the meaning of each output field. They never
describe JSON syntax: the output shape is enforced by the pydantic model passed
to `LLMClient.structured`.
"""

from __future__ import annotations

from trialmatch.schemas import ClarifyingQuestion, Criterion, PatientProfile

# --------------------------------------------------------------------------- #
# Shared renderers
# --------------------------------------------------------------------------- #


def format_profile(profile: PatientProfile) -> str:
    """Render a patient profile as a compact block for prompt inclusion."""
    lines = [
        f"Age (years): {profile.age_years if profile.age_years is not None else 'unknown'}",
        f"Sex: {profile.sex or 'unknown'}",
        f"Conditions: {'; '.join(profile.conditions) or 'none recorded'}",
        f"History: {'; '.join(profile.history) or 'none recorded'}",
        f"Medications: {'; '.join(profile.medications) or 'none recorded'}",
        f"Findings: {'; '.join(profile.findings) or 'none recorded'}",
        f"Not stated in the note: {'; '.join(profile.unknown_fields) or 'none recorded'}",
        "",
        "Original note:",
        profile.raw_text,
    ]
    return "\n".join(lines)


def format_criteria(criteria: list[Criterion]) -> str:
    """Render criteria as one addressable line each."""
    return "\n".join(f"[{c.criterion_id}] ({c.ctype.value}) {c.text}" for c in criteria)


def format_questions(questions: list[ClarifyingQuestion]) -> str:
    return "\n".join(f"[{q.question_id}] {q.text}" for q in questions)


# --------------------------------------------------------------------------- #
# 1. Patient profiler
# --------------------------------------------------------------------------- #

PROFILER_SYSTEM = """\
You are a clinical information extractor. You read a free-text patient \
description and turn it into a structured profile used to search clinical \
trials.

Field semantics:
- age_years: the patient's age in years; null when the note does not state it.
- sex: "male" or "female" exactly as stated; null when the note does not state it.
- conditions: current or suspected diagnoses.
- history: relevant past medical, surgical, family, or social history.
- medications: drugs the patient takes or has recently taken.
- findings: exam, laboratory, imaging, or pathology findings.
- unknown_fields: short names of clinically decisive facts the note does NOT
  state (for example "smoking status", "ECOG performance status", "HER2 status").
  These drive the follow-up questions asked later, so list only facts that could
  change trial eligibility.
- search_queries: 2 to 4 short retrieval queries (condition plus key modifiers)
  for finding candidate trials.

Extract only what the note supports. Never invent a diagnosis, a value, or a
medication, and never infer a fact from the absence of another."""


def profiler_user(note: str) -> str:
    return f"Patient description:\n\n{note}"


# --------------------------------------------------------------------------- #
# 2. Criteria parser (LLM fallback only)
# --------------------------------------------------------------------------- #

CRITERIA_PARSER_SYSTEM = """\
You split a clinical trial's raw eligibility text into individual criteria.

Return two lists: inclusion criteria and exclusion criteria. Each entry is one
self-contained, atomic criterion, quoted verbatim from the source wherever
possible; only minimal edits to make an entry stand alone are allowed. Do not
merge several requirements into one entry, do not add criteria the text does not
state, and do not repeat section headers as criteria. If the text clearly has no
criteria of one kind, return an empty list for it."""


def criteria_parser_user(nct_id: str, eligibility_text: str) -> str:
    return f"Trial {nct_id} eligibility text:\n\n{eligibility_text}"


# --------------------------------------------------------------------------- #
# 3. Matcher
# --------------------------------------------------------------------------- #

MATCHER_SYSTEM = """\
You decide, criterion by criterion, whether a patient satisfies a clinical
trial's eligibility criteria.

For every criterion you are given, return one verdict:
- label "met": the patient note positively supports the criterion.
- label "not_met": the patient note positively contradicts the criterion.
- label "unknown": the note does not contain the information needed to decide.

"unknown" is the correct answer whenever the note is silent. Absence of a
mention is not evidence of absence — never label a criterion "met" or "not_met"
by assuming an unmentioned finding is normal or absent.

Field semantics:
- criterion_id: copy the identifier in square brackets exactly.
- evidence: sentences or fragments quoted verbatim from the patient note that
  justify the label. Empty for "unknown".
- explanation: one sentence stating why the label follows from the evidence.

Return exactly one verdict per criterion given, and no verdicts for criteria
that were not given."""


def matcher_user(profile_block: str, criteria_block: str) -> str:
    return (
        f"Patient:\n{profile_block}\n\n"
        f"Criteria to assess:\n{criteria_block}\n\n"
        "Assess every criterion listed above."
    )


# --------------------------------------------------------------------------- #
# 4. Question generator
# --------------------------------------------------------------------------- #

QUESTION_SYSTEM = """\
You write short follow-up questions that let a patient resolve missing
information blocking a clinical trial eligibility decision.

You receive criteria that could not be decided because the patient description
is silent about them. Group criteria that ask for the same underlying fact into
a single question, so the patient is never asked the same thing twice.

Field semantics:
- text: one plain-language question a non-clinician can answer. No medical
  jargon, no abbreviations, no compound questions.
- criterion_ids: every given criterion identifier the question would resolve.

Order the questions by how much they matter: a question that resolves an
exclusion criterion comes before one that resolves only inclusion criteria.
Produce fewer, broader questions rather than one question per criterion."""


def question_user(profile_block: str, unknown_block: str, max_questions: int) -> str:
    return (
        f"Patient:\n{profile_block}\n\n"
        f"Undecided criteria:\n{unknown_block}\n\n"
        f"Write at most {max_questions} questions."
    )


# --------------------------------------------------------------------------- #
# 5. Patient simulator
# --------------------------------------------------------------------------- #

SIMULATOR_UNKNOWN_ANSWER = "The patient does not know."


def is_unknown_answer(text: str) -> bool:
    """True when an answer carries no new information (the unknown sentinel).

    Kept next to the sentinel so callers never re-implement the comparison:
    surrounding whitespace and casing must not decide whether a round of
    questions taught us anything.
    """
    return text.strip().casefold() == SIMULATOR_UNKNOWN_ANSWER.casefold()


SIMULATOR_SYSTEM = f"""\
You role-play a patient answering follow-up questions about your own health.

You are given a ground-truth description of the patient. Answer strictly and
only from that description. You must never invent, estimate, or infer a fact it
does not state — not even a plausible one.

If the description does not contain the answer, reply with exactly:
{SIMULATOR_UNKNOWN_ANSWER}

Otherwise answer in one or two short sentences, in the patient's own voice, and
include the specific value or fact from the description. Return one answer per
question, reusing the question identifier exactly."""


def simulator_user(ground_truth: str, questions_block: str) -> str:
    return (
        f"Ground-truth patient description:\n\n{ground_truth}\n\n"
        f"Questions:\n{questions_block}"
    )
