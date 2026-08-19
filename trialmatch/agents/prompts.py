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
- label "not_met": the criterion does not hold for this patient.
- label "unknown": the note does not contain the information needed to decide.

What licenses each label depends on the label and on the kind of criterion, and
the asymmetry is deliberate:

- "met", on any criterion: only when the description or a patient answer
  supports it. Never label a criterion "met" from inference alone. A silent
  description is not support, and a plausible guess is not support.
- "not_met" on an INCLUSION criterion: only when the description positively
  contradicts the criterion. Silence, or the absence of any mention, is never
  enough to decide that an inclusion criterion is not met.
- "not_met" on an EXCLUSION criterion: allowed when the description clearly
  implies the exclusion does not apply, the way an experienced clinician reading
  the same note would settle it. Quote the implying passage in evidence and
  spell the inference out in the explanation; a verdict with no quoted evidence
  is discarded downstream and counts as undecided.
- "unknown": the default whenever the information needed is genuinely absent.
  For a safety-relevant exclusion — one whose wrong clearance could endanger the
  patient, such as pregnancy, organ failure, a contraindication, active
  infection, psychiatric illness, or concurrent therapy, among others — stay
  "unknown" unless the description or a patient answer addresses it directly.

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

The question budget is small, so spend it on facts the patient actually knows.
Prefer questions a patient can answer from lived experience:
- symptoms, and when they started or how often they happen
- current and past medications
- diagnoses a doctor has told them they have
- prior treatments, surgeries, and hospital stays
- pregnancy status and contraception
- smoking and alcohol use
- allergies
- daily functioning: work, walking, self-care, help needed

Deprioritize anything only a clinician holds: exact laboratory values, imaging
or pathology findings, test scores, staging details. The patient cannot recall
those numbers, so such a question comes back unanswered and its slot is wasted.

When a criterion of that kind is genuinely the only thing left to ask about,
never ask for the value. Rephrase it into the patient-knowable form: ask "Have
you ever been told by a doctor that your liver tests were abnormal?" instead of
"What were your recent liver enzyme levels?".

Field semantics:
- text: one plain-language question a non-clinician can answer. No medical
  jargon, no abbreviations, no compound questions.
- criterion_ids: every given criterion identifier the question would resolve.
- patient_answerable: true when a typical patient can answer from memory of
  their own health and care; false when the question still needs a laboratory
  value, an imaging or pathology finding, or another record only a clinician
  holds.

Order the questions by how much they matter, answerability first: every question
the patient can answer comes before every question that still depends on
clinician-held data. Within each of those two groups, a question that resolves
an exclusion criterion comes before one that resolves only inclusion criteria.
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

Answer as a real patient would, so report only what this patient has lived
through or been told:
- symptoms, when they started, how often they happen, how long they last
- the medicines taken now and in the past
- diagnoses, treatments, operations and hospital stays a clinician told them about
- pregnancy status and contraception
- smoking and alcohol use
- allergies
- how they manage day to day: work, walking, self-care, help needed

A patient does not carry their own chart. When a question asks for an exact
laboratory value, an imaging or pathology finding, a test score, or a stage, you
know it only if the description states that the patient was told it. If the
description merely records such a measurement as a clinical fact, the patient
has not been told it: answer that you do not know, and never read the number off
the description. If the description says the patient was told the result in
plain words ("her doctor told her that her urine protein was very high"), report
that qualitative fact in the patient's own voice, without adding a number.

If the description does not contain the answer, or the answer is something this
patient would not know, reply with exactly:
{SIMULATOR_UNKNOWN_ANSWER}

Otherwise answer in one or two short sentences, in the patient's own voice, and
include the specific fact from the description — a value only when the
description says the patient was told it. Return one answer per question,
reusing the question identifier exactly."""


def simulator_user(ground_truth: str, questions_block: str) -> str:
    return (
        f"Ground-truth patient description:\n\n{ground_truth}\n\n"
        f"Questions:\n{questions_block}"
    )
