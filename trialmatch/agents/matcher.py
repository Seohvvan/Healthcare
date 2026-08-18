"""Agent 4: criterion-level MET / NOT MET / UNKNOWN judgements with evidence.

Hard demographic rules (sex, age bounds) are decided in code and short-circuit
the LLM call entirely — they are cheap, exact, and a mismatch makes every other
criterion moot. Everything else goes through one structured call covering all
criteria at once.

`trial_label` and `score` stay at their defaults here; the ranker owns them.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import cast

from pydantic import BaseModel, Field

from trialmatch.agents import prompts
from trialmatch.config import ModelConfig
from trialmatch.llm import LLMClient
from trialmatch.schemas import (
    CriterionVerdict,
    EligibilityLabel,
    ParsedCriteria,
    PatientProfile,
    TrialAssessment,
    TrialRecord,
)

_AGE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(year|month|week|day)s?", re.IGNORECASE)
_UNIT_TO_YEARS = {
    "year": 1.0,
    "month": 1.0 / 12.0,
    "week": 1.0 / 52.1775,
    "day": 1.0 / 365.25,
}
_BINARY_SEXES = {"MALE", "FEMALE"}

# Criterion-id protocol for the synthetic demographic verdicts. Defined here
# once: the ranker imports `DEMOGRAPHIC_MARKER` to recognise them, so the string
# is never retyped at the other end of the seam.
_DEMOGRAPHIC = "demographic"
DEMOGRAPHIC_MARKER = f":{_DEMOGRAPHIC}:"
DEMOGRAPHIC_SEX_SUFFIX = f"{_DEMOGRAPHIC}:sex"
DEMOGRAPHIC_AGE_SUFFIX = f"{_DEMOGRAPHIC}:age"


class CriterionAssessment(BaseModel):
    """LLM output schema: one verdict."""

    criterion_id: str
    label: EligibilityLabel
    evidence: list[str] = Field(default_factory=list)
    explanation: str = ""


class AssessmentBatch(BaseModel):
    """LLM output schema: all verdicts for one trial."""

    verdicts: list[CriterionAssessment] = Field(default_factory=list)


def assess_trial(
    llm: LLMClient,
    profile: PatientProfile,
    parsed: ParsedCriteria,
    trial: TrialRecord,
    cfg: ModelConfig,
    criterion_ids: Sequence[str] | None = None,
) -> TrialAssessment:
    """Assess one (patient, trial) pair.

    `criterion_ids` restricts the call to a subset of `parsed` — used by the
    question loop to re-assess only the criteria a new answer can resolve
    (design.md §3). The returned assessment then holds verdicts for that subset
    only; merging it into the previous round is the caller's job. Ids the trial
    does not have are ignored. The hard demographic prefilter is unchanged: a
    mismatch short-circuits the LLM call whatever the subset is.
    """
    demographic = _demographic_mismatch(profile, trial)
    if demographic is not None:
        suffix, explanation = demographic
        return TrialAssessment(
            patient_id=profile.patient_id,
            nct_id=trial.nct_id,
            verdicts=[
                CriterionVerdict(
                    criterion_id=f"{trial.nct_id}:{suffix}",
                    label=EligibilityLabel.NOT_MET,
                    evidence=[],
                    explanation=explanation,
                )
            ],
        )

    criteria = parsed.all_criteria
    if criterion_ids is not None:
        wanted = set(criterion_ids)
        criteria = [c for c in criteria if c.criterion_id in wanted]
    if not criteria:
        return TrialAssessment(patient_id=profile.patient_id, nct_id=trial.nct_id)

    batch = cast(
        AssessmentBatch,
        llm.structured(
            model=cfg.reason_model,
            system=prompts.MATCHER_SYSTEM,
            user=prompts.matcher_user(
                prompts.format_profile(profile),
                prompts.format_criteria(criteria),
            ),
            schema=AssessmentBatch,
            max_tokens=cfg.max_tokens_reason,
        ),
    )
    returned = {item.criterion_id: item for item in batch.verdicts}

    verdicts: list[CriterionVerdict] = []
    for criterion in criteria:
        item = returned.get(criterion.criterion_id)
        if item is None:
            # The model skipped it: undecided by definition, never assumed met.
            verdicts.append(
                CriterionVerdict(
                    criterion_id=criterion.criterion_id,
                    label=EligibilityLabel.UNKNOWN,
                    evidence=[],
                    explanation="No verdict returned for this criterion.",
                )
            )
            continue
        verdicts.append(
            CriterionVerdict(
                criterion_id=criterion.criterion_id,
                label=item.label,
                evidence=item.evidence,
                explanation=item.explanation,
            )
        )

    return TrialAssessment(
        patient_id=profile.patient_id,
        nct_id=trial.nct_id,
        verdicts=verdicts,
        unknown_criterion_ids=[
            v.criterion_id for v in verdicts if v.label is EligibilityLabel.UNKNOWN
        ],
    )


def parse_age_years(raw: str | None) -> float | None:
    """Parse a ClinicalTrials.gov age string ("18 Years", "6 Months") into years."""
    if raw is None:
        return None
    text = raw.strip()
    if not text:
        return None
    match = _AGE_RE.search(text)
    if match is not None:
        return float(match.group(1)) * _UNIT_TO_YEARS[match.group(2).lower()]
    try:
        return float(text)
    except ValueError:
        return None


def _demographic_mismatch(
    profile: PatientProfile, trial: TrialRecord
) -> tuple[str, str] | None:
    """Return (criterion_id suffix, explanation) when a hard rule rules the trial out."""
    trial_sex = (trial.sex or "").strip().upper()
    patient_sex = (profile.sex or "").strip().upper()
    if trial_sex in _BINARY_SEXES and patient_sex in _BINARY_SEXES and trial_sex != patient_sex:
        return (
            DEMOGRAPHIC_SEX_SUFFIX,
            (
                f"The trial enrols {trial_sex.lower()} participants only, "
                f"but the patient is {patient_sex.lower()}."
            ),
        )

    age = profile.age_years
    if age is None:
        return None

    minimum = parse_age_years(trial.minimum_age)
    if minimum is not None and age < minimum:
        return (
            DEMOGRAPHIC_AGE_SUFFIX,
            (
                f"The patient is {age:g} years old, below the trial minimum of "
                f"{trial.minimum_age}."
            ),
        )

    maximum = parse_age_years(trial.maximum_age)
    if maximum is not None and age > maximum:
        return (
            DEMOGRAPHIC_AGE_SUFFIX,
            (
                f"The patient is {age:g} years old, above the trial maximum of "
                f"{trial.maximum_age}."
            ),
        )

    return None
