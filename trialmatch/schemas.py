"""Shared data contracts for all pipeline stages.

Every module (data, retrieval, agents, eval, pipeline) communicates through
these models only. Keep changes backward-compatible: downstream slices are
implemented against this exact interface.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class CriterionType(str, Enum):
    INCLUSION = "inclusion"
    EXCLUSION = "exclusion"


class EligibilityLabel(str, Enum):
    """Criterion-level verdict (TrialGPT-style three-way classification)."""

    MET = "met"
    NOT_MET = "not_met"
    UNKNOWN = "unknown"  # patient note lacks the information to decide


class TrialLabel(str, Enum):
    """Trial-level verdict, aligned with TREC CT qrels (gain 2 / 1 / 0)."""

    ELIGIBLE = "eligible"  # meets inclusion, no exclusion applies (qrel 2)
    EXCLUDED = "excluded"  # matches condition but an exclusion applies (qrel 1)
    NOT_RELEVANT = "not_relevant"  # qrel 0


class TrialRecord(BaseModel):
    """One clinical trial, as stored in the local JSONL snapshot."""

    nct_id: str
    title: str = ""
    conditions: list[str] = Field(default_factory=list)
    brief_summary: str = ""
    eligibility_text: str = ""  # raw criteria text from eligibilityModule
    sex: str | None = None  # "ALL" | "FEMALE" | "MALE" | None
    minimum_age: str | None = None  # raw string, e.g. "18 Years"
    maximum_age: str | None = None
    overall_status: str | None = None
    phase: list[str] = Field(default_factory=list)


class Criterion(BaseModel):
    """One atomic eligibility criterion extracted from a trial."""

    criterion_id: str  # f"{nct_id}:{inclusion|exclusion}:{index}"
    ctype: CriterionType
    text: str  # verbatim criterion text


class ParsedCriteria(BaseModel):
    nct_id: str
    inclusion: list[Criterion] = Field(default_factory=list)
    exclusion: list[Criterion] = Field(default_factory=list)

    @property
    def all_criteria(self) -> list[Criterion]:
        return [*self.inclusion, *self.exclusion]


class PatientProfile(BaseModel):
    """Structured view of a free-text patient description."""

    patient_id: str
    raw_text: str
    age_years: float | None = None
    sex: str | None = None  # "female" | "male" | None
    conditions: list[str] = Field(default_factory=list)  # diagnoses / suspected
    history: list[str] = Field(default_factory=list)  # relevant medical history
    medications: list[str] = Field(default_factory=list)
    findings: list[str] = Field(default_factory=list)  # exam / lab / imaging findings
    unknown_fields: list[str] = Field(default_factory=list)  # info NOT in the note
    search_queries: list[str] = Field(default_factory=list)  # for retrieval


class CriterionVerdict(BaseModel):
    criterion_id: str
    label: EligibilityLabel
    evidence: list[str] = Field(default_factory=list)  # sentences from patient note
    explanation: str = ""


class TrialAssessment(BaseModel):
    """Aggregated result for one (patient, trial) pair.

    `trial_label` and `score` are computed by deterministic code in the ranker,
    never by the LLM.
    """

    patient_id: str
    nct_id: str
    verdicts: list[CriterionVerdict] = Field(default_factory=list)
    trial_label: TrialLabel | None = None
    score: float = 0.0
    unknown_criterion_ids: list[str] = Field(default_factory=list)


class ClarifyingQuestion(BaseModel):
    """A question generated to resolve UNKNOWN criteria."""

    question_id: str
    text: str
    criterion_ids: list[str] = Field(default_factory=list)  # criteria it resolves
    priority: int = 0  # lower = ask first (patient-answerable first, then exclusion)


class QuestionAnswer(BaseModel):
    question_id: str
    answer_text: str


class Recommendation(BaseModel):
    """Final ranked output for one patient."""

    patient_id: str
    ranked: list[TrialAssessment] = Field(default_factory=list)  # best first
    report_markdown: str = ""  # human-readable report incl. disclaimer
