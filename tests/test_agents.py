"""Agent tests. No network, no API key: every LLM call goes through `FakeLLM`."""

from __future__ import annotations

from typing import Any

import pytest
from conftest import FakeLLM  # shared test double, see tests/conftest.py

from trialmatch.agents.criteria_parser import CriteriaSplit, parse_criteria
from trialmatch.agents.matcher import (
    DEMOGRAPHIC_MARKER,
    DEMOGRAPHIC_SEX_SUFFIX,
    AssessmentBatch,
    CriterionAssessment,
    assess_trial,
    parse_age_years,
)
from trialmatch.agents.profiler import ProfileExtraction, profile_patient
from trialmatch.agents.question_gen import GeneratedQuestion, QuestionBatch, generate_questions
from trialmatch.agents.ranker import aggregate, label_grade, rank
from trialmatch.agents.reporter import DISCLAIMER, build_report
from trialmatch.agents.simulator import AnswerBatch, SimulatedAnswer, answer_questions
from trialmatch.config import ModelConfig, RankingConfig
from trialmatch.llm import LLMClient
from trialmatch.schemas import (
    ClarifyingQuestion,
    Criterion,
    CriterionType,
    CriterionVerdict,
    EligibilityLabel,
    ParsedCriteria,
    PatientProfile,
    TrialAssessment,
    TrialLabel,
    TrialRecord,
)

MODELS = ModelConfig()
RANKING = RankingConfig()

ELIGIBILITY_TEXT = """Inclusion Criteria:

* Age 18 years or older
* Histologically confirmed stage III or IV non-small cell lung cancer
* ECOG performance status of 0 or 1

Exclusion Criteria:

* Prior treatment with an EGFR tyrosine kinase inhibitor
* Untreated active brain metastases
"""


def make_trial(**overrides: Any) -> TrialRecord:
    base: dict[str, Any] = {
        "nct_id": "NCT00000001",
        "title": "A study of drug X in advanced NSCLC",
        "eligibility_text": ELIGIBILITY_TEXT,
    }
    base.update(overrides)
    return TrialRecord(**base)


def make_profile(**overrides: Any) -> PatientProfile:
    base: dict[str, Any] = {
        "patient_id": "p1",
        "raw_text": "A 62-year-old man with stage IV lung adenocarcinoma.",
        "age_years": 62.0,
        "sex": "male",
        "conditions": ["stage IV lung adenocarcinoma"],
    }
    base.update(overrides)
    return PatientProfile(**base)


def criterion(criterion_id: str, ctype: CriterionType, text: str = "criterion text") -> Criterion:
    return Criterion(criterion_id=criterion_id, ctype=ctype, text=text)


# --------------------------------------------------------------------------- #
# llm wrapper
# --------------------------------------------------------------------------- #


class _Block:
    def __init__(self, type_: str, text: str = "") -> None:
        self.type = type_
        self.text = text


class _Response:
    def __init__(self, parsed_output: Any = None, content: list[_Block] | None = None) -> None:
        self.parsed_output = parsed_output
        self.content = content or []


class FakeAnthropic:
    """Minimal stand-in for `anthropic.Anthropic` with a recording messages API."""

    def __init__(self, parse_result: Any = None, create_result: Any = None) -> None:
        self.parse_result = parse_result
        self.create_result = create_result
        self.parse_kwargs: dict[str, Any] = {}
        self.create_kwargs: dict[str, Any] = {}
        self.messages = self

    def parse(self, **kwargs: Any) -> Any:
        self.parse_kwargs = kwargs
        return self.parse_result

    def create(self, **kwargs: Any) -> Any:
        self.create_kwargs = kwargs
        return self.create_result


def test_llm_structured_returns_parsed_output_and_builds_the_request() -> None:
    expected = ProfileExtraction(age_years=40.0)
    fake = FakeAnthropic(parse_result=_Response(parsed_output=expected))

    result = LLMClient(fake).structured(
        model="m", system="sys", user="usr", schema=ProfileExtraction, max_tokens=123
    )

    assert result is expected
    assert fake.parse_kwargs["model"] == "m"
    assert fake.parse_kwargs["max_tokens"] == 123
    assert fake.parse_kwargs["system"] == "sys"
    assert fake.parse_kwargs["output_format"] is ProfileExtraction
    assert fake.parse_kwargs["messages"] == [{"role": "user", "content": "usr"}]


def test_llm_structured_marks_the_system_prompt_for_caching() -> None:
    fake = FakeAnthropic(parse_result=_Response(parsed_output=ProfileExtraction()))
    LLMClient(fake).structured(
        model="m", system="sys", user="usr", schema=ProfileExtraction, cache_system=True
    )
    assert fake.parse_kwargs["system"] == [
        {"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}}
    ]


def test_llm_structured_raises_when_parsing_fails() -> None:
    fake = FakeAnthropic(parse_result=_Response(parsed_output=None))
    with pytest.raises(RuntimeError, match="parsed_output"):
        LLMClient(fake).structured(
            model="m", system="sys", user="usr", schema=ProfileExtraction
        )


def test_llm_text_concatenates_text_blocks_only() -> None:
    fake = FakeAnthropic(
        create_result=_Response(
            content=[_Block("thinking"), _Block("text", "Hello "), _Block("text", "world")]
        )
    )
    assert LLMClient(fake).text(model="m", system="sys", user="usr") == "Hello world"


# --------------------------------------------------------------------------- #
# profiler
# --------------------------------------------------------------------------- #


def test_profile_patient_preserves_identity_fields() -> None:
    note = "A 62-year-old man with stage IV lung adenocarcinoma. Smoking status not documented."
    llm = FakeLLM(
        {
            ProfileExtraction: ProfileExtraction(
                age_years=62.0,
                sex="M",
                conditions=["stage IV lung adenocarcinoma"],
                unknown_fields=["smoking status"],
                search_queries=["stage IV lung adenocarcinoma", "NSCLC advanced"],
            )
        }
    )

    profile = profile_patient(llm, "patient-7", note, MODELS)

    assert profile.patient_id == "patient-7"
    assert profile.raw_text == note
    assert profile.age_years == 62.0
    assert profile.sex == "male"  # normalized from "M"
    assert profile.conditions == ["stage IV lung adenocarcinoma"]
    assert profile.unknown_fields == ["smoking status"]
    assert len(profile.search_queries) == 2
    assert llm.calls[0]["model"] == MODELS.extract_model


def test_profile_patient_drops_unrecognized_sex() -> None:
    llm = FakeLLM({ProfileExtraction: ProfileExtraction(sex="not stated")})
    profile = profile_patient(llm, "p1", "note", MODELS)
    assert profile.sex is None


# --------------------------------------------------------------------------- #
# criteria_parser
# --------------------------------------------------------------------------- #


def test_parse_criteria_deterministic_split_offline() -> None:
    llm = FakeLLM()  # any structured call would raise
    parsed = parse_criteria(llm, make_trial(), MODELS, use_llm=False)

    assert len(parsed.inclusion) == 3
    assert len(parsed.exclusion) == 2
    assert llm.calls == []

    assert [c.criterion_id for c in parsed.inclusion] == [
        "NCT00000001:inclusion:0",
        "NCT00000001:inclusion:1",
        "NCT00000001:inclusion:2",
    ]
    assert [c.criterion_id for c in parsed.exclusion] == [
        "NCT00000001:exclusion:0",
        "NCT00000001:exclusion:1",
    ]
    assert parsed.inclusion[0].text == "Age 18 years or older"
    assert parsed.exclusion[1].text == "Untreated active brain metastases"
    assert all(c.ctype is CriterionType.INCLUSION for c in parsed.inclusion)
    assert all(c.ctype is CriterionType.EXCLUSION for c in parsed.exclusion)


def test_parse_criteria_splits_numbered_and_dashed_bullets() -> None:
    text = (
        "INCLUSION CRITERIA\n"
        "1. Adults aged 18-75 years\n"
        "2) Documented type 2 diabetes\n"
        "EXCLUSION CRITERIA\n"
        "- Pregnancy or breastfeeding\n"
    )
    parsed = parse_criteria(FakeLLM(), make_trial(eligibility_text=text), MODELS, use_llm=False)

    assert [c.text for c in parsed.inclusion] == [
        "Adults aged 18-75 years",
        "Documented type 2 diabetes",
    ]
    assert [c.text for c in parsed.exclusion] == ["Pregnancy or breastfeeding"]


def test_parse_criteria_falls_back_to_llm_on_unstructured_blob() -> None:
    blob = (
        "Patients must be adults with confirmed disease and adequate organ function, "
        "and must not have received prior systemic therapy, must not be pregnant, and "
        "must not have any uncontrolled intercurrent illness that in the judgement of "
        "the investigator would interfere with participation in this study protocol."
    )
    llm = FakeLLM(
        {
            CriteriaSplit: CriteriaSplit(
                inclusion=["Adults with confirmed disease", "Adequate organ function"],
                exclusion=["Prior systemic therapy", "Pregnancy"],
            )
        }
    )
    parsed = parse_criteria(llm, make_trial(eligibility_text=blob), MODELS)

    assert len(llm.calls) == 1
    assert llm.calls[0]["cache_system"] is True
    assert [c.criterion_id for c in parsed.inclusion] == [
        "NCT00000001:inclusion:0",
        "NCT00000001:inclusion:1",
    ]
    assert parsed.exclusion[1].text == "Pregnancy"


def test_parse_criteria_offline_blob_returns_empty_instead_of_calling_llm() -> None:
    llm = FakeLLM()
    parsed = parse_criteria(llm, make_trial(eligibility_text="one long blob"), MODELS, use_llm=False)
    assert parsed.inclusion == []
    assert parsed.exclusion == []
    assert llm.calls == []


# --------------------------------------------------------------------------- #
# matcher
# --------------------------------------------------------------------------- #


def test_parse_age_years_units() -> None:
    assert parse_age_years("18 Years") == pytest.approx(18.0)
    assert parse_age_years("6 Months") == pytest.approx(0.5)
    assert parse_age_years("2 Weeks") == pytest.approx(2 / 52.1775)
    assert parse_age_years("30 Days") == pytest.approx(30 / 365.25)
    assert parse_age_years("65") == pytest.approx(65.0)
    assert parse_age_years(None) is None
    assert parse_age_years("N/A") is None


def test_assess_trial_short_circuits_on_sex_conflict() -> None:
    llm = FakeLLM()  # a structured call would raise
    trial = make_trial(sex="FEMALE")
    parsed = parse_criteria(FakeLLM(), trial, MODELS, use_llm=False)

    assessment = assess_trial(llm, make_profile(sex="male"), parsed, trial, MODELS)

    assert llm.calls == []
    assert len(assessment.verdicts) == 1
    verdict = assessment.verdicts[0]
    assert verdict.criterion_id == f"NCT00000001:{DEMOGRAPHIC_SEX_SUFFIX}"
    assert verdict.label is EligibilityLabel.NOT_MET
    assert assessment.unknown_criterion_ids == []
    assert assessment.trial_label is None  # the ranker decides


def test_demographic_mismatch_flows_from_the_matcher_into_a_not_relevant_label() -> None:
    """Seam test for the demographic criterion-id protocol (matcher -> ranker).

    The marker literal is defined once in the matcher; the ranker recognises the
    synthetic verdict through the same constant, so a sex mismatch must come out
    of `aggregate` as NOT_RELEVANT without either side retyping the string.
    """
    trial = make_trial(sex="FEMALE")
    parsed = parse_criteria(FakeLLM(), trial, MODELS, use_llm=False)

    assessment = assess_trial(FakeLLM(), make_profile(sex="male"), parsed, trial, MODELS)
    result = aggregate(assessment, parsed, RANKING)

    assert DEMOGRAPHIC_MARKER in assessment.verdicts[0].criterion_id
    assert result.trial_label is TrialLabel.NOT_RELEVANT
    assert result.score == pytest.approx(-1.0)


def test_assess_trial_restricted_to_a_criterion_subset() -> None:
    trial = make_trial()
    parsed = parse_criteria(FakeLLM(), trial, MODELS, use_llm=False)
    subset = ["NCT00000001:inclusion:2", "NCT00000001:exclusion:1"]
    llm = FakeLLM(
        {
            AssessmentBatch: AssessmentBatch(
                verdicts=[
                    CriterionAssessment(
                        criterion_id="NCT00000001:inclusion:2",
                        label=EligibilityLabel.MET,
                    ),
                    CriterionAssessment(
                        criterion_id="NCT00000001:exclusion:1",
                        label=EligibilityLabel.NOT_MET,
                    ),
                ]
            )
        }
    )

    assessment = assess_trial(
        llm, make_profile(), parsed, trial, MODELS, criterion_ids=[*subset, "NCT00000001:bogus"]
    )

    # Only the requested criteria are sent to the model and returned.
    prompt = llm.calls[0]["user"]
    assert "NCT00000001:inclusion:2" in prompt
    assert "NCT00000001:inclusion:0" not in prompt
    assert [v.criterion_id for v in assessment.verdicts] == subset
    assert assessment.unknown_criterion_ids == []


def test_assess_trial_with_an_empty_criterion_subset_skips_the_llm() -> None:
    trial = make_trial()
    parsed = parse_criteria(FakeLLM(), trial, MODELS, use_llm=False)
    llm = FakeLLM()

    assessment = assess_trial(llm, make_profile(), parsed, trial, MODELS, criterion_ids=[])

    assert llm.calls == []
    assert assessment.verdicts == []


def test_assess_trial_short_circuits_below_minimum_age() -> None:
    llm = FakeLLM()
    trial = make_trial(minimum_age="18 Years")
    parsed = parse_criteria(FakeLLM(), trial, MODELS, use_llm=False)

    assessment = assess_trial(llm, make_profile(age_years=12.0), parsed, trial, MODELS)

    assert llm.calls == []
    assert assessment.verdicts[0].criterion_id == "NCT00000001:demographic:age"
    assert assessment.verdicts[0].label is EligibilityLabel.NOT_MET


def test_assess_trial_allows_sex_all_and_in_range_age() -> None:
    trial = make_trial(sex="ALL", minimum_age="18 Years", maximum_age="80 Years")
    parsed = parse_criteria(FakeLLM(), trial, MODELS, use_llm=False)
    llm = FakeLLM({AssessmentBatch: AssessmentBatch(verdicts=[])})

    assessment = assess_trial(llm, make_profile(age_years=62.0), parsed, trial, MODELS)

    assert len(llm.calls) == 1  # no short circuit
    assert len(assessment.verdicts) == 5


def test_assess_trial_maps_llm_output_and_defaults_missing_to_unknown() -> None:
    trial = make_trial()
    parsed = parse_criteria(FakeLLM(), trial, MODELS, use_llm=False)
    llm = FakeLLM(
        {
            AssessmentBatch: AssessmentBatch(
                verdicts=[
                    CriterionAssessment(
                        criterion_id="NCT00000001:inclusion:0",
                        label=EligibilityLabel.MET,
                        evidence=["A 62-year-old man"],
                        explanation="The patient is 62, above the 18 year threshold.",
                    ),
                    CriterionAssessment(
                        criterion_id="NCT00000001:inclusion:1",
                        label=EligibilityLabel.MET,
                        evidence=["stage IV lung adenocarcinoma"],
                    ),
                    # inclusion:2 deliberately omitted by the model
                    CriterionAssessment(
                        criterion_id="NCT00000001:exclusion:0",
                        label=EligibilityLabel.NOT_MET,
                    ),
                    CriterionAssessment(
                        criterion_id="NCT00000001:exclusion:1",
                        label=EligibilityLabel.UNKNOWN,
                    ),
                    # An id the model invented must not leak into the result.
                    CriterionAssessment(
                        criterion_id="NCT00000001:inclusion:99",
                        label=EligibilityLabel.MET,
                    ),
                ]
            )
        }
    )

    assessment = assess_trial(llm, make_profile(), parsed, trial, MODELS)

    assert llm.calls[0]["model"] == MODELS.reason_model
    by_id = {v.criterion_id: v for v in assessment.verdicts}
    assert set(by_id) == {c.criterion_id for c in parsed.all_criteria}
    assert by_id["NCT00000001:inclusion:0"].label is EligibilityLabel.MET
    assert by_id["NCT00000001:inclusion:0"].evidence == ["A 62-year-old man"]
    assert by_id["NCT00000001:inclusion:2"].label is EligibilityLabel.UNKNOWN
    assert by_id["NCT00000001:inclusion:2"].evidence == []
    assert assessment.unknown_criterion_ids == [
        "NCT00000001:inclusion:2",
        "NCT00000001:exclusion:1",
    ]


def test_assess_trial_without_criteria_skips_the_llm() -> None:
    llm = FakeLLM()
    trial = make_trial(eligibility_text="")
    parsed = ParsedCriteria(nct_id=trial.nct_id)
    assessment = assess_trial(llm, make_profile(), parsed, trial, MODELS)
    assert llm.calls == []
    assert assessment.verdicts == []


# --------------------------------------------------------------------------- #
# ranker
# --------------------------------------------------------------------------- #


def build_parsed(nct_id: str, n_inclusion: int, n_exclusion: int) -> ParsedCriteria:
    return ParsedCriteria(
        nct_id=nct_id,
        inclusion=[
            criterion(f"{nct_id}:inclusion:{i}", CriterionType.INCLUSION)
            for i in range(n_inclusion)
        ],
        exclusion=[
            criterion(f"{nct_id}:exclusion:{i}", CriterionType.EXCLUSION)
            for i in range(n_exclusion)
        ],
    )


def build_assessment(nct_id: str, labels: dict[str, EligibilityLabel]) -> TrialAssessment:
    return TrialAssessment(
        patient_id="p1",
        nct_id=nct_id,
        verdicts=[
            CriterionVerdict(criterion_id=cid, label=label) for cid, label in labels.items()
        ],
        unknown_criterion_ids=[
            cid for cid, label in labels.items() if label is EligibilityLabel.UNKNOWN
        ],
    )


def test_aggregate_exclusion_met_is_excluded() -> None:
    parsed = build_parsed("NCT1", 2, 1)
    assessment = build_assessment(
        "NCT1",
        {
            "NCT1:inclusion:0": EligibilityLabel.MET,
            "NCT1:inclusion:1": EligibilityLabel.MET,
            "NCT1:exclusion:0": EligibilityLabel.MET,
        },
    )
    result = aggregate(assessment, parsed, RANKING)
    assert result.trial_label is TrialLabel.EXCLUDED
    assert result.score == RANKING.excluded_score


def test_aggregate_all_inclusion_met_no_unknowns_is_eligible() -> None:
    parsed = build_parsed("NCT1", 2, 1)
    assessment = build_assessment(
        "NCT1",
        {
            "NCT1:inclusion:0": EligibilityLabel.MET,
            "NCT1:inclusion:1": EligibilityLabel.MET,
            "NCT1:exclusion:0": EligibilityLabel.NOT_MET,
        },
    )
    result = aggregate(assessment, parsed, RANKING)
    assert result.trial_label is TrialLabel.ELIGIBLE
    assert result.score == pytest.approx(1.0)


def test_aggregate_inclusion_not_met_is_not_relevant() -> None:
    parsed = build_parsed("NCT1", 2, 0)
    assessment = build_assessment(
        "NCT1",
        {
            "NCT1:inclusion:0": EligibilityLabel.MET,
            "NCT1:inclusion:1": EligibilityLabel.NOT_MET,
        },
    )
    result = aggregate(assessment, parsed, RANKING)
    assert result.trial_label is TrialLabel.NOT_RELEVANT
    assert result.score == pytest.approx(0.5 - 1.0)


def test_aggregate_unknown_stays_undetermined_and_is_penalised() -> None:
    parsed = build_parsed("NCT1", 4, 0)
    assessment = build_assessment(
        "NCT1",
        {
            "NCT1:inclusion:0": EligibilityLabel.MET,
            "NCT1:inclusion:1": EligibilityLabel.MET,
            "NCT1:inclusion:2": EligibilityLabel.MET,
            "NCT1:inclusion:3": EligibilityLabel.UNKNOWN,
        },
    )
    result = aggregate(assessment, parsed, RANKING)
    assert result.trial_label is None  # candidate for a clarifying question
    assert result.score == pytest.approx(0.75 - RANKING.unknown_penalty * 0.25)


def test_aggregate_unknown_exclusion_blocks_eligible_label() -> None:
    parsed = build_parsed("NCT1", 1, 1)
    assessment = build_assessment(
        "NCT1",
        {
            "NCT1:inclusion:0": EligibilityLabel.MET,
            "NCT1:exclusion:0": EligibilityLabel.UNKNOWN,
        },
    )
    result = aggregate(assessment, parsed, RANKING)
    assert result.trial_label is None
    assert result.score == pytest.approx(1.0)


def test_aggregate_unknown_penalty_orders_more_unknowns_lower() -> None:
    parsed = build_parsed("NCT1", 4, 0)
    one_unknown = aggregate(
        build_assessment(
            "NCT1",
            {
                "NCT1:inclusion:0": EligibilityLabel.MET,
                "NCT1:inclusion:1": EligibilityLabel.MET,
                "NCT1:inclusion:2": EligibilityLabel.MET,
                "NCT1:inclusion:3": EligibilityLabel.UNKNOWN,
            },
        ),
        parsed,
        RANKING,
    )
    two_unknown = aggregate(
        build_assessment(
            "NCT1",
            {
                "NCT1:inclusion:0": EligibilityLabel.MET,
                "NCT1:inclusion:1": EligibilityLabel.MET,
                "NCT1:inclusion:2": EligibilityLabel.UNKNOWN,
                "NCT1:inclusion:3": EligibilityLabel.UNKNOWN,
            },
        ),
        parsed,
        RANKING,
    )
    assert one_unknown.score > two_unknown.score


def test_aggregate_treats_missing_verdicts_as_unknown() -> None:
    parsed = build_parsed("NCT1", 2, 0)
    assessment = build_assessment("NCT1", {"NCT1:inclusion:0": EligibilityLabel.MET})
    result = aggregate(assessment, parsed, RANKING)
    assert result.trial_label is None
    assert result.score == pytest.approx(0.5 - RANKING.unknown_penalty * 0.5)


def test_aggregate_demographic_mismatch_is_not_relevant() -> None:
    parsed = build_parsed("NCT1", 2, 1)
    assessment = TrialAssessment(
        patient_id="p1",
        nct_id="NCT1",
        verdicts=[
            CriterionVerdict(
                criterion_id=f"NCT1:{DEMOGRAPHIC_SEX_SUFFIX}",
                label=EligibilityLabel.NOT_MET,
                explanation="female-only trial",
            )
        ],
    )
    result = aggregate(assessment, parsed, RANKING)
    assert result.trial_label is TrialLabel.NOT_RELEVANT
    assert result.score == pytest.approx(-1.0)


def test_aggregate_returns_a_copy() -> None:
    parsed = build_parsed("NCT1", 1, 0)
    assessment = build_assessment("NCT1", {"NCT1:inclusion:0": EligibilityLabel.MET})
    result = aggregate(assessment, parsed, RANKING)
    assert result is not assessment
    assert assessment.trial_label is None
    assert assessment.score == 0.0


def test_rank_orders_by_score_within_one_label_grade() -> None:
    a = TrialAssessment(patient_id="p1", nct_id="NCT3", score=0.5)
    b = TrialAssessment(patient_id="p1", nct_id="NCT1", score=0.9)
    c = TrialAssessment(patient_id="p1", nct_id="NCT2", score=0.5)

    ordered = rank([a, b, c])
    assert [x.nct_id for x in ordered] == ["NCT1", "NCT2", "NCT3"]
    assert [x.nct_id for x in rank([c, b, a])] == ["NCT1", "NCT2", "NCT3"]


def test_rank_puts_the_label_grade_before_the_score() -> None:
    """design.md §3: eligible > undetermined > excluded > not relevant."""
    eligible = TrialAssessment(
        patient_id="p1", nct_id="NCT4", trial_label=TrialLabel.ELIGIBLE, score=0.2
    )
    undetermined = TrialAssessment(patient_id="p1", nct_id="NCT3", score=0.9)
    excluded = TrialAssessment(
        patient_id="p1", nct_id="NCT2", trial_label=TrialLabel.EXCLUDED, score=1.0
    )
    not_relevant = TrialAssessment(
        patient_id="p1", nct_id="NCT1", trial_label=TrialLabel.NOT_RELEVANT, score=1.0
    )

    ordered = rank([not_relevant, excluded, undetermined, eligible])

    # A high-scoring excluded trial still never outranks a worse eligible one.
    assert [x.nct_id for x in ordered] == ["NCT4", "NCT3", "NCT2", "NCT1"]
    assert [label_grade(x.trial_label) for x in ordered] == [3, 2, 1, 0]


# --------------------------------------------------------------------------- #
# question_gen
# --------------------------------------------------------------------------- #


def test_generate_questions_prioritises_exclusion_and_caps_the_count() -> None:
    parsed = ParsedCriteria(
        nct_id="NCT1",
        inclusion=[
            criterion("NCT1:inclusion:0", CriterionType.INCLUSION, "ECOG 0-1"),
            criterion("NCT1:inclusion:1", CriterionType.INCLUSION, "Adequate renal function"),
        ],
        exclusion=[criterion("NCT1:exclusion:0", CriterionType.EXCLUSION, "Active infection")],
    )
    assessment = TrialAssessment(
        patient_id="p1",
        nct_id="NCT1",
        unknown_criterion_ids=[
            "NCT1:inclusion:0",
            "NCT1:inclusion:1",
            "NCT1:exclusion:0",
        ],
    )
    llm = FakeLLM(
        {
            QuestionBatch: QuestionBatch(
                questions=[
                    GeneratedQuestion(
                        text="How active are you day to day?",
                        criterion_ids=["NCT1:inclusion:0"],
                    ),
                    GeneratedQuestion(
                        text="Do you have any infection right now?",
                        criterion_ids=["NCT1:exclusion:0"],
                    ),
                    GeneratedQuestion(
                        text="Have you had recent kidney blood tests?",
                        criterion_ids=["NCT1:inclusion:1", "NCT1:inclusion:99"],
                    ),
                ]
            )
        }
    )

    questions = generate_questions(llm, make_profile(), [assessment], {"NCT1": parsed}, MODELS)

    assert llm.calls[0]["model"] == MODELS.reason_model
    assert [q.priority for q in questions] == [0, 1, 1]
    assert questions[0].text == "Do you have any infection right now?"
    assert [q.question_id for q in questions] == ["q1", "q2", "q3"]
    # The invented criterion id is filtered out.
    assert questions[2].criterion_ids == ["NCT1:inclusion:1"]

    capped = generate_questions(
        llm, make_profile(), [assessment], {"NCT1": parsed}, MODELS, max_questions=2
    )
    assert len(capped) == 2
    assert capped[0].priority == 0


def test_generate_questions_without_unknowns_skips_the_llm() -> None:
    llm = FakeLLM()
    assessment = TrialAssessment(patient_id="p1", nct_id="NCT1")
    assert generate_questions(llm, make_profile(), [assessment], {}, MODELS) == []
    assert llm.calls == []


# --------------------------------------------------------------------------- #
# simulator
# --------------------------------------------------------------------------- #


def test_answer_questions_defaults_to_dont_know_for_missing_answers() -> None:
    questions = [
        ClarifyingQuestion(question_id="q1", text="Do you smoke?"),
        ClarifyingQuestion(question_id="q2", text="What is your creatinine?"),
    ]
    llm = FakeLLM(
        {
            AnswerBatch: AnswerBatch(
                answers=[SimulatedAnswer(question_id="q1", answer_text="I quit 10 years ago.")]
            )
        }
    )

    answers = answer_questions(llm, "Former smoker, quit 10 years ago.", questions, MODELS)

    assert llm.calls[0]["model"] == MODELS.extract_model
    assert [a.question_id for a in answers] == ["q1", "q2"]
    assert answers[0].answer_text == "I quit 10 years ago."
    assert answers[1].answer_text == "The patient does not know."


def test_answer_questions_with_no_questions_skips_the_llm() -> None:
    llm = FakeLLM()
    assert answer_questions(llm, "ground truth", [], MODELS) == []
    assert llm.calls == []


# --------------------------------------------------------------------------- #
# reporter
# --------------------------------------------------------------------------- #


def test_build_report_contains_disclaimer_and_top_trial() -> None:
    trial = make_trial()
    parsed = parse_criteria(FakeLLM(), trial, MODELS, use_llm=False)
    top = TrialAssessment(
        patient_id="p1",
        nct_id="NCT00000001",
        verdicts=[
            CriterionVerdict(
                criterion_id="NCT00000001:inclusion:0",
                label=EligibilityLabel.MET,
                evidence=["A 62-year-old man"],
                explanation="Patient is 62.",
            ),
            CriterionVerdict(
                criterion_id="NCT00000001:inclusion:2",
                label=EligibilityLabel.UNKNOWN,
            ),
        ],
        trial_label=TrialLabel.ELIGIBLE,
        score=0.8,
        unknown_criterion_ids=["NCT00000001:inclusion:2"],
    )
    other = TrialAssessment(
        patient_id="p1", nct_id="NCT00000002", trial_label=TrialLabel.EXCLUDED, score=-1.0
    )

    report = build_report(
        make_profile(),
        [top, other],
        {"NCT00000001": trial},
        top_k=1,
        parsed_by_nct={"NCT00000001": parsed},
    )

    assert DISCLAIMER in report
    assert "본 결과는 의학적 자문이 아닌 연구·참고용입니다" in report
    assert "NCT00000001" in report
    assert trial.title in report
    assert "NCT00000002" not in report  # top_k=1
    assert "0.800" in report
    assert "적합 (eligible)" in report
    assert "A 62-year-old man" in report
    assert "ECOG performance status of 0 or 1" in report  # unknown rendered with text


def test_build_report_handles_empty_ranking() -> None:
    report = build_report(make_profile(), [], {})
    assert DISCLAIMER in report
    assert "후보 임상시험이 없습니다" in report


def test_build_report_marks_undetermined_trials() -> None:
    assessment = TrialAssessment(patient_id="p1", nct_id="NCT9", score=0.4)
    report = build_report(make_profile(), [assessment], {})
    assert "미정 (추가 확인 필요)" in report
    assert "(제목 정보 없음)" in report
