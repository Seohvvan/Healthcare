"""Contract sanity checks for the shared schemas."""

from trialmatch.schemas import (
    Criterion,
    CriterionType,
    CriterionVerdict,
    EligibilityLabel,
    ParsedCriteria,
    PatientProfile,
    TrialLabel,
    TrialRecord,
)


def test_parsed_criteria_all_criteria_order():
    inc = Criterion(criterion_id="NCT1:inclusion:0", ctype=CriterionType.INCLUSION, text="age > 18")
    exc = Criterion(criterion_id="NCT1:exclusion:0", ctype=CriterionType.EXCLUSION, text="pregnant")
    parsed = ParsedCriteria(nct_id="NCT1", inclusion=[inc], exclusion=[exc])
    assert [c.criterion_id for c in parsed.all_criteria] == [
        "NCT1:inclusion:0",
        "NCT1:exclusion:0",
    ]


def test_schema_round_trip():
    record = TrialRecord(nct_id="NCT00000001", title="A Study", conditions=["Migraine"])
    assert TrialRecord.model_validate_json(record.model_dump_json()) == record

    profile = PatientProfile(patient_id="S005", raw_text="34yo woman with migraine")
    assert profile.unknown_fields == []


def test_enum_values_are_stable():
    # These string values are part of the LLM structured-output contract.
    assert EligibilityLabel.MET.value == "met"
    assert EligibilityLabel.NOT_MET.value == "not_met"
    assert EligibilityLabel.UNKNOWN.value == "unknown"
    assert TrialLabel.ELIGIBLE.value == "eligible"


def test_verdict_defaults():
    verdict = CriterionVerdict(criterion_id="x", label=EligibilityLabel.UNKNOWN)
    assert verdict.evidence == [] and verdict.explanation == ""
