"""Masked-field evaluation tests. No network, no API key: every LLM call is faked.

The metrics are pure functions over run-log dicts, so most tests never touch the
pipeline; the runner tests use a deterministic fake client that only tells the
masked note from the full one by the sentence the mask removes.
"""

from __future__ import annotations

import json
import re
from typing import Any

import pytest
from conftest import FakeLLM  # shared test double, see tests/conftest.py

from trialmatch.agents.criteria_parser import CriteriaSplit
from trialmatch.agents.matcher import AssessmentBatch, CriterionAssessment
from trialmatch.agents.profiler import ProfileExtraction
from trialmatch.agents.prompts import SIMULATOR_UNKNOWN_ANSWER
from trialmatch.agents.question_gen import GeneratedQuestion, QuestionBatch
from trialmatch.agents.simulator import AnswerBatch, SimulatedAnswer
from trialmatch.config import RetrievalConfig, Settings
from trialmatch.data import write_jsonl
from trialmatch.eval import masked
from trialmatch.eval.masked import (
    GeneratedPatient,
    MaskedFact,
    MaskedPatient,
    fact_metrics,
    fact_recovered,
    flipped_count,
    format_table,
    full_note,
    generate_patients,
    label_agreement,
    load_patients,
    main,
    masked_note,
    run_masked_eval,
    save_masked_result,
    summarize,
    top1_label_preserved,
    write_patients,
)
from trialmatch.retrieval import BM25Index
from trialmatch.schemas import EligibilityLabel, TrialRecord

SETTINGS = Settings()

SENTENCES = [
    "A 62-year-old man presents with a persistent cough and weight loss.",
    "He was diagnosed with stage IV lung adenocarcinoma last year.",
    "A biopsy confirmed non-small cell lung cancer.",
    "He is on no active cancer treatment at the moment.",
    "Recent imaging showed no brain metastases.",
]

PATIENT = MaskedPatient(
    patient_id="M001",
    condition="stage IV lung adenocarcinoma",
    sentences=SENTENCES,
    facts=[
        MaskedFact(
            field="biopsy_confirmation",
            sentence_indices=[2],
            keywords=["biopsy", "biopsy confirmed"],
        ),
        MaskedFact(
            field="current_medication",
            sentence_indices=[3],
            keywords=["no active cancer treatment"],
        ),
    ],
)


# --------------------------------------------------------------------------- #
# dataset: masking and validation
# --------------------------------------------------------------------------- #


def test_full_note_joins_every_sentence_in_order() -> None:
    assert full_note(PATIENT) == " ".join(SENTENCES)


def test_masked_note_drops_only_the_facts_sentences_and_keeps_the_order() -> None:
    note = masked_note(
        PATIENT,
        MaskedFact(field="two_sentences", sentence_indices=[1, 3], keywords=["x"]),
    )

    assert note == " ".join([SENTENCES[0], SENTENCES[2], SENTENCES[4]])
    assert SENTENCES[1] not in note
    assert SENTENCES[3] not in note


def test_write_and_load_patients_round_trip(tmp_path) -> None:
    path = write_patients([PATIENT], tmp_path / "nested" / "masked_patients.json")

    loaded = load_patients(path)

    assert [p.patient_id for p in loaded] == ["M001"]
    assert loaded[0].facts[0].field == "biopsy_confirmation"
    assert loaded[0].sentences == SENTENCES
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert set(payload) == {"patients"}


def _write_patients_json(tmp_path, patients: Any):
    path = tmp_path / "patients.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"patients": patients}), encoding="utf-8")
    return path


def _patient_payload(**overrides: Any) -> dict:
    payload = {
        "patient_id": "M001",
        "sentences": ["She has about 4 attacks per month.", "She takes no medication."],
        "facts": [
            {
                "field": "attack_frequency",
                "sentence_indices": [0],
                "keywords": ["4 attacks"],
            }
        ],
    }
    payload.update(overrides)
    return payload


def test_load_patients_rejects_an_out_of_range_sentence_index(tmp_path) -> None:
    path = _write_patients_json(
        tmp_path,
        [
            _patient_payload(
                facts=[{"field": "dose", "sentence_indices": [5], "keywords": ["mg"]}]
            )
        ],
    )

    with pytest.raises(ValueError, match=r"sentence index 5 is out of range \(0\.\.1\)"):
        load_patients(path)


def test_load_patients_rejects_an_empty_mask(tmp_path) -> None:
    path = _write_patients_json(
        tmp_path,
        [_patient_payload(facts=[{"field": "dose", "sentence_indices": [], "keywords": ["mg"]}])],
    )

    with pytest.raises(ValueError, match="sentence_indices must not be empty"):
        load_patients(path)


def test_load_patients_rejects_a_fact_without_keywords(tmp_path) -> None:
    """Without a keyword `fact_recovered` can never fire — the row would be a lie."""
    path = _write_patients_json(
        tmp_path,
        [_patient_payload(facts=[{"field": "dose", "sentence_indices": [0], "keywords": []}])],
    )

    with pytest.raises(ValueError, match="keywords must not be empty"):
        load_patients(path)


def test_load_patients_rejects_a_keyword_absent_from_the_masked_sentences(tmp_path) -> None:
    path = _write_patients_json(
        tmp_path,
        [
            _patient_payload(
                facts=[
                    {
                        "field": "dose",
                        "sentence_indices": [0],
                        "keywords": ["200 mg twice daily"],
                    }
                ]
            )
        ],
    )

    with pytest.raises(ValueError, match="does not appear in the sentences the mask removes"):
        load_patients(path)


def test_load_patients_rejects_a_keyword_the_masked_note_still_states(tmp_path) -> None:
    """Deterministic leakage check: 'she' survives the mask, so recovery is vacuous."""
    path = _write_patients_json(
        tmp_path,
        [
            _patient_payload(
                facts=[{"field": "dose", "sentence_indices": [0], "keywords": ["she"]}]
            )
        ],
    )

    with pytest.raises(ValueError, match="still appears in the masked note"):
        load_patients(path)


def test_load_patients_rejects_a_duplicate_patient_id(tmp_path) -> None:
    path = _write_patients_json(tmp_path, [_patient_payload(), _patient_payload()])

    with pytest.raises(ValueError, match="duplicate patient_id 'M001'"):
        load_patients(path)


def test_load_patients_rejects_a_duplicate_fact_field(tmp_path) -> None:
    fact = {"field": "attack_frequency", "sentence_indices": [0], "keywords": ["4 attacks"]}
    path = _write_patients_json(tmp_path, [_patient_payload(facts=[fact, fact])])

    with pytest.raises(ValueError, match="duplicate fact field 'attack_frequency'"):
        load_patients(path)


def test_load_patients_enforces_only_the_structural_minimums(tmp_path) -> None:
    """A 2-sentence, 1-fact record is short of the generation targets but usable."""
    path = _write_patients_json(tmp_path, [_patient_payload()])

    assert len(load_patients(path)[0].facts) == 1

    too_short = _write_patients_json(
        tmp_path / "short", [_patient_payload(sentences=["Only one sentence."])]
    )
    with pytest.raises(ValueError, match="fewer than 2 sentences"):
        load_patients(too_short)

    no_facts = _write_patients_json(tmp_path / "empty", [_patient_payload(facts=[])])
    with pytest.raises(ValueError, match="has no facts"):
        load_patients(no_facts)


def test_load_patients_rejects_a_file_without_a_patients_list(tmp_path) -> None:
    path = tmp_path / "patients.json"
    path.write_text(json.dumps({"rows": []}), encoding="utf-8")

    with pytest.raises(TypeError, match="'patients' list"):
        load_patients(path)


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #


def run_log(
    rounds: list[dict] | None = None,
    ranking: list[dict] | None = None,
    errors: list[dict] | None = None,
) -> dict:
    return {
        "patient_id": "M001",
        "rounds": rounds or [],
        "ranking": ranking or [],
        "errors": errors or [],
    }


def answers_round(*texts: str, reassessed: list[str] | None = None) -> dict:
    return {
        "round": 1,
        "questions": [],
        "answers": [
            {"question_id": f"q{i + 1}", "answer_text": text} for i, text in enumerate(texts)
        ],
        "reassessed": reassessed or [],
    }


def test_fact_recovered_matches_a_keyword_case_insensitively() -> None:
    log = run_log([answers_round("I get about 4 Attacks Per Month.")])

    assert fact_recovered(log, ["attacks per month", "4 attacks"]) is True


def test_fact_recovered_is_false_when_no_keyword_appears() -> None:
    log = run_log([answers_round("I have had headaches for years.")])

    assert fact_recovered(log, ["attacks per month"]) is False
    assert fact_recovered(log, []) is False  # no keyword can ever match


def test_fact_recovered_ignores_the_unknown_answer() -> None:
    """The unknown sentinel carries no information, whatever it happens to contain."""
    log = run_log([answers_round(SIMULATOR_UNKNOWN_ANSWER)])

    assert fact_recovered(log, [SIMULATOR_UNKNOWN_ANSWER.lower()]) is False


def test_fact_recovered_scans_every_round() -> None:
    log = run_log(
        [
            answers_round(SIMULATOR_UNKNOWN_ANSWER),
            answers_round("I take metformin twice a day."),
        ]
    )

    assert fact_recovered(log, ["metformin"]) is True


def test_flipped_count_counts_reassessed_trials_that_ended_determined() -> None:
    log = run_log(
        rounds=[
            answers_round("yes", reassessed=["NCT0001", "NCT0002"]),
            answers_round("yes", reassessed=["NCT0003"]),
        ],
        ranking=[
            {"nct_id": "NCT0001", "label": "eligible", "score": 1.0},
            {"nct_id": "NCT0002", "label": None, "score": 0.4},  # still undetermined
            {"nct_id": "NCT0003", "label": "excluded", "score": -1.0},
            {"nct_id": "NCT0004", "label": "eligible", "score": 0.9},  # never reassessed
        ],
    )

    assert flipped_count(log) == 2


def test_flipped_count_is_zero_without_a_question_loop() -> None:
    assert flipped_count(run_log(ranking=[{"nct_id": "NCT0001", "label": "eligible"}])) == 0


def test_label_agreement_over_the_shared_trials_treats_none_as_a_label() -> None:
    baseline = run_log(
        ranking=[
            {"nct_id": "NCT0001", "label": "eligible"},
            {"nct_id": "NCT0002", "label": "excluded"},
            {"nct_id": "NCT0003", "label": None},
            {"nct_id": "NCT0009", "label": "eligible"},  # absent from the masked run
        ]
    )
    maskedrun = run_log(
        ranking=[
            {"nct_id": "NCT0001", "label": "eligible"},
            {"nct_id": "NCT0002", "label": None},
            {"nct_id": "NCT0003", "label": None},
        ]
    )

    assert label_agreement(baseline, maskedrun) == pytest.approx(2 / 3)


def test_label_agreement_is_none_without_a_shared_trial() -> None:
    baseline = run_log(ranking=[{"nct_id": "NCT0001", "label": "eligible"}])
    maskedrun = run_log(ranking=[{"nct_id": "NCT0002", "label": "eligible"}])

    assert label_agreement(baseline, maskedrun) is None
    assert label_agreement(run_log(), run_log()) is None


def test_top1_label_preserved_checks_the_label_not_the_rank() -> None:
    baseline = run_log(ranking=[{"nct_id": "NCT0001", "label": "eligible"}])

    same = run_log(
        ranking=[
            {"nct_id": "NCT0002", "label": "eligible"},
            {"nct_id": "NCT0001", "label": "eligible"},
        ]
    )
    flipped = run_log(ranking=[{"nct_id": "NCT0001", "label": None}])
    dropped = run_log(ranking=[{"nct_id": "NCT0002", "label": "eligible"}])

    assert top1_label_preserved(baseline, same) is True  # rank may move, label may not
    assert top1_label_preserved(baseline, flipped) is False
    assert top1_label_preserved(baseline, dropped) is False
    assert top1_label_preserved(run_log(), same) is False  # nothing to preserve


def test_fact_metrics_carries_the_denominators_it_was_computed_over() -> None:
    baseline = run_log(
        ranking=[
            {"nct_id": "NCT0001", "label": "eligible"},
            {"nct_id": "NCT0002", "label": "excluded"},
            {"nct_id": "NCT0009", "label": "eligible"},
        ]
    )
    maskedrun = run_log(
        ranking=[
            {"nct_id": "NCT0001", "label": "eligible"},
            {"nct_id": "NCT0002", "label": "excluded"},
        ],
        errors=[{"nct_id": "NCT0009", "error": "overloaded_error"}],
    )

    metrics = fact_metrics(baseline, maskedrun, ["biopsy"])

    assert metrics["label_agreement"] == pytest.approx(1.0)
    assert metrics["n_baseline"] == 3
    assert metrics["n_masked"] == 2
    assert metrics["n_shared"] == 2  # a perfect agreement over 2 of 3 trials
    assert metrics["masked_errors"] == 1


def _row(**overrides: Any) -> dict:
    row = {
        "patient_id": "M001",
        "field": "a",
        "recovered": True,
        "flipped_count": 2,
        "label_agreement": 1.0,
        "top1_label_preserved": True,
        "n_baseline": 6,
        "n_masked": 6,
        "n_shared": 6,
        "masked_errors": 0,
    }
    row.update(overrides)
    return row


def test_summarize_reports_every_rate_with_its_denominator() -> None:
    rows = [
        _row(),
        _row(
            field="b",
            recovered=False,
            flipped_count=0,
            label_agreement=None,
            top1_label_preserved=False,
            masked_errors=3,
        ),
    ]

    summary = summarize(rows)

    assert summary["num_patients"] == 1
    assert summary["num_facts"] == 2
    assert summary["num_recovered"] == 1
    assert summary["recovery_rate"] == pytest.approx(0.5)
    assert summary["mean_flipped_count"] == pytest.approx(1.0)
    assert summary["mean_label_agreement"] == pytest.approx(1.0)
    assert summary["num_label_agreement"] == 1  # averaged over 1 of the 2 facts
    assert summary["num_top1_label_preserved"] == 1
    assert summary["top1_label_preserved_rate"] == pytest.approx(0.5)
    assert summary["num_masked_errors"] == 3


def test_summarize_of_an_empty_batch_does_not_divide_by_zero() -> None:
    summary = summarize([])

    assert summary["num_facts"] == 0
    assert summary["num_recovered"] == 0
    assert summary["recovery_rate"] == 0.0
    assert summary["mean_label_agreement"] is None
    assert "n/a" in format_table({"results": [], "summary": summary})


def test_format_table_flags_a_thin_comparison_and_prints_the_denominators() -> None:
    rows = [_row(n_baseline=2, n_masked=2, n_shared=2), _row(field="b")]

    table = format_table({"results": rows, "summary": summarize(rows)})

    assert "warning: M001/a compared only 2 shared trial(s)" in table
    assert "M001/b" not in table  # 6 shared trials is not thin
    assert "facts recovered:       2 / 2 (1.0000)" in table
    assert "mean label agreement:  1.0000 (over 2 of 2 facts)" in table
    assert "top-1 label preserved: 2 / 2 (1.0000)" in table


# --------------------------------------------------------------------------- #
# fake client for the runner tests
# --------------------------------------------------------------------------- #


LUNG_CRITERIA = """Inclusion Criteria:

* Adults aged 18 years or older
* Histologically confirmed non-small cell lung cancer

Exclusion Criteria:

* Untreated active brain metastases
"""

SECOND_CRITERIA = """Inclusion Criteria:

* Metastatic lung cancer previously treated with chemotherapy

Exclusion Criteria:

* Pregnancy
"""

# One unstructured blob: no section header and well over the 300-character blob
# threshold, so `parse_criteria` cannot split it and falls back to the LLM.
BLOB_CRITERIA = (
    "Participants must be adults of at least 18 years of age with histologically or "
    "cytologically documented advanced lung cancer that has progressed after at least "
    "one line of systemic therapy, an ECOG performance status of 0 or 1, measurable "
    "disease by RECIST v1.1, adequate bone marrow, hepatic and renal function, and a "
    "life expectancy of at least three months; participants with untreated central "
    "nervous system metastases, another malignancy within the previous five years, or "
    "who are pregnant or breastfeeding at the time of enrolment are not eligible."
)

TRIALS: dict[str, TrialRecord] = {
    trial.nct_id: trial
    for trial in [
        TrialRecord(
            nct_id="NCT0001",
            title="Osimertinib in advanced non-small cell lung cancer",
            conditions=["Non-Small Cell Lung Cancer"],
            brief_summary="A study of osimertinib in adults with advanced lung adenocarcinoma.",
            eligibility_text=LUNG_CRITERIA,
        ),
        TrialRecord(
            nct_id="NCT0002",
            title="Chemotherapy combination for metastatic lung cancer",
            conditions=["Lung Cancer"],
            brief_summary="A chemotherapy study in metastatic lung cancer.",
            eligibility_text=SECOND_CRITERIA,
        ),
        TrialRecord(
            nct_id="NCT0003",
            title="Preventive treatment of chronic migraine",
            conditions=["Migraine"],
            brief_summary="A study of migraine prophylaxis.",
            eligibility_text="Inclusion Criteria:\n\n* Chronic migraine for at least 6 months\n",
        ),
        TrialRecord(
            nct_id="NCT0004",
            title="Second-line therapy for advanced non-small cell lung cancer",
            conditions=["Non-Small Cell Lung Cancer"],
            brief_summary="A second-line study in adults with advanced lung adenocarcinoma.",
            eligibility_text=BLOB_CRITERIA,
        ),
    ]
}

# The index covers the whole snapshot (BM25 idf turns negative on a two-document
# corpus); the smaller `trials` mapping is what limits the candidates.
INDEX = BM25Index.build(TRIALS.values())
LUNG_TRIALS = {nct: TRIALS[nct] for nct in ("NCT0001", "NCT0002")}
# Same corpus plus the unstructured-blob trial, which forces the LLM criteria path.
BLOB_TRIALS = {nct: TRIALS[nct] for nct in ("NCT0001", "NCT0002", "NCT0004")}

PROFILE_EXTRACTION = ProfileExtraction(
    age_years=62.0,
    sex="male",
    conditions=["stage IV lung adenocarcinoma"],
    search_queries=["advanced non-small cell lung cancer", "lung adenocarcinoma"],
)

# Only this criterion depends on the masked sentence.
BIOPSY_CRITERION = "NCT0001:inclusion:1"
BIOPSY_EVIDENCE = "biopsy confirmed"
SIMULATOR_ANSWER = "Yes, a biopsy confirmed non-small cell lung cancer."

_CRITERION_ID = re.compile(r"\[(NCT\d+:(?:inclusion|exclusion):\d+)\]")


class MaskedEvalLLM(FakeLLM):
    """Deterministic client: the biopsy criterion stays UNKNOWN until the text says so.

    Every prompt carries the patient text (the note plus any clarification), so
    this reproduces the real behaviour under masking without any scripting: the
    masked run starts undecided and only the simulator's answer settles it.
    """

    def __init__(self) -> None:
        super().__init__({})
        self.profile_calls = 0

    def structured(self, **kwargs: Any) -> Any:  # type: ignore[override]
        schema = kwargs["schema"]
        if schema is ProfileExtraction:
            self.profile_calls += 1
            self.responses[schema] = PROFILE_EXTRACTION
        elif schema is AssessmentBatch:
            self.responses[schema] = self._assess(kwargs["user"])
        elif schema is CriteriaSplit:
            self.responses[schema] = CriteriaSplit(
                inclusion=["Adults aged 18 years or older", "Advanced lung cancer"],
                exclusion=["Untreated central nervous system metastases"],
            )
        elif schema is QuestionBatch:
            self.responses[schema] = QuestionBatch(
                questions=[
                    GeneratedQuestion(
                        text="Was your lung cancer confirmed with a biopsy?",
                        criterion_ids=[BIOPSY_CRITERION],
                    )
                ]
            )
        elif schema is AnswerBatch:
            self.responses[schema] = AnswerBatch(
                answers=[SimulatedAnswer(question_id="q1", answer_text=SIMULATOR_ANSWER)]
            )
        return super().structured(**kwargs)

    @staticmethod
    def _assess(user: str) -> AssessmentBatch:
        knows_biopsy = BIOPSY_EVIDENCE in user.casefold()
        verdicts = []
        for criterion_id in _CRITERION_ID.findall(user):
            if ":exclusion:" in criterion_id:
                label = EligibilityLabel.NOT_MET
            elif criterion_id == BIOPSY_CRITERION and not knows_biopsy:
                label = EligibilityLabel.UNKNOWN
            else:
                label = EligibilityLabel.MET
            verdicts.append(
                CriterionAssessment(criterion_id=criterion_id, label=label, explanation="because")
            )
        return AssessmentBatch(verdicts=verdicts)


# --------------------------------------------------------------------------- #
# run_masked_eval
# --------------------------------------------------------------------------- #


def test_run_masked_eval_scores_recovery_flips_and_agreement() -> None:
    llm = MaskedEvalLLM()

    result = run_masked_eval(
        llm,
        [PATIENT],
        LUNG_TRIALS,
        INDEX,
        SETTINGS,
        question_rounds=2,
    )

    assert result["errors"] == []
    rows = {row["field"]: row for row in result["results"]}
    assert set(rows) == {"biopsy_confirmation", "current_medication"}

    # 1. The masked fact was asked about and the simulator's answer recovered it,
    #    which flipped the undetermined trial to a determined label.
    recovered = rows["biopsy_confirmation"]
    assert recovered["recovered"] is True
    assert recovered["flipped_count"] == 1
    assert recovered["label_agreement"] == pytest.approx(1.0)
    assert recovered["top1_label_preserved"] is True
    assert recovered["n_baseline"] == recovered["n_masked"] == recovered["n_shared"] == 2
    assert recovered["masked_errors"] == 0

    # 2. Masking a fact no criterion asks about leaves nothing undetermined, so
    #    no question is generated and nothing can be recovered.
    untouched = rows["current_medication"]
    assert untouched["recovered"] is False
    assert untouched["flipped_count"] == 0
    assert untouched["label_agreement"] == pytest.approx(1.0)
    assert result["run_logs"]["M001"]["facts"]["current_medication"]["rounds"] == []

    # 3. One baseline plus one run per fact, all sharing the criteria cache.
    assert llm.profile_calls == 3
    assert "CriteriaSplit" not in {call["schema"].__name__ for call in llm.calls}

    summary = result["summary"]
    assert summary["num_facts"] == 2
    assert summary["num_recovered"] == 1
    assert summary["recovery_rate"] == pytest.approx(0.5)
    assert summary["mean_flipped_count"] == pytest.approx(0.5)
    assert summary["top1_label_preserved_rate"] == pytest.approx(1.0)


def _recording_run_pipeline(calls: list[dict]):
    def fake_run_pipeline(**kwargs: Any):
        calls.append(kwargs)
        return None, {
            "patient_id": kwargs["patient_id"],
            "candidate_ids": [],
            "rounds": [],
            "ranking": [],
            "errors": [],
        }

    return fake_run_pipeline


def test_run_masked_eval_gives_the_baseline_the_same_question_budget(monkeypatch) -> None:
    """A/B control: the arms differ in the removed sentences and nothing else."""
    calls: list[dict] = []
    monkeypatch.setattr(masked, "run_pipeline", _recording_run_pipeline(calls))

    run_masked_eval(
        MaskedEvalLLM(), [PATIENT], LUNG_TRIALS, INDEX, SETTINGS, question_rounds=3
    )

    assert len(calls) == 3  # baseline + one masked run per fact
    assert [call["question_rounds"] for call in calls] == [3, 3, 3]
    assert all(call["answer_provider"] is not None for call in calls)
    # The simulator answers from the full note in every arm, so the note is the
    # only thing that differs.
    assert [call["note"] for call in calls] == [
        full_note(PATIENT),
        masked_note(PATIENT, PATIENT.facts[0]),
        masked_note(PATIENT, PATIENT.facts[1]),
    ]
    assert len({id(call["answer_provider"]) for call in calls}) == 1


def test_run_masked_eval_logs_every_assessed_trial_by_default(monkeypatch) -> None:
    """Comparison metrics read the ranking, so it must not be truncated by top-k."""
    calls: list[dict] = []
    monkeypatch.setattr(masked, "run_pipeline", _recording_run_pipeline(calls))
    settings = SETTINGS.model_copy(update={"retrieval": RetrievalConfig(candidates_k=7)})

    result = run_masked_eval(MaskedEvalLLM(), [PATIENT], LUNG_TRIALS, INDEX, settings)

    assert {call["top_k"] for call in calls} == {7}
    assert result["config"]["top_k"] == 7


def test_run_masked_eval_pins_the_baseline_candidates_for_every_arm(monkeypatch) -> None:
    """Profiling is stochastic, so only the baseline may retrieve; the masked
    arms must judge the exact same trials or the metrics mix in retrieval
    variance."""
    from trialmatch import pipeline as pipeline_module

    retrieve_calls: list[str] = []
    original_retrieve = pipeline_module._retrieve

    def counting_retrieve(profile, index, trials, settings):
        retrieve_calls.append(profile.patient_id)
        return original_retrieve(profile, index, trials, settings)

    monkeypatch.setattr(pipeline_module, "_retrieve", counting_retrieve)

    result = run_masked_eval(MaskedEvalLLM(), [PATIENT], LUNG_TRIALS, INDEX, SETTINGS)

    assert retrieve_calls == ["M001"]  # baseline only, once per patient
    logs = result["run_logs"]["M001"]
    baseline_ids = logs["baseline"]["candidate_ids"]
    assert len(logs["facts"]) == len(PATIENT.facts)
    assert all(
        log["candidate_ids"] == baseline_ids for log in logs["facts"].values()
    )


def test_run_masked_eval_caps_facts_and_reports_what_it_dropped(caplog) -> None:
    with caplog.at_level("WARNING"):
        result = run_masked_eval(
            MaskedEvalLLM(),
            [PATIENT],
            LUNG_TRIALS,
            INDEX,
            SETTINGS,
            question_rounds=1,
            max_facts=1,
        )

    assert [row["field"] for row in result["results"]] == ["biopsy_confirmation"]
    assert "current_medication" in caplog.text  # no silent cap


def test_run_masked_eval_contains_a_failing_masked_run() -> None:
    class SecondFactExplodesLLM(MaskedEvalLLM):
        def structured(self, **kwargs: Any) -> Any:
            # Calls 1 and 2 are the baseline and the first fact; the third run's
            # profiling raises, which run_pipeline does not contain.
            if kwargs["schema"] is ProfileExtraction and self.profile_calls == 2:
                self.profile_calls += 1
                raise RuntimeError("overloaded_error")
            return super().structured(**kwargs)

    result = run_masked_eval(
        SecondFactExplodesLLM(),
        [PATIENT],
        LUNG_TRIALS,
        INDEX,
        SETTINGS,
        question_rounds=1,
    )

    assert [row["field"] for row in result["results"]] == ["biopsy_confirmation"]
    assert result["errors"] == [
        {
            "patient_id": "M001",
            "field": "current_medication",
            "stage": "masked",
            "error": "overloaded_error",
        }
    ]


def test_run_masked_eval_parses_each_trial_once_across_every_arm() -> None:
    """The shared parsed_cache is what keeps the criteria parse off the hot path.

    NCT0004's criteria are one unstructured blob, so the rule-based split cannot
    handle it and `parse_criteria` takes the LLM fallback — exactly once for the
    baseline and both masked runs together.
    """
    llm = MaskedEvalLLM()

    result = run_masked_eval(llm, [PATIENT], BLOB_TRIALS, INDEX, SETTINGS, question_rounds=1)

    logs = result["run_logs"]["M001"]
    every_run = [logs["baseline"], *logs["facts"].values()]
    assert len(every_run) == 3
    # Without the cache each of these runs would parse the blob itself.
    assert all("NCT0004" in log["candidate_ids"] for log in every_run)

    split_calls = [call for call in llm.calls if call["schema"] is CriteriaSplit]
    assert len(split_calls) == 1
    assert "NCT0004" in split_calls[0]["user"]


def test_save_masked_result_writes_one_json_into_the_log_dir(tmp_path) -> None:
    settings = Settings(log_dir=tmp_path / "runs")
    result = run_masked_eval(
        MaskedEvalLLM(), [PATIENT], LUNG_TRIALS, INDEX, settings, question_rounds=1
    )

    path = save_masked_result(result, settings)

    assert path.parent == tmp_path / "runs"
    assert path.name.startswith("masked-") and path.suffix == ".json"
    written = json.loads(path.read_text(encoding="utf-8"))
    assert {"config", "results", "summary", "errors", "run_logs"} == set(written)
    assert written["run_logs"]["M001"]["baseline"]["patient_id"] == "M001"


# --------------------------------------------------------------------------- #
# generation
# --------------------------------------------------------------------------- #


GENERATED = GeneratedPatient(
    condition="chronic migraine",
    sentences=[
        "A 34-year-old woman reports recurrent severe headaches.",
        "She has about 4 attacks per month.",
        "She uses no hormonal contraception.",
    ],
    facts=[
        MaskedFact(
            field="attack_frequency",
            sentence_indices=[1],
            keywords=["attacks per month", "4 attacks"],
        )
    ],
)


def test_generate_patients_replaces_only_the_leading_s() -> None:
    llm = FakeLLM({GeneratedPatient: GENERATED})

    patients, errors = generate_patients(llm, [("S1S5", "A 34-year-old woman.")], SETTINGS)

    assert [p.patient_id for p in patients] == ["M1S5"]
    assert errors == []


def test_generate_patients_drops_a_leaking_fact_and_keeps_the_patient() -> None:
    """'she' survives the mask, so that fact could never be recovered by a question."""
    llm = FakeLLM(
        {
            GeneratedPatient: GENERATED.model_copy(
                update={
                    "facts": [
                        *GENERATED.facts,
                        MaskedFact(
                            field="contraception", sentence_indices=[2], keywords=["she"]
                        ),
                    ]
                }
            )
        }
    )

    patients, errors = generate_patients(llm, [("S005", "A 34-year-old woman.")], SETTINGS)

    assert [fact.field for fact in patients[0].facts] == ["attack_frequency"]
    assert errors == [
        {
            "topic_id": "S005",
            "field": "contraception",
            "error": "keyword 'she' still appears in the masked note (the mask leaks)",
        }
    ]


def test_generator_prompt_asks_only_for_patient_knowable_masks() -> None:
    """Only the load-bearing contracts: a masked fact must be one the simulated
    patient could answer, a measurement only counts as what the patient was told,
    and the structural fact budget the loader relies on is still stated.
    Vocabulary may be reworded freely."""
    system = masked.GENERATOR_SYSTEM.casefold()

    assert "patient-knowable" in system
    assert "was told" in system
    assert f"{masked.MIN_FACTS}-{masked.MAX_FACTS} maskable facts" in system


def test_generate_patients_skips_a_topic_whose_every_fact_is_invalid() -> None:
    llm = FakeLLM(
        {
            GeneratedPatient: GENERATED.model_copy(
                update={
                    "facts": [MaskedFact(field="dose", sentence_indices=[99], keywords=["mg"])]
                }
            )
        }
    )

    patients, errors = generate_patients(llm, [("S005", "A 34-year-old woman.")], SETTINGS)

    assert patients == []
    assert [error.get("field") for error in errors] == ["dose", None]
    assert "has no facts" in errors[-1]["error"]


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #


def _write_topics(tmp_path, topics: list[dict]):
    path = tmp_path / "topics.json"
    path.write_text(json.dumps({"topics": topics}), encoding="utf-8")
    return path


def test_generate_cli_writes_a_patients_file(tmp_path, capsys, monkeypatch) -> None:
    topics = _write_topics(tmp_path, [{"num": "S005", "title": "A 34-year-old woman."}])
    out = tmp_path / "masked_patients.json"
    llm = FakeLLM({GeneratedPatient: GENERATED})
    monkeypatch.setattr(masked, "create_llm", lambda models: llm)

    assert main(["generate", "--topics", str(topics), "--out", str(out)]) == 0

    payload = json.loads(out.read_text(encoding="utf-8"))
    patient = payload["patients"][0]
    assert patient["patient_id"] == "M005"  # topic id with a leading S -> M
    assert patient["facts"][0]["sentence_indices"] == [1]
    assert llm.calls[0]["model"] == SETTINGS.models.reason_model
    assert llm.calls[0]["max_tokens"] == SETTINGS.models.max_tokens_reason
    stdout = capsys.readouterr().out
    assert "M005\t3 sentences\t1 facts" in stdout
    assert f"wrote 1 patients to {out}" in stdout


def test_generate_cli_skips_an_unusable_patient_and_continues(
    tmp_path, capsys, monkeypatch
) -> None:
    topics = _write_topics(
        tmp_path,
        [
            {"num": "S005", "title": "A 34-year-old woman."},
            {"num": "S006", "title": "A 45-year-old man."},
        ],
    )
    out = tmp_path / "masked_patients.json"

    broken = GENERATED.model_copy(
        update={
            "facts": [
                MaskedFact(field="dose", sentence_indices=[99], keywords=["mg"])
            ]
        }
    )

    class SecondTopicIsBrokenLLM(FakeLLM):
        def structured(self, **kwargs: Any) -> Any:
            self.responses[GeneratedPatient] = (
                broken if "45-year-old" in kwargs["user"] else GENERATED
            )
            return super().structured(**kwargs)

    monkeypatch.setattr(masked, "create_llm", lambda models: SecondTopicIsBrokenLLM({}))

    assert main(["generate", "--topics", str(topics), "--out", str(out)]) == 0

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert [p["patient_id"] for p in payload["patients"]] == ["M005"]
    stdout = capsys.readouterr().out
    assert "warning: topic S006 fact dose:" in stdout
    assert "out of range" in stdout
    assert "error: topic S006:" in stdout


def test_generate_cli_does_not_clobber_the_dataset_when_nothing_was_generated(
    tmp_path, capsys, monkeypatch
) -> None:
    topics = _write_topics(tmp_path, [{"num": "S005", "title": "A 34-year-old woman."}])
    out = write_patients([PATIENT], tmp_path / "masked_patients.json")
    before = out.read_text(encoding="utf-8")

    class OverloadedLLM(FakeLLM):
        def structured(self, **kwargs: Any) -> Any:
            raise RuntimeError("overloaded_error")

    monkeypatch.setattr(masked, "create_llm", lambda models: OverloadedLLM({}))

    assert main(["generate", "--topics", str(topics), "--out", str(out)]) == 1

    assert out.read_text(encoding="utf-8") == before  # the dataset survived
    stdout = capsys.readouterr().out
    assert "error: topic S005: overloaded_error" in stdout
    assert "was left untouched" in stdout


def _cli_inputs(tmp_path):
    trials_path = tmp_path / "trials.jsonl"
    write_jsonl(LUNG_TRIALS.values(), trials_path)
    index_path = tmp_path / "bm25.pkl"
    INDEX.save(index_path)
    patients_path = write_patients([PATIENT], tmp_path / "masked_patients.json")
    return patients_path, trials_path, index_path


def _run_argv(tmp_path, *extra: str) -> list[str]:
    patients_path, trials_path, index_path = _cli_inputs(tmp_path)
    return [
        "run",
        "--patients",
        str(patients_path),
        "--trials",
        str(trials_path),
        "--index",
        str(index_path),
        *extra,
    ]


def test_run_cli_prints_the_per_fact_table_and_the_summary(tmp_path, capsys, monkeypatch) -> None:
    monkeypatch.setattr(masked, "create_llm", lambda models: MaskedEvalLLM())

    exit_code = main(
        _run_argv(tmp_path, "--candidates", "5", "--question-rounds", "1")
    )

    assert exit_code == 0
    stdout = capsys.readouterr().out
    rows = [line for line in stdout.splitlines() if line.startswith("M001 ")]
    assert [line.split()[1] for line in rows] == ["biopsy_confirmation", "current_medication"]
    # recovered, flipped, agreement, n_base, n_mask, n_shared, errors, top1_label
    assert rows[0].split()[2:] == ["yes", "1", "1.0000", "2", "2", "2", "0", "yes"]
    assert "compared only 2 shared trial(s)" in stdout  # the denominator is visible
    assert "facts recovered:       1 / 2 (0.5000)" in stdout
    assert "top-1 label preserved: 2 / 2 (1.0000)" in stdout
    assert "warning: --top-k" not in stdout  # defaults to the candidate budget


def test_run_cli_defaults_top_k_to_the_candidate_budget(tmp_path, monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake_run_masked_eval(*args: Any, **kwargs: Any) -> dict:
        captured.update(kwargs)
        return {
            "config": {},
            "results": [],
            "summary": summarize([]),
            "errors": [],
            "run_logs": {},
        }

    monkeypatch.setattr(masked, "create_llm", lambda models: MaskedEvalLLM())
    monkeypatch.setattr(masked, "run_masked_eval", fake_run_masked_eval)

    assert main(_run_argv(tmp_path, "--candidates", "9")) == 0
    assert captured["top_k"] == 9


def test_run_cli_warns_when_an_explicit_top_k_truncates_the_ranking(
    tmp_path, capsys, monkeypatch
) -> None:
    monkeypatch.setattr(masked, "create_llm", lambda models: MaskedEvalLLM())

    assert main(_run_argv(tmp_path, "--candidates", "5", "--top-k", "2")) == 0

    stdout = capsys.readouterr().out
    assert "warning: --top-k 2 is below --candidates 5" in stdout
    assert "truncated ranking" in stdout


@pytest.mark.parametrize(
    "flag", ["--candidates", "--question-rounds", "--top-k"]
)
def test_run_cli_rejects_a_non_positive_budget(tmp_path, flag) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(_run_argv(tmp_path, flag, "0"))

    assert excinfo.value.code == 2
