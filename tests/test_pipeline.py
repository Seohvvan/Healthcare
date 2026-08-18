"""Pipeline + CLI tests. No network, no API key: every LLM call is faked."""

from __future__ import annotations

import json
from typing import Any

import pytest
from conftest import FakeLLM, ScriptedLLM  # shared test doubles, see tests/conftest.py

from trialmatch import cli
from trialmatch.agents.matcher import AssessmentBatch, CriterionAssessment
from trialmatch.agents.profiler import ProfileExtraction
from trialmatch.agents.question_gen import GeneratedQuestion, QuestionBatch
from trialmatch.agents.reporter import DISCLAIMER
from trialmatch.agents.simulator import AnswerBatch, SimulatedAnswer
from trialmatch.config import Settings
from trialmatch.data import write_jsonl
from trialmatch.pipeline import (
    run_pipeline,
    save_run_log,
    simulator_answer_provider,
)
from trialmatch.retrieval import BM25Index
from trialmatch.schemas import (
    ClarifyingQuestion,
    EligibilityLabel,
    QuestionAnswer,
    TrialLabel,
    TrialRecord,
)

SETTINGS = Settings()

NOTE = (
    "A 62-year-old man with stage IV lung adenocarcinoma diagnosed last year. "
    "He is on no active treatment."
)

LUNG_CRITERIA = """Inclusion Criteria:

* Adults aged 18 years or older
* Histologically confirmed advanced non-small cell lung cancer

Exclusion Criteria:

* Untreated active brain metastases
"""

SECOND_CRITERIA = """Inclusion Criteria:

* Metastatic lung cancer previously treated with chemotherapy

Exclusion Criteria:

* Pregnancy
"""

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
            title="Observational registry of lung cancer care",
            conditions=["Lung Cancer"],
            brief_summary="A registry of adults treated for lung adenocarcinoma.",
            eligibility_text="",  # no criteria at all: must be skipped, not judged
        ),
    ]
}

INDEX = BM25Index.build(TRIALS.values())

PROFILE_EXTRACTION = ProfileExtraction(
    age_years=62.0,
    sex="male",
    conditions=["stage IV lung adenocarcinoma"],
    unknown_fields=["ECOG performance status"],
    search_queries=["advanced non-small cell lung cancer", "lung adenocarcinoma"],
)


def verdict(criterion_id: str, label: EligibilityLabel) -> CriterionAssessment:
    return CriterionAssessment(criterion_id=criterion_id, label=label, explanation="because")


def schemas_called(llm: FakeLLM) -> list[str]:
    return [call["schema"].__name__ for call in llm.calls]


def matcher_calls(llm: FakeLLM) -> list[dict[str, Any]]:
    return [call for call in llm.calls if call["schema"] is AssessmentBatch]


# --------------------------------------------------------------------------- #
# end-to-end funnel
# --------------------------------------------------------------------------- #


def test_run_pipeline_funnel_order_skips_criteria_less_trials_and_reports() -> None:
    llm = FakeLLM(
        {
            ProfileExtraction: PROFILE_EXTRACTION,
            AssessmentBatch: AssessmentBatch(
                verdicts=[
                    verdict("NCT0001:inclusion:0", EligibilityLabel.MET),
                    verdict("NCT0001:inclusion:1", EligibilityLabel.MET),
                    verdict("NCT0001:exclusion:0", EligibilityLabel.NOT_MET),
                ]
            ),
        }
    )
    parsed_cache: dict = {}

    recommendation, run_log = run_pipeline(
        patient_id="S001",
        note=NOTE,
        trials=TRIALS,
        index=INDEX,
        llm=llm,
        settings=SETTINGS,
        top_k=3,
        parsed_cache=parsed_cache,
    )

    # 1. profile first, then one matcher call per surviving candidate.
    assert schemas_called(llm)[0] == "ProfileExtraction"
    assert llm.calls[0]["model"] == SETTINGS.models.extract_model
    assert set(schemas_called(llm)[1:]) == {"AssessmentBatch"}
    assert all(call["model"] == SETTINGS.models.reason_model for call in matcher_calls(llm))

    # 2. retrieval fused the note and both generated queries.
    assert set(run_log["candidate_ids"]) == set(TRIALS)

    # 3. the trial without any criterion never reaches the matcher.
    assert run_log["skipped_no_criteria"] == ["NCT0004"]
    assert len(matcher_calls(llm)) == 3
    ranked_ids = [row["nct_id"] for row in run_log["ranking"]]
    assert "NCT0004" not in ranked_ids

    # 4. deterministic aggregation put the fully met trial on top.
    top = recommendation.ranked[0]
    assert top.nct_id == "NCT0001"
    assert top.trial_label is TrialLabel.ELIGIBLE
    assert top.score == pytest.approx(1.0)
    assert run_log["ranking"][0] == {"nct_id": "NCT0001", "label": "eligible", "score": 1.0}

    # 5. the report is the ranked view plus the mandatory disclaimer.
    assert "NCT0001" in recommendation.report_markdown
    assert TRIALS["NCT0001"].title in recommendation.report_markdown
    assert DISCLAIMER in recommendation.report_markdown
    assert recommendation.patient_id == "S001"

    # 6. the criteria cache is filled for every candidate, so a second patient
    #    reuses it (design.md §2).
    assert set(parsed_cache) == set(TRIALS)
    assert parsed_cache["NCT0004"].all_criteria == []
    assert run_log["rounds"] == []  # no answer provider -> no question loop


def test_run_pipeline_truncates_to_top_k_and_reuses_the_cache() -> None:
    llm = FakeLLM({ProfileExtraction: PROFILE_EXTRACTION, AssessmentBatch: AssessmentBatch()})
    parsed_cache: dict = {}

    run_pipeline(
        patient_id="S001",
        note=NOTE,
        trials=TRIALS,
        index=INDEX,
        llm=llm,
        settings=SETTINGS,
        parsed_cache=parsed_cache,
    )
    cached = dict(parsed_cache)

    second = FakeLLM({ProfileExtraction: PROFILE_EXTRACTION, AssessmentBatch: AssessmentBatch()})
    recommendation, run_log = run_pipeline(
        patient_id="S002",
        note=NOTE,
        trials=TRIALS,
        index=INDEX,
        llm=second,
        settings=SETTINGS,
        top_k=2,
        parsed_cache=parsed_cache,
    )

    assert len(recommendation.ranked) == 2
    assert len(run_log["ranking"]) == 2
    # Parsing is deterministic and cached: the same objects come back.
    assert all(parsed_cache[nct] is cached[nct] for nct in cached)
    assert "CriteriaSplit" not in schemas_called(second)


def test_retrieval_drops_zero_score_trials_before_the_matcher() -> None:
    """A trial sharing no term with any query must not cost an LLM call."""
    llm = FakeLLM(
        {
            ProfileExtraction: ProfileExtraction(
                age_years=62.0,
                sex="male",
                conditions=["lung adenocarcinoma"],
                search_queries=["non-small cell lung adenocarcinoma"],
            ),
            AssessmentBatch: AssessmentBatch(),
        }
    )

    _, run_log = run_pipeline(
        patient_id="S001",
        note="Metastatic non-small cell lung adenocarcinoma.",
        trials=TRIALS,
        index=INDEX,
        llm=llm,
        settings=SETTINGS,
    )

    assert "NCT0003" not in run_log["candidate_ids"]  # the migraine trial
    assert all("NCT0003" not in call["user"] for call in matcher_calls(llm))


def test_per_candidate_llm_failure_is_recorded_and_the_run_continues() -> None:
    class FlakyLLM(FakeLLM):
        def structured(self, **kwargs: Any) -> Any:
            if kwargs["schema"] is AssessmentBatch and "NCT0002:" in kwargs["user"]:
                raise RuntimeError("overloaded_error")
            return super().structured(**kwargs)

    llm = FlakyLLM(
        {
            ProfileExtraction: PROFILE_EXTRACTION,
            AssessmentBatch: AssessmentBatch(
                verdicts=[
                    verdict("NCT0001:inclusion:0", EligibilityLabel.MET),
                    verdict("NCT0001:inclusion:1", EligibilityLabel.MET),
                    verdict("NCT0001:exclusion:0", EligibilityLabel.NOT_MET),
                ]
            ),
        }
    )

    recommendation, run_log = run_pipeline(
        patient_id="S001",
        note=NOTE,
        trials=TRIALS,
        index=INDEX,
        llm=llm,
        settings=SETTINGS,
    )

    assert run_log["errors"] == [{"nct_id": "NCT0002", "error": "overloaded_error"}]
    ranked_ids = [a.nct_id for a in recommendation.ranked]
    assert "NCT0002" not in ranked_ids  # dropped, not guessed
    assert ranked_ids[0] == "NCT0001"  # the healthy candidates still rank
    assert DISCLAIMER in recommendation.report_markdown


# --------------------------------------------------------------------------- #
# question loop
# --------------------------------------------------------------------------- #


def question_loop_scripts(second_round_label: EligibilityLabel) -> dict:
    """Round 1 leaves NCT0001:inclusion:1 undecided; round 2 resolves it.

    Round 2 answers the unknown criterion only — that is exactly what the
    pipeline asks for, and it proves the round-1 MET verdicts survive the merge.
    """
    round_one = AssessmentBatch(
        verdicts=[
            verdict("NCT0001:inclusion:0", EligibilityLabel.MET),
            verdict("NCT0001:inclusion:1", EligibilityLabel.UNKNOWN),
            verdict("NCT0001:exclusion:0", EligibilityLabel.NOT_MET),
            verdict("NCT0002:inclusion:0", EligibilityLabel.MET),
            verdict("NCT0002:exclusion:0", EligibilityLabel.NOT_MET),
        ]
    )
    round_two = AssessmentBatch(
        verdicts=[verdict("NCT0001:inclusion:1", second_round_label)]
    )
    return {
        ProfileExtraction: [PROFILE_EXTRACTION],
        # One matcher call per trial in round 1, then the re-assessment.
        AssessmentBatch: [round_one, round_one, round_two],
        QuestionBatch: [
            QuestionBatch(
                questions=[
                    GeneratedQuestion(
                        text="Has a doctor confirmed your lung cancer with a biopsy?",
                        criterion_ids=["NCT0001:inclusion:1"],
                    )
                ]
            )
        ],
    }


def test_question_loop_reassesses_only_trials_with_unknowns_and_improves_them() -> None:
    asked: list[list[ClarifyingQuestion]] = []

    def answer_provider(questions: list[ClarifyingQuestion]) -> list[QuestionAnswer]:
        asked.append(questions)
        return [
            QuestionAnswer(
                question_id=questions[0].question_id,
                answer_text="Yes, a biopsy confirmed non-small cell lung cancer.",
            )
        ]

    trials = {nct: TRIALS[nct] for nct in ("NCT0001", "NCT0002")}
    # The index covers the whole snapshot (BM25 idf turns negative on a
    # two-document corpus, and the pipeline now drops non-positive hits); the
    # smaller `trials` mapping is what limits the run to these two candidates.
    index = INDEX

    baseline_llm = ScriptedLLM(question_loop_scripts(EligibilityLabel.MET))
    baseline, _ = run_pipeline(
        patient_id="S001",
        note=NOTE,
        trials=trials,
        index=index,
        llm=baseline_llm,
        settings=SETTINGS,
        question_rounds=0,
    )
    before = {a.nct_id: a for a in baseline.ranked}["NCT0001"]
    assert before.trial_label is None
    assert before.score == pytest.approx(0.4)  # 1 of 2 inclusion met, 1 unknown

    llm = ScriptedLLM(question_loop_scripts(EligibilityLabel.MET))
    recommendation, run_log = run_pipeline(
        patient_id="S001",
        note=NOTE,
        trials=trials,
        index=index,
        llm=llm,
        settings=SETTINGS,
        question_rounds=1,
        answer_provider=answer_provider,
    )

    # Exactly one re-assessment, and only for the trial that had unknowns.
    calls = matcher_calls(llm)
    assert len(calls) == 3
    # Only the undecided criterion is re-sent; the settled ones are not re-judged.
    assert "NCT0001:inclusion:1" in calls[2]["user"]
    assert "NCT0001:inclusion:0" not in calls[2]["user"]
    assert "NCT0001:exclusion:0" not in calls[2]["user"]
    assert "NCT0002" not in calls[2]["user"]
    # The answer became a profile finding before the re-assessment.
    assert "Clarification — Q: Has a doctor confirmed" in calls[2]["user"]

    assert [q.text for q in asked[0]] == [
        "Has a doctor confirmed your lung cancer with a biopsy?"
    ]
    after = {a.nct_id: a for a in recommendation.ranked}["NCT0001"]
    assert after.trial_label is TrialLabel.ELIGIBLE
    assert after.score == pytest.approx(1.0)
    assert after.score > before.score
    assert recommendation.ranked[0].nct_id == "NCT0001"
    # The merge kept every round-1 verdict, only replacing the unknown one.
    labels = {v.criterion_id: v.label for v in after.verdicts}
    assert labels == {
        "NCT0001:inclusion:0": EligibilityLabel.MET,
        "NCT0001:inclusion:1": EligibilityLabel.MET,
        "NCT0001:exclusion:0": EligibilityLabel.NOT_MET,
    }

    round_log = run_log["rounds"][0]
    assert round_log["round"] == 1
    assert round_log["reassessed"] == ["NCT0001"]
    assert round_log["questions"][0]["criterion_ids"] == ["NCT0001:inclusion:1"]
    assert "biopsy confirmed" in round_log["answers"][0]["answer_text"]


def test_question_loop_stops_when_the_patient_does_not_know() -> None:
    def answer_provider(questions: list[ClarifyingQuestion]) -> list[QuestionAnswer]:
        return [
            QuestionAnswer(question_id=q.question_id, answer_text="The patient does not know.")
            for q in questions
        ]

    trials = {nct: TRIALS[nct] for nct in ("NCT0001", "NCT0002")}
    llm = ScriptedLLM(question_loop_scripts(EligibilityLabel.MET))

    recommendation, run_log = run_pipeline(
        patient_id="S001",
        note=NOTE,
        trials=trials,
        index=INDEX,
        llm=llm,
        settings=SETTINGS,
        question_rounds=1,
        answer_provider=answer_provider,
    )

    assert len(matcher_calls(llm)) == 2  # no re-assessment without new information
    assert run_log["rounds"][0]["reassessed"] == []
    assert {a.nct_id: a for a in recommendation.ranked}["NCT0001"].score == pytest.approx(0.4)


def test_question_loop_ignores_a_settled_trial_that_still_has_unknowns() -> None:
    """design.md §3: only undetermined trials earn a question.

    NCT0001 is excluded (an exclusion criterion is MET) yet keeps an unknown
    inclusion criterion; asking about it could never change the verdict, so no
    question is generated at all — the `FakeLLM` has no `QuestionBatch` canned
    response and would fail loudly if the generator were called.
    """

    def answer_provider(questions: list[ClarifyingQuestion]) -> list[QuestionAnswer]:
        raise AssertionError("no question should be asked")

    trials = {nct: TRIALS[nct] for nct in ("NCT0001", "NCT0002")}
    llm = FakeLLM(
        {
            ProfileExtraction: PROFILE_EXTRACTION,
            AssessmentBatch: AssessmentBatch(
                verdicts=[
                    verdict("NCT0001:inclusion:0", EligibilityLabel.MET),
                    verdict("NCT0001:inclusion:1", EligibilityLabel.UNKNOWN),
                    verdict("NCT0001:exclusion:0", EligibilityLabel.MET),
                    verdict("NCT0002:inclusion:0", EligibilityLabel.MET),
                    verdict("NCT0002:exclusion:0", EligibilityLabel.NOT_MET),
                ]
            ),
        }
    )

    recommendation, run_log = run_pipeline(
        patient_id="S001",
        note=NOTE,
        trials=trials,
        index=INDEX,
        llm=llm,
        settings=SETTINGS,
        question_rounds=1,
        answer_provider=answer_provider,
    )

    excluded = {a.nct_id: a for a in recommendation.ranked}["NCT0001"]
    assert excluded.trial_label is TrialLabel.EXCLUDED
    assert excluded.unknown_criterion_ids == ["NCT0001:inclusion:1"]
    assert run_log["rounds"] == []
    assert "QuestionBatch" not in schemas_called(llm)
    assert len(matcher_calls(llm)) == 2  # no re-assessment either
    # The eligible trial still outranks the excluded one (label grade first).
    assert [a.nct_id for a in recommendation.ranked] == ["NCT0002", "NCT0001"]


def test_simulator_answer_provider_answers_from_ground_truth() -> None:
    llm = FakeLLM(
        {
            AnswerBatch: AnswerBatch(
                answers=[SimulatedAnswer(question_id="q1", answer_text="My ECOG status is 1.")]
            )
        }
    )
    provider = simulator_answer_provider(llm, "The patient has an ECOG status of 1.", SETTINGS)

    answers = provider([ClarifyingQuestion(question_id="q1", text="How active are you?")])

    assert [a.answer_text for a in answers] == ["My ECOG status is 1."]
    assert llm.calls[0]["model"] == SETTINGS.models.extract_model


# --------------------------------------------------------------------------- #
# run log
# --------------------------------------------------------------------------- #


def test_save_run_log_writes_json_into_the_log_dir(tmp_path) -> None:
    settings = Settings(log_dir=tmp_path / "runs")
    llm = FakeLLM({ProfileExtraction: PROFILE_EXTRACTION, AssessmentBatch: AssessmentBatch()})

    _, run_log = run_pipeline(
        patient_id="S001",
        note=NOTE,
        trials=TRIALS,
        index=INDEX,
        llm=llm,
        settings=settings,
    )
    path = save_run_log(run_log, settings)

    assert path.parent == tmp_path / "runs"
    assert path.name.startswith("S001-") and path.suffix == ".json"
    written = json.loads(path.read_text(encoding="utf-8"))
    assert written["patient_id"] == "S001"
    assert written["profile"]["age_years"] == 62.0
    assert written["skipped_no_criteria"] == ["NCT0004"]
    assert {"nct_id", "label", "score"} == set(written["ranking"][0])


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #


def test_cli_topics_lists_ids_and_texts(tmp_path, capsys) -> None:
    path = tmp_path / "topics.json"
    path.write_text(
        json.dumps({"topics": [{"num": "S001", "title": "A 62-year-old man."}]}),
        encoding="utf-8",
    )

    assert cli.main(["topics", "--json", str(path)]) == 0

    assert "S001\tA 62-year-old man." in capsys.readouterr().out


def test_cli_run_without_questions_prints_the_report(tmp_path, capsys, monkeypatch) -> None:
    trials_path = tmp_path / "trials.jsonl"
    write_jsonl(TRIALS.values(), trials_path)
    index_path = tmp_path / "bm25.pkl"
    INDEX.save(index_path)
    note_path = tmp_path / "note.txt"
    note_path.write_text(NOTE, encoding="utf-8")

    llm = FakeLLM(
        {
            ProfileExtraction: PROFILE_EXTRACTION,
            AssessmentBatch: AssessmentBatch(
                verdicts=[
                    verdict("NCT0001:inclusion:0", EligibilityLabel.MET),
                    verdict("NCT0001:inclusion:1", EligibilityLabel.MET),
                    verdict("NCT0001:exclusion:0", EligibilityLabel.NOT_MET),
                ]
            ),
        }
    )
    monkeypatch.setattr(cli, "create_llm", lambda models: llm)

    exit_code = cli.main(
        [
            "run",
            "--note-file",
            str(note_path),
            "--patient-id",
            "S001",
            "--trials",
            str(trials_path),
            "--index",
            str(index_path),
            "--top-k",
            "2",
            "--no-questions",
        ]
    )

    assert exit_code == 0
    stdout = capsys.readouterr().out
    assert "NCT0001" in stdout
    assert DISCLAIMER in stdout
    assert "QuestionBatch" not in schemas_called(llm)  # --no-questions


def test_cli_run_rejects_an_empty_note(tmp_path, capsys) -> None:
    assert (
        cli.main(
            [
                "run",
                "--note",
                "   ",
                "--patient-id",
                "S001",
                "--trials",
                str(tmp_path / "missing.jsonl"),
                "--index",
                str(tmp_path / "missing.pkl"),
            ]
        )
        == 2
    )
    assert "empty" in capsys.readouterr().out


def test_cli_run_reports_a_missing_note_file_without_a_traceback(tmp_path, capsys) -> None:
    exit_code = cli.main(
        [
            "run",
            "--note-file",
            str(tmp_path / "missing.txt"),
            "--patient-id",
            "S001",
            "--trials",
            str(tmp_path / "missing.jsonl"),
            "--index",
            str(tmp_path / "missing.pkl"),
        ]
    )

    assert exit_code == 2
    out = capsys.readouterr().out
    assert out.startswith("error: ")
    assert "missing.txt" in out


def test_cli_index_builds_and_saves_an_index(tmp_path, capsys) -> None:
    trials_path = tmp_path / "trials.jsonl"
    write_jsonl(TRIALS.values(), trials_path)
    out_path = tmp_path / "nested" / "bm25.pkl"

    assert cli.main(["index", "--trials", str(trials_path), "--out", str(out_path)]) == 0

    assert "indexed 4 trials" in capsys.readouterr().out
    assert len(BM25Index.load(out_path)) == 4
