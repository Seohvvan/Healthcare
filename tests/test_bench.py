"""Judged-subset benchmark tests. No network, no API key.

HTTP goes through `httpx.MockTransport`; every LLM call goes through a fake
client built on `FakeLLM` from `tests/conftest.py`.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from conftest import FakeLLM  # shared test double, see tests/conftest.py

from trialmatch.agents.matcher import AssessmentBatch, CriterionAssessment
from trialmatch.agents.profiler import ProfileExtraction
from trialmatch.config import Settings
from trialmatch.data.ctgov import CTGovClient
from trialmatch.data.store import load_trials, write_jsonl
from trialmatch.eval.bench import (
    bm25_judged_rerank,
    fetch_judged_trials,
    llm_judged_rerank,
    load_trial_corpus,
    load_trialgpt_corpus,
    main,
)
from trialmatch.schemas import EligibilityLabel, TrialRecord

SETTINGS = Settings()


# --------------------------------------------------------------------------- #
# load_trialgpt_corpus
# --------------------------------------------------------------------------- #


def _write_json(tmp_path, name: str, payload: Any):
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_load_trialgpt_corpus_dict_keyed_with_split_criteria(tmp_path):
    path = _write_json(
        tmp_path,
        "trial_info.json",
        {
            "NCT00000001": {
                "brief_title": "Sumatriptan for acute migraine",
                "diseases_list": ["Migraine", "Headache"],
                "brief_summary": "A summary.",
                "inclusion_criteria": ["Age 18 years or older", "Diagnosis of migraine"],
                "exclusion_criteria": "Pregnancy",
                "drop_rate": 0.1,  # unknown field: ignored
            }
        },
    )

    corpus = load_trialgpt_corpus(path)

    assert list(corpus) == ["NCT00000001"]
    record = corpus["NCT00000001"]
    assert record.title == "Sumatriptan for acute migraine"
    assert record.conditions == ["Migraine", "Headache"]
    assert record.brief_summary == "A summary."
    assert record.eligibility_text == (
        "Inclusion Criteria:\nAge 18 years or older\nDiagnosis of migraine\n\n"
        "Exclusion Criteria:\nPregnancy"
    )


def test_load_trialgpt_corpus_list_shaped_with_field_variants(tmp_path):
    path = _write_json(
        tmp_path,
        "trials.json",
        [
            {
                "nct_id": "NCT00000002",
                "title": "Insulin study",
                "conditions": "Type 2 Diabetes",
                "summary": "Another summary.",
                "criteria": "Inclusion Criteria:\n* Age 18+\n\nExclusion Criteria:\n* Pregnancy",
            },
            {
                "nctId": "NCT00000003",
                "official_title": "Official only",
                "eligibility_criteria": "Inclusion Criteria:\n* Adults",
            },
        ],
    )

    corpus = load_trialgpt_corpus(path)

    assert sorted(corpus) == ["NCT00000002", "NCT00000003"]
    assert corpus["NCT00000002"].conditions == ["Type 2 Diabetes"]
    assert corpus["NCT00000002"].brief_summary == "Another summary."
    assert corpus["NCT00000002"].eligibility_text.startswith("Inclusion Criteria:")
    assert corpus["NCT00000003"].title == "Official only"
    assert corpus["NCT00000003"].eligibility_text == "Inclusion Criteria:\n* Adults"


def test_load_trialgpt_corpus_skips_records_without_an_id(tmp_path):
    path = _write_json(
        tmp_path,
        "trials.json",
        [
            {"nct_id": "NCT00000004", "title": "Kept"},
            {"title": "No id at all"},
            "not even an object",
        ],
    )

    corpus = load_trialgpt_corpus(path)

    assert list(corpus) == ["NCT00000004"]


def test_load_trialgpt_corpus_rejects_a_non_collection_payload(tmp_path):
    path = _write_json(tmp_path, "trials.json", "nope")
    with pytest.raises(TypeError, match="JSON object or list"):
        load_trialgpt_corpus(path)


def test_load_trial_corpus_dispatches_on_the_extension(tmp_path):
    jsonl = tmp_path / "trials.jsonl"
    write_jsonl([TrialRecord(nct_id="NCT1", title="One")], jsonl)
    assert load_trial_corpus(jsonl)["NCT1"].title == "One"

    js = _write_json(tmp_path, "trials.json", {"NCT2": {"brief_title": "Two"}})
    assert load_trial_corpus(js)["NCT2"].title == "Two"


def _write_jsonl_lines(tmp_path, name: str, records: list[Any]):
    path = tmp_path / name
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8"
    )
    return path


def test_load_trial_corpus_reads_the_trialgpt_jsonl_distribution(tmp_path):
    path = _write_jsonl_lines(
        tmp_path,
        "corpus.jsonl",
        [
            {
                "_id": "NCT00000010",
                "title": "Envelope title",
                "text": "unused blob",
                "metadata": {
                    "brief_title": "Metadata title",
                    "diseases_list": ["Migraine"],
                    "brief_summary": "A summary.",
                    "inclusion_criteria": "inclusion criteria: adults",
                    "exclusion_criteria": "pregnancy",
                    "criteria": "None",  # the distribution stringifies nulls
                },
            }
        ],
    )

    record = load_trial_corpus(path)["NCT00000010"]

    assert record.title == "Metadata title"
    assert record.conditions == ["Migraine"]
    assert record.brief_summary == "A summary."
    # The literal "None" in `criteria` must not win over the split fields.
    assert record.eligibility_text.startswith("Inclusion Criteria:")
    assert "pregnancy" in record.eligibility_text


def test_flatten_metadata_keeps_the_envelope_when_metadata_is_blank(tmp_path):
    # "" is the distribution's missing-value sentinel; it must not shadow a
    # real envelope field of the same name.
    path = _write_jsonl_lines(
        tmp_path,
        "corpus.jsonl",
        [{"_id": "NCT00000011", "title": "Envelope", "metadata": {"title": "", "phase": ""}}],
    )

    assert load_trial_corpus(path)["NCT00000011"].title == "Envelope"


def test_load_trial_corpus_sniffs_metadata_before_a_top_level_id(tmp_path):
    # A distribution-style record that also carries a top-level nct_id must not
    # be routed to the strict snapshot loader, which would load empty records.
    path = _write_jsonl_lines(
        tmp_path,
        "corpus.jsonl",
        [{"nct_id": "NCT00000012", "metadata": {"brief_title": "From metadata"}}],
    )

    assert load_trial_corpus(path)["NCT00000012"].title == "From metadata"


def test_load_trial_corpus_names_the_location_of_a_malformed_jsonl_line(tmp_path):
    path = tmp_path / "corpus.jsonl"
    path.write_text('{"_id": "NCT1", "metadata": {}}\n{broken\n', encoding="utf-8")

    with pytest.raises(ValueError, match=r"corpus\.jsonl:2"):
        load_trial_corpus(path)


def test_load_trial_corpus_tolerates_a_non_mapping_first_record(tmp_path):
    path = tmp_path / "corpus.jsonl"
    path.write_text("5\n", encoding="utf-8")

    assert load_trial_corpus(path) == {}


# --------------------------------------------------------------------------- #
# fetch_judged_trials
# --------------------------------------------------------------------------- #


def _single_study(nct_id: str) -> dict:
    return {
        "protocolSection": {
            "identificationModule": {"nctId": nct_id, "briefTitle": f"Study {nct_id}"},
            "eligibilityModule": {"eligibilityCriteria": "Inclusion Criteria:\n* Age 18+"},
        }
    }


def test_fetch_judged_trials_writes_jsonl_and_skips_missing_ids(tmp_path):
    qrels = {
        "1": {"NCT00000001": 2, "NCT00000002": 1},
        "2": {"NCT00000002": 0, "NCT00000003": 1},
    }
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        nct_id = request.url.path.rsplit("/", 1)[-1]
        if nct_id == "NCT00000003":
            return httpx.Response(404, json={"message": "not found"})
        return httpx.Response(200, json=_single_study(nct_id))

    out_path = tmp_path / "nested" / "judged.jsonl"
    with CTGovClient(transport=httpx.MockTransport(handler)) as client:
        written = fetch_judged_trials(qrels, out_path, client=client)

    assert written == 2
    assert sorted(load_trials(out_path)) == ["NCT00000001", "NCT00000002"]
    # The union of judged ids, one single-study GET each.
    assert [r.url.path.rsplit("/", 1)[-1] for r in requests] == [
        "NCT00000001",
        "NCT00000002",
        "NCT00000003",
    ]
    assert requests[0].url.path.endswith("/studies/NCT00000001")


def test_fetch_judged_trials_retries_a_server_error_on_the_single_study_path(
    tmp_path, monkeypatch
):
    """The per-id fetch goes through `CTGovClient._get`, so it inherits the retry."""
    monkeypatch.setattr("trialmatch.data.ctgov.time.sleep", lambda _seconds: None)
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(503, json={"error": "unavailable"})
        return httpx.Response(200, json=_single_study("NCT00000001"))

    out_path = tmp_path / "judged.jsonl"
    with CTGovClient(transport=httpx.MockTransport(handler)) as client:
        written = fetch_judged_trials({"1": {"NCT00000001": 2}}, out_path, client=client)

    assert calls["n"] == 2  # one failure, one retry
    assert written == 1
    assert list(load_trials(out_path)) == ["NCT00000001"]


def test_fetch_judged_trials_creates_the_file_when_everything_is_missing(tmp_path):
    transport = httpx.MockTransport(lambda request: httpx.Response(404, json={}))
    out_path = tmp_path / "judged.jsonl"
    with CTGovClient(transport=transport) as client:
        written = fetch_judged_trials({"1": {"NCT00000009": 2}}, out_path, client=client)

    assert written == 0
    assert out_path.read_text(encoding="utf-8") == ""


# --------------------------------------------------------------------------- #
# bm25_judged_rerank
# --------------------------------------------------------------------------- #


def _trial(nct_id: str, title: str, conditions: list[str], criteria: str = "") -> TrialRecord:
    return TrialRecord(
        nct_id=nct_id, title=title, conditions=conditions, eligibility_text=criteria
    )


BM25_TRIALS = {
    "NCT01": _trial(
        "NCT01",
        "Sumatriptan for acute migraine with photophobia",
        ["Migraine"],
        "Inclusion Criteria:\n* Migraine with photophobia and nausea",
    ),
    "NCT02": _trial(
        "NCT02",
        "Insulin titration in type 2 diabetes",
        ["Type 2 Diabetes"],
        "Inclusion Criteria:\n* Type 2 diabetes on metformin",
    ),
    "NCT03": _trial(
        "NCT03",
        "Headache prevention programme in chronic migraine",
        ["Headache"],
        "Inclusion Criteria:\n* Recurrent headache",
    ),
}

BM25_TOPICS = [
    ("1", "A 45-year-old woman with migraine, photophobia and nausea."),
    ("2", "A 60-year-old man with type 2 diabetes on metformin."),
]

BM25_QRELS = {
    # NCT04 is judged but absent from the corpus.
    "1": {"NCT01": 2, "NCT02": 0, "NCT03": 1, "NCT04": 1},
    "2": {"NCT01": 0, "NCT02": 2, "NCT03": 0},
}


def test_bm25_judged_rerank_ranks_the_lexical_match_first():
    result = bm25_judged_rerank(BM25_TOPICS, BM25_QRELS, BM25_TRIALS)

    assert result["num_topics"] == 2
    per_topic = result["per_topic"]
    # Ranked [NCT01 (gain 2), NCT03 (gain 1), NCT02 (gain 0)]; the ideal ranking also
    # holds NCT04, which is judged relevant but missing from the corpus, so 1.0 is
    # out of reach: DCG 2.6309 / IDCG 3.1309.
    assert per_topic["1"]["ndcg@5"] == pytest.approx(0.8403, abs=1e-4)
    assert per_topic["1"]["mrr"] == pytest.approx(1.0)  # the best match is rank 1
    assert per_topic["2"]["ndcg@5"] == pytest.approx(1.0)


def test_bm25_judged_rerank_reports_coverage_and_drops_absent_ids():
    result = bm25_judged_rerank(BM25_TOPICS, BM25_QRELS, BM25_TRIALS)

    assert result["coverage"] == {
        "1": {"judged": 4, "in_corpus": 3},
        "2": {"judged": 3, "in_corpus": 3},
    }
    assert result["warnings"] == [
        "topic '1': 1 of 4 judged trials are missing from the corpus"
    ]


def test_bm25_judged_rerank_handles_a_topic_with_no_candidates():
    result = bm25_judged_rerank(
        [("9", "A patient with an unindexed disease.")],
        {"9": {"NCT99": 2}},
        BM25_TRIALS,
    )

    assert result["coverage"]["9"] == {"judged": 1, "in_corpus": 0}
    assert result["per_topic"]["9"]["ndcg@5"] == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# llm_judged_rerank
# --------------------------------------------------------------------------- #

LLM_CRITERIA = """Inclusion Criteria:

* Adults aged 18 years or older
* Diagnosis of migraine

Exclusion Criteria:

* Pregnancy
"""

LLM_TRIALS = {
    nct_id: _trial(nct_id, f"Migraine study {nct_id}", ["Migraine"], LLM_CRITERIA)
    for nct_id in ("NCT01", "NCT02", "NCT03", "NCT04")
}

LLM_TOPICS = [("1", "A 45-year-old woman with migraine and photophobia.")]

# NCT05 is judged but absent from the corpus.
LLM_QRELS = {"1": {"NCT01": 2, "NCT02": 1, "NCT03": 0, "NCT04": 2, "NCT05": 1}}


def _batch(nct_id: str, labels: list[EligibilityLabel]) -> AssessmentBatch:
    """Verdicts for the three criteria of `LLM_CRITERIA`, in order.

    Decided verdicts quote evidence: the ranker reads an evidence-less met or
    cleared verdict as undecided.
    """
    criterion_ids = [
        f"{nct_id}:inclusion:0",
        f"{nct_id}:inclusion:1",
        f"{nct_id}:exclusion:0",
    ]
    return AssessmentBatch(
        verdicts=[
            CriterionAssessment(
                criterion_id=cid,
                label=label,
                evidence=[] if label is UNKNOWN else ["quoted from the topic"],
            )
            for cid, label in zip(criterion_ids, labels, strict=True)
        ]
    )


MET = EligibilityLabel.MET
NOT_MET = EligibilityLabel.NOT_MET
UNKNOWN = EligibilityLabel.UNKNOWN

LLM_ASSESSMENTS = {
    "NCT01": _batch("NCT01", [MET, MET, NOT_MET]),  # eligible   (qrel 2) -> correct
    "NCT02": _batch("NCT02", [MET, MET, MET]),  # excluded       (qrel 1) -> correct
    "NCT03": _batch("NCT03", [NOT_MET, MET, NOT_MET]),  # not relevant (qrel 0) -> correct
    "NCT04": _batch("NCT04", [MET, UNKNOWN, NOT_MET]),  # undetermined (qrel 2) -> wrong
}


class TrialScriptedLLM(FakeLLM):
    """`FakeLLM` plus matcher answers keyed by the nct id present in the prompt."""

    def __init__(self, by_trial: dict[str, AssessmentBatch]) -> None:
        super().__init__({ProfileExtraction: ProfileExtraction(conditions=["migraine"])})
        self.by_trial = by_trial

    def structured(self, *, schema, user: str, **kwargs: Any):
        if schema is AssessmentBatch:
            self.calls.append({"schema": schema, "user": user, **kwargs})
            for nct_id, batch in self.by_trial.items():
                if nct_id in user:
                    return batch
            raise AssertionError(f"no canned AssessmentBatch for prompt: {user[:120]!r}")
        return super().structured(schema=schema, user=user, **kwargs)


def test_llm_judged_rerank_scores_ranking_and_three_class_labels():
    llm = TrialScriptedLLM(LLM_ASSESSMENTS)

    result = llm_judged_rerank(
        llm,
        LLM_TOPICS,
        LLM_QRELS,
        LLM_TRIALS,
        SETTINGS,
        max_topics=5,
        candidates_per_topic=4,
        use_llm_criteria=False,
    )

    # One profiler call for the topic, one matcher call per candidate.
    assert sum(1 for call in llm.calls if call["schema"] is ProfileExtraction) == 1
    assert sum(1 for call in llm.calls if call["schema"] is AssessmentBatch) == 4

    classification = result["classification"]
    assert classification["num_assessed"] == 4
    assert classification["accuracy"] == pytest.approx(0.75)  # NCT04 undetermined -> wrong
    assert classification["macro_f1"] == pytest.approx(7 / 9)
    assert classification["undetermined"] == 1

    ranking = result["ranking"]
    assert ranking["per_topic"]["1"]["ndcg@10"] > 0.8
    assert result["coverage"] == {"1": {"judged": 5, "in_corpus": 4, "assessed": 4}}
    assert result["config"] == {
        "max_topics": 5,
        "candidates_per_topic": 4,
        "use_llm_criteria": False,
    }
    # Serializable end to end (the CLI writes this straight to JSON).
    json.dumps(result)


def test_llm_judged_rerank_orders_by_the_label_grade_then_the_score():
    result = llm_judged_rerank(
        TrialScriptedLLM(LLM_ASSESSMENTS),
        LLM_TOPICS,
        LLM_QRELS,
        LLM_TRIALS,
        SETTINGS,
        candidates_per_topic=4,
        use_llm_criteria=False,
    )
    per_topic = result["ranking"]["per_topic"]["1"]
    assert per_topic["mrr"] == pytest.approx(1.0)
    assert per_topic["p@10"] == pytest.approx(0.3)
    # The bench ranks through the product's `rank`: eligible NCT01 (qrel 2),
    # undetermined NCT04 (qrel 2), excluded NCT02 (qrel 1), not relevant NCT03
    # (qrel 0). Score alone would have put NCT03 (-0.5) above NCT02 (-1.0) and
    # cost ndcg; the label grade matches the TREC gain order instead.
    assert per_topic["ndcg@10"] == pytest.approx(0.8973, abs=1e-4)
    assert result["errors"] == []


def test_llm_judged_rerank_records_a_failing_candidate_and_keeps_going():
    class ExplodingLLM(TrialScriptedLLM):
        def structured(self, *, schema, user: str, **kwargs: Any):
            if schema is AssessmentBatch and "NCT02" in user:
                raise RuntimeError("overloaded_error")
            return super().structured(schema=schema, user=user, **kwargs)

    result = llm_judged_rerank(
        ExplodingLLM(LLM_ASSESSMENTS),
        LLM_TOPICS,
        LLM_QRELS,
        LLM_TRIALS,
        SETTINGS,
        candidates_per_topic=4,
        use_llm_criteria=False,
    )

    assert result["errors"] == [
        {"topic_id": "1", "nct_id": "NCT02", "error": "overloaded_error"}
    ]
    # The other three candidates were still assessed, scored and ranked.
    assert result["classification"]["num_assessed"] == 3
    assert result["ranking"]["per_topic"]["1"]["mrr"] == pytest.approx(1.0)


def test_llm_judged_rerank_honours_the_requested_ndcg_cutoffs():
    result = llm_judged_rerank(
        TrialScriptedLLM(LLM_ASSESSMENTS),
        LLM_TOPICS,
        LLM_QRELS,
        LLM_TRIALS,
        SETTINGS,
        candidates_per_topic=4,
        use_llm_criteria=False,
        ks=[3],
    )

    assert result["ranking"]["metric_keys"][0] == "ndcg@3"
    assert "ndcg@10" not in result["ranking"]["per_topic"]["1"]


def test_llm_judged_rerank_caps_topics_and_candidates():
    topics = [*LLM_TOPICS, ("2", "A second topic that must not be evaluated.")]
    llm = TrialScriptedLLM(LLM_ASSESSMENTS)

    result = llm_judged_rerank(
        llm,
        topics,
        {**LLM_QRELS, "2": {"NCT01": 2}},
        LLM_TRIALS,
        SETTINGS,
        max_topics=1,
        candidates_per_topic=2,
        use_llm_criteria=False,
    )

    assert list(result["coverage"]) == ["1"]  # topic 2 was never touched
    assert result["coverage"]["1"]["assessed"] == 2
    assert result["classification"]["num_assessed"] == 2
    assert sum(1 for call in llm.calls if call["schema"] is AssessmentBatch) == 2
    # Candidates beyond the budget still appear, after the assessed ones.
    assert sorted(result["ranking"]["per_topic"]) == ["1"]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _fixture_files(tmp_path):
    topics = tmp_path / "topics.json"
    topics.write_text(
        json.dumps({"topics": [{"num": tid, "title": text} for tid, text in BM25_TOPICS]}),
        encoding="utf-8",
    )
    qrels = tmp_path / "qrels.txt"
    qrels.write_text(
        "".join(
            f"{tid} 0 {doc} {gain}\n"
            for tid, gains in BM25_QRELS.items()
            for doc, gain in gains.items()
        ),
        encoding="utf-8",
    )
    trials = tmp_path / "trials.jsonl"
    write_jsonl(list(BM25_TRIALS.values()), trials)
    return topics, qrels, trials


def test_main_bm25_prints_a_table_and_writes_the_result(tmp_path, capsys):
    topics, qrels, trials = _fixture_files(tmp_path)
    out = tmp_path / "out" / "bm25.json"

    exit_code = main(
        [
            "bm25",
            "--topics-json",
            str(topics),
            "--qrels",
            str(qrels),
            "--trials",
            str(trials),
            "--ks",
            "5",
            "--out",
            str(out),
        ]
    )

    assert exit_code == 0
    printed = capsys.readouterr().out
    assert "ndcg@5" in printed
    assert "judged trials:" in printed
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["coverage"]["1"] == {"judged": 4, "in_corpus": 3}
    assert payload["metric_keys"][0] == "ndcg@5"


def test_main_requires_exactly_one_topic_source(tmp_path):
    _, qrels, trials = _fixture_files(tmp_path)
    with pytest.raises(SystemExit):
        main(["bm25", "--qrels", str(qrels), "--trials", str(trials)])
