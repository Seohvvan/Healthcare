"""Retrieval tests: BM25 ranking, persistence, and RRF fusion.

No network access and no optional heavy dependencies are required.
"""

from __future__ import annotations

import importlib.util

import pytest

from trialmatch.retrieval import BM25Index, DenseIndex, rrf, tokenize
from trialmatch.schemas import TrialRecord

TRIALS = [
    TrialRecord(
        nct_id="NCT00000001",
        title="Erenumab for Prevention of Episodic Migraine With Aura",
        conditions=["Migraine", "Migraine With Aura"],
        brief_summary="A study of a CGRP antibody in adults with frequent migraine headaches.",
        eligibility_text="Inclusion: 4 to 14 migraine days per month. Exclusion: chronic migraine.",
    ),
    TrialRecord(
        nct_id="NCT00000002",
        title="BCG Immunotherapy in Non-Muscle-Invasive Bladder Cancer",
        conditions=["Bladder Cancer", "Urothelial Carcinoma"],
        brief_summary=(
            "Patients presenting with painless hematuria and a bladder tumor on cystoscopy."
        ),
        eligibility_text="Inclusion: histologically confirmed urothelial carcinoma of the bladder.",
    ),
    TrialRecord(
        nct_id="NCT00000003",
        title="Levothyroxine Dose Adjustment in Subclinical Hypothyroidism",
        conditions=["Hypothyroidism", "Thyroid Disease"],
        brief_summary="Evaluates thyroid hormone replacement guided by TSH levels.",
        eligibility_text="Inclusion: elevated TSH with normal free T4.",
    ),
    TrialRecord(
        nct_id="NCT00000004",
        title="Early Fluid Resuscitation in Acute Pancreatitis",
        conditions=["Acute Pancreatitis"],
        brief_summary="Compares aggressive versus moderate fluid therapy after epigastric pain.",
        eligibility_text="Inclusion: lipase greater than three times the upper limit of normal.",
    ),
    TrialRecord(
        nct_id="NCT00000005",
        title="Nintedanib in Idiopathic Pulmonary Fibrosis",
        conditions=["Idiopathic Pulmonary Fibrosis", "Interstitial Lung Disease"],
        brief_summary="Antifibrotic therapy for progressive dyspnea and honeycombing on HRCT.",
        eligibility_text="Inclusion: FVC at least 50 percent predicted.",
    ),
]


@pytest.fixture(scope="module")
def index() -> BM25Index:
    return BM25Index.build(TRIALS)


def test_tokenize_drops_stopwords_and_short_tokens():
    assert tokenize("The patient's TSH is 5.2 mIU/L!") == ["patient", "tsh", "is", "miu"]
    assert tokenize("") == []


def test_index_size_and_order():
    built = BM25Index.build(TRIALS)
    assert len(built) == 5
    assert built.nct_ids == [t.nct_id for t in TRIALS]


def test_build_rejects_empty_corpus():
    with pytest.raises(ValueError):
        BM25Index.build([])


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("migraine with aura", "NCT00000001"),
        ("painless hematuria bladder tumor", "NCT00000002"),
        ("elevated TSH subclinical hypothyroidism", "NCT00000003"),
        ("acute pancreatitis elevated lipase", "NCT00000004"),
        ("idiopathic pulmonary fibrosis dyspnea", "NCT00000005"),
    ],
)
def test_search_ranks_matching_trial_first(index: BM25Index, query: str, expected: str):
    results = index.search(query, k=3)
    assert results[0][0] == expected
    assert results[0][1] > 0.0


def test_search_respects_k_and_is_sorted(index: BM25Index):
    results = index.search("migraine bladder thyroid", k=2)
    assert len(results) == 2
    assert results[0][1] >= results[1][1]
    assert index.search("migraine", k=0) == []


def test_search_with_only_stopwords_returns_nothing(index: BM25Index):
    assert index.search("the and for", k=5) == []


def test_search_is_deterministic_on_ties(index: BM25Index):
    # A query matching no document leaves every score at 0.0 -> ordered by id.
    results = index.search("zzzqqq nonexistentterm", k=5)
    assert [nct_id for nct_id, _ in results] == sorted(t.nct_id for t in TRIALS)


def test_save_load_round_trip(tmp_path, index: BM25Index):
    path = tmp_path / "nested" / "bm25.pkl"
    index.save(path)
    assert path.exists()

    loaded = BM25Index.load(path)
    assert loaded.nct_ids == index.nct_ids
    for query in ("migraine with aura", "painless hematuria bladder tumor"):
        assert loaded.search(query, k=5) == index.search(query, k=5)


def test_rrf_hand_computed_scores():
    fused = rrf([["a", "b"], ["a", "c"]], k=60)
    scores = dict(fused)
    assert scores["a"] == pytest.approx(1 / 61 + 1 / 61)
    assert scores["b"] == pytest.approx(1 / 62)
    assert scores["c"] == pytest.approx(1 / 62)
    # b and c tie; the id tie-break puts "b" first.
    assert [doc_id for doc_id, _ in fused] == ["a", "b", "c"]


def test_rrf_promotes_consistently_ranked_id():
    # "y" is second in both rankings and beats ids that are first in only one.
    fused = rrf([["x", "y"], ["z", "y"]], k=1)
    assert fused[0][0] == "y"
    assert fused[0][1] == pytest.approx(2 * (1 / 3))


def test_rrf_top_n():
    rankings = [["a", "b", "c", "d"], ["b", "a", "d", "c"]]
    assert [doc_id for doc_id, _ in rrf(rankings, top_n=2)] == ["a", "b"]
    assert rrf(rankings, top_n=0) == []
    assert len(rrf(rankings, top_n=None)) == 4
    assert len(rrf(rankings, top_n=99)) == 4


def test_rrf_determinism_on_full_ties():
    # Mirrored rankings give every id an identical score -> pure id ordering.
    fused = rrf([["b", "a"], ["a", "b"]], k=60)
    assert [doc_id for doc_id, _ in fused] == ["a", "b"]
    assert fused[0][1] == pytest.approx(fused[1][1])


def test_rrf_empty_inputs():
    assert rrf([]) == []
    assert rrf([[], []]) == []


def test_rrf_rejects_non_positive_k():
    with pytest.raises(ValueError):
        rrf([["a"]], k=0)


def test_dense_index_requires_optional_extra():
    if importlib.util.find_spec("sentence_transformers") is not None:
        pytest.skip("sentence-transformers is installed; the failure path cannot be exercised")
    with pytest.raises(RuntimeError, match="uv sync --extra dense"):
        DenseIndex.build(TRIALS)
