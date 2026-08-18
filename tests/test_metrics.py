"""Unit tests for the evaluation metrics and the offline TREC runner."""

import json
import math

import pytest

from trialmatch.eval.metrics import (
    cohens_kappa,
    macro_f1,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    three_class_accuracy,
)
from trialmatch.eval.run_trec import (
    evaluate_rankings,
    format_table,
    load_qrels_file,
    load_rankings_file,
    main,
)

TOL = 1e-9


# --------------------------------------------------------------------------
# nDCG
# --------------------------------------------------------------------------


def test_ndcg_hand_computed():
    # gains a=2, b=1, c=0 ranked as [b, a, c]:
    #   DCG  = 1/log2(2) + 2/log2(3) + 0/log2(4)
    #   IDCG = 2/log2(2) + 1/log2(3)
    gains = {"a": 2, "b": 1, "c": 0}
    expected = (1 / math.log2(2) + 2 / math.log2(3)) / (2 / math.log2(2) + 1 / math.log2(3))
    assert ndcg_at_k(["b", "a", "c"], gains, 3) == pytest.approx(expected, abs=TOL)
    assert round(ndcg_at_k(["b", "a", "c"], gains, 3), 6) == 0.859719


def test_ndcg_perfect_and_empty_cases():
    gains = {"a": 2, "b": 1, "c": 0}
    assert ndcg_at_k(["a", "b", "c"], gains, 3) == pytest.approx(1.0, abs=TOL)
    # Ideal DCG uses all judged docs, so missing "b" cannot reach 1.0.
    assert ndcg_at_k(["a"], gains, 3) < 1.0
    # No judged doc carries a positive gain -> 0.0 rather than a division error.
    assert ndcg_at_k(["a", "b"], {"a": 0, "b": 0}, 2) == 0.0
    assert ndcg_at_k([], gains, 5) == 0.0


def test_ndcg_unjudged_docs_score_zero():
    gains = {"a": 2}
    assert ndcg_at_k(["zzz", "a"], gains, 2) == pytest.approx(1 / math.log2(3), abs=TOL)


def test_ranking_metrics_reject_non_positive_k():
    for func in (ndcg_at_k, precision_at_k, recall_at_k):
        with pytest.raises(ValueError):
            func(["a"], {"a": 2}, 0)
        with pytest.raises(ValueError):
            func(["a"], {"a": 2}, -1)


# --------------------------------------------------------------------------
# precision / recall / MRR
# --------------------------------------------------------------------------


def test_precision_at_k_denominator_is_k():
    gains = {"a": 2, "b": 1, "c": 0}
    assert precision_at_k(["a", "b", "c"], gains, 2) == pytest.approx(1.0)
    assert precision_at_k(["a", "b", "c"], gains, 3) == pytest.approx(2 / 3)
    # k larger than the ranking: the missing ranks count as misses.
    assert precision_at_k(["a"], gains, 10) == pytest.approx(0.1)
    # No relevant document retrieved.
    assert precision_at_k(["c", "zzz"], gains, 2) == 0.0


def test_precision_min_gain_selects_eligible_only():
    gains = {"a": 2, "b": 1}
    assert precision_at_k(["b", "a"], gains, 2, min_gain=2) == pytest.approx(0.5)


def test_recall_at_k():
    gains = {"a": 2, "b": 1, "c": 0}
    assert recall_at_k(["a", "c"], gains, 2) == pytest.approx(0.5)
    assert recall_at_k(["a", "b"], gains, 10) == pytest.approx(1.0)
    # Truncation at k matters.
    assert recall_at_k(["c", "a", "b"], gains, 1) == 0.0
    # No relevant judged document at all -> 0.0.
    assert recall_at_k(["a"], {"a": 0, "c": 0}, 5) == 0.0
    assert recall_at_k(["a"], {}, 5) == 0.0


def test_mrr():
    gains = {"a": 2, "b": 1, "c": 0}
    assert mrr(["a", "b"], gains) == pytest.approx(1.0)
    assert mrr(["c", "b", "a"], gains) == pytest.approx(0.5)
    assert mrr(["zzz", "c", "a"], gains) == pytest.approx(1 / 3)
    # No relevant document in the ranking, and the empty ranking.
    assert mrr(["c", "zzz"], gains) == 0.0
    assert mrr([], gains) == 0.0
    # min_gain restricts to eligible-only relevance.
    assert mrr(["b", "a"], gains, min_gain=2) == pytest.approx(0.5)


# --------------------------------------------------------------------------
# classification metrics
# --------------------------------------------------------------------------


def test_three_class_accuracy():
    y_true = ["met", "not_met", "unknown", "met"]
    y_pred = ["met", "unknown", "unknown", "met"]
    assert three_class_accuracy(y_true, y_pred) == pytest.approx(0.75)


def test_classification_metrics_validate_input():
    labels = ["met", "not_met", "unknown"]
    for call in (
        lambda: three_class_accuracy(["met"], ["met", "met"]),
        lambda: macro_f1(["met"], ["met", "met"], labels),
        lambda: cohens_kappa(["met"], ["met", "met"]),
        lambda: three_class_accuracy([], []),
        lambda: macro_f1([], [], labels),
        lambda: cohens_kappa([], []),
        lambda: macro_f1(["met"], ["met"], []),
    ):
        with pytest.raises(ValueError):
            call()


def test_macro_f1_hand_computed():
    labels = ["met", "not_met", "unknown"]
    y_true = ["met", "met", "not_met", "unknown"]
    y_pred = ["met", "not_met", "not_met", "met"]
    # met:     tp=1 fp=1 fn=1 -> F1 = 2/(2+1+1)   = 0.5
    # not_met: tp=1 fp=1 fn=0 -> F1 = 2/(2+1+0)   = 2/3
    # unknown: tp=0 fp=0 fn=1 -> F1 = 0/(0+0+1)   = 0.0
    expected = (0.5 + 2 / 3 + 0.0) / 3
    assert macro_f1(y_true, y_pred, labels) == pytest.approx(expected, abs=TOL)


def test_macro_f1_perfect_and_worst_case():
    labels = ["met", "not_met"]
    assert macro_f1(["met", "not_met"], ["met", "not_met"], labels) == pytest.approx(1.0)
    assert macro_f1(["met", "not_met"], ["not_met", "met"], labels) == 0.0
    # A label absent from both sequences contributes 0.0 instead of raising.
    assert macro_f1(["met"], ["met"], ["met", "unknown"]) == pytest.approx(0.5)


def test_cohens_kappa_perfect_and_worst_case():
    y_true = ["met", "met", "not_met", "not_met"]
    assert cohens_kappa(y_true, list(y_true)) == pytest.approx(1.0, abs=TOL)
    # Complete disagreement with balanced marginals -> kappa = -1.
    flipped = ["not_met", "not_met", "met", "met"]
    assert cohens_kappa(y_true, flipped) == pytest.approx(-1.0, abs=TOL)


def test_cohens_kappa_hand_computed():
    # 4 agreements out of 6; marginals true(met=3, unknown=3), pred(met=4, unknown=2).
    y_true = ["met", "met", "met", "unknown", "unknown", "unknown"]
    y_pred = ["met", "met", "met", "met", "unknown", "unknown"]
    observed = 5 / 6
    expected_chance = (3 / 6) * (4 / 6) + (3 / 6) * (2 / 6)
    expected = (observed - expected_chance) / (1 - expected_chance)
    assert cohens_kappa(y_true, y_pred) == pytest.approx(expected, abs=TOL)


def test_cohens_kappa_degenerate_single_label():
    # Chance agreement is 1.0, so kappa is undefined; full agreement -> 1.0.
    assert cohens_kappa(["met", "met"], ["met", "met"]) == 1.0
    # Both annotators constant but different -> 0.0.
    assert cohens_kappa(["met", "met"], ["unknown", "unknown"]) == 0.0


# --------------------------------------------------------------------------
# evaluate_rankings / format_table / CLI
# --------------------------------------------------------------------------


QRELS = {"1": {"NCT01": 2, "NCT02": 0, "NCT03": 1}}
RANKINGS = {"1": ["NCT02", "NCT01", "NCT99"], "2": ["NCT01"]}


def test_evaluate_rankings_end_to_end_with_missing_topic():
    result = evaluate_rankings(RANKINGS, QRELS, ks=(5, 10))

    assert result["num_topics"] == 1
    assert result["skipped_topics"] == ["2"]
    assert len(result["warnings"]) == 1 and "2" in result["warnings"][0]
    assert list(result["per_topic"]) == ["1"]
    assert result["metric_keys"] == ["ndcg@5", "ndcg@10", "p@10", "mrr", "recall@50"]

    scores = result["per_topic"]["1"]
    expected_ndcg = (2 / math.log2(3)) / (2 / math.log2(2) + 1 / math.log2(3))
    assert scores["ndcg@5"] == pytest.approx(expected_ndcg, abs=TOL)
    assert scores["ndcg@10"] == pytest.approx(expected_ndcg, abs=TOL)
    assert scores["p@10"] == pytest.approx(0.1, abs=TOL)
    assert scores["mrr"] == pytest.approx(0.5, abs=TOL)
    assert scores["recall@50"] == pytest.approx(0.5, abs=TOL)

    # Only the scored topic contributes to the mean.
    assert result["mean"] == pytest.approx(scores)

    # The result must be JSON-serializable as-is.
    assert json.loads(json.dumps(result))["num_topics"] == 1


def test_evaluate_rankings_no_scorable_topic():
    result = evaluate_rankings({"9": ["NCT01"]}, QRELS)
    assert result["num_topics"] == 0
    assert result["per_topic"] == {}
    assert all(value == 0.0 for value in result["mean"].values())


def test_evaluate_rankings_rejects_bad_k():
    with pytest.raises(ValueError):
        evaluate_rankings(RANKINGS, QRELS, ks=(0,))


def test_format_table_lists_every_mean_metric():
    table = format_table(evaluate_rankings(RANKINGS, QRELS, ks=(5,)))
    assert "topics evaluated: 1" in table
    assert "topics skipped:   1 (2)" in table
    for key in ("ndcg@5", "p@10", "mrr", "recall@50"):
        assert key in table
    assert "0.5000" in table  # mrr and recall@50
    # Fixed-width layout: every metric row has the same length.
    rows = [line for line in table.splitlines() if line.startswith(("ndcg", "p@", "mrr", "recall"))]
    assert len({len(row) for row in rows}) == 1


def test_cli_round_trip(tmp_path, capsys):
    rankings_path = tmp_path / "rankings.json"
    rankings_path.write_text(json.dumps(RANKINGS), encoding="utf-8")
    qrels_path = tmp_path / "qrels.txt"
    qrels_path.write_text(
        "1 0 NCT01 2\n1 0 NCT02 0\n1 0 NCT03 1\nbroken line\n1 0 NCT04 x\n",
        encoding="utf-8",
    )
    out_path = tmp_path / "nested" / "result.json"

    assert main(["--rankings", str(rankings_path), "--qrels", str(qrels_path),
                 "--ks", "5,10", "--out", str(out_path)]) == 0

    stdout = capsys.readouterr().out
    assert "warning:" in stdout and "ndcg@10" in stdout

    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written["num_topics"] == 1
    assert written["skipped_topics"] == ["2"]

    # Malformed qrels lines are skipped, not fatal.
    assert load_qrels_file(qrels_path) == QRELS
    assert load_rankings_file(rankings_path) == RANKINGS


def test_load_rankings_file_rejects_non_object(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(TypeError):
        load_rankings_file(path)
