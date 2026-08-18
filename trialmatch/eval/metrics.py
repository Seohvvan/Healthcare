"""Pure metric functions for ranking and classification evaluation.

Ranking metrics follow the TREC Clinical Trials setup (design.md §5): graded
relevance with `eligible = 2`, `excluded = 1`, `not relevant = 0`. Documents
absent from `gains` are unjudged and treated as gain 0.

Classification metrics cover the criterion-level three-way verdict
(MET / NOT MET / UNKNOWN) and are implemented directly, without scikit-learn.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence

__all__ = [
    "cohens_kappa",
    "macro_f1",
    "mrr",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "three_class_accuracy",
]


def _validate_k(k: int) -> None:
    if k <= 0:
        raise ValueError(f"k must be a positive integer, got {k}")


def _validate_pair(y_true: Sequence[str], y_pred: Sequence[str]) -> None:
    if len(y_true) != len(y_pred):
        raise ValueError(f"length mismatch: len(y_true)={len(y_true)}, len(y_pred)={len(y_pred)}")
    if not y_true:
        raise ValueError("y_true and y_pred must not be empty")


def _discounted_gain(gain: float, rank: int) -> float:
    """Standard TREC discount: gain / log2(rank + 1), with rank starting at 1."""
    return gain / math.log2(rank + 1)


def ndcg_at_k(ranked_ids: Sequence[str], gains: Mapping[str, int], k: int) -> float:
    """Normalized discounted cumulative gain over the top `k` ranked documents.

    The ideal DCG is built from the highest gains among *all* judged documents
    in `gains`, so a system is penalised for relevant documents it never
    retrieved. Returns 0.0 when no judged document carries a positive gain.
    """
    _validate_k(k)

    dcg = 0.0
    for rank, doc_id in enumerate(ranked_ids[:k], start=1):
        gain = gains.get(doc_id, 0)
        if gain:
            dcg += _discounted_gain(gain, rank)

    ideal_gains = sorted((g for g in gains.values() if g > 0), reverse=True)[:k]
    idcg = sum(_discounted_gain(gain, rank) for rank, gain in enumerate(ideal_gains, start=1))
    if idcg <= 0.0:
        return 0.0
    return dcg / idcg


def precision_at_k(
    ranked_ids: Sequence[str],
    gains: Mapping[str, int],
    k: int,
    min_gain: int = 1,
) -> float:
    """Fraction of the top `k` ranks holding a document with gain >= `min_gain`.

    The denominator is always `k` (standard TREC P@k), so a ranking shorter
    than `k` is penalised for the missing ranks.
    """
    _validate_k(k)
    hits = sum(1 for doc_id in ranked_ids[:k] if gains.get(doc_id, 0) >= min_gain)
    return hits / k


def recall_at_k(
    ranked_ids: Sequence[str],
    gains: Mapping[str, int],
    k: int,
    min_gain: int = 1,
) -> float:
    """Share of all relevant judged documents that appear in the top `k`.

    Returns 0.0 when the topic has no judged document with gain >= `min_gain`.
    """
    _validate_k(k)
    total_relevant = sum(1 for gain in gains.values() if gain >= min_gain)
    if total_relevant == 0:
        return 0.0
    hits = sum(1 for doc_id in ranked_ids[:k] if gains.get(doc_id, 0) >= min_gain)
    return hits / total_relevant


def mrr(ranked_ids: Sequence[str], gains: Mapping[str, int], min_gain: int = 1) -> float:
    """Reciprocal rank of the first relevant document, or 0.0 if there is none."""
    for rank, doc_id in enumerate(ranked_ids, start=1):
        if gains.get(doc_id, 0) >= min_gain:
            return 1.0 / rank
    return 0.0


def three_class_accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    """Plain accuracy; named for the MET / NOT MET / UNKNOWN verdict schema."""
    _validate_pair(y_true, y_pred)
    correct = sum(1 for true, pred in zip(y_true, y_pred, strict=True) if true == pred)
    return correct / len(y_true)


def macro_f1(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> float:
    """Unweighted mean of the per-label F1 scores over `labels`.

    A label with no true positives, false positives and false negatives (i.e.
    absent from both sequences) contributes 0.0, matching the convention of
    scikit-learn's `zero_division=0`.
    """
    _validate_pair(y_true, y_pred)
    if not labels:
        raise ValueError("labels must not be empty")

    scores = []
    for label in labels:
        tp = fp = fn = 0
        for true, pred in zip(y_true, y_pred, strict=True):
            if pred == label and true == label:
                tp += 1
            elif pred == label:
                fp += 1
            elif true == label:
                fn += 1
        denominator = 2 * tp + fp + fn
        scores.append(0.0 if denominator == 0 else (2 * tp) / denominator)
    return sum(scores) / len(scores)


def cohens_kappa(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    """Cohen's kappa for two annotators over the labels present in the data.

    When chance agreement is 1.0 (both annotators used a single label for
    everything) kappa is undefined; 1.0 is returned for full agreement and
    0.0 otherwise.
    """
    _validate_pair(y_true, y_pred)

    n = len(y_true)
    observed = sum(1 for true, pred in zip(y_true, y_pred, strict=True) if true == pred) / n

    true_counts = Counter(y_true)
    pred_counts = Counter(y_pred)
    expected = sum(
        (true_counts[label] / n) * (pred_counts[label] / n)
        for label in set(true_counts) | set(pred_counts)
    )

    if math.isclose(expected, 1.0):
        return 1.0 if math.isclose(observed, 1.0) else 0.0
    return (observed - expected) / (1.0 - expected)
