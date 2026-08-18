"""Evaluation layer: ranking / classification metrics and the TREC runner.

`bench` is deliberately NOT re-exported here: it pulls in the data, retrieval
and agent layers, and importing this package must stay cheap enough that
`run_trec` really is usable on its own (which is what justifies its duplicated
qrels parser). Use `trialmatch.eval.bench` directly for the judged-subset bench.
"""

from __future__ import annotations

from trialmatch.eval.metrics import (
    cohens_kappa,
    macro_f1,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    three_class_accuracy,
)
from trialmatch.eval.run_trec import evaluate_rankings, format_table

__all__ = [
    "cohens_kappa",
    "evaluate_rankings",
    "format_table",
    "macro_f1",
    "mrr",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "three_class_accuracy",
]
