"""Reciprocal Rank Fusion for combining lexical and dense rankings."""

from __future__ import annotations

from collections.abc import Sequence


def rrf(
    rankings: Sequence[Sequence[str]],
    k: int = 60,
    top_n: int | None = None,
) -> list[tuple[str, float]]:
    """Fuse several best-first id rankings via Reciprocal Rank Fusion.

    score(id) = sum over rankings of 1 / (k + rank), with `rank` starting at 1.
    Ids absent from a ranking contribute nothing from it. Results are sorted by
    descending score, ties broken by id for determinism.

    Args:
        rankings: sequences of ids, each ordered best-first.
        k: RRF damping constant (60 in the original paper).
        top_n: keep only the best `top_n` fused ids; `None` keeps all.
    """
    if k <= 0:
        raise ValueError("rrf constant k must be positive")

    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)

    fused = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
    if top_n is None:
        return fused
    if top_n <= 0:
        return []
    return fused[:top_n]
