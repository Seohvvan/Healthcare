"""Offline TREC evaluation runner: scores stored rankings against qrels.

This module performs no retrieval and makes no LLM calls. It reads a rankings
JSON produced by the pipeline plus a TREC qrels file, and reports NDCG@k,
P@10, MRR and Recall@50 per topic and averaged over topics (design.md §5.2).

The qrels parser is duplicated here on purpose so this module — and the
`trialmatch.eval` package that exposes it — stays importable without the data
layer. `trialmatch.eval.__init__` therefore re-exports only `metrics` and this
runner; `bench`, which does need the data layer, is imported by its full path.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from trialmatch.eval.metrics import mrr, ndcg_at_k, precision_at_k, recall_at_k

__all__ = [
    "PRECISION_K",
    "RECALL_K",
    "evaluate_rankings",
    "format_table",
    "load_qrels_file",
    "load_rankings_file",
    "main",
]

PRECISION_K = 10
RECALL_K = 50

# Documents with gain >= this count as relevant for P@k / Recall@k / MRR.
# TREC CT gains: eligible = 2, excluded = 1, not relevant = 0.
RELEVANT_MIN_GAIN = 1


def evaluate_rankings(
    rankings: Mapping[str, Sequence[str]],
    qrels: Mapping[str, Mapping[str, int]],
    ks: Sequence[int] = (5, 10),
) -> dict:
    """Score every topic in `rankings` against `qrels`.

    Topics missing from `qrels` cannot be scored and are skipped; their ids are
    reported in `skipped_topics` with a human-readable line in `warnings`.

    Returns a plain nested dict (JSON-serializable) with `per_topic` scores,
    their `mean`, the metric key order, and the skip report.
    """
    ks = [int(k) for k in ks]
    for k in ks:
        if k <= 0:
            raise ValueError(f"k must be a positive integer, got {k}")

    metric_keys = [f"ndcg@{k}" for k in ks]
    metric_keys += [f"p@{PRECISION_K}", "mrr", f"recall@{RECALL_K}"]

    per_topic: dict[str, dict[str, float]] = {}
    skipped_topics: list[str] = []
    warnings: list[str] = []

    for topic_id in sorted(rankings):
        gains = qrels.get(topic_id)
        if gains is None:
            skipped_topics.append(topic_id)
            warnings.append(f"topic {topic_id!r} has no qrels entry; skipped")
            continue

        ranked = list(rankings[topic_id])
        scores = {f"ndcg@{k}": ndcg_at_k(ranked, gains, k) for k in ks}
        scores[f"p@{PRECISION_K}"] = precision_at_k(
            ranked, gains, PRECISION_K, min_gain=RELEVANT_MIN_GAIN
        )
        scores["mrr"] = mrr(ranked, gains, min_gain=RELEVANT_MIN_GAIN)
        scores[f"recall@{RECALL_K}"] = recall_at_k(
            ranked, gains, RECALL_K, min_gain=RELEVANT_MIN_GAIN
        )
        per_topic[topic_id] = scores

    num_topics = len(per_topic)
    mean = {
        key: (sum(scores[key] for scores in per_topic.values()) / num_topics)
        if num_topics
        else 0.0
        for key in metric_keys
    }

    return {
        "metric_keys": metric_keys,
        "num_topics": num_topics,
        "per_topic": per_topic,
        "mean": mean,
        "skipped_topics": skipped_topics,
        "warnings": warnings,
    }


def format_table(result: dict) -> str:
    """Render the mean metrics of `evaluate_rankings` as a fixed-width table."""
    metric_keys = list(result.get("metric_keys") or result.get("mean", {}))
    mean = result.get("mean", {})

    name_width = max((len(key) for key in metric_keys), default=6)
    name_width = max(name_width, len("metric"))
    value_width = 8

    lines = [f"topics evaluated: {result.get('num_topics', 0)}"]
    skipped = result.get("skipped_topics") or []
    if skipped:
        lines.append(f"topics skipped:   {len(skipped)} ({', '.join(map(str, skipped))})")
    lines.append("")
    lines.append(f"{'metric':<{name_width}}  {'mean':>{value_width}}")
    lines.append(f"{'-' * name_width}  {'-' * value_width}")
    for key in metric_keys:
        lines.append(f"{key:<{name_width}}  {mean.get(key, 0.0):>{value_width}.4f}")
    return "\n".join(lines)


def load_rankings_file(path: Path) -> dict[str, list[str]]:
    """Load `{topic_id: [nct_id, ...]}` rankings JSON, coercing ids to strings."""
    with Path(path).open(encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise TypeError(f"rankings file must hold a JSON object, got {type(payload).__name__}")
    return {str(topic_id): [str(doc) for doc in docs] for topic_id, docs in payload.items()}


def load_qrels_file(path: Path) -> dict[str, dict[str, int]]:
    """Parse TREC qrels lines (`topic iter doc rel`); malformed lines are skipped."""
    qrels: dict[str, dict[str, int]] = {}
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            fields = line.split()
            if len(fields) < 4:
                continue
            topic_id, _iteration, doc_id, relevance = fields[:4]
            try:
                gain = int(relevance)
            except ValueError:
                continue
            qrels.setdefault(topic_id, {})[doc_id] = gain
    return qrels


def _parse_ks(raw: str) -> list[int]:
    try:
        ks = [int(part) for part in raw.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid --ks value: {raw!r}") from exc
    if not ks or any(k <= 0 for k in ks):
        raise argparse.ArgumentTypeError(f"--ks must be positive integers, got {raw!r}")
    return ks


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="trialmatch-eval-trec",
        description="Score stored rankings against TREC Clinical Trials qrels.",
    )
    parser.add_argument("--rankings", required=True, type=Path, help="JSON {topic_id: [nct_id]}")
    parser.add_argument("--qrels", required=True, type=Path, help="TREC qrels file")
    parser.add_argument("--ks", default="5,10", type=_parse_ks, help="NDCG cutoffs, e.g. 5,10")
    parser.add_argument("--out", type=Path, default=None, help="write the full result JSON here")
    args = parser.parse_args(argv)

    rankings = load_rankings_file(args.rankings)
    qrels = load_qrels_file(args.qrels)
    result = evaluate_rankings(rankings, qrels, ks=args.ks)

    for warning in result["warnings"]:
        print(f"warning: {warning}")
    print(format_table(result))

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, sort_keys=True)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
