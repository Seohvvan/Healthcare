"""TREC Clinical Trials judged-subset benchmark (design.md §0, §5.2).

TREC CT qrels were assessed against the 2021-04 ClinicalTrials.gov snapshot, so
scoring them against a fresh crawl is invalid: the crawl misses trials that no
longer exist and the criteria text has drifted since. This module therefore
evaluates **re-ranking over each topic's judged trials only** — the candidate
set is exactly the set of NCT ids the assessors looked at.

Corpus preference, best first:
  1. `load_trialgpt_corpus` — the TrialGPT repository's preprocessed 2021 text.
  2. `fetch_judged_trials` — current API v2 records for the judged ids, which
     carry a text-drift caveat that must be reported with any result.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import httpx

from trialmatch.agents import (
    aggregate,
    assess_trial,
    parse_criteria,
    profile_patient,
    rank,
)
from trialmatch.config import Settings, load_dotenv
from trialmatch.data.ctgov import CTGovClient, parse_study
from trialmatch.data.store import load_trials, write_jsonl
from trialmatch.data.trec import load_qrels, load_topics_json, load_topics_xml
from trialmatch.eval.metrics import macro_f1, three_class_accuracy
from trialmatch.eval.run_trec import evaluate_rankings, format_table
from trialmatch.llm import SupportsLLM, create_llm
from trialmatch.retrieval.bm25 import BM25Index
from trialmatch.schemas import TrialAssessment, TrialLabel, TrialRecord

__all__ = [
    "DEFAULT_CANDIDATES_PER_TOPIC",
    "DEFAULT_MAX_TOPICS",
    "bm25_judged_rerank",
    "fetch_judged_trials",
    "llm_judged_rerank",
    "load_trial_corpus",
    "load_trialgpt_corpus",
    "main",
]

logger = logging.getLogger(__name__)

# Budget caps for the LLM bench (design.md §5.2: cap topics and candidates).
# Worst case per topic: 1 profiler call + candidates_per_topic matcher calls
# (+ one criteria-parser call per candidate whose text needs the LLM fallback).
DEFAULT_MAX_TOPICS = 5
DEFAULT_CANDIDATES_PER_TOPIC = 20

# TREC CT graded relevance: eligible = 2, excluded = 1, not relevant = 0.
_ELIGIBLE_GAIN = 2
_EXCLUDED_GAIN = 1

_TRIAL_LABELS: tuple[str, ...] = tuple(label.value for label in TrialLabel)

_ID_FIELDS = ("nct_id", "nctId", "NCTID", "nctid", "id", "_id")
_TITLE_FIELDS = ("brief_title", "title", "official_title")
_CONDITION_FIELDS = ("diseases_list", "conditions")
_SUMMARY_FIELDS = ("brief_summary", "summary")
_CRITERIA_FIELDS = ("criteria", "eligibility_criteria")
_INCLUSION_FIELDS = ("inclusion_criteria",)
_EXCLUSION_FIELDS = ("exclusion_criteria",)

# Records are written in batches so a multi-thousand-id fetch never holds the
# whole corpus in memory and a partial run leaves a usable file behind.
_FETCH_BATCH = 200


# --------------------------------------------------------------------------- #
# 1. TrialGPT corpus loader
# --------------------------------------------------------------------------- #


def load_trialgpt_corpus(path: Path) -> dict[str, TrialRecord]:
    """Load the TrialGPT repository's preprocessed trial-info JSON (2021 text).

    This is the preferred benchmark corpus: its text matches the snapshot the
    TREC assessors judged, so criteria drift cannot distort the scores.

    The exact field layout is only verified at download time, so the loader is
    deliberately permissive. It accepts either a JSON object keyed by NCT id or
    a list of objects each carrying an id field, and maps the known field-name
    variants (`brief_title`/`title`, `diseases_list`/`conditions`,
    `criteria` or `inclusion_criteria` + `exclusion_criteria`, ...). Unknown
    fields are ignored; records with no resolvable NCT id are skipped and the
    total is logged.
    """
    with Path(path).open(encoding="utf-8") as fh:
        payload = json.load(fh)

    if isinstance(payload, dict):
        items: list[tuple[str | None, Any]] = list(payload.items())
    elif isinstance(payload, list):
        items = [(None, entry) for entry in payload]
    else:
        raise TypeError(
            f"trial corpus must be a JSON object or list, got {type(payload).__name__}"
        )

    corpus: dict[str, TrialRecord] = {}
    skipped = 0
    for key, raw in items:
        if not isinstance(raw, Mapping):
            skipped += 1
            continue
        nct_id = _resolve_nct_id(raw, key)
        if not nct_id:
            skipped += 1
            continue
        corpus[nct_id] = TrialRecord(
            nct_id=nct_id,
            title=_first_text(raw, _TITLE_FIELDS),
            conditions=_as_str_list(_first_value(raw, _CONDITION_FIELDS)),
            brief_summary=_first_text(raw, _SUMMARY_FIELDS),
            eligibility_text=_eligibility_text(raw),
        )

    if skipped:
        logger.warning("skipped %d corpus record(s) without a resolvable NCT id", skipped)
    return corpus


def load_trial_corpus(path: Path) -> dict[str, TrialRecord]:
    """Load a corpus by extension: `.jsonl` snapshot vs TrialGPT `.json`."""
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        return load_trials(path)
    return load_trialgpt_corpus(path)


def _resolve_nct_id(raw: Mapping[str, Any], key: str | None) -> str:
    """Resolve the NCT id from the mapping key or any known id field."""
    candidates: list[str] = []
    if isinstance(key, str) and key.strip():
        candidates.append(key.strip())
    for field in _ID_FIELDS:
        value = raw.get(field)
        if isinstance(value, str) and value.strip():
            candidates.append(value.strip())

    for candidate in candidates:
        if candidate.upper().startswith("NCT"):
            return candidate
    return candidates[0] if candidates else ""


def _first_value(raw: Mapping[str, Any], fields: Sequence[str]) -> Any:
    for field in fields:
        value = raw.get(field)
        if value:
            return value
    return None


def _first_text(raw: Mapping[str, Any], fields: Sequence[str]) -> str:
    """First non-empty field as text; list values are joined line by line."""
    value = _first_value(raw, fields)
    return _as_text(value)


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        lines = [str(item).strip() for item in value if str(item).strip()]
        return "\n".join(lines)
    return ""


def _as_str_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _eligibility_text(raw: Mapping[str, Any]) -> str:
    """Raw criteria text, rebuilt from split inclusion/exclusion when needed."""
    direct = _first_text(raw, _CRITERIA_FIELDS)
    if direct:
        return direct

    blocks: list[str] = []
    inclusion = _first_text(raw, _INCLUSION_FIELDS)
    if inclusion:
        blocks.append(f"Inclusion Criteria:\n{inclusion}")
    exclusion = _first_text(raw, _EXCLUSION_FIELDS)
    if exclusion:
        blocks.append(f"Exclusion Criteria:\n{exclusion}")
    return "\n\n".join(blocks)


# --------------------------------------------------------------------------- #
# 2. Judged-id fetch from the live API (fallback corpus)
# --------------------------------------------------------------------------- #


def fetch_judged_trials(
    qrels: Mapping[str, Mapping[str, int]],
    out_path: Path,
    *,
    client: CTGovClient | None = None,
    transport: httpx.BaseTransport | None = None,
) -> int:
    """Fetch the CURRENT record of every judged NCT id into a JSONL snapshot.

    ⚠️ Drift caveat: the text returned here is today's, not the 2021-04 text the
    TREC assessors judged. Criteria are routinely amended, so a benchmark run on
    this snapshot measures the engine against *drifted* documents and every
    reported number must say so. Prefer `load_trialgpt_corpus` whenever the
    TrialGPT preprocessed distribution is available.

    Ids the API no longer serves (404) are skipped and counted in the log.
    Returns the number of records written.
    """
    nct_ids = sorted({str(doc_id) for gains in qrels.values() for doc_id in gains})
    out_path = Path(out_path)

    owns_client = client is None
    ctgov = client if client is not None else CTGovClient(transport=transport)

    written = 0
    missing = 0
    batch: list[TrialRecord] = []
    try:
        for nct_id in nct_ids:
            raw = ctgov.get_study(nct_id)
            if raw is None:
                missing += 1
                continue
            record = parse_study(raw)
            if not record.nct_id:
                record = record.model_copy(update={"nct_id": nct_id})
            batch.append(record)
            if len(batch) >= _FETCH_BATCH:
                written += write_jsonl(batch, out_path, append=written > 0)
                batch = []
        # Always write once, so an all-404 run still truncates/creates the file.
        if batch or written == 0:
            written += write_jsonl(batch, out_path, append=written > 0)
    finally:
        if owns_client:
            ctgov.close()

    logger.info(
        "fetched %d/%d judged trials into %s (%d not found)",
        written,
        len(nct_ids),
        out_path,
        missing,
    )
    return written


# --------------------------------------------------------------------------- #
# 3. BM25 re-ranking bench
# --------------------------------------------------------------------------- #


def _judged_candidates(
    topic_id: str,
    qrels: Mapping[str, Mapping[str, int]],
    trials: Mapping[str, TrialRecord],
) -> tuple[list[str], dict[str, int]]:
    """Judged ids for a topic that are present in `trials`, plus coverage counts."""
    judged = sorted(qrels.get(topic_id) or {})
    candidates = [nct_id for nct_id in judged if nct_id in trials]
    return candidates, {"judged": len(judged), "in_corpus": len(candidates)}


def _bm25_order(
    text: str, candidates: Sequence[str], trials: Mapping[str, TrialRecord]
) -> list[str]:
    """Rank `candidates` by BM25 against `text`; never-retrieved ids trail in id order."""
    if not candidates:
        return []
    index = BM25Index.build([trials[nct_id] for nct_id in candidates])
    ranked = [nct_id for nct_id, _score in index.search(text, k=len(candidates))]
    retrieved = set(ranked)
    ranked.extend(nct_id for nct_id in candidates if nct_id not in retrieved)
    return ranked


def bm25_judged_rerank(
    topics: Sequence[tuple[str, str]],
    qrels: Mapping[str, Mapping[str, int]],
    trials: Mapping[str, TrialRecord],
    ks: Sequence[int] = (5, 10),
) -> dict:
    """BM25 re-ranking of each topic's judged trials — the zero-LLM baseline.

    The candidate set is the topic's judged ids that exist in `trials`; ids the
    corpus lacks are counted per topic under `coverage` and stay unranked, so
    the metrics show the coverage loss instead of hiding it.
    """
    rankings: dict[str, list[str]] = {}
    coverage: dict[str, dict[str, int]] = {}

    for topic_id, text in topics:
        candidates, counts = _judged_candidates(topic_id, qrels, trials)
        coverage[topic_id] = counts
        rankings[topic_id] = _bm25_order(text, candidates, trials)

    result = evaluate_rankings(rankings, qrels, ks=ks)
    result["coverage"] = coverage
    result["warnings"].extend(_coverage_warnings(coverage))
    return result


def _coverage_warnings(coverage: Mapping[str, Mapping[str, int]]) -> list[str]:
    return [
        f"topic {topic_id!r}: {counts['judged'] - counts['in_corpus']} of "
        f"{counts['judged']} judged trials are missing from the corpus"
        for topic_id, counts in sorted(coverage.items())
        if counts["in_corpus"] < counts["judged"]
    ]


# --------------------------------------------------------------------------- #
# 4. LLM matching bench (budget-capped)
# --------------------------------------------------------------------------- #


def _gain_to_label(gain: int) -> TrialLabel:
    """Map a TREC gain onto the trial-level verdict schema."""
    if gain >= _ELIGIBLE_GAIN:
        return TrialLabel.ELIGIBLE
    if gain == _EXCLUDED_GAIN:
        return TrialLabel.EXCLUDED
    return TrialLabel.NOT_RELEVANT


def llm_judged_rerank(
    llm: SupportsLLM,
    topics: Sequence[tuple[str, str]],
    qrels: Mapping[str, Mapping[str, int]],
    trials: Mapping[str, TrialRecord],
    settings: Settings,
    *,
    max_topics: int | None = DEFAULT_MAX_TOPICS,
    candidates_per_topic: int = DEFAULT_CANDIDATES_PER_TOPIC,
    use_llm_criteria: bool = True,
    ks: Sequence[int] = (5, 10),
) -> dict:
    """Full matching pipeline over the judged subset, capped to a fixed budget.

    Per topic: BM25 pre-ranks the judged candidates, the top
    `candidates_per_topic` go through profile → criteria → assessment →
    deterministic aggregation, and `rank` orders them exactly as the product
    does (label grade first, then score). Candidates beyond the cap keep their
    judged status but trail in id order, which is what a real budget-limited run
    would produce.

    A candidate whose assessment raises is skipped, recorded under `errors`, and
    left out of both metric families rather than aborting the whole bench.

    Reports both metric families of design.md §5.2:
      * ranking — `evaluate_rankings` over the re-ranked judged subset;
      * classification — three-class accuracy / macro-F1 of the trial verdict
        against the qrel gain (eligible 2 / excluded 1 / not relevant 0) over
        the *assessed* candidates only. A trial the pipeline leaves undetermined
        (open UNKNOWNs, `trial_label is None`) counts as `not_relevant` for the
        accuracy and is also reported separately as `undetermined`.
    """
    selected = list(topics) if max_topics is None else list(topics)[:max_topics]

    rankings: dict[str, list[str]] = {}
    coverage: dict[str, dict[str, int]] = {}
    errors: list[dict] = []
    y_true: list[str] = []
    y_pred: list[str] = []
    undetermined = 0

    for topic_id, text in selected:
        candidates, counts = _judged_candidates(topic_id, qrels, trials)
        preranked = _bm25_order(text, candidates, trials)
        assessed_ids = preranked[:candidates_per_topic]
        deferred = sorted(set(preranked) - set(assessed_ids))
        coverage[topic_id] = {**counts, "assessed": len(assessed_ids)}

        if not assessed_ids:
            rankings[topic_id] = deferred
            continue

        profile = profile_patient(llm, topic_id, text, settings.models)

        assessed: list[TrialAssessment] = []
        gains = qrels.get(topic_id) or {}
        for nct_id in assessed_ids:
            trial = trials[nct_id]
            try:
                parsed = parse_criteria(llm, trial, settings.models, use_llm=use_llm_criteria)
                assessment = aggregate(
                    assess_trial(llm, profile, parsed, trial, settings.models),
                    parsed,
                    settings.ranking,
                )
            except Exception as exc:  # noqa: BLE001 — one trial must not sink the bench
                errors.append({"topic_id": topic_id, "nct_id": nct_id, "error": str(exc)})
                continue
            assessed.append(assessment)

            if assessment.trial_label is None:
                undetermined += 1
            predicted = assessment.trial_label or TrialLabel.NOT_RELEVANT
            y_pred.append(predicted.value)
            y_true.append(_gain_to_label(gains.get(nct_id, 0)).value)

        failed = sorted({str(err["nct_id"]) for err in errors if err["topic_id"] == topic_id})
        ordered = [assessment.nct_id for assessment in rank(assessed)]
        rankings[topic_id] = ordered + deferred + failed

    ranking_result = evaluate_rankings(rankings, qrels, ks=ks)
    ranking_result["warnings"].extend(_coverage_warnings(coverage))

    classification = {
        "num_assessed": len(y_true),
        "accuracy": three_class_accuracy(y_true, y_pred) if y_true else 0.0,
        "macro_f1": macro_f1(y_true, y_pred, _TRIAL_LABELS) if y_true else 0.0,
        "undetermined": undetermined,
        "labels": list(_TRIAL_LABELS),
    }

    return {
        "config": {
            "max_topics": max_topics,
            "candidates_per_topic": candidates_per_topic,
            "use_llm_criteria": use_llm_criteria,
        },
        "ranking": ranking_result,
        "classification": classification,
        "coverage": coverage,
        "errors": errors,
    }


# --------------------------------------------------------------------------- #
# 5. CLI
# --------------------------------------------------------------------------- #


def format_coverage(coverage: Mapping[str, Mapping[str, int]]) -> str:
    """Summarise judged-vs-available candidate counts over all topics."""
    judged = sum(counts.get("judged", 0) for counts in coverage.values())
    in_corpus = sum(counts.get("in_corpus", 0) for counts in coverage.values())
    assessed = sum(counts.get("assessed", 0) for counts in coverage.values())
    share = (in_corpus / judged) if judged else 0.0
    lines = [
        f"topics with candidates: {len(coverage)}",
        f"judged trials:          {judged}",
        f"present in corpus:      {in_corpus} ({share:.1%})",
    ]
    if assessed:
        lines.append(f"assessed by the LLM:    {assessed}")
    return "\n".join(lines)


def _parse_ks(raw: str) -> list[int]:
    try:
        ks = [int(part) for part in raw.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid --ks value: {raw!r}") from exc
    if not ks or any(k <= 0 for k in ks):
        raise argparse.ArgumentTypeError(f"--ks must be positive integers, got {raw!r}")
    return ks


def _add_topic_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--topics-xml", type=Path, help="TREC CT topics XML")
    group.add_argument("--topics-json", type=Path, help="topics JSON ({'topics': [...]})")


def _load_topics(args: argparse.Namespace) -> list[tuple[str, str]]:
    if args.topics_xml is not None:
        return load_topics_xml(args.topics_xml)
    return load_topics_json(args.topics_json)


def _write_result(result: dict, out: Path | None) -> None:
    if out is None:
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)
    print(f"wrote {out}")


def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv()  # credentials/model overrides from <project>/.env, if present
    parser = argparse.ArgumentParser(
        prog="trialmatch-eval-bench",
        description=(
            "TREC judged-subset re-ranking benchmark. Candidates are a topic's "
            "judged trials only, because the qrels were assessed against the "
            "2021-04 corpus snapshot."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch = subparsers.add_parser(
        "fetch-judged",
        help="fetch the judged NCT ids from the live API (drifted text: see the docs)",
    )
    fetch.add_argument("--qrels", required=True, type=Path, help="TREC qrels file")
    fetch.add_argument("--out", required=True, type=Path, help="JSONL snapshot to write")

    bm25 = subparsers.add_parser("bm25", help="BM25 re-ranking baseline (no LLM calls)")
    _add_topic_args(bm25)
    bm25.add_argument("--qrels", required=True, type=Path)
    bm25.add_argument("--trials", required=True, type=Path, help=".jsonl snapshot or TrialGPT .json")
    bm25.add_argument("--ks", default="5,10", type=_parse_ks, help="NDCG cutoffs, e.g. 5,10")
    bm25.add_argument("--out", type=Path, default=None, help="write the full result JSON here")

    llm = subparsers.add_parser(
        "llm",
        help="LLM matching bench over the judged subset",
        description=(
            "Budget-capped LLM bench. Defaults evaluate "
            f"{DEFAULT_MAX_TOPICS} topics x {DEFAULT_CANDIDATES_PER_TOPIC} candidates, "
            f"i.e. at most {DEFAULT_MAX_TOPICS} profiler calls plus "
            f"{DEFAULT_MAX_TOPICS * DEFAULT_CANDIDATES_PER_TOPIC} matcher calls. "
            "Raise the caps deliberately: cost grows linearly in both."
        ),
    )
    _add_topic_args(llm)
    llm.add_argument("--qrels", required=True, type=Path)
    llm.add_argument("--trials", required=True, type=Path, help=".jsonl snapshot or TrialGPT .json")
    llm.add_argument("--max-topics", type=int, default=DEFAULT_MAX_TOPICS)
    llm.add_argument("--candidates", type=int, default=DEFAULT_CANDIDATES_PER_TOPIC)
    llm.add_argument("--ks", default="5,10", type=_parse_ks, help="NDCG cutoffs, e.g. 5,10")
    llm.add_argument(
        "--provider",
        choices=["anthropic", "gemini"],
        default="anthropic",
        help="LLM provider; picks the model preset in config.py (default: anthropic)",
    )
    llm.add_argument("--out", type=Path, default=None, help="write the full result JSON here")

    args = parser.parse_args(argv)

    if args.command == "fetch-judged":
        written = fetch_judged_trials(load_qrels(args.qrels), args.out)
        print(f"wrote {written} judged trials to {args.out}")
        print(
            "caveat: these are today's records, not the 2021-04 text the TREC "
            "assessors judged; report the drift with any result."
        )
        return 0

    topics = _load_topics(args)
    qrels = load_qrels(args.qrels)
    trials = load_trial_corpus(args.trials)

    if args.command == "bm25":
        result = bm25_judged_rerank(topics, qrels, trials, ks=args.ks)
        for warning in result["warnings"]:
            print(f"warning: {warning}")
        print(format_table(result))
        print()
        print(format_coverage(result["coverage"]))
        _write_result(result, args.out)
        return 0

    # `llm`: build the real client here so importing this module needs no key.
    settings = Settings().with_provider(args.provider)
    result = llm_judged_rerank(
        create_llm(settings.models),
        topics,
        qrels,
        trials,
        settings,
        max_topics=args.max_topics,
        candidates_per_topic=args.candidates,
        ks=args.ks,
    )
    for warning in result["ranking"]["warnings"]:
        print(f"warning: {warning}")
    for error in result["errors"]:
        print(f"error: {error['nct_id']} (topic {error['topic_id']}): {error['error']}")
    print(format_table(result["ranking"]))
    print()
    classification = result["classification"]
    print(
        f"trial-label accuracy: {classification['accuracy']:.4f}  "
        f"macro-F1: {classification['macro_f1']:.4f}  "
        f"(n={classification['num_assessed']}, "
        f"undetermined={classification['undetermined']})"
    )
    print()
    print(format_coverage(result["coverage"]))
    _write_result(result, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
