"""Data layer: ClinicalTrials.gov ingestion, TREC benchmark loaders, JSONL store."""

from __future__ import annotations

from trialmatch.data.ctgov import CTGovClient, download_snapshot, parse_study
from trialmatch.data.store import load_trials, read_jsonl, write_jsonl
from trialmatch.data.trec import load_qrels, load_topics_json, load_topics_xml

__all__ = [
    "CTGovClient",
    "download_snapshot",
    "load_qrels",
    "load_topics_json",
    "load_topics_xml",
    "load_trials",
    "parse_study",
    "read_jsonl",
    "write_jsonl",
]
