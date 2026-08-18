"""Loaders for TREC Clinical Trials benchmark files (offline, no network).

The provided `synthetic-patients.json` uses the same topic shape as TREC CT
(a numbered id plus a free-text patient description), so both loaders return
the same `(topic_id, text)` pairs.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path


def load_topics_json(path: Path) -> list[tuple[str, str]]:
    """Load `{"topics": [{"num": ..., "title": ...}]}` into (topic_id, text)."""
    with Path(path).open(encoding="utf-8") as fh:
        payload = json.load(fh)

    topics: list[tuple[str, str]] = []
    for topic in payload.get("topics") or []:
        topic_id = str(topic.get("num", "")).strip()
        text = str(topic.get("title", "")).strip()
        if topic_id and text:
            topics.append((topic_id, text))
    return topics


def load_topics_xml(path: Path) -> list[tuple[str, str]]:
    """Load TREC CT topics XML: `<topics><topic number="1">text</topic></topics>`."""
    root = ET.parse(Path(path)).getroot()

    topics: list[tuple[str, str]] = []
    for element in root.iter("topic"):
        topic_id = (element.get("number") or element.get("num") or "").strip()
        text = " ".join("".join(element.itertext()).split())
        if topic_id and text:
            topics.append((topic_id, text))
    return topics


def load_qrels(path: Path) -> dict[str, dict[str, int]]:
    """Load TREC qrels (`topic_id iteration doc_id relevance`) lines.

    Blank and malformed lines are skipped rather than raising, so partially
    downloaded or commented qrels files stay usable.
    """
    qrels: dict[str, dict[str, int]] = {}
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            fields = line.split()
            if len(fields) < 4:
                continue
            topic_id, _iteration, doc_id, relevance = fields[:4]
            try:
                score = int(relevance)
            except ValueError:
                continue
            qrels.setdefault(topic_id, {})[doc_id] = score
    return qrels
