"""Lexical BM25 retrieval over the local trial snapshot.

Document text per trial = title + conditions + brief summary + eligibility text
(design.md §2). Scores are deterministic: ties break on `nct_id`.
"""

from __future__ import annotations

import pickle
import re
from collections.abc import Iterable
from pathlib import Path

from rank_bm25 import BM25Okapi

from trialmatch.schemas import TrialRecord

_TOKEN_SPLIT_RE = re.compile(r"[^a-z0-9]+")

# Small English stopword set. Deliberately conservative: clinical terms such as
# "no", "not", "male" must survive tokenization.
STOPWORDS: frozenset[str] = frozenset(
    {
        "the",
        "and",
        "for",
        "are",
        "was",
        "were",
        "with",
        "that",
        "this",
        "these",
        "those",
        "from",
        "have",
        "has",
        "had",
        "been",
        "being",
        "will",
        "would",
        "should",
        "could",
        "into",
        "onto",
        "than",
        "then",
        "there",
        "their",
        "them",
        "they",
        "who",
        "whom",
        "which",
        "what",
        "when",
        "where",
        "such",
        "also",
        "any",
        "all",
        "but",
        "not",
        "its",
        "his",
        "her",
        "our",
        "your",
    }
)


def tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumeric, drop short tokens and stopwords."""
    tokens = _TOKEN_SPLIT_RE.split(text.lower())
    return [t for t in tokens if len(t) >= 2 and t not in STOPWORDS]


def trial_document(trial: TrialRecord) -> str:
    """Concatenate the indexed fields of a trial into one document string."""
    parts = [
        trial.title,
        " ".join(trial.conditions),
        trial.brief_summary,
        trial.eligibility_text,
    ]
    return "\n".join(part for part in parts if part)


class BM25Index:
    """BM25Okapi index over trial documents, keyed by `nct_id`."""

    def __init__(self, nct_ids: list[str], bm25: BM25Okapi) -> None:
        self.nct_ids = nct_ids
        self._bm25 = bm25

    def __len__(self) -> int:
        return len(self.nct_ids)

    @classmethod
    def build(cls, trials: Iterable[TrialRecord]) -> BM25Index:
        """Build an index; document order follows the input iteration order."""
        nct_ids: list[str] = []
        corpus: list[list[str]] = []
        for trial in trials:
            nct_ids.append(trial.nct_id)
            corpus.append(tokenize(trial_document(trial)))
        if not corpus:
            raise ValueError("cannot build a BM25 index from an empty trial collection")
        return cls(nct_ids, BM25Okapi(corpus))

    def search(self, query: str, k: int = 100) -> list[tuple[str, float]]:
        """Return the top-`k` (nct_id, score) pairs, best first.

        Ties break on `nct_id` so repeated runs produce identical orderings.
        """
        if k <= 0:
            return []
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(
            zip(self.nct_ids, (float(s) for s in scores), strict=True),
            key=lambda pair: (-pair[1], pair[0]),
        )
        return ranked[:k]

    def save(self, path: Path) -> None:
        """Pickle the index to `path`, creating parent directories as needed."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump({"nct_ids": self.nct_ids, "bm25": self._bm25}, fh)

    @classmethod
    def load(cls, path: Path) -> BM25Index:
        """Load an index previously written by `save`."""
        with Path(path).open("rb") as fh:
            # Only ever load index files this project produced.
            payload = pickle.load(fh)
        return cls(payload["nct_ids"], payload["bm25"])
