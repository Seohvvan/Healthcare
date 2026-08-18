"""Dense embedding retrieval (optional extra).

Mirrors the `BM25Index` build/search interface so the two can be fused by
`trialmatch.retrieval.fusion.rrf`. sentence-transformers is an optional
dependency and is imported lazily *inside* the methods, never at module import
time, so the base install stays lean (design.md §2).
"""

from __future__ import annotations

import pickle
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from trialmatch.retrieval.bm25 import trial_document
from trialmatch.schemas import TrialRecord

# MedCPT is a two-tower model: documents and queries use different encoders.
DEFAULT_DOC_MODEL = "ncbi/MedCPT-Article-Encoder"
DEFAULT_QUERY_MODEL = "ncbi/MedCPT-Query-Encoder"

_INSTALL_HINT = (
    "DenseIndex requires the optional 'dense' extra (sentence-transformers). "
    "Install it with: uv sync --extra dense"
)


def _load_encoder(model_name: str) -> Any:
    """Import sentence-transformers lazily and return a loaded encoder."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError(_INSTALL_HINT) from exc
    return SentenceTransformer(model_name)


def _normalize(matrix: np.ndarray) -> np.ndarray:
    """L2-normalize rows so dot products are cosine similarities."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


class DenseIndex:
    """Cosine-similarity index over MedCPT (or compatible) embeddings."""

    def __init__(
        self,
        nct_ids: list[str],
        embeddings: np.ndarray,
        query_model: str = DEFAULT_QUERY_MODEL,
    ) -> None:
        self.nct_ids = nct_ids
        self.embeddings = embeddings
        self.query_model = query_model

    def __len__(self) -> int:
        return len(self.nct_ids)

    @classmethod
    def build(
        cls,
        trials: Iterable[TrialRecord],
        doc_model: str = DEFAULT_DOC_MODEL,
        query_model: str = DEFAULT_QUERY_MODEL,
        batch_size: int = 32,
    ) -> DenseIndex:
        """Embed every trial document; raises RuntimeError without the extra."""
        nct_ids: list[str] = []
        documents: list[str] = []
        for trial in trials:
            nct_ids.append(trial.nct_id)
            documents.append(trial_document(trial))
        if not documents:
            raise ValueError("cannot build a dense index from an empty trial collection")

        encoder = _load_encoder(doc_model)
        embeddings = np.asarray(
            encoder.encode(documents, batch_size=batch_size, show_progress_bar=False),
            dtype=np.float32,
        )
        return cls(nct_ids, _normalize(embeddings), query_model=query_model)

    def search(self, query: str, k: int = 100) -> list[tuple[str, float]]:
        """Return the top-`k` (nct_id, score) pairs, best first, ties by id."""
        if k <= 0 or not query.strip():
            return []
        encoder = _load_encoder(self.query_model)
        vector = np.asarray(encoder.encode([query], show_progress_bar=False), dtype=np.float32)
        scores = (self.embeddings @ _normalize(vector)[0]).tolist()
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
            pickle.dump(
                {
                    "nct_ids": self.nct_ids,
                    "embeddings": self.embeddings,
                    "query_model": self.query_model,
                },
                fh,
            )

    @classmethod
    def load(cls, path: Path) -> DenseIndex:
        """Load an index previously written by `save`."""
        with Path(path).open("rb") as fh:
            # Only ever load index files this project produced.
            payload = pickle.load(fh)
        return cls(
            payload["nct_ids"],
            payload["embeddings"],
            query_model=payload.get("query_model", DEFAULT_QUERY_MODEL),
        )
