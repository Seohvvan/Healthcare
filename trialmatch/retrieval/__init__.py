"""Retrieval stage: BM25, optional dense embeddings, and rank fusion.

`DenseIndex` is imported lazily-safe: the module never imports
sentence-transformers at import time, so this package stays usable with the
base dependency set (see design.md §2 — dense is an optional extra).
"""

from __future__ import annotations

from trialmatch.retrieval.bm25 import BM25Index, tokenize
from trialmatch.retrieval.dense import (
    DEFAULT_DOC_MODEL,
    DEFAULT_QUERY_MODEL,
    DenseIndex,
)
from trialmatch.retrieval.fusion import rrf

__all__ = [
    "DEFAULT_DOC_MODEL",
    "DEFAULT_QUERY_MODEL",
    "BM25Index",
    "DenseIndex",
    "rrf",
    "tokenize",
]
