"""
FAISS vector store operations.

Uses IndexFlatIP (inner product on normalized vectors = cosine similarity).
One index per session, in-memory, not persisted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

# pyrefly: ignore [missing-import]
import faiss
import numpy as np

from app.core.config import get_settings
from app.core.sessions import ChunkRecord


@dataclass
class SearchResult:
    """A single retrieval result with its similarity score."""
    chunk: ChunkRecord
    score: float


def create_index() -> faiss.IndexFlatIP:
    """Create a fresh FAISS inner-product index."""
    settings = get_settings()
    return faiss.IndexFlatIP(settings.EMBEDDING_DIM)


def add_vectors(
    index: faiss.IndexFlatIP,
    vectors: np.ndarray,
    chunks: List[ChunkRecord],
) -> None:
    """
    Add vectors to the FAISS index.

    The caller is responsible for keeping `chunks` aligned with the index
    (i.e. the i-th vector corresponds to chunks[i]).
    """
    if vectors.shape[0] == 0:
        return
    index.add(vectors)


def search(
    index: faiss.IndexFlatIP,
    query_vector: np.ndarray,
    chunks: List[ChunkRecord],
    top_k: int | None = None,
    threshold: float | None = None,
) -> List[SearchResult]:
    """
    Search the FAISS index and return top-k results above the similarity threshold.

    Args:
        index: The FAISS index to search.
        query_vector: Shape (1, dim) — the embedded query.
        chunks: The full list of ChunkRecords aligned with the index.
        top_k: Number of results to return (defaults to config TOP_K).
        threshold: Minimum cosine similarity (defaults to config SIMILARITY_THRESHOLD).
    """
    settings = get_settings()
    top_k = top_k or settings.TOP_K
    threshold = threshold if threshold is not None else settings.SIMILARITY_THRESHOLD

    if index.ntotal == 0:
        return []

    # Clamp top_k to the number of indexed vectors
    effective_k = min(top_k, index.ntotal)

    scores, indices = index.search(query_vector, effective_k)

    results: List[SearchResult] = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue  # FAISS returns -1 for missing results
        if score < threshold:
            continue  # Below similarity floor
        results.append(SearchResult(chunk=chunks[idx], score=float(score)))

    return results
