"""
SentenceTransformer embedding wrapper.

Uses all-MiniLM-L6-v2 (384-dim) — free, local, no API cost.
The model is loaded lazily on first call and cached for the process lifetime.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.config import get_settings

# Module-level cache for the model
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """Lazy-load the embedding model (heavy first call, then cached)."""
    global _model
    if _model is None:
        settings = get_settings()
        _model = SentenceTransformer(settings.EMBEDDING_MODEL)
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Embed a list of texts into L2-normalized vectors.

    Returns:
        np.ndarray of shape (len(texts), EMBEDDING_DIM) — float32, unit-normed
        so that inner product == cosine similarity.
    """
    model = _get_model()
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    # Normalize for cosine similarity via IndexFlatIP
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)  # avoid division by zero
    embeddings = embeddings / norms

    return embeddings.astype(np.float32)


def embed_query(query: str) -> np.ndarray:
    """Embed a single query string. Returns shape (1, EMBEDDING_DIM)."""
    return embed_texts([query])
