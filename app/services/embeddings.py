"""
fastembed embedding wrapper.

Uses BAAI/bge-small-en-v1.5 (384-dim, ONNX-quantized) — free, local, no API cost.
~4x lighter than sentence-transformers + torch. Fits in 512MB RAM.
The model is loaded lazily on first call and cached for the process lifetime.
"""

from __future__ import annotations

import numpy as np
from fastembed import TextEmbedding

from app.core.config import get_settings

# Module-level cache for the model
_model: TextEmbedding | None = None


def _get_model() -> TextEmbedding:
    """Lazy-load the embedding model (first call downloads ~130MB ONNX, then cached)."""
    global _model
    if _model is None:
        settings = get_settings()
        import time
        last_err = None
        for attempt in range(3):
            try:
                _model = TextEmbedding(settings.EMBEDDING_MODEL)
                return _model
            except Exception as e:
                last_err = e
                print(f"[embeddings] Model load attempt {attempt + 1}/3 failed: {e}")
                if attempt < 2:
                    time.sleep(2)
        raise RuntimeError(
            f"Could not load model {settings.EMBEDDING_MODEL} after 3 attempts: {last_err}"
        )
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Embed a list of texts into L2-normalized vectors.

    Returns:
        np.ndarray of shape (len(texts), EMBEDDING_DIM) — float32, unit-normed
        so that inner product == cosine similarity.
    """
    model = _get_model()
    embeddings = list(model.embed(texts))
    arr = np.array(embeddings, dtype=np.float32)

    # Normalize for cosine similarity via IndexFlatIP
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)  # avoid division by zero
    arr = arr / norms

    return arr


def embed_query(query: str) -> np.ndarray:
    """Embed a single query string. Returns shape (1, EMBEDDING_DIM)."""
    return embed_texts([query])
