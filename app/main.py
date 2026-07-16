"""
FastAPI application entry point.

- Registers document and chat routers.
- Mounts the frontend static files at /.
- Configures CORS for pilot (allow all origins).
- Health check at /health.
"""

from __future__ import annotations

import os
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.routers import documents, chat


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Validate config and attempt to pre-load the embedding model on startup."""
    import logging
    from app.core.config import get_settings
    from app.services.embeddings import embed_texts

    logger = logging.getLogger(__name__)

    settings = get_settings()
    if not settings.GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to .env or your environment variables."
        )

    # Attempt to warm up the embedding model. If the download fails
    # (e.g. HuggingFace 504), let the app start anyway — the model will
    # be downloaded lazily on the first embedding request.
    try:
        embed_texts(["warmup"])
    except Exception:
        logger.warning("Embedding model warmup failed; will retry on first request")

    yield


app = FastAPI(
    title="OpsPilot",
    description="RAG-powered document Q&A for logistics operations",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS (wide open for pilot) ──────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ──────────────────────────────────────────────────────────────────
app.include_router(documents.router)
app.include_router(chat.router)


# ── Health check ─────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/health/debug")
async def health_debug():
    """Diagnostic endpoint — checks that key dependencies are importable and functional."""
    checks = {}

    # Check pypdf
    try:
        from pypdf import PdfReader
        checks["pypdf"] = "ok"
    except Exception as e:
        checks["pypdf"] = f"error: {e}"

    # Check fastembed
    try:
        from fastembed import TextEmbedding
        checks["fastembed_import"] = "ok"
    except Exception as e:
        checks["fastembed_import"] = f"error: {e}"

    # Check fastembed model loading + embedding
    try:
        from app.services.embeddings import embed_texts
        import numpy as np
        result = embed_texts(["test"])
        checks["fastembed_embed"] = f"ok — shape={result.shape}, dtype={result.dtype}"
    except Exception as e:
        checks["fastembed_embed"] = f"error: {e}"

    # Check faiss
    try:
        import faiss
        index = faiss.IndexFlatIP(384)
        checks["faiss"] = f"ok — ntotal={index.ntotal}"
    except Exception as e:
        checks["faiss"] = f"error: {e}"

    return checks



# ── Static frontend ─────────────────────────────────────────────────────────
# Serve index.html at / and other static assets from the frontend directory.
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


@app.get("/")
async def serve_index():
    return FileResponse(FRONTEND_DIR / "index.html")


# Mount static files (css, js, etc.) — this must come after route registration
# so that API routes take precedence over the static file catch-all.
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")
