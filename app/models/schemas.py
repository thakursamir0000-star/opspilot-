"""
Pydantic request/response models for the API layer.
"""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel


# ── Documents ────────────────────────────────────────────────────────────────

class DocumentInfo(BaseModel):
    """Metadata for a single uploaded document."""
    doc_id: str
    filename: str
    num_pages: int
    num_chunks: int
    status: str  # "ready" | "no_text_extracted" | "error"


class UploadResponse(BaseModel):
    """Response to POST /documents/upload."""
    documents: List[DocumentInfo]


class DocumentListResponse(BaseModel):
    """Response to GET /documents."""
    documents: List[DocumentInfo]


# ── Chat ─────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    """Request body for POST /chat."""
    session_id: str
    message: str


class CitationItem(BaseModel):
    """A single citation pointing back to a source chunk."""
    filename: str
    page: int
    snippet: str


class ChatResponse(BaseModel):
    """Response to POST /chat."""
    answer: str
    citations: List[CitationItem]
