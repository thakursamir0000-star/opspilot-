"""
PDF text extraction and word-count-based chunking.

Strategy:
- pypdf per-page extraction, keeping page numbers for citations.
- Sliding window: ~700 words per chunk, ~120-word overlap.
- Word-based (not token-based) — deliberate simplicity for the pilot.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import List, Tuple

# pyrefly: ignore [missing-import]
from pypdf import PdfReader
import io

from app.core.config import get_settings


@dataclass
class PageText:
    """Raw extracted text from a single PDF page."""
    page_number: int  # 1-indexed
    text: str


@dataclass
class Chunk:
    """A text chunk with source provenance."""
    text: str
    filename: str
    page_number: int  # primary page this chunk came from
    doc_id: str


def extract_text_from_pdf(file_bytes: bytes, filename: str) -> Tuple[List[PageText], int]:
    """
    Extract text from every page of a PDF.

    Returns:
        (list of PageText, total_page_count)
    """
    reader = PdfReader(io.BytesIO(file_bytes))
    pages: List[PageText] = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            pages.append(PageText(page_number=i + 1, text=text))

    return pages, len(reader.pages)


def chunk_pages(
    pages: List[PageText],
    filename: str,
    doc_id: str,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> List[Chunk]:
    """
    Split extracted pages into overlapping word-count-based chunks.

    Each chunk retains the page number of the page where it starts,
    which is used for citation purposes.
    """
    settings = get_settings()
    chunk_size = chunk_size or settings.CHUNK_SIZE_WORDS
    overlap = overlap or settings.CHUNK_OVERLAP_WORDS

    # Build a flat list of (word, page_number) tuples
    word_page_pairs: List[Tuple[str, int]] = []
    for page in pages:
        words = page.text.split()
        for w in words:
            word_page_pairs.append((w, page.page_number))

    if not word_page_pairs:
        return []

    chunks: List[Chunk] = []
    start = 0

    while start < len(word_page_pairs):
        end = min(start + chunk_size, len(word_page_pairs))
        window = word_page_pairs[start:end]

        chunk_text = " ".join(w for w, _ in window)
        # The chunk's page number is the page where the chunk starts
        chunk_page = window[0][1]

        chunks.append(Chunk(
            text=chunk_text,
            filename=filename,
            page_number=chunk_page,
            doc_id=doc_id,
        ))

        # Advance by (chunk_size - overlap) words
        step = chunk_size - overlap
        if step <= 0:
            step = chunk_size  # safety: avoid infinite loop
        start += step

    return chunks


def process_pdf(file_bytes: bytes, filename: str) -> Tuple[List[Chunk], int, str]:
    """
    Full pipeline: extract → chunk.

    Returns:
        (chunks, total_page_count, status)
        status is "ready" or "no_text_extracted"
    """
    doc_id = str(uuid.uuid4())
    pages, total_pages = extract_text_from_pdf(file_bytes, filename)

    if not pages:
        return [], total_pages, "no_text_extracted"

    chunks = chunk_pages(pages, filename, doc_id)
    return chunks, total_pages, "ready"
