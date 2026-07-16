"""
Document upload and listing endpoints.

POST /documents/upload — multipart form: session_id + PDF files
GET  /documents       — list documents for a session
"""

from __future__ import annotations

import logging
import uuid
from typing import List

from fastapi import APIRouter, File, Form, UploadFile, HTTPException

from app.core.sessions import session_store, ChunkRecord, DocumentRecord
from app.models.schemas import DocumentInfo, UploadResponse, DocumentListResponse
from app.services.ingestion import process_pdf
from app.services.embeddings import embed_texts
from app.services import vectorstore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload", response_model=UploadResponse)
async def upload_documents(
    session_id: str = Form(...),
    files: List[UploadFile] = File(...),
):
    """
    Upload one or more PDF files for a session.

    Each file is: extracted → chunked → embedded → added to the session's FAISS index.
    Non-PDF files are rejected with a 400.
    """
    session = session_store.get_or_create(session_id)
    session.ensure_index()

    results: List[DocumentInfo] = []

    for upload_file in files:
        # Validate file type
        filename = upload_file.filename or "unknown.pdf"
        if not filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail=f"Only PDF files are accepted. Got: {filename}"
            )

        # Read file contents
        file_bytes = await upload_file.read()

        # Stage 1: Extract and chunk (captures page count even if embedding fails)
        total_pages = 0
        chunks = []
        status = "error"
        doc_id = str(uuid.uuid4())

        try:
            chunks, total_pages, status = process_pdf(file_bytes, filename)
            doc_id = chunks[0].doc_id if chunks else doc_id
            logger.info("PDF %s: extraction OK — %d pages, %d chunks, status=%s",
                        filename, total_pages, len(chunks), status)
        except Exception as e:
            logger.exception("Failed to extract text from %s", filename)
            status = f"error: extraction failed — {str(e)[:150]}"

        # Stage 2: Embed and index (only if extraction produced chunks)
        if chunks and not status.startswith("error"):
            try:
                texts = [c.text for c in chunks]
                vectors = embed_texts(texts)

                chunk_records = [
                    ChunkRecord(
                        text=c.text,
                        filename=c.filename,
                        page_number=c.page_number,
                        doc_id=doc_id,
                    )
                    for c in chunks
                ]
                session.chunks.extend(chunk_records)

                vectorstore.add_vectors(session.faiss_index, vectors, chunk_records)
                logger.info("PDF %s: embedding + indexing OK (%d vectors)",
                            filename, len(chunk_records))
            except Exception as e:
                logger.exception("Failed to embed/index %s", filename)
                status = f"error: embedding failed — {str(e)[:150]}"
                chunks = []  # clear chunks since they weren't indexed

        # Record the document (always, even on error)
        doc_record = DocumentRecord(
            doc_id=doc_id,
            filename=filename,
            num_pages=total_pages,
            num_chunks=len(chunks),
            status=status,
        )
        session.documents.append(doc_record)

        results.append(DocumentInfo(
            doc_id=doc_id,
            filename=filename,
            num_pages=total_pages,
            num_chunks=len(chunks),
            status=status,
        ))

    return UploadResponse(documents=results)


@router.get("", response_model=DocumentListResponse)
async def list_documents(session_id: str):
    """Return all documents currently loaded for a session (for the sidebar)."""
    session = session_store.get(session_id)
    if session is None:
        return DocumentListResponse(documents=[])

    docs = [
        DocumentInfo(
            doc_id=d.doc_id,
            filename=d.filename,
            num_pages=d.num_pages,
            num_chunks=d.num_chunks,
            status=d.status,
        )
        for d in session.documents
    ]
    return DocumentListResponse(documents=docs)
