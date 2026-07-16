"""
Chat endpoints — synchronous and streaming.

POST /chat        — full answer + citations in one response
POST /chat/stream — SSE token-by-token streaming (stretch goal)
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.core.sessions import session_store
from app.models.schemas import ChatRequest, ChatResponse
from app.services.rag import process_query, process_query_stream

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Send a message and get a grounded answer with citations.

    Returns 400 if no documents have been loaded for the session.
    """
    session_id = request.session_id
    message = request.message.strip()

    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    if not session_store.has_documents(session_id):
        raise HTTPException(
            status_code=400,
            detail="No documents loaded for this session. Upload documents first."
        )

    session = session_store.get_or_create(session_id)

    try:
        response = process_query(session, message)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@router.post("/stream")
async def chat_stream(request: ChatRequest):
    """
    Streaming chat — returns Server-Sent Events with tokens as they arrive.

    Events:
    - {"type": "token", "content": "..."} — individual tokens
    - {"type": "citations", "citations": [...]} — final citation list
    - [DONE] — stream end marker
    """
    session_id = request.session_id
    message = request.message.strip()

    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    if not session_store.has_documents(session_id):
        raise HTTPException(
            status_code=400,
            detail="No documents loaded for this session. Upload documents first."
        )

    session = session_store.get_or_create(session_id)

    return StreamingResponse(
        process_query_stream(session, message),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
