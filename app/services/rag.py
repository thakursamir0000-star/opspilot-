"""
RAG orchestrator — ties together query rewriting, embedding, retrieval, and answer generation.

Pipeline per query:
1. Rewrite the follow-up question into a standalone question (conversational memory).
2. Embed the standalone question.
3. Search FAISS for the top-k most relevant chunks.
4. Build the prompt with context and generate a grounded answer.
5. Extract citations from the retrieved chunks.
"""

from __future__ import annotations

from typing import List, Generator

from app.core.sessions import SessionData
from app.models.schemas import ChatResponse, CitationItem
from app.services import embeddings, vectorstore, llm


def process_query(session: SessionData, message: str) -> ChatResponse:
    """
    Full RAG pipeline: rewrite → embed → retrieve → answer.

    Args:
        session: The current session's state (index, chunks, history).
        message: The raw user message.

    Returns:
        ChatResponse with answer text and citations.
    """
    # 1. Rewrite the question for standalone meaning
    history = session.get_recent_history()
    standalone_question = llm.rewrite_query(history, message)

    # 2. Embed the standalone question
    query_vector = embeddings.embed_query(standalone_question)

    # 3. Search FAISS
    session.ensure_index()
    results = vectorstore.search(
        index=session.faiss_index,
        query_vector=query_vector,
        chunks=session.chunks,
    )

    # 4. Build context for the LLM
    context_chunks = [
        {
            "text": r.chunk.text,
            "filename": r.chunk.filename,
            "page_number": r.chunk.page_number,
        }
        for r in results
    ]

    # 5. Generate grounded answer (use original message, not rewritten)
    answer = llm.generate_answer(message, context_chunks, history)

    # 6. Build citations from the retrieved chunks
    citations: List[CitationItem] = []
    seen = set()
    for r in results:
        key = (r.chunk.filename, r.chunk.page_number)
        if key not in seen:
            seen.add(key)
            # Use the first ~200 chars of the chunk as the snippet
            snippet = r.chunk.text[:200].strip()
            if len(r.chunk.text) > 200:
                snippet += "…"
            citations.append(CitationItem(
                filename=r.chunk.filename,
                page=r.chunk.page_number,
                snippet=snippet,
            ))

    # 7. Update session history
    session.add_to_history("user", message)
    session.add_to_history("assistant", answer)

    return ChatResponse(answer=answer, citations=citations)


def process_query_stream(session: SessionData, message: str) -> Generator[str, None, None]:
    """
    Streaming RAG pipeline — yields answer tokens as they arrive.
    Citations are sent as a final JSON event after the answer completes.
    """
    import json

    # 1. Rewrite
    history = session.get_recent_history()
    standalone_question = llm.rewrite_query(history, message)

    # 2. Embed
    query_vector = embeddings.embed_query(standalone_question)

    # 3. Search
    session.ensure_index()
    results = vectorstore.search(
        index=session.faiss_index,
        query_vector=query_vector,
        chunks=session.chunks,
    )

    # 4. Context
    context_chunks = [
        {
            "text": r.chunk.text,
            "filename": r.chunk.filename,
            "page_number": r.chunk.page_number,
        }
        for r in results
    ]

    # 5. Stream the answer
    full_answer = []
    for token in llm.generate_answer_stream(message, context_chunks, history):
        full_answer.append(token)
        yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

    # 6. Build citations
    citations = []
    seen = set()
    for r in results:
        key = (r.chunk.filename, r.chunk.page_number)
        if key not in seen:
            seen.add(key)
            snippet = r.chunk.text[:200].strip()
            if len(r.chunk.text) > 200:
                snippet += "…"
            citations.append({
                "filename": r.chunk.filename,
                "page": r.chunk.page_number,
                "snippet": snippet,
            })

    # Send citations as final event
    yield f"data: {json.dumps({'type': 'citations', 'citations': citations})}\n\n"
    yield "data: [DONE]\n\n"

    # 7. Update history
    answer_text = "".join(full_answer)
    session.add_to_history("user", message)
    session.add_to_history("assistant", answer_text)
