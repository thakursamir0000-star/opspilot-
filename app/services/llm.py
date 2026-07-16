"""
Groq LLM client — query rewriting for conversational memory and grounded answer generation.

Uses the Groq Python SDK with Llama 3.3 70B (configurable).
"""

from __future__ import annotations

from typing import Dict, List

from groq import Groq

from app.core.config import get_settings

# Module-level client (lazy)
_client: Groq | None = None


def _get_client() -> Groq:
    """Lazy-initialise the Groq client."""
    global _client
    if _client is None:
        settings = get_settings()
        _client = Groq(api_key=settings.GROQ_API_KEY)
    return _client


# ── Prompts ──────────────────────────────────────────────────────────────────

REWRITE_SYSTEM = """You are a query rewriting assistant. Given a conversation history and a follow-up question, rewrite the follow-up question as a standalone question that captures the full meaning without needing any prior context.

Rules:
- Output ONLY the rewritten question, nothing else.
- If the question is already standalone, return it as-is.
- Do not add any explanation or preamble."""

ANSWER_SYSTEM = """You are OpsPilot, an AI assistant that answers questions about uploaded documents for a logistics operations team.

Rules:
1. Answer ONLY based on the provided document context below. Do NOT use any outside knowledge.
2. If the answer is not contained in the provided context, say clearly: "I don't have enough information in the uploaded documents to answer that question."
3. When you reference information, mention the source document name and page number naturally in your answer (e.g., "According to rate_card.pdf, page 12, ...").
4. Be concise and direct. Use bullet points for lists.
5. If the context contains contradictory information from different documents, mention both and note the discrepancy.
6. Maintain a professional, helpful tone appropriate for an operations team."""


# ── Core functions ───────────────────────────────────────────────────────────

def rewrite_query(history: List[Dict[str, str]], new_question: str) -> str:
    """
    Rewrite a follow-up question into a standalone question using conversation context.

    This is what makes "and who does it apply to?" resolve to
    "who does the penalty clause apply to?" when the previous question
    was about penalty clauses.
    """
    if not history:
        return new_question

    client = _get_client()
    settings = get_settings()

    # Build the conversation context for the rewriter
    messages = [{"role": "system", "content": REWRITE_SYSTEM}]

    # Include the last ~4 turns of history for context
    recent = history[-4:]
    for msg in recent:
        messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({
        "role": "user",
        "content": f"Rewrite this follow-up question as a standalone question:\n\n{new_question}"
    })

    try:
        response = client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=messages,
            temperature=0.0,
            max_tokens=256,
        )
        rewritten = response.choices[0].message.content.strip()
        return rewritten if rewritten else new_question
    except Exception:
        # Fallback: use the original question if rewriting fails
        return new_question


def generate_answer(
    question: str,
    context_chunks: List[Dict[str, str]],
    history: List[Dict[str, str]],
) -> str:
    """
    Generate a grounded answer using the retrieved context chunks and conversation history.

    Args:
        question: The original user question (not the rewritten one).
        context_chunks: List of dicts with keys: text, filename, page_number.
        history: Recent chat history for conversational continuity.
    """
    client = _get_client()
    settings = get_settings()

    # Build context block from retrieved chunks
    if not context_chunks:
        context_block = "(No relevant context was found in the uploaded documents.)"
    else:
        context_parts = []
        for i, chunk in enumerate(context_chunks, 1):
            context_parts.append(
                f"[Source {i}: {chunk['filename']}, Page {chunk['page_number']}]\n{chunk['text']}"
            )
        context_block = "\n\n---\n\n".join(context_parts)

    # Assemble messages
    messages = [{"role": "system", "content": ANSWER_SYSTEM}]

    # Add recent history for conversational continuity (last ~6 messages)
    recent = history[-6:]
    for msg in recent:
        messages.append({"role": msg["role"], "content": msg["content"]})

    # The user's question with context
    user_prompt = f"""Document Context:
{context_block}

Question: {question}"""

    messages.append({"role": "user", "content": user_prompt})

    response = client.chat.completions.create(
        model=settings.GROQ_MODEL,
        messages=messages,
        temperature=0.3,
        max_tokens=2048,
    )

    return response.choices[0].message.content.strip()


def generate_answer_stream(
    question: str,
    context_chunks: List[Dict[str, str]],
    history: List[Dict[str, str]],
):
    """
    Streaming version of generate_answer — yields content tokens as they arrive.
    Used by the /chat/stream endpoint (stretch goal).
    """
    client = _get_client()
    settings = get_settings()

    if not context_chunks:
        context_block = "(No relevant context was found in the uploaded documents.)"
    else:
        context_parts = []
        for i, chunk in enumerate(context_chunks, 1):
            context_parts.append(
                f"[Source {i}: {chunk['filename']}, Page {chunk['page_number']}]\n{chunk['text']}"
            )
        context_block = "\n\n---\n\n".join(context_parts)

    messages = [{"role": "system", "content": ANSWER_SYSTEM}]

    recent = history[-6:]
    for msg in recent:
        messages.append({"role": msg["role"], "content": msg["content"]})

    user_prompt = f"""Document Context:
{context_block}

Question: {question}"""

    messages.append({"role": "user", "content": user_prompt})

    stream = client.chat.completions.create(
        model=settings.GROQ_MODEL,
        messages=messages,
        temperature=0.3,
        max_tokens=2048,
        stream=True,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            yield delta.content
