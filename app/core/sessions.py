"""
In-memory session store.

Each browser tab gets a unique session_id. The backend keeps per-session state:
FAISS index, chunk records (text + metadata), chat history, and document list.

Known limitation: all state is lost on server restart. This is acceptable for a
pilot — called out explicitly in the README.
"""

from __future__ import annotations

import faiss
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Any

from app.core.config import get_settings


@dataclass
class ChunkRecord:
    """A single text chunk with provenance metadata."""
    text: str
    filename: str
    page_number: int
    doc_id: str


@dataclass
class DocumentRecord:
    """Metadata about an uploaded document."""
    doc_id: str
    filename: str
    num_pages: int
    num_chunks: int
    status: str  # "ready" | "no_text_extracted" | "error"


@dataclass
class SessionData:
    """Everything the backend remembers about one browser session."""
    faiss_index: Any = None                          # faiss.IndexFlatIP (created lazily)
    chunks: List[ChunkRecord] = field(default_factory=list)
    documents: List[DocumentRecord] = field(default_factory=list)
    chat_history: List[Dict[str, str]] = field(default_factory=list)  # [{role, content}, ...]

    def ensure_index(self) -> None:
        """Create the FAISS index if it doesn't exist yet."""
        if self.faiss_index is None:
            settings = get_settings()
            self.faiss_index = faiss.IndexFlatIP(settings.EMBEDDING_DIM)

    def add_to_history(self, role: str, content: str) -> None:
        """Append a message and trim to the configured window."""
        self.chat_history.append({"role": role, "content": content})
        max_msgs = get_settings().MAX_HISTORY_MESSAGES
        if len(self.chat_history) > max_msgs:
            self.chat_history = self.chat_history[-max_msgs:]

    def get_recent_history(self, n: int | None = None) -> List[Dict[str, str]]:
        """Return the last n messages (defaults to full window)."""
        if n is None:
            return list(self.chat_history)
        return list(self.chat_history[-n:])


class SessionStore:
    """Thread-safe-ish in-memory dict of session_id → SessionData."""

    def __init__(self) -> None:
        self._sessions: Dict[str, SessionData] = {}

    def get_or_create(self, session_id: str) -> SessionData:
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionData()
        return self._sessions[session_id]

    def get(self, session_id: str) -> SessionData | None:
        return self._sessions.get(session_id)

    def has_documents(self, session_id: str) -> bool:
        session = self._sessions.get(session_id)
        if session is None:
            return False
        return any(d.status == "ready" for d in session.documents)


# Global singleton
session_store = SessionStore()
