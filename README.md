# OpsPilot

**RAG-powered document Q&A for logistics operations teams.**

Upload operational PDFs (rate cards, SOPs, vendor contracts), ask questions in natural language, and get grounded answers with page-level citations. Built as a pilot to demonstrate intelligent document querying for internal ops workflows.

---

## Architecture

```
┌────────────────────────────────┐
│      Browser (Frontend)        │
│   index.html / app.js / css    │
│   • Upload PDFs                │
│   • Chat panel                 │
│   • Doc list sidebar           │
│   • Loading/error states       │
└──────────────┬─────────────────┘
               │ REST (fetch)
               ▼
┌────────────────────────────────┐
│       FastAPI Backend          │
│                                │
│  routers/documents.py ──▶ services/ingestion.py (PDF → text → chunks)
│  routers/chat.py      ──▶ services/rag.py (orchestration)
│                                │
│  core/sessions.py              │  ├─▶ services/embeddings.py (SentenceTransformer)
│  (per-session state:           │  ├─▶ services/vectorstore.py (FAISS index)
│   history + FAISS index)       │  └─▶ services/llm.py (Groq client)
└──────────────┬─────────────────┘
               │
               ▼
     Groq API (Llama 3.3 70B)
```

**Single service.** FastAPI serves both the JSON API and the static frontend, so Render only needs one web service — one URL.

**Session model.** Each browser tab gets a `session_id` (generated client-side with `crypto.randomUUID()`, stored in `localStorage`). The backend keeps an in-memory dict: `session_id → {faiss_index, chunk_records, chat_history}`.

---

## Setup

### Prerequisites

- Python 3.10+
- A [Groq API key](https://console.groq.com) (free tier works)

### Local Development

```bash
# Clone the repo
git clone <your-repo-url>
cd opspilot

# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your GROQ_API_KEY

# Start the server
uvicorn app.main:app --reload --port 8000
```

Open http://localhost:8000 in your browser.

> **Note:** The first startup downloads the embedding model (~90MB). Subsequent starts are instant.

### Deployment on Render (Native Python)

1. Push to GitHub.
2. Render dashboard → New → Web Service → connect the repo.
3. **Build command:** `pip install -r requirements.txt`
4. **Start command:** `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
5. Add `GROQ_API_KEY` as an environment variable in the Render dashboard.
6. Deploy on the free tier.

> The `render.yaml` file auto-configures these settings if Render detects it.

> **Cold starts:** Free tier spins down after ~15 min idle. First request after inactivity takes ~30-60s.

---

## Chunking & Retrieval Strategy

### Why these choices?

| Decision | Choice | Rationale |
|---|---|---|
| **PDF extraction** | `pypdf` per-page | Keeps page numbers attached to every chunk — makes citations possible. No OCR (see limitations). |
| **Chunking** | Word-count sliding window, ~700 words, ~120-word overlap | ~500-600 tokens per chunk. Overlap preserves context across chunk boundaries (e.g. a clause split mid-sentence). Word-based instead of tokenizer-based is a deliberate simplicity trade-off. |
| **Embedding** | `all-MiniLM-L6-v2` (384-dim) | Free, runs locally, no API cost/latency. Good enough quality for a pilot corpus of a few hundred pages. |
| **Vector store** | FAISS `IndexFlatIP` per session | Cosine similarity via normalized vectors. In-memory, per-session. No persistence needed — each session re-uploads. |
| **Retrieval** | Top-k = 5, cosine threshold ≥ 0.25 | Top-k is configurable. The similarity floor prevents force-feeding irrelevant context on off-topic questions. |
| **Grounding** | System prompt + low-similarity filtering | Model is instructed to answer only from context and say "I don't know" when info isn't present. Chunks below 0.25 cosine are dropped before reaching the prompt. |

### Conversational Memory

Two-step approach per turn:
1. **Query rewrite:** Chat history (last ~4 turns) + new question → Groq rewrites it as a standalone question. This is what gets embedded and searched ("and who does it apply to?" → "who does the penalty clause apply to?").
2. **Answer generation:** The *original* user question + retrieved context + recent history → main answer call. Response stays natural and conversational.

### What I'd improve with `tiktoken`

The current word-based chunking is a simplicity trade-off. A `tiktoken`-based token chunking approach would give more precise control over prompt budgets and ensure chunks align better with the model's tokenization. This is a "next week" improvement.

---

## Known Limitations

- **In-memory sessions:** All state (FAISS index, chat history, documents) is lost on server restart. Acceptable for a pilot — production would use a persistent vector store (Qdrant, pgvector).
- **No OCR:** Scanned/image PDFs yield no text. The app surfaces this as a warning ("no text extracted") rather than silently failing. An OCR fallback (e.g. Tesseract) would fix this.
- **Single-node FAISS:** Not persisted, not distributed. Fine for a pilot corpus of a few hundred pages.
- **Free-tier cold starts:** Render free tier spins down after ~15 min. First request after inactivity is slow (~30-60s).
- **No auth:** Any session_id is accepted. Not suitable for multi-tenant production use.
- **Context window:** Only the top-5 chunks are passed to the LLM. Very long or complex documents may need more sophisticated retrieval (hybrid search, re-ranking).

---

## What I'd Build Next (One More Week)

1. **Persistent vector store** — Replace in-memory FAISS with Qdrant or pgvector for session persistence across restarts and multi-tenant support.
2. **Hybrid BM25 + dense retrieval** — Add a BM25 pass (e.g. `rank_bm25`) and combine with dense FAISS results via reciprocal rank fusion. Especially useful for exact keyword matches in contracts.
3. **OCR fallback** — Detect scanned PDFs and run Tesseract before chunking.
4. **Auth & multi-tenancy** — Session-based auth, user isolation, document-level access control.
5. **Evaluation harness** — Automated answer quality testing against a ground-truth Q&A set to measure retrieval precision and answer accuracy before/after changes.
6. **Re-ranking** — Add a cross-encoder re-ranker after initial FAISS retrieval to improve precision.
7. **Streaming** — Token-by-token response streaming for a more responsive UI (partially implemented in `/chat/stream`).

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/documents/upload` | Upload PDFs (multipart form) |
| `GET` | `/documents?session_id=...` | List documents for a session |
| `POST` | `/chat` | Send a message, get answer + citations |
| `POST` | `/chat/stream` | Streaming chat (SSE) |
| `GET` | `/health` | Health check |

---

## Tech Stack

- **Backend:** FastAPI, Python 3.10+
- **LLM:** Groq (Llama 3.3 70B Versatile)
- **Embeddings:** SentenceTransformers (all-MiniLM-L6-v2)
- **Vector Store:** FAISS (IndexFlatIP, in-memory)
- **PDF Extraction:** pypdf
- **Frontend:** Vanilla HTML/CSS/JS
- **Deployment:** Render (native Python web service)
