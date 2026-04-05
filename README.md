# 🧠 DocMind — Document Intelligence Agent

A production-grade RAG agent with a **custom FastAPI backend** and **polished dark UI** (no Streamlit). Upload documents, ask questions, get cited answers with streaming responses.

---

## Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI + Uvicorn |
| Agent | LangGraph (multi-step reasoning) |
| Vector DB | Supabase + pgvector |
| Embeddings | sentence-transformers (local, free) |
| LLM | OpenRouter (free) or OpenAI |
| Frontend | Vanilla HTML/CSS/JS (zero frameworks) |
| Deploy | Render (free tier) |

---

## Quick Start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Edit .env with your keys

# 3. Setup Supabase schema (once)
python database.py
# Copy the printed SQL → paste in Supabase SQL Editor → Run

# 4. Start
uvicorn main:app --reload

# Open http://localhost:8000
```

---

## Project Structure

```
docmind/
├── main.py           # FastAPI app + API routes + static file serving
├── agent.py          # LangGraph agent with 4 tools
├── vector_store.py   # Supabase vector store + document processor
├── config.py         # All configuration
├── database.py       # Supabase schema SQL (run once)
├── requirements.txt
├── render.yaml       # One-click Render deploy
├── .env.example
└── frontend/
    ├── index.html    # App shell
    └── static/
        ├── css/main.css   # Full dark UI styles
        └── js/app.js      # All frontend logic
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/api/health` | Health check |
| GET | `/api/stats` | Doc count, chunk count, sources |
| POST | `/api/upload` | Upload files (multipart) |
| DELETE | `/api/document` | Delete a source by name |
| POST | `/api/chat` | Chat (non-streaming) |
| POST | `/api/chat/stream` | Chat with SSE streaming |
| GET | `/api/history/{id}` | Load chat history |

---

## Deploy to Render

1. Push to GitHub
2. Render → New Web Service → connect repo
3. `render.yaml` is auto-detected
4. Add env vars: `SUPABASE_URL`, `SUPABASE_KEY`, `OPENROUTER_API_KEY`
5. Deploy ✅

---

## Customisation

- **Add tools**: New `@tool` functions in `agent.py`
- **Change LLM**: Set `LLM_MODEL` in `.env` to any [OpenRouter model](https://openrouter.ai/models)
- **Tune RAG**: `CHUNK_SIZE`, `RETRIEVAL_K`, `SIM_THRESHOLD` in `config.py`
- **Restyle UI**: Edit `frontend/static/css/main.css` — all CSS variables at the top of `:root`
