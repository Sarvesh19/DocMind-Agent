"""
main.py — DocMind FastAPI Backend
"""

import logging, json
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="DocMind API", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Initialised in startup — NOT at import time (prevents port-bind timeout on Render)
store     = None
processor = None
agent     = None

@app.on_event("startup")
async def startup_event():
    global store, processor, agent
    logger.info("DocMind starting — loading models...")
    from vector_store import SupabaseVectorStore, DocumentProcessor
    from agent import RAGAgent
    store     = SupabaseVectorStore()
    processor = DocumentProcessor(store)
    agent     = RAGAgent()
    logger.info("DocMind ready ✓")

def require_ready():
    if store is None or agent is None:
        raise HTTPException(503, "Server is still starting up, please retry in a few seconds.")

# ── Pydantic models ────────────────────────────────────────────────────────────
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    session_id: str
    history: List[ChatMessage] = []

class DeleteRequest(BaseModel):
    source_name: str

# ── API routes ─────────────────────────────────────────────────────────────────
@app.get("/api/health")
def health():
    return {"status": "ok" if store else "starting", "version": "2.0.0"}

@app.get("/api/stats")
def stats():
    require_ready()
    try:
        sources = store.list_sources()
        return {"chunks": store.count(), "documents": len(sources), "sources": sources}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/api/upload")
async def upload(files: List[UploadFile] = File(...)):
    require_ready()
    results = []
    for f in files:
        try:
            data   = await f.read()
            chunks = processor.process_bytes(data, f.filename)
            results.append({"filename": f.filename, "chunks": chunks, "status": "ok"})
        except Exception as e:
            results.append({"filename": f.filename, "error": str(e), "status": "error"})
    return {"results": results}

@app.delete("/api/document")
def delete_doc(req: DeleteRequest):
    require_ready()
    return {"deleted": store.delete_source(req.source_name), "source": req.source_name}

@app.post("/api/chat")
def chat(req: ChatRequest):
    require_ready()
    from langchain_core.messages import HumanMessage, AIMessage
    history = [
        HumanMessage(content=m.content) if m.role == "user" else AIMessage(content=m.content)
        for m in req.history[-10:]
    ]
    try:
        response = agent.invoke(req.message, history=history)
        store.save_message(req.session_id, "user", req.message)
        store.save_message(req.session_id, "assistant", response)
        return {"response": response, "session_id": req.session_id}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    require_ready()
    from langchain_core.messages import HumanMessage, AIMessage
    history = [
        HumanMessage(content=m.content) if m.role == "user" else AIMessage(content=m.content)
        for m in req.history[-10:]
    ]

    async def event_generator():
        full_response = ""
        try:
            async for chunk in agent.stream(req.message, history=history):
                if chunk:
                    delta = chunk[len(full_response):]
                    if delta:
                        full_response = chunk
                        yield f"data: {json.dumps({'delta': delta})}\n\n"
            store.save_message(req.session_id, "user", req.message)
            store.save_message(req.session_id, "assistant", full_response)
            yield f"data: {json.dumps({'done': True, 'full': full_response})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

@app.get("/api/history/{session_id}")
def get_history(session_id: str):
    require_ready()
    return {"messages": store.load_history(session_id)}

# ── Static files ───────────────────────────────────────────────────────────────
FRONTEND = Path(__file__).parent / "frontend"
if FRONTEND.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND / "static")), name="static")

    @app.get("/")
    def root():
        return FileResponse(str(FRONTEND / "index.html"))

    @app.get("/{full_path:path}")
    def spa_fallback(full_path: str):
        return FileResponse(str(FRONTEND / "index.html"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=10000, reload=True)
