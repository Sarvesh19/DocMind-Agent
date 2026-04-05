import os, uuid, logging
from typing import List, Dict, Any
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from supabase import create_client, Client
from config import *

logger = logging.getLogger(__name__)

class SupabaseVectorStore:
    def __init__(self):
        self.supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def add_documents(self, documents: List[Document], source_name: str = "") -> int:
        chunks = self.splitter.split_documents(documents)
        rows = []
        for chunk in chunks:
            embedding = self.embeddings.embed_query(chunk.page_content)
            rows.append({
                "id": str(uuid.uuid4()),
                "content": chunk.page_content,
                "embedding": embedding,
                "metadata": {**chunk.metadata, "source": source_name or chunk.metadata.get("source", "unknown")},
            })
        for i in range(0, len(rows), 200):
            self.supabase.table(SUPABASE_TABLE).insert(rows[i:i+200]).execute()
        return len(rows)

    def similarity_search(self, query: str, k: int = RETRIEVAL_K, threshold: float = SIM_THRESHOLD) -> List[Dict]:
        vec = self.embeddings.embed_query(query)
        resp = self.supabase.rpc("match_documents", {
            "query_embedding": vec, "match_threshold": threshold, "match_count": k
        }).execute()
        return resp.data or []

    def list_sources(self) -> List[Dict]:
        resp = self.supabase.table(SUPABASE_TABLE).select("metadata, created_at").execute()
        seen, sources = set(), []
        for row in (resp.data or []):
            src = (row.get("metadata") or {}).get("source", "")
            if src and src not in seen:
                seen.add(src)
                sources.append({"name": src, "created_at": row.get("created_at", "")})
        return sources

    def delete_source(self, source_name: str) -> int:
        resp = (self.supabase.table(SUPABASE_TABLE)
            .delete().filter("metadata->>source", "eq", source_name).execute())
        return len(resp.data or [])

    def count(self) -> int:
        resp = self.supabase.table(SUPABASE_TABLE).select("id", count="exact").execute()
        return resp.count or 0

    def save_message(self, session_id: str, role: str, content: str):
        try:
            self.supabase.table("chat_sessions").insert({
                "session_id": session_id, "role": role, "content": content
            }).execute()
        except Exception:
            pass

    def load_history(self, session_id: str, limit: int = 20) -> List[Dict]:
        try:
            resp = (self.supabase.table("chat_sessions")
                .select("role,content,created_at")
                .eq("session_id", session_id)
                .order("created_at", desc=False).limit(limit).execute())
            return resp.data or []
        except Exception:
            return []


class DocumentProcessor:
    SUPPORTED = {".pdf", ".txt", ".md", ".docx"}

    def __init__(self, store: SupabaseVectorStore):
        self.store = store

    def process_bytes(self, data: bytes, filename: str) -> int:
        ext = os.path.splitext(filename)[1].lower()
        if ext not in self.SUPPORTED:
            raise ValueError(f"Unsupported file type: {ext}")
        tmp = f"/tmp/{uuid.uuid4()}_{filename}"
        with open(tmp, "wb") as f:
            f.write(data)
        try:
            if ext == ".pdf":
                from langchain_community.document_loaders import PyPDFLoader
                loader = PyPDFLoader(tmp)
            elif ext == ".docx":
                from langchain_community.document_loaders import Docx2txtLoader
                loader = Docx2txtLoader(tmp)
            else:
                loader = TextLoader(tmp, encoding="utf-8")
            docs = loader.load()
            return self.store.add_documents(docs, source_name=filename)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
