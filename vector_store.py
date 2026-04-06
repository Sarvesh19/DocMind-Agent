import os
import uuid
import logging
from typing import List, Dict, Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_core.documents import Document
from supabase import create_client, Client

from config import (
    SUPABASE_URL,
    SUPABASE_KEY,
    SUPABASE_TABLE,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    RETRIEVAL_K,
    SIM_THRESHOLD,
)

logger = logging.getLogger(__name__)


class SupabaseVectorStore:
    """Vector store using Supabase pgvector and HuggingFace embeddings."""

    def __init__(self):
        self.supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'}
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""]
        )

    def add_documents(self, documents: List[Document], source_name: str = "") -> int:
        """Split documents, embed them, and store in Supabase."""
        chunks = self.splitter.split_documents(documents)
        if not chunks:
            return 0

        # Batch embed all chunks at once
        texts = [chunk.page_content for chunk in chunks]
        embeddings = self.embeddings.embed_documents(texts)

        rows = []
        for chunk, embedding in zip(chunks, embeddings):
            metadata = {**chunk.metadata}
            metadata["source"] = source_name or metadata.get("source", "unknown")
            rows.append({
                "id": str(uuid.uuid4()),
                "content": chunk.page_content,
                "embedding": embedding,
                "metadata": metadata,
            })

        # Insert in batches to avoid request size limits
        for i in range(0, len(rows), 200):
            self.supabase.table(SUPABASE_TABLE).insert(rows[i:i+200]).execute()

        logger.info("Stored %d chunks for source '%s'", len(rows), source_name)
        return len(rows)

    def similarity_search(
        self, query: str, k: int = RETRIEVAL_K, threshold: float = SIM_THRESHOLD
    ) -> List[Dict]:
        """Return relevant chunks from vector DB."""
        query_embedding = self.embeddings.embed_query(query)
        response = self.supabase.rpc(
            "match_documents",
            {
                "query_embedding": query_embedding,
                "match_threshold": threshold,
                "match_count": k,
            },
        ).execute()
        return response.data or []

    def list_sources(self) -> List[Dict]:
        """Return unique source names with their creation times."""
        response = self.supabase.table(SUPABASE_TABLE).select("metadata, created_at").execute()
        seen = set()
        sources = []
        for row in response.data or []:
            src = (row.get("metadata") or {}).get("source", "")
            if src and src not in seen:
                seen.add(src)
                sources.append({
                    "name": src,
                    "created_at": row.get("created_at", "")
                })
        return sources

    def delete_source(self, source_name: str) -> int:
        """Delete all chunks belonging to a source document."""
        response = (
            self.supabase.table(SUPABASE_TABLE)
            .delete()
            .filter("metadata->>source", "eq", source_name)
            .execute()
        )
        return len(response.data or [])

    def count(self) -> int:
        """Return total number of chunks."""
        response = self.supabase.table(SUPABASE_TABLE).select("id", count="exact").execute()
        return response.count or 0

    def save_message(self, session_id: str, role: str, content: str) -> None:
        """Store a chat message in the conversation history."""
        try:
            self.supabase.table("chat_sessions").insert({
                "session_id": session_id,
                "role": role,
                "content": content,
            }).execute()
        except Exception as e:
            logger.warning("Failed to save message: %s", e)

    def load_history(self, session_id: str, limit: int = 20) -> List[Dict]:
        """Load recent chat messages for a session."""
        try:
            response = (
                self.supabase.table("chat_sessions")
                .select("role, content, created_at")
                .eq("session_id", session_id)
                .order("created_at", desc=False)
                .limit(limit)
                .execute()
            )
            return response.data or []
        except Exception as e:
            logger.warning("Failed to load history: %s", e)
            return []


class DocumentProcessor:
    """Handle file uploads and delegate to vector store."""

    SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx"}

    def __init__(self, store: SupabaseVectorStore):
        self.store = store

    def process_bytes(self, data: bytes, filename: str) -> int:
        """Process a file from bytes and index it."""
        ext = os.path.splitext(filename)[1].lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {ext}")

        # Write to temporary file
        tmp_path = f"/tmp/{uuid.uuid4()}_{filename}"
        with open(tmp_path, "wb") as f:
            f.write(data)

        try:
            # Choose loader based on extension
            if ext == ".pdf":
                loader = PyPDFLoader(tmp_path)
            elif ext == ".docx":
                loader = Docx2txtLoader(tmp_path)
            else:  # .txt, .md
                loader = TextLoader(tmp_path, encoding="utf-8")

            documents = loader.load()
            return self.store.add_documents(documents, source_name=filename)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)