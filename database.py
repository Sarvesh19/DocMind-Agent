"""Run `python database.py` to print the SQL — then paste it in Supabase SQL Editor."""

SQL = """
-- 1. Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Documents table
CREATE TABLE IF NOT EXISTS documents (
    id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content    TEXT NOT NULL,
    embedding  VECTOR(384),
    metadata   JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS docs_embed_idx ON documents USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- 3. Chat sessions table (persistent memory)
CREATE TABLE IF NOT EXISTS chat_sessions (
    id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id TEXT NOT NULL,
    role       TEXT NOT NULL,
    content    TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS chat_sid_idx ON chat_sessions (session_id, created_at);

-- 4. Similarity search function
CREATE OR REPLACE FUNCTION match_documents(
    query_embedding VECTOR(384),
    match_threshold FLOAT DEFAULT 0.4,
    match_count     INT   DEFAULT 5
)
RETURNS TABLE (id UUID, content TEXT, metadata JSONB, similarity FLOAT)
LANGUAGE SQL STABLE AS $$
    SELECT d.id, d.content, d.metadata,
           1 - (d.embedding <=> query_embedding) AS similarity
    FROM documents d
    WHERE 1 - (d.embedding <=> query_embedding) > match_threshold
    ORDER BY d.embedding <=> query_embedding
    LIMIT match_count;
$$;
"""

if __name__ == "__main__":
    print("\n" + "="*60)
    print("PASTE THIS SQL IN: Supabase → SQL Editor → New Query")
    print("="*60)
    print(SQL)
