import os
from dotenv import load_dotenv

load_dotenv()

# Supabase (free tier)
SUPABASE_URL   = os.getenv("SUPABASE_URL")
SUPABASE_KEY   = os.getenv("SUPABASE_KEY")
SUPABASE_TABLE = os.getenv("SUPABASE_TABLE", "documents")

# Groq (free tier)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama-3-groq-70b-tool-use")  # or "llama3-70b-8192"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Embeddings (local, free)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM   = 384

# RAG settings
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 50
RETRIEVAL_K   = 5
SIM_THRESHOLD = 0.4

# Optional fallback (if you ever need OpenAI)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")