import os
from dotenv import load_dotenv
load_dotenv()

SUPABASE_URL   = os.getenv("SUPABASE_URL", "https://your-project.supabase.co")
SUPABASE_KEY   = os.getenv("SUPABASE_KEY", "your-anon-key")
SUPABASE_TABLE = os.getenv("SUPABASE_TABLE", "documents")

# ── LLM providers (first one with a key wins, in order: Groq → OpenAI → OpenRouter) ──
GROQ_API_KEY        = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL          = os.getenv("GROQ_MODEL", "llama3-8b-8192")

OPENAI_API_KEY      = os.getenv("OPENAI_API_KEY", "")

OPENROUTER_API_KEY  = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL           = os.getenv("LLM_MODEL", "openrouter/free")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM   = 384

CHUNK_SIZE    = 500
CHUNK_OVERLAP = 50
RETRIEVAL_K   = 5
SIM_THRESHOLD = 0.4
