"""
preload_model.py
================
Run during BUILD so the fastembed model is cached before server starts.
This prevents cold-start delays on first upload/query.
"""
import sys
print("Preloading fastembed model during build...", flush=True)
try:
    from fastembed import TextEmbedding
    model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    # Run a dummy encode to fully download and cache
    list(model.embed(["preload test"]))
    print("✓ Fastembed model cached successfully.", flush=True)
except Exception as e:
    print(f"Warning: model preload failed ({e}). Will download on first request.", flush=True)
    sys.exit(0)
