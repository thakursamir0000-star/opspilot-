"""
Pre-download the fastembed model during build so it's cached for runtime.

Usage (in Render build command):
    python scripts/download_model.py
"""

from fastembed import TextEmbedding

MODEL_NAME = "BAAI/bge-small-en-v1.5"

print(f"Downloading and caching embedding model: {MODEL_NAME}")
model = TextEmbedding(MODEL_NAME)

# Verify it works
result = list(model.embed(["test"]))
print(f"Model loaded OK — embedding dim: {len(result[0])}")
