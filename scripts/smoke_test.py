#!/usr/bin/env python3
import os
import sys
from pathlib import Path
from qdrant_client import QdrantClient

# Ensure scripts is importable
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Use embedder factory for Qwen3 support
try:
    from scripts.embedder import get_embedding_model
    _EMBEDDER_FACTORY = True
except ImportError:
    _EMBEDDER_FACTORY = False
    from fastembed import TextEmbedding
from scripts.utils import sanitize_vector_name

QDRANT_URL = os.environ.get("QDRANT_URL", "http://qdrant:6333")
COLLECTION = os.environ.get("COLLECTION_NAME", "codebase")
MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")

# Derive the named vector consistently with ingest_code
vn = os.environ.get("VECTOR_NAME")
if not vn:
    vn = sanitize_vector_name(MODEL)
VEC_NAME = vn

client = QdrantClient(url=QDRANT_URL)

# Count points
try:
    count = client.count(COLLECTION, exact=True).count
except Exception:
    count = None

# Prepare query embedding
if _EMBEDDER_FACTORY:
    model = get_embedding_model(MODEL)
else:
    model = TextEmbedding(model_name=MODEL)
query = "python code indexer for qdrant"
vec = next(model.embed([query]))

# Search top 3
res = client.search(
    collection_name=COLLECTION,
    query_vector={"name": VEC_NAME, "vector": vec.tolist()},
    limit=3,
    with_payload=True,
)

print("Smoke test:")
print(f"- Qdrant URL: {QDRANT_URL}")
print(f"- Collection: {COLLECTION}")
print(f"- Embedding model: {MODEL}")
print(f"- Vector name: {VEC_NAME}")
print(f"- Count (exact): {count}")
print("Top hits:")
for p in res:
    md = (p.payload or {}).get("metadata", {})
    print(
        f"  score={p.score:.4f} path={md.get('path')} lines={md.get('start_line')}-{md.get('end_line')}"
    )
