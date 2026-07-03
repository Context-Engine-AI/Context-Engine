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
VEC_NAME = os.environ.get("VECTOR_NAME") or sanitize_vector_name(MODEL)

client = QdrantClient(url=QDRANT_URL)
if _EMBEDDER_FACTORY:
    emb = get_embedding_model(MODEL)
else:
    emb = TextEmbedding(model_name=MODEL)
q = "function that chunks code lines with overlap for semantic indexing"
vec = next(emb.embed([q]))
res = client.search(
    collection_name=COLLECTION,
    query_vector={"name": VEC_NAME, "vector": vec.tolist()},
    limit=5,
    with_payload=True,
)
for p in res:
    info = (p.payload or {}).get("information")
    md = (p.payload or {}).get("metadata") or {}
    print(
        {
            "score": round(p.score, 4),
            "information": info,
            "path": md.get("path"),
            "language": md.get("language"),
        }
    )
