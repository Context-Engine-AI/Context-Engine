#!/usr/bin/env python3
import os
import argparse
import sys
from pathlib import Path
from qdrant_client import QdrantClient, models

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.utils import sanitize_vector_name

# Warm start: load embedding model and warm Qdrant HNSW search path with a small query
# Useful to reduce first-query latency and set a higher runtime ef for quality


def derive_vector_name(model_name: str) -> str:
    return sanitize_vector_name(model_name)


def get_embedding_model(model_name: str):
    """Get embedding model with Qwen3 support via embedder factory."""
    try:
        from scripts.embedder import get_embedding_model as _get_model
        return _get_model(model_name)
    except ImportError:
        pass
    # Fallback to direct fastembed
    from fastembed import TextEmbedding
    return TextEmbedding(model_name=model_name)


def main():
    parser = argparse.ArgumentParser(description="Warm start embeddings + Qdrant HNSW")
    parser.add_argument(
        "--query",
        "-q",
        default="warm start probe",
        help="Probe text to embed and search",
    )
    parser.add_argument(
        "--ef", type=int, default=256, help="HNSW ef (search) to warm caches"
    )
    parser.add_argument(
        "--limit", type=int, default=3, help="Number of points to request"
    )
    args = parser.parse_args()

    QDRANT_URL = os.environ.get("QDRANT_URL", "http://qdrant:6333")
    COLLECTION = os.environ.get("COLLECTION_NAME", "codebase")
    MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")

    print(
        f"Warm start: qdrant={QDRANT_URL} collection={COLLECTION} model={MODEL} ef={args.ef}"
    )

    client = QdrantClient(url=QDRANT_URL)
    model = get_embedding_model(MODEL)
    vec_name = derive_vector_name(MODEL)

    # Trigger model download/init
    vec = next(model.embed([args.query])).tolist()

    # Attempt new query_points API first
    try:
        qp = client.query_points(
            collection_name=COLLECTION,
            query=vec,
            using=vec_name,
            search_params=models.SearchParams(hnsw_ef=args.ef),
            limit=args.limit,
            with_payload=False,
        )
        _ = qp
        print("Warm start via query_points: OK")
        return
    except Exception:
        pass

    # Fallback to search API
    try:
        _ = client.search(
            collection_name=COLLECTION,
            query_vector={"name": vec_name, "vector": vec},
            limit=args.limit,
            with_payload=False,
        )
        print("Warm start via search: OK")
    except Exception as e:
        print(f"Warm start failed: {e}")


if __name__ == "__main__":
    main()
