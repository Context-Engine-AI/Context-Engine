#!/usr/bin/env python3
import os
from qdrant_client import QdrantClient, models

QDRANT_URL = os.environ.get("QDRANT_URL", "http://qdrant:6333")
COLLECTION = os.environ.get("COLLECTION_NAME", "codebase")

cli = QdrantClient(url=QDRANT_URL)

# Add an alternative named vector expected by some clients (fastembed naming)
alt_name = "fast-bge-base-en-v1.5"
cli.update_collection(
    collection_name=COLLECTION,
    vectors_config={
        alt_name: models.VectorParams(size=768, distance=models.Distance.COSINE)
    },
)

info = cli.get_collection(COLLECTION)
print(info.config.params.vectors)
