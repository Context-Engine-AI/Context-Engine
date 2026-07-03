#!/usr/bin/env python3
"""
CoIR Corpus Indexer - Thin adapter over shared benchmark indexing core.

Transforms CoIR dataset entries into generic BenchmarkDoc format and delegates
to scripts/benchmarks/core_indexer.py for actual indexing.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from qdrant_client import QdrantClient

from scripts.benchmarks.core_indexer import (
    BenchmarkDoc,
    index_benchmark_corpus,
    compute_corpus_fingerprint as _compute_corpus_fingerprint,
    get_collection_fingerprint as _get_collection_fingerprint,
    collection_matches_corpus as _collection_matches_corpus,
)
from scripts.benchmarks.qdrant_utils import (
    get_qdrant_client,
    get_embedding_model,
)

# Collection prefix for CoIR benchmarks
COIR_COLLECTION_PREFIX = "coir-bench-"


def get_temp_collection(task_name: str = "default") -> str:
    """Get collection name for a CoIR task."""
    return f"{COIR_COLLECTION_PREFIX}{task_name}"


def get_corpus_collection(corpus: List[Dict[str, Any]]) -> str:
    """Get a stable collection name for a specific corpus."""
    fp = compute_corpus_fingerprint(corpus)
    return f"{COIR_COLLECTION_PREFIX}{fp}"


def coir_entry_to_doc(entry: Dict[str, Any]) -> BenchmarkDoc:
    """Convert CoIR entry to BenchmarkDoc."""
    doc_id = entry.get("_id", "")
    text = entry.get("text", "")
    title = entry.get("title", "")

    # Combine title + text for embedding (matches original indexer)
    full_text = f"{title}\n{text}".strip() if title else text

    return BenchmarkDoc(
        doc_id=doc_id,
        text=full_text,
        language="python",  # CoIR is code retrieval, assume Python
        metadata={
            "_id": doc_id,
            "title": title,
            "source": "coir",
        },
    )


def compute_corpus_fingerprint(corpus: List[Dict[str, Any]]) -> str:
    """Compute fingerprint for CoIR corpus."""
    docs = [coir_entry_to_doc(e) for e in corpus]
    return _compute_corpus_fingerprint(docs)


def get_collection_fingerprint(collection: str) -> str | None:
    """Get stored fingerprint from collection."""
    client = get_qdrant_client()
    return _get_collection_fingerprint(client, collection)


def collection_matches_corpus(
    collection: str,
    corpus: List[Dict[str, Any]],
    corpus_fingerprint: str | None = None,
) -> bool:
    """Check if collection matches corpus fingerprint."""
    client = get_qdrant_client()
    docs = [coir_entry_to_doc(e) for e in corpus]
    return _collection_matches_corpus(client, collection, docs)


def index_coir_corpus(
    corpus: List[Dict[str, Any]],
    collection: str,
    batch_size: int = 100,
    recreate: bool = False,
    show_progress: bool = True,
    force: bool = False,
) -> Dict[str, Any]:
    """Index a CoIR corpus using shared benchmark indexer.

    Args:
        corpus: List of {"_id": str, "text": str, "title": str (optional)}
        collection: Qdrant collection name
        batch_size: Batch size for indexing
        recreate: Drop and recreate collection
        show_progress: Show progress (ignored, kept for API compatibility)
        force: Force reindexing even if collection matches corpus

    Returns:
        {"indexed": int, "collection": str, "time_s": float, "reused": bool}
    """
    # Convert CoIR entries to generic BenchmarkDoc format
    docs = [coir_entry_to_doc(e) for e in corpus]

    # Get client and model
    client = get_qdrant_client()
    model = get_embedding_model()

    # Delegate to shared benchmark indexer
    if force:
        recreate = True

    result = index_benchmark_corpus(
        docs=docs,
        client=client,
        model=model,
        collection=collection,
        batch_size=batch_size,
        recreate=recreate,
        skip_if_exists=not force,
    )

    # Convert result format to match old API
    return {
        "indexed": result.get("indexed_count", 0),
        "collection": collection,
        "time_s": result.get("duration_sec", 0.0),
        "reused": result.get("reused", False),
    }


def cleanup_coir_collections(task_names: List[str] | None = None) -> int:
    """Delete CoIR benchmark collections.

    Args:
        task_names: Specific task names to delete, or None for all coir-bench-* collections

    Returns:
        Number of collections deleted
    """
    client = get_qdrant_client()
    deleted = 0

    try:
        collections = client.get_collections().collections
        for coll in collections:
            name = coll.name
            if not name.startswith(COIR_COLLECTION_PREFIX):
                continue

            # If task_names specified, only delete matching ones
            if task_names:
                task_suffix = name[len(COIR_COLLECTION_PREFIX):]
                if task_suffix not in task_names:
                    continue

            try:
                client.delete_collection(name)
                deleted += 1
                print(f"Deleted collection: {name}")
            except Exception as e:
                print(f"Failed to delete {name}: {e}")
    except Exception as e:
        print(f"Failed to list collections: {e}")

    return deleted