#!/usr/bin/env python3
"""
Shared Qdrant + embedder helpers for benchmark indexers.

Benchmarks should not depend on each other (e.g., CoIR importing CoSQA) just to
reuse small utilities. Keep these helpers in a neutral module.
"""
from __future__ import annotations

import os
from typing import Any

from qdrant_client import QdrantClient


def get_qdrant_client() -> QdrantClient:
    """Get Qdrant client with standard configuration."""
    url = os.environ.get("QDRANT_URL") or "http://localhost:6333"
    api_key = os.environ.get("QDRANT_API_KEY")
    timeout = int(os.environ.get("QDRANT_TIMEOUT") or "60")
    return QdrantClient(url=url, api_key=api_key or None, timeout=timeout)


def get_embedding_model() -> Any:
    """Get embedding model using standard Context-Engine factory."""
    try:
        from scripts.embedder import get_embedding_model as _get_model
        return _get_model()
    except ImportError:
        from fastembed import TextEmbedding

        model_name = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
        return TextEmbedding(model_name=model_name)


def get_model_dimension(model: Any) -> int:
    """Get embedding dimension from model."""
    try:
        from scripts.embedder import get_model_dimension as _get_dim

        model_name = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
        return _get_dim(model_name)
    except ImportError:
        # Probe dimension
        vec = list(model.embed(["test"]))[0]
        return len(vec)


def probe_pseudo_tags(collection: str, limit: int = 5) -> None:
    """Probe collection to verify LLM pseudo/tags presence."""
    # Only probe if feature is enabled
    if os.environ.get("REFRAG_PSEUDO_DESCRIBE", "0") != "1":
        return

    try:
        print("  [probe] Verifying LLM pseudo/tags presence...")
        client = get_qdrant_client()
        scroll_res = client.scroll(
            collection_name=collection,
            limit=limit,
            with_payload=["pseudo", "tags", "docstring", "symbol"],
        )
        points, _ = scroll_res
        valid_pseudo = 0
        valid_tags = 0
        
        for p in points:
            payload = p.payload or {}
            # Check pseudo (should exist and be non-empty)
            if payload.get("pseudo"):
                valid_pseudo += 1
            # Check tags (should exist and be list)
            if isinstance(payload.get("tags"), list) and payload["tags"]:
                valid_tags += 1
        
        print(f"  [probe] Checked {len(points)} docs: {valid_pseudo} have pseudo, {valid_tags} have tags")
        
        if len(points) > 0 and (valid_pseudo < len(points) or valid_tags < len(points)):
            print("  [WARN] Some documents missing pseudo/tags! Check ingestion logs.")
            if points:
                from dataclasses import asdict
                p0 = points[0]
                # Safe payload print
                safe_payload = {k: v for k, v in (p0.payload or {}).items() if k in ["pseudo", "tags", "symbol"]}
                print(f"  Sample id={p0.id} payload: {safe_payload}")

    except Exception as e:
        print(f"  [probe] Failed to probe collection: {e}")


def verify_config_compatibility(client: Any, collection_name: str) -> None:
    """Verify that Qdrant collection schema matches current environment configuration.
    
    Raises:
        ValueError: If a critical configuration mismatch is detected (e.g., vector dimensions).
    """
    print(f"  [verify] Checking schema for '{collection_name}'...")
    try:
        info = client.get_collection(collection_name)
    except Exception:
        print(f"  [verify] Collection {collection_name} does not exist. Will be created/verified by indexer.")
        return

    vectors = info.config.params.vectors
    
    # Check Lexical Vector Dimension
    from scripts.ingest.config import LEX_VECTOR_NAME, LEX_VECTOR_DIM
    lex_name = LEX_VECTOR_NAME
    expected_dim = LEX_VECTOR_DIM
    
    # Handle dict-style vector config (named vectors)
    if isinstance(vectors, dict):
        if lex_name in vectors:
            actual_dim = vectors[lex_name].size
            if actual_dim != expected_dim:
                raise ValueError(
                    f"CRITICAL CONFIG MISMATCH: Lexical vector '{lex_name}' dimension mismatch! "
                    f"Env={expected_dim}, Qdrant={actual_dim}. "
                    f"Use --recreate to fix."
                )
        elif hasattr(vectors, lex_name): # Handle object access if wrapper
             actual_dim = getattr(vectors, lex_name).size
             if actual_dim != expected_dim:
                raise ValueError(
                    f"CRITICAL CONFIG MISMATCH: Lexical vector '{lex_name}' dimension mismatch! "
                    f"Env={expected_dim}, Qdrant={actual_dim}."
                )
        else:
            # If expected but missing, that's also a fail
            raise ValueError(f"CRITICAL CONFIG MISMATCH: Vector '{lex_name}' missing in collection.")
            
        print("  [verify] Schema matches environment configuration.")
    else:
        # Fallback for single unnamed vector (shouldn't happen in this pipeline)
        print("  [verify] Single vector configuration detected (skipping deep verification).")
