#!/usr/bin/env python3
"""
mcp/memory.py - Memory store/find implementations for MCP indexer server.

Extracted from mcp_indexer_server.py for better modularity.
Contains:
- _memory_store_impl: Store memory entries (called by thin @mcp.tool() wrapper)
- _memory_find_impl: Find memory entries (called by thin @mcp.tool() wrapper)

Note: The @mcp.tool() decorated memory_store/memory_find functions remain in
mcp_indexer_server.py as thin wrappers that call these implementations.
"""

from __future__ import annotations

__all__ = [
    "_memory_store_impl",
    "_memory_find_impl",
]

import asyncio
import hashlib
import logging
import os
import re
import uuid
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Environment
QDRANT_URL = os.environ.get("QDRANT_URL", "http://qdrant:6333")


async def _memory_store_impl(
    information: str,
    metadata: Optional[Dict[str, Any]] = None,
    collection: Optional[str] = None,
    default_collection_fn=None,
    get_embedding_model_fn=None,
) -> Dict[str, Any]:
    """Store a free-form memory entry in Qdrant using the active collection.

    - Embeds the text and writes both dense and lexical vectors (plus mini vector in ReFRAG mode).
    - Honors explicit collection overrides; otherwise falls back to workspace/env defaults.
    - Returns a payload compatible with context-aware tools.
    """
    try:
        from qdrant_client import QdrantClient, models  # type: ignore
        from fastembed import TextEmbedding  # type: ignore
        from scripts.utils import sanitize_vector_name
        from scripts.ingest_code import ensure_collection as _ensure_collection  # type: ignore
        from scripts.ingest_code import project_mini as _project_mini  # type: ignore

    except Exception as e:  # pragma: no cover
        return {"error": f"deps: {e}"}

    if not information or not str(information).strip():
        return {"error": "information is required"}

    # Get default collection
    if default_collection_fn:
        coll = (collection or default_collection_fn()) or ""
    else:
        from scripts.mcp_impl.workspace import _default_collection
        coll = (collection or _default_collection()) or ""

    model_name = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
    vector_name = sanitize_vector_name(model_name)

    # Minimal lexical hashing (aligns with ingest_code defaults)
    LEX_VECTOR_NAME = os.environ.get("LEX_VECTOR_NAME", "lex")
    LEX_VECTOR_DIM = int(os.environ.get("LEX_VECTOR_DIM", "4096") or 4096)

    def _split_ident_lex(s: str):
        parts = re.split(r"[^A-Za-z0-9]+", s)
        out: list[str] = []
        for p in parts:
            if not p:
                continue
            segs = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+", p)
            out.extend([x for x in segs if x])
        return [x.lower() for x in out if x]

    def _lex_hash_vector(text: str, dim: int = LEX_VECTOR_DIM) -> list[float]:
        # Delegate to shared utility for consistency
        try:
            from scripts.utils import lex_hash_vector_text

            return lex_hash_vector_text(text, dim)
        except Exception:
            # Fallback: minimal hashing
            if not text:
                return [0.0] * dim
            vec = [0.0] * dim
            toks = _split_ident_lex(text)
            if not toks:
                return vec
            for t in toks:
                h = int(
                    hashlib.md5(t.encode("utf-8", errors="ignore")).hexdigest()[:8], 16
                )
                vec[h % dim] += 1.0
            norm = (sum(v * v for v in vec) or 0.0) ** 0.5 or 1.0
            return [v / norm for v in vec]

    # Build vectors (cached embedding model)
    if get_embedding_model_fn:
        model = get_embedding_model_fn(model_name)
    else:
        from scripts.mcp_impl.admin_tools import _get_embedding_model
        model = _get_embedding_model(model_name)

    dense = next(model.embed([str(information)])).tolist()

    lex = _lex_hash_vector(str(information))

    # Upsert
    try:
        client = QdrantClient(
            url=QDRANT_URL,
            api_key=os.environ.get("QDRANT_API_KEY"),
            timeout=float(os.environ.get("QDRANT_TIMEOUT", "20") or 20),
        )
        # Ensure collection and named vectors exist (dense + lexical)
        try:
            await asyncio.to_thread(
                lambda: _ensure_collection(client, coll, len(dense), vector_name)
            )
        except Exception:
            pass
        pid = str(uuid.uuid4())
        payload = {
            "information": str(information),
            "metadata": metadata or {"kind": "memory", "source": "memory"},
        }
        vecs = {vector_name: dense, LEX_VECTOR_NAME: lex}
        try:
            if str(os.environ.get("REFRAG_MODE", "")).strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }:
                mini_name = os.environ.get("MINI_VECTOR_NAME", "mini")
                mini = _project_mini(
                    list(dense), int(os.environ.get("MINI_VEC_DIM", "64") or 64)
                )
                vecs[mini_name] = mini
        except Exception:
            pass
        point = models.PointStruct(id=pid, vector=vecs, payload=payload)
        await asyncio.to_thread(
            lambda: client.upsert(collection_name=coll, points=[point], wait=True)
        )
        return {"ok": True, "id": pid, "collection": coll, "vector_name": vector_name}
    except Exception as e:
        return {"error": str(e)}


async def _memory_find_impl(
    query: str,
    limit: Optional[int] = None,
    collection: Optional[str] = None,
    top_k: Optional[int] = None,
    default_collection_fn=None,
    get_embedding_model_fn=None,
) -> Dict[str, Any]:
    """Find memory entries by vector similarity (dense + lexical fusion).

    - Performs hybrid search using both dense and lexical vectors.
    - Returns scored results with information and metadata.

    Parameters:
    - query: str. Search query text.
    - limit: int (default 5). Maximum results to return.
    - collection: str (optional). Target collection override.
    - top_k: int (optional). Alias for limit.

    Returns:
    - {ok: true, results: [{id, score, information, metadata}, ...], count: N}
    """
    try:
        from qdrant_client import QdrantClient, models  # type: ignore
        from scripts.utils import sanitize_vector_name
    except Exception as e:  # pragma: no cover
        return {"ok": False, "error": f"deps: {e}", "results": [], "count": 0}

    if not query or not str(query).strip():
        return {"ok": False, "error": "query is required", "results": [], "count": 0}

    # Resolve limit (top_k is alias)
    effective_limit = limit if limit is not None else (top_k if top_k is not None else 5)
    effective_limit = max(1, min(100, int(effective_limit)))

    # Get default collection
    if default_collection_fn:
        coll = (collection or default_collection_fn()) or ""
    else:
        from scripts.mcp_impl.workspace import _default_collection
        coll = (collection or _default_collection()) or ""

    if not coll:
        return {"ok": False, "error": "no collection configured", "results": [], "count": 0}

    model_name = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
    vector_name = sanitize_vector_name(model_name)

    # Lexical vector settings
    LEX_VECTOR_NAME = os.environ.get("LEX_VECTOR_NAME", "lex")
    LEX_VECTOR_DIM = int(os.environ.get("LEX_VECTOR_DIM", "4096") or 4096)

    def _split_ident_lex(s: str):
        parts = re.split(r"[^A-Za-z0-9]+", s)
        out: list[str] = []
        for p in parts:
            if not p:
                continue
            segs = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+", p)
            out.extend([x for x in segs if x])
        return [x.lower() for x in out if x]

    def _lex_hash_vector(text: str, dim: int = LEX_VECTOR_DIM) -> list[float]:
        try:
            from scripts.utils import lex_hash_vector_text
            return lex_hash_vector_text(text, dim)
        except Exception:
            if not text:
                return [0.0] * dim
            vec = [0.0] * dim
            toks = _split_ident_lex(text)
            if not toks:
                return vec
            for t in toks:
                h = int(
                    hashlib.md5(t.encode("utf-8", errors="ignore")).hexdigest()[:8], 16
                )
                vec[h % dim] += 1.0
            norm = (sum(v * v for v in vec) or 0.0) ** 0.5 or 1.0
            return [v / norm for v in vec]

    try:
        # Build query vectors
        if get_embedding_model_fn:
            model = get_embedding_model_fn(model_name)
        else:
            from scripts.mcp_impl.admin_tools import _get_embedding_model
            model = _get_embedding_model(model_name)

        dense_query = next(model.embed([str(query)])).tolist()
        lex_query = _lex_hash_vector(str(query))

        client = QdrantClient(
            url=QDRANT_URL,
            api_key=os.environ.get("QDRANT_API_KEY"),
            timeout=float(os.environ.get("QDRANT_TIMEOUT", "20") or 20),
        )

        # Hybrid search using prefetch + RRF fusion
        prefetch_limit = max(effective_limit * 3, 20)

        prefetch = [
            models.Prefetch(
                query=dense_query,
                using=vector_name,
                limit=prefetch_limit,
            ),
            models.Prefetch(
                query=lex_query,
                using=LEX_VECTOR_NAME,
                limit=prefetch_limit,
            ),
        ]

        hits = await asyncio.to_thread(
            lambda: client.query_points(
                collection_name=coll,
                prefetch=prefetch,
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=effective_limit,
                with_payload=True,
            )
        )

        results = []
        for pt in getattr(hits, "points", []):
            payload = pt.payload or {}
            results.append({
                "id": str(pt.id),
                "score": getattr(pt, "score", 0.0),
                "information": payload.get("information", ""),
                "metadata": payload.get("metadata", {}),
            })

        return {"ok": True, "results": results, "count": len(results), "collection": coll}

    except Exception as e:
        return {"ok": False, "error": str(e), "results": [], "count": 0}
