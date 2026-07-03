#!/usr/bin/env python3
"""
Embedding utilities for hybrid search.

Provides cached embedding functionality with support for both the unified
cache system and legacy fallback caching. Handles Qwen3 instruction prefixes
and batch embedding optimizations.

This module is designed to be self-contained and importable by other modules
that need embedding capabilities without pulling in the full hybrid_search module.
"""
from __future__ import annotations

__all__ = [
    "_EMBEDDER_FACTORY", "_get_embedding_model", "_embed_queries_cached",
    "_EMBED_CACHE", "_EMBED_LOCK", "_EMBED_QUERY_CACHE",
    "EmbeddingModel", "get_embedding_model", "embed_queries_cached",
    "clear_embedding_cache", "get_embedding_cache_stats",
    "UNIFIED_CACHE_AVAILABLE", "TextEmbedding",
]

import os
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, List, Optional, TYPE_CHECKING

# ---------------------------------------------------------------------------
# Embedder factory setup
# ---------------------------------------------------------------------------
try:
    from scripts.embedder import get_embedding_model as _get_embedding_model
    _EMBEDDER_FACTORY = True
except ImportError:
    _EMBEDDER_FACTORY = False
    _get_embedding_model = None  # type: ignore

# Always try to import TextEmbedding for backward compatibility with tests
try:
    from fastembed import TextEmbedding
except ImportError:
    TextEmbedding = None  # type: ignore

# Type alias for embedding model (TextEmbedding or compatible)
EmbeddingModel = Any if TextEmbedding is None else TextEmbedding

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------
MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")

# ---------------------------------------------------------------------------
# Unified cache system
# ---------------------------------------------------------------------------
try:
    from scripts.cache_manager import get_embedding_cache
    UNIFIED_CACHE_AVAILABLE = True
except ImportError:
    UNIFIED_CACHE_AVAILABLE = False
    get_embedding_cache = None  # type: ignore

# Legacy cache fallback structures
_EMBED_QUERY_CACHE: OrderedDict[tuple[str, str], List[float]] = OrderedDict()
_EMBED_LOCK = threading.Lock()
def _safe_int(val, default: int) -> int:
    """Safely parse an integer from a value, returning default on failure."""
    try:
        if val is None or (isinstance(val, str) and val.strip() == ""):
            return default
        return int(val)
    except (ValueError, TypeError):
        return default

MAX_EMBED_CACHE = _safe_int(os.environ.get("MAX_EMBED_CACHE"), 8192)

# Unified cache instance (lazy initialized)
_EMBED_CACHE: Optional[Any] = None


def _get_embed_cache() -> Any:
    """Get or initialize the embedding cache (unified or legacy)."""
    global _EMBED_CACHE
    if UNIFIED_CACHE_AVAILABLE:
        if _EMBED_CACHE is None:
            _EMBED_CACHE = get_embedding_cache()
        return _EMBED_CACHE
    return None


# ---------------------------------------------------------------------------
# Public API: get_embedding_model
# ---------------------------------------------------------------------------
def get_embedding_model(model_name: Optional[str] = None) -> EmbeddingModel:
    """
    Get or create an embedding model instance.

    Uses the embedder factory when available, otherwise falls back to
    direct TextEmbedding instantiation.

    Args:
        model_name: Model name override. If None, uses EMBEDDING_MODEL env var.

    Returns:
        Embedding model instance.

    Raises:
        ImportError: If neither embedder factory nor fastembed is available.
    """
    if _EMBEDDER_FACTORY and _get_embedding_model is not None:
        return _get_embedding_model(model_name)

    if TextEmbedding is None:
        raise ImportError(
            "No embedding backend available. Install fastembed or ensure "
            "scripts.embedder is importable."
        )

    name = model_name or MODEL_NAME
    return TextEmbedding(model_name=name)


# ---------------------------------------------------------------------------
# Public API: embed_queries_cached
# ---------------------------------------------------------------------------
def embed_queries_cached(
    model: Any,
    queries: List[str],
) -> List[List[float]]:
    """
    Cache dense query embeddings to avoid repeated compute across expansions/retries.

    Optimized: batch-embeds all missing queries in one model call (2-5x faster).
    Thread-safe with bounded cache size.

    When Qwen3 is enabled and QWEN3_QUERY_INSTRUCTION=1, applies instruction
    prefix to queries before embedding for improved retrieval quality.

    Args:
        model: Embedding model instance (TextEmbedding or compatible).
        queries: List of query strings to embed.

    Returns:
        List of embedding vectors (one per query, guaranteed 1:1 correspondence).

    Raises:
        ValueError: If queries is empty or contains only invalid entries.
        RuntimeError: If embedding fails for any query.
    """
    if not queries:
        raise ValueError("queries must be a non-empty list")

    # Validate and sanitize queries - filter empty/whitespace but track positions
    sanitized: List[str] = []
    original_indices: List[int] = []
    for i, q in enumerate(queries):
        if q and isinstance(q, str) and q.strip():
            sanitized.append(q.strip())
            original_indices.append(i)

    if not sanitized:
        raise ValueError("queries contains no valid non-empty strings")

    try:
        # Best-effort model name extraction; fall back to env
        name = getattr(model, "model_name", None) or os.environ.get(
            "EMBEDDING_MODEL", MODEL_NAME
        )
    except Exception:
        name = os.environ.get("EMBEDDING_MODEL", MODEL_NAME)

    # Apply Qwen3 instruction prefix if enabled (queries only, not documents)
    try:
        from scripts.embedder import prefix_queries
        sanitized = prefix_queries(sanitized, name)
    except ImportError:
        pass

    cache = _get_embed_cache()

    if UNIFIED_CACHE_AVAILABLE and cache is not None:
        embeddings = _embed_with_unified_cache(model, sanitized, name, cache)
    else:
        embeddings = _embed_with_legacy_cache(model, sanitized, name)

    # Reconstruct full result list with None for invalid queries
    # (callers should not pass empty queries, but if they do, fail clearly)
    if len(embeddings) != len(sanitized):
        raise RuntimeError(
            f"Embedding count mismatch: got {len(embeddings)} for {len(sanitized)} queries"
        )

    # If all queries were valid, return directly
    if len(sanitized) == len(queries):
        return embeddings

    # Otherwise, we had some empty queries - raise error (callers must sanitize)
    raise ValueError(
        f"queries contained {len(queries) - len(sanitized)} empty/invalid entries; "
        "caller must filter empty queries before calling embed_queries_cached"
    )


def _embed_with_unified_cache(
    model: Any,
    queries: List[str],
    model_name: str,
    cache: Any,
) -> List[List[float]]:
    """Embed queries using the unified cache system."""
    missing_queries: List[str] = []
    missing_indices: List[int] = []

    # Find missing queries
    for i, q in enumerate(queries):
        key = (str(model_name), str(q))
        if cache.get(key) is None:
            missing_queries.append(str(q))
            missing_indices.append(i)

    # Batch-embed all missing queries in one call
    if missing_queries:
        try:
            vecs = list(model.embed(missing_queries))
            # Cache all new embeddings
            for q, vec in zip(missing_queries, vecs):
                key = (str(model_name), str(q))
                cache.set(key, vec.tolist())
        except Exception:
            # Fallback to one-by-one if batch fails
            for q in missing_queries:
                key = (str(model_name), str(q))
                vec = next(model.embed([q])).tolist()
                cache.set(key, vec)

    # Return embeddings in original order from cache
    out: List[List[float]] = []
    for q in queries:
        key = (str(model_name), str(q))
        v = cache.get(key)
        if v is None:
            raise RuntimeError(f"Failed to embed query: {q!r}")
        out.append(v)
    return out


def _embed_with_legacy_cache(
    model: Any,
    queries: List[str],
    model_name: str,
) -> List[List[float]]:
    """Embed queries using the legacy OrderedDict cache."""
    missing_queries: List[str] = []
    missing_indices: List[int] = []

    with _EMBED_LOCK:
        for i, q in enumerate(queries):
            key = (str(model_name), str(q))
            if key not in _EMBED_QUERY_CACHE:
                missing_queries.append(str(q))
                missing_indices.append(i)

    # Batch-embed all missing queries in one call
    if missing_queries:
        try:
            # Embed all missing queries at once
            vecs = list(model.embed(missing_queries))
            with _EMBED_LOCK:
                # Cache all new embeddings
                for q, vec in zip(missing_queries, vecs):
                    key = (str(model_name), str(q))
                    if key not in _EMBED_QUERY_CACHE:
                        _EMBED_QUERY_CACHE[key] = vec.tolist()
                        # Evict oldest entries if cache exceeds limit
                        while len(_EMBED_QUERY_CACHE) > MAX_EMBED_CACHE:
                            _EMBED_QUERY_CACHE.popitem(last=False)
        except Exception:
            # Fallback to one-by-one if batch fails
            for q in missing_queries:
                key = (str(model_name), str(q))
                vec = next(model.embed([q])).tolist()
                with _EMBED_LOCK:
                    if key not in _EMBED_QUERY_CACHE:
                        _EMBED_QUERY_CACHE[key] = vec
                        # Evict oldest entries if cache exceeds limit
                        while len(_EMBED_QUERY_CACHE) > MAX_EMBED_CACHE:
                            _EMBED_QUERY_CACHE.popitem(last=False)

    # Return embeddings in original order from cache (thread-safe read)
    out: List[List[float]] = []
    with _EMBED_LOCK:
        for q in queries:
            key = (str(model_name), str(q))
            v = _EMBED_QUERY_CACHE.get(key)
            if v is None:
                raise RuntimeError(f"Failed to embed query: {q!r}")
            out.append(v)
    return out


# ---------------------------------------------------------------------------
# Backward compatibility aliases
# ---------------------------------------------------------------------------
# Expose the internal function name used by hybrid_search.py
_embed_queries_cached = embed_queries_cached


def clear_embedding_cache() -> None:
    """Clear the embedding cache (both unified and legacy)."""
    global _EMBED_CACHE

    cache = _get_embed_cache()
    if cache is not None:
        try:
            cache.clear()
        except Exception:
            pass

    with _EMBED_LOCK:
        _EMBED_QUERY_CACHE.clear()


def get_embedding_cache_stats() -> dict:
    """Get statistics about the embedding cache."""
    cache = _get_embed_cache()
    if UNIFIED_CACHE_AVAILABLE and cache is not None:
        try:
            return cache.get_stats()
        except Exception:
            pass

    with _EMBED_LOCK:
        return {
            "cache_type": "legacy",
            "size": len(_EMBED_QUERY_CACHE),
            "max_size": MAX_EMBED_CACHE,
        }
