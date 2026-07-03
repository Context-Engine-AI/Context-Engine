#!/usr/bin/env python3
"""
Qdrant client and query logic extracted from hybrid_search.py.

This module provides:
- Connection pooling setup and client management
- Thread executor for parallel queries
- Point coercion utilities
- Collection caching and management
- Query functions (lex_query, sparse_lex_query, dense_query)
- Lexical vector functions (lex_hash_vector, lex_sparse_vector)
"""

__all__ = [
    "_POOL_AVAILABLE", "get_qdrant_client", "return_qdrant_client", "pooled_qdrant_client",
    "_QUERY_EXECUTOR", "_EXECUTOR_LOCK", "_get_query_executor",
    "_coerce_points", "_legacy_vector_search",
    "_ENSURED_COLLECTIONS", "_get_client_endpoint", "_ensure_collection",
    "lex_hash_vector", "lex_sparse_vector",
    "lex_query", "sparse_lex_query", "dense_query",
    "_sanitize_filter_obj", "_sanitize_vector_name", "_ensure_payload_indexes",
]

import os
import logging
import threading
import re
from typing import List, Dict, Any, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# Core Qdrant imports
try:
    from qdrant_client import QdrantClient, models
except ImportError:
    QdrantClient = None  # type: ignore
    models = None  # type: ignore

logger = logging.getLogger("hybrid_qdrant")

# ---------------------------------------------------------------------------
# Helper functions for safe type conversion
# ---------------------------------------------------------------------------

def _safe_int(val: Any, default: int) -> int:
    try:
        if val is None or (isinstance(val, str) and val.strip() == ""):
            return default
        return int(val)
    except (ValueError, TypeError):
        return default


def _safe_float(val: Any, default: float) -> float:
    try:
        if val is None or (isinstance(val, str) and val.strip() == ""):
            return default
        return float(val)
    except (ValueError, TypeError):
        return default


# ---------------------------------------------------------------------------
# Configuration constants (from environment)
# ---------------------------------------------------------------------------

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
API_KEY = os.environ.get("QDRANT_API_KEY")

# Lexical vector configuration
# Imported from ingest config to ensure Single Source of Truth
from scripts.ingest.config import (
    LEX_VECTOR_NAME,
    LEX_VECTOR_DIM,
    LEX_SPARSE_NAME,
    LEX_SPARSE_MODE,
)

EF_SEARCH = _safe_int(os.environ.get("QDRANT_EF_SEARCH", "128"), 128)

# Quantization search params (for faster search with quantized collections)
QDRANT_QUANTIZATION = os.environ.get("QDRANT_QUANTIZATION", "none").strip().lower()
QDRANT_QUANTIZATION_RESCORE = os.environ.get("QDRANT_QUANTIZATION_RESCORE", "1").strip().lower() in ("1", "true", "yes", "on")
QDRANT_QUANTIZATION_OVERSAMPLING = float(os.environ.get("QDRANT_QUANTIZATION_OVERSAMPLING", "2.0") or 2.0)


def _get_search_params(ef: int) -> models.SearchParams:
    """Build SearchParams with optional quantization settings."""
    if QDRANT_QUANTIZATION in {"scalar", "binary"}:
        return models.SearchParams(
            hnsw_ef=ef,
            quantization=models.QuantizationSearchParams(
                rescore=QDRANT_QUANTIZATION_RESCORE,
                oversampling=QDRANT_QUANTIZATION_OVERSAMPLING,
            )
        )
    return models.SearchParams(hnsw_ef=ef)


# ---------------------------------------------------------------------------
# Connection pooling setup
# ---------------------------------------------------------------------------

try:
    from scripts.qdrant_client_manager import get_qdrant_client, return_qdrant_client, pooled_qdrant_client
    _POOL_AVAILABLE = True
except ImportError:
    _POOL_AVAILABLE = False

    def get_qdrant_client(url=None, api_key=None, force_new=False, use_pool=True):
        """Fallback client creation when pooling is unavailable."""
        if QdrantClient is None:
            raise ImportError(
                "qdrant_client is not installed. Install with: pip install qdrant-client"
            )
        return QdrantClient(
            url=url or os.environ.get("QDRANT_URL", "http://localhost:6333"),
            api_key=api_key or os.environ.get("QDRANT_API_KEY")
        )

    def return_qdrant_client(client):
        """No-op when pooling is unavailable."""
        pass

    class pooled_qdrant_client:
        """Fallback context manager when pooling is unavailable."""
        def __init__(self, url=None, api_key=None):
            self.url = url
            self.api_key = api_key
            self.client = None

        def __enter__(self):
            self.client = get_qdrant_client(self.url, self.api_key)
            return self.client

        def __exit__(self, exc_type, exc_val, exc_tb):
            return_qdrant_client(self.client)


# ---------------------------------------------------------------------------
# Thread executor for parallel queries
# ---------------------------------------------------------------------------

_QUERY_EXECUTOR: ThreadPoolExecutor | None = None
_EXECUTOR_LOCK = threading.Lock()


def _get_query_executor(max_workers: int = 4) -> ThreadPoolExecutor:
    """Get or create a shared ThreadPoolExecutor for parallel queries."""
    global _QUERY_EXECUTOR
    if _QUERY_EXECUTOR is None:
        with _EXECUTOR_LOCK:
            if _QUERY_EXECUTOR is None:
                _QUERY_EXECUTOR = ThreadPoolExecutor(max_workers=max_workers)
    return _QUERY_EXECUTOR


# ---------------------------------------------------------------------------
# Point coercion
# ---------------------------------------------------------------------------

def _coerce_points(result: Any) -> List[Any]:
    """Normalize Qdrant responses to a list of points."""
    if result is None:
        return []
    if isinstance(result, list):
        return result
    try:
        return list(result)
    except TypeError:
        return [result]


# ---------------------------------------------------------------------------
# Legacy search fallback
# ---------------------------------------------------------------------------

def _legacy_vector_search(
    client,
    collection: str,
    vec_name: str,
    vector: List[float],
    per_query: int,
    flt,
) -> List[Any]:
    """Fallback to legacy client.search when query_points is unavailable."""
    try:
        result = client.search(
            collection_name=collection,
            query_vector={"name": vec_name, "vector": vector},
            limit=per_query,
            with_payload=True,
            query_filter=flt,
        )
        return _coerce_points(getattr(result, "points", result))
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Collection caching
# ---------------------------------------------------------------------------

_ENSURED_COLLECTIONS: set[str] = set()
_COLLECTION_VECTOR_NAMES: Dict[str, set[str]] = {}


def _get_client_endpoint(client) -> str:
    """Extract endpoint identifier from Qdrant client for cache scoping."""
    try:
        if hasattr(client, '_client') and hasattr(client._client, '_host'):
            return f"{client._client._host}:{getattr(client._client, '_port', 6333)}"
        if hasattr(client, 'rest_uri'):
            return client.rest_uri
        return os.environ.get("QDRANT_URL", "localhost:6333")
    except Exception:
        return os.environ.get("QDRANT_URL", "localhost:6333")


def _collection_cache_key(client, collection: str) -> str:
    return f"{_get_client_endpoint(client)}:{collection}"


def _cache_collection_vectors(client, collection: str) -> set[str] | None:
    """Cache available vector names (dense + sparse) for a collection."""
    cache_key = _collection_cache_key(client, collection)
    if cache_key in _COLLECTION_VECTOR_NAMES:
        return _COLLECTION_VECTOR_NAMES[cache_key]
    try:
        info = client.get_collection(collection)
    except Exception:
        return None
    try:
        vnames: set[str] = set()
        vcfg = info.config.params.vectors
        if isinstance(vcfg, dict):
            vnames.update(vcfg.keys())
        elif hasattr(vcfg, "size"):
            vnames.add("")  # Default (unnamed) vector
        scfg = info.config.params.sparse_vectors
        if isinstance(scfg, dict):
            vnames.update(scfg.keys())
        _COLLECTION_VECTOR_NAMES[cache_key] = vnames
        return vnames
    except Exception:
        return None


def _vector_available(client, collection: str, vector_name: str | None) -> bool:
    if not vector_name:
        return True
    vnames = _cache_collection_vectors(client, collection)
    if vnames is None:
        return True
    return vector_name in vnames


def _ensure_collection(client, collection: str, dim: int, vec_name: str):
    """Cached wrapper for ensure_collection - only calls once per (endpoint, collection, vec_name) pair.

    IMPORTANT: This is called during SEARCH operations. We must NOT delete/recreate collections
    that already exist with data. The ensure_collection in ingest_code can trigger recreation
    when PATTERN_VECTORS=1 or LEX_SPARSE_MODE=1 if the collection lacks those vectors.

    For search, we only need to verify the collection exists - not modify its schema.
    """
    endpoint = _get_client_endpoint(client)
    cache_key = f"{endpoint}:{collection}:{vec_name}:{dim}"
    if cache_key in _ENSURED_COLLECTIONS:
        return

    # For SEARCH operations, just verify collection exists - don't try to modify schema
    # Schema modifications can trigger deletion of existing data!
    vnames = _cache_collection_vectors(client, collection)
    if vnames is not None:
        _ENSURED_COLLECTIONS.add(cache_key)
        return

    # Collection doesn't exist - only then call ensure_collection to create it
    try:
        from scripts.ingest_code import ensure_collection as _ensure_collection_raw
        _ensure_collection_raw(client, collection, dim, vec_name)
    except ImportError:
        pass

    try:
        _cache_collection_vectors(client, collection)
    except Exception:
        pass
    _ENSURED_COLLECTIONS.add(cache_key)


def clear_ensured_collections():
    """Clear the collection cache (useful for testing)."""
    global _ENSURED_COLLECTIONS, _COLLECTION_VECTOR_NAMES
    _ENSURED_COLLECTIONS = set()
    _COLLECTION_VECTOR_NAMES = {}


# ---------------------------------------------------------------------------
# Collection name resolution
# ---------------------------------------------------------------------------

def _collection(collection_name: str | None = None) -> str:
    """Determine collection name with priority: CLI arg > env > workspace state > default."""
    if collection_name and collection_name.strip():
        return collection_name.strip()

    env_coll = os.environ.get("COLLECTION_NAME", "").strip()
    if env_coll:
        return env_coll

    try:
        import json
        workspace_root = Path(os.environ.get("WORKSPACE_PATH") or os.environ.get("WATCH_ROOT") or "/work")
        state_file = workspace_root / ".codebase" / "state.json"
        if state_file.exists():
            with open(state_file, "r", encoding="utf-8") as f:
                state = json.load(f)
            if isinstance(state, dict):
                coll = state.get("qdrant_collection")
                if isinstance(coll, str) and coll.strip():
                    return coll.strip()
    except Exception:
        pass

    return "codebase"


# ---------------------------------------------------------------------------
# Filter sanitization
# ---------------------------------------------------------------------------

_FILTER_CACHE: Dict[int, Any] = {}
_FILTER_CACHE_LOCK = threading.Lock()
_FILTER_CACHE_MAX = 256


def _sanitize_filter_obj(flt):
    """Sanitize Qdrant filter objects so we never send an empty filter {}.
    
    Qdrant returns 400 if filter has no conditions; return None in that case.
    Uses caching for repeated filter patterns to avoid redundant validation.
    """
    if flt is None:
        return None

    cache_key = id(flt)
    with _FILTER_CACHE_LOCK:
        if cache_key in _FILTER_CACHE:
            return _FILTER_CACHE[cache_key]

    try:
        must = getattr(flt, "must", None)
        should = getattr(flt, "should", None)
        must_not = getattr(flt, "must_not", None)
        if must is None and should is None and must_not is None:
            if isinstance(flt, dict):
                m = [c for c in (flt.get("must") or []) if c is not None]
                s = [c for c in (flt.get("should") or []) if c is not None]
                mn = [c for c in (flt.get("must_not") or []) if c is not None]
                result = None if (not m and not s and not mn) else flt
            else:
                result = None
        else:
            m = [c for c in (must or []) if c is not None]
            s = [c for c in (should or []) if c is not None]
            mn = [c for c in (must_not or []) if c is not None]
            result = None if (not m and not s and not mn) else flt
    except Exception:
        result = None

    with _FILTER_CACHE_LOCK:
        if len(_FILTER_CACHE) < _FILTER_CACHE_MAX:
            _FILTER_CACHE[cache_key] = result

    return result


# ---------------------------------------------------------------------------
# Lexical vector functions
# ---------------------------------------------------------------------------

_STOP = {
    "the", "a", "an", "of", "in", "on", "for", "and", "or", "to",
    "with", "by", "is", "are", "be", "this", "that",
}


def _split_ident_lex(s: str) -> List[str]:
    """Split identifier into tokens (snake_case and camelCase aware)."""
    parts = re.split(r"[^A-Za-z0-9]+", s)
    out: List[str] = []
    for p in parts:
        if not p:
            continue
        segs = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+", p)
        out.extend([x for x in segs if x])
    return [x.lower() for x in out if x and x.lower() not in _STOP]


def lex_hash_vector(phrases: List[str], dim: int | None = None) -> List[float]:
    """Generate dense lexical hash vector for query phrases."""
    if dim is None:
        dim = LEX_VECTOR_DIM
    try:
        from scripts.utils import lex_hash_vector_queries as _lex_hash_vector_queries
        return _lex_hash_vector_queries(phrases, dim)
    except ImportError:
        return _fallback_lex_hash_vector(phrases, dim)


def _fallback_lex_hash_vector(phrases: List[str], dim: int) -> List[float]:
    """Fallback implementation when utils is unavailable."""
    import hashlib
    vec = [0.0] * dim
    for phrase in phrases:
        for tok in _split_ident_lex(phrase):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            idx = h % dim
            vec[idx] += 1.0
    norm = sum(v * v for v in vec) ** 0.5
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


def lex_sparse_vector(phrases: List[str]) -> Dict[str, Any]:
    """Generate sparse vector for query phrases (lossless exact matching)."""
    try:
        from scripts.utils import lex_sparse_vector_queries as _lex_sparse_vector_queries
        return _lex_sparse_vector_queries(phrases)
    except ImportError:
        return _fallback_lex_sparse_vector(phrases)


def _fallback_lex_sparse_vector(phrases: List[str]) -> Dict[str, Any]:
    """Fallback implementation when utils is unavailable."""
    import hashlib
    indices = []
    values = []
    seen = set()
    for phrase in phrases:
        for tok in _split_ident_lex(phrase):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16) % (2**31)
            if h not in seen:
                indices.append(h)
                values.append(1.0)
                seen.add(h)
    return {"indices": indices, "values": values}


# ---------------------------------------------------------------------------
# Query functions
# ---------------------------------------------------------------------------

def lex_query(
    client,
    v: List[float],
    flt,
    per_query: int,
    collection_name: str | None = None
) -> List[Any]:
    """Query using dense lexical hash vector."""
    ef = max(EF_SEARCH, 32 + 4 * int(per_query))
    flt = _sanitize_filter_obj(flt)
    collection = _collection(collection_name)
    if not _vector_available(client, collection, LEX_VECTOR_NAME):
        if os.environ.get("DEBUG_HYBRID_SEARCH"):
            logger.debug(f"Skipping lex query: {LEX_VECTOR_NAME} not in collection vectors")
        return []

    try:
        qp = client.query_points(
            collection_name=collection,
            query=v,
            using=LEX_VECTOR_NAME,
            query_filter=flt,
            search_params=_get_search_params(ef),
            limit=per_query,
            with_payload=True,
        )
        return _coerce_points(getattr(qp, "points", qp))
    except TypeError:
        if os.environ.get("DEBUG_HYBRID_SEARCH"):
            logger.debug("QP_FILTER_KWARG_SWITCH", extra={"using": LEX_VECTOR_NAME})
        qp = client.query_points(
            collection_name=collection,
            query=v,
            using=LEX_VECTOR_NAME,
            filter=flt,
            search_params=_get_search_params(ef),
            limit=per_query,
            with_payload=True,
        )
        return _coerce_points(getattr(qp, "points", qp))
    except AttributeError:
        return _legacy_vector_search(client, collection, LEX_VECTOR_NAME, v, per_query, flt)
    except Exception as e:
        if os.environ.get("DEBUG_HYBRID_SEARCH"):
            try:
                logger.debug("QP_FILTER_DROP", extra={"using": LEX_VECTOR_NAME, "reason": str(e)[:200]})
            except Exception:
                pass
        try:
            qp = client.query_points(
                collection_name=collection,
                query=v,
                using=LEX_VECTOR_NAME,
                query_filter=None,
                search_params=_get_search_params(ef),
                limit=per_query,
                with_payload=True,
            )
            return _coerce_points(getattr(qp, "points", qp))
        except TypeError:
            qp = client.query_points(
                collection_name=collection,
                query=v,
                using=LEX_VECTOR_NAME,
                filter=None,
                search_params=_get_search_params(ef),
                limit=per_query,
                with_payload=True,
            )
            return _coerce_points(getattr(qp, "points", qp))
        except Exception as e2:
            if os.environ.get("DEBUG_HYBRID_SEARCH"):
                try:
                    logger.debug("QP_FILTER_DROP_FAILED", extra={"using": LEX_VECTOR_NAME, "reason": str(e2)[:200]})
                except Exception:
                    pass
        return _legacy_vector_search(client, collection, LEX_VECTOR_NAME, v, per_query, flt)


def sparse_lex_query(
    client,
    sparse_vec: Dict[str, Any],
    flt,
    per_query: int,
    collection_name: str | None = None
) -> List[Any]:
    """Query using sparse lexical vector for lossless exact matching."""
    flt = _sanitize_filter_obj(flt)
    collection = _collection(collection_name)
    
    # Check if sparse vector exists in collection
    vnames = _cache_collection_vectors(client, collection)
    if vnames is not None and LEX_SPARSE_NAME not in vnames:
        if os.environ.get("DEBUG_HYBRID_SEARCH"):
            logger.debug(f"Skipping sparse lex query: {LEX_SPARSE_NAME} not in {vnames}")
        return []

    if not sparse_vec.get("indices"):
        return []

    try:
        qp = client.query_points(
            collection_name=collection,
            query=models.SparseVector(
                indices=sparse_vec["indices"],
                values=sparse_vec["values"],
            ),
            using=LEX_SPARSE_NAME,
            query_filter=flt,
            limit=per_query,
            with_payload=True,
        )
        return _coerce_points(getattr(qp, "points", qp))
    except TypeError:
        try:
            qp = client.query_points(
                collection_name=collection,
                query=models.SparseVector(
                    indices=sparse_vec["indices"],
                    values=sparse_vec["values"],
                ),
                using=LEX_SPARSE_NAME,
                filter=flt,
                limit=per_query,
                with_payload=True,
            )
            return _coerce_points(getattr(qp, "points", qp))
        except Exception:
            return []
    except Exception as e:
        if os.environ.get("DEBUG_HYBRID_SEARCH"):
            logger.debug("SPARSE_LEX_QUERY_ERROR", extra={"error": str(e)[:200]})
        return []


def dense_query(
    client,
    vec_name: str,
    v: List[float],
    flt,
    per_query: int,
    collection_name: str | None = None,
    query_text: str | None = None
) -> List[Any]:
    """Query using dense embedding vector."""
    # Default EF: scale with per_query for adequate recall
    ef = max(EF_SEARCH, 32 + 4 * int(per_query))

    # Apply dynamic EF optimization if query text provided
    if query_text:
        try:
            from scripts.query_optimizer import optimize_query
            result = optimize_query(query_text)
            # Only override EF when adaptive optimization is enabled
            if result.get("adaptive_enabled", False):
                ef = result["recommended_ef"]
                if os.environ.get("DEBUG_HYBRID_SEARCH"):
                    logger.debug(f"Dynamic EF: {ef} (complexity={result['complexity']}, type={result['query_type']})")
        except ImportError:
            pass
        except Exception as e:
            if os.environ.get("DEBUG_HYBRID_SEARCH"):
                logger.debug(f"Query optimizer failed, using default EF: {e}")

    flt = _sanitize_filter_obj(flt)
    collection = _collection(collection_name)
    if not _vector_available(client, collection, vec_name):
        if os.environ.get("DEBUG_HYBRID_SEARCH"):
            logger.debug(f"Skipping dense query: {vec_name} not in collection vectors")
        return []

    try:
        qp = client.query_points(
            collection_name=collection,
            query=v,
            using=vec_name,
            query_filter=flt,
            search_params=_get_search_params(ef),
            limit=per_query,
            with_payload=True,
        )
        return _coerce_points(getattr(qp, "points", qp))
    except TypeError:
        if os.environ.get("DEBUG_HYBRID_SEARCH"):
            logger.debug("QP_FILTER_KWARG_SWITCH", extra={"using": vec_name})
        qp = client.query_points(
            collection_name=collection,
            query=v,
            using=vec_name,
            filter=flt,
            search_params=_get_search_params(ef),
            limit=per_query,
            with_payload=True,
        )
        return _coerce_points(getattr(qp, "points", qp))
    except Exception as e:
        if os.environ.get("DEBUG_HYBRID_SEARCH"):
            try:
                logger.debug("QP_FILTER_DROP", extra={"using": vec_name, "reason": str(e)[:200]})
            except Exception:
                pass
        if not collection:
            return _legacy_vector_search(client, _collection(), vec_name, v, per_query, flt)
        try:
            qp = client.query_points(
                collection_name=collection,
                query=v,
                using=vec_name,
                query_filter=None,
                search_params=_get_search_params(ef),
                limit=per_query,
                with_payload=True,
            )
            return _coerce_points(getattr(qp, "points", qp))
        except TypeError:
            try:
                qp = client.query_points(
                    collection_name=collection,
                    query=v,
                    using=vec_name,
                    filter=None,
                    search_params=_get_search_params(ef),
                    limit=per_query,
                    with_payload=True,
                )
                return _coerce_points(getattr(qp, "points", qp))
            except Exception as e2:
                if os.environ.get("DEBUG_HYBRID_SEARCH"):
                    try:
                        logger.debug("QP_FILTER_DROP_FAILED", extra={"using": vec_name, "reason": str(e2)[:200]})
                    except Exception:
                        pass
        return _legacy_vector_search(client, collection, vec_name, v, per_query, flt)


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

__all__ = [
    # Pool availability flag
    "_POOL_AVAILABLE",
    # Connection pooling
    "get_qdrant_client",
    "return_qdrant_client",
    "pooled_qdrant_client",
    # Thread executor
    "_QUERY_EXECUTOR",
    "_EXECUTOR_LOCK",
    "_get_query_executor",
    # Point coercion
    "_coerce_points",
    # Legacy search
    "_legacy_vector_search",
    # Collection caching
    "_ENSURED_COLLECTIONS",
    "_get_client_endpoint",
    "_ensure_collection",
    "clear_ensured_collections",
    # Collection name resolution
    "_collection",
    # Filter sanitization
    "_sanitize_filter_obj",
    # Lexical vector functions
    "_split_ident_lex",
    "lex_hash_vector",
    "lex_sparse_vector",
    # Query functions
    "lex_query",
    "sparse_lex_query",
    "dense_query",
    # Constants
    "QDRANT_URL",
    "API_KEY",
    "LEX_VECTOR_NAME",
    "LEX_VECTOR_DIM",
    "LEX_SPARSE_NAME",
    "LEX_SPARSE_MODE",
    "EF_SEARCH",
]
