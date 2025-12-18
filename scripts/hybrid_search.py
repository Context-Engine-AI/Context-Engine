#!/usr/bin/env python3
import os
import sys
import argparse
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

# Ensure /work or repo root is in sys.path for scripts imports
_ROOT_DIR = Path(__file__).resolve().parent.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from qdrant_client import QdrantClient, models

# Use embedder factory for Qwen3 support; fallback to direct fastembed
from typing import TYPE_CHECKING, Any

try:
    from scripts.embedder import get_embedding_model as _get_embedding_model
    _EMBEDDER_FACTORY = True
except ImportError:
    _EMBEDDER_FACTORY = False

# Always try to import TextEmbedding for backward compatibility with tests
try:
    from fastembed import TextEmbedding
except ImportError:
    TextEmbedding = None  # type: ignore

# Type alias for embedding model (TextEmbedding or compatible)
EmbeddingModel = Any if TextEmbedding is None else TextEmbedding
import re
import json
import math

# Prefer orjson for faster serialization (2-3x speedup)
try:
    import orjson
    def _json_dumps(obj):
        return orjson.dumps(obj).decode()
except ImportError:
    orjson = None
    def _json_dumps(obj):
        return json.dumps(obj)

import logging
import threading

# Connection pooling imports
try:
    from scripts.qdrant_client_manager import get_qdrant_client, return_qdrant_client, pooled_qdrant_client
    _POOL_AVAILABLE = True
except ImportError:
    _POOL_AVAILABLE = False
    def get_qdrant_client(url=None, api_key=None, force_new=False, use_pool=True):
        return QdrantClient(url=url or os.environ.get("QDRANT_URL", "http://localhost:6333"),
                           api_key=api_key or os.environ.get("QDRANT_API_KEY"))
    def return_qdrant_client(client):
        pass

# ThreadPoolExecutor for parallel queries (reuse across calls)
_QUERY_EXECUTOR = None
_EXECUTOR_LOCK = threading.Lock()

def _get_query_executor(max_workers: int = 4) -> ThreadPoolExecutor:
    """Get or create a shared ThreadPoolExecutor for parallel queries."""
    global _QUERY_EXECUTOR
    if _QUERY_EXECUTOR is None:
        with _EXECUTOR_LOCK:
            if _QUERY_EXECUTOR is None:
                _QUERY_EXECUTOR = ThreadPoolExecutor(max_workers=max_workers)
    return _QUERY_EXECUTOR

# Filter sanitization cache (avoids repeated deep copies)
_FILTER_CACHE = {}
_FILTER_CACHE_LOCK = threading.Lock()
_FILTER_CACHE_MAX = 256

# Cached regex pattern compilation (avoids recompiling same patterns)
@lru_cache(maxsize=128)
def _compile_regex(pattern: str, flags: int = 0):
    """Cached regex compilation for repeated patterns."""
    return re.compile(pattern, flags)

# Import unified caching system
try:
    from scripts.cache_manager import get_search_cache, get_embedding_cache, get_expansion_cache
    UNIFIED_CACHE_AVAILABLE = True
except ImportError:
    UNIFIED_CACHE_AVAILABLE = False

# Import request deduplication system
try:
    from scripts.deduplication import get_deduplicator, is_duplicate_request
    DEDUPLICATION_AVAILABLE = True
except ImportError:
    DEDUPLICATION_AVAILABLE = False

# Import semantic expansion functionality
try:
    from scripts.semantic_expansion import (
        expand_queries_semantically,
        expand_queries_with_prf,
        get_expansion_stats,
        clear_expansion_cache
    )
    SEMANTIC_EXPANSION_AVAILABLE = True
except ImportError:
    SEMANTIC_EXPANSION_AVAILABLE = False

# Import query optimizer for dynamic EF tuning
try:
    from scripts.query_optimizer import get_query_optimizer, optimize_query
    QUERY_OPTIMIZER_AVAILABLE = True
except ImportError:
    QUERY_OPTIMIZER_AVAILABLE = False

# Aho-Corasick for multi-pattern lexical matching (feature flag: AC_LEXICAL_ENABLED=1)
_AC_LEXICAL_ENABLED = os.environ.get("AC_LEXICAL_ENABLED", "0").lower() in {"1", "true", "yes", "on"}
_AhoCorasick = None

def _get_aho_corasick_class():
    """Lazy-load AhoCorasick class."""
    global _AhoCorasick
    if _AhoCorasick is None and _AC_LEXICAL_ENABLED:
        try:
            from scripts.aho_corasick import AhoCorasick
            _AhoCorasick = AhoCorasick
        except ImportError:
            pass
    return _AhoCorasick

logger = logging.getLogger("hybrid_search")


def _collection(collection_name: str | None = None) -> str:
    """Determine collection name with priority: CLI arg > env > default."""

    if collection_name and collection_name.strip():
        return collection_name.strip()

    env_coll = os.environ.get("COLLECTION_NAME", "").strip()
    if env_coll:
        return env_coll

    return "my-collection"


MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
API_KEY = os.environ.get("QDRANT_API_KEY")


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

LEX_VECTOR_NAME = os.environ.get("LEX_VECTOR_NAME", "lex")
LEX_VECTOR_DIM = _safe_int(os.environ.get("LEX_VECTOR_DIM", "4096"), 4096)
# Optional mini vector (ReFRAG gating)
MINI_VECTOR_NAME = os.environ.get("MINI_VECTOR_NAME", "mini")
MINI_VEC_DIM = _safe_int(os.environ.get("MINI_VEC_DIM", "64"), 64)
HYBRID_MINI_WEIGHT = _safe_float(os.environ.get("HYBRID_MINI_WEIGHT", "0.5"), 0.5)


# Legacy cache compatibility - use unified cache if available
if UNIFIED_CACHE_AVAILABLE:
    _EMBED_CACHE = get_embedding_cache()
    _RESULTS_CACHE = get_search_cache()
    _EXPANSION_CACHE = get_expansion_cache()
    MAX_EMBED_CACHE = int(os.environ.get("MAX_EMBED_CACHE", "8192") or 8192)
    # Lightweight local fallback to ensure deterministic hits in tests
    try:
        from collections import OrderedDict as _OD
    except Exception:
        _OD = dict  # pragma: no cover
    _RESULTS_CACHE_OD = _OD()

    MAX_RESULTS_CACHE = int(os.environ.get("HYBRID_RESULTS_CACHE", "32") or 32)
else:
    # Fallback to original caching system
    from threading import Lock as _Lock
    from collections import OrderedDict

    _EMBED_QUERY_CACHE: OrderedDict[tuple[str, str], List[float]] = OrderedDict()
    _EMBED_LOCK = _Lock()
    MAX_EMBED_CACHE = int(os.environ.get("MAX_EMBED_CACHE", "8192") or 8192)

    _RESULTS_CACHE: OrderedDict[tuple, List[Dict[str, Any]]] = OrderedDict()
    _RESULTS_LOCK = _Lock()
    MAX_RESULTS_CACHE = int(os.environ.get("HYBRID_RESULTS_CACHE", "32") or 32)

# Ensure _RESULTS_LOCK exists regardless of cache backend
try:
    _RESULTS_LOCK
except NameError:
    _RESULTS_LOCK = threading.RLock()


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


def _legacy_vector_search(
    client: QdrantClient,
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


def _embed_queries_cached(
    model: Any, queries: List[str]
) -> List[List[float]]:
    """Cache dense query embeddings to avoid repeated compute across expansions/retries.
    Optimized: batch-embeds all missing queries in one model call (2-5x faster).
    Thread-safe with bounded cache size.

    When Qwen3 is enabled and QWEN3_QUERY_INSTRUCTION=1, applies instruction
    prefix to queries before embedding for improved retrieval quality.
    """
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
        queries = prefix_queries(queries, name)
    except ImportError:
        pass

    if UNIFIED_CACHE_AVAILABLE:
        # Use unified caching system
        missing_queries = []
        missing_indices = []

        # Find missing queries
        for i, q in enumerate(queries):
            key = (str(name), str(q))
            if _EMBED_CACHE.get(key) is None:
                missing_queries.append(str(q))
                missing_indices.append(i)

        # Batch-embed all missing queries in one call
        if missing_queries:
            try:
                vecs = list(model.embed(missing_queries))
                # Cache all new embeddings
                for q, vec in zip(missing_queries, vecs):
                    key = (str(name), str(q))
                    _EMBED_CACHE.set(key, vec.tolist())
            except Exception:
                # Fallback to one-by-one if batch fails
                for q in missing_queries:
                    key = (str(name), str(q))
                    try:
                        vec = next(model.embed([q])).tolist()
                        _EMBED_CACHE.set(key, vec)
                    except Exception:
                        pass

        # Return embeddings in original order from cache
        out: List[List[float]] = []
        for q in queries:
            key = (str(name), str(q))
            v = _EMBED_CACHE.get(key)
            if v is not None:
                out.append(v)
        return out
    else:
        # Fallback to original caching system
        missing_queries = []
        missing_indices = []
        with _EMBED_LOCK:
            for i, q in enumerate(queries):
                key = (str(name), str(q))
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
                        key = (str(name), str(q))
                        if key not in _EMBED_QUERY_CACHE:
                            _EMBED_QUERY_CACHE[key] = vec.tolist()
                            # Evict oldest entries if cache exceeds limit
                            while len(_EMBED_QUERY_CACHE) > MAX_EMBED_CACHE:
                                _EMBED_QUERY_CACHE.popitem(last=False)
            except Exception:
                # Fallback to one-by-one if batch fails
                for q in missing_queries:
                    key = (str(name), str(q))
                    try:
                        vec = next(model.embed([q])).tolist()
                        with _EMBED_LOCK:
                            if key not in _EMBED_QUERY_CACHE:
                                _EMBED_QUERY_CACHE[key] = vec
                                # Evict oldest entries if cache exceeds limit
                                while len(_EMBED_QUERY_CACHE) > MAX_EMBED_CACHE:
                                    _EMBED_QUERY_CACHE.popitem(last=False)
                    except Exception:
                        pass

        # Return embeddings in original order from cache (thread-safe read)
        out: List[List[float]] = []
        with _EMBED_LOCK:
            for q in queries:
                key = (str(name), str(q))
                v = _EMBED_QUERY_CACHE.get(key)
                if v is not None:
                    out.append(v)
        return out


# Ensure project root is on sys.path when run as a script (so 'scripts' package imports work)
import sys
from pathlib import Path as _P

_ROOT = _P(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.utils import sanitize_vector_name as _sanitize_vector_name
from scripts.ingest_code import ensure_collection as _ensure_collection
from scripts.ingest_code import project_mini as _project_mini


RRF_K = _safe_int(os.environ.get("HYBRID_RRF_K", "60"), 60)
DENSE_WEIGHT = _safe_float(os.environ.get("HYBRID_DENSE_WEIGHT", "1.0"), 1.0)
LEXICAL_WEIGHT = _safe_float(os.environ.get("HYBRID_LEXICAL_WEIGHT", "0.25"), 0.25)
LEX_VECTOR_WEIGHT = _safe_float(
    os.environ.get("HYBRID_LEX_VECTOR_WEIGHT", str(LEXICAL_WEIGHT)), LEXICAL_WEIGHT
)
EF_SEARCH = _safe_int(os.environ.get("QDRANT_EF_SEARCH", "128"), 128)
# Lightweight, configurable boosts
SYMBOL_BOOST = _safe_float(os.environ.get("HYBRID_SYMBOL_BOOST", "0.15"), 0.15)
RECENCY_WEIGHT = _safe_float(os.environ.get("HYBRID_RECENCY_WEIGHT", "0.1"), 0.1)
CORE_FILE_BOOST = _safe_float(os.environ.get("HYBRID_CORE_FILE_BOOST", "0.1"), 0.1)
SYMBOL_EQUALITY_BOOST = _safe_float(
    os.environ.get("HYBRID_SYMBOL_EQUALITY_BOOST", "0.25"), 0.25
)
VENDOR_PENALTY = _safe_float(os.environ.get("HYBRID_VENDOR_PENALTY", "0.05"), 0.05)
LANG_MATCH_BOOST = _safe_float(os.environ.get("HYBRID_LANG_MATCH_BOOST", "0.05"), 0.05)
CLUSTER_LINES = _safe_int(os.environ.get("HYBRID_CLUSTER_LINES", "15"), 15)
# Penalize test files slightly to prefer implementation over tests
TEST_FILE_PENALTY = _safe_float(os.environ.get("HYBRID_TEST_FILE_PENALTY", "0.15"), 0.15)

# Additional file-type weighting knobs (defaults tuned for Q&A use)
CONFIG_FILE_PENALTY = _safe_float(os.environ.get("HYBRID_CONFIG_FILE_PENALTY", "0.3"), 0.3)
IMPLEMENTATION_BOOST = _safe_float(os.environ.get("HYBRID_IMPLEMENTATION_BOOST", "0.2"), 0.2)
DOCUMENTATION_PENALTY = _safe_float(os.environ.get("HYBRID_DOCUMENTATION_PENALTY", "0.1"), 0.1)

# Modest boost for matches against pseudo/tags produced at index time.
# Default 0.0 = disabled; set HYBRID_PSEUDO_BOOST>0 to experiment.
PSEUDO_BOOST = _safe_float(os.environ.get("HYBRID_PSEUDO_BOOST", "0.0"), 0.0)

# Penalize comment-heavy snippets so code (not comments) ranks higher
COMMENT_PENALTY = _safe_float(os.environ.get("HYBRID_COMMENT_PENALTY", "0.2"), 0.2)
COMMENT_RATIO_THRESHOLD = _safe_float(os.environ.get("HYBRID_COMMENT_RATIO_THRESHOLD", "0.6"), 0.6)

# Micro-span compaction and budgeting (ReFRAG-lite output shaping)
MICRO_OUT_MAX_SPANS = _safe_int(os.environ.get("MICRO_OUT_MAX_SPANS", "3"), 3)
MICRO_MERGE_LINES = _safe_int(os.environ.get("MICRO_MERGE_LINES", "4"), 4)
MICRO_BUDGET_TOKENS = _safe_int(os.environ.get("MICRO_BUDGET_TOKENS", "512"), 512)
MICRO_TOKENS_PER_LINE = _safe_int(os.environ.get("MICRO_TOKENS_PER_LINE", "32"), 32)

# === Large codebase scaling (default ON) ===
# These parameters enable automatic scaling for collections with 10k+ points
LARGE_COLLECTION_THRESHOLD = _safe_int(os.environ.get("HYBRID_LARGE_THRESHOLD", "10000"), 10000)
MAX_RRF_K_SCALE = _safe_float(os.environ.get("HYBRID_MAX_RRF_K_SCALE", "3.0"), 3.0)
SCORE_NORMALIZE_ENABLED = os.environ.get("HYBRID_SCORE_NORMALIZE", "1").lower() in {"1", "true", "yes", "on"}

# Cache for collection stats (avoid repeated Qdrant calls)
_COLL_STATS_CACHE: Dict[str, Tuple[float, Dict[str, Any]]] = {}
_COLL_STATS_TTL = 300  # 5 minutes

def _get_collection_stats(client: QdrantClient, coll_name: str) -> Dict[str, Any]:
    """Get cached collection statistics for scaling decisions."""
    import time
    now = time.time()
    cached = _COLL_STATS_CACHE.get(coll_name)
    if cached and (now - cached[0]) < _COLL_STATS_TTL:
        return cached[1]
    try:
        info = client.get_collection(coll_name)
        stats = {"points_count": info.points_count or 0}
        _COLL_STATS_CACHE[coll_name] = (now, stats)
        return stats
    except Exception:
        return {"points_count": 0}

def _scale_rrf_k(base_k: int, collection_size: int) -> int:
    """Scale RRF k parameter based on collection size.

    For large collections, increase k to spread score distribution.
    Uses logarithmic scaling: k_scaled = k * (1 + log10(size/threshold))
    Capped at MAX_RRF_K_SCALE * base_k.
    """
    if collection_size < LARGE_COLLECTION_THRESHOLD:
        return base_k
    ratio = collection_size / LARGE_COLLECTION_THRESHOLD
    scale = 1.0 + math.log10(max(1, ratio))
    scale = min(scale, MAX_RRF_K_SCALE)
    return int(base_k * scale)

def _adaptive_per_query(base_limit: int, collection_size: int, has_filters: bool) -> int:
    """Increase candidate retrieval for larger collections.

    Uses sublinear sqrt scaling to avoid excessive retrieval.
    Filters reduce the need for extra candidates.
    """
    if collection_size < LARGE_COLLECTION_THRESHOLD:
        return base_limit
    ratio = collection_size / LARGE_COLLECTION_THRESHOLD
    # sqrt scaling: doubles at 4x threshold, triples at 9x
    scale = math.sqrt(ratio)
    if has_filters:
        scale = max(1.0, scale * 0.7)  # reduce scaling when filters are active
    # Cap at 3x base limit
    scaled = int(base_limit * min(scale, 3.0))
    return max(base_limit, min(scaled, 200))

def _normalize_scores(score_map: Dict[str, Dict[str, Any]], collection_size: int) -> None:
    """Normalize scores using z-score + sigmoid for large collections.

    This spreads compressed score distributions to improve discrimination.
    Only applies when SCORE_NORMALIZE_ENABLED=true and collection is large.
    """
    if not SCORE_NORMALIZE_ENABLED:
        return
    if collection_size < LARGE_COLLECTION_THRESHOLD:
        return
    if len(score_map) < 3:
        return

    scores = [rec["s"] for rec in score_map.values()]
    mean_s = sum(scores) / len(scores)
    var_s = sum((s - mean_s) ** 2 for s in scores) / len(scores)
    std_s = math.sqrt(var_s) if var_s > 0 else 1.0

    if std_s < 1e-6:
        return  # All scores identical, nothing to normalize

    # Z-score normalization + sigmoid to [0, 1] range
    for rec in score_map.values():
        z = (rec["s"] - mean_s) / std_s
        # Sigmoid with scale factor for wider spread
        normalized = 1.0 / (1.0 + math.exp(-z * 0.5))
        rec["s"] = normalized


def _merge_and_budget_spans(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Given ranked items with metadata path/start_line/end_line, merge nearby spans
    per path and enforce a token budget using a simple tokens-per-line estimate.
    Returns a filtered/merged list preserving score order as much as possible.
    """
    # Read dynamic knobs at call-time so tests/env can override without reload
    try:
        merge_lines = int(
            os.environ.get("MICRO_MERGE_LINES", str(MICRO_MERGE_LINES))
            or MICRO_MERGE_LINES
        )
    except (ValueError, TypeError):
        merge_lines = MICRO_MERGE_LINES
    try:
        budget_tokens = int(
            os.environ.get("MICRO_BUDGET_TOKENS", str(MICRO_BUDGET_TOKENS))
            or MICRO_BUDGET_TOKENS
        )
    except (ValueError, TypeError):
        budget_tokens = MICRO_BUDGET_TOKENS
    try:
        tokens_per_line = int(
            os.environ.get("MICRO_TOKENS_PER_LINE", str(MICRO_TOKENS_PER_LINE))
            or MICRO_TOKENS_PER_LINE
        )
    except (ValueError, TypeError):
        tokens_per_line = MICRO_TOKENS_PER_LINE
    try:
        out_max_spans = int(
            os.environ.get("MICRO_OUT_MAX_SPANS", str(MICRO_OUT_MAX_SPANS))
            or MICRO_OUT_MAX_SPANS
        )
    except (ValueError, TypeError):
        out_max_spans = MICRO_OUT_MAX_SPANS

    # First cluster adjacent by path using a tighter merge gap for micro spans
    clusters: Dict[str, List[Dict[str, Any]]] = {}
    for m in items:
        # Robust metadata extraction: support both dict-shaped results and Qdrant point payloads
        md = {}
        try:
            if isinstance(m, dict):
                if m.get("path") or m.get("start_line") or m.get("end_line"):
                    md = {"path": m.get("path"), "start_line": m.get("start_line"), "end_line": m.get("end_line")}
                else:
                    pt = m.get("pt", {})
                    if hasattr(pt, "payload") and getattr(pt, "payload"):
                        md = (pt.payload or {}).get("metadata") or {}
        except Exception:
            md = {}
        path = str((md or {}).get("path") or "")
        start_line = int((md or {}).get("start_line") or 0)
        end_line = int((md or {}).get("end_line") or 0)
        if not path or start_line <= 0 or end_line <= 0:
            # skip invalid entries
            continue
        lst = clusters.setdefault(path, [])
        merged = False
        # Fix: use "raw_score" field (what run_hybrid_search emits) not "s"
        item_score = float(m.get("raw_score") or m.get("score") or m.get("s") or 0.0)
        for c in lst:
            if (
                start_line <= c["end"] + merge_lines
                and end_line >= c["start"] - merge_lines
            ):
                # expand bounds; keep higher-score rep
                cluster_score = float(c["m"].get("raw_score") or c["m"].get("score") or c["m"].get("s") or 0.0)
                if item_score > cluster_score:
                    c["m"] = m
                c["start"] = min(c["start"], start_line)
                c["end"] = max(c["end"], end_line)
                merged = True
                break
        if not merged:
            lst.append({"start": start_line, "end": end_line, "m": m, "p": path})

    # Now budget per path with a global token budget
    budget = budget_tokens
    out: List[Dict[str, Any]] = []
    per_path_counts: Dict[str, int] = {}

    def _line_tokens(s: int, e: int) -> int:
        return max(1, (e - s + 1) * tokens_per_line)

    # Flatten clusters preserving original score order
    flattened = []
    for lst in clusters.values():
        for c in lst:
            flattened.append(c)

    def _flat_key(c):
        m = c.get("m", {})
        # Use stored cluster path and start for stable ordering
        path = str(c.get("p") or "")
        start = int(c.get("start") or 0)
        # Fix: use "raw_score" field (what run_hybrid_search emits) not "s"
        score = float(m.get("raw_score") or m.get("score") or m.get("s") or 0.0)
        # Reranker scores are negative (less negative = better), so sort ascending
        # Positive scores (no reranker) sort descending as before
        if score < 0:
            return (score, path, start)  # ascending for negative scores
        else:
            return (-score, path, start)  # descending for positive scores

    flattened.sort(key=_flat_key)

    for c in flattened:
        m = c["m"]
        # Prefer path from cluster key
        path = str(c.get("p") or "")
        # per-path cap
        if per_path_counts.get(path, 0) >= out_max_spans:
            continue
        need = _line_tokens(c["start"], c["end"])
        if need <= budget:
            budget -= need
            per_path_counts[path] = per_path_counts.get(path, 0) + 1
            # rewrite start/end in the representative's metadata clone for emission
            # (we do not mutate original payloads coming from Qdrant objects)
            out.append(
                {"m": m, "start": c["start"], "end": c["end"], "need_tokens": need}
            )
        elif budget > 0 and per_path_counts.get(path, 0) < out_max_spans:
            # Trim to fit remaining budget instead of dropping entirely
            # Calculate how many lines we can afford with remaining budget
            affordable_lines = max(1, budget // tokens_per_line)
            trim_end = c["start"] + affordable_lines - 1
            if trim_end >= c["start"]:
                trimmed_need = _line_tokens(c["start"], trim_end)
                budget -= trimmed_need
                per_path_counts[path] = per_path_counts.get(path, 0) + 1
                out.append(
                    {"m": m, "start": c["start"], "end": trim_end, "need_tokens": trimmed_need, "_trimmed": True}
                )
        if budget <= 0:
            break

    # Map back to the same structure expected downstream: keep representative m
    # and expose start_line/end_line from our merged span via components
    result: List[Dict[str, Any]] = []
    for c in out:
        m = c["m"]
        # Fix: Update the public start_line/end_line fields to reflect merged bounds
        # so citations and file reads use the expanded range
        m["start_line"] = c["start"]
        m["end_line"] = c["end"]
        # Clear the text field since it no longer matches the merged bounds
        # This forces context_answer to re-read from the file with correct line range
        m["text"] = ""
        # Also keep the internal markers for debugging
        m["_merged_start"] = c["start"]
        m["_merged_end"] = c["end"]
        m["_budget_tokens"] = c["need_tokens"]
        result.append(m)
    return result


# Core file patterns (prioritize implementation over tests/docs)
CORE_FILE_PATTERNS = [
    r"\.py$",
    r"\.js$",
    r"\.ts$",
    r"\.tsx$",
    r"\.jsx$",
    r"\.go$",
    r"\.rs$",
    r"\.java$",
    r"\.cpp$",
    r"\.c$",
    r"\.h$",
]
NON_CORE_PATTERNS = [
    r"test",
    r"spec",
    r"__test__",
    r"\.test\.",
    r"\.spec\.",
    r"_test\.py$",
    r"_spec\.py$",
    r"docs?/",
    r"documentation/",
    r"\.md$",
    r"\.txt$",
    r"README",
    r"CHANGELOG",
]

# Test file patterns
TEST_FILE_PATTERNS = [
    r"/tests?/",
    r"(^|/)test_",
    r"_test\.",
    r"\.test\.",
    r"\.spec\.",
]

def is_test_file(path: str) -> bool:
    import re
    p = path.lower()
    for pattern in TEST_FILE_PATTERNS:
        if re.search(pattern, p):
            return True
    return False


def is_core_file(path: str) -> bool:
    """Check if file is core implementation (not test/doc)"""
    import re

    path_lower = path.lower()
    # Skip non-core files
    for pattern in NON_CORE_PATTERNS:
        if re.search(pattern, path_lower):
            return False
    # Check for core file extensions
    for pattern in CORE_FILE_PATTERNS:
        if re.search(pattern, path_lower):
            return True
    return False


# Vendor/third-party detection
VENDOR_PATTERNS = [
    "vendor/",
    "third_party/",
    "node_modules/",
    "/dist/",
    "/build/",
    ".generated/",
    "generated/",
    "autogen/",
    "target/",
]


def is_vendor_path(path: str) -> bool:
    p = path.lower()
    return any(s in p for s in VENDOR_PATTERNS)


# Language extension mapping and checks
LANG_EXTS: Dict[str, List[str]] = {
    "python": [".py"],
    "typescript": [".ts", ".tsx"],
    "javascript": [".js", ".jsx"],
    "go": [".go"],
    "rust": [".rs"],
    "java": [".java"],
    "cpp": [".cpp", ".cc", ".cxx", ".hpp", ".h"],
    "c": [".c", ".h"],
    "csharp": [".cs", ".csx"],
    "razor": [".cshtml", ".razor"],
    "xml": [".csproj", ".resx", ".config"],
}


def lang_matches_path(lang: str, path: str) -> bool:
    if not lang:
        return False
    exts = LANG_EXTS.get(lang.lower(), [])
    pl = path.lower()
    return any(pl.endswith(ext) for ext in exts)


# --- Query DSL parsing (lang:, file:/path, path:, under:, kind:, symbol:) ---
def parse_query_dsl(queries: List[str]) -> Tuple[List[str], Dict[str, str]]:
    clean: List[str] = []
    extracted: Dict[str, str] = {}
    token_re = re.compile(
        r"\b(?:(lang|language|file|path|under|kind|symbol|ext|not|case|repo))\s*:\s*([^\s]+)",
        re.IGNORECASE,
    )
    for q in queries:
        parts = []
        last = 0
        for m in token_re.finditer(q):
            key = m.group(1).lower()
            val = m.group(2)
            if key in ("file", "path"):
                extracted["under"] = val
            elif key in ("lang", "language"):
                extracted["language"] = val
            elif key in ("ext",):
                extracted["ext"] = val
            elif key in ("not",):
                extracted["not"] = val
            elif key in ("case",):
                extracted["case"] = val
            elif key in ("repo",):
                extracted["repo"] = val
            else:
                extracted[key] = val
            parts.append(q[last : m.start()].strip())
            last = m.end()
        parts.append(q[last:].strip())
        remaining = " ".join([p for p in parts if p])
        if remaining:
            clean.append(remaining)
    # Keep at least an empty query if everything was tokens
    if not clean and queries:
        clean = [""]
    return clean, extracted


# --- Tokenization helpers for smarter lexical ---
_STOP = {
    "the",
    "a",
    "an",
    "of",
    "in",
    "on",
    "for",
    "and",
    "or",
    "to",
    "with",
    "by",
    "is",
    "are",
    "be",
    "this",
    "that",
}


def _split_ident(s: str) -> List[str]:
    # split snake_case and camelCase
    parts = re.split(r"[^A-Za-z0-9]+", s)
    out: List[str] = []
    for p in parts:
        if not p:
            continue
        # camelCase split
        segs = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+", p)
        out.extend([x for x in segs if x])
    return [x.lower() for x in out if x and x.lower() not in _STOP]


def tokenize_queries(phrases: List[str]) -> List[str]:
    toks: List[str] = []
    for ph in phrases:
        toks.extend(_split_ident(ph))
    # de-dup preserving order
    seen = set()
    out: List[str] = []
    for t in toks:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out


# Minimal code-aware query expansion (quick win)
CODE_SYNONYMS = {
    "function": ["method", "def", "fn"],
    "class": ["type", "object"],
    "create": ["init", "initialize", "construct"],
    "get": ["fetch", "retrieve"],
    "set": ["assign", "update"],
}


def expand_queries(
    queries: List[str], language: str | None = None, max_extra: int = 2
) -> List[str]:
    out: List[str] = list(queries)
    for q in list(queries):
        ql = q.lower()
        for word, syns in CODE_SYNONYMS.items():
            if word in ql:
                for s in syns[:max_extra]:
                    exp = re.sub(rf"\b{re.escape(word)}\b", s, q, flags=re.IGNORECASE)
                    if exp not in out:
                        out.append(exp)
    return out[: max(8, len(queries))]


# Enhanced query expansion with semantic similarity
def expand_queries_enhanced(
    queries: List[str],
    language: str | None = None,
    max_extra: int = 2,
    client: QdrantClient | None = None,
    model: Any = None,
    collection: str | None = None
) -> List[str]:
    """
    Enhanced query expansion combining synonym-based and semantic similarity approaches.

    Args:
        queries: Original query strings
        language: Optional programming language hint
        max_extra: Maximum number of additional expansions per query
        client: QdrantClient instance for semantic expansion
        model: Embedding model instance for semantic analysis
        collection: Collection name for semantic expansion

    Returns:
        List of expanded queries
    """
    # Start with original queries
    out: List[str] = list(queries)

    # 1. Apply traditional synonym-based expansion
    synonym_expanded = expand_queries(queries, language, max_extra)
    for q in synonym_expanded:
        if q not in out:
            out.append(q)

    # 2. Apply semantic similarity expansion if available
    if SEMANTIC_EXPANSION_AVAILABLE and client and model:
        try:
            semantic_terms = expand_queries_semantically(
                queries, language, client, model, collection, max_extra
            )

            # Create expanded queries using semantic terms
            for q in list(queries):
                for term in semantic_terms:
                    # Add term as a standalone query
                    if term not in out:
                        out.append(term)

                    # Create combined queries with semantic terms
                    combined = f"{q} {term}"
                    if combined not in out:
                        out.append(combined)

            if os.environ.get("DEBUG_HYBRID_SEARCH"):
                logger.debug(f"Semantic expansion added {len(semantic_terms)} terms: {semantic_terms}")

        except Exception as e:
            if os.environ.get("DEBUG_HYBRID_SEARCH"):
                logger.debug(f"Semantic expansion failed: {e}")

    # Limit total number of queries to prevent explosion
    max_queries = max(8, len(queries) * 3)
    return out[:max_queries]


# --- LLM-assisted expansion (optional if configured) and PRF (default-on) ---
def _llm_expand_queries(
    queries: List[str], language: str | None = None, max_new: int = 4
) -> List[str]:
    """Best-effort LLM expansion with preference for a local runtime (Ollama).
    Providers (by env):
      - LLM_PROVIDER=ollama (preferred if OLLAMA_HOST set; default http://localhost:11434)
      - fallback: OPENAI_API_KEY + LLM_EXPAND_MODEL
    On any error or if not configured, returns []."""
    import json
    import urllib.request

    model = os.environ.get("LLM_EXPAND_MODEL", "glm4")
    prompt = (
        "You are a code search expert. Given one or more short queries, suggest up to "
        f"{max_new} semantically diverse, code-oriented expansions. Only return a JSON list of strings.\n"
        f"Language hint: {language or 'any'}. Queries: {queries}"
    )

    # 1) Prefer local Ollama
    prov = (os.environ.get("LLM_PROVIDER") or "").strip().lower()
    ollama_host = (
        os.environ.get("OLLAMA_HOST", "http://localhost:11434").strip()
        or "http://localhost:11434"
    )
    if prov in {"", "ollama"}:  # default to ollama if reachable
        try:
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0.2, "num_predict": 128},
            }
            req = urllib.request.Request(
                ollama_host.rstrip("/") + "/api/generate",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=8) as resp:
                body = json.loads(resp.read().decode("utf-8", "ignore"))
            txt = body.get("response", "")

            def _parse_list(t: str):
                t = t.strip().strip("`")
                import re, json as _json

                try:
                    v = _json.loads(t)
                    if isinstance(v, list):
                        return v
                except Exception:
                    pass
                if t.startswith("```"):
                    t = t.strip("`")
                m = re.search(r"\[.*?\]", t, flags=re.S)
                if m:
                    try:
                        v = _json.loads(m.group(0))
                        if isinstance(v, list):
                            return v
                    except Exception:
                        pass
                return None

            arr = _parse_list(txt)
            if isinstance(arr, list):
                return [str(x) for x in arr[:max_new] if str(x).strip()]
            out: List[str] = []
            for line in txt.splitlines():
                s = line.strip().strip("- ").strip("`")
                if s:
                    out.append(s)
                if len(out) >= max_new:
                    break
            return out
        except Exception:
            pass

    # 2) Fallback to OpenAI if configured
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return []
    try:
        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
        }
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=json.dumps(data).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=6) as resp:
            body = json.loads(resp.read().decode("utf-8", "ignore"))
        txt = body.get("choices", [{}])[0].get("message", {}).get("content", "")

        def _parse_list(t: str):
            t = t.strip().strip("`")
            import re, json as _json

            try:
                v = _json.loads(t)
                if isinstance(v, list):
                    return v
            except Exception:
                pass
            if t.startswith("```"):
                t = t.strip("`")
            m = re.search(r"\[.*?\]", t, flags=re.S)
            if m:
                try:
                    v = _json.loads(m.group(0))
                    if isinstance(v, list):
                        return v
                except Exception:
                    pass
            return None

        arr = _parse_list(txt)
        if isinstance(arr, list):
            return [str(x) for x in arr[:max_new] if str(x).strip()]
        out = []
        for line in txt.splitlines():
            s = line.strip().strip("- ").strip("`")
            if s:
                out.append(s)
            if len(out) >= max_new:
                break
        return out
    except Exception:
        return []


def _prf_terms_from_results(
    score_map: Dict[str, Dict[str, Any]], top_docs: int = 8, max_terms: int = 6
) -> List[str]:
    """Extract pseudo-relevant feedback terms from top documents' metadata."""
    # Rank by current fused score 's'
    ranked = sorted(score_map.values(), key=lambda r: r.get("s", 0.0), reverse=True)[
        : max(1, top_docs)
    ]
    freq: Dict[str, int] = {}
    for rec in ranked:
        md = (rec.get("pt").payload or {}).get("metadata") or {}
        path = str(md.get("path") or md.get("symbol_path") or md.get("file_path") or "")
        symbol = str(md.get("symbol") or "")
        text = f"{symbol} {path}"
        for tok in tokenize_queries([text]):
            if tok:
                freq[tok] = freq.get(tok, 0) + 1
    # sort by frequency desc
    terms = sorted(freq.items(), key=lambda kv: kv[1], reverse=True)
    return [t for t, _ in terms[: max(1, max_terms)]]


def _env_truthy(val: str | None, default: bool) -> bool:
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def rrf(rank: int, k: int = RRF_K) -> float:
    return 1.0 / (k + rank)


def lexical_score(phrases: List[str], md: Dict[str, Any], token_weights: Dict[str, float] | None = None, bm25_weight: float | None = None) -> float:
    """Smarter lexical: split identifiers, weight matches in symbol/path higher.
    If token_weights provided, apply a small BM25-style multiplicative factor per token:
        factor = 1 + bm25_weight * (w - 1) where w are normalized around 1.0
    """
    tokens = tokenize_queries(phrases)
    if not tokens:
        return 0.0
    path = str(md.get("path", "")).lower()
    path_segs = re.split(r"[/\\]", path)
    sym = str(md.get("symbol", "")).lower()
    symp = str(md.get("symbol_path", "")).lower()
    code = str(md.get("code", ""))[:2000].lower()
    # Optional index-time pseudo/tags enrichment
    pseudo = str(md.get("pseudo") or "").lower()
    tags_val = md.get("tags") or []
    if isinstance(tags_val, list):
        tags_text = " ".join(str(x) for x in tags_val).lower()
    else:
        tags_text = str(tags_val).lower()
    s = 0.0
    for t in tokens:
        if not t:
            continue
        contrib = 0.0
        if t in sym or t in symp:
            contrib += 2.0
        if any(t in seg for seg in path_segs):
            contrib += 0.6
        if t in code:
            contrib += 1.0
        # Pseudo/tags signals: gentle, optional boost
        if PSEUDO_BOOST > 0.0:
            if pseudo and t in pseudo:
                contrib += PSEUDO_BOOST
            if tags_text and t in tags_text:
                contrib += 0.5 * PSEUDO_BOOST
        if contrib > 0 and token_weights and bm25_weight:
            w = float(token_weights.get(t, 1.0) or 1.0)
            contrib *= (1.0 + float(bm25_weight) * (w - 1.0))
        s += contrib
    return s


def lexical_score_ac(
    phrases: List[str],
    md: Dict[str, Any],
    automaton: Any,
    token_weights: Dict[str, float] | None = None,
    bm25_weight: float | None = None,
) -> float:
    """Aho-Corasick optimized lexical scoring.

    Uses pre-built automaton for single-pass matching across all fields.
    Same scoring logic as lexical_score but faster for many tokens.
    """
    if automaton is None:
        return lexical_score(phrases, md, token_weights, bm25_weight)

    path = str(md.get("path", "")).lower()
    path_segs = re.split(r"[/\\]", path)
    sym = str(md.get("symbol", "")).lower()
    symp = str(md.get("symbol_path", "")).lower()
    code = str(md.get("code", ""))[:2000].lower()
    pseudo = str(md.get("pseudo") or "").lower()
    tags_val = md.get("tags") or []
    if isinstance(tags_val, list):
        tags_text = " ".join(str(x) for x in tags_val).lower()
    else:
        tags_text = str(tags_val).lower()

    # Find matches in each field using automaton
    sym_matches = {m.pattern for m in automaton.find_all(sym + " " + symp)}
    path_matches = {m.pattern for seg in path_segs for m in automaton.find_all(seg)}
    code_matches = {m.pattern for m in automaton.find_all(code)}
    pseudo_matches = {m.pattern for m in automaton.find_all(pseudo)} if pseudo else set()
    tags_matches = {m.pattern for m in automaton.find_all(tags_text)} if tags_text else set()

    s = 0.0
    for t in automaton.patterns:
        if not t:
            continue
        contrib = 0.0
        if t in sym_matches:
            contrib += 2.0
        if t in path_matches:
            contrib += 0.6
        if t in code_matches:
            contrib += 1.0
        if PSEUDO_BOOST > 0.0:
            if t in pseudo_matches:
                contrib += PSEUDO_BOOST
            if t in tags_matches:
                contrib += 0.5 * PSEUDO_BOOST
        if contrib > 0 and token_weights and bm25_weight:
            w = float(token_weights.get(t, 1.0) or 1.0)
            contrib *= (1.0 + float(bm25_weight) * (w - 1.0))
        s += contrib
    return s


def build_lexical_automaton_for_search(tokens: List[str]):
    """Build Aho-Corasick automaton from tokens if feature enabled."""
    if not _AC_LEXICAL_ENABLED:
        return None
    AhoCorasick = _get_aho_corasick_class()
    if AhoCorasick is None:
        return None
    # Filter out empty/short tokens
    valid_tokens = [t for t in tokens if t and len(t) >= 2]
    if not valid_tokens:
        return None
    return AhoCorasick(valid_tokens, case_sensitive=False)


# --- Adaptive weighting and MMR diversification helpers ---

def _compute_query_stats(queries: List[str]) -> Dict[str, Any]:
    toks = tokenize_queries(queries)
    total = len(toks)
    def _is_camel(t: str) -> bool:
        try:
            return any(c.isupper() for c in t[1:]) and any(c.islower() for c in t)
        except Exception:
            return False
    def _is_identifier_like(t: str) -> bool:
        try:
            return ("_" in t) or t.isupper() or any(ch.isdigit() for ch in t) or _is_camel(t)
        except Exception:
            return False
    id_like = sum(1 for t in toks if _is_identifier_like(t))
    avg_tok_len = (sum(len(t) for t in toks) / max(1, total)) if total else 0.0
    qchars = sum(len(q) for q in queries) if queries else 0
    has_question = any(("?" in q) for q in (queries or []))
    q0 = (queries[0].strip().lower() if queries else "")
    wh_start = q0.startswith(("how", "what", "why", "when", "where", "explain", "describe"))
    stats = {
        "total_tokens": total,
        "identifier_density": (id_like / max(1, total)),
        "avg_token_len": avg_tok_len,
        "avg_query_chars": (qchars / max(1, len(queries))) if queries else 0.0,
        "narrative_hint": bool(has_question or wh_start),
    }
    return stats


def _adaptive_weights(stats: Dict[str, Any]) -> Tuple[float, float, float]:
    """Return per-query weights (dense_w, lex_vec_w, lex_text_w) with gentle clamps.
    Dense/lex-vector vary within ±25%; lexical text component within ±20%.
    """
    # Base weights
    base_d = DENSE_WEIGHT
    base_lv = LEX_VECTOR_WEIGHT
    base_lx = LEXICAL_WEIGHT

    id_density = float(stats.get("identifier_density", 0.0) or 0.0)
    total = int(stats.get("total_tokens", 0) or 0)
    narrative_hint = 1.0 if stats.get("narrative_hint") else 0.0
    longish = 1.0 if total >= 8 else 0.0

    # Build simple signals
    narrative_score = 0.6 * narrative_hint + 0.4 * longish
    id_score = id_density
    delta = max(-1.0, min(1.0, narrative_score - id_score))

    dens_scale = 1.0 + 0.25 * delta
    lv_scale = 1.0 - 0.25 * delta
    lx_scale = 1.0 + 0.20 * (-delta)  # favor lexical text for identifier-heavy queries

    # Clamp
    dens_scale = max(0.75, min(1.25, dens_scale))
    lv_scale = max(0.75, min(1.25, lv_scale))
    lx_scale = max(0.80, min(1.20, lx_scale))

    return base_d * dens_scale, base_lv * lv_scale, base_lx * lx_scale


def _bm25_token_weights_from_results(phrases: List[str], results: List[Any]) -> Dict[str, float]:
    """Compute lightweight per-token IDF-like weights from a small sample of lex results.
    Returns weights normalized to mean 1.0 over tokens present in phrases.
    """
    try:
        tokens = [t for t in tokenize_queries(phrases) if t]
        if not tokens or not results:
            return {}
        tok_set = set(tokens)
        N = max(1, len(results))
        df: Dict[str, int] = {t: 0 for t in tok_set}
        for p in results:
            try:
                md = (p.payload or {}).get("metadata") or {}
            except Exception:
                md = {}
            text = " ".join(
                [
                    str(md.get("symbol") or ""),
                    str(md.get("symbol_path") or ""),
                    str(md.get("path") or ""),
                    str((md.get("code") or ""))[:2000],
                ]
            ).lower()
            doc_toks = set(tokenize_queries([text]))
            for t in tok_set:
                if t in doc_toks:
                    df[t] += 1
        idf: Dict[str, float] = {t: math.log(1.0 + (N / float(df[t] + 1))) for t in tok_set}
        mean = sum(idf.values()) / max(1, len(idf))
        if mean <= 0:
            return {t: 1.0 for t in tok_set}
        return {t: (idf[t] / mean) for t in tok_set}
    except Exception:
        return {}


def _bm25_token_weights_from_results(phrases: List[str], results: List[Any]) -> Dict[str, float]:
    """Compute lightweight per-token IDF-like weights from a small sample of lex results.
    Returns weights normalized to mean 1.0 over tokens present in phrases.
    """
    try:
        tokens = [t for t in tokenize_queries(phrases) if t]
        if not tokens or not results:
            return {}
        tok_set = set(tokens)
        N = max(1, len(results))
        df: Dict[str, int] = {t: 0 for t in tok_set}
        for p in results:
            try:
                md = (p.payload or {}).get("metadata") or {}
            except Exception:
                md = {}
            text = " ".join(
                [
                    str(md.get("symbol") or ""),
                    str(md.get("symbol_path") or ""),
                    str(md.get("path") or ""),
                    str((md.get("code") or ""))[:2000],
                ]
            ).lower()
            doc_toks = set(tokenize_queries([text]))
            for t in tok_set:
                if t in doc_toks:
                    df[t] += 1
        idf: Dict[str, float] = {t: math.log(1.0 + (N / float(df[t] + 1))) for t in tok_set}
        mean = sum(idf.values()) / max(1, len(idf))
        if mean <= 0:
            return {t: 1.0 for t in tok_set}
        return {t: (idf[t] / mean) for t in tok_set}
    except Exception:
        return {}

def _mmr_diversify(ranked: List[Dict[str, Any]], k: int = 60, lambda_: float = 0.7) -> List[Dict[str, Any]]:
    """Maximal Marginal Relevance over fused list.
    Preserves top-1 by relevance, then balances relevance vs. diversity by path/symbol.
    Returns a reordered list (top-k diversified, remainder appended in original order).
    """
    if not ranked:
        return []
    k = max(1, min(int(k or 1), len(ranked)))

    def _path(md: Dict[str, Any]) -> str:
        return str(md.get("path") or "")

    def _symp(md: Dict[str, Any]) -> str:
        return str(md.get("symbol_path") or md.get("symbol") or "")

    def _sim(a: Dict[str, Any], b: Dict[str, Any]) -> float:
        mda = (a["pt"].payload or {}).get("metadata") or {}
        mdb = (b["pt"].payload or {}).get("metadata") or {}
        pa, pb = _path(mda), _path(mdb)
        if pa and pb and pa == pb:
            return 1.0
        sa, sb = _symp(mda), _symp(mdb)
        if sa and sb and sa == sb:
            return 0.8
        if pa and pb:
            ta = set(re.split(r"[/\\]+", pa.lower()))
            tb = set(re.split(r"[/\\]+", pb.lower()))
            ta.discard(""); tb.discard("")
            if ta and tb:
                inter = len(ta & tb)
                union = max(1, len(ta | tb))
                return 0.5 * (inter / union)
        return 0.0

    rel = [float(m.get("s", 0.0)) for m in ranked]
    selected_idx = [0]  # preserve top-1
    candidates = list(range(1, len(ranked)))
    while len(selected_idx) < k and candidates:
        best_idx = None
        best_score = -1e18
        for i in candidates:
            # relevance
            r = rel[i]
            # max similarity to already selected
            if selected_idx:
                max_sim = max(_sim(ranked[i], ranked[j]) for j in selected_idx)
            else:
                max_sim = 0.0
            mmr = lambda_ * r - (1.0 - lambda_) * max_sim
            if mmr > best_score:
                best_score = mmr
                best_idx = i
        selected_idx.append(best_idx)
        candidates.remove(best_idx)

    # Build diversified list: selected top-k in chosen order, then remaining by original order
    sel_set = set(selected_idx)
    diversified = [ranked[i] for i in selected_idx]
    diversified.extend([ranked[i] for i in range(len(ranked)) if i not in sel_set])
    return diversified


# --- Lexical vector (hashing trick) for server-side hybrid ---
def _split_ident_lex(s: str) -> List[str]:
    parts = re.split(r"[^A-Za-z0-9]+", s)
    out: List[str] = []
    for p in parts:
        if not p:
            continue
        segs = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+", p)
        out.extend([x for x in segs if x])
    return [x.lower() for x in out if x and x.lower() not in _STOP]


from scripts.utils import lex_hash_vector_queries as _lex_hash_vector_queries


def lex_hash_vector(phrases: List[str], dim: int = LEX_VECTOR_DIM) -> List[float]:
    return _lex_hash_vector_queries(phrases, dim)

# Defensive: sanitize Qdrant filter objects so we never send an empty filter {}
# Qdrant returns 400 if filter has no conditions; return None in that case.
# Uses caching for repeated filter patterns to avoid redundant validation.
def _sanitize_filter_obj(flt):
    if flt is None:
        return None

    # Try cache first (hash by id for object identity)
    cache_key = id(flt)
    with _FILTER_CACHE_LOCK:
        if cache_key in _FILTER_CACHE:
            return _FILTER_CACHE[cache_key]

    try:
        # Try model-style attributes first
        must = getattr(flt, "must", None)
        should = getattr(flt, "should", None)
        must_not = getattr(flt, "must_not", None)
        if must is None and should is None and must_not is None:
            # Maybe dict-like
            if isinstance(flt, dict):
                m = [c for c in (flt.get("must") or []) if c is not None]
                s = [c for c in (flt.get("should") or []) if c is not None]
                mn = [c for c in (flt.get("must_not") or []) if c is not None]
                result = None if (not m and not s and not mn) else flt
            else:
                # Unknown structure -> drop
                result = None
        else:
            m = [c for c in (must or []) if c is not None]
            s = [c for c in (should or []) if c is not None]
            mn = [c for c in (must_not or []) if c is not None]
            result = None if (not m and not s and not mn) else flt
    except Exception:
        result = None

    # Cache result (with size limit)
    with _FILTER_CACHE_LOCK:
        if len(_FILTER_CACHE) < _FILTER_CACHE_MAX:
            _FILTER_CACHE[cache_key] = result

    return result


def lex_query(client: QdrantClient, v: List[float], flt, per_query: int, collection_name: str | None = None) -> List[Any]:
    ef = max(EF_SEARCH, 32 + 4 * int(per_query))
    flt = _sanitize_filter_obj(flt)
    collection = _collection(collection_name)

    # Prefer modern API; handle kwarg rename between client versions (query_filter -> filter)
    try:
        qp = client.query_points(
            collection_name=collection,
            query=v,
            using=LEX_VECTOR_NAME,
            query_filter=flt,
            search_params=models.SearchParams(hnsw_ef=ef),
            limit=per_query,
            with_payload=True,
        )
        return _coerce_points(getattr(qp, "points", qp))
    except TypeError:
        # Older/newer client may expect 'filter' kw
        if os.environ.get("DEBUG_HYBRID_SEARCH"):
            logger.debug("QP_FILTER_KWARG_SWITCH", extra={"using": LEX_VECTOR_NAME})
        qp = client.query_points(
            collection_name=collection,
            query=v,
            using=LEX_VECTOR_NAME,
            filter=flt,
            search_params=models.SearchParams(hnsw_ef=ef),
            limit=per_query,
            with_payload=True,
        )
        return _coerce_points(getattr(qp, "points", qp))
    except AttributeError:
        return _legacy_vector_search(client, collection, LEX_VECTOR_NAME, v, per_query, flt)
    except Exception as e:
        # Retry without a filter at all (handles servers that reject certain filter shapes)
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
                search_params=models.SearchParams(hnsw_ef=ef),
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
                search_params=models.SearchParams(hnsw_ef=ef),
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


def dense_query(
    client: QdrantClient, vec_name: str, v: List[float], flt, per_query: int, collection_name: str | None = None, query_text: str | None = None
) -> List[Any]:
    ef = max(EF_SEARCH, 32 + 4 * int(per_query))
    
    # Apply dynamic EF optimization if query text provided
    if QUERY_OPTIMIZER_AVAILABLE and query_text and os.environ.get("QUERY_OPTIMIZER_ADAPTIVE", "1") == "1":
        try:
            result = optimize_query(query_text)
            ef = result["recommended_ef"]
            if os.environ.get("DEBUG_HYBRID_SEARCH"):
                logger.debug(f"Dynamic EF: {ef} (complexity={result['complexity']}, type={result['query_type']})")
        except Exception as e:
            if os.environ.get("DEBUG_HYBRID_SEARCH"):
                logger.debug(f"Query optimizer failed, using default EF: {e}")
    
    flt = _sanitize_filter_obj(flt)
    collection = _collection(collection_name)

    try:
        qp = client.query_points(
            collection_name=collection,
            query=v,
            using=vec_name,
            query_filter=flt,
            search_params=models.SearchParams(hnsw_ef=ef),
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
            search_params=models.SearchParams(hnsw_ef=ef),
            limit=per_query,
            with_payload=True,
        )
        return _coerce_points(getattr(qp, "points", qp))
    except Exception as e:
        # Retry without any filter to maximize compatibility across server/client versions
        if os.environ.get("DEBUG_HYBRID_SEARCH"):
            try:
                logger.debug("QP_FILTER_DROP", extra={"using": vec_name, "reason": str(e)[:200]})
            except Exception:
                pass
        try:
            qp = client.query_points(
                collection_name=_collection(),
                query=v,
                using=vec_name,
                query_filter=None,
                search_params=models.SearchParams(hnsw_ef=ef),
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
                    search_params=models.SearchParams(hnsw_ef=ef),
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


# In-process API: run hybrid search and return structured items list
# Optional: pass an existing embedding model instance via model to reuse cache
# Optional: pass mode to adjust implementation/docs weighting (code_first/balanced/docs_first)

def run_hybrid_search(
    queries: List[str],
    limit: int = 10,
    per_path: int = 1,
    language: str | None = None,
    under: str | None = None,
    kind: str | None = None,
    symbol: str | None = None,
    ext: str | None = None,
    not_filter: str | None = None,
    case: str | None = None,
    path_regex: str | None = None,
    path_glob: str | list[str] | None = None,
    not_glob: str | list[str] | None = None,
    expand: bool = True,
    model: Any = None,
    collection: str | None = None,
    mode: str | None = None,
    repo: str | list[str] | None = None,  # Filter by repo name(s); "*" to disable auto-filter
) -> List[Dict[str, Any]]:
    client = QdrantClient(url=os.environ.get("QDRANT_URL", QDRANT_URL), api_key=API_KEY)
    model_name = os.environ.get("EMBEDDING_MODEL", MODEL_NAME)
    if model:
        _model = model
    elif _EMBEDDER_FACTORY:
        _model = _get_embedding_model(model_name)
    else:
        _model = TextEmbedding(model_name=model_name)
    vec_name = _sanitize_vector_name(model_name)

    # Parse Query DSL and merge with explicit args
    raw_queries = list(queries)
    clean_queries, dsl = parse_query_dsl(raw_queries)
    eff_language = language or dsl.get("language")
    eff_under = under or dsl.get("under")
    eff_kind = kind or dsl.get("kind")
    eff_symbol = symbol or dsl.get("symbol")
    eff_ext = ext or dsl.get("ext")
    eff_not = not_filter or dsl.get("not")
    eff_case = case or dsl.get("case") or os.environ.get("HYBRID_CASE", "insensitive")
    # Repo filter: explicit param > DSL > auto-detect from env
    eff_repo = repo or dsl.get("repo")
    # Normalize repo to list for multi-repo support
    if eff_repo and isinstance(eff_repo, str):
        if eff_repo.strip() == "*":
            eff_repo = None  # "*" means search all repos
        else:
            eff_repo = [r.strip() for r in eff_repo.split(",") if r.strip()]
    elif eff_repo and isinstance(eff_repo, (list, tuple)):
        eff_repo = [str(r).strip() for r in eff_repo if str(r).strip() and str(r).strip() != "*"]
        if not eff_repo:
            eff_repo = None
    # Auto-detect repo from env if not specified and auto-filter is enabled
    if eff_repo is None and str(os.environ.get("REPO_AUTO_FILTER", "1")).strip().lower() in {"1", "true", "yes", "on"}:
        auto_repo = os.environ.get("CURRENT_REPO") or os.environ.get("REPO_NAME")
        if auto_repo and auto_repo.strip():
            eff_repo = [auto_repo.strip()]
    eff_path_regex = path_regex

    def _to_list(x):
        if x is None:
            return []
        if isinstance(x, (list, tuple)):
            out = []
            for e in x:
                s = str(e).strip()
                if s:
                    out.append(s)
            return out
        s = str(x).strip()
        return [s] if s else []

    eff_path_globs = _to_list(path_glob)
    eff_not_globs = _to_list(not_glob)

    # Normalize glob patterns: allow repo-relative (e.g., "src/*.py") to match
    # stored absolute paths (e.g., "/work/src/..."). We keep both original and
    # absolute-prefixed variants for matching.
    def _normalize_globs(globs: list[str]) -> list[str]:
        out: list[str] = []
        try:
            for g in (globs or []):
                s = str(g).strip().replace("\\", "/")
                if not s:
                    continue
                out.append(s)
                if not s.startswith("/"):
                    out.append("/work/" + s.lstrip("/"))
        except Exception:
            pass
        # Dedup while preserving order
        seen = set()
        dedup: list[str] = []
        for g in out:
            if g not in seen:
                dedup.append(g)
                seen.add(g)
        return dedup

    eff_path_globs_norm = _normalize_globs(eff_path_globs)
    eff_not_globs_norm = _normalize_globs(eff_not_globs)

    # Normalize under
    def _norm_under(u: str | None) -> str | None:
        if not u:
            return None
        u = str(u).strip().replace("\\", "/")
        u = "/".join([p for p in u.split("/") if p])
        if not u:
            return None
        if not u.startswith("/"):
            v = "/work/" + u
        else:
            v = "/work/" + u.lstrip("/") if not u.startswith("/work/") else u
        return v

    eff_under = _norm_under(eff_under)

    # Results cache: return cached results for identical (queries, filters, knobs)
    _USE_CACHE = (MAX_RESULTS_CACHE > 0) and _env_truthy(os.environ.get("HYBRID_RESULTS_CACHE_ENABLED"), True)
    cache_key = None
    if _USE_CACHE:
        try:
            cache_key = (
                "v1",
                tuple(clean_queries),
                int(limit or 0),
                int(per_path or 0),
                str(eff_language or ""),
                str(eff_under or ""),
                str(eff_kind or ""),
                str(eff_symbol or ""),
                str(eff_ext or ""),
                str(eff_not or ""),
                str(eff_case or ""),
                tuple(eff_path_globs_norm or ()),
                tuple(eff_not_globs_norm or ()),
                str(eff_repo or ""),
                str(eff_path_regex or ""),
                bool(expand),
                str(vec_name),
                str(_collection()),
                _env_truthy(os.environ.get("HYBRID_ADAPTIVE_WEIGHTS"), True),
                _env_truthy(os.environ.get("HYBRID_MMR"), True),
                str(mode or ""),
            )
        except Exception:
            cache_key = None
        if cache_key is not None:
            if UNIFIED_CACHE_AVAILABLE:
                val = _RESULTS_CACHE.get(cache_key)
                if val is not None:
                    if os.environ.get("DEBUG_HYBRID_SEARCH"):
                        logger.debug("cache hit for hybrid results (unified)")
                    return val
                # Fallback to local in-process dict to ensure deterministic hits (esp. in unit tests)
                try:
                    with _RESULTS_LOCK:
                        if cache_key in _RESULTS_CACHE_OD:
                            if os.environ.get("DEBUG_HYBRID_SEARCH"):
                                logger.debug("cache hit for hybrid results (fallback OD)")
                            return _RESULTS_CACHE_OD[cache_key]
                except Exception:
                    pass
            else:
                with _RESULTS_LOCK:
                    if cache_key in _RESULTS_CACHE:
                        val = _RESULTS_CACHE.pop(cache_key)
                        _RESULTS_CACHE[cache_key] = val
                        if os.environ.get("DEBUG_HYBRID_SEARCH"):
                            logger.debug("cache hit for hybrid results (legacy)")
                        return val

    # Build optional filter
    flt = None
    must = []
    if eff_language:
        must.append(
            models.FieldCondition(
                key="metadata.language", match=models.MatchValue(value=eff_language)
            )
        )
    # Repo filter: supports single repo or list of repos (for related codebases)
    if eff_repo:
        if isinstance(eff_repo, list) and len(eff_repo) == 1:
            must.append(
                models.FieldCondition(
                    key="metadata.repo", match=models.MatchValue(value=eff_repo[0])
                )
            )
        elif isinstance(eff_repo, list) and len(eff_repo) > 1:
            # Multiple repos: use MatchAny for OR logic
            must.append(
                models.FieldCondition(
                    key="metadata.repo", match=models.MatchAny(any=eff_repo)
                )
            )
        elif isinstance(eff_repo, str):
            must.append(
                models.FieldCondition(
                    key="metadata.repo", match=models.MatchValue(value=eff_repo)
                )
            )
    if eff_under:
        must.append(
            models.FieldCondition(
                key="metadata.path_prefix", match=models.MatchValue(value=eff_under)
            )
        )
    if eff_kind:
        must.append(
            models.FieldCondition(
                key="metadata.kind", match=models.MatchValue(value=eff_kind)
            )
        )
    if eff_symbol:
        must.append(
            models.FieldCondition(
                key="metadata.symbol", match=models.MatchValue(value=eff_symbol)
            )
        )

    # After attempting cache get, run deduplication. If duplicate, serve cached result if present.
    if DEDUPLICATION_AVAILABLE:
        request_data = {
            'queries': queries,
            'limit': limit,
            'per_path': per_path,
            'language': language,
            'under': under,
            'kind': kind,
            'symbol': symbol,
            'ext': ext,
            'not': not_filter,
            'case': case,
            'path_regex': path_regex,
            'path_glob': path_glob,
            'not_glob': not_glob,
            'expand': expand,
            'collection': _collection(),
            'vector_name': vec_name,
            'mode': mode,
        }
        is_duplicate, similar_fp = is_duplicate_request(request_data)
        if is_duplicate:
            # Prefer serving from cache on duplicate
            if cache_key is not None:
                if UNIFIED_CACHE_AVAILABLE:
                    val = _RESULTS_CACHE.get(cache_key)
                    if val is not None:
                        if os.environ.get("DEBUG_HYBRID_SEARCH"):
                            logger.debug("duplicate served from cache (unified)")
                        return val
                    try:
                        with _RESULTS_LOCK:
                            if cache_key in _RESULTS_CACHE_OD:
                                if os.environ.get("DEBUG_HYBRID_SEARCH"):
                                    logger.debug("duplicate served from cache (fallback OD)")
                                return _RESULTS_CACHE_OD[cache_key]
                    except Exception:
                        pass
                else:
                    with _RESULTS_LOCK:
                        if cache_key in _RESULTS_CACHE:
                            val = _RESULTS_CACHE.pop(cache_key)
                            _RESULTS_CACHE[cache_key] = val
                            if os.environ.get("DEBUG_HYBRID_SEARCH"):
                                logger.debug("duplicate served from cache (legacy)")
                            return val
            if os.environ.get("DEBUG_HYBRID_SEARCH"):
                logger.debug("Duplicate without cache; bypassing dedup and continuing search")


    # Add ext filter (file extension) - server-side when possible
    if eff_ext:
        ext_clean = eff_ext.lower().lstrip(".")
        must.append(
            models.FieldCondition(
                key="metadata.ext", match=models.MatchValue(value=ext_clean)
            )
        )

    # Add not filter (simple text exclusion on path) - server-side when possible
    must_not = []
    if eff_not:
        # Try MatchText for substring exclusion; fallback to post-filter if unsupported
        try:
            must_not.append(
                models.FieldCondition(
                    key="metadata.path", match=models.MatchText(text=eff_not)
                )
            )
        except Exception:
            # Will be handled by post-filter
            pass

    flt = models.Filter(must=must, must_not=must_not) if (must or must_not) else None
    flt = _sanitize_filter_obj(flt)


    # Build query list (LLM-assisted first, then synonym expansion)
    qlist = list(clean_queries)
    try:
        llm_max = int(os.environ.get("LLM_EXPAND_MAX", "0") or 0)
    except (ValueError, TypeError):
        llm_max = 4
    _llm_more = _llm_expand_queries(qlist, eff_language, max_new=llm_max)
    for s in _llm_more:
        if s and s not in qlist:
            qlist.append(s)
    if expand:
        # Use enhanced expansion with semantic similarity if available
        if SEMANTIC_EXPANSION_AVAILABLE:
            qlist = expand_queries_enhanced(
                qlist, eff_language,
                max_extra=max(2, int(os.environ.get("SEMANTIC_EXPANSION_MAX_TERMS", "3") or "3")),
                client=client,
                model=_model,
                collection=_collection()
            )
        else:
            qlist = expand_queries(qlist, eff_language)

    # Query sharpening: derive basename tokens from path_glob to steer retrieval/gating
    try:
        if eff_path_globs or eff_path_globs_norm:
            def _bn(p: str) -> str:
                s = str(p or "").replace("\\", "/").strip()
                # drop any trailing slashes and take last segment
                parts = [t for t in s.split("/") if t]
                return parts[-1] if parts else ""
            globs_src = list(eff_path_globs or []) + list(eff_path_globs_norm or [])
            basenames = []
            for g in globs_src:
                b = _bn(g)
                if b and b not in basenames:
                    basenames.append(b)
            for b in basenames:
                if b and b not in qlist:
                    qlist.append(b)
                # also add stem (filename without extension) as a lexical hint
                stem = b.rsplit(".", 1)[0] if "." in b else b
                if stem and stem not in qlist:
                    qlist.append(stem)
            # Add short path segments (e.g., "scripts/hybrid_search") to steer lexical hashing
            for g in globs_src:
                s = str(g or "").replace("\\", "/").strip()
                parts = [t for t in s.split("/") if t]
                if len(parts) >= 2:
                    last2 = "/".join(parts[-2:])
                    if last2 and last2 not in qlist:
                        qlist.append(last2)
                if len(parts) >= 3:
                    last3 = "/".join(parts[-3:])
                    if last3 and last3 not in qlist:
                        qlist.append(last3)
    except Exception:
        pass

    # === Large codebase scaling (automatic) ===
    _coll_stats = _get_collection_stats(client, _collection(collection))
    _coll_size = _coll_stats.get("points_count", 0)
    _has_filters = bool(eff_language or eff_repo or eff_under or eff_kind or eff_symbol or eff_ext)

    # Scale RRF k for better score discrimination at scale
    _scaled_rrf_k = _scale_rrf_k(RRF_K, _coll_size)

    # Adaptive per_query: retrieve more candidates from larger collections
    _scaled_per_query = _adaptive_per_query(max(24, limit), _coll_size, _has_filters)

    if os.environ.get("DEBUG_HYBRID_SEARCH") and _coll_size >= LARGE_COLLECTION_THRESHOLD:
        logger.debug(f"Large collection scaling: size={_coll_size}, rrf_k={_scaled_rrf_k}, per_query={_scaled_per_query}")

    # Local RRF function using scaled k
    def _scaled_rrf(rank: int) -> float:
        return 1.0 / (_scaled_rrf_k + rank)

    # Lexical vector query (with scaled retrieval)
    score_map: Dict[str, Dict[str, Any]] = {}
    try:
        lex_vec = lex_hash_vector(qlist)
        lex_results = lex_query(client, lex_vec, flt, _scaled_per_query, collection)
    except Exception:
        lex_results = []

    # Per-query adaptive weights (default ON, gentle clamps)
    _USE_ADAPT = _env_truthy(os.environ.get("HYBRID_ADAPTIVE_WEIGHTS"), True)
    if _USE_ADAPT:
        try:
            _AD_DENSE_W, _AD_LEX_VEC_W, _AD_LEX_TEXT_W = _adaptive_weights(_compute_query_stats(qlist))
        except Exception:
            _AD_DENSE_W, _AD_LEX_VEC_W, _AD_LEX_TEXT_W = DENSE_WEIGHT, LEX_VECTOR_WEIGHT, LEXICAL_WEIGHT
    else:
        _AD_DENSE_W, _AD_LEX_VEC_W, _AD_LEX_TEXT_W = DENSE_WEIGHT, LEX_VECTOR_WEIGHT, LEXICAL_WEIGHT

    for rank, p in enumerate(lex_results, 1):
        pid = str(p.id)
        score_map.setdefault(
            pid,
            {
                "pt": p,
                "s": 0.0,
                "d": 0.0,
                "lx": 0.0,
                "sym_sub": 0.0,
                "sym_eq": 0.0,
                "core": 0.0,
                "vendor": 0.0,
                "langb": 0.0,
                "rec": 0.0,
                "test": 0.0,
            },
        )
        lxs = (_AD_LEX_VEC_W * _scaled_rrf(rank)) if _USE_ADAPT else (LEX_VECTOR_WEIGHT * _scaled_rrf(rank))
        score_map[pid]["lx"] += lxs
        score_map[pid]["s"] += lxs

    # Dense queries
    embedded = _embed_queries_cached(_model, qlist)
    # Ensure collection schema is compatible with current search settings (named vectors)
    try:
        if embedded:
            dim = len(embedded[0])
            _ensure_collection(client, _collection(collection), dim, vec_name)
    except Exception:
        pass
    # Optional gate-first using mini vectors to restrict dense search to candidates
    # Adaptive gating: disable for short/ambiguous queries to avoid over-filtering
    flt_gated = flt
    try:
        gate_first = str(os.environ.get("REFRAG_GATE_FIRST", "0")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        refrag_on = str(os.environ.get("REFRAG_MODE", "")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        cand_n = int(os.environ.get("REFRAG_CANDIDATES", "200") or 200)
    except (ValueError, TypeError):
        gate_first, refrag_on, cand_n = False, False, 200

    # Adaptive mini-gate: disable for queries that are too short or lack strong identifiers
    should_bypass_gate = False
    if gate_first and refrag_on:
        # Check query characteristics
        try:
            # Count total tokens across queries
            total_tokens = sum(len(_split_ident(q)) for q in qlist)
            # Check for strong identifiers (ALL_CAPS, camelCase, or has underscore)
            has_strong_id = any(
                any(t.isupper() or "_" in t or any(c.isupper() for c in t[1:]) and any(c.islower() for c in t)
                    for t in _split_ident(q))
                for q in qlist
            )
            # If query is very short (<3 tokens) and has no strong identifiers, bypass gate
            if total_tokens < 3 and not has_strong_id:
                should_bypass_gate = True
                if os.environ.get("DEBUG_HYBRID_SEARCH"):
                    logger.debug(f"Adaptive gate bypass (short query, tokens={total_tokens}, strong_id={has_strong_id})")
            # If we have strict filters (language, under, symbol, ext), relax candidate count
            if eff_language or eff_under or eff_symbol or eff_ext:
                cand_n = max(cand_n, limit * 5)
                if os.environ.get("DEBUG_HYBRID_SEARCH"):
                    logger.debug(f"Adaptive gate relaxed candidate count to {cand_n} due to filters")
        except Exception:
            pass

    if gate_first and refrag_on and not should_bypass_gate:
        try:
            # ReFRAG gate-first: Use MINI vectors to prefilter candidates
            mini_queries = [_project_mini(list(v), MINI_VEC_DIM) for v in embedded]

            # Get top candidates using MINI vectors (fast prefilter)
            candidate_ids = set()
            for mv in mini_queries:
                mini_results = dense_query(client, MINI_VECTOR_NAME, mv, flt, cand_n, collection)
                for result in mini_results:
                    if hasattr(result, 'id'):
                        candidate_ids.add(result.id)

            if candidate_ids:
                # Server-side gating without requiring payload fields: prefer HasIdCondition
                from qdrant_client import models as _models
                try:
                    gating_cond = _models.HasIdCondition(has_id=list(candidate_ids))
                    gating_kind = "has_id"
                except Exception:
                    # Fallback to pid_str if HasIdCondition unavailable
                    id_vals = [str(cid) for cid in candidate_ids]
                    gating_cond = _models.FieldCondition(
                        key="pid_str",
                        match=_models.MatchAny(any=id_vals),
                    )
                    gating_kind = "pid_str"
                if flt is None:
                    flt_gated = _models.Filter(must=[gating_cond])
                else:
                    must = list(flt.must or [])
                    must.append(gating_cond)
                    flt_gated = _models.Filter(must=must, should=flt.should, must_not=flt.must_not)
                if os.environ.get("DEBUG_HYBRID_SEARCH"):
                    logger.debug(f"ReFRAG gate-first (server-side-{gating_kind}): {len(candidate_ids)} candidates")
                    logger.debug(f"flt_gated.must has {len(flt_gated.must or [])} conditions")
                    logger.debug(f"flt_gated.must_not has {len(flt_gated.must_not or [])} conditions")
            else:
                # No candidates -> no gating
                flt_gated = flt
        except Exception as e:
            if os.environ.get("DEBUG_HYBRID_SEARCH"):
                logger.debug(f"ReFRAG gate-first failed: {e}, proceeding without gating")
            # Fallback to normal search (no gating)
            flt_gated = flt
    else:
        flt_gated = flt

    # Sanitize filter: if empty, drop it to avoid Qdrant 400s on invalid filters
    try:
        if flt_gated is not None:
            _m = [c for c in (getattr(flt_gated, "must", None) or []) if c is not None]
            _s = [c for c in (getattr(flt_gated, "should", None) or []) if c is not None]
            _mn = [c for c in (getattr(flt_gated, "must_not", None) or []) if c is not None]
            if not _m and not _s and not _mn:
                flt_gated = None
    except Exception:
        pass

    flt_gated = _sanitize_filter_obj(flt_gated)

    # Parallel dense query execution for multiple queries
    if len(embedded) > 1 and os.environ.get("PARALLEL_DENSE_QUERIES", "1") == "1":
        executor = _get_query_executor()
        futures = [
            executor.submit(
                dense_query,
                client,
                vec_name,
                v,
                flt_gated,
                _scaled_per_query,
                collection,
                queries[i] if i < len(queries) else None,
            )
            for i, v in enumerate(embedded)
        ]
        result_sets: List[List[Any]] = [f.result() for f in futures]
    else:
        result_sets: List[List[Any]] = [
            dense_query(
                client,
                vec_name,
                v,
                flt_gated,
                _scaled_per_query,
                collection,
                query_text=queries[i] if i < len(queries) else None,
            )
            for i, v in enumerate(embedded)
        ]
    if os.environ.get("DEBUG_HYBRID_SEARCH"):
        total_dense_results = sum(len(rs) for rs in result_sets)
        logger.debug(f"Dense query returned {total_dense_results} total results across {len(result_sets)} queries")

    # Optional ReFRAG-style mini-vector gating: add compact-vector RRF if enabled
    try:
        if os.environ.get("REFRAG_MODE", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            try:
                mini_queries = [_project_mini(list(v), MINI_VEC_DIM) for v in embedded]
                mini_sets: List[List[Any]] = [
                    dense_query(client, MINI_VECTOR_NAME, mv, flt, _scaled_per_query, collection)
                    for mv in mini_queries
                ]
                for res in mini_sets:
                    for rank, p in enumerate(res, 1):
                        pid = str(p.id)
                        score_map.setdefault(
                            pid,
                            {
                                "pt": p,
                                "s": 0.0,
                                "d": 0.0,
                                "lx": 0.0,
                                "sym_sub": 0.0,
                                "sym_eq": 0.0,
                                "core": 0.0,
                                "vendor": 0.0,
                                "langb": 0.0,
                                "rec": 0.0,
                                "test": 0.0,
                            },
                        )
                        dens = float(HYBRID_MINI_WEIGHT) * _scaled_rrf(rank)
                        score_map[pid]["d"] += dens
                        score_map[pid]["s"] += dens
            except Exception:
                pass
    except Exception:
        pass

    # Enhanced PRF with semantic similarity
    if SEMANTIC_EXPANSION_AVAILABLE and score_map:
        # Local PRF dense weight (fallback if not set later)
        try:
            prf_dw = float(os.environ.get("PRF_DENSE_WEIGHT", "0.4") or 0.4)
        except Exception:
            prf_dw = 0.4

        try:
            # Get top results for PRF context
            top_results = []
            for pid, rec in score_map.items():
                top_results.append(rec["pt"])
                if len(top_results) >= 8:  # Use top 8 for PRF
                    break

            if top_results:
                semantic_prf_terms = expand_queries_with_prf(
                    clean_queries, top_results, _model, max_terms=4
                )

                # Create PRF queries using semantic terms
                semantic_prf_qs = []
                for term in semantic_prf_terms:
                    base = clean_queries[0] if clean_queries else (qlist[0] if qlist else "")
                    cand = (base + " " + term).strip()
                    if cand and cand not in qlist and cand not in semantic_prf_qs:
                        semantic_prf_qs.append(cand)
                        if len(semantic_prf_qs) >= 3:  # Limit semantic PRF queries
                            break

                if semantic_prf_qs:
                    # Dense semantic PRF pass
                    embedded_sem_prf = _embed_queries_cached(_model, semantic_prf_qs)
                    result_sets_sem_prf: List[List[Any]] = [
                        dense_query(client, vec_name, v, flt, max(8, limit // 3 or 4))
                        for v in embedded_sem_prf
                    ]
                    for res_sem_prf in result_sets_sem_prf:
                        for rank, p in enumerate(res_sem_prf, 1):
                            pid = str(p.id)
                            score_map.setdefault(
                                pid,
                                {
                                    "pt": p,
                                    "s": 0.0,
                                    "d": 0.0,
                                    "lx": 0.0,
                                    "sym_sub": 0.0,
                                    "sym_eq": 0.0,
                                    "core": 0.0,
                                    "vendor": 0.0,
                                    "langb": 0.0,
                                    "rec": 0.0,
                                    "test": 0.0,
                                },
                            )
                            # Lower weight for semantic PRF to avoid over-diversification
                            dens = 0.3 * prf_dw * _scaled_rrf(rank)
                            score_map[pid]["d"] += dens
                            score_map[pid]["s"] += dens

                    if os.environ.get("DEBUG_HYBRID_SEARCH"):
                        logger.debug(f"Semantic PRF added {len(semantic_prf_qs)} queries with terms: {semantic_prf_terms}")

        except Exception as e:
            if os.environ.get("DEBUG_HYBRID_SEARCH"):
                logger.debug(f"Semantic PRF failed: {e}")

    lex_results2: List[Any] = []

    # Pseudo-Relevance Feedback (default-on): mine top terms from current results and run a light second pass
    try:
        prf_enabled = _env_truthy(os.environ.get("PRF_ENABLED"), True)
    except (ValueError, TypeError):
        prf_enabled = True

    # Lightweight BM25-style lexical boost (default ON)
    try:
        _USE_BM25 = _env_truthy(os.environ.get("HYBRID_BM25"), True)
    except Exception:
        _USE_BM25 = True
    try:
        _BM25_W = float(os.environ.get("HYBRID_BM25_WEIGHT", "0.2") or 0.2)
    except Exception:
        _BM25_W = 0.2
    _bm25_tok_w = _bm25_token_weights_from_results(qlist, (lex_results or []) + (lex_results2 or [])) if _USE_BM25 else {}

    if prf_enabled and score_map:
        try:
            top_docs = int(os.environ.get("PRF_TOP_DOCS", "8") or 8)
        except (ValueError, TypeError):
            top_docs = 8
        try:
            max_terms = int(os.environ.get("PRF_MAX_TERMS", "6") or 6)
        except (ValueError, TypeError):
            max_terms = 6
        try:
            extra_q = int(os.environ.get("PRF_EXTRA_QUERIES", "4") or 4)
        except (ValueError, TypeError):
            extra_q = 4
        try:
            prf_dw = float(os.environ.get("PRF_DENSE_WEIGHT", "0.4") or 0.4)
        except (ValueError, TypeError):
            prf_dw = 0.4
        try:
            prf_lw = float(os.environ.get("PRF_LEX_WEIGHT", "0.6") or 0.6)
        except (ValueError, TypeError):
            prf_lw = 0.6
        terms = _prf_terms_from_results(
            score_map, top_docs=top_docs, max_terms=max_terms
        )
        base = clean_queries[0] if clean_queries else (qlist[0] if qlist else "")
        prf_qs: List[str] = []
        for t in terms:
            cand = (base + " " + t).strip()
            if cand and cand not in qlist and cand not in prf_qs:
                prf_qs.append(cand)
                if len(prf_qs) >= extra_q:
                    break
        if prf_qs:
            # Lexical PRF pass
            try:
                lex_vec2 = lex_hash_vector(prf_qs)
                lex_results2 = lex_query(
                    client, lex_vec2, flt, max(12, limit // 2 or 6), collection
                )
            except Exception:
                lex_results2 = []
            for rank, p in enumerate(lex_results2, 1):
                pid = str(p.id)
                score_map.setdefault(
                    pid,
                    {
                        "pt": p,
                        "s": 0.0,
                        "d": 0.0,
                        "lx": 0.0,
                        "sym_sub": 0.0,
                        "sym_eq": 0.0,
                        "core": 0.0,
                        "vendor": 0.0,
                        "langb": 0.0,
                        "rec": 0.0,
                        "test": 0.0,
                    },
                )
                lxs = prf_lw * _scaled_rrf(rank)
                score_map[pid]["lx"] += lxs
                score_map[pid]["s"] += lxs
            # Dense PRF pass
            try:
                embedded2 = _embed_queries_cached(_model, prf_qs)
                _prf_per_query = max(12, _scaled_per_query // 2)
                result_sets2: List[List[Any]] = [
                    dense_query(client, vec_name, v, flt, _prf_per_query, collection)
                    for v in embedded2
                ]
                for res2 in result_sets2:
                    for rank, p in enumerate(res2, 1):
                        pid = str(p.id)
                        score_map.setdefault(
                            pid,
                            {
                                "pt": p,
                                "s": 0.0,
                                "d": 0.0,
                                "lx": 0.0,
                                "sym_sub": 0.0,
                                "sym_eq": 0.0,
                                "core": 0.0,
                                "vendor": 0.0,
                                "langb": 0.0,
                                "rec": 0.0,
                                "test": 0.0,
                            },
                        )
                        dens = prf_dw * _scaled_rrf(rank)
                        score_map[pid]["d"] += dens
                        score_map[pid]["s"] += dens
            except Exception:
                pass

    # Add dense scores (with scaled RRF)
    for res in result_sets:
        for rank, p in enumerate(res, 1):
            pid = str(p.id)
            score_map.setdefault(
                pid,
                {
                    "pt": p,
                    "s": 0.0,
                    "d": 0.0,
                    "lx": 0.0,
                    "sym_sub": 0.0,
                    "sym_eq": 0.0,
                    "core": 0.0,
                    "vendor": 0.0,
                    "langb": 0.0,
                    "rec": 0.0,
                    "test": 0.0,
                },
            )
            dens = (_AD_DENSE_W * _scaled_rrf(rank)) if _USE_ADAPT else (DENSE_WEIGHT * _scaled_rrf(rank))
            score_map[pid]["d"] += dens
            score_map[pid]["s"] += dens

    # Lexical + boosts
    timestamps: List[int] = []
    # Mode-aware tweaks for implementation/docs weighting. Modes:
    # - None / "code_first": full IMPLEMENTATION_BOOST and DOCUMENTATION_PENALTY
    # - "balanced": keep impl boost, halve doc penalty
    # - "docs_first": reduce impl boost slightly and disable doc penalty
    eff_mode = (mode or "").strip().lower()
    impl_boost = IMPLEMENTATION_BOOST
    doc_penalty = DOCUMENTATION_PENALTY
    if eff_mode in {"balanced"}:
        doc_penalty = DOCUMENTATION_PENALTY * 0.5
    elif eff_mode in {"docs_first", "docs-first", "docs"}:
        impl_boost = IMPLEMENTATION_BOOST * 0.5
        doc_penalty = 0.0
    for pid, rec in list(score_map.items()):
        payload = rec["pt"].payload or {}
        base_md = payload.get("metadata") or {}
        # Merge top-level pseudo/tags into the view passed to lexical_score so
        # HYBRID_PSEUDO_BOOST can see index-time GLM/llamacpp labels.
        md = dict(base_md)
        if "pseudo" in payload:
            md["pseudo"] = payload["pseudo"]
        if "tags" in payload:
            md["tags"] = payload["tags"]

        lx = (_AD_LEX_TEXT_W * lexical_score(qlist, md, token_weights=_bm25_tok_w, bm25_weight=_BM25_W)) if _USE_ADAPT else (LEXICAL_WEIGHT * lexical_score(qlist, md, token_weights=_bm25_tok_w, bm25_weight=_BM25_W))
        rec["lx"] += lx
        rec["s"] += lx
        ts = md.get("last_modified_at") or md.get("ingested_at")
        if isinstance(ts, int):
            timestamps.append(ts)
        sym = str(md.get("symbol") or "").lower()
        sym_path = str(md.get("symbol_path") or "").lower()
        sym_text = f"{sym} {sym_path}"
        for q in qlist:
            ql = q.lower()
            if not ql:
                continue
            if ql in sym_text:
                rec["sym_sub"] += SYMBOL_BOOST
                rec["s"] += SYMBOL_BOOST
            if ql == sym or ql == sym_path:
                rec["sym_eq"] += SYMBOL_EQUALITY_BOOST
                rec["s"] += SYMBOL_EQUALITY_BOOST
        path = str(md.get("path") or "")
        if CORE_FILE_BOOST > 0.0 and path and is_core_file(path):
            rec["core"] += CORE_FILE_BOOST
            rec["s"] += CORE_FILE_BOOST
        if VENDOR_PENALTY > 0.0 and path and is_vendor_path(path):
            rec["vendor"] -= VENDOR_PENALTY
            rec["s"] -= VENDOR_PENALTY
        if TEST_FILE_PENALTY > 0.0 and path and is_test_file(path):
            rec["test"] -= TEST_FILE_PENALTY
            rec["s"] -= TEST_FILE_PENALTY

        # Additional file-type weighting
        path_lower = path.lower()
        ext = ("." + path_lower.rsplit(".", 1)[-1]) if "." in path_lower else ""
        # Penalize config/metadata files
        if CONFIG_FILE_PENALTY > 0.0 and path:
            if ext in {".json", ".yml", ".yaml", ".toml", ".ini"} or "/.codebase/" in path_lower or "/.kiro/" in path_lower:
                rec["cfg"] = float(rec.get("cfg", 0.0)) - CONFIG_FILE_PENALTY
                rec["s"] -= CONFIG_FILE_PENALTY
        # Boost likely implementation files (mode-aware)
        if impl_boost > 0.0 and path:
            if ext in {".py", ".ts", ".tsx", ".js", ".jsx", ".go", ".java", ".rs", ".rb", ".php", ".cs", ".cpp", ".c", ".hpp", ".h"}:
                rec["impl"] = float(rec.get("impl", 0.0)) + impl_boost
                rec["s"] += impl_boost
        # Penalize docs (README/docs/markdown) relative to implementation files (mode-aware)
        if doc_penalty > 0.0 and path:
            if (
                "readme" in path_lower
                or "/docs/" in path_lower
                or "/documentation/" in path_lower
                or path_lower.endswith(".md")
            ):
                rec["doc"] = float(rec.get("doc", 0.0)) - doc_penalty
                rec["s"] -= doc_penalty

        if LANG_MATCH_BOOST > 0.0 and path and eff_language:
            lang = str(eff_language).lower()
            md_lang = str((md.get("language") or "").lower())
            if (lang and md_lang and md_lang == lang) or lang_matches_path(lang, path):
                rec["langb"] += LANG_MATCH_BOOST
                rec["s"] += LANG_MATCH_BOOST

        # Memory blending: apply penalty to memory-like entries to prevent swamping code results
        # Only apply if query doesn't explicitly ask for memories/notes
        kind = str(md.get("kind") or "").lower()
        if kind == "memory":
            qlow = " ".join(qlist).lower()
            is_memory_query = any(w in qlow for w in ["remember", "note", "recall", "memo", "stored"])
            if not is_memory_query:
                # Apply penalty and cap at 1 memory result unless explicitly requested
                memory_penalty = float(os.environ.get("HYBRID_MEMORY_PENALTY", "0.15") or 0.15)
                rec["mem_penalty"] = float(rec.get("mem_penalty", 0.0)) - memory_penalty
                rec["s"] -= memory_penalty
                # Simple lexical overlap check: drop memories with no query token overlap
                try:
                    text_lower = str(md.get("text") or md.get("information") or "").lower()
                    query_tokens = set()
                    for q in qlist:
                        query_tokens.update(_split_ident(q))
                    has_overlap = any(t in text_lower for t in query_tokens if len(t) >= 3)
                    if not has_overlap:
                        # Strong penalty for irrelevant memories
                        rec["mem_penalty"] -= 0.5
                        rec["s"] -= 0.5
                except Exception:
                    pass

    if timestamps and RECENCY_WEIGHT > 0.0:
        tmin, tmax = min(timestamps), max(timestamps)
        span = max(1, tmax - tmin)
        for rec in score_map.values():
            md = (rec["pt"].payload or {}).get("metadata") or {}
            ts = md.get("last_modified_at") or md.get("ingested_at")
            if isinstance(ts, int):
                norm = (ts - tmin) / span
                rec_comp = RECENCY_WEIGHT * norm
                rec["rec"] += rec_comp
                rec["s"] += rec_comp

    # === Large codebase score normalization ===
    # Spread compressed score distributions for better discrimination
    _normalize_scores(score_map, _coll_size)

    def _tie_key(m: Dict[str, Any]):
        md = (m["pt"].payload or {}).get("metadata") or {}
        sp = str(md.get("symbol_path") or md.get("symbol") or "")
        path = str(md.get("path") or "")
        start_line = int(md.get("start_line") or 0)
        return (-float(m["s"]), len(sp), path, start_line)

    if os.environ.get("DEBUG_HYBRID_SEARCH"):
        logger.debug(f"score_map has {len(score_map)} items before ranking (coll_size={_coll_size})")
    ranked = sorted(score_map.values(), key=_tie_key)
    if os.environ.get("DEBUG_HYBRID_SEARCH"):
        logger.debug(f"ranked has {len(ranked)} items after sorting")

    # Lightweight keyword bump: prefer spans whose local snippet contains query tokens
    try:
        kb = float(os.environ.get("HYBRID_KEYWORD_BUMP", "0.3") or 0.3)
        kcap = float(os.environ.get("HYBRID_KEYWORD_CAP", "0.6") or 0.6)
    except Exception:
        kb, kcap = 0.3, 0.6
    # Build lowercase keyword set from queries (simple split, keep >=3 chars + special tokens)
    kw: set[str] = set()
    for q in qlist:
        ql = (q or "").lower()
        for tok in re.findall(r"[a-zA-Z0-9_\-]+", ql):
            t = tok.strip()
            if len(t) >= 3:
                kw.add(t)
    # Add a few commonly relevant code tokens (helps for gate-first cases)
    for t in ("hasidcondition", "pid_str", "matchany", "gate-first", "gatefirst"):
        kw.add(t)

    import io as _io

    def _snippet_contains(md: dict) -> int:
        # returns number of keyword hits found in a small local snippet
        try:
            path = str(md.get("path") or "")
            sline = int(md.get("start_line") or 0)
            eline = int(md.get("end_line") or 0)
            txt = (md.get("text") or md.get("code") or "")
            if not txt and path and sline:
                p = path
                try:
                    if not os.path.isabs(p):
                        p = os.path.join("/work", p)
                    realp = os.path.realpath(p)
                    if realp == "/work" or realp.startswith("/work/"):
                        with open(realp, "r", encoding="utf-8", errors="ignore") as f:
                            lines = f.readlines()
                        si = max(1, sline - 3)
                        ei = min(len(lines), max(sline, eline) + 3)
                        txt = "".join(lines[si-1:ei])
                except Exception:
                    txt = txt or ""
            lt = (txt or "").lower()
            if not lt:
                return 0
            hits = 0
            for t in kw:
                if t and t in lt:
                    hits += 1
            return hits
        except Exception:
            return 0

    # Apply bump to top-N ranked (limited for speed)
    topN = min(len(ranked), 200)
    for i in range(topN):
        m = ranked[i]
        md = (m["pt"].payload or {}).get("metadata") or {}
        hits = _snippet_contains(md)
        if hits > 0 and kb > 0.0:
            bump = min(kcap, kb * float(hits))
            m["s"] += bump
        # Apply comment-heavy penalty to de-emphasize comments/doc blocks
        try:
            if COMMENT_PENALTY > 0.0:
                ratio = _snippet_comment_ratio(md)
                thr = float(COMMENT_RATIO_THRESHOLD)
                if ratio >= thr:
                    # Scale penalty with how far ratio exceeds threshold (cap at COMMENT_PENALTY)
                    scale = (ratio - thr) / max(1e-6, 1.0 - thr)
                    pen = min(float(COMMENT_PENALTY), float(COMMENT_PENALTY) * max(0.0, scale))
                    if pen > 0:
                        m["cmt"] = float(m.get("cmt", 0.0)) - pen
                        m["s"] -= pen
        except Exception:
            pass
    # Re-sort after bump
    def _snippet_comment_ratio(md: dict) -> float:
        # Estimate fraction of non-blank lines that are comments (language-agnostic heuristics)
        try:
            path = str(md.get("path") or "")
            sline = int(md.get("start_line") or 0)
            eline = int(md.get("end_line") or 0)
            txt = (md.get("text") or md.get("code") or "")
            if not txt and path and sline:
                p = path
                try:
                    if not os.path.isabs(p):
                        p = os.path.join("/work", p)
                    realp = os.path.realpath(p)
                    if realp == "/work" or realp.startswith("/work/"):
                        with open(realp, "r", encoding="utf-8", errors="ignore") as f:
                            lines = f.readlines()
                        si = max(1, sline - 3)
                        ei = min(len(lines), max(sline, eline) + 3)
                        txt = "".join(lines[si-1:ei])
                except Exception:
                    txt = txt or ""
            if not txt:
                return 0.0
            total = 0
            comment = 0
            in_block = False
            for raw in txt.splitlines():
                line = raw.strip()
                if not line:
                    continue
                total += 1
                # HTML/XML comments
                if line.startswith("<!--"):
                    in_block = True
                    comment += 1
                    continue
                if in_block:
                    comment += 1
                    if "-->" in line:
                        in_block = False
                    continue
                # C/JS block comments
                if line.startswith("/*"):
                    in_block = True
                    comment += 1
                    continue
                if in_block:
                    comment += 1
                    if "*/" in line:
                        in_block = False
                    continue
                # Single-line comments for many languages
                if line.startswith("//") or line.startswith("#"):
                    comment += 1
                    continue
                # Python docstring-like lines (treat as comment-ish for ranking)
                if line.startswith("\"\"\"") or line.startswith("'''"):
                    comment += 1
                    continue
            if total == 0:
                return 0.0
            return comment / float(total)
        except Exception:
            return 0.0

    # Apply bump to top-N ranked (limited for speed)

    ranked = sorted(ranked, key=_tie_key)

    # Cluster by path adjacency
    clusters: Dict[str, List[Dict[str, Any]]] = {}
    for m in ranked:
        md = (m["pt"].payload or {}).get("metadata") or {}
        path = str(md.get("path") or "")
        start_line = int(md.get("start_line") or 0)
        end_line = int(md.get("end_line") or 0)
        lst = clusters.setdefault(path, [])
        merged_flag = False
        for c in lst:
            if (
                start_line <= c["end"] + CLUSTER_LINES
                and end_line >= c["start"] - CLUSTER_LINES
            ):
                if float(m["s"]) > float(c["m"]["s"]):
                    c["m"] = m
                c["start"] = min(c["start"], start_line)
                c["end"] = max(c["end"], end_line)
                merged_flag = True
                break
        if not merged_flag:
            lst.append({"start": start_line, "end": end_line, "m": m})

    ranked = sorted([c["m"] for lst in clusters.values() for c in lst], key=_tie_key)
    if os.environ.get("DEBUG_HYBRID_SEARCH"):
        logger.debug(f"ranked has {len(ranked)} items after clustering")


    # Optional MMR diversification (default ON; preserves top-1)
    if _env_truthy(os.environ.get("HYBRID_MMR"), True):
        try:
            _mmr_k = min(len(ranked), max(20, int(os.environ.get("MMR_K", str((limit or 10) * 3)) or 30)))
        except Exception:
            _mmr_k = min(len(ranked), max(20, (limit or 10) * 3))
        try:
            _mmr_lambda = float(os.environ.get("MMR_LAMBDA", "0.7") or 0.7)
        except Exception:
            _mmr_lambda = 0.7
        if (limit or 0) >= 10 or (not per_path) or (per_path <= 0):
            ranked = _mmr_diversify(ranked, k=_mmr_k, lambda_=_mmr_lambda)

    # Client-side filters and per-path diversification
    import re as _re, fnmatch as _fnm

    case_sensitive = str(eff_case or "").lower() == "sensitive"

    def _match_glob(pat: str, path: str) -> bool:
        if not pat:
            return True
        if case_sensitive:
            return _fnm.fnmatchcase(path, pat)
        return _fnm.fnmatchcase(path.lower(), pat.lower())

    if eff_not or eff_path_regex or eff_ext or eff_path_globs or eff_not_globs:

        def _pass_filters(m: Dict[str, Any]) -> bool:
            md = (m["pt"].payload or {}).get("metadata") or {}
            path = str(md.get("path") or "")
            rel = path[6:] if path.startswith("/work/") else path
            pp = str(md.get("path_prefix") or "")
            p_for_sub = path if case_sensitive else path.lower()
            pp_for_sub = pp if case_sensitive else pp.lower()
            if eff_not:
                nn = eff_not if case_sensitive else eff_not.lower()
                if nn in p_for_sub or nn in pp_for_sub:
                    return False
            if eff_not_globs_norm and any(_match_glob(g, path) or _match_glob(g, rel) for g in eff_not_globs_norm):
                return False
            if eff_ext:
                ex = eff_ext.lower().lstrip(".")
                if not path.lower().endswith("." + ex):
                    return False
            if eff_path_regex:
                flags = 0 if case_sensitive else _re.IGNORECASE
                try:
                    if not _re.search(eff_path_regex, path, flags=flags):
                        return False
                except Exception:
                    pass
            if eff_path_globs_norm and not any(_match_glob(g, path) or _match_glob(g, rel) for g in eff_path_globs_norm):
                return False
            return True

        ranked = [m for m in ranked if _pass_filters(m)]

    # ReFRAG-lite span compaction and budgeting is NOT applied here in run_hybrid_search
    # It's only applied in context_answer where token budgeting is needed for LLM context
    # Removing this to avoid over-filtering search results

    if per_path and per_path > 0:
        counts: Dict[str, int] = {}
        merged: List[Dict[str, Any]] = []
        if os.environ.get("DEBUG_HYBRID_SEARCH"):
            logger.debug(f"Applying per_path={per_path} limiting to {len(ranked)} ranked results")
        for m in ranked:
            md = (m["pt"].payload or {}).get("metadata") or {}
            path = str(md.get("path", ""))
            c = counts.get(path, 0)
            if c < per_path:
                merged.append(m)
                counts[path] = c + 1
            if len(merged) >= limit:
                break
        if os.environ.get("DEBUG_HYBRID_SEARCH"):
            logger.debug(f"After per_path limiting: {len(merged)} results from {len(counts)} unique paths")
    else:
        merged = ranked[:limit]

    # Emit structured items
    # Build directory → paths map for related hints (same dir siblings)
    dir_to_paths: Dict[str, set] = {}
    try:
        for _m in merged:
            _md = (_m["pt"].payload or {}).get("metadata") or {}
            _pp = str(_md.get("path_prefix") or "")
            _p = str(_md.get("path") or "")
            if _pp and _p:
                dir_to_paths.setdefault(_pp, set()).add(_p)
    except Exception:
        dir_to_paths = {}
    # Precompute known paths for quick membership checks
    all_paths: set = set()
    try:
        for _s in dir_to_paths.values():
            all_paths |= set(_s)
    except Exception:
        all_paths = set()

    # Build path -> host_path map so we can emit related_paths in host space
    # when PATH_EMIT_MODE prefers host paths. This keeps human-facing paths
    # consistent while still preserving container paths for backend use.
    host_map: Dict[str, str] = {}
    try:
        for _m in merged:
            _md = (_m["pt"].payload or {}).get("metadata") or {}
            _p = str(_md.get("path") or "").strip()
            _h = str(_md.get("host_path") or "").strip()
            if _p and _h:
                host_map[_p] = _h
    except Exception:
        host_map = {}

    items: List[Dict[str, Any]] = []
    if not merged:
        if _USE_CACHE and cache_key is not None:
            if UNIFIED_CACHE_AVAILABLE:
                # Use unified caching system
                _RESULTS_CACHE.set(cache_key, items)
                if os.environ.get("DEBUG_HYBRID_SEARCH"):
                    logger.debug("cache store for hybrid results")
            else:
                # Fallback to original caching system
                with _RESULTS_LOCK:
                    _RESULTS_CACHE[cache_key] = items
                    if os.environ.get("DEBUG_HYBRID_SEARCH"):
                        logger.debug("cache store for hybrid results")
                    while len(_RESULTS_CACHE) > MAX_RESULTS_CACHE:
                        _RESULTS_CACHE.popitem(last=False)
        return items

    for m in merged:
        md = (m["pt"].payload or {}).get("metadata") or {}
        # Prefer merged bounds if present
        start_line = m.get("_merged_start") or md.get("start_line")
        end_line = m.get("_merged_end") or md.get("end_line")
        # Store both fusion_score (from hybrid) and rerank_score (if available) separately
        fusion_score = float(m.get("s", 0.0))
        rerank_score = m.get("rerank_score")  # None if not reranked

        comp = {
            "dense_rrf": round(float(m.get("d", 0.0)), 4),
            "lexical": round(float(m.get("lx", 0.0)), 4),
            "symbol_substr": round(float(m.get("sym_sub", 0.0)), 4),
            "symbol_exact": round(float(m.get("sym_eq", 0.0)), 4),
            "core_boost": round(float(m.get("core", 0.0)), 4),
            "vendor_penalty": round(float(m.get("vendor", 0.0)), 4),
            "lang_boost": round(float(m.get("langb", 0.0)), 4),
            "recency": round(float(m.get("rec", 0.0)), 4),
            "test_penalty": round(float(m.get("test", 0.0)), 4),
            # new components
            "config_penalty": round(float(m.get("cfg", 0.0)), 4),
            "impl_boost": round(float(m.get("impl", 0.0)), 4),
            "doc_penalty": round(float(m.get("doc", 0.0)), 4),
        }

        # Add reranker info to components if present
        if rerank_score is not None:
            comp["rerank"] = round(float(rerank_score), 4)
        why = []
        if comp["dense_rrf"]:
            why.append(f"dense_rrf:{comp['dense_rrf']}")
        for k in ("lexical", "symbol_substr", "symbol_exact", "core_boost", "lang_boost", "impl_boost"):
            if comp[k]:
                why.append(f"{k}:{comp[k]}")
        if comp["vendor_penalty"]:
            why.append(f"vendor_penalty:{comp['vendor_penalty']}")
        for k in ("test_penalty", "config_penalty", "doc_penalty"):
            if comp.get(k):
                why.append(f"{k}:{comp[k]}")
        if comp["recency"]:
            why.append(f"recency:{comp['recency']}")
        # Related hints
        _imports = md.get("imports") or []
        _calls = md.get("calls") or []
        _symp = md.get("symbol_path") or md.get("symbol") or ""
        _pp = str(md.get("path_prefix") or "")
        _path = str(md.get("path") or "")
        _related_set = set()
        # Same-dir siblings
        try:
            if _pp in dir_to_paths:
                for p in dir_to_paths[_pp]:
                    if p != _path:
                        _related_set.add(p)
        except Exception:
            pass
        # Import-based hints: resolve relative/quoted path-like imports
        try:
            import re as _re, posixpath as _ppath

            def _pathlike_segments(s: str) -> list[str]:
                s = str(s or "")
                segs = []
                # quoted segments first
                for mmm in _re.findall(r"[\"']([^\"']+)[\"']", s):
                    if "/" in mmm or mmm.startswith("."):
                        segs.append(mmm)
                # fall back to whitespace tokens containing '/' or starting with '.'
                for tok in str(s).replace(",", " ").split():
                    if ("/" in tok) or tok.startswith("."):
                        segs.append(tok)
                return segs

            def _resolve(seg: str) -> list[str]:
                try:
                    seg = seg.strip()
                    # base dir from path_prefix
                    base = _pp or ""
                    candidates = []
                    # choose join rule
                    if seg.startswith("./") or seg.startswith("../") or "/" in seg:
                        j = _ppath.normpath(_ppath.join(base, seg)) if not seg.startswith("/") else _ppath.normpath(seg)
                        candidates.append(j)
                        # add extensions if last segment lacks a dot
                        last = j.split("/")[-1]
                        if "." not in last:
                            for ext in [".py", ".js", ".ts", ".tsx", ".jsx", ".mjs", ".cjs"]:
                                candidates.append(j + ext)
                    out = set()
                    for c in candidates:
                        if c in all_paths:
                            out.add(c)
                        if c.startswith("/") and c.lstrip("/") in all_paths:
                            out.add(c.lstrip("/"))
                        if c.startswith("/work/") and c[len("/work/"):] in all_paths:
                            out.add(c[len("/work/"):])
                    return list(out)
                except Exception:
                    return []

            for imp in (_imports or []):
                for seg in _pathlike_segments(imp):
                    for cand in _resolve(seg):
                        if cand != _path:
                            _related_set.add(cand)
        except Exception:
            pass

        _related = sorted(_related_set)[:10]
        # Align related_paths with PATH_EMIT_MODE when possible: in host/auto
        # modes, prefer host paths when we have a mapping; in container mode,
        # keep container/path-space values as-is.
        _related_out = _related
        try:
            _mode_related = str(os.environ.get("PATH_EMIT_MODE", "auto")).strip().lower()
        except Exception:
            _mode_related = "auto"
        if _mode_related in {"host", "auto"}:
            try:
                _mapped: List[str] = []
                for rp in _related:
                    _mapped.append(host_map.get(rp, rp))
                _related_out = _mapped
            except Exception:
                _related_out = _related
        # Best-effort snippet text directly from payload for downstream LLM stitching
        _payload = (m["pt"].payload or {}) if m.get("pt") is not None else {}
        _metadata = _payload.get("metadata", {}) or {}
        _text = (
            _payload.get("code") or
            _metadata.get("code") or
            _payload.get("text") or
            _metadata.get("text") or
            ""
        )
        # Carry through pseudo/tags so downstream consumers (e.g., repo_search reranker)
        # can incorporate index-time GLM/llm labels into their own scoring or display.
        _pseudo = _payload.get("pseudo")
        if _pseudo is None:
            _pseudo = _metadata.get("pseudo")
        _tags = _payload.get("tags")
        if _tags is None:
            _tags = _metadata.get("tags")
        # Skip memory-like points without a real file path
        if not _path or not _path.strip():
            if os.environ.get("DEBUG_HYBRID_FILTER"):
                logger.debug(f"Filtered out item with empty path: {_metadata}")
            continue

        # Emit path: prefer original host path when available; also include container path
        _emit_path = _path
        _host = ""
        _cont = ""
        try:
            _host = str(_metadata.get("host_path") or "").strip()
            _cont = str(_metadata.get("container_path") or "").strip()
            _repo = str(_metadata.get("repo") or "").strip()
            _pp = str(_metadata.get("path_prefix") or "").strip()
            _mode = str(os.environ.get("PATH_EMIT_MODE", "auto")).strip().lower()

            if _mode == "host" and _host:
                _emit_path = _host
            elif _mode == "container" and _cont:
                _emit_path = _cont
            else:
                # Auto mode: prefer host when available, else container; then fallback normalization
                if _host:
                    _emit_path = _host
                elif _cont:
                    _emit_path = _cont
                else:
                    # Auto/compat fallback: normalize to container form if repo+prefix known; else map cwd to /work
                    if _repo and _pp and isinstance(_emit_path, str):
                        _pp_norm = _pp.rstrip("/") + "/"
                        if _emit_path.startswith(_pp_norm):
                            _rel = _emit_path[len(_pp_norm):]
                            if _rel:
                                _emit_path = f"/work/{_repo}/" + _rel.lstrip("/")
                    if isinstance(_emit_path, str):
                        _cwd = os.getcwd().rstrip("/") + "/"
                        if _emit_path.startswith(_cwd):
                            _rel = _emit_path[len(_cwd):]
                            if _rel:
                                _emit_path = "/work/" + _rel
        except Exception:
            pass

        items.append(
            {
                "score": round(float(m["s"]), 4),
                "raw_score": float(m["s"]),  # expose raw fused score for downstream budgeter
                "fusion_score": round(fusion_score, 4),  # Always store fusion score
                "rerank_score": round(float(rerank_score), 4) if rerank_score is not None else None,  # Store rerank separately
                "path": _emit_path,
                "host_path": _host,
                "container_path": _cont,
                "symbol": _symp,
                "start_line": start_line,
                "end_line": end_line,
                "components": comp,
                "why": why,
                "relations": {"imports": _imports, "calls": _calls, "symbol_path": _symp},
                "related_paths": _related_out,
                "span_budgeted": bool(m.get("_merged_start") is not None),
                "budget_tokens_used": m.get("_budget_tokens"),
                "text": _text,
                "pseudo": _pseudo,
                "tags": _tags,
            }
        )
    if _USE_CACHE and cache_key is not None:
        if UNIFIED_CACHE_AVAILABLE:
            _RESULTS_CACHE.set(cache_key, items)
            # Mirror into local fallback dict for deterministic hits in tests
            try:
                with _RESULTS_LOCK:
                    _RESULTS_CACHE_OD[cache_key] = items
                    while len(_RESULTS_CACHE_OD) > MAX_RESULTS_CACHE:
                        # pop oldest inserted (like LRU/FIFO)
                        try:
                            _RESULTS_CACHE_OD.popitem(last=False)
                        except Exception:
                            break
            except Exception:
                pass
            if os.environ.get("DEBUG_HYBRID_SEARCH"):
                logger.debug("cache store for hybrid results")
        else:
            with _RESULTS_LOCK:
                _RESULTS_CACHE[cache_key] = items
                if os.environ.get("DEBUG_HYBRID_SEARCH"):
                    logger.debug("cache store for hybrid results")
                while len(_RESULTS_CACHE) > MAX_RESULTS_CACHE:
                    _RESULTS_CACHE.popitem(last=False)
    return items


def main():
    ap = argparse.ArgumentParser(description="Hybrid search: dense + lexical RRF")
    ap.add_argument(
        "--query",
        "-q",
        action="append",
        required=True,
        help="One or more query strings (multi-query)",
    )
    ap.add_argument("--language", type=str, default=None)
    ap.add_argument("--under", type=str, default=None)
    ap.add_argument("--kind", type=str, default=None)
    ap.add_argument("--symbol", type=str, default=None)
    # Expansion disabled by default; enable via --expand or HYBRID_EXPAND=1
    ap.add_argument(
        "--expand",
        dest="expand",
        action="store_true",
        default=_env_truthy(os.environ.get("HYBRID_EXPAND"), False),
        help="Enable simple query expansion",
    )
    ap.add_argument(
        "--no-expand",
        dest="expand",
        action="store_false",
        help="Disable query expansion",
    )
    # Per-path diversification enabled by default (1) unless overridden by env/flag
    ap.add_argument(
        "--per-path",
        type=int,
        default=int(os.environ.get("HYBRID_PER_PATH", "1") or 1),
        help="Cap results per file path to diversify (0=off)",
    )

    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--per-query", type=int, default=24)
    ap.add_argument(
        "--json",
        dest="json",
        action="store_true",
        help="Emit JSON lines with score breakdown",
    )
    ap.add_argument(
        "--quiet",
        dest="quiet",
        action="store_true",
        help="Suppress human output on empty results; exit with code 1 when no matches",
    )

    # Structured filters to mirror MCP tool fields
    ap.add_argument("--ext", type=str, default=None)
    ap.add_argument("--not", dest="not_filter", type=str, default=None)
    ap.add_argument("--collection", type=str, default=None,
                     help="Target collection name")
    ap.add_argument(
        "--case",
        type=str,
        choices=["sensitive", "insensitive"],
        default=os.environ.get("HYBRID_CASE", "insensitive"),
    )
    ap.add_argument("--path-regex", dest="path_regex", type=str, default=None)
    ap.add_argument("--path-glob", dest="path_glob", type=str, default=None)
    ap.add_argument("--not-glob", dest="not_glob", type=str, default=None)

    args = ap.parse_args()

    # Resolve effective collection early to avoid variable usage errors
    eff_collection = args.collection or os.environ.get("COLLECTION_NAME", "codebase")

    # Use embedder factory for Qwen3 support
    if _EMBEDDER_FACTORY:
        model = _get_embedding_model(MODEL_NAME)
    else:
        model = TextEmbedding(model_name=MODEL_NAME)
    vec_name = _sanitize_vector_name(MODEL_NAME)
    client = QdrantClient(url=QDRANT_URL, api_key=API_KEY or None)

    # Ensure collection exists with dual named vectors before search
    try:
        first_vec = next(model.embed(["__dim__warmup__"]))
        dim = len(first_vec.tolist())
        _ensure_collection(client, _collection(eff_collection), dim, vec_name)
    except Exception:
        pass

    # Parse Query DSL from queries, then build effective filters
    raw_queries = list(args.query)
    clean_queries, dsl = parse_query_dsl(raw_queries)
    eff_language = args.language or dsl.get("language")
    eff_under = args.under or dsl.get("under")
    eff_kind = args.kind or dsl.get("kind")
    eff_symbol = args.symbol or dsl.get("symbol")
    eff_ext = args.ext or dsl.get("ext")
    eff_not = args.not_filter or dsl.get("not")
    eff_case = args.case or dsl.get("case")
    eff_repo = dsl.get("repo")
    eff_path_regex = args.path_regex
    eff_path_glob = getattr(args, "path_glob", None)
    eff_not_glob = getattr(args, "not_glob", None)

    # Normalize 'under' to absolute path_prefix used in payload (defaults to /work/<rel>)
    def _norm_under(u: str | None) -> str | None:
        if not u:
            return None
        u = str(u).strip()
        # Handle common path variations: backslashes, multiple slashes, trailing slashes
        u = u.replace("\\", "/")
        # Collapse multiple slashes and remove empty segments
        u = "/".join([p for p in u.split("/") if p])
        if not u:
            return None
        # Relative path: prepend /work/
        if not u.startswith("/"):
            v = "/work/" + u
        else:
            # Absolute path: ensure it's under /work mount
            v = "/work/" + u.lstrip("/") if not u.startswith("/work/") else u
        # If the normalized path points to a real file under /work, use its parent directory as prefix
        try:
            from pathlib import Path as _P

            p = _P(v)
            # If it's an existing file, use its parent directory as the prefix
            if p.is_file():
                return str(p.parent)
            # Heuristic: if path doesn't exist and looks like a file stem (no dot),

            # treat it as a file name and use its parent directory
            if (not p.exists()) and p.name and ("." not in p.name):
                return str(p.parent) if str(p.parent) else v
        except Exception:
            pass
        # Already normalized /work/... dir path or non-existent path; use as-is
        return v

    eff_under = _norm_under(eff_under)

    # Build optional filter
    flt = None
    must = []
    if eff_language:
        must.append(
            models.FieldCondition(
                key="metadata.language", match=models.MatchValue(value=eff_language)
            )
        )
    if eff_repo:
        must.append(
            models.FieldCondition(
                key="metadata.repo", match=models.MatchValue(value=eff_repo)
            )
        )
    if eff_under:
        must.append(
            models.FieldCondition(
                key="metadata.path_prefix", match=models.MatchValue(value=eff_under)
            )
        )
    # If ext: was provided without an explicit language, infer language from extension
    if eff_ext and not eff_language:
        ex = eff_ext.lower().lstrip(".")
        for lang, exts in LANG_EXTS.items():
            if any(ex == e.lstrip(".").lower() for e in exts):
                eff_language = lang
                break

    if eff_kind:
        must.append(
            models.FieldCondition(
                key="metadata.kind", match=models.MatchValue(value=eff_kind)
            )
        )
    if eff_symbol:
        must.append(
            models.FieldCondition(
                key="metadata.symbol", match=models.MatchValue(value=eff_symbol)
            )
        )
    flt = models.Filter(must=must) if must else None
    flt = _sanitize_filter_obj(flt)

    # === Large codebase scaling (CLI path) ===
    _cli_coll_stats = _get_collection_stats(client, eff_collection)
    _cli_coll_size = _cli_coll_stats.get("points_count", 0)
    _cli_has_filters = bool(eff_language or eff_repo or eff_under or eff_kind or eff_symbol or eff_ext)
    _cli_scaled_rrf_k = _scale_rrf_k(RRF_K, _cli_coll_size)
    _cli_scaled_per_query = _adaptive_per_query(args.per_query, _cli_coll_size, _cli_has_filters)

    def _cli_scaled_rrf(rank: int) -> float:
        return 1.0 / (_cli_scaled_rrf_k + rank)

    # Build query set (optionally expanded)
    queries = list(clean_queries)
    # Initialize score map early so we can accumulate from lex and dense
    score_map: Dict[str, Dict[str, Any]] = {}
    # Server-side lexical vector search (hashing) as an additional ranked list
    try:
        lex_vec = lex_hash_vector(queries)
        lex_results = lex_query(client, lex_vec, flt, _cli_scaled_per_query, eff_collection)
    except Exception:
        lex_results = []

    # Per-query adaptive weights for CLI path (default ON)
    _USE_ADAPT2 = _env_truthy(os.environ.get("HYBRID_ADAPTIVE_WEIGHTS"), True)
    if _USE_ADAPT2:
        try:
            _AD_DENSE_W2, _AD_LEX_VEC_W2, _AD_LEX_TEXT_W2 = _adaptive_weights(_compute_query_stats(queries))
        except Exception:
            _AD_DENSE_W2, _AD_LEX_VEC_W2, _AD_LEX_TEXT_W2 = DENSE_WEIGHT, LEX_VECTOR_WEIGHT, LEXICAL_WEIGHT
    else:
        _AD_DENSE_W2, _AD_LEX_VEC_W2, _AD_LEX_TEXT_W2 = DENSE_WEIGHT, LEX_VECTOR_WEIGHT, LEXICAL_WEIGHT


    if args.expand:
        # Use enhanced expansion with semantic similarity if available
        if SEMANTIC_EXPANSION_AVAILABLE:
            queries = expand_queries_enhanced(
                queries, eff_language,
                max_extra=max(2, int(os.environ.get("SEMANTIC_EXPANSION_MAX_TERMS", "3") or "3")),
                client=client,
                model=model
            )
        else:
            queries = expand_queries(queries, eff_language)

    # Add server-side lexical vector ranking into fusion (with scaled RRF)
    for rank, p in enumerate(lex_results, 1):
        pid = str(p.id)
        score_map.setdefault(
            pid,
            {
                "pt": p,
                "s": 0.0,
                "d": 0.0,
                "lx": 0.0,
                "sym_sub": 0.0,
                "sym_eq": 0.0,
                "core": 0.0,
                "vendor": 0.0,
                "langb": 0.0,
                "rec": 0.0,
                "test": 0.0,
            },
        )
        lxs = (_AD_LEX_VEC_W2 * _cli_scaled_rrf(rank)) if _USE_ADAPT2 else (LEX_VECTOR_WEIGHT * _cli_scaled_rrf(rank))
        score_map[pid]["lx"] += lxs
        score_map[pid]["s"] += lxs

    embedded = _embed_queries_cached(model, queries)
    result_sets: List[List[Any]] = [
        dense_query(
            client,
            vec_name,
            v,
            flt,
            _cli_scaled_per_query,
            eff_collection,
            query_text=queries[i] if i < len(queries) else None,
        )
        for i, v in enumerate(embedded)
    ]

    # RRF fusion (weighted, with scaled RRF)
    for res in result_sets:
        for rank, p in enumerate(res, 1):
            pid = str(p.id)
            score_map.setdefault(
                pid,
                {
                    "pt": p,
                    "s": 0.0,
                    "d": 0.0,
                    "lx": 0.0,
                    "sym_sub": 0.0,
                    "sym_eq": 0.0,
                    "core": 0.0,
                    "vendor": 0.0,
                    "langb": 0.0,
                    "rec": 0.0,
                    "test": 0.0,
                },
            )
            dens = (_AD_DENSE_W2 * _cli_scaled_rrf(rank)) if _USE_ADAPT2 else (DENSE_WEIGHT * _cli_scaled_rrf(rank))
            score_map[pid]["d"] += dens
            score_map[pid]["s"] += dens

    # Lexical bump + symbol boost; also collect recency
    # Lightweight BM25-style lexical boost (default ON) for CLI path
    try:
        _USE_BM25_CLI = _env_truthy(os.environ.get("HYBRID_BM25"), True)
    except Exception:
        _USE_BM25_CLI = True
    try:
        _BM25_W2 = float(os.environ.get("HYBRID_BM25_WEIGHT", "0.2") or 0.2)
    except Exception:
        _BM25_W2 = 0.2
    _bm25_tok_w2 = _bm25_token_weights_from_results(queries, lex_results) if _USE_BM25_CLI else {}

    # Build Aho-Corasick automaton once for all results (if AC_LEXICAL_ENABLED=1)
    _ac_automaton = None
    if _AC_LEXICAL_ENABLED:
        _ac_tokens = tokenize_queries(queries)
        _ac_automaton = build_lexical_automaton_for_search(_ac_tokens)

    timestamps: List[int] = []
    for pid, rec in list(score_map.items()):
        md = (rec["pt"].payload or {}).get("metadata") or {}
        # Use AC-optimized scoring if automaton available, else fall back to original
        if _ac_automaton is not None:
            lx_raw = lexical_score_ac(queries, md, _ac_automaton, token_weights=_bm25_tok_w2, bm25_weight=_BM25_W2)
        else:
            lx_raw = lexical_score(queries, md, token_weights=_bm25_tok_w2, bm25_weight=_BM25_W2)
        lx = (_AD_LEX_TEXT_W2 * lx_raw) if _USE_ADAPT2 else (LEXICAL_WEIGHT * lx_raw)
        rec["lx"] += lx
        rec["s"] += lx
        ts = md.get("last_modified_at") or md.get("ingested_at")
        if isinstance(ts, int):
            timestamps.append(ts)

        # Symbol-based boosts
        sym = str(md.get("symbol") or "").lower()
        sym_path = str(md.get("symbol_path") or "").lower()
        sym_text = f"{sym} {sym_path}"
        for q in queries:
            ql = q.lower()
            if not ql:
                continue
            # substring match boost
            if ql in sym_text:
                rec["sym_sub"] += SYMBOL_BOOST
                rec["s"] += SYMBOL_BOOST
            # exact match boost (symbol or symbol_path)
            if ql == sym or ql == sym_path:
                rec["sym_eq"] += SYMBOL_EQUALITY_BOOST
                rec["s"] += SYMBOL_EQUALITY_BOOST

        # Path-based adjustments
        path = str(md.get("path") or "")
        if CORE_FILE_BOOST > 0.0 and path and is_core_file(path):
            rec["core"] += CORE_FILE_BOOST
            rec["s"] += CORE_FILE_BOOST
        if VENDOR_PENALTY > 0.0 and path and is_vendor_path(path):
            rec["vendor"] -= VENDOR_PENALTY
            rec["s"] -= VENDOR_PENALTY
        if TEST_FILE_PENALTY > 0.0 and path and is_test_file(path):
            rec["test"] -= TEST_FILE_PENALTY
            rec["s"] -= TEST_FILE_PENALTY


        # Language match boost if requested
        if (
            LANG_MATCH_BOOST > 0.0
            and path
            and (eff_language or getattr(args, "language", None))
        ):
            lang = str((eff_language or args.language or "")).lower()
            md_lang = str((md.get("language") or "")).lower()
            if (lang and md_lang and md_lang == lang) or lang_matches_path(lang, path):
                rec["langb"] += LANG_MATCH_BOOST
                rec["s"] += LANG_MATCH_BOOST

    # Recency bump (normalize across results)
    if timestamps and RECENCY_WEIGHT > 0.0:
        tmin, tmax = min(timestamps), max(timestamps)
        span = max(1, tmax - tmin)
        for rec in score_map.values():
            md = (rec["pt"].payload or {}).get("metadata") or {}
            ts = md.get("last_modified_at") or md.get("ingested_at")
            if isinstance(ts, int):
                norm = (ts - tmin) / span
                rec_comp = RECENCY_WEIGHT * norm
                rec["rec"] += rec_comp
                rec["s"] += rec_comp

    # === Large codebase score normalization (CLI path) ===
    _normalize_scores(score_map, _cli_coll_size)

    # Rank with deterministic tie-breakers
    def _tie_key(m: Dict[str, Any]):
        md = (m["pt"].payload or {}).get("metadata") or {}
        sp = str(md.get("symbol_path") or md.get("symbol") or "")
        path = str(md.get("path") or "")
        start_line = int(md.get("start_line") or 0)
        return (-float(m["s"]), len(sp), path, start_line)

    ranked = sorted(score_map.values(), key=_tie_key)

    # Adjacent-hit clustering by path
    clusters: Dict[str, List[Dict[str, Any]]] = {}
    for m in ranked:
        md = (m["pt"].payload or {}).get("metadata") or {}
        path = str(md.get("path") or "")
        start_line = int(md.get("start_line") or 0)
        end_line = int(md.get("end_line") or 0)
        lst = clusters.setdefault(path, [])
        merged_flag = False
        for c in lst:
            if (
                start_line <= c["end"] + CLUSTER_LINES
                and end_line >= c["start"] - CLUSTER_LINES
            ):
                # Near/overlapping: keep the higher-scoring rep and expand bounds
                if float(m["s"]) > float(c["m"]["s"]):
                    c["m"] = m
                c["start"] = min(c["start"], start_line)
                c["end"] = max(c["end"], end_line)
                merged_flag = True
                break
        if not merged_flag:
            lst.append({"start": start_line, "end": end_line, "m": m})

    ranked = sorted([c["m"] for lst in clusters.values() for c in lst], key=_tie_key)

    # Optional MMR diversification (default ON; preserves top-1)
    if _env_truthy(os.environ.get("HYBRID_MMR"), True):
        try:
            _mmr_k = min(len(ranked), max(20, int(os.environ.get("MMR_K", str((args.limit or 10) * 3)) or 30)))
        except Exception:
            _mmr_k = min(len(ranked), max(20, (args.limit or 10) * 3))
        try:
            _mmr_lambda = float(os.environ.get("MMR_LAMBDA", "0.7") or 0.7)
        except Exception:
            _mmr_lambda = 0.7
        if (args.limit or 0) >= 10 or (not args.per_path) or (args.per_path <= 0):
            ranked = _mmr_diversify(ranked, k=_mmr_k, lambda_=_mmr_lambda)


    # Apply client-side filters: NOT substring, path regex, glob, and ext
    import re as _re, fnmatch as _fnm

    case_sensitive = str(eff_case or "").lower() == "sensitive"
    if eff_not or eff_path_regex or eff_ext or eff_path_glob or eff_not_glob:

        def _match_glob(pat: str, path: str) -> bool:
            if not pat:
                return True
            if case_sensitive:
                return _fnm.fnmatchcase(path, pat)
            return _fnm.fnmatchcase(path.lower(), pat.lower())

        def _pass_filters(m: Dict[str, Any]) -> bool:
            md = (m["pt"].payload or {}).get("metadata") or {}
            path = str(md.get("path") or "")
            rel = path[6:] if path.startswith("/work/") else path
            pp = str(md.get("path_prefix") or "")
            p_for_sub = path if case_sensitive else path.lower()
            pp_for_sub = pp if case_sensitive else pp.lower()
            # NOT substring filter
            if eff_not:
                nn = eff_not if case_sensitive else eff_not.lower()
                if nn in p_for_sub or nn in pp_for_sub:
                    return False
            # not_glob exclusion
            if eff_not_glob and (
                _match_glob(eff_not_glob, path)
                or _match_glob(eff_not_glob, rel)
                or (not str(eff_not_glob).startswith("/") and _match_glob("/work/" + str(eff_not_glob).lstrip("/"), path))
            ):
                return False
            # Extension filter (normalize to .ext)
            if eff_ext:
                ex = eff_ext.lower().lstrip(".")
                if not path.lower().endswith("." + ex):
                    return False
            # Path regex filter
            if eff_path_regex:
                flags = 0 if case_sensitive else _re.IGNORECASE
                try:
                    if not _re.search(eff_path_regex, path, flags=flags):
                        return False
                except Exception:
                    # Ignore invalid regex
                    pass
            # path_glob inclusion
            if eff_path_glob and not (
                _match_glob(eff_path_glob, path)
                or _match_glob(eff_path_glob, rel)
                or (not str(eff_path_glob).startswith("/") and _match_glob("/work/" + str(eff_path_glob).lstrip("/"), path))
            ):
                return False
            return True

        ranked = [m for m in ranked if _pass_filters(m)]

    # Optional diversification by path
    if args.per_path and args.per_path > 0:
        counts: Dict[str, int] = {}
        merged: List[Dict[str, Any]] = []
        for m in ranked:
            md = (m["pt"].payload or {}).get("metadata") or {}
            path = str(md.get("path", ""))
            c = counts.get(path, 0)
            if c < args.per_path:
                merged.append(m)
                counts[path] = c + 1
                if len(merged) >= args.limit:
                    break
    # Build directory → paths map for related path hints
    dir_to_paths: Dict[str, set] = {}
    try:
        for m in merged:
            md = (m["pt"].payload or {}).get("metadata") or {}
            pp = str(md.get("path_prefix") or "")
            p = str(md.get("path") or "")
            if pp and p:

                dir_to_paths.setdefault(pp, set()).add(p)
    except Exception:
        dir_to_paths = {}

    else:
        merged = ranked[: args.limit]

    # Empty result handling
    if not merged:
        if getattr(args, "json", False):
            try:
                print(json.dumps({"results": [], "query": clean_queries, "count": 0}))
            except Exception:
                print("{}")
            return
        if getattr(args, "quiet", False):
            sys.exit(1)
        print("No results.")
        sys.exit(1)

    for m in merged:
        md = (m["pt"].payload or {}).get("metadata") or {}
        if getattr(args, "json", False):
            # Related hints
            _imports = md.get("imports") or []
            _calls = md.get("calls") or []
            _symp = md.get("symbol_path") or md.get("symbol") or ""
            _pp = str(md.get("path_prefix") or "")
            _related = []
            try:
                if _pp in dir_to_paths:
    # Empty result handling

                    _related = [p for p in sorted(dir_to_paths[_pp]) if p != md.get("path")][:5]
            except Exception:
                _related = []
            item = {
                "score": round(float(m["s"]), 4),
                "path": md.get("path"),
                "symbol": _symp,
                "start_line": md.get("start_line"),
                "end_line": md.get("end_line"),
                "components": {
                    "dense_rrf": round(float(m.get("d", 0.0)), 4),
                    "lexical": round(float(m.get("lx", 0.0)), 4),
                    "symbol_substr": round(float(m.get("sym_sub", 0.0)), 4),
                    "symbol_exact": round(float(m.get("sym_eq", 0.0)), 4),
                    "core_boost": round(float(m.get("core", 0.0)), 4),
                    "vendor_penalty": round(float(m.get("vendor", 0.0)), 4),
                    "lang_boost": round(float(m.get("langb", 0.0)), 4),
                    "recency": round(float(m.get("rec", 0.0)), 4),
                    "test_penalty": round(float(m.get("test", 0.0)), 4),
                    # new components
                    "config_penalty": round(float(m.get("cfg", 0.0)), 4),
                    "impl_boost": round(float(m.get("impl", 0.0)), 4),
                    "doc_penalty": round(float(m.get("doc", 0.0)), 4),
                },
                "relations": {"imports": _imports, "calls": _calls, "symbol_path": _symp},
                "related_paths": _related,
            }
            # Build a human friendly why list
            why = []
            if item["components"]["dense_rrf"]:
                why.append(f"dense_rrf:{item['components']['dense_rrf']}")
            for k in (
                "lexical",
                "symbol_substr",
                "symbol_exact",
                "core_boost",
                "lang_boost",
                "impl_boost",
            ):
                if item["components"][k]:
                    why.append(f"{k}:{item['components'][k]}")
            if item["components"]["vendor_penalty"]:
                why.append(f"vendor_penalty:{item['components']['vendor_penalty']}")
            for k in ("test_penalty", "config_penalty", "doc_penalty"):
                if item["components"].get(k):
                    why.append(f"{k}:{item['components'][k]}")
            if item["components"]["recency"]:
                why.append(f"recency:{item['components']['recency']}")
            item["why"] = why
            print(json.dumps(item))
        else:
            print(
                f"{m['s']:.3f}\t{md.get('path')}\t{md.get('symbol_path') or md.get('symbol') or ''}\t{md.get('start_line')}-{md.get('end_line')}"
            )


if __name__ == "__main__":
    main()
