#!/usr/bin/env python3
"""
Configuration constants and helper functions for hybrid search.

This module extracts all environment-based configuration, scoring constants,
and utility functions from hybrid_search.py for reuse across the codebase.
"""
__all__ = [
    "_safe_int", "_safe_float", "_env_truthy", "_get_micro_defaults", "_collection",
    "MODEL_NAME", "QDRANT_URL", "API_KEY",
    "LEX_VECTOR_NAME", "LEX_VECTOR_DIM", "LEX_SPARSE_NAME", "LEX_SPARSE_MODE",
    "MINI_VECTOR_NAME", "MINI_VEC_DIM", "HYBRID_MINI_WEIGHT",
    "RRF_K", "DENSE_WEIGHT", "LEXICAL_WEIGHT", "LEX_VECTOR_WEIGHT", "EF_SEARCH",
    "SYMBOL_BOOST", "SYMBOL_EQUALITY_BOOST", "FNAME_BOOST", "RECENCY_WEIGHT", "CORE_FILE_BOOST",
    "VENDOR_PENALTY", "LANG_MATCH_BOOST", "CLUSTER_LINES", "TEST_FILE_PENALTY",
    "CONFIG_FILE_PENALTY", "IMPLEMENTATION_BOOST", "DOCUMENTATION_PENALTY",
    "PSEUDO_BOOST", "COMMENT_PENALTY", "COMMENT_RATIO_THRESHOLD", "INTENT_IMPL_BOOST",
    "SPARSE_LEX_MAX_SCORE", "SPARSE_RRF_MAX", "SPARSE_RRF_MIN",
    "MICRO_OUT_MAX_SPANS", "MICRO_MERGE_LINES", "MICRO_BUDGET_TOKENS", "MICRO_TOKENS_PER_LINE",
    "LARGE_COLLECTION_THRESHOLD", "MAX_RRF_K_SCALE", "SCORE_NORMALIZE_ENABLED",
    "MAX_EMBED_CACHE", "MAX_RESULTS_CACHE",
    "INCLUDE_WHY",
]
import os
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Helper functions for safe parsing of environment variables
# ---------------------------------------------------------------------------

def _safe_int(val: Any, default: int) -> int:
    """Safely parse an integer from a value, returning default on failure."""
    try:
        if val is None or (isinstance(val, str) and val.strip() == ""):
            return default
        return int(val)
    except (ValueError, TypeError):
        return default


def _safe_float(val: Any, default: float) -> float:
    """Safely parse a float from a value, returning default on failure."""
    try:
        if val is None or (isinstance(val, str) and val.strip() == ""):
            return default
        return float(val)
    except (ValueError, TypeError):
        return default


def _env_truthy(val: str | None, default: bool) -> bool:
    """Check if an environment variable value is truthy or falsy."""
    if val is None:
        return default
    v = val.strip().lower()
    if v in {"1", "true", "yes", "on"}:
        return True
    if v in {"0", "false", "no", "off"}:
        return False
    return default


# ---------------------------------------------------------------------------
# Core environment-based constants
# ---------------------------------------------------------------------------

MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
API_KEY = os.environ.get("QDRANT_API_KEY")

# Lexical vector configuration
# Lexical vector configuration
# Imported from ingest config to ensure Single Source of Truth
from scripts.ingest.config import (
    LEX_VECTOR_NAME,
    LEX_VECTOR_DIM,
    LEX_SPARSE_NAME,
    LEX_SPARSE_MODE,
    MINI_VECTOR_NAME,
    MINI_VEC_DIM,
)

# Optional mini vector (ReFRAG gating) weights (hybrid-specific)
HYBRID_MINI_WEIGHT = _safe_float(os.environ.get("HYBRID_MINI_WEIGHT", "0.5"), 0.5)


# ---------------------------------------------------------------------------
# RRF and scoring constants
# ---------------------------------------------------------------------------

RRF_K = _safe_int(os.environ.get("HYBRID_RRF_K", "30"), 30)
DENSE_WEIGHT = _safe_float(os.environ.get("HYBRID_DENSE_WEIGHT", "1.5"), 1.5)
LEXICAL_WEIGHT = _safe_float(os.environ.get("HYBRID_LEXICAL_WEIGHT", "0.20"), 0.20)
LEX_VECTOR_WEIGHT = _safe_float(
    os.environ.get("HYBRID_LEX_VECTOR_WEIGHT", str(LEXICAL_WEIGHT)), LEXICAL_WEIGHT
)
EF_SEARCH = _safe_int(os.environ.get("QDRANT_EF_SEARCH", "128"), 128)

# Symbol and path-based boosts
SYMBOL_BOOST = _safe_float(os.environ.get("HYBRID_SYMBOL_BOOST", "0.15"), 0.15)
SYMBOL_EQUALITY_BOOST = _safe_float(
    os.environ.get("HYBRID_SYMBOL_EQUALITY_BOOST", "0.25"), 0.25
)
FNAME_BOOST = _safe_float(
    os.environ.get("HYBRID_FNAME_BOOST", str(SYMBOL_EQUALITY_BOOST * 0.5)),
    SYMBOL_EQUALITY_BOOST * 0.5,
)
RECENCY_WEIGHT = _safe_float(os.environ.get("HYBRID_RECENCY_WEIGHT", "0.1"), 0.1)
CORE_FILE_BOOST = _safe_float(os.environ.get("HYBRID_CORE_FILE_BOOST", "0.1"), 0.1)
VENDOR_PENALTY = _safe_float(os.environ.get("HYBRID_VENDOR_PENALTY", "0.05"), 0.05)
LANG_MATCH_BOOST = _safe_float(os.environ.get("HYBRID_LANG_MATCH_BOOST", "0.05"), 0.05)
CLUSTER_LINES = _safe_int(os.environ.get("HYBRID_CLUSTER_LINES", "15"), 15)

# Test file penalty (increased to ensure implementations rank above tests)
TEST_FILE_PENALTY = _safe_float(os.environ.get("HYBRID_TEST_FILE_PENALTY", "0.35"), 0.35)

# Additional file-type weighting knobs
CONFIG_FILE_PENALTY = _safe_float(os.environ.get("HYBRID_CONFIG_FILE_PENALTY", "0.3"), 0.3)
IMPLEMENTATION_BOOST = _safe_float(os.environ.get("HYBRID_IMPLEMENTATION_BOOST", "0.3"), 0.3)
DOCUMENTATION_PENALTY = _safe_float(os.environ.get("HYBRID_DOCUMENTATION_PENALTY", "0.25"), 0.25)

# Pseudo/tags boost (disabled by default)
PSEUDO_BOOST = _safe_float(os.environ.get("HYBRID_PSEUDO_BOOST", "0.0"), 0.0)

# Comment penalty for comment-heavy snippets
COMMENT_PENALTY = _safe_float(os.environ.get("HYBRID_COMMENT_PENALTY", "0.2"), 0.2)
COMMENT_RATIO_THRESHOLD = _safe_float(os.environ.get("HYBRID_COMMENT_RATIO_THRESHOLD", "0.6"), 0.6)

# Query intent detection boost
INTENT_IMPL_BOOST = _safe_float(os.environ.get("HYBRID_INTENT_IMPL_BOOST", "0.15"), 0.15)


# ---------------------------------------------------------------------------
# Sparse lexical vector scoring constants
# ---------------------------------------------------------------------------

SPARSE_LEX_MAX_SCORE = _safe_float(os.environ.get("HYBRID_SPARSE_LEX_MAX", "15.0"), 15.0)
SPARSE_RRF_MAX = 1.0 / (RRF_K + 1)   # Best rank (1)
SPARSE_RRF_MIN = 1.0 / (RRF_K + 50)  # Worst rank we care about (50)


# ---------------------------------------------------------------------------
# Micro-span compaction and budgeting (ReFRAG-lite output shaping)
# ---------------------------------------------------------------------------

def _get_micro_defaults() -> tuple[int, int, int, int]:
    """Return (max_spans, merge_lines, budget_tokens, tokens_per_line) based on runtime and micro chunk mode.

    Budget tokens floor is 5000 to ensure context_answer has enough context for quality answers.
    """
    micro_enabled = os.environ.get("INDEX_MICRO_CHUNKS", "1").strip().lower() in {"1", "true", "yes", "on"}

    try:
        from scripts.refrag_glm import detect_glm_runtime
        is_glm = detect_glm_runtime()
    except ImportError:
        is_glm = False

    if is_glm:
        if micro_enabled:
            return (24, 6, 8192, 32)
        else:
            return (12, 4, 6000, 32)
    else:
        # Non-GLM: still need reasonable budget for quality context_answer
        return (8, 4, 5000, 32)


_MICRO_DEFAULTS = _get_micro_defaults()
MICRO_OUT_MAX_SPANS = _safe_int(os.environ.get("MICRO_OUT_MAX_SPANS", str(_MICRO_DEFAULTS[0])), _MICRO_DEFAULTS[0])
MICRO_MERGE_LINES = _safe_int(os.environ.get("MICRO_MERGE_LINES", str(_MICRO_DEFAULTS[1])), _MICRO_DEFAULTS[1])
MICRO_BUDGET_TOKENS = _safe_int(os.environ.get("MICRO_BUDGET_TOKENS", str(_MICRO_DEFAULTS[2])), _MICRO_DEFAULTS[2])
MICRO_TOKENS_PER_LINE = _safe_int(os.environ.get("MICRO_TOKENS_PER_LINE", str(_MICRO_DEFAULTS[3])), _MICRO_DEFAULTS[3])


# ---------------------------------------------------------------------------
# Large collection scaling constants
# ---------------------------------------------------------------------------

LARGE_COLLECTION_THRESHOLD = _safe_int(os.environ.get("HYBRID_LARGE_THRESHOLD", "10000"), 10000)
MAX_RRF_K_SCALE = _safe_float(os.environ.get("HYBRID_MAX_RRF_K_SCALE", "3.0"), 3.0)
SCORE_NORMALIZE_ENABLED = os.environ.get("HYBRID_SCORE_NORMALIZE", "1").lower() in {"1", "true", "yes", "on"}


# ---------------------------------------------------------------------------
# Collection name resolution helper
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
# Cache configuration
# ---------------------------------------------------------------------------

MAX_EMBED_CACHE = _safe_int(os.environ.get("MAX_EMBED_CACHE"), 8192)
MAX_RESULTS_CACHE = _safe_int(os.environ.get("HYBRID_RESULTS_CACHE"), 32)


# ---------------------------------------------------------------------------
# Output configuration
# ---------------------------------------------------------------------------

# Include "why" explanations in search results (disabled by default to reduce tokens)
INCLUDE_WHY = os.environ.get("INCLUDE_WHY", "0").lower() in {"1", "true", "yes", "on"}
