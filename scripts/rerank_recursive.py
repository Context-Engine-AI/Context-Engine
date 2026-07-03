#!/usr/bin/env python3
"""
Backwards-compatibility shim for rerank_recursive.

This file re-exports all symbols from the scripts.rerank_recursive package
to maintain backwards compatibility for existing imports.

All new code should import from scripts.rerank_recursive (the package) directly.

Usage (both work):
    # Old style (still works)
    from scripts.rerank_recursive import RecursiveReranker, TinyScorer

    # New style (preferred)
    from scripts.rerank_recursive import RecursiveReranker, TinyScorer
"""
from __future__ import annotations

# Re-export everything from the package
from scripts.rerank_recursive import (
    # State
    RefinementState,
    # Core classes
    TinyScorer,
    LatentRefiner,
    VICReg,
    LearnedProjection,
    LearnedHybridWeights,
    QueryExpander,
    ConfidenceEstimator,
    RecursiveReranker,
    ONNXRecursiveReranker,
    SessionAwareReranker,
    # Utilities
    _COMMON_TOKENS,
    _split_identifier,
    _normalize_token,
    _tokenize_for_fname_boost,
    _candidate_path_for_fname_boost,
    _compute_fname_boost,
    _cache_key,
    _get_cached_embedding,
    _cache_embedding,
    # Functions
    rerank_recursive,
    rerank_recursive_inprocess,
    rerank_with_learning,
    rerank_with_session,
    get_recursive_reranker,
    _get_learning_reranker,
    # Constants
    HAS_ONNX,
)

__all__ = [
    # State
    "RefinementState",
    # Core classes
    "TinyScorer",
    "LatentRefiner",
    "VICReg",
    "LearnedProjection",
    "LearnedHybridWeights",
    "QueryExpander",
    "ConfidenceEstimator",
    "RecursiveReranker",
    "ONNXRecursiveReranker",
    "SessionAwareReranker",
    # Utilities
    "_COMMON_TOKENS",
    "_split_identifier",
    "_normalize_token",
    "_tokenize_for_fname_boost",
    "_candidate_path_for_fname_boost",
    "_compute_fname_boost",
    "_cache_key",
    "_get_cached_embedding",
    "_cache_embedding",
    # Functions
    "rerank_recursive",
    "rerank_recursive_inprocess",
    "rerank_with_learning",
    "rerank_with_session",
    "get_recursive_reranker",
    "_get_learning_reranker",
    # Constants
    "HAS_ONNX",
]
