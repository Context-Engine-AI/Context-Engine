"""
Recursive Reranker Package - TRM-inspired iterative refinement for code search.

This package provides modular components for recursive reranking:

Core Components:
- TinyScorer: 2-layer MLP for scoring query-document pairs
- LatentRefiner: Refines latent state based on current results
- RecursiveReranker: Main reranking pipeline

Regularization:
- VICReg: Variance-Invariance-Covariance regularization

Learnable Components:
- LearnedProjection: Learnable embedding projection
- LearnedHybridWeights: Learns dense vs. lexical balance
- QueryExpander: Learns query expansions from usage

Utilities:
- RefinementState: Dataclass for latent state
- ConfidenceEstimator: Early stopping logic
"""
from __future__ import annotations

# State dataclass
from scripts.rerank_recursive.state import RefinementState

# Utilities
from scripts.rerank_recursive.utils import (
    _COMMON_TOKENS,
    _split_identifier,
    _normalize_token,
    _tokenize_for_fname_boost,
    _candidate_path_for_fname_boost,
    _compute_fname_boost,
    _cache_key,
    _get_cached_embedding,
    _cache_embedding,
)

# Core scorer and refiner
from scripts.rerank_recursive.scorer import TinyScorer
from scripts.rerank_recursive.refiner import LatentRefiner

# Regularization
from scripts.rerank_recursive.vicreg import VICReg

# Learnable components
from scripts.rerank_recursive.projection import LearnedProjection
from scripts.rerank_recursive.hybrid_weights import LearnedHybridWeights
from scripts.rerank_recursive.expander import QueryExpander

# Early stopping
from scripts.rerank_recursive.confidence import ConfidenceEstimator

# Alpha scheduling
from scripts.rerank_recursive.alpha_scheduler import (
    CosineAlphaScheduler,
    LearnedAlphaWeights,
)

# Main rerankers and functions
from scripts.rerank_recursive.recursive import (
    RecursiveReranker,
    ONNXRecursiveReranker,
    FastEmbedRecursiveReranker,
    SessionAwareReranker,
    rerank_recursive,
    rerank_recursive_inprocess,
    rerank_with_learning,
    rerank_with_session,
    get_recursive_reranker,
    _get_learning_reranker,
    HAS_ONNX,
    HAS_RERANKER_FACTORY,
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
    "CosineAlphaScheduler",
    "LearnedAlphaWeights",
    "RecursiveReranker",
    "ONNXRecursiveReranker",
    "FastEmbedRecursiveReranker",
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
    "HAS_RERANKER_FACTORY",
]
