#!/usr/bin/env python3
"""
Semantic similarity-based query expansion for Context-Engine.

This module provides intelligent query expansion using semantic similarity
to improve search relevance by finding conceptually related terms.
"""

import os
import math
import re
from typing import List, Dict, Any, Tuple, Optional, Set
from collections import defaultdict
import logging

logger = logging.getLogger("semantic_expansion")

# Import embedding functionality (prefer embedder factory for Qwen3 support)
try:
    from scripts.embedder import get_embedding_model as _get_embedding_model
    _EMBEDDER_FACTORY = True
    FASTEMBED_AVAILABLE = True
except ImportError:
    _EMBEDDER_FACTORY = False
    try:
        from fastembed import TextEmbedding
        FASTEMBED_AVAILABLE = True
    except ImportError:
        FASTEMBED_AVAILABLE = False
        TextEmbedding = None

# Import Qdrant client for vector operations
try:
    from qdrant_client import QdrantClient, models
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    QdrantClient = None
    models = None

# Import local utilities
try:
    from scripts.utils import (
        lex_hash_vector_queries as _lex_hash_vector_queries,
        sanitize_vector_name as _sanitize_vector_name,
    )
except ImportError:
    _lex_hash_vector_queries = None
    _sanitize_vector_name = None

# Configuration defaults
# NOTE: SEMANTIC_EXPANSION_ENABLED is intentionally *not* a module-level constant.
# Tests and callers may toggle SEMANTIC_EXPANSION_ENABLED at runtime; reading the
# env var at import-time makes that unreliable.
def _semantic_expansion_enabled() -> bool:
    v = (os.environ.get("SEMANTIC_EXPANSION_ENABLED", "1") or "1").strip().lower()
    return v in {"1", "true", "yes", "on"}
SEMANTIC_EXPANSION_TOP_K = int(os.environ.get("SEMANTIC_EXPANSION_TOP_K", "5") or "5")
SEMANTIC_EXPANSION_SIMILARITY_THRESHOLD = float(os.environ.get("SEMANTIC_EXPANSION_SIMILARITY_THRESHOLD", "0.7") or "0.7")
SEMANTIC_EXPANSION_MAX_TERMS = int(os.environ.get("SEMANTIC_EXPANSION_MAX_TERMS", "3") or "3")
SEMANTIC_EXPANSION_CACHE_SIZE = int(os.environ.get("SEMANTIC_EXPANSION_CACHE_SIZE", "1000") or "1000")
SEMANTIC_EXPANSION_CACHE_TTL = float(os.environ.get("SEMANTIC_EXPANSION_CACHE_TTL", "3600") or "3600")

# Use UnifiedCache for proper LRU eviction instead of simple FIFO
try:
    from scripts.cache_manager import UnifiedCache, EvictionPolicy
    _expansion_cache = UnifiedCache(
        name="semantic_expansion",
        max_size=SEMANTIC_EXPANSION_CACHE_SIZE,
        eviction_policy=EvictionPolicy.LRU,
        default_ttl=SEMANTIC_EXPANSION_CACHE_TTL,
    )
    _UNIFIED_CACHE = True
except ImportError:
    _expansion_cache: Dict[str, List[str]] = {}  # type: ignore
    _UNIFIED_CACHE = False

_cache_hits = 0
_cache_misses = 0


def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0
    
    try:
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    except Exception:
        return 0.0


def _coerce_embedding_vector(raw: Any) -> Optional[List[float]]:
    """Best-effort conversion of embedding output into a flat float list."""
    try:
        if raw is None:
            return None
        vec = raw.tolist() if hasattr(raw, "tolist") else raw
        if isinstance(vec, (list, tuple)):
            if vec and isinstance(vec[0], (list, tuple)):
                vec = vec[0]
            return [float(x) for x in vec]
        return [float(x) for x in list(vec)]
    except Exception:
        return None


def _select_vector_name(
    client: Any,
    collection: str,
    model_dim: Optional[int] = None,
    model_name: Optional[str] = None,
) -> Optional[str]:
    """Pick a dense vector name by matching embedding dimension or model name."""
    try:
        info = client.get_collection(collection)
        cfg = info.config.params.vectors
        if isinstance(cfg, dict) and cfg:
            # Try dimension match first
            if model_dim:
                for name, params in cfg.items():
                    psize = getattr(params, "size", None) or getattr(params, "dim", None)
                    if psize and int(psize) == int(model_dim):
                        return name

            # Try matching by sanitized model name
            if model_name:
                sanitized = str(model_name).replace("/", "-").replace("_", "-").lower()
                for name in cfg.keys():
                    if sanitized in name.lower() or name.lower() in sanitized:
                        return name

            # Fallback: return first non-lex vector
            lex_name = os.environ.get("LEX_VECTOR_NAME", "lex")
            mini_name = os.environ.get("MINI_VECTOR_NAME", "mini")
            for name in cfg.keys():
                if name not in (lex_name, mini_name):
                    return name
        return None
    except Exception:
        return None


def _get_expansion_cache_key(queries: List[str], language: Optional[str] = None) -> str:
    """Generate a cache key for query expansion."""
    # Normalize queries for consistent caching
    normalized = [q.lower().strip() for q in queries if q.strip()]
    lang_part = f"lang:{language}" if language else ""
    return "|".join(sorted(normalized)) + f"#{lang_part}"


def _get_cached_expansion(cache_key: str) -> Optional[List[str]]:
    """Get cached expansion results."""
    global _cache_hits
    if _UNIFIED_CACHE:
        result = _expansion_cache.get(cache_key)
        if result is not None:
            _cache_hits += 1
            return result.copy() if isinstance(result, list) else result
        return None
    else:
        if cache_key in _expansion_cache:
            _cache_hits += 1
            return _expansion_cache[cache_key].copy()
        return None


def _cache_expansion(cache_key: str, expansions: List[str]) -> None:
    """Cache expansion results with LRU eviction."""
    global _cache_misses
    _cache_misses += 1
    
    if _UNIFIED_CACHE:
        _expansion_cache.set(cache_key, expansions.copy())
    else:
        _expansion_cache[cache_key] = expansions.copy()
        if len(_expansion_cache) > SEMANTIC_EXPANSION_CACHE_SIZE:
            keys_to_remove = list(_expansion_cache.keys())[:len(_expansion_cache) - SEMANTIC_EXPANSION_CACHE_SIZE]
            for key in keys_to_remove:
                del _expansion_cache[key]


def _extract_code_tokens(text: str) -> List[str]:
    """Extract code-relevant tokens from text."""
    # Split on common delimiters and filter
    tokens = re.split(r'[^A-Za-z0-9_]+', text)
    
    # Filter and normalize tokens
    filtered = []
    for token in tokens:
        token = token.strip().lower()
        if (len(token) >= 3 and 
            not token.isdigit() and 
            token not in {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'does', 'let', 'put', 'say', 'she', 'too', 'use'}):
            filtered.append(token)
    
    return filtered


def _extract_terms_from_results(results: List[Any], max_terms: int = 20) -> List[str]:
    """Extract relevant terms from search results for expansion."""
    if not results:
        return []
    
    term_freq = defaultdict(int)
    
    for result in results[:10]:  # Limit to top 10 results for performance
        try:
            # Extract metadata
            if hasattr(result, 'payload') and result.payload:
                metadata = result.payload.get('metadata', {})
            else:
                metadata = {}
            
            # Extract text from various fields
            text_fields = [
                metadata.get('text', ''),
                metadata.get('code', ''),
                metadata.get('symbol', ''),
                metadata.get('symbol_path', ''),
                metadata.get('path', '')
            ]
            
            combined_text = ' '.join(str(field) for field in text_fields if field)
            
            # Extract tokens
            tokens = _extract_code_tokens(combined_text)
            
            # Count frequency
            for token in tokens:
                term_freq[token] += 1
                
        except Exception as e:
            logger.debug(f"Error extracting terms from result: {e}")
            continue
    
    # Sort by frequency and return top terms
    sorted_terms = sorted(term_freq.items(), key=lambda x: x[1], reverse=True)
    return [term for term, freq in sorted_terms[:max_terms]]


def _expand_with_lexical_similarity(queries: List[str], candidate_terms: List[str]) -> List[str]:
    """Expand queries using lexical similarity when embeddings aren't available."""
    expansions = []
    
    for query in queries:
        query_tokens = set(_extract_code_tokens(query))
        
        for term in candidate_terms:
            term_tokens = set(_extract_code_tokens(term))
            
            # Calculate Jaccard similarity
            intersection = query_tokens.intersection(term_tokens)
            union = query_tokens.union(term_tokens)
            
            if union:
                similarity = len(intersection) / len(union)
                if similarity >= 0.3:  # Threshold for lexical similarity
                    expansions.append(term)
    
    return expansions[:SEMANTIC_EXPANSION_MAX_TERMS]


def expand_queries_semantically(
    queries: List[str], 
    language: Optional[str] = None,
    client: Optional['QdrantClient'] = None,
    model: Optional['TextEmbedding'] = None,
    collection: Optional[str] = None,
    max_expansions: int = None
) -> List[str]:
    """
    Expand queries using semantic similarity to improve search relevance.
    
    Args:
        queries: Original query strings
        language: Optional programming language hint
        client: QdrantClient instance (optional, will create if None)
        model: TextEmbedding instance (optional, will create if None)
        collection: Collection name to search in
        max_expansions: Maximum number of expansion terms to return
        
    Returns:
        List of semantically related expansion terms
    """
    if not _semantic_expansion_enabled() or not queries:
        return []
    
    max_expansions = max_expansions or SEMANTIC_EXPANSION_MAX_TERMS
    
    # Check cache first
    cache_key = _get_expansion_cache_key(queries, language)
    cached_result = _get_cached_expansion(cache_key)
    if cached_result:
        return cached_result[:max_expansions]
    
    try:
        # Initialize components if not provided
        if client is None and QDRANT_AVAILABLE:
            qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
            api_key = os.environ.get("QDRANT_API_KEY")
            client = QdrantClient(url=qdrant_url, api_key=api_key)
        
        if model is None and FASTEMBED_AVAILABLE:
            model_name = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
            if _EMBEDDER_FACTORY:
                model = _get_embedding_model(model_name)
            else:
                model = TextEmbedding(model_name=model_name)
        else:
            # When caller injects a model, prefer its name for vector selection if available
            model_name = getattr(model, "model_name", os.environ.get("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5"))
        
        # Qdrant collections with multiple vectors require the vector name
        vector_name = None
        if collection is None:
            collection = os.environ.get("COLLECTION_NAME", "codebase")

        model_dim = getattr(model, "dim", None) if model is not None else None
        if client and collection:
            vector_name = _select_vector_name(
                client,
                collection,
                model_dim=model_dim,
                model_name=model_name,
            )

        # If we don't have the required components, fall back to lexical expansion
        if not (client and model):
            logger.debug("Semantic expansion unavailable: missing client or model")
            return []
        
        # Get initial search results to extract terms from
        # Use a hybrid approach: combine original queries for initial search
        combined_query = " ".join(queries)
        
        # Get embeddings for the query
        query_embeddings = list(model.embed([combined_query]))
        if not query_embeddings:
            return []

        # Accept either vector objects with tolist() or plain (nested) lists
        try:
            query_vector = _coerce_embedding_vector(query_embeddings[0])
        except Exception:
            query_vector = None
        if not query_vector:
            return []

        # Search for similar documents
        try:
            search_vector = query_vector
            if vector_name:
                try:
                    search_vector = models.NamedVector(name=vector_name, vector=query_vector)
                except Exception:
                    search_vector = query_vector

            search_results = client.search(
                collection_name=collection,
                query_vector=search_vector,
                limit=SEMANTIC_EXPANSION_TOP_K,
                with_payload=True,
                with_vectors=False  # We don't need vectors for term extraction
            )
        except Exception as e:
            logger.debug(f"Search failed during semantic expansion: {e}")
            return []
        
        # Extract candidate terms from search results
        candidate_terms = _extract_terms_from_results(search_results)
        
        if not candidate_terms:
            return []
        
        # Calculate semantic similarity between query and candidates
        # Get embeddings for candidate terms
        candidate_embeddings = list(model.embed(candidate_terms))
        if not candidate_embeddings:
            return []

        # Calculate similarities and filter by threshold
        similar_terms = []
        for i, term in enumerate(candidate_terms):
            if i < len(candidate_embeddings):
                try:
                    candidate_vector = _coerce_embedding_vector(candidate_embeddings[i])
                except Exception:
                    continue
                if not candidate_vector:
                    continue
                similarity = _cosine_similarity(query_vector, candidate_vector)

                if similarity >= SEMANTIC_EXPANSION_SIMILARITY_THRESHOLD:
                    similar_terms.append((term, similarity))

        # Sort by similarity and return top terms
        similar_terms.sort(key=lambda x: x[1], reverse=True)
        expansions = [term for term, _ in similar_terms[:max_expansions]]

        # Cache the result
        _cache_expansion(cache_key, expansions)

        return expansions

    except Exception as e:
        logger.debug(f"Semantic expansion failed: {e}")
        return []


def expand_queries_with_prf(
    queries: List[str],
    initial_results: List[Any],
    model: Optional['TextEmbedding'] = None,
    max_expansions: int = None
) -> List[str]:
    """
    Expand queries using pseudo-relevance feedback from initial search results.
    
    Args:
        queries: Original query strings
        initial_results: Initial search results to use for feedback
        model: TextEmbedding instance for semantic analysis
        max_expansions: Maximum number of expansion terms
        
    Returns:
        List of expansion terms derived from initial results
    """
    if not initial_results or not queries:
        return []
    
    max_expansions = max_expansions or SEMANTIC_EXPANSION_MAX_TERMS
    
    try:
        # Extract candidate terms from initial results
        candidate_terms = _extract_terms_from_results(initial_results)
        
        if not candidate_terms:
            return []
        
        # If we have a model, use semantic similarity
        if model and FASTEMBED_AVAILABLE:
            # Get embeddings for queries
            query_text = " ".join(queries)
            query_embeddings = list(model.embed([query_text]))
            
            if not query_embeddings:
                return _expand_with_lexical_similarity(queries, candidate_terms)

            query_vector = _coerce_embedding_vector(query_embeddings[0])
            if not query_vector:
                return _expand_with_lexical_similarity(queries, candidate_terms)
            
            # Get embeddings for candidates
            candidate_embeddings = list(model.embed(candidate_terms))
            
            if not candidate_embeddings:
                return _expand_with_lexical_similarity(queries, candidate_terms)
            
            # Calculate similarities
            similar_terms = []
            for i, term in enumerate(candidate_terms):
                if i < len(candidate_embeddings):
                    candidate_vector = _coerce_embedding_vector(candidate_embeddings[i])
                    if not candidate_vector:
                        continue
                    similarity = _cosine_similarity(query_vector, candidate_vector)
                    
                    if similarity >= SEMANTIC_EXPANSION_SIMILARITY_THRESHOLD:
                        similar_terms.append((term, similarity))
            
            # Sort by similarity and return top terms
            similar_terms.sort(key=lambda x: x[1], reverse=True)
            return [term for term, _ in similar_terms[:max_expansions]]
        else:
            # Fall back to lexical similarity
            return _expand_with_lexical_similarity(queries, candidate_terms)
            
    except Exception as e:
        logger.debug(f"PRF expansion failed: {e}")
        return []


def get_expansion_stats() -> Dict[str, Any]:
    """Get statistics about the expansion cache performance."""
    total_requests = _cache_hits + _cache_misses
    hit_rate = (_cache_hits / total_requests * 100) if total_requests > 0 else 0
    
    return {
        "cache_hits": _cache_hits,
        "cache_misses": _cache_misses,
        "hit_rate_percent": round(hit_rate, 2),
        "cache_size": len(_expansion_cache),
        "max_cache_size": SEMANTIC_EXPANSION_CACHE_SIZE
    }


def clear_expansion_cache() -> None:
    """Clear the expansion cache."""
    global _cache_hits, _cache_misses, _expansion_cache
    _cache_hits = 0
    _cache_misses = 0
    _expansion_cache.clear()
