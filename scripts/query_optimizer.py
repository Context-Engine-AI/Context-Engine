#!/usr/bin/env python3
"""
Dynamic Query Performance Optimizer

Implements adaptive HNSW_EF tuning and intelligent query routing to optimize
retrieval performance based on query complexity and collection characteristics.
"""

import os
import re
import time
import math
import threading
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger("query_optimizer")


class QueryType(Enum):
    """Classification of query types for optimized routing."""
    SIMPLE = "simple"  # Simple keyword, exact match likely
    SEMANTIC = "semantic"  # Natural language, needs deep search
    COMPLEX = "complex"  # Multi-faceted, benefits from extensive search
    HYBRID = "hybrid"  # Mix of keywords and semantic


@dataclass
class QueryProfile:
    """Profile of a query for optimization decisions."""
    query: str
    query_type: QueryType
    complexity_score: float
    recommended_ef: int
    use_dense_only: bool
    estimated_latency_ms: float


@dataclass
class OptimizationStats:
    """Statistics for monitoring optimizer performance."""
    total_queries: int = 0
    simple_queries: int = 0
    semantic_queries: int = 0
    complex_queries: int = 0
    hybrid_queries: int = 0
    avg_ef_used: float = 0.0
    total_latency_ms: float = 0.0
    cache_hits: int = 0


class QueryOptimizer:
    """
    Adaptive query optimizer that dynamically tunes HNSW_EF and routing.
    
    Features:
    - Query complexity analysis
    - Dynamic HNSW_EF calculation based on query type
    - Intelligent routing between dense and hybrid search
    - Performance monitoring and adaptive learning
    """
    
    def __init__(
        self,
        base_ef: int = 128,
        min_ef: int = 64,
        max_ef: int = 512,
        collection_size: int = 10000,
        enable_adaptive: bool = True
    ):
        """
        Initialize the query optimizer.
        
        Args:
            base_ef: Default HNSW_EF value
            min_ef: Minimum allowed EF value
            max_ef: Maximum allowed EF value
            collection_size: Approximate collection size for scaling
            enable_adaptive: Enable adaptive EF tuning
        """
        self.base_ef = base_ef
        self.min_ef = min_ef
        self.max_ef = max_ef
        self.collection_size = collection_size
        self.enable_adaptive = enable_adaptive
        
        # Statistics tracking
        self.stats = OptimizationStats()
        self._query_cache: Dict[str, QueryProfile] = {}
        self._performance_history: List[Tuple[float, int, float]] = []  # (complexity, ef, latency)
        
        # Load configuration from environment
        self._load_config()
        
        logger.info(
            f"QueryOptimizer initialized: base_ef={base_ef}, range=[{min_ef}, {max_ef}], "
            f"adaptive={enable_adaptive}"
        )
    
    def _load_config(self):
        """Load optimizer configuration from environment variables."""
        self.enable_adaptive = os.environ.get("QUERY_OPTIMIZER_ADAPTIVE", "1").lower() in {
            "1", "true", "yes", "on"
        }
        
        # Complexity thresholds for query classification
        self.simple_threshold = float(os.environ.get("QUERY_OPTIMIZER_SIMPLE_THRESHOLD", "0.3") or 0.3)
        self.complex_threshold = float(os.environ.get("QUERY_OPTIMIZER_COMPLEX_THRESHOLD", "0.7") or 0.7)
        
        # EF scaling factors
        self.simple_ef_factor = float(os.environ.get("QUERY_OPTIMIZER_SIMPLE_FACTOR", "0.5") or 0.5)
        self.semantic_ef_factor = float(os.environ.get("QUERY_OPTIMIZER_SEMANTIC_FACTOR", "1.0") or 1.0)
        self.complex_ef_factor = float(os.environ.get("QUERY_OPTIMIZER_COMPLEX_FACTOR", "2.0") or 2.0)
        
        # Dense-only routing threshold (lower complexity = prefer dense)
        self.dense_only_threshold = float(os.environ.get("QUERY_OPTIMIZER_DENSE_THRESHOLD", "0.2") or 0.2)
        
        if os.environ.get("DEBUG_QUERY_OPTIMIZER"):
            logger.debug(f"Optimizer config loaded: adaptive={self.enable_adaptive}, thresholds=({self.simple_threshold}, {self.complex_threshold})")
    
    def analyze_query(self, query: str, language: Optional[str] = None) -> QueryProfile:
        """
        Analyze query and generate optimization profile.
        
        Args:
            query: Query string to analyze
            language: Optional programming language hint
        
        Returns:
            QueryProfile with optimization recommendations
        """
        # Check cache first
        cache_key = f"{query}:{language or ''}"
        if cache_key in self._query_cache:
            self.stats.cache_hits += 1
            return self._query_cache[cache_key]
        
        # Calculate complexity score
        complexity = self._calculate_complexity(query, language)
        
        # Classify query type
        query_type = self._classify_query(query, complexity)
        
        # Calculate optimal EF
        recommended_ef = self._calculate_optimal_ef(complexity, query_type)
        
        # Decide on routing
        use_dense_only = self._should_use_dense_only(query, complexity, query_type)
        
        # Estimate latency (rough heuristic)
        estimated_latency = self._estimate_latency(complexity, recommended_ef, use_dense_only)
        
        profile = QueryProfile(
            query=query,
            query_type=query_type,
            complexity_score=complexity,
            recommended_ef=recommended_ef,
            use_dense_only=use_dense_only,
            estimated_latency_ms=estimated_latency
        )
        
        # Cache the profile
        if len(self._query_cache) < 1000:  # Limit cache size
            self._query_cache[cache_key] = profile
        
        # Update stats
        self.stats.total_queries += 1
        if query_type == QueryType.SIMPLE:
            self.stats.simple_queries += 1
        elif query_type == QueryType.SEMANTIC:
            self.stats.semantic_queries += 1
        elif query_type == QueryType.COMPLEX:
            self.stats.complex_queries += 1
        else:
            self.stats.hybrid_queries += 1
        
        if os.environ.get("DEBUG_QUERY_OPTIMIZER"):
            logger.debug(
                f"Query analyzed: type={query_type.value}, complexity={complexity:.3f}, "
                f"ef={recommended_ef}, dense_only={use_dense_only}"
            )
        
        return profile
    
    def _calculate_complexity(self, query: str, language: Optional[str] = None) -> float:
        """
        Calculate query complexity score (0.0 to 1.0).
        
        Higher scores indicate more complex queries needing deeper search.
        
        Factors:
        - Query length (longer = more complex)
        - Number of terms
        - Natural language indicators (questions, connectors)
        - Code-specific patterns (operators, symbols)
        - Language-specific keywords
        """
        score = 0.0
        query_lower = query.lower().strip()
        
        # 1. Length factor (normalize to ~100 chars)
        length_score = min(len(query) / 100.0, 1.0)
        score += length_score * 0.2
        
        # 2. Term count (more terms = more complex)
        terms = re.findall(r'\b\w+\b', query)
        term_score = min(len(terms) / 10.0, 1.0)
        score += term_score * 0.15
        
        # 3. Natural language indicators
        question_words = ['what', 'how', 'why', 'when', 'where', 'which', 'who', 'explain', 'describe']
        if any(word in query_lower for word in question_words):
            score += 0.2
        
        # Connectors indicate complex multi-part queries
        connectors = ['and', 'or', 'but', 'with', 'that', 'also', 'including']
        connector_count = sum(1 for word in connectors if f' {word} ' in f' {query_lower} ')
        score += min(connector_count * 0.1, 0.2)
        
        # 4. Code-specific complexity
        # Special characters often mean precise searches
        special_chars = len(re.findall(r'[(){}[\]<>.:;,]', query))
        if special_chars > 0:
            score -= 0.1  # Special chars = more specific = simpler
        
        # CamelCase or snake_case (likely looking for specific symbols)
        if re.search(r'[A-Z][a-z]+[A-Z]', query) or '_' in query:
            score -= 0.1
        
        # Quoted strings (exact matches)
        if '"' in query or "'" in query:
            score -= 0.15
        
        # 5. Language-specific adjustments
        if language:
            # If language specified, query is more focused
            score -= 0.1
        
        # 6. Regex patterns (very specific)
        if re.search(r'[*+?\\|^$]', query):
            score -= 0.15
        
        # Clamp to [0, 1]
        return max(0.0, min(1.0, score))
    
    def _classify_query(self, query: str, complexity: float) -> QueryType:
        """Classify query type based on complexity and patterns."""
        query_lower = query.lower().strip()
        
        # Simple: Low complexity, likely exact match
        if complexity < self.simple_threshold:
            # Extra checks for simple patterns
            if re.match(r'^[a-z_][a-z0-9_]*$', query_lower):  # Single identifier
                return QueryType.SIMPLE
            if len(query.split()) <= 2 and not any(c in query for c in '(){}[]'):
                return QueryType.SIMPLE
        
        # Complex: High complexity, multi-faceted
        if complexity > self.complex_threshold:
            return QueryType.COMPLEX
        
        # Semantic: Natural language questions
        question_indicators = ['what', 'how', 'why', 'explain', 'describe', 'show me', 'find all']
        if any(query_lower.startswith(ind) for ind in question_indicators):
            return QueryType.SEMANTIC
        
        # Hybrid: Everything else
        return QueryType.HYBRID
    
    def _calculate_optimal_ef(self, complexity: float, query_type: QueryType) -> int:
        """
        Calculate optimal HNSW_EF based on complexity and query type.
        
        Strategy:
        - Simple queries: Lower EF for speed
        - Complex queries: Higher EF for quality
        - Scale with collection size
        """
        if not self.enable_adaptive:
            return self.base_ef
        
        # Base factor by query type
        if query_type == QueryType.SIMPLE:
            factor = self.simple_ef_factor
        elif query_type == QueryType.SEMANTIC:
            factor = self.semantic_ef_factor
        elif query_type == QueryType.COMPLEX:
            factor = self.complex_ef_factor
        else:  # HYBRID
            factor = (self.simple_ef_factor + self.semantic_ef_factor) / 2
        
        # Adjust by complexity within type
        complexity_adjustment = 0.5 + (complexity * 0.5)  # Range [0.5, 1.0]
        
        # Calculate EF
        calculated_ef = int(self.base_ef * factor * complexity_adjustment)
        
        # Collection size scaling (larger collections may benefit from higher EF)
        if self.collection_size > 100000:
            scale_factor = min(1.5, 1.0 + (self.collection_size / 1000000.0))
            calculated_ef = int(calculated_ef * scale_factor)
        
        # Clamp to valid range
        return max(self.min_ef, min(self.max_ef, calculated_ef))
    
    def _should_use_dense_only(
        self, query: str, complexity: float, query_type: QueryType
    ) -> bool:
        """
        Decide whether to use dense-only search vs hybrid.
        
        Dense-only is faster but may miss exact matches.
        Hybrid is more thorough but slower.
        
        Returns:
            True if dense-only search is recommended
        """
        # Very simple queries benefit from hybrid (lexical matching)
        if complexity < self.dense_only_threshold:
            return False
        
        # Natural language questions: dense is fine
        if query_type == QueryType.SEMANTIC:
            return True
        
        # Has special characters or exact match indicators: use hybrid
        if any(char in query for char in ['"', "'", '(', ')', '{', '}', '[', ']']):
            return False
        
        # CamelCase or specific symbol names: use hybrid
        if re.search(r'[A-Z][a-z]+[A-Z]', query) or '_' in query:
            return False
        
        # Default: use hybrid for safety
        return False
    
    def _estimate_latency(
        self, complexity: float, ef: int, dense_only: bool
    ) -> float:
        """
        Estimate query latency in milliseconds.
        
        This is a rough heuristic based on:
        - EF value (higher = slower)
        - Dense vs hybrid (hybrid = ~1.5x slower)
        - Collection size
        """
        # Base latency per EF unit (ms)
        base_latency_per_ef = 0.1
        
        # EF contribution
        latency = ef * base_latency_per_ef
        
        # Hybrid search overhead
        if not dense_only:
            latency *= 1.5
        
        # Collection size overhead (log scale)
        if self.collection_size > 1000:
            size_factor = 1.0 + (math.log10(self.collection_size) / 10.0)
            latency *= size_factor
        
        # Complexity overhead (reranking, post-processing)
        latency += complexity * 10.0
        
        return latency
    
    def record_query_performance(
        self, complexity: float, ef: int, actual_latency_ms: float
    ):
        """
        Record actual query performance for adaptive learning.
        
        Args:
            complexity: Query complexity score
            ef: EF value used
            actual_latency_ms: Actual query latency
        """
        self._performance_history.append((complexity, ef, actual_latency_ms))
        
        # Keep last 1000 samples
        if len(self._performance_history) > 1000:
            self._performance_history = self._performance_history[-1000:]
        
        # Update rolling average
        if self._performance_history:
            total_ef = sum(h[1] for h in self._performance_history)
            self.stats.avg_ef_used = total_ef / len(self._performance_history)
            
            total_latency = sum(h[2] for h in self._performance_history)
            self.stats.total_latency_ms = total_latency
    
    def get_stats(self) -> Dict[str, Any]:
        """Get optimizer statistics for monitoring."""
        return {
            "total_queries": self.stats.total_queries,
            "query_types": {
                "simple": self.stats.simple_queries,
                "semantic": self.stats.semantic_queries,
                "complex": self.stats.complex_queries,
                "hybrid": self.stats.hybrid_queries
            },
            "avg_ef_used": round(self.stats.avg_ef_used, 2),
            "avg_latency_ms": (
                round(self.stats.total_latency_ms / len(self._performance_history), 2)
                if self._performance_history else 0.0
            ),
            "cache_hits": self.stats.cache_hits,
            "cache_hit_rate": (
                round(self.stats.cache_hits / self.stats.total_queries * 100, 2)
                if self.stats.total_queries > 0 else 0.0
            ),
            "config": {
                "adaptive_enabled": self.enable_adaptive,
                "base_ef": self.base_ef,
                "ef_range": [self.min_ef, self.max_ef],
                "collection_size": self.collection_size
            }
        }
    
    def reset_stats(self):
        """Reset statistics counters."""
        self.stats = OptimizationStats()
        self._performance_history = []
        logger.info("QueryOptimizer stats reset")


# Global optimizer instance (lazy initialization)
_optimizer: Optional[QueryOptimizer] = None
_optimizer_lock = threading.Lock()


def get_query_optimizer(
    collection_size: Optional[int] = None,
    reset: bool = False
) -> QueryOptimizer:
    """
    Get or create global query optimizer instance.
    
    Args:
        collection_size: Approximate collection size for optimization
        reset: Force recreation of optimizer
    
    Returns:
        QueryOptimizer instance
    """
    global _optimizer
    
    with _optimizer_lock:
        if _optimizer is None or reset:
            base_ef = int(os.environ.get("QDRANT_EF_SEARCH", "128") or 128)
            min_ef = int(os.environ.get("QUERY_OPTIMIZER_MIN_EF", "64") or 64)
            max_ef = int(os.environ.get("QUERY_OPTIMIZER_MAX_EF", "512") or 512)
            
            size = collection_size or int(os.environ.get("QUERY_OPTIMIZER_COLLECTION_SIZE", "10000") or 10000)

            # Read adaptive flag from environment
            enable_adaptive = os.environ.get("QUERY_OPTIMIZER_ADAPTIVE", "1").lower() in {
                "1", "true", "yes", "on"
            }

            _optimizer = QueryOptimizer(
                base_ef=base_ef,
                min_ef=min_ef,
                max_ef=max_ef,
                collection_size=size,
                enable_adaptive=enable_adaptive
            )
        
        return _optimizer


# Convenience functions for integration
def optimize_query(
    query: str,
    language: Optional[str] = None,
    collection_size: Optional[int] = None
) -> Dict[str, Any]:
    """
    Analyze query and return optimization recommendations.
    
    Returns dict with:
        - recommended_ef: Optimal HNSW_EF value
        - use_dense_only: Whether to use dense-only search
        - query_type: Classification of query
        - complexity: Complexity score
        - estimated_latency_ms: Estimated latency
    """
    optimizer = get_query_optimizer(collection_size)
    profile = optimizer.analyze_query(query, language)

    return {
        "recommended_ef": profile.recommended_ef,
        "use_dense_only": profile.use_dense_only,
        "query_type": profile.query_type.value,
        "complexity": round(profile.complexity_score, 3),
        "estimated_latency_ms": round(profile.estimated_latency_ms, 2),
        "adaptive_enabled": optimizer.enable_adaptive,
    }


def get_optimizer_stats() -> Dict[str, Any]:
    """Get current optimizer statistics."""
    if _optimizer is None:
        return {"error": "Optimizer not initialized"}
    return _optimizer.get_stats()


if __name__ == "__main__":
    # Example usage and testing
    import json
    
    # Enable debug logging
    logging.basicConfig(level=logging.DEBUG)
    os.environ["DEBUG_QUERY_OPTIMIZER"] = "1"
    
    test_queries = [
        "UserManager",  # Simple
        "function to parse json",  # Semantic
        "How does the authentication flow work and what middleware is used?",  # Complex
        "error handling in api.py",  # Hybrid
        "calculate_total_price",  # Simple with underscore
        'class named "DatabaseConnection"',  # Simple with quotes
    ]
    
    optimizer = get_query_optimizer(collection_size=50000)
    
    print("Query Optimization Analysis:")
    print("=" * 80)
    
    for query in test_queries:
        result = optimize_query(query, language="python", collection_size=50000)
        print(f"\nQuery: {query}")
        print(f"  Type: {result['query_type']}")
        print(f"  Complexity: {result['complexity']}")
        print(f"  Recommended EF: {result['recommended_ef']}")
        print(f"  Dense Only: {result['use_dense_only']}")
        print(f"  Est. Latency: {result['estimated_latency_ms']}ms")
    
    print("\n" + "=" * 80)
    print("Optimizer Statistics:")
    print(json.dumps(get_optimizer_stats(), indent=2))
