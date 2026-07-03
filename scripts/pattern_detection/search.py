"""
Pattern Search - Structural code similarity search via Qdrant.

This module provides the search interface for finding structurally similar code
across all supported languages. It integrates:

1. PatternExtractor - Extracts normalized AST structure from code
2. PatternEncoder - Converts structure to searchable vectors
3. Qdrant - Vector database for fast similarity search
4. OnlinePatternLearner - Discovers patterns as you search

Key features:
- Cross-language search: Python pattern matches Go/Rust/Java/etc.
- Example-based: "Find code like this" without writing queries
- Pattern-aware: Discovers and surfaces common idioms
- Hybrid mode: Combines structural + semantic similarity
- TOON output: Token-efficient output format when enabled
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple, Union
from collections import Counter

logger = logging.getLogger(__name__)


# =============================================================================
# Helper Classes
# =============================================================================

class ScoredPoint:
    """Wrapper to add a score attribute to scroll results for uniform handling."""
    __slots__ = ('id', 'payload', 'score')

    def __init__(self, point, score: float):
        self.id = point.id
        self.payload = point.payload
        self.score = score


def _get_line_value(primary: Optional[int], fallback: Optional[int], default: int = 1) -> int:
    """Get line number with proper None handling (0 is valid, None falls back)."""
    if primary is not None:
        return primary
    if fallback is not None:
        return fallback
    return default


# =============================================================================
# TOON Support
# =============================================================================

def _is_toon_enabled() -> bool:
    """Check if TOON output format is enabled globally via TOON_ENABLED env var."""
    return os.environ.get("TOON_ENABLED", "0").lower() in ("1", "true", "yes")


def _should_use_toon(output_format: Any) -> bool:
    """Determine if TOON format should be used based on explicit param or env flag."""
    if output_format is not None:
        fmt = str(output_format).strip().lower()
        return fmt == "toon"
    return _is_toon_enabled()


def _format_pattern_results_as_toon(
    response: Dict[str, Any],
    compact: bool = False,
) -> Dict[str, Any]:
    """Convert pattern search response to TOON format.

    Args:
        response: Pattern search response dict with 'results' key
        compact: If True, use minimal fields only

    Returns:
        Modified response with TOON-encoded results
    """
    try:
        results = response.get("results", [])
        if isinstance(results, list):
            # Encode results to TOON format
            toon_results = encode_pattern_results(results, compact=compact)
            response["results"] = toon_results
        response["output_format"] = "toon"
        return response
    except Exception as e:
        logger.debug(f"TOON encoding failed: {e}")
        return response


def encode_pattern_results(
    results: List[Dict[str, Any]],
    delimiter: str = ",",
    compact: bool = False,
) -> str:
    """Encode pattern search results to TOON tabular format.

    Args:
        results: List of pattern search result dicts
        delimiter: Field delimiter (default: ",")
        compact: If True, only include core location fields

    Returns:
        TOON-formatted pattern results string
    """
    if not results:
        return "results[0]:"

    # Determine fields based on compact mode
    if compact:
        fields = ["path", "start_line", "end_line", "score", "language", "matched_lines"]
    else:
        # Full fields for pattern results
        fields = [
            "path", "start_line", "end_line", "score", "language",
            "control_flow_signature", "matched_patterns", "matched_lines", "snippet",
            "semantic_score", "combined_score"
        ]

    # Filter to fields actually present
    all_present: set = set()
    for r in results:
        all_present.update(r.keys())
    fields = [f for f in fields if f in all_present]

    # Build TOON output
    bracket = f"[{len(results)}]"
    fields_part = "{" + delimiter.join(fields) + "}"

    lines = [f"results{bracket}{fields_part}:"]
    for r in results:
        values = []
        for f in fields:
            val = r.get(f)
            values.append(_encode_toon_value(val, delimiter))
        lines.append(f"  {delimiter.join(values)}")

    return "\n".join(lines)


def _encode_toon_value(value: Any, delimiter: str) -> str:
    """Encode a single value to TOON format."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)
    if isinstance(value, str):
        # Check if quoting needed
        needs_quote = (
            not value or
            value[0].isspace() or value[-1].isspace() or
            value in ("true", "false", "null") or
            delimiter in value or
            any(c in value for c in (':', '"', '\\', '[', ']', '{', '}', '\n', '\r', '\t'))
        )
        if needs_quote:
            escaped = value.replace('\\', '\\\\').replace('"', '\\"')
            escaped = escaped.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
            return f'"{escaped}"'
        return value
    if isinstance(value, list):
        # Compact list encoding
        if not value:
            return "[]"
        items = [_encode_toon_value(v, delimiter) for v in value]
        return "[" + ";".join(items) + "]"
    # Objects: compact JSON fallback
    import json
    return json.dumps(value, separators=(",", ":"))

# Lazy imports for optional dependencies
_qdrant_client = None
_extractor = None
_encoder = None
_learner = None


def _get_qdrant_client():
    """Get or create Qdrant client."""
    global _qdrant_client
    if _qdrant_client is None:
        try:
            from qdrant_client import QdrantClient
            # Support QDRANT_URL (docker/k8s) or QDRANT_HOST/PORT (local dev)
            url = os.environ.get("QDRANT_URL")
            if url:
                _qdrant_client = QdrantClient(url=url)
            else:
                host = os.environ.get("QDRANT_HOST", "localhost")
                port = int(os.environ.get("QDRANT_PORT", "6333"))
                _qdrant_client = QdrantClient(host=host, port=port)
        except Exception as e:
            logger.warning(f"Failed to connect to Qdrant: {e}")
            return None
    return _qdrant_client


def _get_extractor():
    """Get or create pattern extractor."""
    global _extractor
    if _extractor is None:
        from .extractor import PatternExtractor
        _extractor = PatternExtractor()
    return _extractor


def _get_encoder():
    """Get or create pattern encoder."""
    global _encoder
    if _encoder is None:
        from .encoder import PatternEncoder
        _encoder = PatternEncoder()
    return _encoder


def _get_learner():
    """Get or create pattern learner."""
    global _learner
    if _learner is None:
        from .catalog import get_pattern_learner
        _learner = get_pattern_learner()
    return _learner


@dataclass
class PatternSearchResult:
    """A single result from pattern search."""
    path: str
    start_line: int
    end_line: int
    score: float  # Structural similarity score (0-1)
    language: str
    snippet: Optional[str] = None

    # Pattern analysis
    matched_patterns: List[str] = field(default_factory=list)
    control_flow_signature: str = ""

    # Line numbers within snippet that match query pattern (absolute, 1-indexed)
    matched_lines: List[int] = field(default_factory=list)

    # Combined scoring (if hybrid search)
    semantic_score: Optional[float] = None
    combined_score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        d = {
            "path": self.path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "score": round(self.score, 4),
            "language": self.language,
            "snippet": self.snippet,
            "matched_patterns": self.matched_patterns,
            "control_flow_signature": self.control_flow_signature,
            "semantic_score": round(self.semantic_score, 4) if self.semantic_score is not None else None,
            "combined_score": round(self.combined_score, 4) if self.combined_score is not None else None,
        }
        if self.matched_lines:
            d["matched_lines"] = self.matched_lines
        return d


@dataclass
class PatternSearchResponse:
    """Response from pattern search."""
    results: List[PatternSearchResult]
    total: int
    query_signature: str  # Control flow signature of query
    discovered_patterns: List[str] = field(default_factory=list)
    languages_searched: List[str] = field(default_factory=list)
    search_mode: str = "structural"  # structural, semantic, hybrid, error
    output_format: str = "json"  # json or toon

    def to_dict(self, compact: bool = False) -> Dict[str, Any]:
        """Convert to dictionary for JSON/TOON serialization."""
        # ok=False when search_mode is "error" to signal failure to callers
        is_ok = self.search_mode != "error"
        return {
            "ok": is_ok,
            "results": [r.to_dict() for r in self.results],
            "total": self.total,
            "query_signature": self.query_signature,
            "discovered_patterns": self.discovered_patterns,
            "languages_searched": self.languages_searched,
            "search_mode": self.search_mode,
            "output_format": self.output_format,
        }

    def format(self, output_format: Any = None, compact: bool = False) -> Dict[str, Any]:
        """Format response as JSON or TOON based on output_format parameter."""
        response = self.to_dict(compact=compact)

        if _should_use_toon(output_format):
            return _format_pattern_results_as_toon(response, compact=compact)
        return response


# =============================================================================
# Core Search Functions
# =============================================================================

def pattern_search(
    example: str,
    language: str = "python",
    *,
    limit: int = 10,
    collection: Optional[str] = None,
    min_score: float = 0.5,
    include_snippet: bool = True,
    context_lines: int = 3,
    target_languages: Optional[List[str]] = None,
    hybrid: bool = False,
    semantic_weight: float = 0.3,
    output_format: Any = None,  # "json" (default) or "toon"
    compact: bool = False,
    client: Any = None,  # Optional QdrantClient override for testing
    aroma_rerank: bool = True,  # Enable AROMA-style pruning + reranking (default ON)
    aroma_alpha: float = 0.6,  # Weight for pruned similarity (vs original score)
) -> Union[PatternSearchResponse, Dict[str, Any]]:
    """
    Find code structurally similar to the given example.

    This is the primary search interface. Given a code example, it finds
    other code with similar structure across ALL supported languages.

    Args:
        example: Code snippet to find similar code for
        language: Language of the example code
        limit: Maximum results to return
        collection: Qdrant collection (defaults to COLLECTION_NAME env)
        min_score: Minimum similarity score (0-1)
        include_snippet: Include code snippets in results
        context_lines: Lines of context around matches
        target_languages: Filter to specific languages (None = all)
        hybrid: Combine structural + semantic similarity
        semantic_weight: Weight for semantic score in hybrid mode (0-1)
        output_format: "json" (default) or "toon" for token-efficient format
        compact: If True with TOON, use minimal fields only
        aroma_rerank: Enable AROMA-style pruning + reranking (default True)
        aroma_alpha: Weight for pruned similarity vs original score (0-1, default 0.6)

    Returns:
        PatternSearchResponse or Dict (TOON format) with matching code and patterns

    Example:
        >>> results = pattern_search('''
        ... for i in range(retries):
        ...     try:
        ...         return do_request()
        ...     except Exception:
        ...         time.sleep(2 ** i)
        ... ''', language="python", output_format="toon")
        >>> # Finds retry patterns in Python, Go, Rust, Java, etc.
    """
    # Determine if TOON output is requested
    use_toon = _should_use_toon(output_format)

    # AROMA/hybrid reranking requires snippets for pruning/semantic scoring
    if (aroma_rerank or hybrid) and not include_snippet:
        logger.debug("Forcing include_snippet=True (required for AROMA/hybrid scoring)")
        include_snippet = True

    extractor = _get_extractor()
    encoder = _get_encoder()
    # Use provided client or fall back to global
    if client is None:
        client = _get_qdrant_client()

    if client is None:
        response = PatternSearchResponse(
            results=[],
            total=0,
            query_signature="ERROR:NO_QDRANT",
            search_mode="error",
        )
        return response.format(output_format, compact) if use_toon else response

    # Extract structural signature from example
    signature = extractor.extract(example, language)
    query_vector = encoder.encode(signature)
    cf_sig = signature.control_flow.get("signature", "")

    # Determine collection
    if collection is None:
        collection = os.environ.get("COLLECTION_NAME", "codebase")

    # Build search filter
    search_filter = None
    if target_languages:
        from qdrant_client.models import Filter, FieldCondition, MatchAny
        search_filter = Filter(
            must=[
                FieldCondition(
                    key="language",
                    match=MatchAny(any=target_languages)
                )
            ]
        )

    try:
        # Search for similar patterns
        # First check if collection has pattern_vector field
        collection_info = client.get_collection(collection)
        vectors_config = collection_info.config.params.vectors
        # vectors can be a dict (named) or VectorParams (single unnamed vector)
        has_pattern_vector = (
            isinstance(vectors_config, dict) and "pattern_vector" in vectors_config
        )

        # Fetch size: over-fetch only when reranking is enabled
        needs_reranking = aroma_rerank or hybrid
        fetch_limit = limit * 3 if needs_reranking else limit

        if has_pattern_vector:
            # Use dedicated pattern vector
            response = client.query_points(
                collection_name=collection,
                query=query_vector,
                using="pattern_vector",
                limit=fetch_limit,
                query_filter=search_filter,
                with_payload=True,
            )
            results = response.points
        else:
            # Fallback: search using semantic vector with pattern-based reranking
            results = _fallback_pattern_search(
                client, collection, query_vector, signature,
                fetch_limit, search_filter
            )

        # Process candidates - collect full pool when reranking, early-stop otherwise
        search_results = []
        seen_paths = set()

        for hit in results:
            payload = hit.payload or {}
            # Support both flat payload (path at top level) and nested (metadata.path)
            meta = payload.get("metadata", {})
            path = meta.get("path") or payload.get("path") or payload.get("file_path", "")

            # Deduplicate: prefer path, fall back to point id when path is empty
            dedup_key = path if path else str(getattr(hit, 'id', id(hit)))
            if dedup_key in seen_paths:
                continue
            seen_paths.add(dedup_key)

            result = PatternSearchResult(
                path=path,
                start_line=_get_line_value(meta.get("start_line"), payload.get("start_line"), 1),
                end_line=_get_line_value(meta.get("end_line"), payload.get("end_line"), 1),
                score=hit.score,
                language=meta.get("language") or payload.get("language", "unknown"),
                control_flow_signature=payload.get("cf_signature", ""),
            )

            # Include snippet if requested (forced True when aroma_rerank - see line ~366)
            if include_snippet:
                result.snippet = _get_snippet(
                    path, result.start_line, result.end_line, context_lines
                )
                # Highlight lines matching query pattern (AST-based)
                if result.snippet and signature.control_flow:
                    result.matched_lines = extractor.find_matching_lines(
                        result.snippet,
                        result.language,
                        signature.control_flow,
                        line_offset=result.start_line - 1,
                    )

            search_results.append(result)

            # Early stop only when:
            # 1. Not reranking (reranking needs full pool to reorder)
            # 2. No min_score filter (filtering may discard results, need extras)
            if not needs_reranking and min_score <= 0 and len(search_results) >= limit:
                break

        # AROMA-style reranking: prune each result w.r.t. query, rerank by pruned similarity
        if aroma_rerank and search_results:
            search_results = _apply_aroma_reranking(
                search_results, signature, extractor, aroma_alpha
            )

        # Hybrid mode: blend AROMA score (if present) with semantic score
        if hybrid and search_results:
            search_results = _apply_hybrid_scoring(
                search_results, example, language, semantic_weight
            )

        # Apply min_score filter AFTER reranking (on combined_score if available)
        if min_score > 0:
            search_results = [
                r for r in search_results
                if (r.combined_score if r.combined_score is not None else r.score) >= min_score
            ]

        # Final slice to requested limit
        search_results = search_results[:limit]

        # Discover patterns from results
        discovered = _discover_patterns_from_results(search_results, signature)

        search_mode = "structural"
        if aroma_rerank:
            search_mode = "aroma" if not hybrid else "aroma_hybrid"
        elif hybrid:
            search_mode = "hybrid"

        response = PatternSearchResponse(
            results=search_results,
            total=len(search_results),
            query_signature=cf_sig,
            discovered_patterns=discovered,
            languages_searched=list(set(r.language for r in search_results)),
            search_mode=search_mode,
        )
        return response.format(output_format, compact) if use_toon else response

    except Exception as e:
        logger.error(f"Pattern search failed: {e}")
        response = PatternSearchResponse(
            results=[],
            total=0,
            query_signature=cf_sig,
            search_mode="error",
        )
        return response.format(output_format, compact) if use_toon else response


def find_similar_patterns(
    code: str,
    language: str = "python",
    *,
    limit: int = 5,
    collection: Optional[str] = None,
    output_format: Any = None,
    compact: bool = False,
) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Simplified interface: find code with similar patterns.

    Args:
        code: Code snippet to find similar patterns for
        language: Language of the code
        limit: Maximum results
        collection: Qdrant collection
        output_format: "json" (default) or "toon" for token-efficient format
        compact: If True with TOON, use minimal fields

    Returns:
        List of dicts with path, score, snippet (JSON) or TOON-formatted dict
    """
    response = pattern_search(
        example=code,
        language=language,
        limit=limit,
        collection=collection,
        include_snippet=True,
        output_format=output_format,
        compact=compact,
    )
    # If TOON was used, response is already formatted
    if isinstance(response, dict):
        return response
    return [r.to_dict() for r in response.results]


def search_by_pattern_description(
    description: str,
    *,
    limit: int = 10,
    min_score: float = 0.0,
    collection: Optional[str] = None,
    target_languages: Optional[List[str]] = None,
    output_format: Any = None,
    compact: bool = False,
) -> Union[PatternSearchResponse, Dict[str, Any]]:
    """
    Search using natural language pattern description.

    Examples:
        - "retry with exponential backoff"
        - "resource cleanup with finally"
        - "null check guard clause"

    Args:
        description: Natural language description of the pattern
        limit: Maximum results
        min_score: Minimum similarity score (0-1), applied before TOON encoding
        collection: Qdrant collection
        target_languages: Filter to specific languages
        output_format: "json" (default) or "toon" for token-efficient format
        compact: If True with TOON, use minimal fields

    This uses the OnlinePatternLearner to match descriptions to
    discovered patterns, then searches for code matching those patterns.
    """
    use_toon = _should_use_toon(output_format)
    learner = _get_learner()

    # Find patterns matching the description
    matched_patterns = learner.natural_language_query(description, top_k=3)

    if not matched_patterns:
        # Fallback: use description as pseudo-code
        return pattern_search(
            example=description,
            language="python",
            limit=limit,
            min_score=min_score,
            collection=collection,
            target_languages=target_languages,
            output_format=output_format,
            compact=compact,
        )

    # Use best matching pattern's centroid for search
    best_pattern = matched_patterns[0]

    client = _get_qdrant_client()
    if client is None or not best_pattern.centroid:
        # Mark as error so ok=False propagates to callers
        response = PatternSearchResponse(
            results=[],
            total=0,
            query_signature="NL:" + description[:30],
            discovered_patterns=[p.auto_description for p in matched_patterns],
            search_mode="error",
        )
        return response.format(output_format, compact) if use_toon else response

    if collection is None:
        collection = os.environ.get("COLLECTION_NAME", "codebase")

    # Build filter
    search_filter = None
    if target_languages:
        from qdrant_client.models import Filter, FieldCondition, MatchAny
        search_filter = Filter(
            must=[FieldCondition(key="language", match=MatchAny(any=target_languages))]
        )

    try:
        # Check if collection has pattern_vector field
        collection_info = client.get_collection(collection)
        vectors_config = collection_info.config.params.vectors
        has_pattern_vector = (
            isinstance(vectors_config, dict) and "pattern_vector" in vectors_config
        )

        if has_pattern_vector:
            response = client.query_points(
                collection_name=collection,
                query=best_pattern.centroid,
                using="pattern_vector",
                limit=limit,
                query_filter=search_filter,
                with_payload=True,
                with_vectors=False,  # Only need payload and scores
            )
            results = response.points
        else:
            # No pattern_vector in collection - use scroll + in-memory reranking
            # Cannot use vector search: pattern centroid (64-dim) != semantic vector (384-dim)
            logger.debug("No pattern_vector field, using scroll + rerank for NL search")
            scroll_results, _ = client.scroll(
                collection_name=collection,
                scroll_filter=search_filter,
                limit=limit * 3,
                with_payload=True,
                with_vectors=False,
            )
            # Rerank by pattern description match (simple keyword overlap)
            description_words = set(description.lower().split())
            scored_results = []
            for point in scroll_results:
                payload = point.payload or {}
                meta = payload.get("metadata", {})
                # Check common text field names used in code collections
                text = (
                    payload.get("text", "") or
                    payload.get("content", "") or
                    payload.get("code", "") or
                    meta.get("text", "") or
                    meta.get("code", "")
                )
                text_words = set(text.lower().split())
                overlap = len(description_words & text_words) / max(len(description_words), 1)
                # Wrap in ScoredPoint to preserve the overlap score
                scored_results.append(ScoredPoint(point, overlap))
            scored_results.sort(key=lambda x: x.score, reverse=True)
            results = scored_results[:limit]

        search_results = []
        for hit in results:
            payload = hit.payload or {}
            # Support both flat payload and nested metadata (consistent with pattern_search)
            meta = payload.get("metadata", {})
            path = meta.get("path") or payload.get("path") or payload.get("file_path", "")
            # Use .score attribute (works for both search results and ScoredPoint)
            score = getattr(hit, 'score', 0.5)
            search_results.append(PatternSearchResult(
                path=path,
                start_line=_get_line_value(meta.get("start_line"), payload.get("start_line"), 1),
                end_line=_get_line_value(meta.get("end_line"), payload.get("end_line"), 1),
                score=score,
                language=meta.get("language") or payload.get("language", "unknown"),
                matched_patterns=[best_pattern.auto_description],
            ))

        # Apply min_score filtering BEFORE TOON encoding (so it works regardless of output format)
        if min_score > 0:
            search_results = [r for r in search_results if r.score >= min_score]

        response = PatternSearchResponse(
            results=search_results,
            total=len(search_results),
            query_signature="NL:" + description[:30],
            discovered_patterns=[p.auto_description for p in matched_patterns],
            languages_searched=list(set(r.language for r in search_results)),
            search_mode="natural_language",
        )
        return response.format(output_format, compact) if use_toon else response

    except Exception as e:
        logger.error(f"NL pattern search failed: {e}")
        response = PatternSearchResponse(
            results=[],
            total=0,
            query_signature="NL:" + description[:30],
            search_mode="error",
        )
        return response.format(output_format, compact) if use_toon else response



# =============================================================================
# Helper Functions
# =============================================================================

def _fallback_pattern_search(
    client,
    collection: str,
    pattern_vector: List[float],
    signature,
    limit: int,
    search_filter,
) -> List:
    """
    Fallback search when collection doesn't have pattern_vector field.

    Uses scroll with in-memory pattern reranking since we can't do
    vector search with mismatched dimensions.

    Note: limit is already inflated by caller for reranking; don't multiply again.
    """
    # Can't use pattern_vector (64-dim) against semantic vectors (384-dim)
    # Instead, scroll through relevant documents and rerank by pattern similarity
    try:
        # Scroll with filter to get candidate documents
        # limit is pre-inflated by caller (e.g., limit*3 for reranking)
        semantic_results, _ = client.scroll(
            collection_name=collection,
            scroll_filter=search_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
    except Exception as e:
        logger.debug(f"Fallback scroll failed: {e}")
        return []

    # Rerank by structural similarity using cf_signature in payload
    query_cf = signature.control_flow.get("normalized_sequence", [])

    # Baseline score for documents without cf_sequence or when query has no CF nodes:
    # - Set BELOW default min_score (0.5) so unverified matches don't pass by default
    # - This prioritizes precision: only items with actual pattern similarity surface
    # - Callers can lower min_score (e.g., 0.3) to include these fallback results
    BASELINE_SCORE_NO_CF = 0.4

    reranked = []
    for point in semantic_results:
        payload = point.payload or {}

        # Calculate similarity from stored pattern info
        stored_cf = payload.get("cf_sequence", [])
        if stored_cf and query_cf:
            # Jaccard similarity of control flow sequences
            set_query = set(query_cf)
            set_stored = set(stored_cf)
            intersection = len(set_query & set_stored)
            union = len(set_query | set_stored)
            cf_similarity = intersection / union if union > 0 else 0.0
        elif not stored_cf or not query_cf:
            # No cf_sequence in payload OR query has no control-flow nodes
            # Assign baseline score - filtered by default min_score unless caller lowers it
            cf_similarity = BASELINE_SCORE_NO_CF

        reranked.append(ScoredPoint(point, cf_similarity))

    # Sort by score and return top results
    reranked.sort(key=lambda x: x.score, reverse=True)
    return reranked[:limit]


def _get_snippet(
    path: str,
    start_line: int,
    end_line: int,
    context_lines: int,
) -> Optional[str]:
    """Read snippet from file with context."""
    try:
        # Resolve path
        workspace = os.environ.get("WORKSPACE_PATH", "/work")
        full_path = os.path.join(workspace, path) if not os.path.isabs(path) else path

        if not os.path.exists(full_path):
            return None

        with open(full_path, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()

        # Calculate range with context
        start = max(0, start_line - 1 - context_lines)
        end = min(len(lines), end_line + context_lines)

        return ''.join(lines[start:end])

    except Exception as e:
        logger.debug(f"Failed to read snippet from {path}: {e}")
        return None


def _apply_hybrid_scoring(
    results: List[PatternSearchResult],
    query: str,
    language: str,
    semantic_weight: float,
) -> List[PatternSearchResult]:
    """
    Apply hybrid scoring combining structural/AROMA and semantic similarity.

    If AROMA reranking ran first, blends AROMA's combined_score with semantic.
    Otherwise blends raw structural score with semantic.
    """
    try:
        # Get embeddings for semantic comparison
        from scripts.embeddings import get_embedding_model
        model = get_embedding_model()
        query_embedding = model.encode(query)

        for result in results:
            if result.snippet:
                snippet_embedding = model.encode(result.snippet)
                # Cosine similarity
                dot = sum(a * b for a, b in zip(query_embedding, snippet_embedding))
                norm_q = sum(a * a for a in query_embedding) ** 0.5
                norm_s = sum(a * a for a in snippet_embedding) ** 0.5
                semantic_sim = dot / (norm_q * norm_s) if norm_q and norm_s else 0

                result.semantic_score = semantic_sim
                # Blend with existing combined_score (from AROMA) or raw score
                base_score = result.combined_score if result.combined_score is not None else result.score
                result.combined_score = (
                    (1 - semantic_weight) * base_score +
                    semantic_weight * semantic_sim
                )
            # If no snippet but has AROMA score, keep it; otherwise use raw score
            elif result.combined_score is None:
                result.combined_score = result.score

        # Re-sort by combined score
        results.sort(key=lambda r: r.combined_score or r.score, reverse=True)

    except Exception as e:
        logger.debug(f"Hybrid scoring failed, using structural only: {e}")

    return results


def _apply_aroma_reranking(
    results: List[PatternSearchResult],
    query_signature,
    extractor,
    alpha: float = 0.6,
) -> List[PatternSearchResult]:
    """
    Apply AROMA-style pruning and reranking.

    For each result:
    1. Extract its pattern signature
    2. Prune it w.r.t. query to find maximal similar subtree
    3. Compute combined score: alpha * pruned_similarity + (1-alpha) * original_score
    4. Re-sort by combined score
    """
    try:
        from .prune import AromaPruner
        pruner = AromaPruner(extractor)

        reranked = []
        for result in results:
            # Extract signature from result's snippet if available
            if result.snippet:
                result_sig = extractor.extract(result.snippet, result.language)
                prune_result = pruner.prune(query_signature, result_sig)

                # Combined score
                combined = alpha * prune_result.similarity_score + (1 - alpha) * result.score
                result.combined_score = combined

                # Store pruning info in matched_patterns for debugging
                if prune_result.similarity_score > 0:
                    result.matched_patterns.append(
                        f"aroma_sim:{prune_result.similarity_score:.2f},"
                        f"retained:{prune_result.pruned_feature_count}/{prune_result.original_feature_count}"
                    )
            else:
                # No snippet - can't extract signature, use original score
                result.combined_score = result.score

            reranked.append(result)

        # Re-sort by combined score
        reranked.sort(key=lambda r: r.combined_score or r.score, reverse=True)
        return reranked

    except Exception as e:
        logger.debug(f"AROMA reranking failed, using original order: {e}")
        return results


def _discover_patterns_from_results(
    results: List[PatternSearchResult],
    query_signature,
) -> List[str]:
    """Identify common patterns across search results."""
    if not results:
        return []

    # Count control flow signatures
    cf_counter = Counter()
    for r in results:
        if r.control_flow_signature:
            cf_counter[r.control_flow_signature] += 1

    # Return most common patterns
    common = cf_counter.most_common(3)
    return [sig for sig, count in common if count >= 2]


# =============================================================================
# Convenience Aliases
# =============================================================================

def search_similar_code(
    code: str,
    output_format: Any = None,
    compact: bool = False,
    **kwargs,
) -> Union[PatternSearchResponse, Dict[str, Any]]:
    """Alias for pattern_search with output_format support."""
    return pattern_search(code, output_format=output_format, compact=compact, **kwargs)


def find_code_like(
    example: str,
    output_format: Any = None,
    compact: bool = False,
    **kwargs,
) -> Union[PatternSearchResponse, Dict[str, Any]]:
    """Alias for pattern_search with friendlier name."""
    return pattern_search(example, output_format=output_format, compact=compact, **kwargs)


# =============================================================================
# Export
# =============================================================================

__all__ = [
    # Core search functions
    "pattern_search",
    "find_similar_patterns",
    "search_by_pattern_description",
    # Aliases
    "search_similar_code",
    "find_code_like",
    # Data classes
    "PatternSearchResult",
    "PatternSearchResponse",
    # TOON support
    "encode_pattern_results",
    "_should_use_toon",
    "_format_pattern_results_as_toon",
]
