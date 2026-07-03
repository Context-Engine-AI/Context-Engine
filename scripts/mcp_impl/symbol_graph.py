#!/usr/bin/env python3
"""
mcp_impl/symbol_graph.py - Symbol graph navigation for code understanding.

Provides Qdrant-native queries for:
- "who calls X" (callers)
- "where is X defined" (definition)
- "what imports Y" (importers)
- "who is called by X" (called_by - post-index computed)

Note:
This MIT-branch implementation does NOT include the larger "graph edges collection" /
GraphRAG / Neo4j-backed graph traversal stack that existed on other branches.
If you want to revisit that approach later, see related commits (by SHA) in repo history:
4da7e55, 21359d7, 795406b, 64be33e.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

__all__ = [
    "_symbol_graph_impl",
    "_format_symbol_graph_toon",
    "_compute_called_by",
]


def _parse_int_or_default(value: Any, default: int = 0) -> int:
    """Defensive integer parser that returns default on failure."""
    if value is None:
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except (ValueError, TypeError):
            return default
    return default


# Environment - use same patterns as rest of engine
QDRANT_URL = os.environ.get("QDRANT_URL", "http://qdrant:6333")


def _normalize_symbol(symbol: str) -> str:
    """Normalize a symbol name for robust matching.

    Handles:
    - Whitespace trimming
    - Qualified names (obj.method -> method for base match)
    - Common prefixes/suffixes
    """
    s = str(symbol).strip()
    if not s:
        return ""
    # Remove leading/trailing underscores for matching (but preserve for exact)
    return s


def _symbol_variants(symbol: str) -> List[str]:
    """Generate symbol variants for fuzzy matching.

    Given "MyClass.my_method", returns:
    - "MyClass.my_method" (exact)
    - "my_method" (base name)
    - "MyClass" (container)
    """
    s = _normalize_symbol(symbol)
    if not s:
        return []

    variants = [s]

    # Handle qualified names: obj.method, Class.method, module.func
    if "." in s:
        parts = s.split(".")
        # Add base name (last part)
        if parts[-1]:
            variants.append(parts[-1])
        # Add container name (first part) for class lookups
        if len(parts) >= 2 and parts[0]:
            variants.append(parts[0])

    # Handle C++/Rust namespace paths: Namespace::Class::method
    if "::" in s:
        parts = s.split("::")
        if parts[-1]:
            variants.append(parts[-1])
        if len(parts) >= 2 and parts[-2]:
            variants.append(parts[-2])

    # Handle arrow notation: obj->method
    if "->" in s:
        parts = s.split("->")
        if parts[-1]:
            variants.append(parts[-1])

    return list(dict.fromkeys(variants))  # Dedupe preserving order

def _norm_under(u: Optional[str]) -> Optional[str]:
    """Normalize an `under` path to match ingest's stored `metadata.path_prefix` values.

    This mirrors the engine's convention: normalize to a /work/... style path.
    Note: `under` in this engine is an exact directory filter (not recursive).
    """
    if not u:
        return None
    s = str(u).strip().replace("\\", "/")
    s = "/".join([p for p in s.split("/") if p])
    if not s:
        return None
    # Normalize to /work/...
    if not s.startswith("/"):
        v = "/work/" + s
    else:
        v = "/work/" + s.lstrip("/") if not s.startswith("/work/") else s
    return v.rstrip("/")


async def _symbol_graph_impl(
    symbol: str,
    query_type: str = "callers",
    limit: int = 20,
    language: Optional[str] = None,
    under: Optional[str] = None,
    collection: Optional[str] = None,
    session: Optional[str] = None,
    ctx: Any = None,
) -> Dict[str, Any]:
    """
    Query the symbol graph to find callers, definitions, or importers.

    Args:
        symbol: The symbol name to search for (function, class, module name)
        query_type: One of "callers", "definition", "importers"
        limit: Maximum number of results
        language: Optional language filter
        under: Optional path prefix filter
        collection: Optional collection override
        session: Optional session ID for collection routing
        ctx: MCP context (optional)

    Returns:
        Dict with "results" list and metadata
    """
    from qdrant_client import QdrantClient
    from qdrant_client import models as qmodels

    # Get collection using engine's standard approach
    coll = str(collection or "").strip()
    if not coll:
        try:
            from scripts.mcp_impl.workspace import _default_collection
            coll = _default_collection() or ""
        except Exception:
            coll = os.environ.get("COLLECTION_NAME", "codebase")
    if not coll:
        coll = os.environ.get("COLLECTION_NAME", "codebase")

    # Connect to Qdrant using engine's standard env vars
    try:
        client = QdrantClient(
            url=QDRANT_URL,
            api_key=os.environ.get("QDRANT_API_KEY"),
            timeout=float(os.environ.get("QDRANT_TIMEOUT", "20") or 20),
        )
    except Exception as e:
        logger.error(f"Failed to connect to Qdrant: {e}")
        return {
            "results": [],
            "error": f"Qdrant connection failed: {e}",
            "symbol": symbol,
            "query_type": query_type,
            "collection": coll,
        }

    # Validate query_type
    if query_type not in ("callers", "definition", "importers"):
        return {
            "results": [],
            "error": f"Invalid query_type: {query_type}. Use 'callers', 'definition', or 'importers'",
            "symbol": symbol,
            "query_type": query_type,
            "collection": coll,
        }

    results = []

    try:
        if query_type == "callers":
            # Find chunks where metadata.calls array contains the symbol (exact match)
            results = await _query_array_field(
                client=client,
                collection=coll,
                field_key="metadata.calls",
                value=symbol,
                limit=limit,
                language=language,
                under=_norm_under(under),
            )
        elif query_type == "definition":
            # Find chunks where symbol_path matches the symbol
            results = await _query_definition(
                client=client,
                collection=coll,
                symbol=symbol,
                limit=limit,
                language=language,
                under=_norm_under(under),
            )
        elif query_type == "importers":
            # Find chunks where metadata.imports array contains the symbol
            results = await _query_array_field(
                client=client,
                collection=coll,
                field_key="metadata.imports",
                value=symbol,
                limit=limit,
                language=language,
                under=_norm_under(under),
            )

        # If no results, fall back to semantic search
        if not results:
            results = await _fallback_semantic_search(
                symbol=symbol,
                query_type=query_type,
                limit=limit,
                language=language,
                collection=coll,
                session=session,
            )

    except Exception as e:
        logger.warning(f"symbol_graph query failed: {e}")
        # Fall back to semantic search
        results = await _fallback_semantic_search(
            symbol=symbol,
            query_type=query_type,
            limit=limit,
            language=language,
            collection=coll,
            session=session,
        )

    return {
        "results": results,
        "symbol": symbol,
        "query_type": query_type,
        "count": len(results),
        "collection": coll,
    }


async def _query_array_field(
    client: Any,
    collection: str,
    field_key: str,
    value: str,
    limit: int,
    language: Optional[str] = None,
    under: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Query for points where an array field contains a specific value.

    Uses a multi-strategy approach for robust matching:
    1. MatchAny for exact array element matching
    2. MatchAny with symbol variants (qualified names)
    3. MatchText for substring fallback
    """
    from qdrant_client import models as qmodels

    results: List[Any] = []
    seen_ids: Set[str] = set()

    # Build base conditions for optional filters
    base_conditions = []
    if language:
        base_conditions.append(
            qmodels.FieldCondition(
                key="metadata.language",
                match=qmodels.MatchValue(value=language.lower()),
            )
        )
    if under:
        base_conditions.append(
            qmodels.FieldCondition(
                key="metadata.path_prefix",
                match=qmodels.MatchValue(value=under),
            )
        )

    # Strategy 1: Exact match with MatchAny (most reliable for array fields)
    try:
        filter1 = qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key=field_key,
                    match=qmodels.MatchAny(any=[value]),
                )
            ] + base_conditions
        )

        def scroll1():
            return client.scroll(
                collection_name=collection,
                scroll_filter=filter1,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )

        scroll_result = await asyncio.to_thread(scroll1)
        points = scroll_result[0] if scroll_result else []
        for pt in points:
            pt_id = str(getattr(pt, "id", id(pt)))
            if pt_id not in seen_ids:
                seen_ids.add(pt_id)
                results.append(pt)
    except Exception as e:
        logger.debug(f"Strategy 1 (MatchAny exact) failed: {e}")

    # Strategy 2: Try symbol variants (e.g., "MyClass.method" -> also try "method")
    if len(results) < limit:
        variants = _symbol_variants(value)
        for variant in variants[1:]:  # Skip first (exact match already tried)
            if len(results) >= limit:
                break
            try:
                filter2 = qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key=field_key,
                            match=qmodels.MatchAny(any=[variant]),
                        )
                    ] + base_conditions
                )

                def scroll2():
                    return client.scroll(
                        collection_name=collection,
                        scroll_filter=filter2,
                        limit=limit - len(results),
                        with_payload=True,
                        with_vectors=False,
                    )

                scroll_result = await asyncio.to_thread(scroll2)
                points = scroll_result[0] if scroll_result else []
                for pt in points:
                    pt_id = str(getattr(pt, "id", id(pt)))
                    if pt_id not in seen_ids:
                        seen_ids.add(pt_id)
                        results.append(pt)
            except Exception as e:
                logger.debug(f"Strategy 2 (variant '{variant}') failed: {e}")

    # Strategy 3: MatchText substring fallback for partial matches
    if len(results) < limit:
        try:
            filter3 = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key=field_key,
                        match=qmodels.MatchText(text=value),
                    )
                ] + base_conditions
            )

            def scroll3():
                return client.scroll(
                    collection_name=collection,
                    scroll_filter=filter3,
                    limit=limit - len(results),
                    with_payload=True,
                    with_vectors=False,
                )

            scroll_result = await asyncio.to_thread(scroll3)
            points = scroll_result[0] if scroll_result else []
            for pt in points:
                pt_id = str(getattr(pt, "id", id(pt)))
                if pt_id not in seen_ids:
                    seen_ids.add(pt_id)
                    results.append(pt)
        except Exception as e:
            # MatchText may not be supported on array fields in all Qdrant versions
            logger.debug(f"Strategy 3 (MatchText substring) failed: {e}")

    return [_format_point(pt) for pt in results[:limit]]


async def _query_definition(
    client: Any,
    collection: str,
    symbol: str,
    limit: int,
    language: Optional[str] = None,
    under: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Query for symbol definitions using symbol_path or symbol fields.
    """
    from qdrant_client import models as qmodels

    results = []

    # Build base conditions for optional filters
    base_conditions = []
    if language:
        base_conditions.append(
            qmodels.FieldCondition(
                key="metadata.language",
                match=qmodels.MatchValue(value=language.lower()),
            )
        )
    if under:
        base_conditions.append(
            qmodels.FieldCondition(
                key="metadata.path_prefix",
                match=qmodels.MatchValue(value=under),
            )
        )

    # Strategy 1: Exact match on symbol_path (e.g., "MyClass.my_method")
    try:
        filter1 = qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="metadata.symbol_path",
                    match=qmodels.MatchValue(value=symbol),
                )
            ] + base_conditions
        )

        def scroll1():
            return client.scroll(
                collection_name=collection,
                scroll_filter=filter1,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )

        scroll_result = await asyncio.to_thread(scroll1)
        points = scroll_result[0] if scroll_result else []
        results.extend(points)
    except Exception as e:
        logger.debug(f"symbol_path exact match failed: {e}")

    # Strategy 2: Exact match on symbol field
    if len(results) < limit:
        try:
            filter2 = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="metadata.symbol",
                        match=qmodels.MatchValue(value=symbol),
                    )
                ] + base_conditions
            )

            def scroll2():
                return client.scroll(
                    collection_name=collection,
                    scroll_filter=filter2,
                    limit=limit - len(results),
                    with_payload=True,
                    with_vectors=False,
                )

            scroll_result = await asyncio.to_thread(scroll2)
            points = scroll_result[0] if scroll_result else []
            results.extend(points)
        except Exception as e:
            logger.debug(f"symbol exact match failed: {e}")

    # Strategy 3: Text search on symbol_path for partial matches (e.g., "my_method" in "MyClass.my_method")
    if len(results) < limit:
        try:
            filter3 = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="metadata.symbol_path",
                        match=qmodels.MatchText(text=symbol),
                    )
                ] + base_conditions
            )

            def scroll3():
                return client.scroll(
                    collection_name=collection,
                    scroll_filter=filter3,
                    limit=limit - len(results),
                    with_payload=True,
                    with_vectors=False,
                )

            scroll_result = await asyncio.to_thread(scroll3)
            points = scroll_result[0] if scroll_result else []
            results.extend(points)
        except Exception as e:
            logger.debug(f"symbol_path text match failed: {e}")

    # Deduplicate by point ID
    seen_ids = set()
    unique_results = []
    for pt in results:
        pt_id = getattr(pt, "id", None)
        if pt_id not in seen_ids:
            seen_ids.add(pt_id)
            unique_results.append(pt)

    return [_format_point(pt) for pt in unique_results[:limit]]


def _get_path(pt: Any) -> str:
    """Extract path from point payload."""
    payload = getattr(pt, "payload", {}) or {}
    md = payload.get("metadata", payload)
    return str(md.get("path") or md.get("file_path") or "")


def _format_point(pt: Any) -> Dict[str, Any]:
    """Format a Qdrant point for the API response."""
    payload = getattr(pt, "payload", {}) or {}
    md = payload.get("metadata", payload)

    # Get code snippet from correct field: "information" is the indexed text
    snippet = ""
    info = payload.get("information") or payload.get("document") or ""
    if info:
        # The information field contains: "HEADER\n<CODE>\ncode here\n</CODE>"
        # Extract code from between markers if present
        if "<CODE>" in info and "</CODE>" in info:
            try:
                start = info.index("<CODE>") + 6
                end = info.index("</CODE>")
                snippet = info[start:end].strip()[:500]
            except Exception:
                snippet = info[:500]
        else:
            snippet = info[:500]

    result = {
        "path": str(md.get("path") or md.get("file_path") or ""),
        "start_line": _parse_int_or_default(md.get("start_line") or md.get("start"), default=0),
        "end_line": _parse_int_or_default(md.get("end_line") or md.get("end"), default=0),
        "symbol": str(md.get("symbol") or ""),
        "symbol_path": str(md.get("symbol_path") or ""),
        "language": str(md.get("language") or ""),
        "snippet": snippet,
        "calls": md.get("calls") or [],
        "imports": md.get("imports") or [],
    }

    return result


async def _fallback_semantic_search(
    symbol: str,
    query_type: str,
    limit: int = 20,
    language: Optional[str] = None,
    collection: Optional[str] = None,
    session: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Fallback to semantic search when filter-based search returns no results.
    """
    # Construct a query based on what we're looking for
    query_prefixes = {
        "callers": f"code that calls {symbol}",
        "definition": f"definition of {symbol} function class",
        "importers": f"code that imports {symbol}",
    }
    query = query_prefixes.get(query_type, symbol)

    try:
        from scripts.mcp_impl.search import _repo_search_impl

        search_result = await _repo_search_impl(
            query=query,
            limit=limit,
            language=language,
            session=session,
            output_format="json",  # Avoid TOON encoding for internal calls
        )

        # Handle case where results might be TOON-encoded string (shouldn't happen with output_format="json")
        results = search_result.get("results", [])
        if isinstance(results, str):
            # If somehow still a string, return empty - TOON decoding is not worth it here
            logger.debug("Fallback search returned TOON-encoded results, skipping")
            return []
        return results

    except Exception as e:
        logger.warning(f"Fallback semantic search failed: {e}")
        return []


def _format_symbol_graph_toon(result: Dict[str, Any]) -> str:
    """Format symbol graph results in TOON format for token efficiency."""
    lines = []
    query_type = result.get("query_type", "")
    symbol = result.get("symbol", "")
    results = result.get("results", [])

    if not results:
        return f"≡ SYMBOL_GRAPH | {query_type} | {symbol}\n⚠ No results found"

    lines.append(f"≡ SYMBOL_GRAPH | {query_type} | {symbol} | {len(results)} results")

    for r in results:
        path = r.get("path", "")
        start = r.get("start_line", 0)
        end = r.get("end_line", 0)
        sym = r.get("symbol_path") or r.get("symbol") or ""

        line = f"→ {path}:{start}-{end}"
        if sym:
            line += f" | {sym}"

        lines.append(line)

    return "\n".join(lines)


async def _compute_called_by(
    symbol: str,
    limit: int = 50,
    language: Optional[str] = None,
    under: Optional[str] = None,
    collection: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compute "called_by" - the inverse of what a symbol calls.

    Given a symbol (e.g., function or method), finds all functions that:
    1. Are defined in the codebase
    2. Have this symbol in their metadata.calls list

    This is the ego-graph concept simplified: "Who references me?"

    Args:
        symbol: The symbol name to find callers for
        limit: Maximum number of callers to return
        language: Optional language filter
        under: Optional path prefix filter
        collection: Optional collection override

    Returns:
        Dict with:
        - symbol: The queried symbol
        - called_by: List of {path, symbol, symbol_path, line} for callers
        - count: Number of callers found
    """
    from qdrant_client import QdrantClient
    from qdrant_client import models as qmodels

    # Get collection
    coll = str(collection or "").strip()
    if not coll:
        try:
            from scripts.mcp_impl.workspace import _default_collection
            coll = _default_collection() or ""
        except Exception:
            coll = os.environ.get("COLLECTION_NAME", "codebase")
    if not coll:
        coll = os.environ.get("COLLECTION_NAME", "codebase")

    try:
        client = QdrantClient(
            url=QDRANT_URL,
            api_key=os.environ.get("QDRANT_API_KEY"),
            timeout=float(os.environ.get("QDRANT_TIMEOUT", "20") or 20),
        )
    except Exception as e:
        logger.error(f"Failed to connect to Qdrant: {e}")
        return {
            "symbol": symbol,
            "called_by": [],
            "count": 0,
            "error": f"Qdrant connection failed: {e}",
        }

    # Build filter: find chunks where metadata.calls contains symbol
    base_conditions = []
    if language:
        base_conditions.append(
            qmodels.FieldCondition(
                key="metadata.language",
                match=qmodels.MatchValue(value=language.lower()),
            )
        )
    norm_under = _norm_under(under)
    if norm_under:
        base_conditions.append(
            qmodels.FieldCondition(
                key="metadata.path_prefix",
                match=qmodels.MatchValue(value=norm_under),
            )
        )

    callers: List[Dict[str, Any]] = []
    seen_ids: Set[str] = set()

    # Try exact match first
    try:
        variants = _symbol_variants(symbol)
        for variant in variants:
            if len(callers) >= limit:
                break

            query_filter = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="metadata.calls",
                        match=qmodels.MatchAny(any=[variant]),
                    )
                ] + base_conditions
            )

            def do_scroll():
                return client.scroll(
                    collection_name=coll,
                    scroll_filter=query_filter,
                    limit=limit - len(callers),
                    with_payload=True,
                    with_vectors=False,
                )

            scroll_result = await asyncio.to_thread(do_scroll)
            points = scroll_result[0] if scroll_result else []

            for pt in points:
                pt_id = str(getattr(pt, "id", id(pt)))
                if pt_id in seen_ids:
                    continue
                seen_ids.add(pt_id)

                payload = getattr(pt, "payload", {}) or {}
                md = payload.get("metadata", payload)

                # Only include if this chunk has a symbol (is a function/class definition)
                chunk_symbol = str(md.get("symbol") or "")
                if not chunk_symbol:
                    continue

                caller_info = {
                    "path": str(md.get("path") or ""),
                    "symbol": chunk_symbol,
                    "symbol_path": str(md.get("symbol_path") or ""),
                    "start_line": _parse_int_or_default(md.get("start_line") or md.get("start"), default=0),
                    "end_line": _parse_int_or_default(md.get("end_line") or md.get("end"), default=0),
                    "language": str(md.get("language") or ""),
                }
                callers.append(caller_info)

    except Exception as e:
        logger.warning(f"_compute_called_by query failed: {e}")

    return {
        "symbol": symbol,
        "called_by": callers[:limit],
        "count": len(callers[:limit]),
        "collection": coll,
    }


async def _get_symbol_calls(
    symbol_path: str,
    collection: Optional[str] = None,
) -> List[str]:
    """
    Get the list of calls made by a specific symbol.

    Useful for building call graphs: given a function, what does it call?

    Args:
        symbol_path: The symbol_path to look up (e.g., "MyClass.my_method")
        collection: Optional collection override

    Returns:
        List of function/method names called by this symbol
    """
    from qdrant_client import QdrantClient
    from qdrant_client import models as qmodels

    coll = str(collection or "").strip()
    if not coll:
        try:
            from scripts.mcp_impl.workspace import _default_collection
            coll = _default_collection() or ""
        except Exception:
            coll = os.environ.get("COLLECTION_NAME", "codebase")
    if not coll:
        coll = os.environ.get("COLLECTION_NAME", "codebase")

    try:
        client = QdrantClient(
            url=QDRANT_URL,
            api_key=os.environ.get("QDRANT_API_KEY"),
            timeout=float(os.environ.get("QDRANT_TIMEOUT", "20") or 20),
        )
    except Exception as e:
        logger.error(f"Failed to connect to Qdrant: {e}")
        return []

    # Find the symbol definition
    try:
        query_filter = qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="metadata.symbol_path",
                    match=qmodels.MatchValue(value=symbol_path),
                )
            ]
        )

        def do_scroll():
            return client.scroll(
                collection_name=coll,
                scroll_filter=query_filter,
                limit=1,
                with_payload=True,
                with_vectors=False,
            )

        scroll_result = await asyncio.to_thread(do_scroll)
        points = scroll_result[0] if scroll_result else []

        if points:
            payload = getattr(points[0], "payload", {}) or {}
            md = payload.get("metadata", payload)
            return md.get("calls") or []

    except Exception as e:
        logger.warning(f"_get_symbol_calls failed: {e}")

    return []
