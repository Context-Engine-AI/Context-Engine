#!/usr/bin/env python3
"""
mcp/context_search.py - Context search helpers for MCP indexer server.

Extracted from mcp_indexer_server.py for better modularity.
Contains:
- _context_search_impl: Main implementation (called by thin @mcp.tool() wrapper)
- _cs_* helper functions for context_search (blended code + memory search)

Note: The @mcp.tool() decorated context_search function remains in mcp_indexer_server.py
as a thin wrapper that calls _context_search_impl.
"""

from __future__ import annotations

__all__ = [
    "_context_search_impl",
]

import asyncio
import os
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Imports from sibling modules
# ---------------------------------------------------------------------------
from scripts.mcp_impl.utils import (
    _coerce_bool,
    _coerce_int,
    _to_str_list_relaxed,
    _looks_jsonish_string,
    _maybe_parse_jsonish,
    _extract_kwargs_payload,
)
from scripts.mcp_impl.workspace import _default_collection, _MEM_COLL_CACHE
from scripts.mcp_impl.toon import _should_use_toon, _format_context_results_as_toon

# Environment
QDRANT_URL = os.environ.get("QDRANT_URL", "http://qdrant:6333")

async def _context_search_impl(
    # Core query + limits
    query: Any = None,
    limit: Any = None,
    per_path: Any = None,
    # Include memory hits and blending controls
    include_memories: Any = None,
    memory_weight: Any = None,
    per_source_limits: Any = None,  # e.g., {"code": 5, "memory": 3}
    # Pass-through structured filters (same as repo_search)
    include_snippet: Any = None,
    context_lines: Any = None,
    rerank_enabled: Any = None,
    rerank_top_n: Any = None,
    rerank_return_m: Any = None,
    rerank_timeout_ms: Any = None,
    highlight_snippet: Any = None,
    collection: Any = None,
    language: Any = None,
    under: Any = None,
    kind: Any = None,
    symbol: Any = None,
    path_regex: Any = None,
    path_glob: Any = None,
    not_glob: Any = None,
    ext: Any = None,
    not_: Any = None,
    case: Any = None,
    session: Any = None,
    compact: Any = None,
    # Repo scoping (cross-codebase isolation)
    repo: Any = None,  # str, list[str], or "*" to search all repos
    # Output format
    output_format: Any = None,  # "json" (default) or "toon" for token-efficient format
    kwargs: Any = None,
    # Injected dependencies from facade
    *,
    repo_search_fn: Any = None,  # async callable for repo_search
    get_embedding_model_fn: Any = None,  # callable for _get_embedding_model
) -> Dict[str, Any]:
    """Blend code search results with memory-store entries (notes, docs) for richer context.

    When to use:
    - You want code spans plus relevant memories in one response.
    - Prefer repo_search for code-only; use context_answer when you need an LLM-written answer.

    Key parameters:
    - query: str or list[str]
    - include_memories: bool (opt-in). If true, queries the memory collection and merges with code results.
    - memory_weight: float (default 1.0). Scales memory scores relative to code.
    - per_source_limits: dict, e.g. {"code": 5, "memory": 3}
    - All repo_search filters are supported and passed through.
    - output_format: "json" (default) or "toon" for token-efficient TOON format.
    - rerank_enabled: bool (default true). ONNX reranker is ON by default for better relevance.
    - repo: str or list[str]. Filter by repo name(s). Use "*" to search all repos (disable auto-filter).
      By default, auto-detects current repo from CURRENT_REPO env and filters to it.

    Returns:
    - {"results": [{"source": "code"| "memory", ...}, ...], "total": N[, "memory_note": str]}
    - In compact mode, results are reduced to lightweight records.

    Example:
    - include_memories=true, per_source_limits={"code": 6, "memory": 2}, path_glob="docs/**"
    """
    # Unwrap kwargs if MCP client sent everything in a single kwargs string
    if kwargs and not query and not limit:
        # If all named params are None and kwargs has content, assume wrapped call
        query = kwargs.get("query", query)
        limit = kwargs.get("limit", limit)
        per_path = kwargs.get("per_path", per_path)
        include_memories = kwargs.get("include_memories", include_memories)
        memory_weight = kwargs.get("memory_weight", memory_weight)
        per_source_limits = kwargs.get("per_source_limits", per_source_limits)
        include_snippet = kwargs.get("include_snippet", include_snippet)
        context_lines = kwargs.get("context_lines", context_lines)
        rerank_enabled = kwargs.get("rerank_enabled", rerank_enabled)
        rerank_top_n = kwargs.get("rerank_top_n", rerank_top_n)
        rerank_return_m = kwargs.get("rerank_return_m", rerank_return_m)
        rerank_timeout_ms = kwargs.get("rerank_timeout_ms", rerank_timeout_ms)
        highlight_snippet = kwargs.get("highlight_snippet", highlight_snippet)
        collection = kwargs.get("collection", collection)
        language = kwargs.get("language", language)
        under = kwargs.get("under", under)
        kind = kwargs.get("kind", kind)
        symbol = kwargs.get("symbol", symbol)
        path_regex = kwargs.get("path_regex", path_regex)
        path_glob = kwargs.get("path_glob", path_glob)
        not_glob = kwargs.get("not_glob", not_glob)
        ext = kwargs.get("ext", ext)
        not_ = kwargs.get("not_", not_)
        case = kwargs.get("case", case)
        compact = kwargs.get("compact", compact)

    # Unwrap nested payloads that some MCP clients send (kwargs/arguments fields or json strings)
    def _maybe_dict(val: Any) -> Dict[str, Any]:
        if isinstance(val, dict):
            return val
        if isinstance(val, str) and _looks_jsonish_string(val):
            parsed = _maybe_parse_jsonish(val)
            if isinstance(parsed, dict):
                return parsed
        return {}

    payloads: List[Dict[str, Any]] = []
    if isinstance(kwargs, dict):
        arg_payload = _maybe_dict(kwargs.get("arguments"))
        if arg_payload:
            payloads.append(arg_payload)
        nested_kwargs = _extract_kwargs_payload(kwargs)
        if nested_kwargs:
            payloads.append(nested_kwargs)
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        if (
            query is None or (isinstance(query, str) and query.strip() == "")
        ) and payload.get("query") is not None:
            query = payload.get("query")
        if (
            query is None or (isinstance(query, str) and query.strip() == "")
        ) and payload.get("queries") is not None:
            query = payload.get("queries")
        if (
            limit is None or (isinstance(limit, str) and limit.strip() == "")
        ) and payload.get("limit") is not None:
            limit = payload.get("limit")
        if (
            per_path is None
            or (isinstance(per_path, str) and str(per_path).strip() == "")
        ) and payload.get("per_path") is not None:
            per_path = payload.get("per_path")
        if include_memories is None and payload.get("include_memories") is not None:
            include_memories = payload.get("include_memories")
        if include_memories is None and payload.get("includeMemories") is not None:
            include_memories = payload.get("includeMemories")
        if memory_weight is None and payload.get("memory_weight") is not None:
            memory_weight = payload.get("memory_weight")
        if memory_weight is None and payload.get("memoryWeight") is not None:
            memory_weight = payload.get("memoryWeight")
        if per_source_limits is None and payload.get("per_source_limits") is not None:
            per_source_limits = payload.get("per_source_limits")
        if per_source_limits is None and payload.get("perSourceLimits") is not None:
            per_source_limits = payload.get("perSourceLimits")
        if (include_snippet is None or include_snippet == "") and payload.get(
            "include_snippet"
        ) is not None:
            include_snippet = payload.get("include_snippet")
        if (include_snippet is None or include_snippet == "") and payload.get(
            "includeSnippet"
        ) is not None:
            include_snippet = payload.get("includeSnippet")
        if (
            context_lines is None
            or (isinstance(context_lines, str) and context_lines.strip() == "")
        ) and payload.get("context_lines") is not None:
            context_lines = payload.get("context_lines")
        if (
            context_lines is None
            or (isinstance(context_lines, str) and context_lines.strip() == "")
        ) and payload.get("contextLines") is not None:
            context_lines = payload.get("contextLines")
        if (rerank_enabled is None or rerank_enabled == "") and payload.get(
            "rerank_enabled"
        ) is not None:
            rerank_enabled = payload.get("rerank_enabled")
        if (rerank_enabled is None or rerank_enabled == "") and payload.get(
            "rerankEnabled"
        ) is not None:
            rerank_enabled = payload.get("rerankEnabled")
        if (
            rerank_top_n is None
            or (isinstance(rerank_top_n, str) and rerank_top_n.strip() == "")
        ) and payload.get("rerank_top_n") is not None:
            rerank_top_n = payload.get("rerank_top_n")
        if (
            rerank_top_n is None
            or (isinstance(rerank_top_n, str) and rerank_top_n.strip() == "")
        ) and payload.get("rerankTopN") is not None:
            rerank_top_n = payload.get("rerankTopN")
        if (
            rerank_return_m is None
            or (isinstance(rerank_return_m, str) and rerank_return_m.strip() == "")
        ) and payload.get("rerank_return_m") is not None:
            rerank_return_m = payload.get("rerank_return_m")
        if (
            rerank_return_m is None
            or (isinstance(rerank_return_m, str) and rerank_return_m.strip() == "")
        ) and payload.get("rerankReturnM") is not None:
            rerank_return_m = payload.get("rerankReturnM")
        if (
            rerank_timeout_ms is None
            or (isinstance(rerank_timeout_ms, str) and rerank_timeout_ms.strip() == "")
        ) and payload.get("rerank_timeout_ms") is not None:
            rerank_timeout_ms = payload.get("rerank_timeout_ms")
        if (
            rerank_timeout_ms is None
            or (isinstance(rerank_timeout_ms, str) and rerank_timeout_ms.strip() == "")
        ) and payload.get("rerankTimeoutMs") is not None:
            rerank_timeout_ms = payload.get("rerankTimeoutMs")
        if (highlight_snippet is None or highlight_snippet == "") and payload.get(
            "highlight_snippet"
        ) is not None:
            highlight_snippet = payload.get("highlight_snippet")
        if (highlight_snippet is None or highlight_snippet == "") and payload.get(
            "highlightSnippet"
        ) is not None:
            highlight_snippet = payload.get("highlightSnippet")
        if (
            collection is None
            or (isinstance(collection, str) and collection.strip() == "")
        ) and payload.get("collection") is not None:
            collection = payload.get("collection")
        if (
            language is None or (isinstance(language, str) and language.strip() == "")
        ) and payload.get("language") is not None:
            language = payload.get("language")
        if (
            under is None or (isinstance(under, str) and under.strip() == "")
        ) and payload.get("under") is not None:
            under = payload.get("under")
        if (
            kind is None or (isinstance(kind, str) and kind.strip() == "")
        ) and payload.get("kind") is not None:
            kind = payload.get("kind")
        if (
            symbol is None or (isinstance(symbol, str) and symbol.strip() == "")
        ) and payload.get("symbol") is not None:
            symbol = payload.get("symbol")
        if (
            path_regex is None
            or (isinstance(path_regex, str) and path_regex.strip() == "")
        ) and payload.get("path_regex") is not None:
            path_regex = payload.get("path_regex")
        if (
            path_regex is None
            or (isinstance(path_regex, str) and path_regex.strip() == "")
        ) and payload.get("pathRegex") is not None:
            path_regex = payload.get("pathRegex")
        if (
            path_glob is None
            or (isinstance(path_glob, str) and str(path_glob).strip() == "")
        ) and payload.get("path_glob") is not None:
            path_glob = payload.get("path_glob")
        if (
            path_glob is None
            or (isinstance(path_glob, str) and str(path_glob).strip() == "")
        ) and payload.get("pathGlob") is not None:
            path_glob = payload.get("pathGlob")
        if (
            not_glob is None
            or (isinstance(not_glob, str) and str(not_glob).strip() == "")
        ) and payload.get("not_glob") is not None:
            not_glob = payload.get("not_glob")
        if (
            not_glob is None
            or (isinstance(not_glob, str) and str(not_glob).strip() == "")
        ) and payload.get("notGlob") is not None:
            not_glob = payload.get("notGlob")
        if (
            ext is None or (isinstance(ext, str) and ext.strip() == "")
        ) and payload.get("ext") is not None:
            ext = payload.get("ext")
        if (
            not_ is None or (isinstance(not_, str) and not_.strip() == "")
        ) and payload.get("not") is not None:
            not_ = payload.get("not")
        if (
            not_ is None or (isinstance(not_, str) and not_.strip() == "")
        ) and payload.get("not_") is not None:
            not_ = payload.get("not_")
        if (
            case is None or (isinstance(case, str) and case.strip() == "")
        ) and payload.get("case") is not None:
            case = payload.get("case")
        if (
            compact is None or (isinstance(compact, str) and compact.strip() == "")
        ) and payload.get("compact") is not None:
            compact = payload.get("compact")

    # Leniency: absorb nested 'kwargs' JSON payload some clients send (string or dict)
    try:
        _extra = _extract_kwargs_payload(kwargs)
        if _extra:
            if (query is None) or (isinstance(query, str) and query.strip() == ""):
                query = _extra.get("query") or _extra.get("queries") or query
            if (limit in (None, "")) and (_extra.get("limit") is not None):
                limit = _extra.get("limit")
            if (per_path in (None, "")) and (_extra.get("per_path") is not None):
                per_path = _extra.get("per_path")
            # Memory blending controls
            if include_memories is None and (
                (_extra.get("include_memories") is not None)
                or (_extra.get("includeMemories") is not None)
            ):
                include_memories = _extra.get(
                    "include_memories", _extra.get("includeMemories")
                )
            if memory_weight is None and (
                (_extra.get("memory_weight") is not None)
                or (_extra.get("memoryWeight") is not None)
            ):
                memory_weight = _extra.get("memory_weight", _extra.get("memoryWeight"))
            if per_source_limits is None and (
                (_extra.get("per_source_limits") is not None)
                or (_extra.get("perSourceLimits") is not None)
            ):
                per_source_limits = _extra.get(
                    "per_source_limits", _extra.get("perSourceLimits")
                )
            # Passthrough search filters
            if (include_snippet in (None, "")) and (
                _extra.get("include_snippet") is not None
            ):
                include_snippet = _extra.get("include_snippet")
            if (context_lines in (None, "")) and (
                _extra.get("context_lines") is not None
            ):
                context_lines = _extra.get("context_lines")
            if (rerank_enabled in (None, "")) and (
                _extra.get("rerank_enabled") is not None
            ):
                rerank_enabled = _extra.get("rerank_enabled")
            if (rerank_top_n in (None, "")) and (
                _extra.get("rerank_top_n") is not None
            ):
                rerank_top_n = _extra.get("rerank_top_n")
            if (rerank_return_m in (None, "")) and (
                _extra.get("rerank_return_m") is not None
            ):
                rerank_return_m = _extra.get("rerank_return_m")
            if (rerank_timeout_ms in (None, "")) and (
                _extra.get("rerank_timeout_ms") is not None
            ):
                rerank_timeout_ms = _extra.get("rerank_timeout_ms")
            if (highlight_snippet in (None, "")) and (
                _extra.get("highlight_snippet") is not None
            ):
                highlight_snippet = _extra.get("highlight_snippet")
            if (
                collection is None
                or (isinstance(collection, str) and collection.strip() == "")
            ) and _extra.get("collection"):
                collection = _extra.get("collection")
            if (
                language is None
                or (isinstance(language, str) and language.strip() == "")
            ) and _extra.get("language"):
                language = _extra.get("language")
            if (
                under is None or (isinstance(under, str) and under.strip() == "")
            ) and _extra.get("under"):
                under = _extra.get("under")
            if (
                kind is None or (isinstance(kind, str) and kind.strip() == "")
            ) and _extra.get("kind"):
                kind = _extra.get("kind")
            if (
                symbol is None or (isinstance(symbol, str) and symbol.strip() == "")
            ) and _extra.get("symbol"):
                symbol = _extra.get("symbol")
            if (
                path_regex is None
                or (isinstance(path_regex, str) and path_regex.strip() == "")
            ) and _extra.get("path_regex"):
                path_regex = _extra.get("path_regex")
            if (path_glob in (None, "")) and (_extra.get("path_glob") is not None):
                path_glob = _extra.get("path_glob")
            if (not_glob in (None, "")) and (_extra.get("not_glob") is not None):
                not_glob = _extra.get("not_glob")
            if (
                ext is None or (isinstance(ext, str) and ext.strip() == "")
            ) and _extra.get("ext"):
                ext = _extra.get("ext")
            if (not_ is None or (isinstance(not_, str) and not_.strip() == "")) and (
                _extra.get("not") or _extra.get("not_")
            ):
                not_ = _extra.get("not") or _extra.get("not_")
            if (
                case is None or (isinstance(case, str) and case.strip() == "")
            ) and _extra.get("case"):
                case = _extra.get("case")
            if (compact in (None, "")) and (_extra.get("compact") is not None):
                compact = _extra.get("compact")
    except Exception:
        pass

    # Normalize inputs
    coll = (collection or _default_collection()) or ""
    mcoll = (os.environ.get("MEMORY_COLLECTION_NAME") or coll) or ""
    use_sse_memory = str(os.environ.get("MEMORY_SSE_ENABLED", "false")).lower() in (
        "1",
        "true",
        "yes",
    )
    # Auto-detect memory collection if not explicitly set
    if include_memories and not os.environ.get("MEMORY_COLLECTION_NAME"):
        try:
            from qdrant_client import QdrantClient  # type: ignore

            # Optional: disable auto-detect and/or use cached result
            if str(os.environ.get("MEMORY_AUTODETECT", "1")).lower() not in (
                "1",
                "true",
                "yes",
                "on",
            ):
                raise RuntimeError("auto-detect disabled")
            import time

            ttl = float(os.environ.get("MEMORY_COLLECTION_TTL_SECS", "300") or 300)
            if (
                _MEM_COLL_CACHE["name"]
                and (time.time() - float(_MEM_COLL_CACHE["ts"] or 0.0)) < ttl
            ):
                mcoll = _MEM_COLL_CACHE["name"]
                raise RuntimeError("use cache")
            client = QdrantClient(
                url=QDRANT_URL,
                api_key=os.environ.get("QDRANT_API_KEY"),
                timeout=float(os.environ.get("QDRANT_TIMEOUT", "20") or 20),
            )
            info = await asyncio.to_thread(client.get_collections)
            best_name = None
            best_hits = -1
            for c in info.collections:
                name = getattr(c, "name", None)
                if not name:
                    continue
                # Sample a small page for memory-like payloads
                try:
                    pts, _ = await asyncio.to_thread(
                        lambda: client.scroll(
                            collection_name=name,
                            with_payload=True,
                            with_vectors=False,
                            limit=300,
                        )
                    )
                    hits = 0
                    for pt in pts:
                        pl = getattr(pt, "payload", {}) or {}
                        md = pl.get("metadata") or {}
                        path = md.get("path")
                        content = (
                            pl.get("content")
                            or pl.get("text")
                            or pl.get("information")
                            or md.get("information")
                        )
                        if not path and content:
                            hits += 1
                    if hits > best_hits:
                        best_hits = hits
                        best_name = name
                except Exception:
                    continue
            if best_name and best_hits > 0:
                mcoll = best_name
                try:
                    import time

                    _MEM_COLL_CACHE["name"] = best_name
                    _MEM_COLL_CACHE["ts"] = time.time()
                except Exception:
                    pass
        except Exception:
            pass

    try:
        lim = int(limit) if (limit is not None and str(limit).strip() != "") else 10
    except (ValueError, TypeError):
        lim = 10
    try:
        per_path_val = (
            int(per_path)
            if (per_path is not None and str(per_path).strip() != "")
            else 2
        )
    except (ValueError, TypeError):
        per_path_val = 2

    # Normalize queries to list (accept q/text aliases)
    queries: List[str] = []
    if query is None or (isinstance(query, str) and query.strip() == ""):
        q_alt = kwargs.get("q") or kwargs.get("text")
        if q_alt is not None:
            query = q_alt
    if isinstance(query, (list, tuple)):
        queries = [str(q).strip() for q in query if str(q).strip()]
    elif isinstance(query, str):
        queries = _to_str_list_relaxed(query)
    elif query is not None and str(query).strip() != "":
        queries = [str(query).strip()]

    # Accept common alias keys and camelCase from clients
    if kwargs and (limit is None or (isinstance(limit, str) and limit.strip() == "")) and (
        "top_k" in kwargs
    ):
        limit = kwargs.get("top_k")
    if kwargs and include_memories is None and ("includeMemories" in kwargs):
        include_memories = kwargs.get("includeMemories")
    if kwargs and memory_weight is None and ("memoryWeight" in kwargs):
        memory_weight = kwargs.get("memoryWeight")
    if kwargs and per_source_limits is None and ("perSourceLimits" in kwargs):
        per_source_limits = kwargs.get("perSourceLimits")

    # Smart defaults inspired by stored preferences, but without external calls
    compact_raw = compact
    smart_compact = False
    if len(queries) > 1 and (
        compact_raw is None
        or (isinstance(compact_raw, str) and compact_raw.strip() == "")
    ):
        smart_compact = True
    # If snippets are requested, disable compact to preserve snippet field
    if include_snippet and str(include_snippet).lower() not in ("", "false", "0", "no"):
        smart_compact = False
        compact_raw = False
    eff_compact = (
        True if (smart_compact or (str(compact_raw).lower() == "true")) else False
    )

    # Per-source limits
    code_limit = lim
    mem_limit = 0
    include_mem = False
    if include_memories is not None and str(include_memories).lower() in (
        "true",
        "1",
        "yes",
    ):  # opt-in
        include_mem = True
        # Parse per_source_limits if provided; accept JSON-ish strings as well
        code_limit = lim
        mem_limit = min(3, lim)  # sensible default
        try:
            psl = per_source_limits
            # Some clients stringify payloads; parse if JSON-ish
            if isinstance(psl, str) and _looks_jsonish_string(psl):
                _ps = _maybe_parse_jsonish(psl)
                if isinstance(_ps, dict):
                    psl = _ps
            if isinstance(psl, dict):
                code_limit = int(psl.get("code", code_limit))
                mem_limit = int(psl.get("memory", mem_limit))
        except (ValueError, TypeError):
            pass

    # First: run code search via internal repo_search for consistent behavior
    code_res = await repo_search_fn(
        query=queries if len(queries) > 1 else (queries[0] if queries else ""),
        limit=code_limit,
        per_path=per_path_val,
        include_snippet=include_snippet,
        context_lines=context_lines,
        rerank_enabled=rerank_enabled,
        rerank_top_n=rerank_top_n,
        rerank_return_m=rerank_return_m,
        rerank_timeout_ms=rerank_timeout_ms,
        highlight_snippet=highlight_snippet,
        collection=coll,
        language=language,
        under=under,
        kind=kind,
        symbol=symbol,
        path_regex=path_regex,
        path_glob=path_glob,
        not_glob=not_glob,
        ext=ext,
        not_=not_,
        case=case,
        compact=False,
        repo=repo,  # Cross-codebase isolation
        session=session,
    )

    # Optional debug
    if os.environ.get("DEBUG_CONTEXT_SEARCH"):
        try:
            logger.debug(
                "DBG_CTX_SRCH_START",
                extra={
                    "queries": queries,
                    "coll": coll,
                    "limit": int(code_limit),
                    "per_path": int(per_path_val),
                },
            )
        except Exception:
            pass

    # Shape code results to a common schema
    code_hits: List[Dict[str, Any]] = []
    if isinstance(code_res, dict):
        items = code_res.get("results") or code_res.get("data") or code_res.get("items")
        # If compact mode was used, results may be a list; support both shapes
        items = items if items is not None else code_res.get("results", code_res)
    else:
        items = code_res
    # Normalize list
    if isinstance(items, list):
        for r in items:
            if isinstance(r, dict):
                ch = {
                    "source": "code",
                    "score": float(r.get("score") or r.get("s") or 0.0),
                    "path": r.get("path"),
                    "symbol": r.get("symbol", ""),
                    "start_line": r.get("start_line"),
                    "end_line": r.get("end_line"),
                    "_raw": r,
                }
                code_hits.append(ch)
    # More debug after shaping
    if os.environ.get("DEBUG_CONTEXT_SEARCH"):
        try:
            logger.debug(
                "DBG_CTX_SRCH_CODE_RES",
                extra={
                    "type": type(code_res).__name__,
                    "has_results": bool(
                        isinstance(code_res, dict)
                        and isinstance(code_res.get("results"), list)
                    ),
                    "len_results": (
                        len(code_res.get("results"))
                        if isinstance(code_res, dict)
                        and isinstance(code_res.get("results"), list)
                        else None
                    ),
                    "code_hits": len(code_hits),
                },
            )
        except Exception:
            pass

    # HTTP fallback: if still empty, call our own repo_search over HTTP (safeguarded)
    used_http_fallback = False
    if not code_hits:
        try:
            from scripts.mcp_router import call_tool_http  # type: ignore

            base = (
                os.environ.get("MCP_INDEXER_HTTP_URL") or "http://localhost:8003/mcp"
            ).rstrip("/")
            http_args = {
                "query": (
                    queries if len(queries) > 1 else (queries[0] if queries else "")
                ),
                "limit": int(code_limit),
                "per_path": int(per_path_val),
                "include_snippet": bool(include_snippet),
                "context_lines": int(context_lines)
                if context_lines not in (None, "")
                else 2,
                "collection": coll,
                "language": language or "",
                "under": under or "",
                "kind": kind or "",
                "symbol": symbol or "",
                "path_regex": path_regex or "",
                "path_glob": path_glob or [],
                "not_glob": not_glob or [],
                "ext": ext or "",
                "not": not_ or "",
                "case": case or "",
                "compact": bool(eff_compact),
            }
            timeout = float(os.environ.get("CONTEXT_SEARCH_HTTP_TIMEOUT", "20") or 20)
            resp = await asyncio.to_thread(
                lambda: call_tool_http(base, "repo_search", http_args, timeout=timeout)
            )
            r = ((resp.get("result") or {}).get("structuredContent") or {}).get(
                "result"
            ) or {}
            http_items = r.get("results") or []
            if isinstance(http_items, list):
                for obj in http_items:
                    if isinstance(obj, dict):
                        code_hits.append(
                            {
                                "source": "code",
                                "score": float(obj.get("score") or obj.get("s") or 0.0),
                                "path": obj.get("path"),
                                "symbol": obj.get("symbol", ""),
                                "start_line": int(obj.get("start_line") or 0),
                                "end_line": int(obj.get("end_line") or 0),
                                "_raw": obj,
                            }
                        )
            used_http_fallback = True
            if os.environ.get("DEBUG_CONTEXT_SEARCH"):
                try:
                    logger.debug(
                        "DBG_CTX_SRCH_HTTP_FALLBACK", extra={"count": len(code_hits)}
                    )
                except Exception:
                    pass
        except Exception:
            pass

    # Fallback: if internal repo_search yielded no code hits, try direct in-process hybrid search
    used_hybrid_fallback = False
    if not code_hits and queries:
        try:
            from scripts.hybrid_search import run_hybrid_search  # type: ignore

            model_name = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
            model = get_embedding_model_fn(model_name) if get_embedding_model_fn else None
            items2 = await asyncio.to_thread(
                lambda: run_hybrid_search(
                    queries=queries,
                    limit=int(code_limit),
                    per_path=int(per_path_val),
                    language=language or None,
                    under=under or None,
                    kind=kind or None,
                    symbol=symbol or None,
                    ext=ext or None,
                    not_filter=not_ or None,
                    case=case or None,
                    path_regex=path_regex or None,
                    path_glob=path_glob or None,
                    not_glob=not_glob or None,
                    expand=str(os.environ.get("HYBRID_EXPAND", "0")).strip().lower()
                    in {"1", "true", "yes", "on"},
                    model=model,
                    collection=coll,
                )
            )
            if isinstance(items2, list):
                for obj in items2:
                    if isinstance(obj, dict):
                        code_hits.append(
                            {
                                "source": "code",
                                "score": float(obj.get("score") or obj.get("s") or 0.0),
                                "path": obj.get("path"),
                                "symbol": obj.get("symbol", ""),
                                "start_line": int(obj.get("start_line") or 0),
                                "end_line": int(obj.get("end_line") or 0),
                                "_raw": obj,
                            }
                        )
            used_hybrid_fallback = True
        except Exception:
            pass

    # Option A: Query the memory MCP server over SSE and blend results (real integration)
    mem_hits: List[Dict[str, Any]] = []
    memory_note: str = ""
    if include_mem and mem_limit > 0 and queries and use_sse_memory:
        try:
            # Import the FastMCP client if available; record a helpful note otherwise
            try:
                from fastmcp import Client  # use FastMCP client for SSE interop
            except ImportError:
                memory_note = "SSE memory disabled: fastmcp client not installed"
                raise
            import asyncio

            timeout = float(os.environ.get("MEMORY_MCP_TIMEOUT", "6"))
            base_url = os.environ.get("MEMORY_MCP_URL") or "http://mcp:8000/sse"
            # Best-effort: poll memory MCP /readyz on its health port to avoid init race
            try:
                from urllib.parse import urlparse
                import urllib.request, time

                ready_attempts = int(
                    os.environ.get("MEMORY_MCP_READY_RETRIES", "5") or 5
                )
                ready_backoff = float(
                    os.environ.get("MEMORY_MCP_READY_BACKOFF", "0.2") or 0.2
                )
                health_port = int(
                    os.environ.get("MEMORY_MCP_HEALTH_PORT", "18000") or 18000
                )
                pu = urlparse(base_url)
                host = pu.hostname or "mcp"
                scheme = pu.scheme or "http"
                readyz = f"{scheme}://{host}:{health_port}/readyz"

                def _poll_ready():
                    for i in range(max(1, ready_attempts)):
                        try:
                            with urllib.request.urlopen(readyz, timeout=1.5) as r:
                                if getattr(r, "status", 200) == 200:
                                    return True
                        except Exception:
                            time.sleep(ready_backoff * (i + 1))
                    return False

                try:
                    await asyncio.to_thread(_poll_ready)
                except Exception:
                    pass
            except Exception:
                pass

            async with Client(base_url) as c:
                tools = None
                attempts = int(os.environ.get("MEMORY_MCP_LIST_RETRIES", "3") or 3)
                backoff = float(os.environ.get("MEMORY_MCP_LIST_BACKOFF", "0.2") or 0.2)
                last_err = None
                for i in range(max(1, attempts)):
                    try:
                        tools = await asyncio.wait_for(c.list_tools(), timeout=timeout)
                        if tools:
                            break
                    except Exception as e:
                        last_err = e
                        try:
                            await asyncio.sleep(backoff * (i + 1))
                        except Exception:
                            pass
                if tools is None:
                    raise last_err or RuntimeError(
                        "list_tools failed before initialization"
                    )
                tool_name = None
                # Prefer canonical names
                for t in tools:
                    tn = (getattr(t, "name", None) or "").strip()
                    tl = tn.lower()
                    if tl in ("find", "memory.find"):
                        tool_name = tn
                        break
                if tool_name is None:
                    for t in tools:
                        tn = (getattr(t, "name", None) or "").strip()
                        if "find" in tn.lower():
                            tool_name = tn
                            break
                if tool_name:
                    qtext = " ".join([q for q in queries if q]).strip() or queries[0]
                    arg_variants: List[Dict[str, Any]] = [
                        {"query": qtext, "limit": mem_limit, "collection": mcoll},
                        {"q": qtext, "limit": mem_limit, "collection": mcoll},
                        {"text": qtext, "limit": mem_limit, "collection": mcoll},
                    ]
                    res_obj = None
                    for args in arg_variants:
                        try:
                            res_obj = await asyncio.wait_for(
                                c.call_tool(tool_name, args), timeout=timeout
                            )
                            break
                        except Exception:
                            continue
                    if res_obj is not None:
                        # Normalize FastMCP result content -> rd-like dict
                        rd = {"content": []}
                        try:
                            for item in getattr(res_obj, "content", []) or []:
                                txt = getattr(item, "text", None)
                                if isinstance(txt, str):
                                    rd["content"].append({"type": "text", "text": txt})
                        except Exception:
                            rd = {}

                        # Parse common MCP tool result shapes
                        def push_text(
                            txt: str,
                            md: Dict[str, Any] | None = None,
                            score: float | int | None = None,
                        ):
                            if not txt:
                                return
                            mem_hits.append(
                                {
                                    "source": "memory",
                                    "score": float(score or 1.0),
                                    "content": txt,
                                    "metadata": (md or {}),
                                }
                            )

                        if isinstance(rd, dict):
                            cont = rd.get("content")
                            if isinstance(cont, list):
                                for c in cont:
                                    try:
                                        ctype = c.get("type")
                                        if ctype == "text" and isinstance(
                                            c.get("text"), str
                                        ):
                                            push_text(c["text"], {})
                                        elif ctype == "json":
                                            j = c.get("json")
                                            if isinstance(j, list):
                                                for it in j:
                                                    if isinstance(it, dict):
                                                        push_text(
                                                            str(
                                                                it.get("text")
                                                                or it.get("content")
                                                                or it.get("information")
                                                                or ""
                                                            ),
                                                            it.get("metadata") or {},
                                                            it.get("score") or 1.0,
                                                        )
                                            elif isinstance(j, dict):
                                                items = (
                                                    j.get("results")
                                                    or j.get("items")
                                                    or j.get("memories")
                                                    or j.get("data")
                                                )
                                                if isinstance(items, list):
                                                    for it in items:
                                                        if isinstance(it, dict):
                                                            push_text(
                                                                str(
                                                                    it.get("text")
                                                                    or it.get("content")
                                                                    or it.get(
                                                                        "information"
                                                                    )
                                                                    or ""
                                                                ),
                                                                it.get("metadata")
                                                                or {},
                                                                it.get("score") or 1.0,
                                                            )
                                    except Exception:
                                        continue
                            # Fallback if provider returns flat dict
                            if not mem_hits:
                                items = rd.get("results") or rd.get("items")
                                if isinstance(items, list):
                                    for it in items:
                                        if isinstance(it, dict):
                                            push_text(
                                                str(
                                                    it.get("text")
                                                    or it.get("content")
                                                    or it.get("information")
                                                    or ""
                                                ),
                                                it.get("metadata") or {},
                                                it.get("score") or 1.0,
                                            )
        except Exception:
            pass

    # If SSE memory didn’t yield hits, try local Qdrant memory-like retrieval as fallback
    if include_mem and mem_limit > 0 and not mem_hits and queries:
        try:
            from qdrant_client import QdrantClient  # type: ignore

            from scripts.utils import sanitize_vector_name  # local util

            client = QdrantClient(
                url=QDRANT_URL,
                api_key=os.environ.get("QDRANT_API_KEY"),
                timeout=float(os.environ.get("QDRANT_TIMEOUT", "20") or 20),
            )
            model_name = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
            vec_name = sanitize_vector_name(model_name)
            model = get_embedding_model_fn(model_name) if get_embedding_model_fn else None

            qtext = " ".join([q for q in queries if q]).strip() or queries[0]
            v = next(model.embed([qtext])).tolist()
            k = max(mem_limit, 5)
            res = await asyncio.to_thread(
                lambda: client.search(
                    collection_name=mcoll,
                    query_vector={"name": vec_name, "vector": v},
                    limit=k,
                    with_payload=True,
                )
            )
            for pt in res:
                payload = getattr(pt, "payload", {}) or {}
                md = payload.get("metadata") or {}
                path = str(md.get("path") or "")
                start_line = md.get("start_line")
                end_line = md.get("end_line")
                content = (
                    payload.get("content")
                    or payload.get("text")
                    or payload.get("information")
                    or md.get("information")
                )
                kind = (md.get("kind") or payload.get("kind") or "").lower()
                source_tag = (md.get("source") or payload.get("source") or "").lower()
                flagged = kind in (
                    "memory",
                    "preference",
                    "note",
                    "policy",
                    "infra",
                    "chat",
                ) or source_tag in ("memory", "chat")
                is_memory_like = (
                    (not path)
                    or (start_line in (None, 0) and end_line in (None, 0))
                    or flagged
                )
                if is_memory_like and content:
                    mem_hits.append(
                        {
                            "source": "memory",
                            "score": float(getattr(pt, "score", 0.0) or 0.0),
                            "content": content,
                            "metadata": md,
                        }
                    )
        except Exception:  # pragma: no cover
            pass

    # Fallback: lightweight substring scan over a capped scroll if vector name mismatch
    if include_mem and mem_limit > 0 and not mem_hits and queries:
        try:
            from qdrant_client import QdrantClient  # type: ignore

            client = QdrantClient(
                url=QDRANT_URL,
                api_key=os.environ.get("QDRANT_API_KEY"),
                timeout=float(os.environ.get("QDRANT_TIMEOUT", "20") or 20),
            )
            import re

            terms = [str(t).lower() for t in queries if t]
            tokens = set()
            for t in terms:
                tokens.update([w for w in re.split(r"[^a-z0-9_]+", t) if len(w) >= 3])
            if not tokens:
                tokens = set(terms)
            checked = 0
            cap = 2000
            page = None
            while len(mem_hits) < mem_limit and checked < cap:
                sc, page = await asyncio.to_thread(
                    lambda: client.scroll(
                        collection_name=mcoll,
                        with_payload=True,
                        with_vectors=False,
                        limit=500,
                        offset=page,
                    )
                )
                if not sc:
                    break
                for pt in sc:
                    payload = getattr(pt, "payload", {}) or {}
                    md = payload.get("metadata") or {}
                    path = str(md.get("path") or "")
                    start_line = md.get("start_line")
                    end_line = md.get("end_line")
                    content = (
                        payload.get("content")
                        or payload.get("text")
                        or payload.get("information")
                        or md.get("information")
                    )
                    kind = (md.get("kind") or payload.get("kind") or "").lower()
                    source_tag = (
                        md.get("source") or payload.get("source") or ""
                    ).lower()
                    flagged = kind in (
                        "memory",
                        "preference",
                        "note",
                        "policy",
                        "infra",
                        "chat",
                    ) or source_tag in ("memory", "chat")
                    is_memory_like = (
                        (not path)
                        or (start_line in (None, 0) and end_line in (None, 0))
                        or flagged
                    )
                    if not (is_memory_like and content):
                        continue
                    low = str(content).lower()
                    if any(tok in low for tok in tokens):
                        mem_hits.append(
                            {
                                "source": "memory",
                                "score": 0.5,  # nominal score for substring match; blended via memory_weight
                                "content": content,
                                "metadata": md,
                            }
                        )
                        if len(mem_hits) >= mem_limit:
                            break
                checked += len(sc)
        except Exception:
            pass

    # Blend results
    try:
        mw = (
            float(memory_weight)
            if (memory_weight is not None and str(memory_weight).strip() != "")
            else 0.3
        )
    except (ValueError, TypeError):
        mw = 0.3

    # Build per-source lists with adjusted scores
    code_scored = [{**h, "score": float(h.get("score", 0.0))} for h in code_hits]
    mem_scored = [{**h, "score": float(h.get("score", 0.0)) * mw} for h in mem_hits]

    # Enforce per-source limits before final slice so callers actually get memory hits
    if include_mem and mem_limit > 0:
        code_scored.sort(key=lambda x: -float(x.get("score", 0.0)))
        mem_scored.sort(key=lambda x: -float(x.get("score", 0.0)))
        m_keep = min(len(mem_scored), mem_limit, lim)
        sel_mem = mem_scored[:m_keep]
        c_keep = max(0, min(len(code_scored), code_limit, lim - m_keep))
        sel_code = code_scored[:c_keep]
        blended = sel_code + sel_mem
        blended.sort(
            key=lambda x: (
                -float(x.get("score", 0.0)),
                x.get("source", ""),
                str(x.get("path", "")),
            )
        )
        # No need to slice again; sel_code+sel_mem already <= lim
    else:
        blended = code_scored
        blended.sort(
            key=lambda x: (
                -float(x.get("score", 0.0)),
                x.get("source", ""),
                str(x.get("path", "")),
            )
        )
        blended = blended[:lim]

    # Compact shaping if requested
    if eff_compact:
        compacted: List[Dict[str, Any]] = []
        for b in blended:
            if b.get("source") == "code":
                compacted.append(
                    {
                        "source": "code",
                        "path": b.get("path"),
                        "start_line": b.get("start_line") or 0,
                        "end_line": b.get("end_line") or 0,
                    }
                )
            else:
                compacted.append(
                    {
                        "source": "memory",
                        "content": (b.get("content") or "")[:500],
                    }
                )
        ret = {"results": compacted, "total": len(compacted)}
        if memory_note:
            ret["memory_note"] = memory_note
        ret["diag"] = {
            "code_hits": len(code_hits),
            "mem_hits": len(mem_hits),
            "used_http_fallback": bool(locals().get("used_http_fallback", False)),
            "used_hybrid_fallback": bool(locals().get("used_hybrid_fallback", False)),
        }
        ret["args"] = {
            "queries": queries,
            "collection": coll,
            "limit": int(code_limit),
            "per_path": int(per_path_val),
            "include_memories": bool(include_mem),
            "memory_weight": float(mw),
            "include_snippet": bool(include_snippet),
            "context_lines": int(context_lines)
            if context_lines not in (None, "")
            else 2,
            "compact": bool(eff_compact),
        }
        try:
            if isinstance(code_res, dict):
                ret["diag"]["rerank"] = {
                    "used_rerank": bool(code_res.get("used_rerank")),
                    "counters": code_res.get("rerank_counters") or {},
                }
        except Exception:
            pass
        # Apply TOON formatting if requested or enabled globally
        if _should_use_toon(output_format):
            return _format_context_results_as_toon(ret, compact=True)
        return ret

    ret = {"results": blended, "total": len(blended)}
    if memory_note:
        ret["memory_note"] = memory_note
    ret["diag"] = {
        "code_hits": len(code_hits),
        "mem_hits": len(mem_hits),
        "used_http_fallback": bool(locals().get("used_http_fallback", False)),
        "used_hybrid_fallback": bool(locals().get("used_hybrid_fallback", False)),
    }
    ret["args"] = {
        "queries": queries,
        "collection": coll,
        "limit": int(code_limit),
        "per_path": int(per_path_val),
        "include_memories": bool(include_mem),
        "memory_weight": float(mw),
        "include_snippet": bool(include_snippet),
        "context_lines": int(context_lines) if context_lines not in (None, "") else 2,
        "compact": bool(eff_compact),
    }
    # Apply TOON formatting if requested or enabled globally (use context-aware encoder for memory support)
    if _should_use_toon(output_format):
        return _format_context_results_as_toon(ret, compact=bool(eff_compact))
    return ret

