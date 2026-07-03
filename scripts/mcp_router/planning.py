"""
mcp_router/planning.py - Tool planning and selection.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

from .config import HTTP_URL_INDEXER
from .intent import (
    classify_intent,
    INTENT_ANSWER,
    INTENT_SEARCH,
    INTENT_SEARCH_TESTS,
    INTENT_SEARCH_CONFIG,
    INTENT_SEARCH_CALLERS,
    INTENT_SEARCH_IMPORTERS,
    INTENT_MEMORY_STORE,
    INTENT_MEMORY_FIND,
    INTENT_SYMBOL_GRAPH,
    INTENT_INDEX,
    INTENT_PRUNE,
    INTENT_STATUS,
    INTENT_LIST,
)
from .memory import parse_memory_store_payload
from .hints import parse_repo_hints, clean_query_and_dsl, select_best_search_tool_by_signature
from .scratchpad import load_scratchpad, looks_like_repeat, looks_like_same_filters
from .client import discover_tool_endpoints


def build_plan(q: str) -> List[Tuple[str, Dict[str, Any]]]:
    """Build execution plan for a query."""
    intent = classify_intent(q)
    include_snippet = str(os.environ.get("ROUTER_INCLUDE_SNIPPET", "1")).lower() in {"1", "true", "yes", "on"}
    search_limit = int(os.environ.get("ROUTER_SEARCH_LIMIT", "8") or 8)
    max_tokens_env = os.environ.get("ROUTER_MAX_TOKENS", "").strip()

    def _reuse_last_filters(args: Dict[str, Any]) -> None:
        try:
            if looks_like_same_filters(q):
                sp = load_scratchpad()
                lf = sp.get("last_filters") if isinstance(sp, dict) else None
                if isinstance(lf, dict):
                    for k in ("language", "under", "symbol", "ext", "path_glob", "not_glob"):
                        if k not in args and lf.get(k) not in (None, ""):
                            args[k] = lf.get(k)
        except Exception:
            pass

    # Repeat/redo handling
    try:
        if looks_like_repeat(q):
            sp = load_scratchpad()
            lp = sp.get("last_plan")
            if isinstance(lp, list) and lp:
                norm: list[tuple] = []
                for it in lp:
                    if isinstance(it, (list, tuple)) and len(it) == 2:
                        norm.append((it[0], it[1]))
                if norm:
                    return norm
    except Exception:
        pass

    # Multi-intent: memory store + reindex
    lowq = q.lower()
    if any(w in lowq for w in ["remember this", "store memory", "save memory", "remember that"]) and any(w in lowq for w in ["reindex", "index now", "recreate", "fresh index"]):
        idx_args: Dict[str, Any] = {}
        if any(w in lowq for w in ["recreate", "fresh", "from scratch", "fresh index"]):
            idx_args["recreate"] = True
        info, meta = parse_memory_store_payload(q)
        store_args: Dict[str, Any] = {"information": info or q.strip()}
        if meta:
            allowed = {"priority", "tags", "topic", "category", "owner"}
            cleaned = {k: v for k, v in meta.items() if k in allowed and v not in (None, "", [])}
            if cleaned:
                store_args["metadata"] = cleaned
        return [("store", store_args), ("qdrant_index_root", idx_args)]

    if intent == INTENT_INDEX:
        recreate = True if any(w in q.lower() for w in ["recreate", "fresh", "from scratch"]) else None
        args = {}
        if recreate is True:
            args["recreate"] = True
        return [("qdrant_index_root", args)]

    if intent == INTENT_PRUNE:
        return [("qdrant_prune", {})]

    if intent == INTENT_STATUS:
        return [("qdrant_status", {})]

    if intent == INTENT_LIST:
        return [("qdrant_list", {})]

    if intent == INTENT_SEARCH:
        hints = parse_repo_hints(q)
        clean_q, dsl_filters = clean_query_and_dsl(q)
        args = {"query": clean_q}
        if search_limit:
            args["limit"] = search_limit
        if include_snippet:
            args["include_snippet"] = True
        _reuse_last_filters(args)

        for k in ("language", "under", "symbol", "ext", "path_glob", "not_glob"):
            v = dsl_filters.get(k)
            if v not in (None, "") and k not in args:
                args[k] = v
        for k in ("language", "under", "symbol", "ext", "path_glob", "not_glob"):
            v = hints.get(k)
            if v not in (None, "") and k not in args:
                args[k] = v
        try:
            tool_servers = discover_tool_endpoints(allow_network=False)
            picked = select_best_search_tool_by_signature(q, tool_servers, allow_network=False) or "repo_search"
        except Exception:
            picked = "repo_search"
        return [(picked, args)]

    if intent == INTENT_SEARCH_TESTS:
        hints = parse_repo_hints(q)
        clean_q, dsl_filters = clean_query_and_dsl(q)
        args = {"query": clean_q}
        if search_limit:
            args["limit"] = search_limit
        if include_snippet:
            args["include_snippet"] = True
        _reuse_last_filters(args)

        for k in ("language", "under", "symbol", "ext", "path_glob", "not_glob"):
            v = dsl_filters.get(k)
            if v not in (None, "") and k not in args:
                args[k] = v
        for k in ("language", "under", "symbol", "ext", "path_glob", "not_glob"):
            v = hints.get(k)
            if v not in (None, "") and k not in args:
                args[k] = v
        return [("search_tests_for", args)]

    if intent == INTENT_SEARCH_CONFIG:
        hints = parse_repo_hints(q)
        clean_q, dsl_filters = clean_query_and_dsl(q)
        args = {"query": clean_q}
        if search_limit:
            args["limit"] = search_limit
        if include_snippet:
            args["include_snippet"] = True
        _reuse_last_filters(args)

        for k in ("language", "under", "symbol", "ext", "path_glob", "not_glob"):
            v = dsl_filters.get(k)
            if v not in (None, "") and k not in args:
                args[k] = v
        for k in ("language", "under", "symbol", "ext", "path_glob", "not_glob"):
            v = hints.get(k)
            if v not in (None, "") and k not in args:
                args[k] = v
        return [("search_config_for", args)]

    if intent == INTENT_MEMORY_STORE:
        info, meta = parse_memory_store_payload(q)
        payload: Dict[str, Any] = {"information": info or q.strip()}
        if meta:
            allowed = {"priority", "tags", "topic", "category", "owner"}
            cleaned = {k: v for k, v in meta.items() if k in allowed and v not in (None, "", [])}
            if cleaned:
                payload["metadata"] = cleaned
        return [("store", payload)]

    if intent == INTENT_MEMORY_FIND:
        args = {"query": q}
        if search_limit:
            args["limit"] = max(5, search_limit)
        return [("find", args)]

    if intent == INTENT_SYMBOL_GRAPH:
        hints = parse_repo_hints(q)
        clean_q, dsl_filters = clean_query_and_dsl(q)
        args = {"symbol": clean_q, "query_type": "callers"}
        if search_limit:
            args["limit"] = search_limit
        for k in ("language", "under"):
            v = dsl_filters.get(k) or hints.get(k)
            if v not in (None, ""):
                args[k] = v
        return [("symbol_graph", args)]

    if intent == INTENT_SEARCH_CALLERS:
        hints = parse_repo_hints(q)
        clean_q, dsl_filters = clean_query_and_dsl(q)
        args = {"query": clean_q}
        if search_limit:
            args["limit"] = search_limit
        _reuse_last_filters(args)

        for k in ("language", "under", "symbol", "ext", "path_glob", "not_glob"):
            v = dsl_filters.get(k)
            if v not in (None, "") and k not in args:
                args[k] = v
        for k in ("language", "under", "symbol", "ext", "path_glob", "not_glob"):
            v = hints.get(k)
            if v not in (None, "") and k not in args:
                args[k] = v
        return [("search_callers_for", args)]

    if intent == INTENT_SEARCH_IMPORTERS:
        hints = parse_repo_hints(q)
        clean_q, dsl_filters = clean_query_and_dsl(q)
        args = {"query": clean_q}
        if search_limit:
            args["limit"] = search_limit
        _reuse_last_filters(args)
        for k in ("language", "under", "symbol", "ext", "path_glob", "not_glob"):
            v = dsl_filters.get(k)
            if v not in (None, "") and k not in args:
                args[k] = v
        for k in ("language", "under", "symbol", "ext", "path_glob", "not_glob"):
            v = hints.get(k)
            if v not in (None, "") and k not in args:
                args[k] = v
        return [("search_importers_for", args)]

    if intent == INTENT_ANSWER:
        def _looks_like_design_recap(s: str) -> bool:
            low = s.lower()
            return any(w in low for w in ["recap", "design doc", "architecture", "adr", "retrospective", "postmortem"]) and any(w in low for w in ["design", "summary", "recap", "explain"])

        args: Dict[str, Any] = {"query": q}
        if max_tokens_env:
            try:
                mt = int(max_tokens_env)
                if mt > 0:
                    args["max_tokens"] = mt
            except Exception:
                pass

        hints = parse_repo_hints(q)
        lowq = q.lower()
        if "router" in lowq:
            router_globs = ["**/mcp_router.py", "**/*router*.py"]
            if not hints.get("path_glob"):
                hints["path_glob"] = router_globs
            if not hints.get("language"):
                hints["language"] = "python"
        for k in ("language", "under", "symbol", "ext", "path_glob", "not_glob"):
            v = hints.get(k)
            if v not in (None, ""):
                args[k] = v

        plan: List[Tuple[str, Dict[str, Any]]] = []
        if _looks_like_design_recap(q):
            plan.append(("find", {"query": q, "limit": 3}))
        plan.extend([
            ("context_answer_compat", dict(args)),
            ("context_answer", dict(args)),
            ("repo_search", {**{k: v for k, v in args.items() if k != "max_tokens"}, "limit": max(5, search_limit)}),
        ])
        return plan

    # Fallback
    return [("repo_search", {"query": q, "limit": search_limit})]


async def route_query(query: str) -> Dict[str, Any]:
    """
    Route a query to the appropriate tool.
    
    Returns dict with:
    - tool: Selected tool name
    - confidence: Routing confidence (0.0-1.0)
    - intent: Classified intent
    - args: Tool arguments
    """
    intent = classify_intent(query)
    plan = build_plan(query)
    
    if not plan:
        return {
            "tool": "repo_search",
            "confidence": 0.3,
            "intent": "fallback",
            "args": {"query": query},
        }
    
    # First tool in plan is the primary selection
    tool_name, tool_args = plan[0]
    
    # Map intent to confidence (higher for more specific intents)
    intent_confidence = {
        INTENT_ANSWER: 0.9,
        INTENT_SEARCH: 0.7,
        INTENT_SEARCH_TESTS: 0.85,
        INTENT_SEARCH_CONFIG: 0.85,
        INTENT_SEARCH_CALLERS: 0.85,
        INTENT_SEARCH_IMPORTERS: 0.85,
        INTENT_SYMBOL_GRAPH: 0.9,
        INTENT_MEMORY_STORE: 0.9,
        INTENT_MEMORY_FIND: 0.9,
        INTENT_INDEX: 0.95,
        INTENT_PRUNE: 0.95,
        INTENT_STATUS: 0.95,
        INTENT_LIST: 0.95,
    }
    
    return {
        "tool": tool_name,
        "confidence": intent_confidence.get(intent, 0.5),
        "intent": intent,
        "args": tool_args,
    }
