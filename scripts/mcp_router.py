#!/usr/bin/env python3
"""
Backwards-compatibility shim for mcp_router.

NOTE: This file is NOT used by Python when scripts/mcp_router/ package exists.
Python's import system prioritizes packages (directories with __init__.py) over
modules (.py files) with the same name. This file exists for:

1. Documentation: Shows all available exports at a glance
2. Symmetry: Matches the pattern used by rerank_recursive.py
3. Fallback: Would work if the package directory were removed

All imports resolve to scripts.mcp_router/ (the package):
    from scripts.mcp_router import build_plan, classify_intent
    # or
    from scripts import mcp_router
    mcp_router.build_plan("query")

Usage:
  python -m scripts.mcp_router --plan "How do I ...?"
  python -m scripts.mcp_router --run  "What is hybrid search?"
"""
from __future__ import annotations

# Re-export everything from the package
from scripts.mcp_router import (
    # Config
    HTTP_URL_INDEXER,
    HTTP_URL_MEMORY,
    DEFAULT_HTTP_URL,
    HEALTH_PORT_INDEXER,
    HEALTH_PORT_MEMORY,
    LANGS,
    cache_ttl_sec,
    scratchpad_ttl_sec,
    divergence_thresholds,
    divergence_is_fatal_for,
    # Intent constants
    INTENT_ANSWER,
    INTENT_SEARCH,
    INTENT_SEARCH_TESTS,
    INTENT_SEARCH_CONFIG,
    INTENT_SEARCH_CALLERS,
    INTENT_SEARCH_IMPORTERS,
    INTENT_MEMORY_STORE,
    INTENT_MEMORY_FIND,
    INTENT_INDEX,
    INTENT_PRUNE,
    INTENT_STATUS,
    INTENT_LIST,
    # Intent functions
    classify_intent,
    get_last_intent_debug,
    _classify_intent_rules,
    # Memory
    parse_memory_store_payload,
    # Client
    call_tool_http,
    is_failure_response,
    discover_tool_endpoints,
    default_tool_endpoints,
    tools_describe_cached,
    _mcp_handshake,
    _post_raw,
    _post_raw_retry,
    _parse_stream_or_json,
    _filter_args,
    # Scratchpad
    scratchpad_path,
    load_scratchpad,
    save_scratchpad,
    looks_like_repeat,
    looks_like_same_filters,
    looks_like_expand,
    # Hints
    parse_repo_hints,
    clean_query_and_dsl,
    select_best_search_tool_by_signature,
    # Batching
    BatchingContextAnswerClient,
    get_batch_client,
    # Validation
    is_result_good,
    extract_metric_from_resp,
    material_drop,
    # Planning
    build_plan,
    # CLI
    main,
    # Legacy aliases
    _is_failure_response,
    _is_result_good,
    _discover_tool_endpoints,
)

__all__ = [
    # Config
    "HTTP_URL_INDEXER",
    "HTTP_URL_MEMORY",
    "DEFAULT_HTTP_URL",
    "HEALTH_PORT_INDEXER",
    "HEALTH_PORT_MEMORY",
    "LANGS",
    "cache_ttl_sec",
    "scratchpad_ttl_sec",
    "divergence_thresholds",
    "divergence_is_fatal_for",
    # Intent
    "INTENT_ANSWER",
    "INTENT_SEARCH",
    "INTENT_SEARCH_TESTS",
    "INTENT_SEARCH_CONFIG",
    "INTENT_SEARCH_CALLERS",
    "INTENT_SEARCH_IMPORTERS",
    "INTENT_MEMORY_STORE",
    "INTENT_MEMORY_FIND",
    "INTENT_INDEX",
    "INTENT_PRUNE",
    "INTENT_STATUS",
    "INTENT_LIST",
    "classify_intent",
    "get_last_intent_debug",
    # Memory
    "parse_memory_store_payload",
    # Client
    "call_tool_http",
    "is_failure_response",
    "discover_tool_endpoints",
    "default_tool_endpoints",
    "tools_describe_cached",
    # Scratchpad
    "scratchpad_path",
    "load_scratchpad",
    "save_scratchpad",
    "looks_like_repeat",
    "looks_like_same_filters",
    "looks_like_expand",
    # Hints
    "parse_repo_hints",
    "clean_query_and_dsl",
    "select_best_search_tool_by_signature",
    # Batching
    "BatchingContextAnswerClient",
    "get_batch_client",
    # Validation
    "is_result_good",
    "extract_metric_from_resp",
    "material_drop",
    # Planning
    "build_plan",
    # CLI
    "main",
]

if __name__ == "__main__":
    import sys
    raise SystemExit(main(sys.argv[1:]))
