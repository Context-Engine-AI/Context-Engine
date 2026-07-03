"""
mcp_router - Modular MCP routing package.

This package provides intent classification, tool planning, and HTTP execution
for routing queries to the appropriate MCP tools.

Public API:
- classify_intent: Determine query intent
- build_plan: Create execution plan for query
- call_tool_http: Execute MCP tool over HTTP
- discover_tool_endpoints: Find available tools
"""
from __future__ import annotations

# Config exports
from .config import (
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
)

# Intent exports
from .intent import (
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
    classify_intent,
    get_last_intent_debug,
)

# Memory exports
from .memory import parse_memory_store_payload

# Client exports
from .client import (
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
)

# Scratchpad exports
from .scratchpad import (
    scratchpad_path,
    load_scratchpad,
    save_scratchpad,
    looks_like_repeat,
    looks_like_same_filters,
    looks_like_expand,
)

# Hints exports
from .hints import (
    parse_repo_hints,
    clean_query_and_dsl,
    select_best_search_tool_by_signature,
)

# Batching exports
from .batching import (
    BatchingContextAnswerClient,
    get_batch_client,
)

# Validation exports
from .validation import (
    is_result_good,
    extract_metric_from_resp,
    material_drop,
)

# Planning exports
from .planning import build_plan, route_query

# ---------------------------------------------------------------------------
# Private function imports for backward compatibility
# ---------------------------------------------------------------------------
from .intent import _classify_intent_rules

# ---------------------------------------------------------------------------
# Legacy aliases (underscore-prefixed) for backward compatibility
# ---------------------------------------------------------------------------
_LAST_INTENT_DEBUG = {}  # Use get_last_intent_debug() instead
_BATCH_CLIENT = None  # Lazy initialized

def _get_batch_client():
    global _BATCH_CLIENT
    if _BATCH_CLIENT is None:
        _BATCH_CLIENT = get_batch_client()
    return _BATCH_CLIENT

# Function aliases
_parse_memory_store_payload = parse_memory_store_payload
_looks_like_repeat = looks_like_repeat
_looks_like_same_filters = looks_like_same_filters
_looks_like_expand = looks_like_expand
_load_scratchpad = load_scratchpad
_save_scratchpad = save_scratchpad
_scratchpad_path = scratchpad_path
_scratchpad_ttl_sec = scratchpad_ttl_sec
_cache_ttl_sec = cache_ttl_sec
_discover_tool_endpoints = discover_tool_endpoints
_default_tool_endpoints = default_tool_endpoints
_tools_describe_cached = tools_describe_cached
_is_failure_response = is_failure_response
_is_result_good = is_result_good
_extract_metric_from_resp = extract_metric_from_resp
_material_drop = material_drop
_divergence_thresholds = divergence_thresholds
_divergence_is_fatal_for = divergence_is_fatal_for
_parse_repo_hints = parse_repo_hints
_clean_query_and_dsl = clean_query_and_dsl
_select_best_search_tool_by_signature = select_best_search_tool_by_signature

# Health port aliases
_HEALTH_PORT_INDEXER = HEALTH_PORT_INDEXER
_HEALTH_PORT_MEMORY = HEALTH_PORT_MEMORY

# Language set alias
_LANGS = LANGS


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
    "route_query",
    # CLI
    "main",
]

# Import main for CLI compatibility
from .cli import main
