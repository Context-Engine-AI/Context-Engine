#!/usr/bin/env python3
"""
mcp/search.py - Search tool implementation for MCP indexer server.

Extracted from mcp_indexer_server.py for better modularity.
Contains:
- _repo_search_impl: Main implementation (called by thin @mcp.tool() wrapper)

Note: The @mcp.tool() decorated repo_search function remains in mcp_indexer_server.py
as a thin wrapper that calls _repo_search_impl.
"""

from __future__ import annotations

__all__ = [
    "_repo_search_impl",
]

import json
import os
import re
import logging
import asyncio
import subprocess
import hashlib
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Imports from sibling modules
# ---------------------------------------------------------------------------
from scripts.mcp_impl.utils import (
    _coerce_bool,
    _coerce_int,
    _coerce_str,
    _to_str_list_relaxed,
    _extract_kwargs_payload,
    _tokens_from_queries,
    safe_int,
)
from scripts.mcp_impl.workspace import _default_collection, _work_script
from scripts.mcp_impl.admin_tools import _detect_current_repo, _run_async
from scripts.mcp_toon import _should_use_toon, _format_results_as_toon
from scripts.mcp_auth import require_collection_access as _require_collection_access

# Constants
QDRANT_URL = os.environ.get("QDRANT_URL", "http://qdrant:6333")
SNIPPET_MAX_BYTES = safe_int(
    os.environ.get("MCP_SNIPPET_MAX_BYTES", "8192"),
    default=8192,
    logger=logger,
    context="MCP_SNIPPET_MAX_BYTES",
)


async def _repo_search_impl(
    query: Any = None,
    queries: Any = None,  # Alias for query (many clients use this)
    limit: Any = None,
    # DEBUG: remove after timing investigation
    _debug_entry: bool = True,
    per_path: Any = None,
    include_snippet: Any = None,
    context_lines: Any = None,
    rerank_enabled: Any = None,
    rerank_top_n: Any = None,
    rerank_return_m: Any = None,
    rerank_timeout_ms: Any = None,
    highlight_snippet: Any = None,
    collection: Any = None,
    workspace_path: Any = None,
    mode: Any = None,
    session: Any = None,
    ctx: Any = None,  # MCP Context (passed from wrapper)
    # Structured filters (optional; mirrors hybrid_search flags)
    language: Any = None,
    under: Any = None,
    kind: Any = None,
    symbol: Any = None,
    # Additional structured parity
    path_regex: Any = None,
    path_glob: Any = None,
    not_glob: Any = None,
    ext: Any = None,
    not_: Any = None,
    case: Any = None,
    # Repo scoping (cross-codebase isolation)
    repo: Any = None,  # str, list[str], or "*" to search all repos
    # Response shaping
    compact: Any = None,
    output_format: Any = None,  # "json" (default) or "toon" for token-efficient format
    args: Any = None,  # Compatibility shim for mcp-remote/Claude wrappers that send args/kwargs
    kwargs: Any = None,
    # Injected dependencies from facade
    *,
    get_embedding_model_fn: Any = None,  # callable for _get_embedding_model
    require_auth_session_fn: Any = None,  # callable for _require_auth_session
    do_highlight_snippet_fn: Any = None,  # callable for _do_highlight_snippet
    run_async_fn: Any = None,  # callable for _run_async (subprocess runner)
) -> Dict[str, Any]:
    """Zero-config code search over repositories (hybrid: vector + lexical RRF, rerank ON by default).

    When to use:
    - Find relevant code spans quickly; prefer this over embedding-only search.
    - Use context_answer when you need a synthesized explanation; use context_search to blend with memory notes.

    Key parameters:
    - query: str or list[str]. Multiple queries are fused; accepts "queries" alias.
    - limit: int (default 10). Total results across files.
    - per_path: int (default 2). Max results per file.
    - include_snippet/context_lines: return inline snippets near hits when true.
    - rerank_*: ONNX reranker is ON by default for best relevance; timeouts fall back to hybrid.
    - output_format: "json" (default) or "toon" for token-efficient TOON format.
      Set TOON_ENABLED=1 env var to enable TOON by default.
    - collection: str. Target collection; defaults to workspace state or env COLLECTION_NAME.
    - repo: str or list[str]. Filter by repo name(s). Use "*" to search all repos (disable auto-filter).
      By default, auto-detects current repo from CURRENT_REPO env and filters to it.
      Use repo=["frontend","backend"] to search related repos together.
    - Filters (optional): language, under (path prefix), kind, symbol, ext, path_regex,
      path_glob (str or list[str]), not_glob (str or list[str]), not_ (negative text), case.

    Returns:
    - Dict with keys:
      - results: list of {score, path, symbol, start_line, end_line, why[, components][, relations][, related_paths][, snippet]}
      - total: int; used_rerank: bool; rerank_counters: dict
    - If compact=true (and snippets not requested), results contain only {path,start_line,end_line}.

    Examples:
    - path_glob=["scripts/**","**/*.py"], language="python"
    - symbol="context_answer", under="scripts"
    """
    sess = require_auth_session_fn(session) if require_auth_session_fn else session

    # Use injected run_async or fall back to module import
    _run_async_fn = run_async_fn if run_async_fn is not None else _run_async

    # Handle queries alias (explicit parameter)
    if queries is not None and (query is None or (isinstance(query, str) and str(query).strip() == "")):
        query = queries

    # Accept common alias keys from clients (top-level)
    try:
        if kwargs and (
            limit is None or (isinstance(limit, str) and str(limit).strip() == "")
        ) and ("top_k" in kwargs):
            limit = kwargs.get("top_k")
        if kwargs and (query is None or (isinstance(query, str) and str(query).strip() == "")):
            q_alt = kwargs.get("q") or kwargs.get("text")
            if q_alt is not None:
                query = q_alt
    except Exception:
        pass

    # Leniency: absorb nested 'kwargs' JSON payload some clients send
    try:
        _extra = _extract_kwargs_payload(kwargs)
        if _extra:
            if query is None or (isinstance(query, str) and query.strip() == ""):
                query = _extra.get("query") or _extra.get("queries")
            if limit in (None, "") and _extra.get("limit") is not None:
                limit = _extra.get("limit")
            if per_path in (None, "") and _extra.get("per_path") is not None:
                per_path = _extra.get("per_path")
            if (
                include_snippet in (None, "")
                and _extra.get("include_snippet") is not None
            ):
                include_snippet = _extra.get("include_snippet")
            if context_lines in (None, "") and _extra.get("context_lines") is not None:
                context_lines = _extra.get("context_lines")
            if (
                rerank_enabled in (None, "")
                and _extra.get("rerank_enabled") is not None
            ):
                rerank_enabled = _extra.get("rerank_enabled")
            if rerank_top_n in (None, "") and _extra.get("rerank_top_n") is not None:
                rerank_top_n = _extra.get("rerank_top_n")
            if (
                rerank_return_m in (None, "")
                and _extra.get("rerank_return_m") is not None
            ):
                rerank_return_m = _extra.get("rerank_return_m")
            if (
                rerank_timeout_ms in (None, "")
                and _extra.get("rerank_timeout_ms") is not None
            ):
                rerank_timeout_ms = _extra.get("rerank_timeout_ms")
            if (
                highlight_snippet in (None, "")
                and _extra.get("highlight_snippet") is not None
            ):
                highlight_snippet = _extra.get("highlight_snippet")
            if (
                collection is None
                or (isinstance(collection, str) and collection.strip() == "")
            ) and _extra.get("collection"):
                collection = _extra.get("collection")
            # Optional session token for session-scoped defaults
            if (
                (session is None) or (isinstance(session, str) and str(session).strip() == "")
            ) and _extra.get("session") is not None:
                session = _extra.get("session")

            # Optional workspace_path routing
            if (
                (workspace_path is None)
                or (
                    isinstance(workspace_path, str)
                    and str(workspace_path).strip() == ""
                )
            ) and _extra.get("workspace_path") is not None:
                workspace_path = _extra.get("workspace_path")

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
            if path_glob in (None, "") and _extra.get("path_glob") is not None:
                path_glob = _extra.get("path_glob")
            if not_glob in (None, "") and _extra.get("not_glob") is not None:
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
            if compact in (None, "") and _extra.get("compact") is not None:
                compact = _extra.get("compact")
            # Optional mode hint: "code_first", "docs_first", "balanced"
            if (
                mode is None or (isinstance(mode, str) and str(mode).strip() == "")
            ) and _extra.get("mode") is not None:
                mode = _extra.get("mode")
    except Exception:
        pass

    # Leniency shim: coerce null/invalid args to sane defaults so buggy clients don't fail schema
    def _to_int(x, default):
        try:
            if x is None or (isinstance(x, str) and x.strip() == ""):
                return default
            return int(x)
        except Exception:
            return default

    def _to_bool(x, default):
        if x is None or (isinstance(x, str) and x.strip() == ""):
            return default
        if isinstance(x, bool):
            return x
        s = str(x).strip().lower()
        if s in {"1", "true", "yes", "on"}:
            return True
        if s in {"0", "false", "no", "off"}:
            return False
        return default

    # Session token (top-level or parsed from nested kwargs above)
    sid = (str(session).strip() if session is not None else "")


    def _to_str(x, default=""):
        if x is None:
            return default
        return str(x)

    # Coerce incoming args (which may be null) to proper types
    limit = _to_int(limit, 10)
    per_path = _to_int(per_path, 2)
    include_snippet = _to_bool(include_snippet, True)
    context_lines = _to_int(context_lines, 2)
    # Reranker: default ON; can be disabled via env or client args
    rerank_env_default = str(
        os.environ.get("RERANKER_ENABLED", "1")
    ).strip().lower() in {"1", "true", "yes", "on"}
    rerank_enabled = _to_bool(rerank_enabled, rerank_env_default)
    # Default rerank_top_n=20 balances quality vs latency; increase for benchmarks
    rerank_top_n = _to_int(
        rerank_top_n, int(os.environ.get("RERANKER_TOPN", "20") or 20)
    )
    rerank_return_m = _to_int(
        rerank_return_m, int(os.environ.get("RERANKER_RETURN_M", "20") or 20)
    )
    rerank_timeout_ms = _to_int(
        rerank_timeout_ms, int(os.environ.get("RERANKER_TIMEOUT_MS", "3000") or 3000)
    )
    highlight_snippet = _to_bool(highlight_snippet, True)

    # Resolve collection and related hints: explicit > per-connection defaults > token defaults > env
    coll_hint = _to_str(collection, "").strip()
    mode_hint = _to_str(mode, "").strip()
    under_hint = _to_str(under, "").strip()
    lang_hint = _to_str(language, "").strip()

    # 1) Per-connection defaults via ctx (no token required)
    if ctx is not None and getattr(ctx, "session", None) is not None:
        try:
            with _SESSION_CTX_LOCK:
                _d2 = SESSION_DEFAULTS_BY_SESSION.get(ctx.session) or {}
                if not coll_hint:
                    _sc2 = str((_d2.get("collection") or "")).strip()
                    if _sc2:
                        coll_hint = _sc2
                if not mode_hint:
                    _sm2 = str((_d2.get("mode") or "")).strip()
                    if _sm2:
                        mode_hint = _sm2
                if not under_hint:
                    _su2 = str((_d2.get("under") or "")).strip()
                    if _su2:
                        under_hint = _su2
                if not lang_hint:
                    _sl2 = str((_d2.get("language") or "")).strip()
                    if _sl2:
                        lang_hint = _sl2
        except Exception:
            pass

    # 2) Legacy token-based defaults
    if sid:
        try:
            with _SESSION_LOCK:
                _d = SESSION_DEFAULTS.get(sid) or {}
                if not coll_hint:
                    _sc = str((_d.get("collection") or "")).strip()
                    if _sc:
                        coll_hint = _sc
                if not mode_hint:
                    _sm = str((_d.get("mode") or "")).strip()
                    if _sm:
                        mode_hint = _sm
                if not under_hint:
                    _su = str((_d.get("under") or "")).strip()
                    if _su:
                        under_hint = _su
                if not lang_hint:
                    _sl = str((_d.get("language") or "")).strip()
                    if _sl:
                        lang_hint = _sl
        except Exception:
            pass

    # 3) Environment default (collection only for now)
    env_coll = (os.environ.get("DEFAULT_COLLECTION") or os.environ.get("COLLECTION_NAME") or "").strip()
    if (not coll_hint) and env_coll:
        coll_hint = env_coll

    # Final fallback
    env_fallback = (os.environ.get("DEFAULT_COLLECTION") or os.environ.get("COLLECTION_NAME") or "codebase").strip()
    collection = coll_hint or env_fallback

    _require_collection_access((sess or {}).get("user_id") if sess else None, collection, "read")

    # Optional mode knob: "code_first" (default for IDE), "docs_first", "balanced"
    if not mode:
        mode = mode_hint
    mode_str = _to_str(mode, "").strip().lower()
    dense_mode = mode_str == "dense"

    # Apply defaults for language / under when explicit args are empty
    if not language:
        language = lang_hint
    if not under:
        under = under_hint

    language = _to_str(language, "").strip()
    under = _to_str(under, "").strip()
    kind = _to_str(kind, "").strip()
    symbol = _to_str(symbol, "").strip()
    path_regex = _to_str(path_regex, "").strip()

    # Normalize globs to lists (accept string or list)
    def _to_str_list(x):
        if x is None:
            return []
        if isinstance(x, (list, tuple)):
            out = []
            for e in x:
                s = str(e).strip()
                if s:
                    out.append(s)
            return out
        s = str(x).strip()
        if not s:
            return []
        # support comma-separated shorthand
        return [t.strip() for t in s.split(",") if t.strip()]

    path_globs = _to_str_list(path_glob)
    not_globs = _to_str_list(not_glob)
    ext = _to_str(ext, "").strip()
    not_ = _to_str(not_, "").strip()
    case = _to_str(case, "").strip()

    # Normalize repo filter: str, list[str], or "*" (search all)
    # Default: auto-detect current repo unless REPO_AUTO_FILTER=0
    repo_filter = None
    if repo is not None:
        if isinstance(repo, str):
            r = repo.strip()
            if r == "*":
                repo_filter = "*"  # Explicit "search all repos"
            elif r:
                # Support comma-separated list
                repo_filter = [x.strip() for x in r.split(",") if x.strip()]
        elif isinstance(repo, (list, tuple)):
            repo_filter = [str(x).strip() for x in repo if str(x).strip() and str(x).strip() != "*"]
            if not repo_filter:
                repo_filter = "*"  # Empty list after filtering means search all

    # Auto-detect current repo if not explicitly specified and auto-filter is enabled
    if repo_filter is None and str(os.environ.get("REPO_AUTO_FILTER", "1")).strip().lower() in {"1", "true", "yes", "on"}:
        detected_repo = _detect_current_repo()
        if detected_repo:
            repo_filter = [detected_repo]

    compact_raw = compact
    compact = _to_bool(compact, False)
    # If snippets are requested, do not compact (we need snippet field in results)
    if include_snippet:
        compact = False

    # Default behavior: exclude commit-history docs (which use path=".git") from
    # generic repo_search calls, unless the caller explicitly asks for git
    # content. This prevents normal code queries from surfacing commit-index
    # points as if they were source files.
    if (not language or language.lower() != "git") and (
        not kind or kind.lower() != "git_message"
    ):
        if ".git" not in not_globs:
            not_globs.append(".git")

    # Accept top-level alias `queries` as a drop-in for `query`
    # Many clients send queries=[...] instead of query=[...]
    if kwargs and "queries" in kwargs and kwargs.get("queries") is not None:
        query = kwargs.get("queries")

    # Normalize queries to a list[str] (robust for JSON strings and arrays)
    queries: list[str] = []
    if isinstance(query, (list, tuple)):
        queries = [str(q).strip() for q in query if str(q).strip()]
    elif isinstance(query, str):
        queries = _to_str_list_relaxed(query)
    elif query is not None:
        s = str(query).strip()
        if s:
            queries = [s]

    if not queries:
        return {"error": "query required"}

    # --- Code signal detection for intelligent targeting ---
    # Analyze query for code-like patterns and extract potential symbols
    code_signals = {"has_code_signals": False, "signal_strength": 0.0, "extracted_symbols": [], "detected_patterns": [], "suggested_boosts": {}}
    try:
        combined_query = " ".join(queries)
        code_signals = _detect_code_signals(combined_query)
    except Exception:
        pass

    # If code signals detected and no explicit symbol filter, use extracted symbols for boosting
    auto_symbol_hints: list[str] = []
    if code_signals.get("has_code_signals") and code_signals.get("extracted_symbols"):
        auto_symbol_hints = code_signals["extracted_symbols"]

    env = os.environ.copy()
    env["QDRANT_URL"] = QDRANT_URL
    env["COLLECTION_NAME"] = collection

    # Apply dynamic boosts based on code signal strength
    if (not dense_mode) and code_signals.get("has_code_signals"):
        boosts = code_signals.get("suggested_boosts", {})
        # Boost symbol matching weight dynamically
        if "symbol_boost_multiplier" in boosts:
            base_sym_boost = float(os.environ.get("HYBRID_SYMBOL_BOOST", "0.15"))
            base_sym_eq_boost = float(os.environ.get("HYBRID_SYMBOL_EQUALITY_BOOST", "0.25"))
            mult = boosts["symbol_boost_multiplier"]
            env["HYBRID_SYMBOL_BOOST"] = str(round(base_sym_boost * mult, 3))
            env["HYBRID_SYMBOL_EQUALITY_BOOST"] = str(round(base_sym_eq_boost * mult, 3))
        # Boost implementation files over tests/docs when looking for code
        if "impl_boost_multiplier" in boosts:
            base_impl_boost = float(os.environ.get("HYBRID_IMPLEMENTATION_BOOST", "0.2"))
            mult = boosts["impl_boost_multiplier"]
            env["HYBRID_IMPLEMENTATION_BOOST"] = str(round(base_impl_boost * mult, 3))

    # Pass extracted symbols as additional search hints (augments existing queries)
    if (not dense_mode) and auto_symbol_hints:
        env["CODE_SIGNAL_SYMBOLS"] = ",".join(auto_symbol_hints[:5])

    results = []
    json_lines = []

    # Default subprocess result placeholder (for consistent response shape)
    res = {"ok": True, "code": 0, "stdout": "", "stderr": ""}

    if dense_mode:
        # Use run_pure_dense_search for improved recall via candidate expansion
        # while preserving base-query dense ranking (no fusion/boosts).
        try:
            from scripts.hybrid_search import run_pure_dense_search
        except Exception as e:
            return {"error": f"dense mode unavailable: {e}"}

        # Determine effective candidate pool (respect rerank_top_n if rerank is enabled)
        try:
            base_limit = int(limit)
        except Exception:
            base_limit = 10
        eff_limit = base_limit
        if rerank_enabled:
            try:
                rt = int(rerank_top_n)
            except Exception:
                rt = 0
            if rt > eff_limit:
                eff_limit = rt

        query_text = " ".join(queries)
        
        # run_pure_dense_search handles embedding + candidate expansion
        items = await asyncio.to_thread(
            lambda: run_pure_dense_search(
                query=query_text,
                limit=eff_limit,
                collection=collection,
                language=language or None,
                under=under or None,
                repo=repo_filter,
            )
        )
        
        # Apply post-filters (path_regex, path_glob, not_glob, not_) that aren't
        # supported by run_pure_dense_search's server-side filters
        case_sensitive = str(case or "").strip().lower() in {"sensitive", "true", "1", "yes", "on"}
        import fnmatch as _fnm
        import re as _re

        def _norm_path(p: str) -> str:
            return p if case_sensitive else p.lower()

        path_globs_norm = [g if case_sensitive else g.lower() for g in path_globs]
        not_globs_norm = [g if case_sensitive else g.lower() for g in not_globs]
        path_regex_norm = path_regex or ""

        def _match_glob(glob_pat: str, path_val: str) -> bool:
            if not glob_pat:
                return False
            return _fnm.fnmatchcase(_norm_path(path_val), glob_pat)

        for item in items:
            path = item.get("path") or ""
            
            # Apply path_regex filter
            if path_regex_norm:
                flags = 0 if case_sensitive else _re.IGNORECASE
                try:
                    if not _re.search(path_regex_norm, path, flags=flags):
                        continue
                except Exception:
                    pass
            
            # Apply path_glob filter
            if path_globs_norm and not any(_match_glob(g, path) for g in path_globs_norm):
                continue
            
            # Apply not_glob filter
            if not_globs_norm and any(_match_glob(g, path) for g in not_globs_norm):
                continue
            
            # Apply not_ text filter
            if not_ and not_.lower() in _norm_path(path):
                continue

            payload = item.get("payload") or {}
            if rerank_enabled and isinstance(payload, dict):
                payload_out = payload
            else:
                payload_out = {
                    k: payload[k]
                    for k in ("_id", "code_id", "id")
                    if k in payload
                }
            json_lines.append(
                {
                    "score": float(item.get("score", 0.0)),
                    "path": path,
                    "symbol": item.get("symbol") or "",
                    "start_line": int(item.get("start_line") or 0),
                    "end_line": int(item.get("end_line") or 0),
                    "payload": payload_out,
                }
            )
    else:
        # In-process hybrid search (optional)
        use_hybrid_inproc = str(
            os.environ.get("HYBRID_IN_PROCESS", "")
        ).strip().lower() in {"1", "true", "yes", "on"}
        if use_hybrid_inproc:
            try:
                from scripts.hybrid_search import run_hybrid_search  # type: ignore

                model_name = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
                model = get_embedding_model_fn(model_name) if get_embedding_model_fn else None
                # Determine effective hybrid candidate limit: if rerank is enabled, search up to rerank_top_n
                try:
                    base_limit = int(limit)
                except Exception:
                    base_limit = 10
                eff_limit = base_limit
                if rerank_enabled:
                    try:
                        rt = int(rerank_top_n)
                    except Exception:
                        rt = 0
                    if rt > eff_limit:
                        eff_limit = rt
                # In-process path_glob/not_glob accept a single string; reduce list inputs safely
                print(f"[debug] DEBUG_SEARCH_TIMING={os.environ.get('DEBUG_SEARCH_TIMING', 'not set')}", flush=True)
                items = await asyncio.to_thread(
                    lambda: run_hybrid_search(
                        queries=queries,
                        limit=eff_limit,
                        per_path=(
                            int(per_path)
                            if (per_path is not None and str(per_path).strip() != "")
                            else 1
                        ),
                        language=language or None,
                        under=under or None,
                        kind=kind or None,
                        symbol=symbol or None,
                        ext=ext or None,
                        not_filter=not_ or None,
                        case=case or None,
                        path_regex=path_regex or None,
                        path_glob=(path_globs or None),
                        not_glob=(not_globs or None),
                        expand=str(os.environ.get("HYBRID_EXPAND", "1")).strip().lower()
                        in {"1", "true", "yes", "on"},
                        model=model,
                        collection=collection,
                        mode=mode_str or None,
                        repo=repo_filter,  # Cross-codebase isolation
                    )
                )
                # items are already in structured dict form
                json_lines = items  # reuse downstream shaping
            except Exception as e:
                # Fallback to subprocess path if in-process fails
                logger.debug(f"In-process hybrid search failed, falling back to subprocess: {type(e).__name__}: {e}")
                # VISIBLE ERROR for debugging silent failures during benchmark runs
                print(f"[ERROR] In-process hybrid failed: {type(e).__name__}: {e}", flush=True)
                import traceback
                traceback.print_exc()
                use_hybrid_inproc = False

        if not use_hybrid_inproc:
            # Try hybrid search via subprocess (JSONL output)
            try:
                base_limit = int(limit)
            except Exception:
                base_limit = 10
            eff_limit = base_limit
            if rerank_enabled:
                try:
                    rt = int(rerank_top_n)
                except Exception:
                    rt = 0
                if rt > eff_limit:
                    eff_limit = rt
            cmd = [
                "python",
                _work_script("hybrid_search.py"),
                "--limit",
                str(eff_limit),
                "--json",
            ]
            if per_path is not None and str(per_path).strip() != "":
                cmd += ["--per-path", str(int(per_path))]
            if language:
                cmd += ["--language", language]
            if under:
                cmd += ["--under", under]
            if kind:
                cmd += ["--kind", kind]
            if symbol:
                cmd += ["--symbol", symbol]
            if ext:
                cmd += ["--ext", ext]
            if not_:
                cmd += ["--not", not_]
            if case:
                cmd += ["--case", case]
            if path_regex:
                cmd += ["--path-regex", path_regex]
            for g in path_globs:
                cmd += ["--path-glob", g]
            for g in not_globs:
                cmd += ["--not-glob", g]
            for q in queries:
                cmd += ["--query", q]
            if collection:
                cmd += ["--collection", str(collection)]

            res = await _run_async_fn(cmd, env=env)
            for line in (res.get("stdout") or "").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    json_lines.append(obj)
                except json.JSONDecodeError:
                    continue
            # Fallback: if subprocess yielded nothing (e.g., local dev without /work), try in-process once
            if not json_lines:
                try:
                    from scripts.hybrid_search import run_hybrid_search  # type: ignore

                    model_name = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
                    model = get_embedding_model_fn(model_name) if get_embedding_model_fn else None
                    items = await asyncio.to_thread(
                        lambda: run_hybrid_search(
                            queries=queries,
                            limit=int(limit),
                            per_path=(
                                int(per_path)
                                if (per_path is not None and str(per_path).strip() != "")
                                else 1
                            ),
                            language=language or None,
                            under=under or None,
                            kind=kind or None,
                            symbol=symbol or None,
                            ext=ext or None,
                            not_filter=not_ or None,
                            case=case or None,
                            path_regex=path_regex or None,
                            path_glob=(path_globs or None),
                            not_glob=(not_globs or None),
                            expand=str(os.environ.get("HYBRID_EXPAND", "0")).strip().lower()
                            in {"1", "true", "yes", "on"},
                            model=model,
                            collection=collection,
                            mode=mode_str or None,
                            repo=repo_filter,  # Cross-codebase isolation
                        )
                    )
                    json_lines = items
                except Exception:
                    pass

    # Optional rerank fallback path: if enabled, attempt; on timeout or error, keep hybrid
    used_rerank = False
    rerank_counters = {
        "inproc_hybrid": 0,
        "inproc_dense": 0,
        "subprocess": 0,
        "timeout": 0,
        "error": 0,
        "learning": 0,  # Learning-enabled recursive reranker
    }
    if rerank_enabled:
        # Check for learning reranker mode (learns from ONNX teacher)
        use_learning_rerank = str(
            os.environ.get("RERANK_LEARNING", "")
        ).strip().lower() in {"1", "true", "yes", "on"}

        if use_learning_rerank and json_lines:
            try:
                from scripts.rerank_recursive import rerank_with_learning

                rq = queries[0] if queries else ""
                cand_objs = list(json_lines[: int(rerank_top_n)])

                # Run learning-enabled reranking (collection-aware for weight isolation)
                reranked = rerank_with_learning(
                    query=rq,
                    candidates=cand_objs,
                    limit=int(rerank_return_m),
                    n_iterations=int(os.environ.get("RERANK_LEARNING_ITERS", "3")),
                    collection=collection or "default",
                )

                if reranked:
                    # Format results for output
                    tmp = []
                    for obj in reranked:
                        # Copy the list to avoid mutating the original object
                        why_parts = list(obj.get("why", []))
                        why_parts.append(f"refine:{obj.get('recursive_iterations', 0)}")
                        why_parts.append(f"score:{float(obj.get('score', 0)):.3f}")

                        # Build components with optional fname_boost
                        components = (obj.get("components") or {}) | {
                            "learning_score": float(obj.get("recursive_score", 0)),
                            "refinement_iterations": int(obj.get("recursive_iterations", 0)),
                        }
                        if obj.get("fname_boost"):
                            components["fname_boost"] = float(obj.get("fname_boost", 0))
                            why_parts.append(f"fname:{float(obj.get('fname_boost', 0)):.2f}")

                        # Extract benchmark IDs from payload for CoIR/CoSQA
                        _payload = obj.get("payload") if isinstance(obj, dict) else None
                        if not isinstance(_payload, dict):
                            _payload = {}
                        _doc_id = _payload.get("_id") or _payload.get("code_id") or _payload.get("id")
                        _code_id = _payload.get("code_id")

                        item = {
                            "score": float(obj.get("score", 0)),
                            "path": obj.get("path", ""),
                            "symbol": obj.get("symbol", ""),
                            "start_line": int(obj.get("start_line") or 0),
                            "end_line": int(obj.get("end_line") or 0),
                            "why": why_parts,
                            "components": components,
                            # Benchmark IDs (preserved through rerank)
                            "doc_id": str(_doc_id) if _doc_id is not None else None,
                            "code_id": str(_code_id) if _code_id is not None else None,
                        }
                        # Preserve dual-path metadata
                        if obj.get("host_path"):
                            item["host_path"] = obj["host_path"]
                        if obj.get("container_path"):
                            item["container_path"] = obj["container_path"]
                        tmp.append(item)

                    if tmp:
                        results = tmp
                        used_rerank = True
                        rerank_counters["learning"] += 1
            except Exception:
                pass  # Fall through to standard reranking

        # Resolve in-process gating once and reuse
        use_rerank_inproc = str(
            os.environ.get("RERANK_IN_PROCESS", "")
        ).strip().lower() in {"1", "true", "yes", "on"}
        # Prefer fusion-aware reranking over hybrid candidates when available, but only if in-process reranker is enabled
        if use_rerank_inproc and not used_rerank:
            try:
                if json_lines:
                    from scripts.rerank_local import rerank_local as _rr_local  # type: ignore
                    import concurrent.futures as _fut

                    rq = queries[0] if queries else ""
                    # Prepare candidate docs from top-N hybrid hits (path+symbol + pseudo/tags + small snippet)
                    cand_objs = list(json_lines[: int(rerank_top_n)])

                    def _doc_for(obj: dict) -> str:
                        path = str(obj.get("path") or "")
                        symbol = str(obj.get("symbol") or "")
                        header = f"{symbol} — {path}".strip()

                        # Try to enrich with pseudo/tags from underlying payload when available.
                        # We expect hybrid to have preserved metadata in obj["components"] or
                        # direct fields; if not, we fall back to header+code only.
                        meta_lines: list[str] = [header] if header else []
                        try:
                            # Prefer explicit pseudo/tags fields on the top-level object when present
                            pseudo_val = obj.get("pseudo")
                            tags_val = obj.get("tags")
                            if pseudo_val is None or tags_val is None:
                                # Fallback: inspect a nested metadata view when present
                                md = obj.get("metadata") or {}
                                if pseudo_val is None:
                                    pseudo_val = md.get("pseudo")
                                if tags_val is None:
                                    tags_val = md.get("tags")
                            pseudo_s = str(pseudo_val).strip() if pseudo_val is not None else ""
                            if pseudo_s:
                                # Keep pseudo short to avoid bloating rerank input
                                meta_lines.append(f"Summary: {pseudo_s[:256]}")
                            if tags_val:
                                try:
                                    if isinstance(tags_val, (list, tuple)):
                                        tags_text = ", ".join(
                                            str(x) for x in tags_val
                                        )[:128]
                                        if tags_text:
                                            meta_lines.append(f"Tags: {tags_text}")
                                    else:
                                        tags_text = str(tags_val)[:128]
                                        if tags_text:
                                            meta_lines.append(f"Tags: {tags_text}")
                                except Exception:
                                    pass
                        except Exception:
                            # If any of the above fails, we just keep header-only
                            pass

                        # Priority 1: Check for inline content (CoSQA/CoIR benchmarks store text in payload)
                        inline_text = (
                            obj.get("text") or obj.get("code") or obj.get("snippet") or
                            (obj.get("payload") or {}).get("text") or
                            (obj.get("payload") or {}).get("code")
                        )
                        if inline_text:
                            # Use inline content directly (truncate for reranker input limit)
                            inline_text = str(inline_text).strip()[:2000]
                            if inline_text:
                                meta = "\n".join(meta_lines) if meta_lines else header
                                return (meta + "\n\n" + inline_text).strip()

                        # Priority 2: Read from disk (for file-based corpora like SWE-bench)
                        sl = int(obj.get("start_line") or 0)
                        el = int(obj.get("end_line") or 0)
                        if not path or not sl:
                            return "\n".join(meta_lines) if meta_lines else header
                        try:
                            p = path
                            # Use rerank_base_path from env or workspace_path, fallback to /work
                            base_path = os.environ.get("RERANK_BASE_PATH") or workspace_path or "/work"
                            if not os.path.isabs(p):
                                p = os.path.join(base_path, p)
                            realp = os.path.realpath(p)
                            # Allow any path under the base_path
                            base_real = os.path.realpath(base_path)
                            if not (realp == base_real or realp.startswith(base_real + os.sep)):
                                return "\n".join(meta_lines) if meta_lines else header
                            with open(
                                realp, "r", encoding="utf-8", errors="ignore"
                            ) as f:
                                lines = f.readlines()
                            ctx = (
                                max(1, int(context_lines))
                                if "context_lines" in locals()
                                else 2
                            )
                            si = max(1, sl - ctx)
                            ei = min(len(lines), max(sl, el) + ctx)
                            snippet = "".join(lines[si - 1 : ei]).strip()
                            if snippet:
                                meta = "\n".join(meta_lines) if meta_lines else header
                                return (meta + "\n\n" + snippet).strip()
                            return "\n".join(meta_lines) if meta_lines else header
                        except Exception:
                            return "\n".join(meta_lines) if meta_lines else header

                    # Build docs concurrently
                    max_workers = min(16, (os.cpu_count() or 4) * 4)
                    with _fut.ThreadPoolExecutor(max_workers=max_workers) as ex:
                        docs = list(ex.map(_doc_for, cand_objs))

                    # Debug: log what text reranker is seeing
                    if os.environ.get("DEBUG_RERANK_TEXT"):
                        logger.info(f"[rerank] Query: {rq[:100]}...")
                        for i, doc in enumerate(docs[:3]):
                            preview = doc[:300].replace('\n', '\\n')
                            logger.info(f"[rerank] Doc[{i}] ({len(doc)} chars): {preview}...")

                    # Capture before-rerank order for comparison
                    _before_paths = [(o.get("path", "?").split("/")[-1], o.get("score", 0)) for o in cand_objs[:10]]

                    pairs = [(rq, d) for d in docs]
                    scores = _rr_local(pairs)
                    # Blend rerank with fusion score to preserve pre-rerank boosts
                    # (symbol_exact, impl_boost, path boosts are otherwise lost)
                    _rerank_blend = float(os.environ.get("RERANK_BLEND_WEIGHT", "0.6") or 0.6)
                    _rerank_blend = max(0.0, min(1.0, _rerank_blend))  # clamp [0,1]
                    # Post-rerank symbol boost: apply symbol boosts directly to blended score
                    # This ensures exact symbol matches rank higher even when reranker disagrees
                    _post_symbol_boost = float(os.environ.get("POST_RERANK_SYMBOL_BOOST", "1.0") or 1.0)
                    blended = []
                    # Detect reranker score range for proper normalization
                    # FastEmbed rerankers return 0-1, raw ONNX can return -12 to +12
                    rr_min, rr_max = min(scores), max(scores)
                    rr_range = rr_max - rr_min if rr_max > rr_min else 1.0
                    # Fusion scores are typically 0-3
                    fus_scores = [float(o.get("score", 0.0) or 0.0) for o in cand_objs]
                    fus_max = max(fus_scores) if fus_scores else 3.0
                    fus_max = max(fus_max, 1.0)  # Avoid div by zero

                    for rr_score, obj in zip(scores, cand_objs):
                        fusion_score = float(obj.get("score", 0.0) or 0.0)
                        # Normalize both to 0-1 range for fair blending
                        norm_rerank = (rr_score - rr_min) / rr_range if rr_range > 0 else 0.5
                        norm_fusion = fusion_score / fus_max
                        blended_score = _rerank_blend * norm_rerank + (1.0 - _rerank_blend) * norm_fusion
                        # Apply post-rerank symbol boost: extract symbol boosts from components
                        # and add them directly to blended score (not diluted by blend weight)
                        comps = obj.get("components") or {}
                        sym_sub = float(comps.get("symbol_substr", 0.0) or 0.0)
                        sym_eq = float(comps.get("symbol_exact", 0.0) or 0.0)
                        post_boost = (sym_sub + sym_eq) * _post_symbol_boost
                        blended_score += post_boost
                        blended.append((blended_score, rr_score, obj, post_boost))
                    ranked = sorted(blended, key=lambda x: x[0], reverse=True)
                    tmp = []
                    for blended_s, rr_s, obj, post_b in ranked[: int(rerank_return_m)]:
                        why_parts = obj.get("why", []) + [f"rerank_onnx:{float(rr_s):.3f}", f"blend:{float(blended_s):.3f}"]
                        if post_b > 0:
                            why_parts.append(f"post_sym:{float(post_b):.3f}")
                        # Extract benchmark IDs from payload for CoIR/CoSQA
                        _payload = obj.get("payload") if isinstance(obj, dict) else None
                        if not isinstance(_payload, dict):
                            _payload = {}
                        _doc_id = _payload.get("_id") or _payload.get("code_id") or _payload.get("id")
                        _code_id = _payload.get("code_id")
                        item = {
                            "score": float(blended_s),
                            "path": obj.get("path", ""),
                            "symbol": obj.get("symbol", ""),
                            "start_line": int(obj.get("start_line") or 0),
                            "end_line": int(obj.get("end_line") or 0),
                            "why": why_parts,
                            "components": (obj.get("components") or {})
                            | {"rerank_onnx": float(rr_s), "blended": float(blended_s), "post_symbol_boost": float(post_b)},
                            # Benchmark IDs (preserved through rerank)
                            "doc_id": str(_doc_id) if _doc_id is not None else None,
                            "code_id": str(_code_id) if _code_id is not None else None,
                        }
                        # Preserve dual-path metadata when available so clients can prefer host paths
                        _hostp = obj.get("host_path")
                        _contp = obj.get("container_path")
                        if _hostp:
                            item["host_path"] = _hostp
                        if _contp:
                            item["container_path"] = _contp
                        tmp.append(item)
                    if tmp:
                        results = tmp
                        used_rerank = True
                        rerank_counters["inproc_hybrid"] += 1

                        # Debug: log before/after comparison
                        if os.environ.get("DEBUG_RERANK_AB"):
                            _after_paths = [(t.get("path", "?").split("/")[-1], t.get("score", 0)) for t in tmp[:10]]
                            logger.info(f"[rerank A/B] BEFORE (fusion): {_before_paths}")
                            logger.info(f"[rerank A/B] AFTER (reranked): {_after_paths}")
                            # Show rerank scores
                            _rr_scores = [(t.get("path", "?").split("/")[-1], t.get("why", [])) for t in tmp[:5]]
                            for p, w in _rr_scores:
                                logger.info(f"[rerank A/B] {p}: {w}")
            except Exception:
                used_rerank = False
        # Fallback paths (in-process reranker dense candidates, then subprocess)
        if not used_rerank:
            if use_rerank_inproc:
                try:
                    from scripts.rerank_local import rerank_in_process  # type: ignore

                    model_name = os.environ.get(
                        "EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5"
                    )
                    model = get_embedding_model_fn(model_name) if get_embedding_model_fn else None
                    rq = queries[0] if queries else ""
                    items = rerank_in_process(
                        query=rq,
                        topk=int(rerank_top_n),
                        limit=int(rerank_return_m),
                        language=language or None,
                        under=under or None,
                        model=model,
                        collection=collection,
                    )
                    if items:
                        results = items
                        used_rerank = True
                        rerank_counters["inproc_dense"] += 1
                except Exception:
                    use_rerank_inproc = False
            if (not use_rerank_inproc) and (not used_rerank):
                try:
                    rq = queries[0] if queries else ""
                    rcmd = [
                        "python",
                        _work_script("rerank_local.py"),
                        "--query",
                        rq,
                        "--topk",
                        str(int(rerank_top_n)),
                        "--limit",
                        str(int(rerank_return_m)),
                    ]
                    if collection:
                        rcmd += ["--collection", str(collection)]
                    if language:
                        rcmd += ["--language", language]
                    if under:
                        rcmd += ["--under", under]
                    if os.environ.get("MCP_DEBUG_RERANK", "").strip():
                        try:
                            logger.debug("RERANK_CMD", extra={"cmd": " ".join(rcmd)})
                        except (ValueError, TypeError):
                            pass
                    _floor_ms = int(os.environ.get("RERANK_TIMEOUT_FLOOR_MS", "1000"))
                    try:
                        _req_ms = int(rerank_timeout_ms)
                    except Exception:
                        _req_ms = _floor_ms
                    _eff_ms = max(_floor_ms, _req_ms)
                    _t_sec = max(0.1, _eff_ms / 1000.0)
                    rres = await _run_async_fn(rcmd, env=env, timeout=_t_sec)
                    if os.environ.get("MCP_DEBUG_RERANK", "").strip():
                        logger.debug(
                            "RERANK_RET",
                            extra={
                                "code": rres.get("code"),
                                "out_len": len((rres.get("stdout") or "").strip()),
                                "err_tail": (rres.get("stderr") or "")[-200:],
                            },
                        )
                    if not rres.get("ok"):
                        _stderr = (rres.get("stderr") or "").lower()
                        if rres.get("code") == -1 or "timed out" in _stderr:
                            rerank_counters["timeout"] += 1
                    if rres.get("ok") and (rres.get("stdout") or "").strip():
                        rerank_counters["subprocess"] += 1
                        tmp = []
                        for ln in (rres.get("stdout") or "").splitlines():
                            parts = ln.strip().split("\t")
                            if len(parts) != 4:
                                continue
                            score_s, path, symbol, range_s = parts
                            try:
                                start_s, end_s = range_s.split("-", 1)
                                start_line = int(start_s)
                                end_line = int(end_s)
                            except (ValueError, TypeError):
                                start_line = 0
                                end_line = 0
                            try:
                                score = float(score_s)
                            except (ValueError, TypeError):
                                score = 0.0
                            item = {
                                "score": score,
                                "path": path,
                                "symbol": symbol,
                                "start_line": start_line,
                                "end_line": end_line,
                                "why": [f"rerank_onnx:{score:.3f}"],
                            }
                            tmp.append(item)
                        if tmp:
                            results = tmp
                            used_rerank = True
                            rerank_counters["subprocess"] += 1
                except Exception:
                    rerank_counters["error"] += 1
                    used_rerank = False

    if not used_rerank:
        # Build results from hybrid JSON lines
        for obj in json_lines:
            # NOTE: hybrid_search.py emits a "payload" field intended for benchmarks.
            # We do NOT pass through the full payload here (it can include large code/text),
            # but we *do* extract stable document identifiers for standard corpora (CoSQA/CoIR).
            _payload = obj.get("payload") if isinstance(obj, dict) else None
            if not isinstance(_payload, dict):
                _payload = {}
            # Prefer CoIR's "_id", else CoSQA's "code_id", else any generic "id".
            _doc_id = _payload.get("_id") or _payload.get("code_id") or _payload.get("id")
            _code_id = _payload.get("code_id")
            item = {
                "score": float(obj.get("score", 0.0)),
                "path": obj.get("path", ""),
                "symbol": obj.get("symbol", ""),
                "start_line": int(obj.get("start_line") or 0),
                "end_line": int(obj.get("end_line") or 0),
                "why": obj.get("why", []),
                "components": obj.get("components", {}),
                # Benchmark IDs (small, safe to include in normal responses)
                "doc_id": str(_doc_id) if _doc_id is not None else None,
                "code_id": str(_code_id) if _code_id is not None else None,
            }
            # Preserve dual-path metadata when available so clients can prefer host paths
            _hostp = obj.get("host_path")
            _contp = obj.get("container_path")
            if _hostp:
                item["host_path"] = _hostp
            if _contp:
                item["container_path"] = _contp
            # Pass-through optional relation hints
            if obj.get("relations"):
                item["relations"] = obj.get("relations")
            if obj.get("related_paths"):
                item["related_paths"] = obj.get("related_paths")
            if obj.get("span_budgeted") is not None:
                item["span_budgeted"] = bool(obj.get("span_budgeted"))
            if obj.get("budget_tokens_used") is not None:
                item["budget_tokens_used"] = int(obj.get("budget_tokens_used"))
            # Pass-through index-time pseudo/tags metadata so downstream consumers
            # (e.g., MCP clients, rerankers, IDEs) can optionally incorporate
            # GLM/LLM labels into their own scoring or display logic.
            if obj.get("pseudo") is not None:
                item["pseudo"] = obj.get("pseudo")
            if obj.get("tags") is not None:
                item["tags"] = obj.get("tags")
            results.append(item)

    # Mode-aware reordering: nudge core implementation code vs docs and non-core when requested
    def _is_doc_path(p: str) -> bool:
        pl = str(p or "").lower()
        return (
            "readme" in pl
            or "/docs/" in pl
            or "/documentation/" in pl
            or pl.endswith(".md")
            or pl.endswith(".rst")
            or pl.endswith(".txt")
        )

    def _is_core_code_item(item: dict) -> bool:
        """Classify a result as core implementation code for mode-aware reordering.

        This intentionally reuses hybrid_search's notion of core/test/vendor files
        instead of duplicating extension and path heuristics here. We only apply
        lightweight checks on top (docs/config/tests components) and delegate the
        rest to helpers from hybrid_search when available.
        """
        try:
            raw_path = item.get("path") or ""
            p = str(raw_path)
        except Exception:
            return False
        if not p:
            return False
        # Never treat docs as core code
        if _is_doc_path(p):
            return False

        # Prefer items that were not explicitly tagged as docs/config/tests in hybrid components
        comps = item.get("components") or {}
        try:
            if comps:
                if comps.get("config_penalty") or comps.get("test_penalty") or comps.get("doc_penalty"):
                    return False
        except Exception:
            pass

        # Defer to hybrid_search helpers when available to avoid duplicating
        # extension and path-based logic.
        try:
            from scripts.hybrid_search import (  # type: ignore
                is_core_file as _hy_core_file,
                is_test_file as _hy_is_test_file,
                is_vendor_path as _hy_is_vendor_path,
            )
        except Exception:
            _hy_core_file = None
            _hy_is_test_file = None
            _hy_is_vendor_path = None

        if _hy_core_file:
            try:
                if not _hy_core_file(p):
                    return False
            except Exception:
                return False
        if _hy_is_test_file:
            try:
                if _hy_is_test_file(p):
                    return False
            except Exception:
                pass
        if _hy_is_vendor_path:
            try:
                if _hy_is_vendor_path(p):
                    return False
            except Exception:
                pass

        # If helper imports failed, fall back to a permissive classification:
        # treat the item as core code (we already filtered obvious docs/config/tests).
        return True

    if mode_str in {"code_first", "code-first", "code"}:
        core_items: list[dict] = []
        other_code: list[dict] = []
        doc_items: list[dict] = []
        for it in results:
            p = it.get("path") or ""
            if p and _is_doc_path(p):
                doc_items.append(it)
            elif _is_core_code_item(it):
                core_items.append(it)
            else:
                other_code.append(it)
        results = core_items + other_code + doc_items

        try:
            _min_core = int(os.environ.get("REPO_SEARCH_CODE_FIRST_MIN_CORE", "2") or 0)
        except Exception:
            _min_core = 2
        try:
            _top_k = int(os.environ.get("REPO_SEARCH_CODE_FIRST_TOP_K", "8") or 8)
        except Exception:
            _top_k = 8
        if _min_core > 0 and results:
            top_k = max(0, min(_top_k, len(results)))
            if top_k > 0:
                flags = [_is_core_code_item(it) for it in results]
                cur_core = sum(1 for i in range(top_k) if flags[i])
                if cur_core < _min_core:
                    for src in range(top_k, len(results)):
                        if not flags[src]:
                            continue
                        for dst in range(top_k - 1, -1, -1):
                            if not flags[dst]:
                                results[dst], results[src] = results[src], results[dst]
                                flags[dst], flags[src] = flags[src], flags[dst]
                                cur_core += 1
                                break
                        if cur_core >= _min_core:
                            break
    elif mode_str in {"docs_first", "docs-first", "docs"}:
        core_items = []
        other_code = []
        doc_items = []
        for it in results:
            p = it.get("path") or ""
            if p and _is_doc_path(p):
                doc_items.append(it)
            elif _is_core_code_item(it):
                core_items.append(it)
            else:
                other_code.append(it)
        results = doc_items + core_items + other_code

    # Enforce user-requested limit on final result count
    try:
        _limit_n = int(limit)
    except Exception:
        _limit_n = 0
    if _limit_n > 0 and len(results) > _limit_n:
        results = results[:_limit_n]

    # Optionally add snippets (with highlighting)
    toks = _tokens_from_queries(queries)
    if include_snippet:
        import concurrent.futures as _fut

        def _read_snip(args):
            i, item = args
            try:
                path = item.get("path")
                sl = int(item.get("start_line") or 0)
                el = int(item.get("end_line") or 0)
                if not path or not sl:
                    return (i, "")
                raw_path = (
                    str(item.get("container_path"))
                    if item.get("container_path")
                    else str(path)
                )
                p = (
                    raw_path
                    if os.path.isabs(raw_path)
                    else os.path.join("/work", raw_path)
                )
                realp = os.path.realpath(p)
                if not (realp == "/work" or realp.startswith("/work/")):
                    return (i, "")
                with open(realp, "r", encoding="utf-8", errors="ignore") as f:
                    lines = f.readlines()
                ctx = max(1, int(context_lines))
                si = max(1, sl - ctx)
                ei = min(len(lines), max(sl, el) + ctx)
                snippet = "".join(lines[si - 1 : ei])
                if highlight_snippet:
                    snippet = (
                        do_highlight_snippet_fn(snippet, toks)
                        if do_highlight_snippet_fn
                        else snippet
                    )
                if len(snippet.encode("utf-8", "ignore")) > SNIPPET_MAX_BYTES:
                    _suffix = "\n...[snippet truncated]"
                    _sb = _suffix.encode("utf-8")
                    _bytes = snippet.encode("utf-8", "ignore")
                    _keep = max(0, SNIPPET_MAX_BYTES - len(_sb))
                    _trimmed = _bytes[:_keep]
                    snippet = _trimmed.decode("utf-8", "ignore") + _suffix
                return (i, snippet)
            except Exception:
                return (i, "")

        max_workers = min(16, (os.cpu_count() or 4) * 4)
        with _fut.ThreadPoolExecutor(max_workers=max_workers) as ex:
            for i, snip in ex.map(_read_snip, list(enumerate(results))):
                try:
                    results[i]["snippet"] = snip
                except Exception:
                    pass

    # Smart default: compact true for multi-query calls if compact not explicitly set
    if (len(queries) > 1) and (
        compact_raw is None
        or (isinstance(compact_raw, str) and compact_raw.strip() == "")
    ):
        compact = True

    # Compact mode: return only path and line range
    if os.environ.get("DEBUG_REPO_SEARCH"):
        logger.debug(
            "DEBUG_REPO_SEARCH",
            extra={
                "count": len(results),
                "sample": [
                    {
                        "path": r.get("path"),
                        "symbol": r.get("symbol"),
                        "range": f"{r.get('start_line')}-{r.get('end_line')}",
                    }
                    for r in results[:5]
                ],
            },
        )

    # ─── Filename boost fallback ───────────────────────────────────────────────
    # Apply filename-query correlation boost for results that don't have it yet.
    # The learning reranker applies fname_boost when enabled; this catches:
    #   - Reranking disabled
    #   - Reranking timed out / failed
    #   - Subprocess hybrid search without reranking
    if not dense_mode:
        _fname_boost_factor = float(os.environ.get("FNAME_BOOST", "0.15") or 0.15)
        if _fname_boost_factor > 0 and results:
            _q_str = " ".join(queries).lower()
            _q_toks = {t for t in re.findall(r"[a-z0-9_]{3,}", _q_str) if len(t) >= 3}
            if _q_toks:
                for r in results:
                    # Skip if fname_boost already applied by reranker
                    if r.get("fname_boost") or (r.get("components") or {}).get("fname_boost"):
                        continue

                    # Extract path from various possible keys
                    _path = ""
                    for _pk in ("path", "rel_path", "host_path", "container_path", "client_path"):
                        _pv = r.get(_pk) or (r.get("metadata") or {}).get(_pk)
                        if isinstance(_pv, str) and _pv.strip():
                            _path = _pv.lower()
                            break
                    if not _path:
                        continue

                    # Extract filename base (strip extension)
                    _fname = _path.rsplit("/", 1)[-1] if "/" in _path else _path
                    _fname_base = re.sub(r"\.[^.]+$", "", _fname)
                    _fname_toks = {t for t in re.split(r"[_\-.]", _fname_base) if t and len(t) >= 3}

                    # Require 2+ matching tokens for boost
                    _match_count = len(_q_toks & _fname_toks)
                    if _match_count >= 2:
                        _boost = float(_fname_boost_factor) * _match_count
                        r["score"] = float(r.get("score", 0)) + _boost
                        r["fname_boost"] = _boost
                        # Update components dict if present
                        if "components" in r and isinstance(r["components"], dict):
                            r["components"]["fname_boost"] = _boost
                        # Update why array if present
                        if "why" in r and isinstance(r["why"], list):
                            r["why"].append(f"fname:{_boost:.2f}")
            # Re-sort results by updated score so fname_boost affects ranking
            results = sorted(results, key=lambda x: float(x.get("score", 0)), reverse=True)

    if compact:
        results = [
            {
                "path": r.get("path", ""),
                "start_line": int(r.get("start_line") or 0),
                "end_line": int(r.get("end_line") or 0),
            }
            for r in results
        ]

    response = {
        "args": {
            "queries": queries,
            "limit": int(limit),
            "per_path": int(per_path),
            "include_snippet": bool(include_snippet),
            "context_lines": int(context_lines),
            "rerank_enabled": bool(rerank_enabled),
            "rerank_top_n": int(rerank_top_n),
            "rerank_return_m": int(rerank_return_m),
            "rerank_timeout_ms": int(rerank_timeout_ms),
            "collection": collection,
            "language": language,
            "under": under,
            "kind": kind,
            "symbol": symbol,
            "ext": ext,
            "not": not_,
            "case": case,
            "path_regex": path_regex,
            "path_glob": path_globs,
            "not_glob": not_globs,
            # Echo the user-provided compact flag in args, normalized via _to_bool to respect strings like "false"/"0"
            "compact": (_to_bool(compact_raw, compact)),
        },
        "used_rerank": bool(used_rerank),
        "rerank_counters": rerank_counters,
        "code_signals": code_signals if code_signals.get("has_code_signals") else None,
        "total": len(results),
        "results": results,
        **res,
    }

    # Apply TOON formatting if requested or enabled globally
    # Full mode (compact=False) still saves tokens vs JSON while preserving all fields
    if _should_use_toon(output_format):
        return _format_results_as_toon(response, compact=bool(compact))
    return response
