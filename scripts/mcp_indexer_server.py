#!/usr/bin/env python3
"""
Minimal MCP (SSE) companion server exposing:
- qdrant-list: list collections
- qdrant-index: index the currently mounted path (/work or /work/<subdir>)
- qdrant-prune: prune stale points for the mounted path

This server is designed to run in a Docker container with the repository
bind-mounted at /work (read-only is fine). It reuses the same Python deps as the
indexer image and shells out to our existing scripts to keep behavior consistent.

Environment:
- FASTMCP_HOST (default: 0.0.0.0)
- FASTMCP_INDEXER_PORT (default: 8001)
- QDRANT_URL (e.g., http://qdrant:6333) — server expects Qdrant reachable via this env
- COLLECTION_NAME (default: codebase) — unified collection for seamless cross-repo search

Conventions:
- Repo content must be mounted at /work inside containers
- Clients must not send null values for tool args; omit field or pass empty string ""
- To index repo root: use qdrant_index_root with no args, or qdrant_index with subdir=""

Note: We use the fastmcp library for quick SSE hosting. If you change to another
MCP server framework, keep the tool names and args stable.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# CRITICAL: OpenLit must be initialized BEFORE any qdrant_client imports
# to properly instrument vector DB calls. This import must come first!
# ---------------------------------------------------------------------------
import os as _os
import sys as _sys
_roots_env = _os.environ.get("WORK_ROOTS", "")
_roots = [p.strip() for p in _roots_env.split(",") if p.strip()] or ["/work", "/app"]
for _root in _roots:
    if _root and _root not in _sys.path:
        _sys.path.insert(0, _root)

# Now import OpenLit init (before any other scripts imports)
try:
    from scripts import openlit_init  # noqa: F401 - triggers early instrumentation
except ImportError:
    pass  # OpenLit not available
import json
import asyncio
import re
import uuid

# Prefer orjson for faster serialization (2-3x speedup on large payloads)
try:
    import orjson
    def _json_dumps(obj) -> str:
        return orjson.dumps(obj).decode("utf-8")
    def _json_dumps_bytes(obj) -> bytes:
        return orjson.dumps(obj)
except ImportError:
    orjson = None  # type: ignore
    def _json_dumps(obj) -> str:
        return json.dumps(obj)
    def _json_dumps_bytes(obj) -> bytes:
        return json.dumps(obj).encode("utf-8")

import os
import subprocess
import threading
import time
from typing import Any, Dict, Optional, List, Tuple

from pathlib import Path
import sys

# Import structured logging and error handling (after sys.path setup)
# Will be imported after sys.path is configured below

import contextlib

# Ensure code roots are on sys.path so absolute imports like 'from scripts.x import y' work
# when this file is executed directly (sys.path[0] may be /work/scripts).
# Supports multiple roots via WORK_ROOTS env (comma-separated), defaults to /work and /app.
_roots_env = os.environ.get("WORK_ROOTS", "")
_roots = [p.strip() for p in _roots_env.split(",") if p.strip()] or ["/work", "/app"]
try:
    for _root in _roots:
        if _root and _root not in sys.path:
            sys.path.insert(0, _root)
except Exception:
    pass

# Note: OpenLit initialization is handled by early import of scripts.openlit_init
# at the top of this file (before any qdrant_client imports)

# Session state imported from mcp_workspace shim (-> scripts.mcp.workspace)
# Must be after sys.path setup
from scripts.mcp_workspace import (
    _MEM_COLL_CACHE,
    SESSION_DEFAULTS,
    SESSION_DEFAULTS_BY_SESSION,
    _SESSION_LOCK,
    _SESSION_CTX_LOCK,
)

# Import structured logging and error handling (after sys.path setup)
try:
    from scripts.logger import (
        get_logger,
        ContextLogger,
        RetrievalError,
        IndexingError,
        DecoderError,
        ValidationError,
        ConfigurationError,
        safe_int,
        safe_float,
        safe_bool,
    )

    logger = get_logger(__name__)
except ImportError:
    # Fallback if logger module not available
    import logging

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    # Import safe conversion functions from utils (single source of truth)
    from scripts.mcp_impl.utils import safe_int, safe_float, safe_bool


from scripts.mcp_auth import (
    require_auth_session as _require_auth_session,
    require_collection_access as _require_collection_access,
)

# ---------------------------------------------------------------------------
# Re-exports from extracted modules (backwards compatibility)
# ---------------------------------------------------------------------------
from scripts.mcp_utils import (
    _coerce_bool,
    _coerce_int,
    _coerce_str,
    _coerce_value_string,
    _maybe_parse_jsonish,
    _looks_jsonish_string,
    _parse_kv_string,
    _extract_kwargs_payload,
    _to_str_list_relaxed,
    _split_ident,
    _tokens_from_queries,
    _STOP,
    _env_overrides,
    _primary_identifier_from_queries,
)

from scripts.mcp_toon import (
    _is_toon_output_enabled,
    _should_use_toon,
    _format_results_as_toon,
    _format_context_results_as_toon,
)

# Import implementations from extracted modules
from scripts.mcp_impl.context_search import _context_search_impl
from scripts.mcp_impl.query_expand import _expand_query_impl
from scripts.mcp_impl.search import _repo_search_impl
from scripts.mcp_impl.info_request import (
    _extract_symbols_from_query,
    _extract_related_concepts,
    _format_information_field,
    _extract_relationships,
    _calculate_confidence,
)
from scripts.mcp_impl.admin_tools import _collection_map_impl
from scripts.mcp_impl.search_specialized import (
    _search_tests_for_impl,
    _search_config_for_impl,
    _search_callers_for_impl,
    _search_importers_for_impl,
)
from scripts.mcp_impl.search_history import (
    _search_commits_for_impl,
    _change_history_for_path_impl,
)
from scripts.mcp_impl.symbol_graph import (
    _symbol_graph_impl,
    _format_symbol_graph_toon,
)
from scripts.mcp_impl.pattern_search import _pattern_search_impl

# Global lock to guard temporary env toggles used during ReFRAG retrieval/decoding
_ENV_LOCK = threading.Lock()

# Shared utilities (lex hashing, snippet highlighter)
try:
    from scripts.utils import highlight_snippet as _do_highlight_snippet
except Exception as e:
    logger.warning(f"Failed to import rich for syntax highlighting: {e}")
    _do_highlight_snippet = None  # fallback guarded at call site


# Back-compat shim for tests expecting _highlight_snippet in this module
# Delegates to scripts.utils.highlight_snippet when available
try:

    def _highlight_snippet(snippet, tokens):  # type: ignore
        return (
            _do_highlight_snippet(snippet, tokens) if _do_highlight_snippet else snippet
        )
except Exception:

    def _highlight_snippet(snippet, tokens):  # type: ignore
        return snippet


try:
    from mcp.server.fastmcp import FastMCP, Context  # type: ignore
except Exception as e:  # pragma: no cover
    raise SystemExit("mcp package is required inside the container: pip install mcp")

# TransportSecuritySettings only exists in mcp >= 1.x with transport_security module
try:
    from mcp.server.transport_security import TransportSecuritySettings  # type: ignore
except ImportError:
    TransportSecuritySettings = None  # type: ignore

APP_NAME = os.environ.get("FASTMCP_SERVER_NAME", "qdrant-indexer-mcp")
HOST = os.environ.get("FASTMCP_HOST", "0.0.0.0")
PORT = safe_int(
    os.environ.get("FASTMCP_INDEXER_PORT", "8001"),
    default=8001,
    logger=logger,
    context="FASTMCP_INDEXER_PORT",
)

# Note: _env_overrides and _primary_identifier_from_queries are now imported from scripts.mcp_impl.utils

QDRANT_URL = os.environ.get("QDRANT_URL", "http://qdrant:6333")
DEFAULT_COLLECTION = (
    os.environ.get("DEFAULT_COLLECTION")
    or os.environ.get("COLLECTION_NAME")
    or "codebase"
)
try:
    from scripts.workspace_state import get_collection_name as _ws_get_collection_name  # type: ignore

    if DEFAULT_COLLECTION in {"", "default-collection", "my-collection", "codebase"}:
        resolved = _ws_get_collection_name(None)
        if resolved:
            DEFAULT_COLLECTION = resolved
except Exception:
    pass

MAX_LOG_TAIL = safe_int(
    os.environ.get("MCP_MAX_LOG_TAIL", "4000"),
    default=4000,
    logger=logger,
    context="MCP_MAX_LOG_TAIL",
)
SNIPPET_MAX_BYTES = safe_int(
    os.environ.get("MCP_SNIPPET_MAX_BYTES", "8192"),
    default=8192,
    logger=logger,
    context="MCP_SNIPPET_MAX_BYTES",
)

MCP_TOOL_TIMEOUT_SECS = safe_float(
    os.environ.get("MCP_TOOL_TIMEOUT_SECS", "3600"),
    default=3600.0,
    logger=logger,
    context="MCP_TOOL_TIMEOUT_SECS",
)

# Set default environment variables for context_answer functionality
# DEBUG_CONTEXT_ANSWER defaults to 0 for production; enable explicitly if needed
os.environ.setdefault("DEBUG_CONTEXT_ANSWER", "0")
os.environ.setdefault("REFRAG_DECODER", "1")
os.environ.setdefault("LLAMACPP_URL", "http://localhost:8080")
os.environ.setdefault("USE_GPU_DECODER", "0")
os.environ.setdefault(
    "CTX_REQUIRE_IDENTIFIER", "0"
)  # Disable strict identifier requirement


# --- TOON functions imported from scripts.mcp_toon ---
# (see imports at top of file for backwards compatibility re-exports)

# --- Workspace state functions imported from mcp_workspace shim ---
from scripts.mcp_workspace import (
    _state_file_path,
    _read_ws_state,
    _default_collection,
    _work_script,
)

# Disable DNS rebinding protection - breaks Docker internal networking (Host: mcp:8000)
_security_settings = (
    TransportSecuritySettings(enable_dns_rebinding_protection=False)
    if TransportSecuritySettings
    else None
)
mcp = FastMCP(APP_NAME, transport_security=_security_settings)


# Capture tool registry automatically by wrapping the decorator once
_TOOLS_REGISTRY: list[dict] = []
try:
    _orig_tool = mcp.tool

    def _tool_capture_wrapper(*dargs, **dkwargs):
        orig_deco = _orig_tool(*dargs, **dkwargs)

        def _inner(fn):
            try:
                _TOOLS_REGISTRY.append(
                    {
                        "name": dkwargs.get("name") or getattr(fn, "__name__", ""),
                        "description": (getattr(fn, "__doc__", None) or "").strip(),
                    }
                )
            except (AttributeError, TypeError) as e:
                logger.warning(f"Failed to capture tool metadata for {fn}", exc_info=e)
            return orig_deco(fn)

        return _inner

    mcp.tool = _tool_capture_wrapper  # type: ignore
except (AttributeError, TypeError) as e:
    logger.warning("Failed to wrap mcp.tool decorator", exc_info=e)


def _relax_var_kwarg_defaults() -> None:
    """Allow tools that rely on **kwargs compatibility shims to be invoked without
    callers supplying an explicit 'kwargs' or 'arguments' field."""
    try:
        from pydantic_core import PydanticUndefined as _PydanticUndefined  # type: ignore
    except Exception:  # pragma: no cover - defensive

        class _Sentinel:  # type: ignore
            pass

        _PydanticUndefined = _Sentinel()  # type: ignore

    try:
        tool_manager = getattr(mcp, "_tool_manager", None)
        tools = getattr(tool_manager, "_tools", {}) if tool_manager is not None else {}
    except Exception:
        tools = {}

    for tool in tools.values():
        try:
            model = getattr(tool.fn_metadata, "arg_model", None)
            if model is None:
                continue
            fields = getattr(model, "model_fields", {})
            changed = False
            for key in ("kwargs", "arguments"):
                fld = fields.get(key)
                if fld is None:
                    continue
                default = getattr(fld, "default", None)
                default_factory = getattr(fld, "default_factory", None)
                if default is _PydanticUndefined and default_factory is None:
                    try:
                        fld.default_factory = dict  # type: ignore[attr-defined]
                    except Exception:
                        fld.default_factory = lambda: {}  # type: ignore
                    fld.default = None
                    changed = True
            if changed:
                try:
                    model.model_rebuild(force=True)
                except Exception:
                    pass
        except Exception:
            continue


# Lightweight readiness endpoint on a separate health port (non-MCP), optional
# Exposes GET /readyz returning {ok: true, app: <name>} once process is up.
HEALTH_PORT = safe_int(
    os.environ.get("FASTMCP_HEALTH_PORT", "18001"),
    default=18001,
    logger=logger,
    context="FASTMCP_HEALTH_PORT",
)


def _start_readyz_server():
    try:
        from http.server import BaseHTTPRequestHandler, HTTPServer

        class H(BaseHTTPRequestHandler):
            def do_GET(self):
                try:
                    if self.path == "/readyz":
                        self.send_response(200)
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        payload = {"ok": True, "app": APP_NAME}
                        self.wfile.write(_json_dumps_bytes(payload))
                    elif self.path == "/tools":
                        self.send_response(200)
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        # Hide expand_query when decoder is disabled
                        tools = _TOOLS_REGISTRY
                        try:
                            from scripts.refrag_llamacpp import is_decoder_enabled  # type: ignore
                        except Exception:
                            is_decoder_enabled = lambda: False  # type: ignore
                        try:
                            if not is_decoder_enabled():
                                tools = [
                                    t
                                    for t in tools
                                    if (t.get("name") or "") != "expand_query"
                                ]
                        except Exception:
                            pass
                        payload = {"ok": True, "tools": tools}
                        self.wfile.write(_json_dumps_bytes(payload))
                    else:
                        self.send_response(404)
                        self.end_headers()
                except Exception:
                    try:
                        self.send_response(500)
                        self.end_headers()
                    except Exception:
                        pass

            def log_message(self, *args, **kwargs):
                # Quiet health server logs
                return

        srv = HTTPServer((HOST, HEALTH_PORT), H)
        th = threading.Thread(target=srv.serve_forever, daemon=True)
        th.start()
        return True
    except Exception:
        return False


# Import the new subprocess manager
try:
    from scripts.subprocess_manager import run_subprocess_async
except ImportError:
    # Fallback if subprocess_manager not available
    logger.warning("subprocess_manager not available, using fallback implementation")

    async def run_subprocess_async(
        cmd: List[str],
        timeout: Optional[float] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Fallback subprocess runner if subprocess_manager is not available."""
        proc: Optional[asyncio.subprocess.Process] = None
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            # Default timeout from env if not provided by caller
            if timeout is None:
                timeout = MCP_TOOL_TIMEOUT_SECS
            try:
                stdout_b, stderr_b = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout
                )
                code = proc.returncode
            except asyncio.TimeoutError:
                try:
                    proc.kill()
                except Exception:
                    pass
                return {
                    "ok": False,
                    "code": -1,
                    "stdout": "",
                    "stderr": f"Command timed out after {timeout}s",
                }
            stdout = (stdout_b or b"").decode("utf-8", errors="ignore")
            stderr = (stderr_b or b"").decode("utf-8", errors="ignore")

            def _cap_tail(s: str) -> str:
                if not s:
                    return s
                return (
                    s
                    if len(s) <= MAX_LOG_TAIL
                    else ("...[tail truncated]\n" + s[-MAX_LOG_TAIL:])
                )

            return {
                "ok": code == 0,
                "code": code,
                "stdout": _cap_tail(stdout),
                "stderr": _cap_tail(stderr),
            }
        except Exception as e:
            return {"ok": False, "code": -2, "stdout": "", "stderr": str(e)}
        finally:
            try:
                if proc is not None:
                    if proc.stdout is not None:
                        proc.stdout.close()
                    if proc.stderr is not None:
                        proc.stderr.close()
                    # Ensure the process is reaped
                    with contextlib.suppress(Exception):
                        await proc.wait()
            except Exception:
                pass


# --- Admin tool helpers imported from mcp_admin_tools shim ---
from scripts.mcp_admin_tools import (
    _EMBED_MODEL_CACHE,
    _EMBED_MODEL_LOCKS,
    _run_async,
    _get_embedding_model,
    _invalidate_router_scratchpad,
    _detect_current_repo,
)

# Lenient argument normalization to tolerate buggy clients (e.g., JSON-in-kwargs, booleans where strings expected)
# Note: _maybe_parse_jsonish and other parsing helpers are now imported from scripts.mcp_utils
from typing import Any as _Any, Dict as _Dict

# Extra parsing helpers for quirky clients that send stringified kwargs
import urllib.parse as _urlparse, ast as _ast

# --- Utility functions imported from scripts.mcp_utils ---
# (see imports at top of file for backwards compatibility re-exports:
#  _parse_kv_string, _coerce_value_string, _to_str_list_relaxed,
#  _extract_kwargs_payload, _looks_jsonish_string, _coerce_bool,
#  _coerce_int, _coerce_str, _STOP, _split_ident, _tokens_from_queries)


@mcp.tool()
async def qdrant_index_root(
    recreate: Optional[bool] = None, collection: Optional[str] = None, session: Optional[str] = None
) -> Dict[str, Any]:
    """Initialize or refresh the vector index for the workspace root (/work).

    When to use:
    - First-time setup for a repo, or to reindex the whole workspace
    - After large refactors or schema changes (set recreate=true)
    - If you want a clean collection or to switch the target collection

    Parameters:
    - recreate: bool (default: false). Drop/recreate the collection before indexing.
    - collection: str (optional). Target collection; defaults to workspace state or env COLLECTION_NAME.

    Returns: subprocess result from ingest_code.py with args echoed. On success code==0.
    Notes:
    - Omit fields instead of sending null values.
    - Safe to call repeatedly; unchanged files are skipped by the indexer.
    """
    sess = _require_auth_session(session)

    # Leniency: if clients embed JSON in 'collection' (and include 'recreate'), parse it
    try:
        if _looks_jsonish_string(collection):
            _parsed = _maybe_parse_jsonish(collection)
            if isinstance(_parsed, dict):
                collection = _parsed.get("collection", collection)
                if recreate is None and "recreate" in _parsed:
                    recreate = _coerce_bool(_parsed.get("recreate"), False)
    except Exception:
        pass

    # Resolve collection: prefer explicit value; otherwise use workspace state
    try:
        _c = (collection or "").strip()
    except Exception:
        _c = ""
    # Empty string means use workspace state default (codebase)
    if _c:
        coll = _c
    else:
        try:
            from scripts.workspace_state import (
                get_collection_name as _ws_get_collection_name,
                is_multi_repo_mode as _ws_is_multi_repo_mode,
            )  # type: ignore

            if _ws_is_multi_repo_mode():
                coll = _ws_get_collection_name("/work") or _default_collection()
            else:
                coll = _ws_get_collection_name(None) or _default_collection()
        except Exception:
            coll = _default_collection()

    _require_collection_access((sess or {}).get("user_id") if sess else None, coll, "write")

    env = os.environ.copy()
    env["QDRANT_URL"] = QDRANT_URL
    env["COLLECTION_NAME"] = coll

    cmd = ["python", _work_script("ingest_code.py"), "--root", "/work"]
    if recreate:
        cmd.append("--recreate")

    res = await _run_async(cmd, env=env)
    ret = {"args": {"root": "/work", "collection": coll, "recreate": recreate}, **res}
    try:
        if ret.get("ok") and int(ret.get("code", 1)) == 0:
            if _invalidate_router_scratchpad("/work"):
                ret["invalidated_router_scratchpad"] = True
    except Exception:
        pass
    return ret


@mcp.tool()
async def qdrant_list(kwargs: Any = None) -> Dict[str, Any]:
    """List available Qdrant collections.

    When to use:
    - Inspect which collections exist before indexing/searching
    - Debug collection naming in multi-workspace setups

    Parameters:
    - (none). Extra params are ignored.

    Returns:
    - {"collections": [str, ...]} or {"error": "..."}
    """
    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(
            url=QDRANT_URL,
            api_key=os.environ.get("QDRANT_API_KEY"),
            timeout=float(os.environ.get("QDRANT_TIMEOUT", "20") or 20),
        )
        cols_info = await asyncio.to_thread(client.get_collections)
        return {"collections": [c.name for c in cols_info.collections]}
    except ImportError:
        return {"error": "qdrant_client is not installed in this container"}
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
async def workspace_info(
    workspace_path: Optional[str] = None, kwargs: Any = None
) -> Dict[str, Any]:
    """Read .codebase/state.json for the current workspace and resolve defaults.

    When to use:
    - Determine the default collection used by this workspace
    - Inspect indexing status and metadata saved by indexer/watch

    Parameters:
    - workspace_path: str (optional). Defaults to "/work".

    Returns:
    - {"workspace_path": str, "default_collection": str, "source": "state_file"|"env", "state": dict}
    """
    ws_path = (workspace_path or "/work").strip() or "/work"


    st = _read_ws_state(ws_path) or {}
    coll = (
        (st.get("qdrant_collection") if isinstance(st, dict) else None)
        or os.environ.get("DEFAULT_COLLECTION")
        or os.environ.get("COLLECTION_NAME")
        or DEFAULT_COLLECTION
    )
    return {
        "workspace_path": ws_path,
        "default_collection": coll,
        "source": ("state_file" if st else "env"),
        "state": st or {},
    }


@mcp.tool()
async def list_workspaces(search_root: Optional[str] = None) -> Dict[str, Any]:
    """Scan search_root recursively for .codebase/state.json and summarize workspaces.

    When to use:
    - Multi-repo environments; pick a workspace/collection to operate on

    Parameters:
    - search_root: str (optional). Directory to scan; defaults to parent of /work.

    Returns:
    - {"workspaces": [{"workspace_path": str, "collection_name": str, "last_updated": str|int, "indexing_state": str}, ...]}
    """
    try:
        from scripts.workspace_state import list_workspaces as _lw  # type: ignore

        items = await asyncio.to_thread(lambda: _lw(search_root))
        return {"workspaces": items}
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# collection_map - thin wrapper delegating to _collection_map_impl
# ---------------------------------------------------------------------------
@mcp.tool()
async def collection_map(
    search_root: Optional[str] = None,
    collection: Optional[str] = None,
    repo_name: Optional[str] = None,
    include_samples: Optional[bool] = None,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Return collection↔repo mappings with optional Qdrant payload samples."""
    return await _collection_map_impl(
        search_root=search_root,
        collection=collection,
        repo_name=repo_name,
        include_samples=include_samples,
        limit=limit,
        coerce_bool_fn=_coerce_bool,
    )


@mcp.tool()
async def qdrant_status(
    collection: Optional[str] = None,
    max_points: Optional[int] = None,
    batch: Optional[int] = None,
    kwargs: Any = None,
) -> Dict[str, Any]:
    """Summarize collection size and recent index timestamps.

    When to use:
    - Check whether indexing ran recently and overall point count

    Parameters:
    - collection: str (optional). Defaults to env COLLECTION_NAME.
    - max_points: int. Cap scanned points when estimating timestamps (default 5000).
    - batch: int. Scroll page size (default 1000).

    Returns:
    - {"collection": str, "count": int, "scanned_points": int,
       "last_ingested_at": {"unix": int, "iso": str},
       "last_modified_at": {"unix": int, "iso": str}}
    - or {"error": "..."}
    """
    # Leniency: absorb 'kwargs' JSON payload some clients send instead of top-level args
    try:
        _extra = _extract_kwargs_payload(kwargs)
        if _extra and not collection:
            collection = _extra.get("collection", collection)
        if _extra and max_points in (None, "") and _extra.get("max_points") is not None:
            max_points = _coerce_int(_extra.get("max_points"), None)
        if _extra and batch in (None, "") and _extra.get("batch") is not None:
            batch = _coerce_int(_extra.get("batch"), None)
    except Exception:
        pass
    coll = collection or _default_collection()
    try:
        from qdrant_client import QdrantClient
        import datetime as _dt

        client = QdrantClient(
            url=QDRANT_URL,
            api_key=os.environ.get("QDRANT_API_KEY"),
            timeout=float(os.environ.get("QDRANT_TIMEOUT", "20") or 20),
        )
        # Count points
        try:
            cnt_res = await asyncio.to_thread(
                lambda: client.count(collection_name=coll, exact=True)
            )
            total = int(getattr(cnt_res, "count", 0))
        except Exception:
            total = 0
        # Scan a limited number of points to estimate last timestamps
        max_points = (
            int(max_points)
            if max_points not in (None, "")
            else int(os.environ.get("MCP_STATUS_MAX_POINTS", "5000"))
        )
        batch = int(batch) if batch not in (None, "") else 1000
        scanned = 0
        last_ing = None
        last_mod = None
        next_page = None
        while scanned < max_points:
            limit = min(batch, max_points - scanned)
            try:
                pts, next_page = await asyncio.to_thread(
                    lambda: client.scroll(
                        collection_name=coll,
                        limit=limit,
                        offset=next_page,
                        with_payload=True,
                        with_vectors=False,
                    )
                )
            except Exception:
                # Fallback without offset keyword (older clients)
                pts, next_page = await asyncio.to_thread(
                    lambda: client.scroll(
                        collection_name=coll,
                        limit=limit,
                        with_payload=True,
                        with_vectors=False,
                    )
                )
            if not pts:
                break
            scanned += len(pts)
            for p in pts:
                md = (p.payload or {}).get("metadata") or {}
                ti = md.get("ingested_at")
                tm = md.get("last_modified_at")
                if isinstance(ti, int):
                    last_ing = ti if last_ing is None else max(last_ing, ti)
                if isinstance(tm, int):
                    last_mod = tm if last_mod is None else max(last_mod, tm)
            if not next_page:
                break

        def _iso(ts):
            if isinstance(ts, int) and ts > 0:
                try:
                    return _dt.datetime.fromtimestamp(ts, _dt.timezone.utc).isoformat()
                except Exception:
                    return ""
            return ""

        return {
            "collection": coll,
            "count": total,
            "scanned_points": scanned,
            "last_ingested_at": {"unix": last_ing or 0, "iso": _iso(last_ing)},
            "last_modified_at": {"unix": last_mod or 0, "iso": _iso(last_mod)},
        }
    except Exception as e:
        return {"collection": coll, "error": str(e)}


@mcp.tool()
async def qdrant_index(
    subdir: Optional[str] = None,
    recreate: Optional[bool] = None,
    collection: Optional[str] = None,
    session: Optional[str] = None,
) -> Dict[str, Any]:
    """Index the workspace (/work) or a specific subdirectory.

    Use this when you want to index only part of the repo (e.g., "scripts" or "backend/api").
    For full-repo indexing, prefer qdrant_index_root.

    Parameters:
    - subdir: str. "" or omit to index the root; or a relative path under /work (e.g., "scripts").
    - recreate: bool (default: false). Drop/recreate the collection before indexing.
    - collection: str (optional). Target collection; defaults to workspace state or env COLLECTION_NAME.

    Returns: subprocess result from ingest_code.py with args echoed. On success code==0.
    Notes:
    - Paths are sandboxed to /work; attempts to escape will be rejected.
    - Omit fields rather than sending null values.
    """
    sess = _require_auth_session(session)

    # Leniency: parse JSON-ish payloads mistakenly sent in 'collection' or 'subdir'
    try:
        if _looks_jsonish_string(collection):
            _parsed = _maybe_parse_jsonish(collection)
            if isinstance(_parsed, dict):
                subdir = _parsed.get("subdir", subdir)
                collection = _parsed.get("collection", collection)
                if recreate is None and "recreate" in _parsed:
                    recreate = _coerce_bool(_parsed.get("recreate"), False)
        if _looks_jsonish_string(subdir):
            _parsed2 = _maybe_parse_jsonish(subdir)
            if isinstance(_parsed2, dict):
                subdir = _parsed2.get("subdir", subdir)
                collection = _parsed2.get("collection", collection)
                if recreate is None and "recreate" in _parsed2:
                    recreate = _coerce_bool(_parsed2.get("recreate"), False)
    except Exception:
        pass

    root = "/work"
    if subdir:
        subdir = subdir.lstrip("/")
        root = os.path.join(root, subdir)
    # Enforce /work sandbox
    real_root = os.path.realpath(root)
    if not (real_root == "/work" or real_root.startswith("/work/")):
        return {"ok": False, "error": "subdir escapes /work sandbox"}
    root = real_root
    # Resolve collection: prefer explicit value; otherwise use workspace state (use workspace root)
    try:
        _c2 = (collection or "").strip()
    except Exception:
        _c2 = ""
    # Empty string means use workspace state default (codebase)
    if _c2:
        coll = _c2
    else:
        try:
            from scripts.workspace_state import (
                get_collection_name as _ws_get_collection_name,
                is_multi_repo_mode as _ws_is_multi_repo_mode,
            )  # type: ignore

            if _ws_is_multi_repo_mode():
                coll = _ws_get_collection_name(root) or _default_collection()
            else:
                coll = _ws_get_collection_name(None) or _default_collection()
        except Exception:
            coll = _default_collection()

    _require_collection_access((sess or {}).get("user_id") if sess else None, coll, "write")

    env = os.environ.copy()
    env["QDRANT_URL"] = QDRANT_URL
    env["COLLECTION_NAME"] = coll

    cmd = [
        "python",
        _work_script("ingest_code.py"),
        "--root",
        root,
    ]
    if recreate:
        cmd.append("--recreate")

    res = await _run_async(cmd, env=env)
    ret = {"args": {"root": root, "collection": coll, "recreate": recreate}, **res}
    try:
        if ret.get("ok") and int(ret.get("code", 1)) == 0:
            if _invalidate_router_scratchpad("/work"):
                ret["invalidated_router_scratchpad"] = True
    except Exception:
        pass
    return ret


@mcp.tool()
async def set_session_defaults(
    collection: Any = None,
    mode: Any = None,
    under: Any = None,
    language: Any = None,
    session: Any = None,
    ctx: Context = None,
    **kwargs,
) -> Dict[str, Any]:
    """Set defaults (e.g., collection, mode, under) for subsequent calls.

    Behavior:
    - If request Context is available, persist defaults per-connection so later calls on
      the same MCP session automatically use them (no token required).
    - Optionally also stores token-scoped defaults for cross-connection reuse.
    """
    try:
        _extra = _extract_kwargs_payload(kwargs)
        if _extra:
            if (collection is None or (isinstance(collection, str) and collection.strip() == "")) and _extra.get("collection") is not None:
                collection = _extra.get("collection")
            if (mode is None or (isinstance(mode, str) and str(mode).strip() == "")) and _extra.get("mode") is not None:
                mode = _extra.get("mode")
            if (under is None or (isinstance(under, str) and str(under).strip() == "")) and _extra.get("under") is not None:
                under = _extra.get("under")
            if (language is None or (isinstance(language, str) and str(language).strip() == "")) and _extra.get("language") is not None:
                language = _extra.get("language")
            if (session is None or (isinstance(session, str) and str(session).strip() == "")) and _extra.get("session") is not None:
                session = _extra.get("session")
    except Exception:
        pass

    defaults: Dict[str, Any] = {}
    unset_keys: set[str] = set()
    for _key, _val in (("collection", collection), ("mode", mode), ("under", under), ("language", language)):
        if isinstance(_val, str):
            _s = _val.strip()
            if _s:
                defaults[_key] = _s
            else:
                unset_keys.add(_key)

    # Per-connection storage (preferred)
    try:
        if ctx is not None and getattr(ctx, "session", None) is not None and (defaults or unset_keys):
            with _SESSION_CTX_LOCK:
                existing2 = SESSION_DEFAULTS_BY_SESSION.get(ctx.session) or {}
                for _k in unset_keys:
                    existing2.pop(_k, None)
                existing2.update(defaults)
                SESSION_DEFAULTS_BY_SESSION[ctx.session] = existing2
    except Exception:
        pass

    # Optional token storage
    sid = str(session).strip() if session is not None else ""
    if not sid:
        sid = uuid.uuid4().hex[:12]
    try:
        if defaults or unset_keys:
            with _SESSION_LOCK:
                existing = SESSION_DEFAULTS.get(sid) or {}
                for _k in unset_keys:
                    existing.pop(_k, None)
                existing.update(defaults)
                SESSION_DEFAULTS[sid] = existing
    except Exception:
        pass

    return {
        "ok": True,
        "session": sid,
        "defaults": SESSION_DEFAULTS.get(sid, {}),
        "applied": ("connection" if (ctx is not None and getattr(ctx, "session", None) is not None) else "token"),
    }

@mcp.tool()
async def qdrant_prune(kwargs: Any = None, **ignored: Any) -> Dict[str, Any]:
    """Remove stale points for /work (files deleted/moved but still in the index).

    Extra arguments are accepted for forward compatibility but ignored.
    Returns the subprocess result from ``prune.py`` with status information.
    """
    env = os.environ.copy()
    env["PRUNE_ROOT"] = "/work"

    cmd = ["python", _work_script("prune.py")]
    res = await _run_async(cmd, env=env)
    return res


# ---------------------------------------------------------------------------
# Code signal detection imported from mcp_code_signals shim
# ---------------------------------------------------------------------------
from scripts.mcp_code_signals import (
    _CODE_INTENT_CACHE,
    _CODE_INTENT_LOCK,
    _CODE_QUERY_ARCHETYPES,
    _PROSE_QUERY_ARCHETYPES,
    _CODE_SIGNAL_PATTERNS,
    _CODE_KEYWORDS,
    _init_code_intent_centroids,
    _detect_code_intent_embedding,
    _detect_code_signals,
)


# ---------------------------------------------------------------------------
# repo_search - thin wrapper delegating to _repo_search_impl
# ---------------------------------------------------------------------------
@mcp.tool()
async def repo_search(
    query: Any = None,
    queries: Any = None,
    limit: Any = None,
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
    ctx: Context = None,
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
    repo: Any = None,
    compact: Any = None,
    output_format: Any = None,
    args: Any = None,
    kwargs: Any = None,
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
    - collection: str. Target collection; defaults to workspace state or env COLLECTION_NAME.
    - repo: str or list[str]. Filter by repo name(s). Use "*" to search all repos.

    Returns:
    - Dict with keys: results, total, used_rerank, rerank_counters
    """
    return await _repo_search_impl(
        query=query,
        queries=queries,
        limit=limit,
        per_path=per_path,
        include_snippet=include_snippet,
        context_lines=context_lines,
        rerank_enabled=rerank_enabled,
        rerank_top_n=rerank_top_n,
        rerank_return_m=rerank_return_m,
        rerank_timeout_ms=rerank_timeout_ms,
        highlight_snippet=highlight_snippet,
        collection=collection,
        workspace_path=workspace_path,
        mode=mode,
        session=session,
        ctx=ctx,
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
        repo=repo,
        compact=compact,
        output_format=output_format,
        args=args,
        kwargs=kwargs,
        get_embedding_model_fn=_get_embedding_model,
        require_auth_session_fn=_require_auth_session,
        do_highlight_snippet_fn=_do_highlight_snippet,
        run_async_fn=_run_async,
    )


@mcp.tool()
async def repo_search_compat(**arguments) -> Dict[str, Any]:
    """Compatibility wrapper for repo_search (lenient argument handling).

    When to use:
    - Clients that only send a single dict payload or use aliases (q/text/top_k)
    - Avoids schema errors by normalizing and forwarding to repo_search

    Returns: same shape as repo_search.
    Note: Prefer calling repo_search directly when possible.
    """
    try:
        args = arguments or {}
        # Core query: prefer explicit query, else q/text; allow queries list passthrough
        query = args.get("query") or args.get("q") or args.get("text")
        queries = args.get("queries")
        # top_k alias for limit
        limit = args.get("limit")
        if (
            limit is None or (isinstance(limit, str) and str(limit).strip() == "")
        ) and ("top_k" in args):
            limit = args.get("top_k")
        # not/ not_ normalization
        not_value = args.get("not_") if ("not_" in args) else args.get("not")

        # Build forward kwargs; pass alias keys too so repo_search's leniency picks them up
        forward = {
            "query": query,
            "limit": limit,
            "per_path": args.get("per_path"),
            "include_snippet": args.get("include_snippet"),
            "context_lines": args.get("context_lines"),
            "rerank_enabled": args.get("rerank_enabled"),
            "rerank_top_n": args.get("rerank_top_n"),
            "rerank_return_m": args.get("rerank_return_m"),
            "rerank_timeout_ms": args.get("rerank_timeout_ms"),
            "highlight_snippet": args.get("highlight_snippet"),
            "collection": args.get("collection"),
            "session": args.get("session"),
            "workspace_path": args.get("workspace_path"),
            "language": args.get("language"),
            "under": args.get("under"),
            "kind": args.get("kind"),
            "symbol": args.get("symbol"),
            "path_regex": args.get("path_regex"),
            "path_glob": args.get("path_glob"),
            "not_glob": args.get("not_glob"),
            "ext": args.get("ext"),
            "not_": not_value,
            "case": args.get("case"),
            "compact": args.get("compact"),
            "mode": args.get("mode"),
            "repo": args.get("repo"),  # Cross-codebase isolation
            "output_format": args.get("output_format"),  # "json" or "toon"
            # Alias passthroughs captured by repo_search(**kwargs)
            "queries": queries,
            "q": args.get("q"),
            "text": args.get("text"),
            "top_k": args.get("top_k"),
        }
        # Drop Nones to avoid overriding repo_search defaults unnecessarily
        clean = {k: v for k, v in forward.items() if v is not None}
        return await repo_search(**clean)
    except Exception as e:
        return {"error": f"repo_search_compat failed: {e}"}


@mcp.tool()
async def context_answer_compat(arguments: Any = None) -> Dict[str, Any]:
    """Compatibility wrapper for context_answer (lenient argument handling).

    When to use:
    - Clients that send a single 'arguments' dict or alternate keys (q/text)
    - Avoids schema errors by normalizing/forwarding to context_answer

    Returns: same shape as context_answer.
    Note: Prefer calling context_answer directly when possible.
    """
    try:
        args = arguments or {}
        query = args.get("query") or args.get("q") or args.get("text")
        forward = {
            "query": query,
            "limit": args.get("limit"),
            "per_path": args.get("per_path"),
            "budget_tokens": args.get("budget_tokens"),
            "include_snippet": args.get("include_snippet"),
            "collection": args.get("collection"),
            "max_tokens": args.get("max_tokens"),
            "temperature": args.get("temperature"),
            "mode": args.get("mode"),
            "expand": args.get("expand"),
            # ---- Forward retrieval filters so router hints are honored ----
            "language": args.get("language"),
            "under": args.get("under"),
            "kind": args.get("kind"),
            "symbol": args.get("symbol"),
            "ext": args.get("ext"),
            "path_regex": args.get("path_regex"),
            "path_glob": args.get("path_glob"),
            "not_glob": args.get("not_glob"),
            "case": args.get("case"),
            # pass through NOT filter under either key
            "not_": args.get("not_") or args.get("not"),
        }
        clean = {k: v for k, v in forward.items() if v is not None}
        return await context_answer(**clean)
    except Exception as e:
        return {"error": f"context_answer_compat failed: {e}"}


# ---------------------------------------------------------------------------
# Specialized search tools - thin wrappers delegating to extracted impls
# ---------------------------------------------------------------------------
@mcp.tool()
async def search_tests_for(
    query: Any = None,
    limit: Any = None,
    include_snippet: Any = None,
    context_lines: Any = None,
    under: Any = None,
    language: Any = None,
    session: Any = None,
    compact: Any = None,
    kwargs: Any = None,
    ctx: Context = None,
) -> Dict[str, Any]:
    """Find test files related to a query.

    What it does:
    - Presets common test file globs and forwards to repo_search
    - Accepts extra filters via kwargs (e.g., language, under, case)

    Parameters:
    - query: str or list[str]; limit; include_snippet/context_lines; under; language; compact

    Returns: repo_search result shape.
    """
    return await _search_tests_for_impl(
        query=query,
        limit=limit,
        include_snippet=include_snippet,
        context_lines=context_lines,
        under=under,
        language=language,
        session=session,
        compact=compact,
        kwargs=kwargs,
        ctx=ctx,
        repo_search_fn=repo_search,
    )


@mcp.tool()
async def search_config_for(
    query: Any = None,
    limit: Any = None,
    include_snippet: Any = None,
    context_lines: Any = None,
    under: Any = None,
    session: Any = None,
    compact: Any = None,
    kwargs: Any = None,
    ctx: Context = None,
) -> Dict[str, Any]:
    """Find likely configuration files for a service/query.

    What it does:
    - Presets config file globs (yaml/json/toml/etc.) and forwards to repo_search
    - Accepts extra filters via kwargs

    Returns: repo_search result shape.
    """
    return await _search_config_for_impl(
        query=query,
        limit=limit,
        include_snippet=include_snippet,
        context_lines=context_lines,
        under=under,
        session=session,
        compact=compact,
        kwargs=kwargs,
        ctx=ctx,
        repo_search_fn=repo_search,
    )


@mcp.tool()
async def search_callers_for(
    query: Any = None,
    limit: Any = None,
    language: Any = None,
    session: Any = None,
    kwargs: Any = None,
    ctx: Context = None,
) -> Dict[str, Any]:
    """Heuristic search for callers/usages of a symbol.

    When to use:
    - You want files that reference/invoke a function/class

    Notes:
    - Thin wrapper over repo_search today; pass language or path_glob to narrow
    - Returns repo_search result shape
    """
    return await _search_callers_for_impl(
        query=query,
        limit=limit,
        language=language,
        session=session,
        kwargs=kwargs,
        ctx=ctx,
        repo_search_fn=repo_search,
    )


@mcp.tool()
async def search_importers_for(
    query: Any = None,
    limit: Any = None,
    language: Any = None,
    session: Any = None,
    kwargs: Any = None,
    ctx: Context = None,
) -> Dict[str, Any]:
    """Find files likely importing or referencing a module/symbol.

    What it does:
    - Presets code globs across common languages; forwards to repo_search
    - Accepts additional filters via kwargs (e.g., under, case)

    Returns: repo_search result shape.
    """
    return await _search_importers_for_impl(
        query=query,
        limit=limit,
        language=language,
        session=session,
        kwargs=kwargs,
        ctx=ctx,
        repo_search_fn=repo_search,
    )


@mcp.tool()
async def symbol_graph(
    symbol: str = None,
    query_type: str = "callers",
    limit: Any = None,
    language: Any = None,
    under: Any = None,
    session: Any = None,
    output_format: Any = None,
    ctx: Context = None,
) -> Dict[str, Any]:
    """Query the symbol graph to find callers, definitions, or importers.

    When to use:
    - "Who calls X?" → query_type="callers"
    - "Where is X defined?" → query_type="definition"
    - "What imports Y?" → query_type="importers"

    Key parameters:
    - symbol: str. The function, class, or module name to search for.
    - query_type: str. One of "callers", "definition", "importers".
    - limit: int (default 20). Maximum results to return.
    - language: str (optional). Filter by programming language.
    - under: str (optional). Filter by path prefix.
    - output_format: "json" (default) or "toon" for token-efficient format.

    Returns:
    - {"results": [...], "symbol": str, "query_type": str, "count": int}
    - Each result includes path, start_line, end_line, symbol_path, and relevant context.

    Example:
    - symbol_graph(symbol="get_embedding_model", query_type="callers")
    - symbol_graph(symbol="ASTAnalyzer", query_type="definition")
    - symbol_graph(symbol="qdrant_client", query_type="importers")
    """
    if not symbol or not str(symbol).strip():
        return {"error": "symbol parameter is required", "results": []}

    _limit = safe_int(limit, default=20, logger=logger, context="symbol_graph.limit")

    result = await _symbol_graph_impl(
        symbol=str(symbol).strip(),
        query_type=query_type or "callers",
        limit=_limit,
        language=str(language).strip() if language else None,
        under=str(under).strip() if under else None,
        session=str(session).strip() if session else None,
        ctx=ctx,
    )

    # Format output
    use_toon = _should_use_toon(output_format)
    if use_toon:
        return {"text": _format_symbol_graph_toon(result), **result}

    return result


@mcp.tool()
async def search_commits_for(
    query: Any = None,
    path: Any = None,
    collection: Any = None,
    limit: Any = None,
    max_points: Any = None,
) -> Dict[str, Any]:
    """Search git commit history indexed in Qdrant.

    What it does:
    - Queries commit documents ingested by scripts/ingest_history.py
    - Filters by optional file path (metadata.files contains path)

    Parameters:
    - query: str or list[str]; matched lexically against commit message/text
    - path: str (optional). Relative path under /work; filters commits that touched this file
    - collection: str (optional). Defaults to env/WS collection
    - limit: int (optional, default 10). Max commits to return
    - max_points: int (optional). Safety cap on scanned points (default 1000)

    Returns:
    - {"ok": true, "results": [{"commit_id", "author_name", "authored_date", "message", "files"}, ...], "scanned": int}
    - On error: {"ok": false, "error": "..."}
    """
    return await _search_commits_for_impl(
        query=query,
        path=path,
        collection=collection,
        limit=limit,
        max_points=max_points,
        default_collection_fn=_default_collection,
        get_embedding_model_fn=_get_embedding_model,
    )


@mcp.tool()
async def change_history_for_path(
    path: Any,
    collection: Any = None,
    max_points: Any = None,
    include_commits: Any = None,
) -> Dict[str, Any]:
    """Summarize recent change metadata for a file path from the index.

    Parameters:
    - path: str. Relative path under /work.
    - collection: str (optional). Defaults to env/WS default.
    - max_points: int (optional). Safety cap on scanned points.
    - include_commits: bool (optional). If true, attach a small list of recent commits
      touching this path based on the commit index.

    Returns:
    - {"ok": true, "summary": {...}} or {"ok": false, "error": "..."}.
    """
    return await _change_history_for_path_impl(
        path=path,
        collection=collection,
        max_points=max_points,
        include_commits=include_commits,
        default_collection_fn=_default_collection,
        search_commits_fn=search_commits_for,
    )

# --- context_answer helpers imported from mcp_context_answer shim ---
from scripts.mcp_context_answer import (
    _cleanup_answer,
    _answer_style_guidance,
    _strip_preamble_labels,
    _validate_answer_output,
    _ca_unwrap_and_normalize,
    _ca_prepare_filters_and_retrieve,
    _ca_fallback_and_budget,
    _ca_build_citations_and_context,
    _ca_ident_supplement,
    _ca_decoder_params,
    _ca_build_prompt,
    _ca_decode,
    _ca_postprocess_answer,
    _synthesize_from_citations,
    _context_answer_impl,
)


@mcp.tool()
async def context_answer(
    query: Any = None,
    limit: Any = None,
    per_path: Any = None,
    budget_tokens: Any = None,
    include_snippet: Any = None,
    collection: Any = None,
    max_tokens: Any = None,
    temperature: Any = None,
    mode: Any = None,  # "stitch" (default) or "pack"
    expand: Any = None,  # whether to LLM-expand queries (up to 2 alternates)
    # Retrieval filter parameters (passed through to hybrid_search)
    language: Any = None,
    under: Any = None,
    kind: Any = None,
    symbol: Any = None,
    ext: Any = None,
    path_regex: Any = None,
    path_glob: Any = None,
    not_glob: Any = None,
    case: Any = None,
    not_: Any = None,
    # Repo scoping (cross-codebase isolation)
    repo: Any = None,  # str, list[str], or "*" to search all repos
    kwargs: Any = None,
) -> Dict[str, Any]:
    """Natural-language Q&A over the repo using retrieval + local LLM (llama.cpp).

    What it does:
    - Retrieves relevant code (hybrid vector+lexical with reranking enabled by default).
    - Budgets/merges micro-spans, builds citations, and asks the LLM to answer.
    - Returns a concise answer plus file/line citations.

    When to use:
    - You need an explanation or "how to" grounded in code.
    - Prefer repo_search for raw hits; prefer context_search to blend code + memory.

    Key parameters:
    - query: str or list[str]; may be expanded if expand=true.
    - budget_tokens: int. Token budget across code spans (defaults from MICRO_BUDGET_TOKENS).
    - include_snippet: bool (default true). Include code snippets sent to the LLM and return them when requested.
    - max_tokens, temperature: decoding controls.
    - mode: "stitch" (default) or "pack" for prompt assembly.
    - expand: bool. Use tiny local LLM to propose up to 2 alternate queries.
    - Filters: language, under, kind, symbol, ext, path_regex, path_glob, not_glob, not_, case.
    - repo: str or list[str]. Filter by repo name(s). Use "*" to search all repos (disable auto-filter).
      By default, auto-detects current repo from CURRENT_REPO env and filters to it.

    Returns:
    - {"answer": str, "citations": [{"path": str, "start_line": int, "end_line": int}], "query": list[str], "used": {...}}
    - On decoder disabled/error, returns {"error": "...", "citations": [...], "query": [...]}

    Notes:
    - Reranking is enabled by default for optimal retrieval quality.
    - Honors env knobs such as REFRAG_MODE, REFRAG_GATE_FIRST, MICRO_BUDGET_TOKENS, DECODER_*.
    - Keeps answers brief (2–4 sentences) and grounded; rejects ungrounded output.
    """
    return await _context_answer_impl(
        query=query,
        limit=limit,
        per_path=per_path,
        budget_tokens=budget_tokens,
        include_snippet=include_snippet,
        collection=collection,
        max_tokens=max_tokens,
        temperature=temperature,
        mode=mode,
        expand=expand,
        language=language,
        under=under,
        kind=kind,
        symbol=symbol,
        ext=ext,
        path_regex=path_regex,
        path_glob=path_glob,
        not_glob=not_glob,
        case=case,
        not_=not_,
        repo=repo,
        kwargs=kwargs,
        get_embedding_model_fn=_get_embedding_model,
        expand_query_fn=expand_query,
        env_lock=_ENV_LOCK,
        prepare_filters_and_retrieve_fn=_ca_prepare_filters_and_retrieve,
    )
 
@mcp.tool()
async def code_search(
    query: Any = None,
    limit: Any = None,
    per_path: Any = None,
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
    kwargs: Any = None,
) -> Dict[str, Any]:
    """Exact alias of repo_search (hybrid code search with reranking enabled by default).

    Prefer repo_search; this name exists for discoverability in some IDEs/agents.
    Same parameters and return shape as repo_search.
    Reranking (rerank_enabled=true) is ON by default for optimal result quality.
    """
    return await repo_search(
        query=query,
        limit=limit,
        per_path=per_path,
        include_snippet=include_snippet,
        context_lines=context_lines,
        rerank_enabled=rerank_enabled,
        rerank_top_n=rerank_top_n,
        rerank_return_m=rerank_return_m,
        rerank_timeout_ms=rerank_timeout_ms,
        highlight_snippet=highlight_snippet,
        collection=collection,
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
        session=session,
        compact=compact,
        kwargs=kwargs,
    )


# ---------------------------------------------------------------------------
# info_request: Simplified codebase retrieval with explanation mode
# (helpers imported from scripts.mcp_impl.info_request)
# ---------------------------------------------------------------------------
@mcp.tool()
async def info_request(
    # Primary parameter
    info_request: str = None,
    information_request: str = None,  # Alias
    # Explanation mode
    include_explanation: bool = None,
    # Relationship mapping
    include_relationships: bool = None,
    # Auth/session (passed through to repo_search)
    session: str = None,
    # Optional filters (pass-through to repo_search)
    limit: int = None,
    language: str = None,
    under: str = None,
    repo: Any = None,
    path_glob: Any = None,
    # Additional options
    include_snippet: bool = None,
    context_lines: int = None,
    # Output format
    output_format: Any = None,  # "json" (default) or "toon" for token-efficient format
    kwargs: Any = None,
) -> Dict[str, Any]:
    """Simplified codebase retrieval with optional explanation mode.

    When to use:
    - Simple, single-parameter code search with human-readable descriptions
    - When you want optional explanation mode for richer context
    - Drop-in replacement for basic codebase retrieval tools

    Key parameters:
    - info_request: str. Natural language description of the code you're looking for.
    - information_request: str. Alias for info_request.
    - include_explanation: bool (default false). Add summary, primary_locations, related_concepts.
    - include_relationships: bool (default false). Add imports_from, calls, related_paths to results.
    - limit: int (default 10). Maximum results to return.
    - language: str. Filter by programming language.
    - under: str. Limit search to specific directory.
    - repo: str or list[str]. Filter by repository name(s).
    - output_format: "json" (default) or "toon" for token-efficient TOON format.

    Returns:
    - Compact mode (default): results with information field and relevance_score alias
    - Explanation mode: adds summary, primary_locations, related_concepts, query_understanding

    Example:
    - {"info_request": "database connection pooling"}
    - {"info_request": "authentication middleware", "include_explanation": true}
    """
    # Resolve query from either parameter
    query = info_request or information_request
    if not query or not str(query).strip():
        return {"ok": False, "error": "info_request parameter is required", "results": []}
    query = str(query).strip()

    # Resolve defaults from env
    _default_limit = safe_int(
        os.environ.get("INFO_REQUEST_LIMIT", "10"), default=10, logger=logger
    )
    _default_context = safe_int(
        os.environ.get("INFO_REQUEST_CONTEXT_LINES", "5"), default=5, logger=logger
    )
    _default_explain = str(
        os.environ.get("INFO_REQUEST_EXPLAIN_DEFAULT", "0")
    ).strip().lower() in {"1", "true", "yes", "on"}
    _default_relationships = str(
        os.environ.get("INFO_REQUEST_RELATIONSHIPS", "0")
    ).strip().lower() in {"1", "true", "yes", "on"}

    # Apply defaults
    eff_limit = limit if limit is not None else _default_limit
    eff_context = context_lines if context_lines is not None else _default_context
    eff_snippet = include_snippet if include_snippet is not None else True
    eff_explain = include_explanation if include_explanation is not None else _default_explain
    eff_relationships = include_relationships if include_relationships is not None else _default_relationships

    # Smart limits based on query characteristics (only if user didn't override)
    if limit is None:
        query_words = len(query.split())
        query_lower = query.lower()
        if query_words <= 2:  # Short query like "auth handler"
            eff_limit = 15  # More results for broad queries
        elif "how does" in query_lower or "what is" in query_lower:
            eff_limit = 8   # Questions need focused results

    # Call repo_search (always JSON - we format TOON ourselves after enhancement)
    search_result = await repo_search(
        query=query,
        limit=eff_limit,
        per_path=3,  # Better default for info requests
        session=session,
        include_snippet=eff_snippet,
        context_lines=eff_context,
        language=language,
        under=under,
        repo=repo,
        path_glob=path_glob,
        output_format="json",  # Always get JSON to iterate results
        kwargs=kwargs,
    )

    # Extract results
    results = search_result.get("results", [])
    total = search_result.get("total", len(results))
    used_rerank = search_result.get("used_rerank", False)

    # Enhance each result with information field and optional relationships
    enhanced_results = []
    for r in results:
        enhanced = dict(r)
        enhanced["information"] = _format_information_field(r)
        enhanced["relevance_score"] = r.get("score", 0.0)  # Alias
        # Add relationships if requested
        if eff_relationships:
            enhanced["relationships"] = _extract_relationships(r)
        enhanced_results.append(enhanced)

    # Build better search strategy string
    strategy_parts = ["hybrid"]
    if used_rerank:
        strategy_parts.append("rerank")
    if repo:
        strategy_parts.append("repo_filtered")
    if language:
        strategy_parts.append(f"lang:{language}")
    if under:
        strategy_parts.append("path_filtered")
    search_strategy = "+".join(strategy_parts)

    # Build response
    response: Dict[str, Any] = {
        "ok": True,
        "results": enhanced_results,
        "total": total,
        "search_strategy": search_strategy,
    }

    # Add explanation if requested
    if eff_explain:
        # Primary locations: unique file paths
        seen_paths = set()
        primary_locations = []
        for r in results:
            p = r.get("path", "")
            if p and p not in seen_paths:
                seen_paths.add(p)
                primary_locations.append(p)
                if len(primary_locations) >= 5:
                    break

        # Related concepts
        related_concepts = _extract_related_concepts(query, results)

        # Detected symbols from query
        detected_symbols = _extract_symbols_from_query(query)

        # Summary
        n_files = len(seen_paths)
        summary = f"Found {total} results related to '{query}' across {n_files} file{'s' if n_files != 1 else ''}"

        # Group results by file
        files_map: Dict[str, list] = {}
        for r in enhanced_results:
            p = r.get("path", "")
            if p not in files_map:
                files_map[p] = []
            files_map[p].append({
                "symbol": r.get("symbol", ""),
                "line": r.get("start_line", 0),
                "score": r.get("score", 0.0),
            })

        grouped_results = {
            "by_file": {
                path: {
                    "count": len(items),
                    "top_symbols": [i["symbol"] for i in sorted(items, key=lambda x: -x["score"])[:3] if i["symbol"]],
                }
                for path, items in files_map.items()
            }
        }

        # Calculate confidence
        confidence = _calculate_confidence(query, enhanced_results)

        response["summary"] = summary
        response["primary_locations"] = primary_locations
        response["related_concepts"] = related_concepts
        response["grouped_results"] = grouped_results
        response["confidence"] = confidence
        response["query_understanding"] = {
            "intent": "search_for_code",
            "detected_language": language or None,
            "detected_symbols": detected_symbols,
            "search_strategy": search_strategy,
        }

    # Apply TOON formatting if requested or enabled globally
    if _should_use_toon(output_format):
        return _format_results_as_toon(response, compact=False)  # Keep info_request fields
    return response


# ---------------------------------------------------------------------------
# context_search - thin wrapper delegating to _context_search_impl
# ---------------------------------------------------------------------------
@mcp.tool()
async def context_search(
    query: Any = None,
    limit: Any = None,
    per_path: Any = None,
    include_memories: Any = None,
    memory_weight: Any = None,
    per_source_limits: Any = None,
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
    repo: Any = None,
    output_format: Any = None,
    kwargs: Any = None,
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
    return await _context_search_impl(
        query=query,
        limit=limit,
        per_path=per_path,
        include_memories=include_memories,
        memory_weight=memory_weight,
        per_source_limits=per_source_limits,
        include_snippet=include_snippet,
        context_lines=context_lines,
        rerank_enabled=rerank_enabled,
        rerank_top_n=rerank_top_n,
        rerank_return_m=rerank_return_m,
        rerank_timeout_ms=rerank_timeout_ms,
        highlight_snippet=highlight_snippet,
        collection=collection,
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
        session=session,
        compact=compact,
        repo=repo,
        output_format=output_format,
        kwargs=kwargs,
        repo_search_fn=repo_search,
        get_embedding_model_fn=_get_embedding_model,
    )


# ---------------------------------------------------------------------------
# expand_query - thin wrapper delegating to _expand_query_impl
# ---------------------------------------------------------------------------
@mcp.tool()
async def expand_query(
    query: Any = None,
    max_new: Any = None,
    session: Optional[str] = None,
) -> Dict[str, Any]:
    """LLM-assisted query expansion (local llama.cpp, if enabled).

    When to use:
    - Generate 1–2 compact alternates before repo_search/context_answer

    Parameters:
    - query: str or list[str]
    - max_new: int in [0,5] (default 3)

    Returns:
    - {"alternates": list[str]} or {"alternates": [], "hint": "..."} if decoder disabled
    """
    return await _expand_query_impl(query=query, max_new=max_new, session=session)


# ---------------------------------------------------------------------------
# Pattern Search - Structural code similarity (conditional on PATTERN_VECTORS=1)
# ---------------------------------------------------------------------------
_PATTERN_SEARCH_ENABLED = str(os.environ.get("PATTERN_VECTORS", "")).strip().lower() in {
    "1", "true", "yes", "on"
}

if _PATTERN_SEARCH_ENABLED:
    @mcp.tool()
    async def pattern_search(
        query: Any = None,
        language: Any = None,
        limit: Any = None,
        min_score: Any = None,
        include_snippet: Any = None,
        context_lines: Any = None,
        target_languages: Any = None,
        output_format: Any = None,
        compact: Any = None,
        aroma_rerank: Any = None,
        aroma_alpha: Any = None,
        query_mode: Any = None,
    ) -> Dict[str, Any]:
        """Find structurally similar code patterns across all languages.

        Accepts EITHER code examples OR natural language descriptions - auto-detects which.

        When to use:
        - Find code with similar control flow (retry loops, error handling, etc.)
        - Cross-language pattern matching (Python pattern → Go/Rust/Java matches)
        - Detect code duplication based on structure, not syntax
        - Search by pattern description ("retry with backoff", "resource cleanup")

        Key parameters:
        - query: str. Code snippet OR natural language description of pattern.
        - query_mode: str. "code", "description", or "auto" (default). Explicit override for detection.
        - language: str. Language hint for code examples (also triggers code mode in auto).
        - limit: int (default 10). Maximum results to return.
        - min_score: float (default 0.3). Minimum similarity score threshold.
        - include_snippet: bool (default false). Include code snippets in results.
        - target_languages: list[str]. Filter to specific target languages.
        - output_format: "json" (default) or "toon" for token-efficient format.
        - compact: bool. If true with TOON, use minimal fields.
        - aroma_rerank: bool (default true). Enable AROMA-style pruning and reranking.
        - aroma_alpha: float (default 0.6). Weight for pruned similarity vs original score.

        Returns:
        - {ok, results: [{path, start_line, end_line, score, language, ...}], total, query_signature}

        Examples:
        - pattern_search(query="for i in range(3): try: ... except: time.sleep(2**i)")
        - pattern_search(query="retry with exponential backoff", query_mode="description")
        - pattern_search(query="if err != nil { return err }", language="go")
        """
        return await _pattern_search_impl(
            query=query,
            language=language,
            limit=limit,
            min_score=min_score,
            include_snippet=include_snippet,
            context_lines=context_lines,
            hybrid=None,
            semantic_weight=None,
            collection=None,
            target_languages=target_languages,
            output_format=output_format,
            compact=compact,
            aroma_rerank=aroma_rerank,
            aroma_alpha=aroma_alpha,
            query_mode=query_mode,
            coerce_bool_fn=_coerce_bool,
            coerce_int_fn=_coerce_int,
            coerce_float_fn=lambda v, d: safe_float(v, default=d, logger=logger, context="pattern_search"),
        )


_relax_var_kwarg_defaults()

if __name__ == "__main__":
    # Configure log level from environment
    import logging as _logging
    _log_level_str = os.environ.get("LOG_LEVEL", "INFO").upper()
    _log_level = getattr(_logging, _log_level_str, _logging.INFO)
    _logging.getLogger().setLevel(_log_level)
    
    # Startup logging with configuration info
    logger.info("=" * 60)
    logger.info("MCP Indexer Server starting...")
    logger.info("=" * 60)
    logger.info(f"  Host: {HOST}")
    logger.info(f"  Port: {PORT}")
    logger.info(f"  Log Level: {_log_level_str}")
    logger.info(f"  Qdrant URL: {os.environ.get('QDRANT_URL', 'not set')}")
    logger.info(f"  Collection: {os.environ.get('COLLECTION_NAME', 'codebase')}")
    logger.info(f"  Transport: {os.environ.get('FASTMCP_TRANSPORT', 'sse')}")
    logger.info(f"  Embedding Model: {os.environ.get('EMBEDDING_MODEL', 'BAAI/bge-base-en-v1.5')}")
    logger.info(f"  Embedding Provider: {os.environ.get('EMBEDDING_PROVIDER', 'fastembed')}")
    logger.info(f"  ReFRAG Decoder: {os.environ.get('REFRAG_DECODER', '1')}")
    logger.info(f"  Rerank Learning: {os.environ.get('RERANK_LEARNING', '1')}")
    logger.info(f"  Semantic Chunks: {os.environ.get('INDEX_SEMANTIC_CHUNKS', '1')}")
    logger.info(f"  Micro Chunks: {os.environ.get('INDEX_MICRO_CHUNKS', '1')}")
    logger.info(f"  Micro Chunk Tokens: {os.environ.get('MICRO_CHUNK_TOKENS', '128')}")
    logger.info(f"  Micro Chunk Stride: {os.environ.get('MICRO_CHUNK_STRIDE', '64')}")
    logger.info(f"  Max Micro Chunks/File: {os.environ.get('MAX_MICRO_CHUNKS_PER_FILE', '200')}")
    logger.info(f"  ReFRAG Mode: {os.environ.get('REFRAG_MODE', '0')}")
    logger.info(f"  ReFRAG Gate First: {os.environ.get('REFRAG_GATE_FIRST', '0')}")
    logger.info(f"  Lexical Vector Dim: {os.environ.get('LEX_VECTOR_DIM', '4096')}")
    logger.info(f"  Lexical Multi Hash: {os.environ.get('LEX_MULTI_HASH', '1')}")
    logger.info(f"  Lexical Bigrams: {os.environ.get('LEX_BIGRAMS', '0')}")
    logger.info(f"  Lexical Bigram Weight: {os.environ.get('LEX_BIGRAM_WEIGHT', '0.7')}")
    logger.info(f"  Lexical Sparse Mode: {os.environ.get('LEX_SPARSE_MODE', '0')}")
    logger.info(f"  Reranker Enabled: {os.environ.get('RERANKER_ENABLED', '0')}")
    logger.info(f"  Rerank Top N: {os.environ.get('RERANK_TOP_N', '20')}")
    logger.info(f"  Rerank Timeout MS: {os.environ.get('RERANK_TIMEOUT_MS', '500')}")
    logger.info(f"  Pattern Search: {'enabled' if _PATTERN_SEARCH_ENABLED else 'disabled (set PATTERN_VECTORS=1)'}")
    logger.info("=" * 60)

    # Optional warmups: gated by env flags to avoid delaying readiness on fresh containers
    try:
        if str(os.environ.get("EMBEDDING_WARMUP", "")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            _ = _get_embedding_model(
                os.environ.get("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
            )
    except Exception:
        pass
    try:
        if str(os.environ.get("RERANK_WARMUP", "")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        } and str(os.environ.get("RERANKER_ENABLED", "")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            if str(os.environ.get("RERANK_IN_PROCESS", "")).strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }:
                try:
                    from scripts.rerank_local import _get_rerank_session  # type: ignore

                    _ = _get_rerank_session()
                except Exception:
                    pass
            else:
                # Fire a tiny warmup rerank once via subprocess; ignore failures
                _env = os.environ.copy()
                _env["QDRANT_URL"] = QDRANT_URL
                _env["COLLECTION_NAME"] = _default_collection()
                _cmd = [
                    "python",
                    "/work/scripts/rerank_local.py",
                    "--query",
                    "warmup",
                    "--topk",
                    "3",
                    "--limit",
                    "1",
                ]
                subprocess.run(
                    _cmd, capture_output=True, text=True, env=_env, timeout=10
                )
    except Exception:
        pass

    # Start lightweight /readyz health endpoint in background (best-effort)
    try:
        _start_readyz_server()
    except Exception:
        pass

    transport = os.environ.get("FASTMCP_TRANSPORT", "sse").strip().lower()
    if transport == "stdio":
        # Run over stdio (for clients that don't support network transports)
        mcp.run(transport="stdio")
    elif transport in {"http", "streamable", "streamable_http", "streamable-http"}:
        # Streamable HTTP (recommended) — endpoint at /mcp (FastMCP default)
        try:
            mcp.settings.host = HOST
            mcp.settings.port = PORT
        except Exception:
            pass
        # Use the correct FastMCP transport name
        try:
            mcp.run(transport="streamable-http")
        except Exception:
            # Fallback to SSE only if HTTP truly unavailable
            mcp.settings.host = HOST
            mcp.settings.port = PORT
            mcp.run(transport="sse")
    else:
        # SSE (legacy) — endpoint at /sse
        mcp.settings.host = HOST
        mcp.settings.port = PORT
        mcp.run(transport="sse")
