"""
mcp_router/client.py - HTTP/MCP client helpers.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Tuple
from urllib import request

from .config import (
    HTTP_URL_INDEXER,
    HTTP_URL_MEMORY,
    HEALTH_PORT_INDEXER,
    HEALTH_PORT_MEMORY,
    cache_ttl_sec,
)

# Caches
_TOOL_ENDPOINTS_CACHE_MAP: Dict[str, str] = {}
_TOOL_ENDPOINTS_CACHE_TS: float = 0.0
_TOOLS_DESCR_CACHE: Dict[str, list] = {}
_TOOLS_DESCR_TS: Dict[str, float] = {}


def _post_raw(url: str, payload: Dict[str, Any], headers: Dict[str, str], timeout: float = 60.0) -> Tuple[Dict[str, str], bytes]:
    req = request.Request(url, method="POST")
    for k, v in headers.items():
        req.add_header(k, v)
    data = json.dumps(payload).encode("utf-8")
    with request.urlopen(req, data=data, timeout=timeout) as resp:
        body = resp.read()
        hdrs = {k.lower(): v for k, v in resp.headers.items()}
    return hdrs, body


def _post_raw_retry(url: str, payload: Dict[str, Any], headers: Dict[str, str],
                    timeout: float = 60.0, retries: int = 2, backoff: float = 0.5) -> Tuple[Dict[str, str], bytes]:
    last_exc: Exception | None = None
    for i in range(max(0, retries) + 1):
        try:
            return _post_raw(url, payload, headers, timeout=timeout)
        except Exception as e:
            last_exc = e
            if i < retries:
                try:
                    time.sleep(backoff * (2 ** i))
                except Exception:
                    pass
            else:
                raise last_exc


def _parse_stream_or_json(body: bytes) -> Dict[str, Any]:
    txt = body.decode("utf-8", errors="ignore")
    if "data:" in txt and ("event:" in txt or txt.strip().startswith("data:")):
        last = None
        for line in txt.splitlines():
            if line.startswith("data:"):
                last = line[len("data:"):].strip()
        if last:
            try:
                return json.loads(last)
            except Exception:
                pass
    return json.loads(txt)


def _filter_args(d: Dict[str, Any]) -> Dict[str, Any]:
    """Remove None/empty values from args dict."""
    return {k: v for k, v in d.items() if v not in (None, "")}


def _mcp_handshake(base_url: str, timeout: float = 30.0) -> Dict[str, str]:
    """Perform MCP handshake and return headers with session ID."""
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    init_payload = {
        "jsonrpc": "2.0",
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "router", "version": "0.1.0"},
        },
        "id": 1,
    }
    hdrs, body = _post_raw_retry(base_url, init_payload, headers, timeout=timeout)
    sid = hdrs.get("mcp-session-id") or hdrs.get("Mcp-Session-Id")
    if not sid:
        try:
            j = _parse_stream_or_json(body)
            sid = j.get("sessionId")
        except Exception:
            sid = None
    if sid:
        headers["Mcp-Session-Id"] = sid
    try:
        _post_raw_retry(base_url, {"jsonrpc": "2.0", "method": "notifications/initialized"}, headers, timeout=timeout)
    except Exception:
        pass
    return headers


def _extract_iserror_text(resp: Dict[str, Any]) -> str | None:
    try:
        r = resp.get("result") or {}
        if isinstance(r, dict) and r.get("isError"):
            content = r.get("content")
            if isinstance(content, list) and content and isinstance(content[0], dict):
                if content[0].get("type") == "text":
                    return content[0].get("text")
    except Exception:
        pass
    return None


def call_tool_http(base_url: str, tool_name: str, args: Dict[str, Any], timeout: float = 120.0) -> Dict[str, Any]:
    """Call an MCP tool over HTTP."""
    headers = _mcp_handshake(base_url, timeout=min(timeout, 30.0))

    def _do_call(arguments: Dict[str, Any]) -> Dict[str, Any]:
        payload = {
            "jsonrpc": "2.0",
            "id": "router-1",
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments,
            },
        }
        _, body = _post_raw_retry(base_url, payload, headers, timeout=timeout)
        return _parse_stream_or_json(body)

    args1 = _filter_args(args)
    if tool_name.endswith("_compat"):
        resp = _do_call({"arguments": args1})
    else:
        resp = _do_call(args1)

    def _get_structured_error(r: Dict[str, Any]) -> str | None:
        try:
            rr = r.get("result") or {}
            sc = rr.get("structuredContent") or {}
            rs = sc.get("result") or {}
            err = rs.get("error")
            if isinstance(err, str):
                return err
        except Exception:
            pass
        return None

    msg = _extract_iserror_text(resp)
    serr = _get_structured_error(resp)
    if msg:
        low = msg.lower()
        if ("kwargs" in low) and ("field required" in low or "missing" in low):
            return _do_call({"kwargs": args1})
        if ("arguments" in low) and ("field required" in low or "missing" in low):
            return _do_call({"arguments": args1})
    if (serr and serr.strip().lower() == "query required") and ("query" in args1 or "queries" in args1):
        resp4 = _do_call({"kwargs": args1})
        serr2 = _get_structured_error(resp4)
        if not (serr2 and serr2.strip().lower() == "query required"):
            return resp4
        resp5 = _do_call({"arguments": {"kwargs": args1}})
        serr3 = _get_structured_error(resp5)
        if not (serr3 and serr3.strip().lower() == "query required"):
            return resp5
        return _do_call({"arguments": args1})
    return resp


def is_failure_response(resp: Dict[str, Any]) -> bool:
    """Check if response indicates a failure."""
    try:
        r = resp.get("result") or {}
        if r.get("isError") is True:
            return True
        sc = r.get("structuredContent") or {}
        rs = sc.get("result") or {}
        if isinstance(rs, dict) and isinstance(rs.get("error"), str):
            return True
    except Exception:
        return False
    return False


def _tools_describe_from_health(base_url: str, timeout: float = 3.0) -> list[dict]:
    """Fetch tool descriptors from health /tools endpoint."""
    try:
        import urllib.request
        if base_url == HTTP_URL_INDEXER:
            url = f"http://localhost:{HEALTH_PORT_INDEXER}/tools"
        elif base_url == HTTP_URL_MEMORY:
            url = f"http://localhost:{HEALTH_PORT_MEMORY}/tools"
        else:
            return []
        with urllib.request.urlopen(url, timeout=timeout) as r:
            if getattr(r, "status", 200) != 200:
                return []
            body = r.read()
            j = _parse_stream_or_json(body)
            tools = (j.get("tools") if isinstance(j, dict) else None) or []
            out = []
            for t in tools:
                if not isinstance(t, dict):
                    continue
                nm = t.get("name")
                if not nm:
                    continue
                out.append({"name": nm, "description": (t.get("description") or "").strip()})
            return out
    except Exception:
        return []


def _mcp_tools_list(base_url: str, timeout: float = 30.0) -> List[str]:
    """Get list of tool names from MCP server."""
    try:
        headers = _mcp_handshake(base_url, timeout=min(timeout, 15.0))
        payload = {"jsonrpc": "2.0", "id": "router-list", "method": "tools/list"}
        _, body = _post_raw_retry(base_url, payload, headers, timeout=timeout)
        j = _parse_stream_or_json(body)
        tools = ((j.get("result") or {}).get("tools") or [])
        names: List[str] = []
        for t in tools:
            try:
                n = t.get("name") if isinstance(t, dict) else None
                if isinstance(n, str) and n:
                    names.append(n)
            except Exception:
                continue
        return names
    except Exception:
        return []


def _mcp_tools_describe(base_url: str, timeout: float = 20.0) -> list[dict]:
    """Return tool dicts from tools/list."""
    try:
        headers = _mcp_handshake(base_url, timeout=min(timeout, 10.0))
        payload = {"jsonrpc": "2.0", "id": "router-list2", "method": "tools/list"}
        _, body = _post_raw_retry(base_url, payload, headers, timeout=timeout)
        j = _parse_stream_or_json(body)
        tools = ((j.get("result") or {}).get("tools") or [])
        out = []
        for t in tools:
            if not isinstance(t, dict):
                continue
            name = (t.get("name") or "").strip()
            if not name:
                continue
            out.append(t)
        return out
    except Exception:
        return []


def tools_describe_cached(base_url: str, allow_network: bool = True, timeout: float = 20.0) -> list[dict]:
    """Get tool descriptions with caching."""
    now = time.time()
    ts = _TOOLS_DESCR_TS.get(base_url, 0.0)
    if base_url in _TOOLS_DESCR_CACHE and (now - ts) <= cache_ttl_sec():
        return _TOOLS_DESCR_CACHE[base_url]
    if not allow_network:
        return _TOOLS_DESCR_CACHE.get(base_url, [])
    desc = _tools_describe_from_health(base_url, timeout=min(timeout, 3.0)) or _mcp_tools_describe(base_url, timeout=timeout)
    _TOOLS_DESCR_CACHE[base_url] = desc
    _TOOLS_DESCR_TS[base_url] = now
    return desc


def default_tool_endpoints() -> Dict[str, str]:
    """Return default tool -> endpoint mapping."""
    idx = HTTP_URL_INDEXER
    mem = HTTP_URL_MEMORY
    mapping: Dict[str, str] = {}
    for n in [
        "repo_search", "context_answer", "context_answer_compat", "expand_query",
        "search_tests_for", "search_config_for", "search_callers_for", "search_importers_for",
        "qdrant_index_root", "qdrant_prune", "qdrant_status", "qdrant_list",
        "workspace_info", "list_workspaces", "change_history_for_path", "code_search", "context_search",
    ]:
        mapping[n] = idx
    mapping["store"] = mem
    mapping["find"] = mem
    return mapping


def discover_tool_endpoints(force: bool = False, allow_network: bool = True) -> Dict[str, str]:
    """Discover tool -> endpoint mapping from servers."""
    global _TOOL_ENDPOINTS_CACHE_TS, _TOOL_ENDPOINTS_CACHE_MAP
    now = time.time()
    ttl = cache_ttl_sec()
    if not force and _TOOL_ENDPOINTS_CACHE_MAP and (now - _TOOL_ENDPOINTS_CACHE_TS) <= ttl:
        return _TOOL_ENDPOINTS_CACHE_MAP
    if not allow_network:
        return _TOOL_ENDPOINTS_CACHE_MAP or default_tool_endpoints()
    mapping: Dict[str, str] = {}
    idx_desc = tools_describe_cached(HTTP_URL_INDEXER, allow_network=allow_network)
    for t in idx_desc:
        n = t.get("name") if isinstance(t, dict) else None
        if n:
            mapping[n] = HTTP_URL_INDEXER
    mem_desc = tools_describe_cached(HTTP_URL_MEMORY, allow_network=allow_network)
    for t in mem_desc:
        n = t.get("name") if isinstance(t, dict) else None
        if n and n not in mapping:
            mapping[n] = HTTP_URL_MEMORY
    if mapping:
        _TOOL_ENDPOINTS_CACHE_MAP.clear()
        _TOOL_ENDPOINTS_CACHE_MAP.update(mapping)
        _TOOL_ENDPOINTS_CACHE_TS = now
    return mapping or (_TOOL_ENDPOINTS_CACHE_MAP or default_tool_endpoints())
