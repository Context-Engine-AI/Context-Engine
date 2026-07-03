"""
mcp_router/batching.py - Context answer batching client.
"""
from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
from typing import Any, Dict

from .config import HTTP_URL_INDEXER
from .client import call_tool_http


class BatchingContextAnswerClient:
    """Lightweight in-memory batching for context_answer calls.

    - Queues short-lived requests keyed by (base_url, collection, filters_fingerprint)
    - Flushes after a small window or when batch size cap is hit
    - For multi-item batches, sends query=[...] with mode="pack"
    - Shares the same response with all enqueued requests
    """

    def __init__(self, call_func=None, enable: bool | None = None, window_ms: int | None = None,
                 max_batch: int | None = None, budget_ms: int | None = None):
        self._call = call_func or call_tool_http
        if enable is None:
            env_enabled = os.environ.get("ROUTER_BATCH_ENABLED")
            if env_enabled is None:
                env_enabled = os.environ.get("ROUTER_BATCH_ENABLE", "0")
            self.enabled = str(env_enabled).strip().lower() in {"1", "true", "yes", "on"}
        else:
            self.enabled = bool(enable)
        self.window_ms = int(os.environ.get("ROUTER_BATCH_WINDOW_MS", str(window_ms if window_ms is not None else 100)) or 100)
        env_max = os.environ.get("ROUTER_BATCH_MAX_SIZE")
        if env_max is None:
            env_max = os.environ.get("ROUTER_BATCH_MAX")
        self.max_batch = int(env_max or (max_batch if max_batch is not None else 8))
        env_budget = os.environ.get("ROUTER_BATCH_LATENCY_BUDGET_MS")
        if env_budget is None:
            env_budget = os.environ.get("ROUTER_BATCH_BUDGET_MS")
        self.budget_ms = int(env_budget or (budget_ms if budget_ms is not None else 2000))
        self._lock = threading.RLock()
        self._groups: dict[str, dict[str, Any]] = {}

    def _should_bypass(self, args: Dict[str, Any]) -> bool:
        try:
            if isinstance(args, dict):
                v = args.get("immediate")
                if v is not None and str(v).strip().lower() in {"1", "true", "yes", "on"}:
                    return True
        except Exception:
            pass
        if str(os.environ.get("ROUTER_BATCH_BYPASS", "0")).strip().lower() in {"1", "true", "yes", "on"}:
            return True
        try:
            q = str((args or {}).get("query") or "")
            if "immediate answer" in q.lower():
                return True
        except Exception:
            pass
        return False

    def _norm_query(self, q: str) -> str:
        try:
            return re.sub(r"\s+", " ", str(q or "").strip())
        except Exception:
            return str(q)

    def _filters_fingerprint(self, args: Dict[str, Any]) -> str:
        keep = {
            "collection", "language", "under", "kind", "symbol", "ext",
            "path_regex", "path_glob", "not_glob", "not_", "case",
            "limit", "per_path", "include_snippet",
        }
        try:
            filt = {k: args.get(k) for k in keep if k in args}
            def _norm(v):
                if v is None:
                    return None
                if isinstance(v, (list, tuple)):
                    return [str(x) for x in v]
                return v
            clean = {k: _norm(v) for k, v in filt.items()}
            return json.dumps(clean, sort_keys=True, ensure_ascii=False)
        except Exception:
            return "{}"

    def _group_key(self, base_url: str, args: Dict[str, Any]) -> str:
        coll = str(args.get("collection") or "")
        fp = self._filters_fingerprint(args)
        repo = os.getcwd()
        return f"{base_url}|{coll}|answer|{fp}|{repo}"

    def call_or_enqueue(self, base_url: str, tool: str, args: Dict[str, Any], timeout: float = 120.0) -> Dict[str, Any]:
        if not self.enabled:
            return self._call(base_url, tool, args, timeout=timeout)
        if self._should_bypass(args):
            return self._call(base_url, tool, args, timeout=timeout)

        start_ts = time.time()
        key = self._group_key(base_url, args or {})
        norm_q = self._norm_query((args or {}).get("query") or "")
        ev = threading.Event()
        slot = {"event": ev, "result": None, "error": None, "query": norm_q, "args": dict(args or {})}

        with self._lock:
            g = self._groups.get(key)
            if not g:
                g = {
                    "created": time.time(),
                    "items": [],
                    "timer": None,
                }
                self._groups[key] = g
            g["items"].append(slot)
            if g["timer"] is None:
                delay = max(0.0, float(self.window_ms) / 1000.0)
                t = threading.Timer(delay, self._flush, args=(key,))
                g["timer"] = t
                t.daemon = True
                t.start()
            if len(g["items"]) >= self.max_batch:
                t = g.get("timer")
                if t:
                    try:
                        t.cancel()
                    except Exception:
                        pass
                    g["timer"] = None
                threading.Thread(target=self._flush, args=(key,), daemon=True).start()

        remain = max(0.05, self.budget_ms / 1000.0)
        ev.wait(timeout=min(timeout, remain))
        if not ev.is_set():
            try:
                res = self._call(base_url, tool, args, timeout=timeout)
                slot["result"] = res
                ev.set()
                try:
                    with self._lock:
                        gg = self._groups.get(key)
                        if gg:
                            lst = gg.get("items") or []
                            if slot in lst:
                                try:
                                    lst.remove(slot)
                                except Exception:
                                    pass
                            if not lst:
                                t2 = gg.get("timer")
                                if t2:
                                    try:
                                        t2.cancel()
                                    except Exception:
                                        pass
                                self._groups.pop(key, None)
                except Exception:
                    pass
                try:
                    print(json.dumps({"router": {"batch_fallback": True, "elapsed_ms": int((time.time()-start_ts)*1000)}}), file=sys.stderr)
                except Exception:
                    pass
                return res
            except Exception as e:
                slot["error"] = e
                ev.set()
                raise

        if slot.get("error") is not None:
            raise slot["error"]
        return slot.get("result") or {}

    def _flush(self, key: str) -> None:
        with self._lock:
            g = self._groups.get(key)
            if not g:
                return
            items = g.get("items") or []
            g["items"] = []
            g["timer"] = None
            if not items:
                self._groups.pop(key, None)
                return

        unique_q: list[str] = []
        seen_q = set()
        for it in items:
            q = it.get("query") or ""
            if q not in seen_q:
                seen_q.add(q)
                unique_q.append(q)
        first_args = dict(items[0].get("args") or {})
        forward = {k: v for k, v in first_args.items() if k not in {"query", "queries"}}
        base_url = None
        try:
            base_url = key.split("|")[0]
        except Exception:
            base_url = HTTP_URL_INDEXER

        started = time.time()
        results_by_q: Dict[str, Any] = {}
        errors_by_q: Dict[str, Exception] = {}
        calls = 0
        try:
            import copy as _copy
        except Exception:
            _copy = None

        if len(unique_q) > 1:
            args_all = dict(forward)
            args_all["query"] = list(unique_q)
            args_all["mode"] = args_all.get("mode") or "pack"
            try:
                agg_res = self._call(base_url, "context_answer", args_all, timeout=120.0)
                calls = 1
                try:
                    payload = ((agg_res or {}).get("result") or {}).get("structuredContent") or {}
                    body = (payload.get("result") or {})
                except Exception:
                    payload, body = {}, {}

                abq = None
                try:
                    abq = body.get("answers_by_query")
                except Exception:
                    abq = None
                if isinstance(abq, list) and abq:
                    _map: Dict[str, Any] = {}
                    by_idx = (len(abq) >= len(unique_q))
                    for i, entry in enumerate(abq):
                        try:
                            qv = entry.get("query")
                            qk = None
                            if isinstance(qv, list) and qv:
                                qk = str(qv[0])
                            elif isinstance(qv, str):
                                qk = qv
                        except Exception:
                            qk = None
                        entry_key = qk if qk else (unique_q[i] if by_idx and i < len(unique_q) else None)
                        if not entry_key:
                            continue
                        per = _copy.deepcopy(agg_res) if _copy else json.loads(json.dumps(agg_res))
                        try:
                            per_body = (per.get("result") or {}).get("structuredContent", {}).get("result", {})
                        except Exception:
                            per_body = None
                        try:
                            ans_i = str(entry.get("answer") or "")
                            cits_i = entry.get("citations") or []
                            if per_body is not None:
                                per_body["answer"] = ans_i
                                per_body["citations"] = cits_i
                                per_body["query"] = [entry_key]
                        except Exception:
                            pass
                        _map[str(entry_key)] = per
                    for uq in unique_q:
                        if str(uq) in _map:
                            results_by_q[uq] = _map[str(uq)]
                    remaining = [uq for uq in unique_q if uq not in results_by_q]
                else:
                    remaining = list(unique_q)

                if remaining:
                    for uq in remaining:
                        args_i = dict(forward)
                        args_i["query"] = uq
                        try:
                            results_by_q[uq] = self._call(base_url, "context_answer", args_i, timeout=120.0)
                        except Exception as e:
                            errors_by_q[uq] = e
                    calls += len(remaining)
            except Exception as e:
                for uq in unique_q:
                    errors_by_q[uq] = e
                calls = 1
        else:
            args1 = dict(forward)
            args1["query"] = unique_q[0] if unique_q else ""
            try:
                results_by_q[args1["query"]] = self._call(base_url, "context_answer", args1, timeout=120.0)
            except Exception as e:
                errors_by_q[args1["query"]] = e
            calls = 1

        elapsed_ms = int((time.time() - started) * 1000)
        try:
            print(json.dumps({
                "router": {
                    "batch_flushed": True,
                    "n_items": len(items),
                    "unique_q": len(unique_q),
                    "calls": int(calls),
                    "elapsed_ms": elapsed_ms,
                    "ok": (len(errors_by_q) == 0),
                }
            }), file=sys.stderr)
        except Exception:
            pass

        for it in items:
            q = it.get("query") or ""
            it["result"] = results_by_q.get(q)
            it["error"] = errors_by_q.get(q)
            ev = it.get("event")
            try:
                if hasattr(ev, "set"):
                    ev.set()
            except Exception:
                pass
        with self._lock:
            gg = self._groups.get(key)
            if gg and not gg.get("items"):
                self._groups.pop(key, None)


# Global client singleton
_BATCH_CLIENT: BatchingContextAnswerClient | None = None


def get_batch_client() -> BatchingContextAnswerClient:
    """Get or create global batch client."""
    global _BATCH_CLIENT
    if _BATCH_CLIENT is None:
        _BATCH_CLIENT = BatchingContextAnswerClient()
    return _BATCH_CLIENT
