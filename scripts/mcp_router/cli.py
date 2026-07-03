#!/usr/bin/env python3
"""
mcp_router/cli.py - CLI entrypoint for MCP router.

Usage:
  python -m scripts.mcp_router --plan "How do I ...?"
  python -m scripts.mcp_router --run  "What is hybrid search?"
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from typing import Any, Dict, List

from .config import HTTP_URL_INDEXER, scratchpad_ttl_sec, divergence_thresholds
from .client import call_tool_http, is_failure_response, discover_tool_endpoints
from .scratchpad import (
    load_scratchpad,
    save_scratchpad,
    looks_like_repeat,
    looks_like_expand,
)
from .validation import (
    is_result_good,
    extract_metric_from_resp,
    material_drop,
)
from .batching import get_batch_client
from .planning import build_plan
from .config import divergence_is_fatal_for


def main(argv: List[str] | None = None) -> int:
    """Main CLI entrypoint."""
    if argv is None:
        argv = sys.argv[1:]
    
    ap = argparse.ArgumentParser()
    ap.add_argument("query", help="User query to route")
    ap.add_argument("--plan", action="store_true", help="Only print routing plan (no execution)")
    ap.add_argument("--run", action="store_true", help="Execute the routed tool(s) over HTTP")
    ap.add_argument("--timeout", type=float, default=180.0, help="HTTP timeout for tool calls")
    args = ap.parse_args(argv)

    plan = build_plan(args.query)
    print(json.dumps({"router": {"url": HTTP_URL_INDEXER, "plan": plan}}, indent=2))

    if args.plan and not args.run:
        return 0

    # Load scratchpad for prior context
    sp = {}
    fresh = False
    prior_answer = None
    prior_citations = None
    prior_paths = None
    try:
        sp = load_scratchpad()
        ts = float(sp.get("timestamp") or 0.0)
        fresh = bool(ts and (time.time() - ts) <= scratchpad_ttl_sec())
        if fresh:
            prior_answer = sp.get("last_answer")
            prior_citations = sp.get("last_citations")
            prior_paths = sp.get("last_paths")
    except Exception:
        pass

    # Execute sequentially until one succeeds
    last_err = None
    last = None
    tool_servers = discover_tool_endpoints()
    mem_snippets: list[str] = list(sp.get("mem_snippets") or []) if fresh else []
    batch_client = get_batch_client()

    for idx, (tool, targs) in enumerate(plan):
        base_url = tool_servers.get(tool, HTTP_URL_INDEXER)
        
        # Skip memory.find if we already have fresh snippets and this is a repeat/expand
        if (tool.lower().endswith("find") or tool.lower() in {"find", "memory.find"}) and mem_snippets and fresh and (looks_like_repeat(args.query) or looks_like_expand(args.query)):
            try:
                print(json.dumps({"tool": tool, "skipped": "scratchpad_fresh"}))
            except Exception:
                pass
            continue

        # Augment answer queries with context
        if tool in {"context_answer", "context_answer_compat"} and (mem_snippets or (fresh and (prior_answer or prior_citations or prior_paths))):
            try:
                tq = str((targs or {}).get("query") or args.query)
                sections = [tq]
                if mem_snippets:
                    bullets = []
                    for s in mem_snippets[:3]:
                        ss = re.sub(r"\s+", " ", str(s)).strip()
                        if len(ss) > 200:
                            ss = ss[:197] + "..."
                        bullets.append(f"- {ss}")
                    sections.append("Memory context:\n" + "\n".join(bullets))
                if fresh and (looks_like_expand(args.query) or looks_like_repeat(args.query)):
                    if isinstance(prior_answer, str) and prior_answer.strip():
                        pa = re.sub(r"\s+", " ", prior_answer).strip()
                        if len(pa) > 400:
                            pa = pa[:397] + "..."
                        sections.append("Prior summary:\n" + pa)
                    paths_list = []
                    if isinstance(prior_paths, list) and prior_paths:
                        paths_list = [str(p) for p in prior_paths[:5]]
                    elif isinstance(prior_citations, list) and prior_citations:
                        uniq = []
                        for c in prior_citations:
                            if isinstance(c, dict) and c.get("path") and c["path"] not in uniq:
                                uniq.append(c["path"])
                        paths_list = uniq[:5]
                    if paths_list:
                        sections.append("Citations context:\n" + "\n".join(f"- {p}" for p in paths_list))
                aug = "\n\n".join(sections)
                targs = {**(targs or {}), "query": aug}
            except Exception:
                pass

        try:
            if tool in {"context_answer", "context_answer_compat"}:
                res = batch_client.call_or_enqueue(base_url, tool, targs, timeout=args.timeout)
            else:
                res = call_tool_http(base_url, tool, targs, timeout=args.timeout)
            print(json.dumps({"tool": tool, "result": res}, indent=2))
            last = res

            # Capture memory snippets
            try:
                if tool.lower().endswith("find") or tool.lower() in {"find", "memory.find"}:
                    r = res.get("result") or {}
                    items = []
                    sc = r.get("structuredContent")
                    if isinstance(sc, dict):
                        rs0 = sc.get("result") or sc
                        if isinstance(rs0, dict):
                            items = rs0.get("results") or rs0.get("hits") or []
                    if not items:
                        content = r.get("content")
                        if isinstance(content, list):
                            for c in content:
                                if not isinstance(c, dict):
                                    continue
                                if "json" in c:
                                    j = c.get("json")
                                    if isinstance(j, (dict, list)):
                                        container = j.get("result") if isinstance(j, dict) and "result" in j else j
                                        if isinstance(container, dict):
                                            items = container.get("results") or container.get("hits") or []
                                            if items:
                                                break
                                if c.get("type") == "text":
                                    ttxt = c.get("text")
                                    if isinstance(ttxt, str) and ttxt.strip():
                                        try:
                                            j = json.loads(ttxt)
                                        except Exception:
                                            continue
                                        container = j.get("result") if isinstance(j, dict) and "result" in j else j
                                        if isinstance(container, dict):
                                            items = container.get("results") or container.get("hits") or []
                                            if items:
                                                break
                    for it in items:
                        if isinstance(it, dict):
                            txt = it.get("information") or it.get("content") or it.get("text")
                            if isinstance(txt, str) and txt.strip():
                                mem_snippets.append(txt.strip())
            except Exception:
                pass

            # Determine if we should treat this step as terminal
            has_future_answer = any(tn in {"context_answer", "context_answer_compat"} for (tn, _) in plan[idx + 1:])
            if (not is_failure_response(res)) and is_result_good(tool, res):
                if tool.lower() in {"find", "memory.find"} and has_future_answer:
                    continue

                # Persist scratchpad
                try:
                    last_filters: Dict[str, Any] = {}
                    for (tn, ta) in plan:
                        if tn == "repo_search" or tn.startswith("search_"):
                            if isinstance(ta, dict):
                                for k in ("language", "under", "symbol", "ext", "path_glob", "not_glob"):
                                    if ta.get(k) not in (None, ""):
                                        last_filters[k] = ta.get(k)
                            break
                    
                    last_answer_text = None
                    last_citations_list = None
                    last_paths_list: list[str] | None = None
                    if tool in {"context_answer", "context_answer_compat"}:
                        try:
                            r0 = res.get("result") or {}
                            sc0 = r0.get("structuredContent") or {}
                            rs0 = sc0.get("result") or sc0
                            if isinstance(rs0, dict):
                                ans0 = rs0.get("answer")
                                if isinstance(ans0, str):
                                    last_answer_text = ans0
                                cites0 = rs0.get("citations")
                                if isinstance(cites0, list):
                                    last_citations_list = cites0
                                    uniqp: list[str] = []
                                    for c in cites0:
                                        if isinstance(c, dict) and c.get("path") and c["path"] not in uniqp:
                                            uniqp.append(c["path"])
                                    last_paths_list = uniqp
                        except Exception:
                            pass

                    # Divergence detection
                    divergence_should_abort = False
                    last_metrics_prev = {}
                    try:
                        last_metrics_prev = sp.get("last_metrics") or {}
                        if not isinstance(last_metrics_prev, dict):
                            last_metrics_prev = {}
                    except Exception:
                        last_metrics_prev = {}
                    metric = extract_metric_from_resp(tool, res)
                    last_metrics_map = dict(last_metrics_prev)
                    if metric is not None:
                        mname, mval = metric
                        prev_val = None
                        try:
                            prev_val = last_metrics_prev.get(tool, {}).get(mname)
                            if prev_val is not None:
                                prev_val = float(prev_val)
                        except Exception:
                            prev_val = None
                        drop_frac, min_base = divergence_thresholds()
                        if material_drop(prev_val, float(mval), drop_frac, min_base):
                            fatal = divergence_is_fatal_for(tool)
                            try:
                                print(json.dumps({
                                    "divergence": {
                                        "tool": tool,
                                        "metric": mname,
                                        "previous": prev_val,
                                        "current": float(mval),
                                        "drop_frac": drop_frac,
                                        "fatal": fatal,
                                    }
                                }))
                            except Exception:
                                pass
                            if fatal:
                                divergence_should_abort = True
                        try:
                            last_metrics_map.setdefault(tool, {})[mname] = float(mval)
                        except Exception:
                            pass
                    else:
                        last_metrics_map = last_metrics_prev

                    success_criteria = {
                        "context_answer": {"expected_fields": ["answer"], "min_citations": 0},
                        "context_answer_compat": {"expected_fields": ["answer"], "min_citations": 0},
                        "repo_search": {"min_results": 1},
                        "search_config_for": {"min_results": 1},
                        "search_tests_for": {"min_results": 1},
                        "search_callers_for": {"min_results": 1},
                        "search_importers_for": {"min_results": 1},
                        "find": {"min_results": 1},
                    }
                    sp = {
                        "last_query": args.query,
                        "last_plan": plan,
                        "last_filters": last_filters or None,
                        "mem_snippets": mem_snippets[:5],
                        "last_answer": last_answer_text,
                        "last_citations": last_citations_list,
                        "last_paths": last_paths_list,
                        "success_criteria": success_criteria,
                        "last_metrics": last_metrics_map,
                        "timestamp": time.time(),
                    }
                    save_scratchpad(sp)
                except Exception:
                    pass

                if divergence_should_abort:
                    continue

                return 0
        except Exception as e:
            last_err = e
            try:
                print(json.dumps({"tool": tool, "server": base_url, "error": str(e)}), file=sys.stderr)
            except Exception:
                pass
            continue

    if last_err:
        print(f"Router: all attempts failed: {last_err}", file=sys.stderr)
    return 1 if last is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
