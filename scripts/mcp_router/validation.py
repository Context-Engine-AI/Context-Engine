"""
mcp_router/validation.py - Response validation and metric extraction.
"""
from __future__ import annotations

from typing import Any, Dict

from .client import is_failure_response
from .config import divergence_thresholds, divergence_is_fatal_for


def is_result_good(tool: str, resp: Dict[str, Any]) -> bool:
    """Check if result is good enough to stop the plan."""
    try:
        r = resp.get("result") or {}
        sc = r.get("structuredContent") or {}
        rs = sc.get("result") or {}
        
        if tool in {"context_answer", "context_answer_compat"}:
            ans = rs.get("answer") if isinstance(rs, dict) else None
            if isinstance(ans, str):
                s = ans.strip()
                if s and not any(p in s.lower() for p in [
                    "insufficient context", "not enough context", "no relevant", "don't know", "cannot answer"
                ]):
                    return True
            cites = rs.get("citations") if isinstance(rs, dict) else None
            if isinstance(cites, list) and len(cites) > 0:
                return True
            return False
        
        if tool.startswith("search_") or tool == "repo_search":
            total = rs.get("total") if isinstance(rs, dict) else None
            if isinstance(total, int) and total > 0:
                return True
            results = rs.get("results") if isinstance(rs, dict) else None
            if isinstance(results, list) and len(results) > 0:
                return True
            return False
        
        return not is_failure_response(resp)
    except Exception:
        return not is_failure_response(resp)


def extract_metric_from_resp(tool: str, resp: Dict[str, Any]) -> tuple[str, float] | None:
    """Extract metric for divergence detection."""
    try:
        r = resp.get("result") or {}
        sc = r.get("structuredContent") or {}
        rs = sc.get("result") or {}
        
        if tool in {"repo_search", "code_search", "context_search", "search_tests_for", "search_config_for", "search_callers_for", "search_importers_for"}:
            tot = rs.get("total")
            if isinstance(tot, (int, float)):
                return ("total_results", float(tot))
            results = rs.get("results")
            if isinstance(results, list):
                return ("total_results", float(len(results)))
            return None
        
        if tool in {"context_answer", "context_answer_compat"}:
            cites = rs.get("citations")
            if isinstance(cites, list):
                return ("citations", float(len(cites)))
            return ("citations", 0.0)
        
        if tool == "qdrant_status":
            cnt = rs.get("count")
            if isinstance(cnt, (int, float)):
                return ("points", float(cnt))
            return None
    except Exception:
        return None
    return None


def material_drop(prev: float | None, curr: float, drop_frac: float, min_base: int) -> bool:
    """Check if there's a material drop in metrics."""
    try:
        if prev is None:
            return False
        if prev < float(min_base):
            return False
        return curr < (float(prev) * float(drop_frac))
    except Exception:
        return False
