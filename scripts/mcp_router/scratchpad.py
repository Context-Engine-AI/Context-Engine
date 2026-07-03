"""
mcp_router/scratchpad.py - Persistent scratchpad for context preservation.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict

from .config import scratchpad_ttl_sec


def scratchpad_path() -> str:
    """Get scratchpad file path."""
    base = os.path.join(os.getcwd(), ".codebase")
    try:
        os.makedirs(base, exist_ok=True)
    except Exception:
        pass
    return os.path.join(base, "router_scratchpad.json")


def load_scratchpad() -> Dict[str, Any]:
    """Load scratchpad with TTL handling."""
    import sys
    p = scratchpad_path()
    try:
        with open(p, "r", encoding="utf-8") as f:
            j = json.load(f)
            if isinstance(j, dict):
                try:
                    ts = float(j.get("timestamp") or 0.0)
                except Exception:
                    ts = 0.0
                ttl = scratchpad_ttl_sec()
                if ts and ttl >= 0 and (time.time() - ts) > ttl:
                    stale_keys = (
                        "last_plan",
                        "last_filters",
                        "mem_snippets",
                        "last_answer",
                        "last_citations",
                        "last_paths",
                        "last_metrics",
                    )
                    removed = False
                    for stale_key in stale_keys:
                        if stale_key in j:
                            j.pop(stale_key, None)
                            removed = True
                    if removed:
                        j["timestamp"] = 0.0
                        try:
                            print(
                                json.dumps({
                                    "router": {
                                        "scratchpad": "stale_cleared",
                                        "age_sec": round(time.time() - ts, 2),
                                    }
                                }),
                                file=sys.stderr,
                            )
                        except Exception:
                            pass
                return j
    except Exception:
        pass
    return {}


def save_scratchpad(d: Dict[str, Any]) -> None:
    """Save scratchpad atomically."""
    p = scratchpad_path()
    tmp = p + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(d, f)
            try:
                f.flush()
                os.fsync(f.fileno())
            except Exception:
                pass
        os.replace(tmp, p)
    except Exception:
        try:
            if os.path.exists(tmp):
                os.unlink(tmp)
        except Exception:
            pass


def looks_like_repeat(q: str) -> bool:
    """Check if query looks like a repeat request."""
    s = q.strip().lower()
    pats = [
        "repeat", "again", "same thing", "do that again", "rerun", "run it again", "same as before",
    ]
    return any(p in s for p in pats)


def looks_like_same_filters(q: str) -> bool:
    """Check if query asks to reuse filters."""
    s = q.strip().lower()
    return any(p in s for p in ["same filters", "reuse filters", "previous filters"])


def looks_like_expand(q: str) -> bool:
    """Check if query asks for expansion."""
    s = q.strip().lower()
    pats = [
        "expand on", "expand that", "expand the summary", "elaborate",
        "more detail", "more details", "go deeper", "add details",
    ]
    return any(p in s for p in pats)
