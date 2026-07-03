"""
mcp_router/memory.py - Memory store payload parsing.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Tuple

_MEMORY_TRIGGER_RE = re.compile(
    r"^(?:remember(?:\s+(?:this|that|me|to))?|save\s+memory|store\s+memory)\s*[:,\-]?\s*",
    re.IGNORECASE,
)
_MEMORY_INTENT_SPLIT_RE = re.compile(
    r"\b(?:then|and|also)\s+(?:reindex|index|recreate|prune|clean\s+up)\b",
    re.IGNORECASE,
)
_MEMORY_META_KEYS = {"priority", "tag", "tags", "topic", "category", "owner"}


def parse_memory_store_payload(q: str) -> Tuple[str, Dict[str, Any]]:
    """Parse memory store command, extracting content and metadata."""
    raw = str(q or "").strip()
    if not raw:
        return "", {}
    cleaned = _MEMORY_TRIGGER_RE.sub("", raw, count=1).lstrip()
    meta: Dict[str, Any] = {}

    def _assign_meta(key: str, value: str) -> None:
        k = key.lower()
        v = value.strip().strip(" \t\r\n,;.")
        if not v:
            return
        if k in {"tag", "tags"}:
            tags = [t.strip() for t in re.split(r"[,\s/]+", v) if t.strip()]
            if tags:
                meta["tags"] = tags
        else:
            meta[k] = v

    if cleaned.startswith("["):
        m = re.match(r"\[([^\]]+)\]\s*(.*)", cleaned, flags=re.S)
        if m:
            meta_block = m.group(1)
            cleaned = m.group(2)
            for key, val in re.findall(r"(\w+)\s*=\s*([^\s,;]+(?:,[^\s,;]+)*)", meta_block):
                if key.strip().lower() in _MEMORY_META_KEYS:
                    _assign_meta(key, val)

    while True:
        m = re.match(
            r"^(?P<key>(?:priority|tag|tags|topic|category|owner))\s*=\s*(?P<val>[^\s;:]+)\s*[,;:]?\s*(?P<rest>.*)$",
            cleaned,
            flags=re.IGNORECASE | re.S,
        )
        if not m:
            break
        _assign_meta(m.group("key"), m.group("val"))
        cleaned = m.group("rest")

    cleaned = cleaned.lstrip(":- ").lstrip()

    split = _MEMORY_INTENT_SPLIT_RE.search(cleaned)
    if split:
        cleaned = cleaned[: split.start()].rstrip(" ,;.")

    cleaned = cleaned.strip().strip('"').strip()
    if not cleaned:
        cleaned = raw
    return cleaned, meta
