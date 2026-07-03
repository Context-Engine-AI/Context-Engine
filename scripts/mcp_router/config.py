"""
mcp_router/config.py - Shared configuration and constants.
"""
from __future__ import annotations

import os

# HTTP endpoints
HTTP_URL_INDEXER = os.environ.get("MCP_INDEXER_HTTP_URL", "http://localhost:8003/mcp").rstrip("/")
HTTP_URL_MEMORY = os.environ.get("MCP_MEMORY_HTTP_URL", "http://localhost:8002/mcp").rstrip("/")
DEFAULT_HTTP_URL = HTTP_URL_INDEXER

# Health ports
try:
    HEALTH_PORT_INDEXER = int(os.environ.get("FASTMCP_INDEXER_HTTP_HEALTH_PORT", "18003") or 18003)
except (ValueError, TypeError):
    HEALTH_PORT_INDEXER = 18003

try:
    HEALTH_PORT_MEMORY = int(os.environ.get("FASTMCP_HTTP_HEALTH_PORT", "18002") or 18002)
except (ValueError, TypeError):
    HEALTH_PORT_MEMORY = 18002


def cache_ttl_sec() -> int:
    try:
        return int(os.environ.get("ROUTER_TOOLS_CACHE_TTL_SEC", "60") or 60)
    except Exception:
        return 60


def scratchpad_ttl_sec() -> int:
    try:
        return int(os.environ.get("ROUTER_SCRATCHPAD_TTL_SEC", "300") or 300)
    except Exception:
        return 300


def divergence_thresholds() -> tuple[float, int]:
    try:
        drop_frac = float(os.environ.get("ROUTER_DIVERGENCE_DROP_FRAC", "0.5") or 0.5)
    except Exception:
        drop_frac = 0.5
    try:
        min_base = int(os.environ.get("ROUTER_DIVERGENCE_MIN_BASE", "3") or 3)
    except Exception:
        min_base = 3
    return drop_frac, min_base


def divergence_is_fatal_for(tool: str) -> bool:
    try:
        s = (os.environ.get("ROUTER_DIVERGENCE_FATAL_TOOLS", "") or "").strip()
        if not s:
            return False
        low = s.lower()
        if low in {"*", "all", "1", "true"}:
            return True
        names = {t.strip().lower() for t in s.split(",") if t.strip()}
        return tool.strip().lower() in names
    except Exception:
        return False


# Language set for hint parsing
LANGS = {
    "python", "typescript", "javascript", "go", "java", "rust", "kotlin",
    "c++", "cpp", "csharp", "c#", "ruby", "php", "scala", "swift", "bash", "shell"
}
