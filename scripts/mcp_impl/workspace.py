#!/usr/bin/env python3
"""
mcp/workspace.py - Workspace state and collection resolution utilities.

Extracted from mcp_indexer_server.py for better modularity.
Contains:
- Workspace state file reading/writing
- Default collection resolution
- Session defaults management
- Script path resolution
"""

from __future__ import annotations

__all__ = [
    # Constants
    "DEFAULT_COLLECTION",
    "QDRANT_URL",
    # Session state
    "SESSION_DEFAULTS",
    "SESSION_DEFAULTS_BY_SESSION",
    "_SESSION_LOCK",
    "_SESSION_CTX_LOCK",
    "_MEM_COLL_CACHE",
    # Functions
    "_state_file_path",
    "_read_ws_state",
    "_default_collection",
    "_work_script",
]

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, Optional
from weakref import WeakKeyDictionary

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_COLLECTION = os.environ.get("DEFAULT_COLLECTION") or os.environ.get("COLLECTION_NAME") or "codebase"
QDRANT_URL = os.environ.get("QDRANT_URL", "http://qdrant:6333")

# ---------------------------------------------------------------------------
# Session state (guarded by locks for thread safety)
# ---------------------------------------------------------------------------
# Cache for memory collection autodetection (name + timestamp)
_MEM_COLL_CACHE: Dict[str, Any] = {"name": None, "ts": 0.0}

# Session defaults map (token -> defaults). Guarded for concurrency.
_SESSION_LOCK = threading.Lock()
SESSION_DEFAULTS: Dict[str, Dict[str, Any]] = {}

# Per-connection defaults keyed by ctx.session (no token required)
_SESSION_CTX_LOCK = threading.Lock()
SESSION_DEFAULTS_BY_SESSION: "WeakKeyDictionary[Any, Dict[str, Any]]" = WeakKeyDictionary()


# ---------------------------------------------------------------------------
# Workspace state file utilities
# ---------------------------------------------------------------------------
def _state_file_path(ws_path: str = "/work") -> str:
    """Get path to workspace state file, delegating to workspace_state module."""
    try:
        from scripts.workspace_state import _state_file_path as _ws_state_file_path
        return str(_ws_state_file_path(workspace_path=ws_path, repo_name=None))
    except Exception as exc:
        logger.warning(f"State file path construction failed, using fallback: {exc}")
        return os.path.join(ws_path, ".codebase", "state.json")


def _read_ws_state(ws_path: str = "/work") -> Optional[Dict[str, Any]]:
    """Read and parse workspace state JSON file."""
    try:
        p = _state_file_path(ws_path)
        if not os.path.exists(p):
            return None
        with open(p, "r", encoding="utf-8") as f:
            obj = json.load(f)
            return obj if isinstance(obj, dict) else None
    except Exception as e:
        logger.debug(f"Failed to read workspace state: {e}")
        return None


def _default_collection() -> str:
    """Resolve default collection name from env or workspace state."""
    env_coll = (os.environ.get("DEFAULT_COLLECTION") or os.environ.get("COLLECTION_NAME") or "").strip()
    if env_coll:
        return env_coll
    st = _read_ws_state("/work")
    if st:
        coll = st.get("qdrant_collection")
        if isinstance(coll, str) and coll.strip():
            return coll.strip()
    return DEFAULT_COLLECTION


def _work_script(name: str) -> str:
    """Return path to script respecting bind mounts first, then /app, then local fallback."""
    try:
        work_path = os.path.join("/work", "scripts", name)
        if os.path.exists(work_path):
            return work_path
    except Exception:
        pass

    try:
        app_path = os.path.join("/app", "scripts", name)
        if os.path.exists(app_path):
            return app_path
    except Exception:
        pass

    return os.path.join(os.getcwd(), "scripts", name)

