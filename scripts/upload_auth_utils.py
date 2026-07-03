import os
import json
import time
from typing import Any


def get_auth_session(upload_endpoint: str) -> str:
    """Resolve auth session from environment or ~/.ctxce/auth.json.

    This mirrors the existing behavior used by the upload clients:
    - Prefer CTXCE_UPLOAD_SESSION_ID / CTXCE_SESSION_ID from the environment.
    - Fall back to ~/.ctxce/auth.json keyed by the upload endpoint (with and without
      a trailing slash), honoring an optional numeric expiresAt/expires_at field.
    - Treat expiresAt <= 0 or missing as non-expiring.

    Returns an empty string when no usable session is found.
    """
    try:
        sess = (os.environ.get("CTXCE_UPLOAD_SESSION_ID") or os.environ.get("CTXCE_SESSION_ID") or "").strip()
    except Exception:
        sess = ""
    if sess:
        return sess

    try:
        home = os.path.expanduser("~")
        cfg_path = os.path.join(home, ".ctxce", "auth.json")
        if not os.path.exists(cfg_path):
            return ""
        with open(cfg_path, "r", encoding="utf-8") as f:
            raw: Any = json.load(f)
        if not isinstance(raw, dict):
            return ""
        key = upload_endpoint.rstrip("/")
        entry = raw.get(key) or raw.get(upload_endpoint)
        if not isinstance(entry, dict):
            return ""
        sid = entry.get("sessionId") or entry.get("session_id")
        exp = entry.get("expiresAt") or entry.get("expires_at")
        now_secs = int(time.time())
        if isinstance(exp, (int, float)) and exp > 0:
            if exp >= now_secs:
                return (sid or "").strip()
            return ""
        return (sid or "").strip()
    except Exception:
        return ""
