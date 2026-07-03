"""Authentication backend for Context-Engine services.

Provides a minimal, SQLite-backed user and session store with
password hashing and optional shared-token based session issuance.

Design notes (PoC-friendly, forward compatible):

- Storage schema is intentionally simple and portable (TEXT/INTEGER fields
  only) so it can be migrated to a real RDBMS (Postgres/MySQL) later without
  changing the logical model.
- AUTH_DB_URL accepts a SQLite-style URL today, but callers should treat it
  as an abstract database URL; future versions may use Alembic-style schema
  migrations and support multiple engines.
- Current focus is users + sessions only. In a fuller deployment, this module
  is the natural place to grow organization and collection metadata, including
  mapping users/orgs to existing Qdrant collections and enforcing collection-
  level ACLs.

Auth is fully opt-in via environment variables and can be reused by
multiple services (upload, dedicated auth service, MCP indexers, etc.).
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, Optional, List

# Configuration
WORK_DIR = os.environ.get("WORK_DIR", "/work")
AUTH_ENABLED = (
    str(os.environ.get("CTXCE_AUTH_ENABLED", "0"))
    .strip()
    .lower()
    in {"1", "true", "yes", "on"}
)
_default_auth_db_path = os.path.join(WORK_DIR, ".codebase", "ctxce_auth.sqlite")
AUTH_DB_URL = os.environ.get("CTXCE_AUTH_DB_URL") or f"sqlite:///{_default_auth_db_path}"
AUTH_SHARED_TOKEN = os.environ.get("CTXCE_AUTH_SHARED_TOKEN")
COLLECTION_REGISTRY_ENABLED = AUTH_ENABLED
ACL_ALLOW_ALL = (
    str(os.environ.get("CTXCE_ACL_ALLOW_ALL", "0")).strip().lower()
    in {"1", "true", "yes", "on"}
)
ALLOW_OPEN_TOKEN_LOGIN = (
    str(os.environ.get("CTXCE_AUTH_ALLOW_OPEN_TOKEN_LOGIN", "0"))
    .strip()
    .lower()
    in {"1", "true", "yes", "on"}
)

_SESSION_TTL_SECONDS_DEFAULT = 0
try:
    _raw_ttl = os.environ.get("CTXCE_AUTH_SESSION_TTL_SECONDS")
    if _raw_ttl is not None and str(_raw_ttl).strip() != "":
        AUTH_SESSION_TTL_SECONDS = int(str(_raw_ttl).strip())
    else:
        AUTH_SESSION_TTL_SECONDS = _SESSION_TTL_SECONDS_DEFAULT
except Exception:
    AUTH_SESSION_TTL_SECONDS = _SESSION_TTL_SECONDS_DEFAULT


class AuthDisabledError(Exception):
    """Raised when auth is disabled via configuration."""


class AuthInvalidToken(Exception):
    """Raised when a shared token login attempt fails validation."""


def _get_auth_db_path() -> str:
    raw = AUTH_DB_URL or ""
    if raw.startswith("sqlite///"):
        return raw[len("sqlite///") :]
    if raw.startswith("sqlite://"):
        return raw[len("sqlite://") :]
    return raw


@contextmanager
def _db_connection():
    path = _get_auth_db_path()
    conn = sqlite3.connect(path)
    try:
        yield conn
    finally:
        conn.close()


def _ensure_db() -> None:
    path = _get_auth_db_path()
    if not path:
        return
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    except Exception:
        pass
    with _db_connection() as conn:
        with conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, user_id TEXT, created_at INTEGER, expires_at INTEGER, metadata_json TEXT)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS users (id TEXT PRIMARY KEY, username TEXT UNIQUE NOT NULL, password_hash TEXT NOT NULL, created_at INTEGER NOT NULL, metadata_json TEXT, role TEXT NOT NULL DEFAULT 'user')"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS collections (id TEXT PRIMARY KEY, qdrant_collection TEXT UNIQUE NOT NULL, created_at INTEGER NOT NULL, metadata_json TEXT, is_deleted INTEGER NOT NULL DEFAULT 0)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS collection_acl (collection_id TEXT NOT NULL, user_id TEXT NOT NULL, permission TEXT NOT NULL, created_at INTEGER NOT NULL, PRIMARY KEY (collection_id, user_id))"
            )
            try:
                cur = conn.cursor()
                cur.execute("PRAGMA table_info(users)")
                cols = [r[1] for r in cur.fetchall() or []]
                if "role" not in cols:
                    conn.execute(
                        "ALTER TABLE users ADD COLUMN role TEXT NOT NULL DEFAULT 'user'"
                    )
            except Exception:
                pass


def _hash_password(password: str) -> str:
    if not isinstance(password, str) or not password:
        raise ValueError("Password is required")
    salt = os.urandom(16)
    iterations = 200_000
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return f"pbkdf2_sha256${iterations}${salt.hex()}${dk.hex()}"


def _verify_password(password: str, encoded: str) -> bool:
    try:
        scheme, iter_s, salt_hex, hash_hex = encoded.split("$", 3)
        if scheme != "pbkdf2_sha256":
            return False
        iterations = int(iter_s)
        salt = bytes.fromhex(salt_hex)
        dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
        return dk.hex() == hash_hex
    except Exception:
        return False


def create_user(
    username: str,
    password: str,
    metadata: Optional[Dict[str, Any]] = None,
    role: Optional[str] = None,
) -> Dict[str, Any]:
    if not AUTH_ENABLED:
        raise AuthDisabledError("Auth not enabled")
    _ensure_db()
    path = _get_auth_db_path()
    now_ts = int(datetime.now().timestamp())
    password_hash = _hash_password(password)
    meta_json: Optional[str] = None
    if metadata:
        try:
            meta_json = json.dumps(metadata)
        except Exception:
            meta_json = None
    user_id = uuid.uuid4().hex
    with _db_connection() as conn:
        with conn:
            desired_role = (str(role).strip().lower() if role is not None else "")
            if desired_role and desired_role not in {"user", "admin"}:
                raise ValueError("Invalid role")
            role_val = desired_role or "user"
            try:
                cur = conn.cursor()
                cur.execute("SELECT 1 FROM users LIMIT 1")
                if not cur.fetchone():
                    role_val = "admin"
            except Exception:
                role_val = "user"
            conn.execute(
                "INSERT INTO users (id, username, password_hash, created_at, metadata_json, role) VALUES (?, ?, ?, ?, ?, ?)",
                (user_id, username, password_hash, now_ts, meta_json, role_val),
            )
    return {"id": user_id, "user_id": user_id, "username": username, "role": role_val}


def _get_user_by_username(username: str) -> Optional[Dict[str, Any]]:
    _ensure_db()
    with _db_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, username, password_hash, created_at, metadata_json, role FROM users WHERE username = ?",
            (username,),
        )
        row = cur.fetchone()
    if not row:
        return None
    return {
        "id": row[0],
        "username": row[1],
        "password_hash": row[2],
        "created_at": row[3],
        "metadata_json": row[4],
        "role": row[5],
    }


def _get_user_role(user_id: str) -> Optional[str]:
    uid = (user_id or "").strip()
    if not uid:
        return None
    _ensure_db()
    with _db_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT role FROM users WHERE id = ?", (uid,))
        row = cur.fetchone()
    if not row:
        return None
    return str(row[0] or "").strip() or None


def is_admin_user(user_id: str) -> bool:
    return (_get_user_role(user_id) or "").lower() == "admin"


def has_any_users() -> bool:
    """Return True if at least one user exists.

    Used by HTTP layers to allow first-user bootstrap flows when the
    database is empty. Raises AuthDisabledError when auth is disabled.
    """
    if not AUTH_ENABLED:
        raise AuthDisabledError("Auth not enabled")
    _ensure_db()
    with _db_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM users LIMIT 1")
        row = cur.fetchone()
    return bool(row)


def authenticate_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    user = _get_user_by_username(username)
    if not user:
        return None
    if not _verify_password(password, user.get("password_hash") or ""):
        return None
    return user


def create_session(
    user_id: str,
    metadata: Optional[Dict[str, Any]] = None,
    ttl_seconds: int = AUTH_SESSION_TTL_SECONDS,
) -> Dict[str, Any]:
    if not AUTH_ENABLED:
        raise AuthDisabledError("Auth not enabled")
    _ensure_db()
    path = _get_auth_db_path()
    now_ts = int(datetime.now().timestamp())
    ttl_val = int(ttl_seconds or 0)
    if ttl_val <= 0:
        expires_ts = 0
    else:
        expires_ts = now_ts + ttl_val
    meta_json: Optional[str] = None
    if metadata:
        try:
            meta_json = json.dumps(metadata)
        except Exception:
            meta_json = None
    session_id = uuid.uuid4().hex
    with _db_connection() as conn:
        with conn:
            conn.execute(
                "INSERT OR REPLACE INTO sessions (id, user_id, created_at, expires_at, metadata_json) VALUES (?, ?, ?, ?, ?)",
                (session_id, user_id, now_ts, expires_ts, meta_json),
            )
    return {"session_id": session_id, "user_id": user_id, "expires_at": expires_ts}


def create_session_for_token(
    client: str,
    workspace: Optional[str] = None,
    token: Optional[str] = None,
) -> Dict[str, Any]:
    if not AUTH_ENABLED:
        raise AuthDisabledError("Auth not enabled")
    if AUTH_SHARED_TOKEN:
        # When a shared token is configured, require it for all token-based sessions.
        if not token or token != AUTH_SHARED_TOKEN:
            raise AuthInvalidToken("Invalid auth token")
    else:
        # Harden default behavior: when auth is enabled but no shared token is configured,
        # disable token-based login unless explicitly allowed via env.
        if not ALLOW_OPEN_TOKEN_LOGIN:
            raise AuthInvalidToken(
                "Token-based login disabled (no shared token configured; set CTXCE_AUTH_SHARED_TOKEN "
                "or CTXCE_AUTH_ALLOW_OPEN_TOKEN_LOGIN=1 to enable)"
            )
    user_id = client or "ctxce"
    meta: Dict[str, Any] = {}
    if workspace:
        meta["workspace"] = workspace
    return create_session(user_id=user_id, metadata=meta)


def validate_session(session_id: str) -> Optional[Dict[str, Any]]:
    """Validate a session id and return its record if active.

    Returns a dict with keys {id, user_id, created_at, expires_at, metadata}
    when valid, or None when missing/expired/unknown. Raises AuthDisabledError
    when auth is disabled.
    """
    if not AUTH_ENABLED:
        raise AuthDisabledError("Auth not enabled")
    sid = (session_id or "").strip()
    if not sid:
        return None
    _ensure_db()
    with _db_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, user_id, created_at, expires_at, metadata_json FROM sessions WHERE id = ?",
            (sid,),
        )
        row = cur.fetchone()
    if not row:
        return None
    now_ts = int(datetime.now().timestamp())
    expires_ts = int(row[3] or 0)
    if expires_ts and expires_ts < now_ts:
        return None
    if AUTH_SESSION_TTL_SECONDS > 0 and expires_ts:
        remaining = expires_ts - now_ts
        if remaining < AUTH_SESSION_TTL_SECONDS // 2:
            new_expires_ts = now_ts + AUTH_SESSION_TTL_SECONDS
            try:
                with _db_connection() as conn2:
                    with conn2:
                        conn2.execute(
                            "UPDATE sessions SET expires_at = ? WHERE id = ?",
                            (new_expires_ts, sid),
                        )
                expires_ts = new_expires_ts
            except Exception:
                pass
    meta: Optional[Dict[str, Any]] = None
    raw_meta = row[4]
    if isinstance(raw_meta, str) and raw_meta.strip():
        try:
            obj = json.loads(raw_meta)
            if isinstance(obj, dict):
                meta = obj
        except Exception:
            meta = None
    return {
        "id": row[0],
        "user_id": row[1],
        "created_at": int(row[2] or 0),
        "expires_at": expires_ts,
        "metadata": meta or {},
    }


def ensure_collection(qdrant_collection: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if not COLLECTION_REGISTRY_ENABLED:
        raise AuthDisabledError("Collection registry not enabled")
    name = (qdrant_collection or "").strip()
    if not name:
        raise ValueError("qdrant_collection is required")
    undelete_on_discovery = (
        str(os.environ.get("CTXCE_COLLECTION_REGISTRY_UNDELETE_ON_DISCOVERY", "0")).strip().lower()
        in {"1", "true", "yes", "on"}
    )
    _ensure_db()
    now_ts = int(datetime.now().timestamp())
    meta_json: Optional[str] = None
    if metadata:
        try:
            meta_json = json.dumps(metadata)
        except Exception:
            meta_json = None
    with _db_connection() as conn:
        with conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT id, qdrant_collection, created_at, metadata_json, is_deleted FROM collections WHERE qdrant_collection = ?",
                (name,),
            )
            row = cur.fetchone()
            if row:
                is_deleted = int(row[4] or 0)
                if undelete_on_discovery and is_deleted != 0:
                    try:
                        conn.execute(
                            "UPDATE collections SET is_deleted = 0 WHERE id = ?",
                            (row[0],),
                        )
                        is_deleted = 0
                    except Exception:
                        pass
                return {
                    "id": row[0],
                    "qdrant_collection": row[1],
                    "created_at": int(row[2] or 0),
                    "metadata_json": row[3],
                    "is_deleted": is_deleted,
                }

            coll_id = uuid.uuid4().hex
            conn.execute(
                "INSERT INTO collections (id, qdrant_collection, created_at, metadata_json, is_deleted) VALUES (?, ?, ?, ?, 0)",
                (coll_id, name, now_ts, meta_json),
            )
            return {
                "id": coll_id,
                "qdrant_collection": name,
                "created_at": now_ts,
                "metadata_json": meta_json,
                "is_deleted": 0,
            }


def ensure_collections(collections: List[str]) -> int:
    if not COLLECTION_REGISTRY_ENABLED:
        raise AuthDisabledError("Collection registry not enabled")
    names = [str(c).strip() for c in (collections or []) if str(c).strip()]
    if not names:
        return 0
    _ensure_db()
    before_count = 0
    after_count = 0
    failures: List[str] = []
    try:
        with _db_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(1) FROM collections")
            row = cur.fetchone()
            if row:
                before_count = int(row[0] or 0)
    except Exception:
        before_count = 0
    for name in names:
        try:
            ensure_collection(name)
        except Exception as e:
            failures.append(f"{name}: {e}")
            continue
    try:
        with _db_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(1) FROM collections")
            row = cur.fetchone()
            if row:
                after_count = int(row[0] or 0)
    except Exception:
        after_count = before_count

    delta = max(0, after_count - before_count)
    if failures and len(failures) >= len(names) and delta == 0:
        raise RuntimeError("Failed to sync collections registry: " + "; ".join(failures[:3]))
    return delta


def grant_collection_access(user_id: str, qdrant_collection: str, permission: str = "read") -> Dict[str, Any]:
    if not AUTH_ENABLED:
        raise AuthDisabledError("Auth not enabled")
    _ensure_db()
    uid = (user_id or "").strip()
    perm = (permission or "read").strip() or "read"
    if not uid:
        raise ValueError("user_id is required")
    coll = ensure_collection(qdrant_collection)
    now_ts = int(datetime.now().timestamp())
    with _db_connection() as conn:
        with conn:
            conn.execute(
                "INSERT OR REPLACE INTO collection_acl (collection_id, user_id, permission, created_at) VALUES (?, ?, ?, ?)",
                (coll.get("id"), uid, perm, now_ts),
            )
    return {"collection_id": coll.get("id"), "user_id": uid, "permission": perm}


def revoke_collection_access(user_id: str, qdrant_collection: str) -> bool:
    if not AUTH_ENABLED:
        raise AuthDisabledError("Auth not enabled")
    _ensure_db()
    uid = (user_id or "").strip()
    name = (qdrant_collection or "").strip()
    if not uid:
        raise ValueError("user_id is required")
    if not name:
        raise ValueError("qdrant_collection is required")
    with _db_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT c.id FROM collections c WHERE c.qdrant_collection = ? AND c.is_deleted = 0",
            (name,),
        )
        row = cur.fetchone()
        if not row:
            return False
        coll_id = row[0]
        with conn:
            cur.execute(
                "DELETE FROM collection_acl WHERE collection_id = ? AND user_id = ?",
                (coll_id, uid),
            )
            return bool(cur.rowcount and cur.rowcount > 0)


def has_collection_access(user_id: str, qdrant_collection: str, permission: str = "read") -> bool:
    if ACL_ALLOW_ALL:
        return True
    if not AUTH_ENABLED:
        raise AuthDisabledError("Auth not enabled")
    uid = (user_id or "").strip()
    if not uid:
        return False
    if is_admin_user(uid):
        return True
    name = (qdrant_collection or "").strip()
    if not name:
        return False
    _ensure_db()
    with _db_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT c.id FROM collections c WHERE c.qdrant_collection = ? AND c.is_deleted = 0",
            (name,),
        )
        row = cur.fetchone()
        if not row:
            return False
        coll_id = row[0]
        cur.execute(
            "SELECT permission FROM collection_acl WHERE collection_id = ? AND user_id = ?",
            (coll_id, uid),
        )
        perm_row = cur.fetchone()
    if not perm_row:
        return False
    granted = str(perm_row[0] or "").strip().lower()
    want = (permission or "read").strip().lower()
    if granted == "admin":
        return True
    if want == "read":
        return granted in {"read", "write"}
    if want == "write":
        return granted == "write"
    return granted == want


def mark_collection_deleted(qdrant_collection: str) -> bool:
    if not AUTH_ENABLED:
        raise AuthDisabledError("Auth not enabled")
    name = (qdrant_collection or "").strip()
    if not name:
        raise ValueError("qdrant_collection is required")
    _ensure_db()
    with _db_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT id FROM collections WHERE qdrant_collection = ?",
            (name,),
        )
        row = cur.fetchone()
        if not row:
            return False
        coll_id = row[0]
        with conn:
            conn.execute(
                "UPDATE collections SET is_deleted = 1 WHERE id = ?",
                (coll_id,),
            )
            conn.execute(
                "DELETE FROM collection_acl WHERE collection_id = ?",
                (coll_id,),
            )
    return True


def list_users() -> List[Dict[str, Any]]:
    if not AUTH_ENABLED:
        raise AuthDisabledError("Auth not enabled")
    _ensure_db()
    with _db_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT id, username, created_at, role FROM users ORDER BY created_at ASC")
        rows = cur.fetchall() or []
    out: List[Dict[str, Any]] = []
    for r in rows:
        out.append(
            {
                "id": r[0],
                "username": r[1],
                "created_at": int(r[2] or 0),
                "role": r[3],
            }
        )
    return out


def list_collections(include_deleted: bool = False) -> List[Dict[str, Any]]:
    if not AUTH_ENABLED:
        raise AuthDisabledError("Auth not enabled")
    _ensure_db()
    with _db_connection() as conn:
        cur = conn.cursor()
        if include_deleted:
            cur.execute(
                "SELECT id, qdrant_collection, created_at, is_deleted FROM collections ORDER BY created_at ASC"
            )
        else:
            cur.execute(
                "SELECT id, qdrant_collection, created_at, is_deleted FROM collections WHERE is_deleted = 0 ORDER BY created_at ASC"
            )
        rows = cur.fetchall() or []
    out: List[Dict[str, Any]] = []
    for r in rows:
        out.append(
            {
                "id": r[0],
                "qdrant_collection": r[1],
                "created_at": int(r[2] or 0),
                "is_deleted": int(r[3] or 0),
            }
        )
    return out


def list_collection_acl() -> List[Dict[str, Any]]:
    if not AUTH_ENABLED:
        raise AuthDisabledError("Auth not enabled")
    _ensure_db()
    with _db_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT a.collection_id, c.qdrant_collection, a.user_id, u.username, u.role, a.permission, a.created_at "
            "FROM collection_acl a "
            "JOIN collections c ON c.id = a.collection_id "
            "LEFT JOIN users u ON u.id = a.user_id "
            "WHERE c.is_deleted = 0 "
            "ORDER BY c.qdrant_collection ASC, u.username ASC"
        )
        rows = cur.fetchall() or []
    out: List[Dict[str, Any]] = []
    for r in rows:
        out.append(
            {
                "collection_id": r[0],
                "qdrant_collection": r[1],
                "user_id": r[2],
                "username": r[3],
                "user_role": r[4],
                "permission": r[5],
                "created_at": int(r[6] or 0),
            }
        )
    return out
