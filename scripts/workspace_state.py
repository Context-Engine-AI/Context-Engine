#!/usr/bin/env python3
"""
Workspace state management for .codebase/state.json files.

This module provides functionality to track workspace-specific state including:
- Collection information and indexing status
- Progress tracking during indexing operations
- Activity logging with structured metadata
- Multi-repo support with per-repo state files
"""
import json
import os
import re
import uuid
import subprocess
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Literal, TypedDict
import threading
import time

_CANONICAL_SLUG_RE = re.compile(r"^.+-[0-9a-f]{16}$")
_SLUGGED_REPO_RE = re.compile(r"^.+-[0-9a-f]{16}(?:_old)?$")
_managed_slug_cache_lock = threading.Lock()
_managed_slug_cache: set[str] = set()
_managed_slug_cache_neg: set[str] = set()

_cache_memo_lock = threading.Lock()
_cache_memo: Dict[str, Dict[str, Any]] = {}
_cache_memo_sig: Dict[str, tuple[int, int]] = {}
_cache_memo_last_check: Dict[str, float] = {}


def is_staging_enabled() -> bool:
    raw = os.environ.get("CTXCE_STAGING_ENABLED", "")
    v = (raw or "").strip().lower()
    if not v:
        return False
    return v in {"1", "true", "yes", "on"}


def _cache_memo_recheck_seconds() -> float:
    try:
        return float(os.environ.get("CACHE_MEMO_RECHECK_SECONDS", "60") or 60)
    except Exception:
        return 60.0


def _normalize_cache_key_path(file_path: str) -> str:
    """Normalize a file path for cache keys.

    Prefer os.path.abspath (no filesystem calls) over Path.resolve(), since resolve
    can trigger expensive metadata operations on network filesystems.
    """
    try:
        return os.path.abspath(file_path)
    except Exception:
        try:
            return str(Path(file_path))
        except Exception:
            return str(file_path)


def _memoize_cache_obj(cache_path: Path, obj: Dict[str, Any]) -> None:
    key = str(cache_path)
    now = time.time()
    sig = (-1, -1)
    try:
        st = cache_path.stat()
        mtime_ns = int(
            getattr(st, "st_mtime_ns", int(getattr(st, "st_mtime", 0) * 1_000_000_000))
        )
        sig = (mtime_ns, int(getattr(st, "st_size", 0)))
    except OSError:
        sig = (-1, -1)
    with _cache_memo_lock:
        _cache_memo[key] = obj
        _cache_memo_last_check[key] = now
        _cache_memo_sig[key] = sig


def _cache_file_sig(cache_path: Path) -> Optional[tuple[int, int]]:
    try:
        st = cache_path.stat()
    except OSError:
        return None
    try:
        mtime_ns = int(
            getattr(st, "st_mtime_ns", int(getattr(st, "st_mtime", 0) * 1_000_000_000))
        )
    except Exception:
        mtime_ns = int(getattr(st, "st_mtime", 0) * 1_000_000_000)
    return (mtime_ns, int(getattr(st, "st_size", 0)))


def _server_managed_slug_from_path(path: Path) -> Optional[str]:
    base = path if path.is_dir() else path.parent
    try:
        parts = base.resolve().parts
    except OSError:
        parts = base.parts

    slug = next((seg for seg in reversed(parts) if _SLUGGED_REPO_RE.match(seg or "")), None)
    if not slug:
        return None

    with _managed_slug_cache_lock:
        if slug in _managed_slug_cache:
            return slug
        if slug in _managed_slug_cache_neg:
            return None

    work_dir = Path(os.environ.get("WORK_DIR") or os.environ.get("WORKDIR") or "/work")
    marker = work_dir / ".codebase" / "repos" / slug / ".ctxce_managed_upload"
    try:
        is_managed = marker.exists()
    except OSError:
        is_managed = False

    with _managed_slug_cache_lock:
        if is_managed:
            _managed_slug_cache.add(slug)
        else:
            _managed_slug_cache_neg.add(slug)

    return slug if is_managed else None

# Type definitions
IndexingState = Literal['idle', 'initializing', 'scanning', 'indexing', 'watching', 'error']
ActivityAction = Literal['indexed', 'deleted', 'skipped', 'scan-completed', 'initialized', 'moved']

# Constants
STATE_DIRNAME = ".codebase"
STATE_FILENAME = "state.json"
CACHE_FILENAME = "cache.json"
PLACEHOLDER_COLLECTION_NAMES = {"", "default-collection", "my-collection"}

class IndexingProgress(TypedDict, total=False):
    files_processed: int
    total_files: Optional[int]
    current_file: Optional[str]

class IndexingStatus(TypedDict, total=False):
    state: IndexingState
    started_at: Optional[str]
    progress: Optional[IndexingProgress]
    error: Optional[str]

class ActivityDetails(TypedDict, total=False):
    block_count: Optional[int]
    reason: Optional[str]
    files_processed: Optional[int]
    total_blocks: Optional[int]
    git_commit: Optional[str]
    git_branch: Optional[str]
    chunk_count: Optional[int]
    file_size: Optional[int]

class LastActivity(TypedDict, total=False):
    timestamp: str
    action: ActivityAction
    file_path: Optional[str]
    details: Optional[ActivityDetails]

class OriginInfo(TypedDict, total=False):
    repo_name: Optional[str]
    container_path: Optional[str]
    source_path: Optional[str]
    collection_name: Optional[str]
    updated_at: Optional[str]


class StagingInfo(TypedDict, total=False):
    collection: Optional[str]
    status: Optional[IndexingStatus]
    started_at: Optional[str]
    updated_at: Optional[str]
    env_hash: Optional[str]
    environment: Optional[Dict[str, str]]
    indexing_config: Optional[Dict[str, Any]]
    indexing_config_hash: Optional[str]
    workspace_path: Optional[str]
    repo_name: Optional[str]


class WorkspaceState(TypedDict, total=False):
    created_at: str
    updated_at: str
    qdrant_collection: str
    indexing_status: Optional[IndexingStatus]
    last_activity: Optional[LastActivity]
    qdrant_stats: Optional[Dict[str, Any]]
    origin: Optional[OriginInfo]
    logical_repo_id: Optional[str]
    indexing_config: Optional[Dict[str, Any]]
    indexing_config_hash: Optional[str]
    indexing_env: Optional[Dict[str, str]]
    indexing_config_pending: Optional[Dict[str, Any]]
    indexing_config_pending_hash: Optional[str]
    indexing_env_pending: Optional[Dict[str, str]]
    previous_collection: Optional[str]
    serving_collection: Optional[str]
    active_repo_slug: Optional[str]
    serving_repo_slug: Optional[str]
    staging: Optional[StagingInfo]

def is_multi_repo_mode() -> bool:
    """Check if multi-repo mode is enabled."""
    return os.environ.get("MULTI_REPO_MODE", "0").strip().lower() in {
        "1", "true", "yes", "on"
    }


def logical_repo_reuse_enabled() -> bool:
    """Feature flag for logical-repo / collection reuse.

    Controlled by LOGICAL_REPO_REUSE env var: 1/true/yes/on => enabled.
    When disabled, behavior falls back to legacy per-repo collection logic
    and does not write logical_repo_id into workspace state.
    """
    return os.environ.get("LOGICAL_REPO_REUSE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

_state_lock = threading.Lock()
# Track last-used timestamps for cleanup of idle workspace locks
_state_locks: Dict[str, threading.RLock] = {}
_state_lock_last_used: Dict[str, float] = {}

def _resolve_workspace_root() -> str:
    """Determine the default workspace root path."""
    return os.environ.get("WORKSPACE_PATH") or os.environ.get("WATCH_ROOT") or "/work"

def _resolve_repo_context(
    workspace_path: Optional[str] = None,
    repo_name: Optional[str] = None,
) -> tuple[str, Optional[str]]:
    """Normalize workspace/repo context, ensuring multi-repo callers map to repo state."""
    resolved_workspace = workspace_path or _resolve_workspace_root()

    if is_multi_repo_mode():
        if repo_name:
            return resolved_workspace, repo_name

        if workspace_path:
            detected = _detect_repo_name_from_path(Path(workspace_path))
            if detected:
                return resolved_workspace, detected

        return resolved_workspace, None

    return resolved_workspace, repo_name

def _get_state_lock(workspace_path: Optional[str] = None, repo_name: Optional[str] = None) -> threading.RLock:
    """Get or create a lock for the workspace or repo state and track usage."""
    if repo_name and is_multi_repo_mode():
        key = f"repo::{repo_name}"
    else:
        key = str(Path(workspace_path or _resolve_workspace_root()).resolve())

    with _state_lock:
        if key not in _state_locks:
            _state_locks[key] = threading.RLock()
        _state_lock_last_used[key] = time.time()
        return _state_locks[key]

def _get_repo_state_dir(repo_name: str) -> Path:
    """Get the state directory for a repository."""
    base_dir = Path(os.environ.get("WORKSPACE_PATH") or os.environ.get("WATCH_ROOT") or "/work")
    if is_multi_repo_mode():
        return base_dir / STATE_DIRNAME / "repos" / repo_name
    return base_dir / STATE_DIRNAME

def _get_state_path(workspace_path: str) -> Path:
    """Get the path to the state.json file for a workspace."""
    workspace = Path(workspace_path).resolve()
    state_dir = workspace / STATE_DIRNAME
    return state_dir / STATE_FILENAME


def _get_global_state_dir(workspace_path: Optional[str] = None) -> Path:
    """Return the root .codebase directory used for workspace metadata."""

    base_dir = Path(workspace_path or _resolve_workspace_root()).resolve()
    return base_dir / STATE_DIRNAME

def _ensure_state_dir(workspace_path: str) -> Path:
    """Ensure the .codebase directory exists and return the state file path."""
    workspace = Path(workspace_path).resolve()
    state_dir = workspace / STATE_DIRNAME
    state_dir.mkdir(exist_ok=True)
    return state_dir / STATE_FILENAME

def _sanitize_name(s: str, max_len: int = 64) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9_.-]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    if not s:
        s = "workspace"
    return s[:max_len]


def _detect_git_common_dir(start: Path) -> Optional[Path]:
    try:
        base = start if start.is_dir() else start.parent
        r = subprocess.run(
            ["git", "-C", str(base), "rev-parse", "--git-common-dir"],
            capture_output=True,
            text=True,
        )
        raw = (r.stdout or "").strip()
        if r.returncode != 0 or not raw:
            return None
        p = Path(raw)
        if not p.is_absolute():
            p = base / p
        return p.resolve()
    except Exception:
        return None


def compute_logical_repo_id(workspace_path: str) -> str:
    try:
        p = Path(workspace_path).resolve()
    except Exception:
        p = Path(workspace_path)

    common = _detect_git_common_dir(p)
    if common is not None:
        key = str(common)
        prefix = "git:"
    else:
        key = str(p)
        prefix = "fs:"

    h = hashlib.sha1(key.encode("utf-8", errors="ignore")).hexdigest()[:16]
    return f"{prefix}{h}"


def ensure_logical_repo_id(state: WorkspaceState, workspace_path: str) -> WorkspaceState:
    if not isinstance(state, dict):
        return state
    if not logical_repo_reuse_enabled():
        # Gate: when logical repo reuse is disabled, leave state untouched
        return state
    if state.get("logical_repo_id"):
        return state
    lrid = compute_logical_repo_id(workspace_path)
    state["logical_repo_id"] = lrid
    origin = dict(state.get("origin", {}) or {})
    origin.setdefault("logical_repo_id", lrid)
    state["origin"] = origin
    return state


# Cross-process file locking (POSIX fcntl), falls back to no-op if unavailable
try:
    import fcntl  # type: ignore
except Exception:  # pragma: no cover
    fcntl = None  # type: ignore

from contextlib import contextmanager

@contextmanager
def _cross_process_lock(lock_path: Path):
    """Advisory cross-process exclusive lock using a companion .lock file.
    Safe across container/process boundaries; pairs with atomic rename writes.
    Ensures group-writable permissions so non-root indexers/watchers can operate.
    """

    lock_path.parent.mkdir(parents=True, exist_ok=True)

    lock_file = None
    fd = None
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o664)
        lock_file = os.fdopen(fd, "a+")
    except PermissionError:
        # If we cannot create or open the requested lock, fall back to /tmp (permissive)
        tmp_path = Path("/tmp") / (lock_path.name)
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(tmp_path, os.O_CREAT | os.O_RDWR, 0o664)
        lock_file = os.fdopen(fd, "a+")
        lock_path = tmp_path

    try:
        try:
            os.chmod(lock_path, 0o664)
        except PermissionError:
            pass

        if fcntl is not None:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            except Exception:
                pass
        yield
    finally:
        try:
            if fcntl is not None:
                try:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                except Exception:
                    pass
        finally:
            try:
                lock_file.close()
            except Exception:
                pass


# Per-file locking for indexer/watcher coordination
# Uses /work/.codebase/file_locks (shared volume) for cross-container coordination in Docker
# Falls back to /tmp for local development
_SHARED_LOCK_DIR = Path("/work/.codebase")
# ALWAYS create the locks dir if /work exists - don't check if .codebase exists yet
# (it might not exist at import time but will be created during indexing)
if Path("/work").exists():
    _FILE_LOCKS_DIR = _SHARED_LOCK_DIR / "file_locks"
    try:
        _FILE_LOCKS_DIR.mkdir(parents=True, exist_ok=True)
    except OSError:
        # Read-only filesystem (e.g. Docker bind mount), fall back to /tmp
        _FILE_LOCKS_DIR = Path("/tmp/context-engine-locks")
        _FILE_LOCKS_DIR.mkdir(parents=True, exist_ok=True)
else:
    _FILE_LOCKS_DIR = Path("/tmp/context-engine-locks")
    _FILE_LOCKS_DIR.mkdir(parents=True, exist_ok=True)

# Lock timeout - if lock file is older than this, consider it stale
_FILE_LOCK_TIMEOUT_SECONDS = 300  # 5 min max per file (generous for LLM calls)


def _get_file_lock_path(file_path: str) -> Path:
    """Get the lock file path for a given file."""
    # Use hash of file path to avoid filesystem path issues
    import hashlib
    path_hash = hashlib.md5(file_path.encode()).hexdigest()[:16]
    return _FILE_LOCKS_DIR / f"{path_hash}.lock"


def is_file_locked(file_path: str) -> bool:
    """Check if a specific file is currently being indexed.

    Uses file-based locking with timestamp for cross-container coordination.
    Returns True if file is locked (lock file exists and is recent).
    """
    try:
        lock_path = _get_file_lock_path(file_path)
        if not lock_path.exists():
            return False
        # Check if lock is stale
        mtime = lock_path.stat().st_mtime
        age = time.time() - mtime
        if age > _FILE_LOCK_TIMEOUT_SECONDS:
            # Stale lock - remove it
            try:
                lock_path.unlink()
            except Exception:
                pass
            return False
        return True
    except Exception:
        return False


@contextmanager
def file_indexing_lock(file_path: str):
    """Acquire a lock for indexing a specific file.

    Use this before processing a file to prevent indexer/watcher collision.
    Non-blocking - raises FileExistsError if file is already locked.

    Uses O_CREAT | O_EXCL for atomic lock acquisition to prevent race conditions
    where two processes could both pass is_file_locked() and overwrite each other.
    """
    lock_path = _get_file_lock_path(file_path)
    fd = None

    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)

        # First check if there's a stale lock we should clean up
        if lock_path.exists():
            try:
                mtime = lock_path.stat().st_mtime
                age = time.time() - mtime
                if age > _FILE_LOCK_TIMEOUT_SECONDS:
                    # Stale lock - try to remove it
                    lock_path.unlink()
                else:
                    # Fresh lock held by another process
                    raise FileExistsError(f"File is already being indexed: {file_path}")
            except FileNotFoundError:
                # Lock was removed between exists() and stat() - continue to acquire
                pass

        # Atomic lock acquisition: O_CREAT | O_EXCL fails if file exists
        # This prevents race condition where two processes both pass the check above
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            # Another process created the lock between our check and open
            raise FileExistsError(f"File is already being indexed: {file_path}")

        # Write lock metadata
        lock_data = json.dumps({
            "file": file_path,
            "locked_at": time.time(),
            "pid": os.getpid(),
        })
        os.write(fd, lock_data.encode())
        os.close(fd)
        fd = None

    except FileExistsError:
        raise
    except Exception as e:
        if fd is not None:
            try:
                os.close(fd)
            except Exception:
                pass
        raise RuntimeError(f"Could not acquire file lock: {e}")

    try:
        yield
    finally:
        # Release lock
        try:
            lock_path.unlink()
        except Exception:
            pass


# Legacy global lock for backward compatibility (deprecated)
INDEXING_LOCK_PATH = _FILE_LOCKS_DIR.parent / "indexing.lock" if _SHARED_LOCK_DIR.exists() else Path("/tmp/context-engine-indexing.lock")


def is_indexing_locked() -> bool:
    """DEPRECATED: Use is_file_locked() for per-file coordination.

    Check if global indexing lock is held. Returns False (no global lock).
    """
    return False  # Per-file locking means no global lock needed


@contextmanager
def indexing_lock():
    """DEPRECATED: Use file_indexing_lock() for per-file coordination.

    No-op for backward compatibility.
    """
    yield

def _git_remote_repo_name(repo_path: Path) -> Optional[str]:
    """Return canonical repo name from git remote origin URL or toplevel."""
    try:
        r = subprocess.run(
            ["git", "-C", str(repo_path), "config", "--get", "remote.origin.url"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if r.returncode == 0 and r.stdout.strip():
            url = r.stdout.strip()
            name = url.rstrip("/").rsplit("/", 1)[-1]
            if name.endswith(".git"):
                name = name[:-4]
            if name:
                return name
    except Exception:
        pass

    try:
        r = subprocess.run(
            ["git", "-C", str(repo_path), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        top = (r.stdout or "").strip()
        if r.returncode == 0 and top:
            return Path(top).name
    except Exception:
        pass
    return None


def _detect_repo_name_from_path(path: Path) -> str:
    """Detect repository name from path using git remote origin URL.

    This ensures consistency with how the MCP server detects repos during search.
    Priority:
    1. Fast-path for server-managed uploads and workspace-relative paths
    2. Git remote origin URL (canonical repo name like 'Context-Engine')
    3. Git toplevel directory name (folder name like 'Context-Engine-hash')
    4. Walk up to find .git and return that folder name
    5. Return parent folder name as fallback
    """
    slug = _server_managed_slug_from_path(path)
    if slug:
        return slug

    # Fast-path for managed upload workspaces or when workspace_path == /work:
    # derive the repo name from the first path segment relative to the workspace
    # root instead of spawning git processes or falling back to "work".
    try:
        ws_root = Path(_resolve_workspace_root()).resolve()
    except Exception:
        ws_root = Path(_resolve_workspace_root())

    try:
        resolved = path.resolve()
    except Exception:
        resolved = path if path.is_dir() else path.parent

    try:
        rel = resolved.relative_to(ws_root)
        if rel.parts:
            candidate = rel.parts[0]
            if candidate not in {".codebase", ".git", "__pycache__"}:
                return candidate
    except Exception:
        pass

    try:
        base = path if path.is_dir() else path.parent
        git_name = _git_remote_repo_name(base)
        if git_name:
            return git_name
    except Exception:
        pass
    try:
        # Walk up to find .git
        cur = path if path.is_dir() else path.parent
        for p in [cur] + list(cur.parents):
            try:
                if (p / ".git").exists():
                    return p.name
            except Exception:
                continue
    except Exception:
        pass

    try:
        structure_name = _detect_repo_name_from_path_by_structure(path)
        if structure_name:
            return structure_name
    except Exception:
        pass

    return (path if path.is_dir() else path.parent).name or "workspace"


def _generate_collection_name(workspace_path: str) -> str:
    ws = Path(workspace_path).resolve()
    repo = _sanitize_name(_detect_repo_name_from_path(ws))
    # stable suffix from absolute path
    h = hashlib.sha1(str(ws).encode("utf-8", errors="ignore")).hexdigest()[:6]
    return _sanitize_name(f"{repo}-{h}")

def _atomic_write_state(state_path: Path, state: WorkspaceState) -> None:
    """Atomically write state to prevent corruption during concurrent access."""
    # Write to temp file first, then rename (atomic on most filesystems)
    temp_path = state_path.with_suffix(f".tmp.{uuid.uuid4().hex[:8]}")
    try:
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        temp_path.replace(state_path)
        # Ensure state/cache files are group-writable so multiple processes
        # (upload service, watcher, indexer) can update them.
        try:
            os.chmod(state_path, 0o664)
        except PermissionError:
            pass
    except Exception:
        # Clean up temp file if something went wrong
        try:
            temp_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise

def get_workspace_state(
    workspace_path: Optional[str] = None, repo_name: Optional[str] = None
) -> WorkspaceState:
    """Get the current workspace state, creating it if it doesn't exist."""

    workspace_path, repo_name = _resolve_repo_context(workspace_path, repo_name)

    if is_multi_repo_mode() and repo_name is None:
        print(
            f"[workspace_state] Multi-repo: Skipping state read for workspace={workspace_path} without repo_name"
        )
        return {}

    lock = _get_state_lock(workspace_path, repo_name)
    with lock:
        state_path: Path
        lock_scope_path: Path

        if is_multi_repo_mode() and repo_name:
            state_dir = _get_repo_state_dir(repo_name)
            try:
                ws_root = Path(_resolve_workspace_root())
                ws_dir = ws_root / repo_name
            except Exception:
                ws_dir = None
            try:
                if not state_dir.exists() and (ws_dir is None or not ws_dir.exists()):
                    return {}
            except Exception:
                return {}
            state_dir.mkdir(parents=True, exist_ok=True)
            # Ensure repo state dir is group-writable so root upload service and
            # non-root watcher/indexer processes can both write state/cache files.
            try:
                os.chmod(state_dir, 0o775)
            except Exception:
                pass
            state_path = state_dir / STATE_FILENAME
            lock_scope_path = state_dir
        else:
            try:
                state_path = _ensure_state_dir(workspace_path)
                lock_scope_path = state_path.parent
            except PermissionError:
                lock_scope_path = _get_global_state_dir(workspace_path)
                lock_scope_path.mkdir(parents=True, exist_ok=True)
                state_path = lock_scope_path / STATE_FILENAME

        lock_path = lock_scope_path / (STATE_FILENAME + ".lock")
        with _cross_process_lock(lock_path):
            if state_path.exists():
                try:
                    with open(state_path, "r", encoding="utf-8-sig") as f:
                        state = json.load(f)
                    if isinstance(state, dict):
                        if logical_repo_reuse_enabled():
                            workspace_real = str(Path(workspace_path or _resolve_workspace_root()).resolve())
                            state = ensure_logical_repo_id(state, workspace_real)
                            try:
                                _atomic_write_state(state_path, state)
                            except Exception as e:
                                print(f"[workspace_state] Failed to persist logical_repo_id to {state_path}: {e}")
                        modified = _ensure_repo_slug_defaults(state, repo_name)
                        if modified:
                            try:
                                _atomic_write_state(state_path, state)
                            except Exception:
                                pass
                        return state
                except (json.JSONDecodeError, ValueError, OSError) as e:
                    print(f"[workspace_state] Failed to read state from {state_path}: {e}")

            now = datetime.now().isoformat()
            collection_name = get_collection_name(repo_name)

            state: WorkspaceState = {
                "workspace_path": str(Path(workspace_path or _resolve_workspace_root()).resolve()),
                "created_at": now,
                "updated_at": now,
                "qdrant_collection": collection_name,
                "indexing_status": {"state": "idle"},
            }
            _ensure_repo_slug_defaults(state, repo_name)

            if logical_repo_reuse_enabled():
                try:
                    state = ensure_logical_repo_id(state, state.get("workspace_path", workspace_path or _resolve_workspace_root()))
                except Exception as e:
                    print(f"[workspace_state] Failed to ensure logical_repo_id for {workspace_path}: {e}")

            _atomic_write_state(state_path, state)
            return state


def update_workspace_state(
    workspace_path: Optional[str] = None,
    updates: Optional[Dict[str, Any]] = None,
    repo_name: Optional[str] = None,
) -> WorkspaceState:
    """Update workspace state with the given changes."""

    workspace_path, repo_name = _resolve_repo_context(workspace_path, repo_name)
    updates = updates or {}

    if is_multi_repo_mode() and repo_name is None:
        print(
            f"[workspace_state] Multi-repo: Skipping state update for workspace={workspace_path} without repo_name"
        )
        return {}

    if is_multi_repo_mode() and repo_name:
        try:
            ws_root = Path(_resolve_workspace_root())
            # Allow updates when the repo state dir exists, even if the workspace
            # directory is not present (e.g. dev-remote simulations where only
            # .codebase state is persisted).
            state_dir = _get_repo_state_dir(repo_name)
            if not (ws_root / repo_name).exists() and not state_dir.exists():
                return {}
        except Exception:
            return {}


    lock = _get_state_lock(workspace_path, repo_name)
    with lock:
        state = get_workspace_state(workspace_path, repo_name)
        if not state:
            return {}

        for key, value in updates.items():
            if key in state or key in WorkspaceState.__annotations__:
                state[key] = value

        _ensure_repo_slug_defaults(state, repo_name)
        state["updated_at"] = datetime.now().isoformat()

        if is_multi_repo_mode() and repo_name:
            state_dir = _get_repo_state_dir(repo_name)
            state_dir.mkdir(parents=True, exist_ok=True)
            state_path = state_dir / STATE_FILENAME
        else:
            try:
                state_path = _ensure_state_dir(workspace_path)
            except PermissionError:
                state_dir = _get_global_state_dir(workspace_path)
                state_dir.mkdir(parents=True, exist_ok=True)
                state_path = state_dir / STATE_FILENAME

        _atomic_write_state(state_path, state)
        return state


def initialize_watcher_state(
    workspace_path: str,
    multi_repo_enabled: bool,
    default_collection: str
) -> None:
    """Initialize workspace state for the watcher."""
    if multi_repo_enabled:
        root_repo_name = _extract_repo_name_from_path(workspace_path)
        if root_repo_name:
            root_collection = get_collection_name(root_repo_name)
            try:
                if persist_indexing_config:
                    persist_indexing_config(
                        workspace_path=workspace_path,
                        repo_name=root_repo_name,
                        pending=True,
                    )
            except Exception:
                pass
            update_indexing_status(
                repo_name=root_repo_name,
                status={"state": "watching"},
            )
            print(
                f"[workspace_state] Initialized repo state: {root_repo_name} -> {root_collection}"
            )
        else:
            print(
                "[workspace_state] Multi-repo: root path is not a repo; skipping state initialization"
            )
    else:
        updates = {"qdrant_collection": default_collection}
        try:
            if get_indexing_config_snapshot and compute_indexing_config_hash:
                cfg = get_indexing_config_snapshot()
                updates["indexing_config"] = cfg
                updates["indexing_config_hash"] = compute_indexing_config_hash(cfg)
        except Exception:
            pass
        update_workspace_state(workspace_path=workspace_path, updates=updates)
        try:
            if persist_indexing_config:
                persist_indexing_config(
                    workspace_path=workspace_path,
                    repo_name=None,
                    pending=True,
                )
        except Exception:
            pass
        update_indexing_status(status={"state": "watching"})


def set_indexing_started(workspace_path: str, total_files: int) -> None:
    """Set status to indexing with start time."""
    try:
        repo_name = _extract_repo_name_from_path(Path(workspace_path))
        update_indexing_status(
            repo_name=repo_name,
            status={
                "state": "indexing",
                "started_at": datetime.now().isoformat(),
                "progress": {"files_processed": 0, "total_files": int(total_files)},
            },
        )
    except Exception:
        pass


def set_indexing_progress(
    workspace_path: str,
    started_at: str,
    processed: int,
    total: int,
    current_file: Optional[str],
) -> None:
    """Update indexing progress."""
    try:
        repo_name = _extract_repo_name_from_path(Path(workspace_path))
        update_indexing_status(
            repo_name=repo_name,
            status={
                "state": "indexing",
                "started_at": started_at,
                "progress": {
                    "files_processed": int(processed),
                    "total_files": int(total),
                    "current_file": str(current_file) if current_file else None,
                },
            },
        )
    except Exception:
        pass


def log_watcher_activity(
    workspace_path: str, action: str, file_path: str, details: Dict | None = None
) -> None:
    """Log watcher activity with validation."""
    try:
        repo_name = _extract_repo_name_from_path(Path(workspace_path))

        valid_actions = {
            "indexed",
            "deleted",
            "skipped",
            "scan-completed",
            "initialized",
            "moved",
        }
        if action not in valid_actions:
            action = "indexed"

        log_activity(
            repo_name=repo_name,
            action=action,  # type: ignore[arg-type]
            file_path=str(file_path),
            details=details,
        )
    except Exception:
        pass

def update_indexing_status(
    workspace_path: Optional[str] = None,
    status: Optional[IndexingStatus] = None,
    repo_name: Optional[str] = None,
) -> WorkspaceState:
    """Update indexing status in workspace state."""
    workspace_path, repo_name = _resolve_repo_context(workspace_path, repo_name)

    if is_multi_repo_mode() and repo_name is None:
        print(
            f"[workspace_state] Multi-repo: Skipping indexing status update for workspace={workspace_path} without repo_name"
        )
        return {}

    if status is None:
        status = {"state": "idle"}

    return update_workspace_state(
        workspace_path=workspace_path,
        updates={"indexing_status": status},
        repo_name=repo_name,
    )


def set_staging_state(
    *,
    workspace_path: Optional[str] = None,
    repo_name: Optional[str] = None,
    staging: Optional[StagingInfo] = None,
) -> WorkspaceState:
    """Persist staging metadata for a workspace/repo."""
    updates: Dict[str, Any] = {"staging": staging}
    if staging:
        staging.setdefault("updated_at", datetime.now().isoformat())
    return update_workspace_state(
        workspace_path=workspace_path,
        repo_name=repo_name,
        updates=updates,
    )


def promote_pending_indexing_config(
    *,
    workspace_path: Optional[str] = None,
    repo_name: Optional[str] = None,
) -> WorkspaceState:
    """Promote pending indexing config/env snapshots to active."""
    state = get_workspace_state(workspace_path, repo_name) or {}
    pending_cfg = state.get("indexing_config_pending")
    pending_hash = state.get("indexing_config_pending_hash")
    pending_env = state.get("indexing_env_pending")

    if not pending_cfg and not pending_env:
        return state

    cfg = pending_cfg or state.get("indexing_config") or get_indexing_config_snapshot()
    cfg_hash = pending_hash or compute_indexing_config_hash(cfg)
    env_snapshot = pending_env or state.get("indexing_env") or dict(os.environ)

    return persist_indexing_config(
        workspace_path=workspace_path,
        repo_name=repo_name,
        config=cfg,
        config_hash=cfg_hash,
        environment=env_snapshot,
        pending=False,
    )


def get_collection_state_snapshot(
    *,
    workspace_path: Optional[str],
    repo_name: Optional[str],
) -> Dict[str, Any]:
    """Return current active + staging collection metadata for bridge/search consumers."""
    state = get_workspace_state(workspace_path, repo_name) or {}
    if not isinstance(state, dict):
        state = {}

    resolved_workspace = state.get("workspace_path") or workspace_path
    active_collection = state.get("qdrant_collection")
    serving_collection = state.get("serving_collection") or active_collection
    active_repo_slug = state.get("active_repo_slug") or repo_name
    serving_repo_slug = state.get("serving_repo_slug") or active_repo_slug

    snapshot: Dict[str, Any] = {
        "workspace_path": resolved_workspace,
        "repo_name": repo_name,
        "active_collection": active_collection,
        "serving_collection": serving_collection,
        "previous_collection": state.get("previous_collection"),
        "active_repo_slug": active_repo_slug,
        "serving_repo_slug": serving_repo_slug,
        "indexing_status": state.get("indexing_status"),
        "staging": None,
    }

    staging_enabled = bool(is_staging_enabled() if callable(is_staging_enabled) else False)
    staging_info = state.get("staging")
    if staging_enabled and isinstance(staging_info, dict) and staging_info.get("collection"):
        snapshot["staging"] = {
            "collection": staging_info.get("collection"),
            "status": staging_info.get("status"),
            "started_at": staging_info.get("started_at"),
            "updated_at": staging_info.get("updated_at"),
            "env_hash": staging_info.get("env_hash"),
            "workspace_path": staging_info.get("workspace_path"),
            "repo_name": staging_info.get("repo_name") or repo_name,
        }

    return snapshot


def get_staging_targets(
    *,
    workspace_path: Optional[str],
    repo_name: Optional[str],
) -> Dict[str, Any]:
    """Return canonical/old slug hints plus staging metadata if staging is active."""

    snapshot = get_collection_state_snapshot(workspace_path=workspace_path, repo_name=repo_name)
    if not snapshot:
        return {}

    active_slug = snapshot.get("active_repo_slug") or repo_name
    serving_slug = snapshot.get("serving_repo_slug") or active_slug
    staging_info = snapshot.get("staging")

    canonical_slug: Optional[str] = None
    if isinstance(active_slug, str) and active_slug.strip():
        canonical_slug = active_slug[:-4] if active_slug.endswith("_old") else active_slug
    elif isinstance(serving_slug, str) and serving_slug.strip():
        canonical_slug = serving_slug[:-4] if serving_slug.endswith("_old") else serving_slug
    elif repo_name:
        canonical_slug = str(repo_name)

    result: Dict[str, Any] = {
        "workspace_path": snapshot.get("workspace_path") or workspace_path,
        "repo_name": repo_name,
        "active_slug": active_slug,
        "serving_slug": serving_slug,
        "staging": staging_info if isinstance(staging_info, dict) else None,
    }

    if canonical_slug:
        result["canonical_slug"] = canonical_slug
        result["old_slug"] = f"{canonical_slug}_old"

    return result


def update_staging_status(
    *,
    workspace_path: Optional[str],
    repo_name: Optional[str],
    status: IndexingStatus,
) -> WorkspaceState:
    state = get_workspace_state(workspace_path, repo_name)
    staging = dict(state.get("staging", {}) or {})
    if not staging:
        return state
    staging["status"] = status
    staging["updated_at"] = datetime.now().isoformat()
    return set_staging_state(
        workspace_path=workspace_path,
        repo_name=repo_name,
        staging=staging,
    )


def clear_staging_collection(
    *,
    workspace_path: Optional[str],
    repo_name: Optional[str],
) -> WorkspaceState:
    return set_staging_state(workspace_path=workspace_path, repo_name=repo_name, staging=None)


def activate_staging_collection(
    *,
    workspace_path: Optional[str],
    repo_name: Optional[str],
) -> WorkspaceState:
    state = get_workspace_state(workspace_path, repo_name)
    staging = dict(state.get("staging", {}) or {})
    collection = staging.get("collection")
    if not collection:
        return state

    updates: Dict[str, Any] = {
        "previous_collection": state.get("qdrant_collection"),
        "qdrant_collection": collection,
        "staging": None,
    }

    status = staging.get("status")
    if isinstance(status, dict):
        updates["indexing_status"] = status

    if staging.get("indexing_config"):
        updates["indexing_config"] = staging.get("indexing_config")
    if staging.get("indexing_config_hash"):
        updates["indexing_config_hash"] = staging.get("indexing_config_hash")
    if staging.get("environment"):
        updates["indexing_env"] = staging.get("environment")
    # Clear pending snapshots once activation succeeds
    updates["indexing_config_pending"] = None
    updates["indexing_config_pending_hash"] = None
    updates["indexing_env_pending"] = None

    return update_workspace_state(
        workspace_path=workspace_path,
        repo_name=repo_name,
        updates=updates,
    )


def update_repo_origin(
    workspace_path: Optional[str] = None,
    repo_name: Optional[str] = None,
    *,
    container_path: Optional[str] = None,
    source_path: Optional[str] = None,
    collection_name: Optional[str] = None,
) -> WorkspaceState:
    """Update origin metadata for a repository/workspace."""

    resolved_workspace, resolved_repo = _resolve_repo_context(workspace_path, repo_name)

    if is_multi_repo_mode() and resolved_repo is None:
        return {}

    state = get_workspace_state(resolved_workspace, resolved_repo)
    if not state:
        state = {}

    origin: OriginInfo = dict(state.get("origin", {}))  # type: ignore[arg-type]
    if resolved_repo:
        origin["repo_name"] = resolved_repo
    if container_path or workspace_path:
        origin["container_path"] = container_path or workspace_path
    if source_path:
        origin["source_path"] = source_path
    if collection_name:
        origin["collection_name"] = collection_name
    origin["updated_at"] = datetime.now().isoformat()

    updates: Dict[str, Any] = {"origin": origin}
    if collection_name:
        updates.setdefault("qdrant_collection", collection_name)

    return update_workspace_state(
        workspace_path=resolved_workspace,
        updates=updates,
        repo_name=resolved_repo,
    )


def log_activity(
    repo_name: Optional[str] = None,
    action: Optional[ActivityAction] = None,
    file_path: Optional[str] = None,
    details: Optional[ActivityDetails] = None,
    workspace_path: Optional[str] = None,
) -> None:
    """Log activity to workspace state."""

    if not action:
        return

    activity = {
        "timestamp": datetime.now().isoformat(),
        "action": action,
        "file_path": file_path,
        "details": details or {},
    }

    resolved_workspace = workspace_path or _resolve_workspace_root()

    if is_multi_repo_mode() and repo_name:
        try:
            ws_root = Path(_resolve_workspace_root())
            if not (ws_root / repo_name).exists():
                return
        except Exception:
            return
        state_dir = _get_repo_state_dir(repo_name)
        state_dir.mkdir(parents=True, exist_ok=True)
        state_path = state_dir / STATE_FILENAME
        lock_path = state_path.with_suffix(".lock")

        with _cross_process_lock(lock_path):
            try:
                if state_path.exists():
                    with open(state_path, "r", encoding="utf-8-sig") as f:
                        state = json.load(f)
                else:
                    state = {"created_at": datetime.now().isoformat()}
            except Exception:
                state = {"created_at": datetime.now().isoformat()}

            state["last_activity"] = activity
            state["updated_at"] = datetime.now().isoformat()
            _atomic_write_state(state_path, state)
    else:
        update_workspace_state(
            workspace_path=resolved_workspace,
            updates={"last_activity": activity},
            repo_name=repo_name,
        )


def _generate_collection_name_from_repo(repo_name: str) -> str:
    """Generate collection name with 8-char hash for local workspaces.

    Used by local indexer/watcher. Remote uploads use 16+8 char pattern
    for collision avoidance when folder names may be identical.
    """
    hash_obj = hashlib.sha256(repo_name.encode())
    short_hash = hash_obj.hexdigest()[:8]
    return f"{repo_name}-{short_hash}"

def _normalize_repo_name_for_collection(repo_name: str) -> str:
    """Normalize repo identifier to a stable base name for collection naming.

    In multi-repo remote mode, repo_name may be a slug like "name-<16hex>" used
    for folder collision avoidance. For Qdrant collections we always want the
    base repo directory name, so strip a trailing 16-hex segment when present.
    """
    try:
        # Special-case staging clone slugs ("..._old"): we still want to strip the
        # remote-upload 16-hex suffix from the *base* repo name.
        is_old = False
        raw = repo_name
        try:
            if raw.endswith("_old"):
                is_old = True
                raw = raw[:-4]
        except Exception:
            is_old = False
            raw = repo_name

        m = re.match(r"^(.*)-([0-9a-f]{16})$", raw)
        if m:
            base = (m.group(1) or "").strip()
            if base:
                return f"{base}_old" if is_old else base
    except Exception:
        pass
    return repo_name


def _collection_name_for_repo_slug(normalized_repo: str, *, is_old_slug: bool) -> Optional[str]:
    if not normalized_repo:
        return None

    # If repo_name is a staging clone slug ("..._old"), compute the canonical collection name
    # from the base repo and then append "_old".
    if is_old_slug:
        try:
            base_repo = normalized_repo[:-4] if normalized_repo.endswith("_old") else normalized_repo
            base_coll = _generate_collection_name_from_repo(base_repo)
            return f"{base_coll}_old"
        except Exception:
            return None

    if is_multi_repo_mode():
        return _generate_collection_name_from_repo(normalized_repo)

    return None


def get_collection_name(repo_name: Optional[str] = None) -> str:
    """Get collection name for repository or workspace.

    Priority:
    1. Explicit COLLECTION_NAME env var - master override when set to a real value
       (if repo_name is an *_old clone, append _old to the override unless already present)
    2. Derive from repo slug (including *_old suffix handling)
    3. Fallback: "global-collection"

    This ensures COLLECTION_NAME works as a master override in both local dev
    and container environments, while still allowing deterministic derivation
    from repo slugs (including staging clone slugs).
    """

    # COLLECTION_NAME always wins when explicitly set to a real value.
    env_coll = os.environ.get("COLLECTION_NAME", "").strip()
    if env_coll and env_coll not in PLACEHOLDER_COLLECTION_NAMES and not (is_multi_repo_mode() and repo_name):
        try:
            if isinstance(repo_name, str) and repo_name.endswith("_old") and not env_coll.endswith("_old"):
                return f"{env_coll}_old"
        except Exception:
            pass
        return env_coll

    normalized = _normalize_repo_name_for_collection(repo_name) if repo_name else None
    is_old_slug = False
    try:
        if isinstance(repo_name, str) and repo_name.endswith("_old"):
            is_old_slug = True
    except Exception:
        is_old_slug = False

    derived = None
    if normalized:
        derived = _collection_name_for_repo_slug(normalized, is_old_slug=is_old_slug)
    if derived:
        return derived

    # Default fallback
    return "global-collection"

def _detect_repo_name_from_path_by_structure(path: Path) -> str:
    """Detect repository name from path structure (fallback when git is unavailable)."""
    try:
        resolved_path = path.resolve()
    except Exception:
        return None

    candidate_roots: List[Path] = []
    for root_str in (
        os.environ.get("WATCH_ROOT"),
        os.environ.get("WORKSPACE_PATH"),
        "/work",
        os.environ.get("HOST_ROOT"),
    ):
        if not root_str:
            continue
        try:
            root_path = Path(root_str).resolve()
        except Exception:
            continue
        if root_path not in candidate_roots:
            candidate_roots.append(root_path)

    for base in candidate_roots:
        try:
            rel_path = resolved_path.relative_to(base)
        except ValueError:
            continue

        if not rel_path.parts:
            continue

        repo_name = rel_path.parts[0]
        if repo_name in (".codebase", ".git", "__pycache__"):
            continue

        repo_path = base / repo_name
        if repo_path.exists() or resolved_path == repo_path or str(resolved_path).startswith(str(repo_path) + os.sep):
            return repo_name

    return None

def _normalize_repo_slug(candidate: Optional[str]) -> Optional[str]:
    if not candidate:
        return None
    if _SLUGGED_REPO_RE.match(candidate):
        return candidate
    return None


def _extract_repo_name_from_path(workspace_path: str) -> str:
    """Extract repository slug or canonical name from workspace path.

    Accepts canonical slugs (repo-hash), `_old` slugs, and falls back to git remote name.
    """
    if not workspace_path:
        return ""

    try:
        path = Path(workspace_path).resolve()
    except Exception:
        path = Path(workspace_path)

    slug = _server_managed_slug_from_path(path)
    if slug:
        return slug

    try:
        repo_path = path if path.is_dir() else path.parent
        if (repo_path / ".git").exists():
            name = _git_remote_repo_name(repo_path)
            if name:
                return name
    except Exception:
        pass

    try:
        candidate = _normalize_repo_slug(path.name)
        if candidate:
            return candidate
    except Exception:
        pass

    try:
        candidate = _normalize_repo_slug(path.parent.name)
        if candidate:
            return candidate
    except Exception:
        pass

    return path.name


def _ensure_repo_slug_defaults(state: WorkspaceState, repo_name: Optional[str]) -> bool:
    modified = False
    active_slug = state.get("active_repo_slug")
    if not active_slug and repo_name:
        state["active_repo_slug"] = repo_name
        active_slug = repo_name
        modified = True
    serving_slug = state.get("serving_repo_slug")
    if not serving_slug:
        state["serving_repo_slug"] = active_slug or repo_name
        modified = True
    if not state.get("serving_collection") and state.get("qdrant_collection"):
        # serving_collection is primarily used for staging/migration workflows.
        # Avoid persisting a workspace-level placeholder collection (e.g. "codebase") into
        # per-repo state in multi-repo mode when staging is not enabled.
        allow = True
        try:
            if is_multi_repo_mode() and not is_staging_enabled():
                allow = False
        except Exception:
            pass

        if allow:
            try:
                qc = str(state.get("qdrant_collection") or "").strip()
            except Exception:
                qc = ""
            if qc and qc not in PLACEHOLDER_COLLECTION_NAMES:
                state["serving_collection"] = qc
                modified = True
    return modified

def _get_cache_path(workspace_path: str) -> Path:
    """Get the path to the cache.json file."""
    try:
        workspace = Path(os.path.abspath(workspace_path))
    except Exception:
        workspace = Path(workspace_path)
    return workspace / STATE_DIRNAME / CACHE_FILENAME


def _read_cache_file_uncached(cache_path: Path) -> Dict[str, Any]:
    if not cache_path.exists():
        now = datetime.now().isoformat()
        return {"file_hashes": {}, "created_at": now, "updated_at": now}
    try:
        with open(cache_path, "r", encoding="utf-8-sig") as f:
            obj = json.load(f)
            if isinstance(obj, dict) and isinstance(obj.get("file_hashes"), dict):
                return obj
    except (OSError, json.JSONDecodeError, ValueError):
        pass
    now = datetime.now().isoformat()
    return {"file_hashes": {}, "created_at": now, "updated_at": now}


def _read_cache_file_cached(cache_path: Path) -> Dict[str, Any]:
    key = str(cache_path)
    now = time.time()

    with _cache_memo_lock:
        last_check = _cache_memo_last_check.get(key, 0.0)
        if key in _cache_memo and (now - last_check) < _cache_memo_recheck_seconds():
            return _cache_memo[key]

    sig = _cache_file_sig(cache_path)
    with _cache_memo_lock:
        _cache_memo_last_check[key] = now
        if sig is not None and _cache_memo_sig.get(key) == sig and key in _cache_memo:
            return _cache_memo[key]

    obj = _read_cache_file_uncached(cache_path)
    with _cache_memo_lock:
        _cache_memo[key] = obj
        _cache_memo_sig[key] = sig or (-1, -1)
        return obj


def _read_cache_cached(workspace_path: str) -> Dict[str, Any]:
    return _read_cache_file_cached(_get_cache_path(workspace_path))


def _read_cache(workspace_path: str) -> Dict[str, Any]:
    """Read cache file, return empty dict if it doesn't exist or is invalid."""

    cache_path = _get_cache_path(workspace_path)
    return _read_cache_file_uncached(cache_path)


def _write_cache(workspace_path: str, cache: Dict[str, Any]) -> None:
    """Atomic write of cache file with cross-process locking."""

    lock = _get_state_lock(workspace_path)
    with lock:
        cache_path = _get_cache_path(workspace_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = cache_path.with_suffix(cache_path.suffix + ".lock")
        with _cross_process_lock(lock_path):
            tmp = cache_path.with_suffix(f".tmp.{uuid.uuid4().hex[:8]}")
            try:
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(cache, f, ensure_ascii=False, indent=2)
                tmp.replace(cache_path)
            finally:
                try:
                    tmp.unlink(missing_ok=True)
                except Exception:
                    pass


def get_cached_file_hash(file_path: str, repo_name: Optional[str] = None) -> str:
    """Get cached file hash for tracking changes."""
    if is_multi_repo_mode() and repo_name:
        state_dir = _get_repo_state_dir(repo_name)
        cache_path = state_dir / CACHE_FILENAME

        cache = _read_cache_file_cached(cache_path)
        file_hashes = cache.get("file_hashes", {})
        fp = _normalize_cache_key_path(file_path)
        val = file_hashes.get(fp, "")
        if isinstance(val, dict):
            return str(val.get("hash") or "")
        return str(val or "")
    else:
        cache = _read_cache_cached(_resolve_workspace_root())
        fp = _normalize_cache_key_path(file_path)
        val = cache.get("file_hashes", {}).get(fp, "")
        if isinstance(val, dict):
            return str(val.get("hash") or "")
        return str(val or "")

    return ""


def set_cached_file_hash(file_path: str, file_hash: str, repo_name: Optional[str] = None) -> None:
    """Set cached file hash for tracking changes."""
    fp = _normalize_cache_key_path(file_path)

    st_size: Optional[int] = None
    st_mtime: Optional[int] = None
    try:
        st = Path(file_path).stat()
        st_size = int(getattr(st, "st_size", 0))
        st_mtime = int(getattr(st, "st_mtime", 0))
    except Exception:
        st_size = None
        st_mtime = None

    if is_multi_repo_mode() and repo_name:
        try:
            ws_root = Path(_resolve_workspace_root())
            if not (ws_root / repo_name).exists():
                return
        except Exception:
            return
        state_dir = _get_repo_state_dir(repo_name)
        cache_path = state_dir / CACHE_FILENAME
        state_dir.mkdir(parents=True, exist_ok=True)

        if cache_path.exists():
            cache = _read_cache_file_cached(cache_path)
        else:
            cache = {"file_hashes": {}, "created_at": datetime.now().isoformat()}

        existing = cache.get("file_hashes", {}).get(fp)
        if isinstance(existing, dict) and st_size is not None and st_mtime is not None:
            if (
                str(existing.get("hash") or "") == str(file_hash or "")
                and int(existing.get("size") or 0) == int(st_size)
                and int(existing.get("mtime") or 0) == int(st_mtime)
            ):
                return

        entry: Any = file_hash
        try:
            if st_size is not None and st_mtime is not None:
                entry = {"hash": file_hash, "size": st_size, "mtime": st_mtime}
            else:
                st = Path(file_path).stat()
                entry = {
                    "hash": file_hash,
                    "size": int(getattr(st, "st_size", 0)),
                    "mtime": int(getattr(st, "st_mtime", 0)),
                }
        except OSError:
            pass

        cache.setdefault("file_hashes", {})[fp] = entry
        cache["updated_at"] = datetime.now().isoformat()

        _atomic_write_state(cache_path, cache)  # reuse atomic writer for files
        _memoize_cache_obj(cache_path, cache)
        return

    cache = _read_cache_cached(_resolve_workspace_root())
    existing = cache.get("file_hashes", {}).get(fp)
    if isinstance(existing, dict) and st_size is not None and st_mtime is not None:
        if (
            str(existing.get("hash") or "") == str(file_hash or "")
            and int(existing.get("size") or 0) == int(st_size)
            and int(existing.get("mtime") or 0) == int(st_mtime)
        ):
            return
    entry: Any = file_hash
    try:
        if st_size is not None and st_mtime is not None:
            entry = {"hash": file_hash, "size": st_size, "mtime": st_mtime}
        else:
            st = Path(file_path).stat()
            entry = {
                "hash": file_hash,
                "size": int(getattr(st, "st_size", 0)),
                "mtime": int(getattr(st, "st_mtime", 0)),
            }
    except OSError:
        pass
    cache.setdefault("file_hashes", {})[fp] = entry
    cache["updated_at"] = datetime.now().isoformat()
    _write_cache(_resolve_workspace_root(), cache)
    _memoize_cache_obj(_get_cache_path(_resolve_workspace_root()), cache)


def get_cached_file_meta(file_path: str, repo_name: Optional[str] = None) -> Dict[str, Any]:
    fp = _normalize_cache_key_path(file_path)
    if is_multi_repo_mode() and repo_name:
        state_dir = _get_repo_state_dir(repo_name)
        cache_path = state_dir / CACHE_FILENAME

        cache = _read_cache_file_cached(cache_path)
        file_hashes = cache.get("file_hashes", {})
        val = file_hashes.get(fp)
    else:
        cache = _read_cache_cached(_resolve_workspace_root())
        val = cache.get("file_hashes", {}).get(fp)

    if isinstance(val, dict):
        return {
            "hash": str(val.get("hash") or ""),
            "size": val.get("size"),
            "mtime": val.get("mtime"),
        }
    if isinstance(val, str):
        return {"hash": val}
    return {}


def remove_cached_file(file_path: str, repo_name: Optional[str] = None) -> None:
    """Remove file entry from cache."""
    if is_multi_repo_mode() and repo_name:
        state_dir = _get_repo_state_dir(repo_name)
        cache_path = state_dir / CACHE_FILENAME

        if cache_path.exists():
            cache = _read_cache_file_cached(cache_path)
            file_hashes = cache.get("file_hashes", {})

            fp = _normalize_cache_key_path(file_path)
            if fp in file_hashes:
                file_hashes.pop(fp, None)
                cache["updated_at"] = datetime.now().isoformat()

                _atomic_write_state(cache_path, cache)
                _memoize_cache_obj(cache_path, cache)
        return

    cache = _read_cache_cached(_resolve_workspace_root())
    fp = _normalize_cache_key_path(file_path)
    if fp in cache.get("file_hashes", {}):
        cache["file_hashes"].pop(fp, None)
        cache["updated_at"] = datetime.now().isoformat()
        _write_cache(_resolve_workspace_root(), cache)
        _memoize_cache_obj(_get_cache_path(_resolve_workspace_root()), cache)


def cleanup_old_cache_locks(max_idle_seconds: int = 900) -> int:
    """Best-effort cleanup of idle cache locks.

    Removes locks that have been idle (not requested via _get_state_lock) for longer than max_idle_seconds
    and whose lock can be acquired without blocking (i.e., not held).
    Returns the number of locks removed.
    """
    now = time.time()
    removed = 0
    with _state_lock:
        stale_keys = []
        for ws, lock in list(_state_locks.items()):
            last = _state_lock_last_used.get(ws, 0.0)
            # Prefer also pruning locks whose workspace no longer exists
            ws_exists = True
            try:
                ws_exists = Path(ws).exists()
            except Exception:
                ws_exists = False
            if (now - last) > max_idle_seconds or not ws_exists:
                acquired = False
                try:
                    acquired = lock.acquire(blocking=False)
                except Exception:
                    acquired = False
                if acquired:
                    try:
                        stale_keys.append(ws)
                    finally:
                        try:
                            lock.release()
                        except Exception:
                            pass
        for ws in stale_keys:
            _state_locks.pop(ws, None)
            _state_lock_last_used.pop(ws, None)
            removed += 1
    return removed


def get_collection_mappings(search_root: Optional[str] = None) -> List[Dict[str, Any]]:
    """Enumerate collection mappings with origin metadata."""

    root_path = Path(search_root or _resolve_workspace_root()).resolve()
    mappings: List[Dict[str, Any]] = []

    try:
        if is_multi_repo_mode():
            repos_root = root_path / STATE_DIRNAME / "repos"
            if repos_root.exists():
                for repo_dir in sorted(p for p in repos_root.iterdir() if p.is_dir()):
                    repo_name = repo_dir.name
                    state_path = repo_dir / STATE_FILENAME
                    if not state_path.exists():
                        continue
                    try:
                        with open(state_path, "r", encoding="utf-8-sig") as f:
                            state = json.load(f) or {}
                    except Exception as e:
                        print(f"[workspace_state] Failed to read repo state from {state_path}: {e}")
                        continue

                    origin = state.get("origin", {}) or {}
                    mappings.append(
                        {
                            "repo_name": repo_name,
                            "collection_name": state.get("qdrant_collection")
                            or get_collection_name(repo_name),
                            "container_path": origin.get("container_path")
                            or str((Path(_resolve_workspace_root()) / repo_name).resolve()),
                            "source_path": origin.get("source_path"),
                            "state_file": str(state_path),
                            "updated_at": state.get("updated_at"),
                        }
                    )
        else:
            state_path = root_path / STATE_DIRNAME / STATE_FILENAME
            if state_path.exists():
                try:
                    with open(state_path, "r", encoding="utf-8-sig") as f:
                        state = json.load(f) or {}
                except Exception:
                    state = {}

                origin = state.get("origin", {}) or {}
                repo_name = origin.get("repo_name") or Path(root_path).name
                mappings.append(
                    {
                        "repo_name": repo_name,
                        "collection_name": state.get("qdrant_collection")
                        or get_collection_name(repo_name),
                        "container_path": origin.get("container_path")
                        or str(root_path),
                        "source_path": origin.get("source_path"),
                        "state_file": str(state_path),
                        "updated_at": state.get("updated_at"),
                    }
                )
    except Exception:
        return mappings

    return mappings


def _env_truthy(name: str, default: bool = False) -> bool:
    try:
        v = os.environ.get(name)
        if v is None:
            return bool(default)
        return str(v).strip().lower() in {"1", "true", "yes", "on"}
    except Exception:
        return bool(default)


def _env_int(name: str) -> Optional[int]:
    try:
        v = os.environ.get(name)
        if v is None:
            return None
        v = str(v).strip()
        if not v:
            return None
        return int(v)
    except Exception:
        return None


def get_indexing_config_snapshot() -> Dict[str, Any]:
    return {
        "embedding_model": os.environ.get("EMBEDDING_MODEL"),
        "embedding_provider": os.environ.get("EMBEDDING_PROVIDER"),
        "refrag_mode": _env_truthy("REFRAG_MODE", False),
        "qwen3_embedding_enabled": _env_truthy("QWEN3_EMBEDDING_ENABLED", False),
        "index_semantic_chunks": _env_truthy("INDEX_SEMANTIC_CHUNKS", True),
        "index_micro_chunks": _env_truthy("INDEX_MICRO_CHUNKS", False),
        "micro_chunk_tokens": _env_int("MICRO_CHUNK_TOKENS"),
        "micro_chunk_stride": _env_int("MICRO_CHUNK_STRIDE"),
        "max_micro_chunks_per_file": _env_int("MAX_MICRO_CHUNKS_PER_FILE"),
        "index_chunk_lines": _env_int("INDEX_CHUNK_LINES"),
        "index_chunk_overlap": _env_int("INDEX_CHUNK_OVERLAP"),
        "use_tree_sitter": _env_truthy("USE_TREE_SITTER", False),
        "index_use_enhanced_ast": _env_truthy("INDEX_USE_ENHANCED_AST", False),
        "mini_vec_dim": _env_int("MINI_VEC_DIM"),
        "lex_sparse_mode": _env_truthy("LEX_SPARSE_MODE", False),
    }


def compute_indexing_config_hash(cfg: Dict[str, Any]) -> str:
    try:
        payload = json.dumps(cfg, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        payload = str(cfg)
    return hashlib.sha1(payload.encode("utf-8", errors="ignore")).hexdigest()


def persist_indexing_config(
    *,
    workspace_path: Optional[str] = None,
    repo_name: Optional[str] = None,
    environment: Optional[Dict[str, str]] = None,
    config: Optional[Dict[str, Any]] = None,
    config_hash: Optional[str] = None,
    pending: bool = False,
) -> WorkspaceState:
    state = get_workspace_state(workspace_path, repo_name) or {}
    if not isinstance(state, dict):
        state = {}

    cfg = config or get_indexing_config_snapshot()
    cfg_hash = config_hash or compute_indexing_config_hash(cfg)
    env_snapshot = environment or dict(os.environ)

    updates: Dict[str, Any] = {}
    if pending:
        # Avoid clobbering an existing pending snapshot (e.g. staging pending env)
        # unless the caller explicitly provides an override.
        if config is not None or state.get("indexing_config_pending") is None:
            updates["indexing_config_pending"] = cfg
            updates["indexing_config_pending_hash"] = cfg_hash
        if environment is not None or state.get("indexing_env_pending") is None:
            updates["indexing_env_pending"] = env_snapshot
    else:
        updates["indexing_config"] = cfg
        updates["indexing_config_hash"] = cfg_hash
        # Only overwrite indexing_env when explicitly provided or missing.
        # This prevents background services (watcher/indexer) from clobbering an
        # already persisted env snapshot (including staging-promoted env).
        if environment is not None or not state.get("indexing_env"):
            updates["indexing_env"] = env_snapshot
        updates["indexing_config_pending"] = None
        updates["indexing_config_pending_hash"] = None
        updates["indexing_env_pending"] = None

    return update_workspace_state(
        workspace_path=workspace_path,
        repo_name=repo_name,
        updates=updates,
    )


def find_collection_for_logical_repo(logical_repo_id: str, search_root: Optional[str] = None) -> Optional[str]:
    if not logical_repo_reuse_enabled():
        return None

    root_path = Path(search_root or _resolve_workspace_root()).resolve()

    try:
        if is_multi_repo_mode():
            repos_root = root_path / STATE_DIRNAME / "repos"
            if repos_root.exists():
                for repo_dir in repos_root.iterdir():
                    if not repo_dir.is_dir():
                        continue
                    state_path = repo_dir / STATE_FILENAME
                    if not state_path.exists():
                        continue
                    try:
                        with open(state_path, "r", encoding="utf-8-sig") as f:
                            state = json.load(f) or {}
                    except Exception:
                        continue

                    ws = state.get("workspace_path") or str(root_path)
                    state = ensure_logical_repo_id(state, ws)
                    if state.get("logical_repo_id") == logical_repo_id:
                        coll = state.get("qdrant_collection")
                        if coll:
                            try:
                                _atomic_write_state(state_path, state)
                            except Exception as e:
                                print(f"[workspace_state] Failed to persist logical_repo_id mapping to {state_path}: {e}")
                            return coll

        state_path = root_path / STATE_DIRNAME / STATE_FILENAME
        if state_path.exists():
            try:
                with open(state_path, "r", encoding="utf-8-sig") as f:
                    state = json.load(f) or {}
            except Exception as e:
                print(f"[workspace_state] Failed to read workspace state from {state_path}: {e}")
                state = {}

            ws = state.get("workspace_path") or str(root_path)
            state = ensure_logical_repo_id(state, ws)
            if state.get("logical_repo_id") == logical_repo_id:
                coll = state.get("qdrant_collection")
                if coll:
                    try:
                        _atomic_write_state(state_path, state)
                    except Exception as e:
                        print(f"[workspace_state] Failed to persist logical_repo_id mapping to {state_path}: {e}")
                    return coll
    except Exception as e:
        print(f"[workspace_state] Error while searching collections for logical_repo_id={logical_repo_id}: {e}")
        return None

    return None


def get_or_create_collection_for_logical_repo(
    workspace_path: str,
    preferred_repo_name: Optional[str] = None,
) -> str:
    # Gate entire logical-repo based resolution behind feature flag
    if not logical_repo_reuse_enabled():
        base_repo = preferred_repo_name
        try:
            coll = get_collection_name(base_repo)
        except Exception:
            coll = get_collection_name(None)
        try:
            update_workspace_state(
                workspace_path=workspace_path,
                updates={"qdrant_collection": coll},
                repo_name=preferred_repo_name,
            )
        except Exception as e:
            print(f"[workspace_state] Failed to persist legacy qdrant_collection for {workspace_path}: {e}")
        return coll
    try:
        ws = Path(workspace_path).resolve()
    except Exception:
        ws = Path(workspace_path)

    common = _detect_git_common_dir(ws)
    if common is not None:
        canonical_root = common.parent
    else:
        canonical_root = ws

    ws_path = str(canonical_root)

    try:
        state = get_workspace_state(workspace_path=ws_path, repo_name=preferred_repo_name)
    except Exception:
        state = {}

    if not isinstance(state, dict):
        state = {}

    try:
        state = ensure_logical_repo_id(state, ws_path)
    except Exception:
        pass

    lrid = state.get("logical_repo_id")
    if isinstance(lrid, str) and lrid:
        coll = find_collection_for_logical_repo(lrid, search_root=ws_path)
        if isinstance(coll, str) and coll:
            if state.get("qdrant_collection") != coll:
                try:
                    update_workspace_state(
                        workspace_path=ws_path,
                        updates={"qdrant_collection": coll, "logical_repo_id": lrid},
                        repo_name=preferred_repo_name,
                    )
                except Exception:
                    pass
            return coll

    coll = state.get("qdrant_collection")
    if not isinstance(coll, str) or not coll:
        base_repo = preferred_repo_name
        try:
            coll = get_collection_name(base_repo)
        except Exception:
            coll = get_collection_name(None)
        try:
            update_workspace_state(
                workspace_path=ws_path,
                updates={"qdrant_collection": coll},
                repo_name=preferred_repo_name,
            )
        except Exception:
            pass

    return coll


# ===== Symbol-Level Cache for Smart Reindexing =====

def _get_symbol_cache_path(file_path: str) -> Path:
    """Get symbol cache file path for a given file."""
    try:
        fp = _normalize_cache_key_path(file_path)
        # Create symbol cache using file hash to handle renames
        file_hash = hashlib.md5(fp.encode('utf-8')).hexdigest()[:8]
        if is_multi_repo_mode():
            repo_name = _detect_repo_name_from_path(Path(file_path))
            if repo_name:
                state_dir = _get_repo_state_dir(repo_name)
                return state_dir / "symbols" / f"{file_hash}.json"
        return _get_cache_path(_resolve_workspace_root()).parent / "symbols" / f"{file_hash}.json"
    except Exception:
        # Fallback: use file name
        return _get_cache_path(_resolve_workspace_root()).parent / "symbols" / f"{Path(file_path).name}.json"


def get_cached_symbols(file_path: str) -> dict:
    """Load cached symbol metadata for a file."""
    cache_path = _get_symbol_cache_path(file_path)

    if not cache_path.exists():
        return {}

    try:
        with open(cache_path, 'r', encoding='utf-8-sig') as f:
            cache_data = json.load(f)
            return cache_data.get("symbols", {})
    except Exception:
        return {}


def set_cached_symbols(file_path: str, symbols: dict, file_hash: str) -> None:
    """Save symbol metadata for a file. Extends existing to include pseudo data."""
    cache_path = _get_symbol_cache_path(file_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        cache_data = {
            "file_path": str(file_path),
            "file_hash": file_hash,
            "updated_at": datetime.now().isoformat(),
            "symbols": symbols
        }

        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=2)

        # Ensure symbol cache files are group-writable so both indexer and
        # watcher processes (potentially different users sharing a group)
        # can update them on shared volumes.
        try:
            os.chmod(cache_path, 0o664)
        except PermissionError:
            pass
    except Exception as e:
        print(f"[SYMBOL_CACHE_WARNING] Failed to save symbol cache for {file_path}: {e}")


def get_cached_pseudo(file_path: str, symbol_id: str) -> tuple[str, list[str]]:
    """Load cached pseudo description and tags for a specific symbol.

    Returns:
        (pseudo, tags) tuple, or ("", []) if not found
    """
    cached_symbols = get_cached_symbols(file_path)

    if symbol_id in cached_symbols:
        symbol_info = cached_symbols[symbol_id]
        pseudo = symbol_info.get("pseudo", "")
        tags = symbol_info.get("tags", [])

        # Ensure correct types
        if isinstance(pseudo, str):
            pseudo = pseudo
        else:
            pseudo = ""

        if isinstance(tags, list):
            tags = [str(tag) for tag in tags]
        else:
            tags = []

        return pseudo, tags

    return "", []


def set_cached_pseudo(file_path: str, symbol_id: str, pseudo: str, tags: list[str], file_hash: str) -> None:
    """Update pseudo data for a specific symbol in the cache.

    This function updates only the pseudo data without recreating the entire symbol cache,
    making it efficient for incremental updates during indexing.
    """
    cached_symbols = get_cached_symbols(file_path)

    # Update the symbol with pseudo data
    if symbol_id in cached_symbols:
        cached_symbols[symbol_id]["pseudo"] = pseudo
        cached_symbols[symbol_id]["tags"] = tags

        # Save the updated cache only when we actually have symbol entries, to
        # avoid creating empty symbol cache files before the base symbol set
        # has been seeded by the indexer/smart reindex path.
        set_cached_symbols(file_path, cached_symbols, file_hash)


def update_symbols_with_pseudo(file_path: str, symbols_with_pseudo: dict, file_hash: str) -> None:
    """Update symbols cache with pseudo data for multiple symbols at once.

    Args:
        file_path: Path to the file
        symbols_with_pseudo: Dict mapping symbol_id to (symbol_info, pseudo, tags) tuples
        file_hash: Current file hash
    """
    cached_symbols = get_cached_symbols(file_path)

    # Update symbols with their new pseudo data
    for symbol_id, (symbol_info, pseudo, tags) in symbols_with_pseudo.items():
        if symbol_id in cached_symbols:
            # Update existing symbol with pseudo data
            cached_symbols[symbol_id]["pseudo"] = pseudo
            cached_symbols[symbol_id]["tags"] = tags

            # Update content hash from symbol_info if available
            if isinstance(symbol_info, dict):
                cached_symbols[symbol_id].update(symbol_info)

    # Save the updated cache
    set_cached_symbols(file_path, cached_symbols, file_hash)


def remove_cached_symbols(file_path: str) -> None:
    """Remove symbol cache for a file (when file is deleted)."""
    cache_path = _get_symbol_cache_path(file_path)
    try:
        if cache_path.exists():
            cache_path.unlink()
    except Exception:
        pass


def clear_symbol_cache(
    workspace_path: Optional[str] = None,
    repo_name: Optional[str] = None,
) -> int:
    """
    Clear symbol cache files for a workspace/repo.

    Returns the number of symbol cache directories removed.
    """
    dirs_removed = 0
    workspace_root = workspace_path or _resolve_workspace_root()

    target_dirs: List[Path] = []
    if is_multi_repo_mode() and repo_name:
        target_dirs.append(_get_repo_state_dir(repo_name) / "symbols")
    else:
        try:
            cache_parent = _get_cache_path(workspace_root).parent
        except Exception:
            cache_parent = Path(workspace_root) / ".codebase"
        target_dirs.append(cache_parent / "symbols")

    for symbols_dir in target_dirs:
        if not symbols_dir.exists():
            continue
        for cache_file in symbols_dir.glob("*.json"):
            file_path = ""
            try:
                with cache_file.open("r", encoding="utf-8-sig") as f:
                    data = json.load(f)
                file_path = str(data.get("file_path") or "")
            except Exception:
                file_path = ""
            if file_path:
                remove_cached_symbols(file_path)
            else:
                try:
                    cache_file.unlink()
                except Exception:
                    pass

        # Best-effort cleanup of empty symbols directory
        try:
            next(symbols_dir.iterdir())
        except StopIteration:
            try:
                symbols_dir.rmdir()
                dirs_removed += 1
            except Exception:
                pass
        except Exception:
            pass

    return dirs_removed


def compare_symbol_changes(old_symbols: dict, new_symbols: dict) -> tuple[list, list]:
    """
    Compare old and new symbols to identify changes.

    Returns:
        (unchanged_symbols, changed_symbols)
    """
    unchanged = []
    changed = []

    for symbol_id, symbol_info in new_symbols.items():
        if symbol_id in old_symbols:
            old_info = old_symbols[symbol_id]
            # Compare content hash
            if old_info.get("content_hash") == symbol_info.get("content_hash"):
                unchanged.append(symbol_id)
            else:
                changed.append(symbol_id)
        else:
            # New symbol
            changed.append(symbol_id)

    return unchanged, changed


def list_workspaces(
    search_root: Optional[str] = None,
    use_qdrant_fallback: bool = True,
) -> List[Dict[str, Any]]:
    """Scan for workspaces via local filesystem or Qdrant collections.

    Supports both local/mounted and remote client-server scenarios:
    - Local: Scans filesystem for .codebase/state.json files
    - Remote: Falls back to querying Qdrant collections for workspace metadata

    Args:
        search_root: Directory to scan for local mode; defaults to parent of /work.
        use_qdrant_fallback: If True and no local workspaces found, query Qdrant.

    Returns:
        List of workspace info dicts with keys:
        - workspace_path: str
        - collection_name: str
        - last_updated: str or int (ISO timestamp or unix)
        - indexing_state: str
        - source: "local" or "qdrant" (indicates discovery method)
    """
    if search_root is None:
        # Default to parent of workspace root
        try:
            search_root = str(Path(_resolve_workspace_root()).parent)
        except Exception:
            search_root = "/work"

    root_path = Path(search_root).resolve()
    workspaces: List[Dict[str, Any]] = []
    seen_paths: set = set()

    # --- Local filesystem scan ---
    try:
        # Find all state.json files
        for state_file in root_path.rglob(f"{STATE_DIRNAME}/{STATE_FILENAME}"):
            try:
                # Skip if in repos subdirectory (multi-repo per-repo states)
                if "repos" in state_file.parts:
                    continue

                workspace_path = str(state_file.parent.parent.resolve())

                # Skip duplicates
                if workspace_path in seen_paths:
                    continue
                seen_paths.add(workspace_path)

                # Read state file
                with open(state_file, "r", encoding="utf-8-sig") as f:
                    state = json.load(f)

                if not isinstance(state, dict):
                    continue

                # Extract info
                collection_name = state.get("qdrant_collection", "")
                updated_at = state.get("updated_at", "")

                indexing_status = state.get("indexing_status", {})
                if isinstance(indexing_status, dict):
                    indexing_state = indexing_status.get("state", "unknown")
                else:
                    indexing_state = "unknown"

                workspaces.append({
                    "workspace_path": workspace_path,
                    "collection_name": collection_name,
                    "last_updated": updated_at,
                    "indexing_state": indexing_state,
                    "source": "local",
                })
            except Exception:
                continue

        # Also check multi-repo states
        if is_multi_repo_mode():
            repos_root = root_path / STATE_DIRNAME / "repos"
            if repos_root.exists():
                for repo_dir in repos_root.iterdir():
                    if not repo_dir.is_dir():
                        continue
                    state_file = repo_dir / STATE_FILENAME
                    if not state_file.exists():
                        continue
                    try:
                        with open(state_file, "r", encoding="utf-8-sig") as f:
                            state = json.load(f)

                        if not isinstance(state, dict):
                            continue

                        repo_name = repo_dir.name
                        workspace_path = state.get("workspace_path", str(root_path / repo_name))

                        if workspace_path in seen_paths:
                            continue
                        seen_paths.add(workspace_path)

                        collection_name = state.get("qdrant_collection", "")
                        updated_at = state.get("updated_at", "")

                        indexing_status = state.get("indexing_status", {})
                        if isinstance(indexing_status, dict):
                            indexing_state = indexing_status.get("state", "unknown")
                        else:
                            indexing_state = "unknown"

                        workspaces.append({
                            "workspace_path": workspace_path,
                            "collection_name": collection_name,
                            "last_updated": updated_at,
                            "indexing_state": indexing_state,
                            "repo_name": repo_name,
                            "source": "local",
                        })
                    except Exception:
                        continue
    except Exception:
        pass

    # --- Qdrant fallback for remote scenarios ---
    if not workspaces and use_qdrant_fallback:
        try:
            workspaces = _list_workspaces_from_qdrant(seen_paths)
        except Exception:
            pass

    # Sort by last_updated descending
    try:
        workspaces.sort(key=lambda w: w.get("last_updated", ""), reverse=True)
    except Exception:
        pass

    return workspaces


def _list_workspaces_from_qdrant(seen_paths: set) -> List[Dict[str, Any]]:
    """Query Qdrant collections to discover workspaces (for remote scenarios).

    Samples points from each collection to extract workspace metadata.
    """
    workspaces: List[Dict[str, Any]] = []

    try:
        from qdrant_client import QdrantClient
    except ImportError:
        return workspaces

    qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    qdrant_key = os.environ.get("QDRANT_API_KEY")

    try:
        client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_key,
            timeout=float(os.environ.get("QDRANT_TIMEOUT", "10") or 10),
        )

        # List all collections
        collections = client.get_collections().collections

        for coll in collections:
            coll_name = coll.name
            if not coll_name:
                continue

            # Sample a few points to extract workspace metadata
            try:
                points, _ = client.scroll(
                    collection_name=coll_name,
                    limit=5,
                    with_payload=True,
                    with_vectors=False,
                )

                if not points:
                    continue

                # Extract workspace info from sampled points
                workspace_path = None
                repo_name = None
                last_ingested = None

                for pt in points:
                    payload = getattr(pt, "payload", {}) or {}
                    md = payload.get("metadata", {}) or {}

                    # Try to get workspace path from metadata
                    if not workspace_path:
                        workspace_path = (
                            md.get("workspace_path")
                            or md.get("source_root")
                            or payload.get("workspace_path")
                        )

                    # Try to get repo name
                    if not repo_name:
                        repo_name = md.get("repo") or md.get("repo_name")

                    # Get ingestion timestamp
                    ts = md.get("ingested_at") or payload.get("ingested_at")
                    if ts and (last_ingested is None or ts > last_ingested):
                        last_ingested = ts

                # Build workspace entry
                ws_path = workspace_path or f"/work/{repo_name}" if repo_name else f"[{coll_name}]"

                if ws_path in seen_paths:
                    continue
                seen_paths.add(ws_path)

                workspaces.append({
                    "workspace_path": ws_path,
                    "collection_name": coll_name,
                    "last_updated": last_ingested or "",
                    "indexing_state": "indexed",  # If points exist, it's indexed
                    "repo_name": repo_name or "",
                    "source": "qdrant",
                })
            except Exception:
                # Collection exists but couldn't sample - still report it
                workspaces.append({
                    "workspace_path": f"[{coll_name}]",
                    "collection_name": coll_name,
                    "last_updated": "",
                    "indexing_state": "unknown",
                    "source": "qdrant",
                })
    except Exception:
        pass

    return workspaces


# Add missing functions that callers expect (already defined above)