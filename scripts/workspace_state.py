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

_SLUGGED_REPO_RE = re.compile(r"^.+-[0-9a-f]{16}$")
_managed_slug_cache_lock = threading.Lock()
_managed_slug_cache: set[str] = set()
_managed_slug_cache_neg: set[str] = set()

_cache_memo_lock = threading.Lock()
_cache_memo: Dict[str, Dict[str, Any]] = {}
_cache_memo_sig: Dict[str, tuple[int, int]] = {}
_cache_memo_last_check: Dict[str, float] = {}


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


class WorkspaceState(TypedDict, total=False):
    created_at: str
    updated_at: str
    qdrant_collection: str
    indexing_status: Optional[IndexingStatus]
    last_activity: Optional[LastActivity]
    qdrant_stats: Optional[Dict[str, Any]]
    origin: Optional[OriginInfo]
    logical_repo_id: Optional[str]

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
        # First try: get repo name from git remote origin URL (canonical name)
        try:
            r = subprocess.run(
                ["git", "-C", str(base), "config", "--get", "remote.origin.url"],
                capture_output=True, text=True, timeout=5
            )
            if r.returncode == 0 and r.stdout.strip():
                url = r.stdout.strip()
                # Extract repo name from URL (e.g., git@github.com:user/repo.git -> repo)
                name = url.rstrip("/").rsplit("/", 1)[-1]
                if name.endswith(".git"):
                    name = name[:-4]
                if name:
                    return name
        except Exception:
            pass

        # Second try: get git toplevel directory name
        r = subprocess.run(["git", "-C", str(base), "rev-parse", "--show-toplevel"],
                           capture_output=True, text=True, timeout=5)
        top = (r.stdout or "").strip()
        if r.returncode == 0 and top:
            return Path(top).name
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
            if not (ws_root / repo_name).exists():
                return {}
        except Exception:
            return {}

    lock = _get_state_lock(workspace_path, repo_name)
    with lock:
        state = get_workspace_state(workspace_path, repo_name)
        for key, value in updates.items():
            if key in state or key in WorkspaceState.__annotations__:
                state[key] = value

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
        m = re.match(r"^(.*)-([0-9a-f]{16})$", repo_name)
        if m:
            base = (m.group(1) or "").strip()
            if base:
                return base
    except Exception:
        pass
    return repo_name


def get_collection_name(repo_name: Optional[str] = None) -> str:
    """Get collection name for repository or workspace."""
    normalized = _normalize_repo_name_for_collection(repo_name) if repo_name else None

    # In multi-repo mode, prioritize repo-specific collection names
    if is_multi_repo_mode() and normalized:
        return _generate_collection_name_from_repo(normalized)

    # Check environment for single-repo mode or fallback
    env_coll = os.environ.get("COLLECTION_NAME", "").strip()
    if env_coll and env_coll not in PLACEHOLDER_COLLECTION_NAMES:
        return env_coll

    # Use repo name if provided (for single-repo mode with repo name)
    if normalized:
        return _generate_collection_name_from_repo(normalized)

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

def _extract_repo_name_from_path(workspace_path: str) -> str:
    """Extract repository name from workspace path.

    Uses git-based detection first (canonical repo name from remote origin URL),
    falls back to folder structure detection when git is unavailable.
    """
    path = Path(workspace_path)

    # First try git-based detection (uses remote origin URL for canonical name)
    git_name = _detect_repo_name_from_path(path)
    if git_name:
        try:
            ws_root = Path(_resolve_workspace_root()).resolve()
        except Exception:
            ws_root = Path(_resolve_workspace_root())
        if ws_root.name and git_name == ws_root.name:
            git_name = None
    if git_name and git_name != "workspace":
        return git_name

    # Fallback to structure-based detection when git is unavailable
    structure_name = _detect_repo_name_from_path_by_structure(path)
    if structure_name:
        return structure_name

    # Final fallback
    return git_name or "workspace"

# Cache functions for file hash tracking
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


def compare_symbol_changes(old_symbols: dict, new_symbols: dict) -> tuple[list, list]:
    """
    Compare old and new symbols to identify changes.

    Uses symbol_diff for move detection when available (SYMBOL_DIFF_ENABLED=1).
    Moved symbols (same content, different location) are treated as unchanged
    for embedding reuse purposes.

    Returns:
        (unchanged_symbols, changed_symbols)
        - unchanged includes symbols that are identical OR moved (reusable embeddings)
        - changed includes added, modified, or removed symbols
    """
    # Check if enhanced diff is enabled
    use_symbol_diff = os.environ.get("SYMBOL_DIFF_ENABLED", "0").lower() in {"1", "true", "yes", "on"}

    if use_symbol_diff:
        try:
            from scripts.symbol_diff import SymbolInfo, diff_symbols

            # Convert dict format to SymbolInfo objects
            def _to_symbol_info(symbol_id: str, info: dict) -> SymbolInfo:
                return SymbolInfo(
                    name=info.get("name", ""),
                    kind=info.get("type", info.get("kind", "")),
                    content_hash=info.get("content_hash", ""),
                    start_line=info.get("start_line", 0),
                    end_line=info.get("end_line", 0),
                )

            old_list = [_to_symbol_info(sid, info) for sid, info in old_symbols.items()]
            new_list = [_to_symbol_info(sid, info) for sid, info in new_symbols.items()]

            changes = diff_symbols(old_list, new_list)

            unchanged = []
            changed = []

            for c in changes:
                if c.new:
                    # Build symbol_id from new symbol
                    new_id = f"{c.new.kind}_{c.new.name}_{c.new.start_line}"
                    if c.change_type in ("unchanged", "moved"):
                        unchanged.append(new_id)
                    else:
                        changed.append(new_id)
                elif c.old and c.change_type == "removed":
                    # Removed symbols go to changed
                    old_id = f"{c.old.kind}_{c.old.name}_{c.old.start_line}"
                    changed.append(old_id)

            return unchanged, changed
        except ImportError:
            pass  # Fall through to basic comparison

    # Basic comparison (no move detection)
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