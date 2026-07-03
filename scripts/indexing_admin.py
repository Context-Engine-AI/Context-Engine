import os
import re
import sys
import time
import subprocess
import shutil
import traceback
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    from qdrant_client import QdrantClient
except Exception:
    QdrantClient = None  # type: ignore

try:
    from scripts.embedder import get_model_dimension
except Exception:
    get_model_dimension = None  # type: ignore

try:
    from scripts.collection_admin import copy_collection_qdrant
except Exception:
    copy_collection_qdrant = None  # type: ignore

try:
    from scripts.ingest_code import (
        ensure_collection_and_indexes_once,
        ensure_payload_indexes,
        _sanitize_vector_name,
        MINI_VECTOR_NAME as _MINI_VECTOR_NAME,
        LEX_SPARSE_NAME as _LEX_SPARSE_NAME,
    )
except Exception:
    ensure_collection_and_indexes_once = None  # type: ignore
    ensure_payload_indexes = None  # type: ignore
    _sanitize_vector_name = None  # type: ignore
    _MINI_VECTOR_NAME = os.environ.get("MINI_VECTOR_NAME", "mini")
    _LEX_SPARSE_NAME = os.environ.get("LEX_SPARSE_NAME", "lex_sparse")

try:
    from scripts.workspace_state import (
        get_collection_mappings,
        get_workspace_state,
        update_workspace_state,
        update_indexing_status,
        get_indexing_config_snapshot,
        compute_indexing_config_hash,
        is_staging_enabled,
        set_staging_state,
        update_staging_status,
        clear_staging_collection,
        activate_staging_collection,
        promote_pending_indexing_config,
        persist_indexing_config,
    )
except Exception:
    get_collection_mappings = None  # type: ignore
    get_workspace_state = None  # type: ignore
    update_workspace_state = None  # type: ignore
    update_indexing_status = None  # type: ignore
    get_indexing_config_snapshot = None  # type: ignore
    compute_indexing_config_hash = None  # type: ignore
    is_staging_enabled = None  # type: ignore
    set_staging_state = None  # type: ignore
    update_staging_status = None  # type: ignore
    clear_staging_collection = None  # type: ignore
    activate_staging_collection = None  # type: ignore
    promote_pending_indexing_config = None  # type: ignore
    persist_indexing_config = None  # type: ignore


def _staging_enabled() -> bool:
    return bool(is_staging_enabled() if callable(is_staging_enabled) else False)


def _workspace_base_dir() -> Path:
    return Path(os.environ.get("WORKSPACE_PATH") or os.environ.get("WATCH_ROOT") or "/work")


def _repo_state_dir(repo_name: str) -> Path:
    return _workspace_base_dir() / ".codebase" / "repos" / repo_name


def _copy_repo_state_for_clone(
    *,
    repo_name: str,
    clone_repo_name: str,
    src_workspace: str,
    clone_workspace: str,
) -> None:
    src_dir = _repo_state_dir(repo_name)
    if not src_dir.exists():
        return
    dst_dir = _repo_state_dir(clone_repo_name)
    shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)

    cache_path = dst_dir / "cache.json"
    if not cache_path.exists():
        return

    try:
        with cache_path.open("r", encoding="utf-8") as fh:
            cache = json.load(fh)
        file_hashes = cache.get("file_hashes")
        if not isinstance(file_hashes, dict):
            return

        src_root = str(Path(src_workspace).resolve())
        clone_root = str(Path(clone_workspace).resolve())
        updated = False
        new_hashes: Dict[str, Any] = {}
        for key, value in file_hashes.items():
            new_key = key
            if isinstance(key, str):
                if key.startswith(src_root):
                    new_key = clone_root + key[len(src_root) :]
                else:
                    token = f"/{repo_name}"
                    repl = f"/{clone_repo_name}"
                    if token in key:
                        new_key = key.replace(token, repl, 1)
                if new_key != key:
                    updated = True
            new_hashes[new_key] = value

        if updated:
            cache["file_hashes"] = new_hashes
            cache["updated_at"] = datetime.now(timezone.utc).isoformat()
            with cache_path.open("w", encoding="utf-8") as fh:
                json.dump(cache, fh, ensure_ascii=False, indent=2, sort_keys=True)
    except Exception as exc:
        print(f"[staging] Warning: failed to retarget cache.json for {clone_repo_name}: {exc}")


_COLLECTION_SCHEMA_CACHE: Dict[str, Dict[str, Any]] = {}
_SNAPSHOT_REFRESHED: Set[str] = set()
_MAPPING_INDEX_CACHE: Dict[str, Any] = {"ts": 0.0, "work_dir": "", "value": {}}


def _probe_collection_schema(collection: str) -> Optional[Dict[str, Any]]:
    if not collection or QdrantClient is None:
        return None
    cached = _COLLECTION_SCHEMA_CACHE.get(collection)
    if cached:
        return cached

    qdrant_url = os.environ.get("QDRANT_URL", "http://qdrant:6333")
    api_key = os.environ.get("QDRANT_API_KEY") or None
    try:
        client = QdrantClient(url=qdrant_url, api_key=api_key)
    except Exception:
        return None

    try:
        info = client.get_collection(collection_name=collection)
    except Exception:
        try:
            client.close()
        except Exception:
            pass
        return None

    try:
        vectors: Dict[str, Optional[int]] = {}
        raw_vectors = getattr(getattr(info, "config", None), "params", None)
        raw_vectors = getattr(raw_vectors, "vectors", None)
        if raw_vectors:
            items = None
            if isinstance(raw_vectors, dict):
                items = raw_vectors.items()
            else:
                try:
                    items = raw_vectors.items()
                except Exception:
                    items = None
            if items:
                for name, params in items:
                    try:
                        size = getattr(params, "size", None)
                    except Exception:
                        size = None
                    vectors[str(name)] = size

        sparse_vectors: set[str] = set()
        raw_sparse = getattr(getattr(info.config, "params", None), "sparse_vectors", None)
        if raw_sparse:
            keys_iter = None
            if isinstance(raw_sparse, dict):
                keys_iter = raw_sparse.keys()
            else:
                try:
                    keys_iter = raw_sparse.keys()
                except Exception:
                    keys_iter = None
            if keys_iter:
                for name in keys_iter:
                    sparse_vectors.add(str(name))

        schema = {
            "vectors": vectors,
            "sparse_vectors": sparse_vectors,
            "payload_indexes": getattr(getattr(info.config, "params", None), "payload_indexes", None),
        }
        _COLLECTION_SCHEMA_CACHE[collection] = schema
        return schema
    except Exception:
        return None
    finally:
        try:
            client.close()
        except Exception:
            pass


def _filter_snapshot_only_recreate_keys(
    collection: str,
    drift_keys: List[str],
    env_config: Dict[str, Any],
) -> Tuple[List[str], List[str]]:
    if not drift_keys or not collection or not env_config:
        return drift_keys, []
    schema = _probe_collection_schema(collection)
    if not schema:
        return drift_keys, []

    vectors = schema.get("vectors") or {}
    sparse_vectors = schema.get("sparse_vectors") or set()
    mini_dim = vectors.get(_MINI_VECTOR_NAME)
    has_sparse = _LEX_SPARSE_NAME in sparse_vectors

    snapshot_only: List[str] = []
    remaining: List[str] = []
    for key in drift_keys:
        suppress = False
        if key == "embedding_model":
            desired = env_config.get("embedding_model")
            if desired and vectors.get(_sanitize_vector_name(desired) if _sanitize_vector_name else desired):
                suppress = True
        elif key == "embedding_provider":
            suppress = False  # provider changes require rebuild
        if key == "lex_sparse_mode":
            desired_sparse = bool(env_config.get("lex_sparse_mode"))
            if desired_sparse and has_sparse:
                suppress = True
        elif key == "mini_vec_dim":
            desired_dim = env_config.get("mini_vec_dim")
            refrag_on = bool(env_config.get("refrag_mode"))
            if refrag_on and desired_dim and mini_dim == desired_dim:
                suppress = True
        if suppress:
            snapshot_only.append(key)
        else:
            remaining.append(key)
    return remaining, snapshot_only


def _auto_refresh_snapshot_if_needed(
    *,
    collection: str,
    workspace_path: Optional[str],
    repo_name: Optional[str],
    snapshot_only_keys: List[str],
    staging_status: str = "none",
    indexing_state: str = "",
) -> bool:
    if not snapshot_only_keys or not collection:
        return False
    if promote_pending_indexing_config is None:
        return False
    if persist_indexing_config is None:
        return False
    ws = (workspace_path or "").strip()
    if not ws:
        return False
    cache_key = f"{ws}:{repo_name or ''}"
    if cache_key in _SNAPSHOT_REFRESHED:
        return False

    # Safety: Do not auto-refresh during active staging rebuild.
    # The old collection depends on the saved .env being accurate.
    if staging_status == "active":
        return False

    # Skip when indexing is currently running; wait for a quiescent window.
    try:
        normalized_index_state = (indexing_state or "").strip().lower()
    except Exception:
        normalized_index_state = ""
    if normalized_index_state in {"initializing", "indexing", "running"}:
        return False

    try:
        if get_workspace_state is not None:
            st = get_workspace_state(ws, repo_name) or {}
            if isinstance(st, dict):
                has_pending = bool(st.get("indexing_config_pending") or st.get("indexing_env_pending"))
                applied_hash = str(st.get("indexing_config_hash") or "")
                pending_hash = str(st.get("indexing_config_pending_hash") or "")
            else:
                has_pending = False
                applied_hash = ""
                pending_hash = ""
        else:
            has_pending = False
            applied_hash = ""
            pending_hash = ""
    except Exception:
        has_pending = False
        applied_hash = ""
        pending_hash = ""

    # If the pending hash already matches applied, there's nothing to refresh.
    if applied_hash and pending_hash and applied_hash == pending_hash:
        return False

    # Auto-capture current config as pending when safe:
    # - No active staging rebuild (checked above)
    # - No pending config exists (avoid clobbering existing pending)
    # - Schema already validated (snapshot_only_keys non-empty)
    if not has_pending:
        try:
            persist_indexing_config(
                workspace_path=ws,
                repo_name=repo_name,
                pending=True,
            )
        except Exception as exc:
            try:
                print(
                    f"[snapshot_refresh] Failed to capture pending config for {collection}: {exc}"
                )
            except Exception:
                pass
            return False

    dry_run = str(os.environ.get("CTXCE_SNAPSHOT_REFRESH_DRY_RUN", "0")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    try:
        if dry_run:
            print(
                f"[snapshot_refresh] DRY RUN: would promote pending config for {collection} "
                f"(workspace={ws}, repo={repo_name or 'default'}) after validating schema: "
                f"{', '.join(snapshot_only_keys)}"
            )
            return True
        promote_pending_indexing_config(workspace_path=ws, repo_name=repo_name)
        _SNAPSHOT_REFRESHED.add(cache_key)
        try:
            print(
                f"[snapshot_refresh] promoted pending indexing config for {collection} "
                f"(workspace={ws}, repo={repo_name or 'default'}) after validating schema: "
                f"{', '.join(snapshot_only_keys)}"
            )
        except Exception:
            pass
        return True
    except Exception as exc:
        try:
            print(
                f"[snapshot_refresh] Failed to promote indexing config for {collection}: {exc}"
            )
        except Exception:
            pass
        return False


def _delete_path_tree(p: Path) -> bool:
    try:
        if not p.exists():
            return False
        if p.is_dir():
            try:
                shutil.rmtree(p)
                return True
            except Exception:
                # Best-effort permission fixup for shared volumes / root-owned files.
                try:
                    for sub in p.rglob("*"):
                        try:
                            if sub.is_dir():
                                os.chmod(sub, 0o777)
                            else:
                                os.chmod(sub, 0o666)
                        except Exception:
                            pass
                    try:
                        os.chmod(p, 0o777)
                    except Exception:
                        pass
                except Exception:
                    pass
                try:
                    shutil.rmtree(p)
                    return True
                except Exception:
                    return False
        p.unlink()
        return True
    except Exception:
        return False


def _resolve_codebase_root(work_root: Path) -> Path:
    env_root = (
        os.environ.get("CTXCE_CODEBASE_ROOT")
        or os.environ.get("CODEBASE_ROOT")
        or ""
    ).strip()

    candidates: List[Path] = []
    if env_root:
        candidates.append(Path(env_root))
    candidates.append(work_root)
    candidates.append(work_root.parent)

    for candidate in candidates:
        try:
            base = candidate.resolve()
        except Exception:
            base = candidate
        try:
            if (base / ".codebase" / "repos").exists():
                return base
        except Exception:
            continue

    return work_root


def _cleanup_old_clone(
    *,
    collection: str,
    repo_name: Optional[str],
    delete_collection: bool = True,
    work_dir: Optional[str] = None,
    workspace_root: Optional[str] = None,
) -> None:
    """Best-effort cleanup of cloned *_old collection and workspace/meta."""

    old_collection = f"{collection}_old"
    if delete_collection:
        try:
            delete_collection_qdrant(
                qdrant_url=os.environ.get("QDRANT_URL", "http://qdrant:6333"),
                api_key=os.environ.get("QDRANT_API_KEY") or None,
                collection=old_collection,
            )
        except Exception:
            pass

    if not repo_name:
        return

    old_slug = f"{repo_name}_old"

    def _resolve_base_work_root() -> Path:
        env_work = work_dir or os.environ.get("WORK_DIR") or os.environ.get("WORKDIR") or "/work"
        try:
            return Path(env_work).resolve()
        except Exception:
            return Path(env_work)

    base_work_root = _resolve_base_work_root()

    def _clamp_to_base(candidate: Optional[Path]) -> Path:
        if candidate is None:
            return base_work_root
        try:
            resolved_candidate = candidate.resolve()
        except Exception:
            resolved_candidate = candidate
        try:
            resolved_base = base_work_root.resolve()
        except Exception:
            resolved_base = base_work_root
        try:
            if resolved_base == resolved_candidate or resolved_base in resolved_candidate.parents:
                return resolved_candidate
        except Exception:
            pass
        try:
            print(
                f"[staging] refusing to treat {resolved_candidate} as work root; falling back to {resolved_base}"
            )
        except Exception:
            pass
        return resolved_base

    work_root: Optional[Path] = None
    if workspace_root:
        try:
            work_root = Path(workspace_root).resolve().parent
        except Exception:
            try:
                work_root = Path(workspace_root).parent
            except Exception:
                work_root = None
    if work_root is None:
        try:
            if work_dir:
                work_root = Path(work_dir).resolve()
            else:
                work_root = Path(os.environ.get("WORK_DIR") or os.environ.get("WORKDIR") or "/work").resolve()
        except Exception:
            work_root = Path("/work")
    work_root = _clamp_to_base(work_root)

    codebase_root = _resolve_codebase_root(work_root)

    def _safe_delete(target: Path) -> None:
        try:
            resolved_target = target.resolve()
            resolved_base = work_root.resolve()
        except Exception:
            try:
                resolved_target = target
                resolved_base = work_root
            except Exception:
                return
        try:
            if resolved_base == resolved_target or resolved_base in resolved_target.parents:
                _delete_path_tree(resolved_target)
            else:
                print(f"[staging] refusing to delete {resolved_target}: outside work root {resolved_base}")
        except Exception:
            pass

    # Workspace clone dir
    try:
        _safe_delete((work_root / old_slug).resolve())
    except Exception:
        pass

    # Repo metadata dir
    try:
        _safe_delete((codebase_root / ".codebase" / "repos" / old_slug).resolve())
    except Exception:
        pass

    # If codebase_root differs from work_root, also try under work_root for safety.
    try:
        if str(codebase_root.resolve()) != str(work_root.resolve()):
            _safe_delete((work_root / ".codebase" / "repos" / old_slug).resolve())
    except Exception:
        pass


CONFIG_DRIFT_RULES: Dict[str, str] = {
    # Schema / embedding changes
    "embedding_model": "recreate",
    "embedding_provider": "recreate",
    "refrag_mode": "recreate",
    "qwen3_embedding_enabled": "recreate",
    "mini_vec_dim": "recreate",
    "lex_sparse_mode": "recreate",
    # Chunking / AST changes
    "index_semantic_chunks": "reindex",
    "index_chunk_lines": "reindex",
    "index_chunk_overlap": "reindex",
    "index_micro_chunks": "reindex",
    "micro_chunk_tokens": "reindex",
    "micro_chunk_stride": "reindex",
    "max_micro_chunks_per_file": "reindex",
    "use_tree_sitter": "reindex",
    "index_use_enhanced_ast": "reindex",
}


_INDEXING_CONFIG_DEFAULTS: Dict[str, Any] = {
    # Keep in sync with scripts.workspace_state.get_indexing_config_snapshot defaults.
    "refrag_mode": False,
    "qwen3_embedding_enabled": False,
    "index_semantic_chunks": True,
    "index_micro_chunks": False,
    "micro_chunk_tokens": None,
    "micro_chunk_stride": None,
    "max_micro_chunks_per_file": None,
    "index_chunk_lines": None,
    "index_chunk_overlap": None,
    "use_tree_sitter": False,
    "index_use_enhanced_ast": False,
    "mini_vec_dim": None,
    "lex_sparse_mode": False,
}


def current_env_indexing_hash() -> str:
    try:
        if get_indexing_config_snapshot and compute_indexing_config_hash:
            return compute_indexing_config_hash(get_indexing_config_snapshot())
    except Exception:
        return ""
    return ""


def _current_env_indexing_config_and_hash() -> Tuple[Dict[str, Any], str]:
    cfg: Dict[str, Any] = {}
    cfg_hash = ""
    if get_indexing_config_snapshot and compute_indexing_config_hash:
        try:
            snapshot = get_indexing_config_snapshot()
            if isinstance(snapshot, dict):
                cfg = snapshot
            cfg_hash = compute_indexing_config_hash(snapshot or {})
        except Exception:
            cfg = {}
            cfg_hash = ""
    if not cfg_hash:
        cfg_hash = current_env_indexing_hash()
    return cfg, cfg_hash


def _classify_indexing_drift(
    applied_config: Dict[str, Any],
    current_config: Dict[str, Any],
) -> Tuple[List[str], List[str]]:
    recreate_keys: List[str] = []
    reindex_keys: List[str] = []
    if not applied_config or not current_config:
        return recreate_keys, reindex_keys
    keys = set(applied_config.keys()) | set(current_config.keys())
    for key in sorted(keys):
        applied_has = key in applied_config
        current_has = key in current_config
        applied_val = applied_config.get(key)
        current_val = current_config.get(key)

        # Back-compat: older state snapshots may not include newly added keys.
        # Treat missing keys as their default values to avoid false drift.
        if not applied_has and key in _INDEXING_CONFIG_DEFAULTS:
            applied_val = _INDEXING_CONFIG_DEFAULTS.get(key)
        if not current_has and key in _INDEXING_CONFIG_DEFAULTS:
            current_val = _INDEXING_CONFIG_DEFAULTS.get(key)

        if applied_val == current_val:
            continue
        drift_class = CONFIG_DRIFT_RULES.get(key, "recreate")
        if drift_class == "recreate":
            recreate_keys.append(key)
        elif drift_class == "reindex":
            reindex_keys.append(key)
        else:
            recreate_keys.append(key)
    return recreate_keys, reindex_keys


def collection_mapping_index(*, work_dir: str) -> Dict[str, List[Dict[str, Any]]]:
    if not get_collection_mappings:
        return {}
    try:
        ttl = float(os.environ.get("CTXCE_COLLECTION_MAPPING_INDEX_TTL_SECS", "5") or 5)
    except Exception:
        ttl = 5.0
    try:
        now = time.time()
    except Exception:
        now = 0.0
    try:
        if (
            ttl > 0
            and _MAPPING_INDEX_CACHE.get("work_dir") == work_dir
            and (now - float(_MAPPING_INDEX_CACHE.get("ts") or 0.0)) < ttl
        ):
            cached = _MAPPING_INDEX_CACHE.get("value")
            if isinstance(cached, dict):
                return cached
    except Exception:
        pass
    try:
        mappings = get_collection_mappings(search_root=work_dir) or []
    except Exception:
        mappings = []
    out: Dict[str, List[Dict[str, Any]]] = {}
    for m in mappings:
        try:
            name = str(m.get("collection_name") or "").strip()
        except Exception:
            name = ""
        if not name:
            continue
        out.setdefault(name, []).append(m)
    try:
        _MAPPING_INDEX_CACHE["ts"] = now
        _MAPPING_INDEX_CACHE["work_dir"] = work_dir
        _MAPPING_INDEX_CACHE["value"] = out
    except Exception:
        pass
    return out


def resolve_collection_root(*, collection: str, work_dir: str) -> Tuple[Optional[str], Optional[str]]:
    name = (collection or "").strip()
    if not name:
        return None, None
    idx = collection_mapping_index(work_dir=work_dir)
    candidates = idx.get(name) or []
    if not candidates:
        return None, None
    chosen = candidates[0]
    repo_name = chosen.get("repo_name")
    container_path = chosen.get("container_path")
    try:
        repo_name = str(repo_name) if repo_name is not None else None
    except Exception:
        repo_name = None
    try:
        container_path = str(container_path) if container_path is not None else None
    except Exception:
        container_path = None
    if container_path:
        return container_path, repo_name
    return work_dir, repo_name


def _get_workspace_state_safe(workspace_path: str, repo_name: Optional[str]) -> Dict[str, Any]:
    if not get_workspace_state:
        return {}
    try:
        state = get_workspace_state(workspace_path, repo_name) or {}
        if isinstance(state, dict):
            return state
        return {}
    except Exception:
        return {}


def get_indexing_state(*, workspace_path: str, repo_name: Optional[str]) -> str:
    if not get_workspace_state:
        return ""
    try:
        st = get_workspace_state(workspace_path, repo_name) or {}
        idx_status = st.get("indexing_status") or {}
        if isinstance(idx_status, dict):
            return str(idx_status.get("state") or "")
    except Exception:
        return ""
    return ""


def build_admin_collections_view(*, collections: Any, work_dir: str) -> List[Dict[str, Any]]:
    env_config, env_hash = _current_env_indexing_config_and_hash()
    mapping_index = collection_mapping_index(work_dir=work_dir)

    enriched: List[Dict[str, Any]] = []
    for c in collections or []:
        try:
            coll_name = str(getattr(c, "qdrant_collection", None) or c.get("qdrant_collection") or "").strip()  # type: ignore[union-attr]
        except Exception:
            coll_name = ""

        applied_hash = ""
        applied_config: Dict[str, Any] = {}
        pending_hash = ""
        indexing_state = ""
        indexing_started_at = ""
        progress_files_processed: Optional[int] = None
        progress_total_files: Optional[int] = None
        progress_current_file = ""
        repo_name = None
        container_path = None
        mapping_count = 0

        try:
            matches = mapping_index.get(coll_name) or []
            mapping_count = len(matches)
            if matches:
                m = matches[0]
                repo_name = m.get("repo_name")
                container_path = m.get("container_path")
        except Exception:
            mapping_count = 0

        st: Dict[str, Any] = {}
        if container_path and get_workspace_state:
            try:
                st_raw = get_workspace_state(str(container_path), repo_name) or {}
                st = st_raw if isinstance(st_raw, dict) else {}
            except Exception:
                st = {}
        if st:
            try:
                applied_hash = str(st.get("indexing_config_hash") or "")
                cfg = st.get("indexing_config") or {}
                applied_config = cfg if isinstance(cfg, dict) else {}
                pending_hash = str(st.get("indexing_config_pending_hash") or "")
                idx_status = st.get("indexing_status") or {}
                if isinstance(idx_status, dict):
                    indexing_state = str(idx_status.get("state") or "")
                    indexing_started_at = str(idx_status.get("started_at") or "")
                    progress = idx_status.get("progress") or {}
                    if isinstance(progress, dict):
                        try:
                            progress_files_processed = int(progress.get("files_processed"))
                        except Exception:
                            progress_files_processed = None
                        try:
                            progress_total_files = int(progress.get("total_files"))
                        except Exception:
                            progress_total_files = None
                        try:
                            progress_current_file = str(progress.get("current_file") or "")
                        except Exception:
                            progress_current_file = ""
            except Exception:
                applied_hash = ""
                applied_config = {}
                pending_hash = ""
                indexing_state = ""
                indexing_started_at = ""
                progress_files_processed = None
                progress_total_files = None
                progress_current_file = ""

        # Add staging information
        staging_status = "none"
        staging_collection = ""
        staging_state = ""
        if st:
            try:
                staging_info = st.get("staging") or {}
                if isinstance(staging_info, dict) and staging_info.get("collection"):
                    staging_status = "active"
                    staging_collection = str(staging_info.get("collection") or "")
                    staging_status_info = staging_info.get("status") or {}
                    if isinstance(staging_status_info, dict):
                        staging_state = str(staging_status_info.get("state") or "")
                else:
                    staging_status = "none"
            except Exception:
                staging_status = "none"

        # "maintenance needed" should reflect actual config drift requiring a maintenance reindex.
        # Pending hashes can exist during staging / queued rebuild flows; keep them visible in the UI
        # but do not treat them as drift.
        drift_detected = bool(env_hash and applied_hash and env_hash != applied_hash)
        drift_recreate_keys: List[str] = []
        drift_reindex_keys: List[str] = []
        drift_unknown = False
        snapshot_only_recreate_keys: List[str] = []
        snapshot_refresh_triggered = False
        if drift_detected:
            if applied_config and env_config:
                drift_recreate_keys, drift_reindex_keys = _classify_indexing_drift(applied_config, env_config)
                if drift_recreate_keys:
                    drift_recreate_keys, snapshot_only_recreate_keys = _filter_snapshot_only_recreate_keys(
                        coll_name, drift_recreate_keys, env_config
                    )
            else:
                drift_unknown = True

        needs_recreate = bool(drift_recreate_keys)
        if drift_unknown and not needs_recreate:
            needs_recreate = True

        needs_reindex_only = bool(not needs_recreate and drift_reindex_keys)
        needs_snapshot_refresh = bool(
            snapshot_only_recreate_keys and not needs_recreate and not needs_reindex_only
        )
        if needs_snapshot_refresh:
            snapshot_refresh_triggered = _auto_refresh_snapshot_if_needed(
                collection=coll_name,
                workspace_path=container_path or work_dir,
                repo_name=repo_name,
                snapshot_only_keys=snapshot_only_recreate_keys,
                staging_status=staging_status,
                indexing_state=indexing_state,
            )

        needs_reindex = needs_recreate or needs_reindex_only

        try:
            cid = getattr(c, "id", None) if hasattr(c, "id") else c.get("id")  # type: ignore[union-attr]
        except Exception:
            cid = None

        enriched.append(
            {
                "id": cid,
                "qdrant_collection": coll_name,
                "container_path": str(container_path) if container_path else "",
                "repo_name": str(repo_name) if repo_name else "",
                "mapping_count": mapping_count,
                "indexing_state": indexing_state,
                "indexing_started_at": indexing_started_at,
                "progress_files_processed": progress_files_processed,
                "progress_total_files": progress_total_files,
                "progress_current_file": progress_current_file,
                "applied_indexing_hash": applied_hash,
                "pending_indexing_hash": pending_hash,
                "current_indexing_hash": env_hash,
                "needs_reindex": needs_reindex,
                "needs_recreate": needs_recreate,
                "needs_reindex_only": needs_reindex_only,
                "needs_snapshot_refresh": needs_snapshot_refresh,
                "snapshot_refresh_triggered": snapshot_refresh_triggered,
                "drift_recreate_keys": drift_recreate_keys,
                "drift_reindex_keys": drift_reindex_keys,
                "snapshot_only_recreate_keys": snapshot_only_recreate_keys,
                "drift_unknown": drift_unknown,
                "has_mapping": bool(container_path),
                "staging_status": staging_status,
                "staging_collection": staging_collection,
                "staging_state": staging_state,
            }
        )

    return enriched


def delete_collection_qdrant(*, qdrant_url: str, api_key: Optional[str], collection: str) -> None:
    if QdrantClient is None:
        return
    name = (collection or "").strip()
    if not name:
        return
    try:
        cli = QdrantClient(url=qdrant_url, api_key=api_key or None)
    except Exception:
        return
    try:
        cli.delete_collection(collection_name=name)
    except Exception:
        pass
    finally:
        try:
            cli.close()
        except Exception:
            pass


def recreate_collection_qdrant(*, qdrant_url: str, api_key: Optional[str], collection: str) -> None:
    if QdrantClient is None:
        return
    name = (collection or "").strip()
    if not name:
        return
    try:
        cli = QdrantClient(url=qdrant_url, api_key=api_key or None)
    except Exception as connect_error:
        raise RuntimeError(f"Failed to connect to Qdrant for collection '{name}': {connect_error}") from connect_error
    try:
        try:
            cli.delete_collection(collection_name=name)
        except Exception as delete_error:
            raise RuntimeError(f"Failed to delete existing collection '{name}' in Qdrant: {delete_error}") from delete_error
    finally:
        try:
            cli.close()
        except Exception:
            pass


def spawn_ingest_code(
    *,
    root: str,
    work_dir: str,
    collection: str,
    recreate: bool,
    repo_name: Optional[str],
    env_overrides: Optional[Dict[str, Any]] = None,
    clear_caches: bool = False,
) -> None:
    script_path = str((Path(__file__).resolve().parent / "ingest_code.py").resolve())
    cmd = [sys.executable or "python3", script_path, "--root", root, "--no-skip-unchanged"]
    if recreate:
        cmd.append("--recreate")
    if clear_caches:
        cmd.append("--clear-indexing-caches")

    env = os.environ.copy()
    # Apply per-run env overrides (e.g. pending env snapshots for staging rebuild).
    # Merge overrides on top of current env, removing keys for None values.
    if isinstance(env_overrides, dict):
        for k, v in env_overrides.items():
            if v is None:
                env.pop(k, None)
            else:
                env[str(k)] = str(v)
        # When we provide env overrides for a run (e.g. staging rebuild), we also want to
        # force ingest_code to honor the explicit COLLECTION_NAME instead of routing based
        # on per-repo state/serving_collection in multi-repo mode.
        # CTXCE_FORCE_COLLECTION_NAME is only used for these subprocess runs; normal watcher
        # and indexer flows do not set it.
        env["CTXCE_FORCE_COLLECTION_NAME"] = "1"  # Force ingest_code to use COLLECTION_NAME for staging/pending env overrides
    env["COLLECTION_NAME"] = collection
    env["WATCH_ROOT"] = work_dir
    env["WORKSPACE_PATH"] = work_dir

    try:
        if update_indexing_status:
            update_indexing_status(
                workspace_path=root,
                repo_name=repo_name,
                status={
                    "state": "initializing",
                    "started_at": datetime.now(timezone.utc).isoformat(),
                    "progress": {"files_processed": 0, "total_files": None, "current_file": None},
                },
            )
    except Exception:
        pass

    # Spawn the ingest process and validate it started successfully
    try:
        proc = subprocess.Popen(cmd, env=env)
        try:
            poller = proc.poll  # type: ignore[attr-defined]
        except AttributeError:
            poller = None
        if callable(poller) and poller() is not None:
            raise RuntimeError(f"Failed to start ingest process: {cmd}")
    except Exception as exc:
        raise RuntimeError(f"Failed to spawn ingest_code for {root}: {exc}") from exc


def _determine_embedding_dim(model_name: str) -> int:
    if get_model_dimension:
        try:
            return int(get_model_dimension(model_name))
        except Exception:
            pass
    try:
        from fastembed import TextEmbedding  # type: ignore

        model = TextEmbedding(model_name=model_name)
        return len(next(model.embed(["dimension probe"])))
    except Exception:
        return 1536


def _normalize_cloned_collection_schema(*, collection_name: str, qdrant_url: str) -> None:
    if QdrantClient is None:
        return
    vector_name = None
    model_name = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
    dim = _determine_embedding_dim(model_name)
    if _sanitize_vector_name is not None:
        try:
            vector_name = _sanitize_vector_name(model_name)
        except Exception:
            vector_name = None
    try:
        client = QdrantClient(url=qdrant_url, api_key=os.environ.get("QDRANT_API_KEY") or None)
    except Exception:
        return
    try:
        # IMPORTANT: This function is called on a freshly cloned "*_old" collection which may
        # be actively serving traffic during a migration. We must not recreate/modify the
        # vector schema here, since Qdrant can't add vector names in-place and the fallback
        # logic in ingest_code can delete+recreate collections (which would drop copied points).
        #
        # We only ensure payload indexes for query performance.
        if ensure_payload_indexes is not None:
            ensure_payload_indexes(client, collection_name)
    except Exception as exc:
        try:
            print(f"[staging] Warning: failed to normalize cloned collection {collection_name}: {exc}")
        except Exception:
            pass
    finally:
        try:
            client.close()
        except Exception:
            pass


def _get_collection_point_count(*, collection_name: str, qdrant_url: str) -> Optional[int]:
    if QdrantClient is None:
        return None
    try:
        client = QdrantClient(url=qdrant_url, api_key=os.environ.get("QDRANT_API_KEY") or None)
    except Exception:
        return None
    try:
        try:
            result = client.count(collection_name=collection_name, exact=True)
            return int(getattr(result, "count", 0))
        except Exception:
            return None
    finally:
        try:
            client.close()
        except Exception:
            pass


def _wait_for_clone_points(
    *,
    source_collection: str,
    cloned_collection: str,
    qdrant_url: str,
    expected_count: Optional[int],
    timeout_seconds: int = 60,
) -> None:
    if QdrantClient is None:
        return
    try:
        client = QdrantClient(url=qdrant_url, api_key=os.environ.get("QDRANT_API_KEY") or None)
    except Exception:
        return

    try:
        if expected_count is None or expected_count <= 0:
            # Nothing to wait for (empty collection or unknown count)
            return

        deadline = time.time() + timeout_seconds
        while True:
            try:
                result = client.count(collection_name=cloned_collection, exact=True)
                clone_count = int(getattr(result, "count", 0))
            except Exception:
                clone_count = 0

            if clone_count >= expected_count:
                try:
                    print(
                        f"[staging] Clone verification succeeded: {cloned_collection} has "
                        f"{clone_count} points (expected >= {expected_count})."
                    )
                except Exception:
                    pass
                return

            if time.time() > deadline:
                raise RuntimeError(
                    f"Cloned collection {cloned_collection} only has {clone_count} points "
                    f"(expected at least {expected_count})"
                )

            time.sleep(2)
    finally:
        try:
            client.close()
        except Exception:
            pass


def start_staging_rebuild(*, collection: str, work_dir: str) -> str:
    if not _staging_enabled():
        raise RuntimeError("Staging is disabled (set CTXCE_STAGING_ENABLED=1 to enable)")
    root, repo_name = resolve_collection_root(collection=collection, work_dir=work_dir)
    if not root:
        raise RuntimeError("No workspace mapping found for collection")

    state = _get_workspace_state_safe(root, repo_name)
    current_staging = state.get("staging") or {}
    staging_status_state = ""
    try:
        staging_status_state = str(((current_staging.get("status") or {}).get("state") or "")).strip().lower()
    except Exception:
        staging_status_state = ""
    queued_marker = staging_status_state in {"", "queued"}
    if isinstance(current_staging, dict) and current_staging.get("collection") and not queued_marker:
        raise RuntimeError("A staging collection is already running for this workspace")

    # Copy current collection to <collection>_old
    old_collection = f"{collection}_old"
    qdrant_url = os.environ.get("QDRANT_URL", "http://qdrant:6333")
    source_point_count = _get_collection_point_count(collection_name=collection, qdrant_url=qdrant_url)

    # Use local import for thread-safety and determinism
    _copy_fn: Any = copy_collection_qdrant
    if _copy_fn is None:
        # Re-import for container environments where module-level import may have failed
        from scripts.collection_admin import copy_collection_qdrant as _ccq
        _copy_fn = _ccq

    if not callable(_copy_fn):
        raise RuntimeError("copy_collection_qdrant unavailable (import failed)")

    try:
        print(f"[staging] Copying collection {collection} -> {old_collection} (overwrite=True)")
        try:
            print(
                f"[staging] copy_collection_qdrant callable={callable(_copy_fn)} type={type(_copy_fn)} module={getattr(_copy_fn, '__module__', '?')}"
            )
        except Exception:
            pass
        _copy_fn(
            source=collection,
            target=old_collection,
            qdrant_url=qdrant_url,
            overwrite=True,
        )
        print(f"[staging] Copy completed for {old_collection}")
    except Exception as exc:
        print(f"[staging] ERROR copying {collection} -> {old_collection}: {exc!r}")
        try:
            print("[staging] TRACEBACK (copy)")
            print(traceback.format_exc())
        except Exception:
            pass
        raise

    try:
        _wait_for_clone_points(
            source_collection=collection,
            cloned_collection=old_collection,
            qdrant_url=qdrant_url,
            expected_count=source_point_count,
            timeout_seconds=90,
        )
    except Exception as exc:
        print(f"[staging] ERROR verifying clone {old_collection}: {exc}")
        raise

    # IMPORTANT: switch serving to *_old as soon as the clone is verified.
    # Large repos can make the filesystem copy below slow; don't block the traffic cutover.
    try:
        print(f"[staging] Switching serving to clone {old_collection}")
        update_workspace_state(
            workspace_path=root,
            repo_name=repo_name,
            updates={
                "serving_collection": old_collection,
                "serving_repo_slug": f"{repo_name}_old" if repo_name else "",
                "active_repo_slug": repo_name,
                "qdrant_collection": collection,
            },
        )
    except Exception as exc:
        print(f"[staging] ERROR updating serving state to {old_collection}: {exc}")
        raise

    # Best-effort: ensure payload indexes on the clone. Failures here must not break cutover.
    try:
        _normalize_cloned_collection_schema(collection_name=old_collection, qdrant_url=qdrant_url)
    except Exception as exc:
        print(f"[staging] Warning: failed to normalize cloned collection {old_collection}: {exc}")

    # Duplicate workspace slug/state to <slug>_old so watcher/indexer can serve reads from the clone.
    # This can be slow for large repos; treat as best-effort and do not block staging cutover.
    if repo_name:
        work_root = Path(os.environ.get("WORK_DIR") or os.environ.get("WORKDIR") or "/work")
        canonical_dir = work_root / repo_name
        old_dir = work_root / f"{repo_name}_old"
        try:
            if canonical_dir.exists():
                t0 = time.time()
                print(f"[staging] Copying workspace tree {canonical_dir} -> {old_dir}")
                shutil.copytree(canonical_dir, old_dir, dirs_exist_ok=True)
                print(f"[staging] Workspace copy completed in {time.time() - t0:.1f}s")
        except Exception as exc:
            print(f"[staging] Warning: failed to copy workspace tree for {repo_name}: {exc}")

        try:
            _copy_repo_state_for_clone(
                repo_name=repo_name,
                clone_repo_name=f"{repo_name}_old",
                src_workspace=str(canonical_dir),
                clone_workspace=str(old_dir),
            )
        except Exception as exc:
            print(f"[staging] Warning: failed to copy repo state for {repo_name}: {exc}")

        try:
            old_state = state.copy()
            # Preserve the env/config that was serving traffic; drop pending snapshots.
            old_state["qdrant_collection"] = old_collection
            old_state["serving_collection"] = old_collection
            old_state["serving_repo_slug"] = f"{repo_name}_old"
            old_state["active_repo_slug"] = old_state.get("active_repo_slug") or repo_name
            try:
                old_state["indexing_status"] = {"state": "idle"}
            except Exception:
                pass
            old_state["indexing_config_pending"] = None
            old_state["indexing_config_pending_hash"] = None
            old_state["indexing_env_pending"] = None
            old_state["staging"] = None
            update_workspace_state(
                workspace_path=str(old_dir),
                repo_name=f"{repo_name}_old",
                updates=old_state,
            )
        except Exception as exc:
            print(f"[staging] Warning: failed to write *_old state for {repo_name}: {exc}")

    # Prepare canonical slug for rebuild (pending env)
    pending_cfg = state.get("indexing_config_pending") or state.get("indexing_config")
    pending_hash = state.get("indexing_config_pending_hash") or state.get("indexing_config_hash")
    pending_env = state.get("indexing_env_pending") or dict(os.environ)
    env_hash = pending_hash or current_env_indexing_hash()

    if not pending_cfg and get_indexing_config_snapshot:
        pending_cfg = get_indexing_config_snapshot() if callable(get_indexing_config_snapshot) else get_indexing_config_snapshot
    if not pending_hash and pending_cfg and compute_indexing_config_hash:
        pending_hash = compute_indexing_config_hash(pending_cfg)

    if set_staging_state:
        staging_info = {
            "collection": old_collection,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "env_hash": pending_hash or env_hash,
            "indexing_config": pending_cfg,
            "indexing_config_hash": pending_hash,
            "environment": pending_env,
            "status": {"state": "initializing"},
            "workspace_path": root,
            "repo_name": repo_name,
        }
        set_staging_state(workspace_path=root, repo_name=repo_name, staging=staging_info)

    if update_staging_status:
        update_staging_status(
            workspace_path=root,
            repo_name=repo_name,
            status={
                "state": "initializing",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "progress": {"files_processed": 0, "total_files": None, "current_file": None},
            },
        )

    recreate_collection_qdrant(
        qdrant_url=os.environ.get("QDRANT_URL", "http://qdrant:6333"),
        api_key=os.environ.get("QDRANT_API_KEY") or None,
        collection=collection,
    )
    spawn_ingest_code(
        root=root,
        work_dir=work_dir,
        collection=collection,
        recreate=True,
        repo_name=repo_name,
        env_overrides=pending_env if isinstance(pending_env, dict) else None,
    )
    return collection


def activate_staging_rebuild(*, collection: str, work_dir: str) -> None:
    if not _staging_enabled():
        raise RuntimeError("Staging is disabled (set CTXCE_STAGING_ENABLED=1 to enable)")
    root, repo_name = resolve_collection_root(collection=collection, work_dir=work_dir)
    if not root:
        raise RuntimeError("No workspace mapping found for collection")
    state = _get_workspace_state_safe(root, repo_name)
    staging = state.get("staging") or {}

    # Idempotent: allow Activate to run even if staging metadata was lost,
    # as long as serving is still pointed at *_old (or the clone artifacts exist).
    staging_active = False
    try:
        if isinstance(staging, dict) and staging.get("collection"):
            staging_active = True
    except Exception:
        staging_active = False
    try:
        if str(state.get("serving_collection") or "").strip() == f"{collection}_old":
            staging_active = True
    except Exception:
        pass
    try:
        if str(state.get("serving_repo_slug") or "").strip().endswith("_old"):
            staging_active = True
    except Exception:
        pass

    if not staging_active:
        # Nothing to activate.
        return
    # This staging workflow serves reads from <collection>_old while recreating/reindexing
    # the primary <collection>. "Activate" means: switch serving back to the rebuilt primary
    # and remove the *_old clone.
    cleanup_kwargs = {
        "collection": collection,
        "repo_name": repo_name,
        "delete_collection": True,
        "work_dir": work_dir,
        "workspace_root": root,
    }

    try:
        # Reset serving state back to the primary collection and clear staging metadata.
        if update_workspace_state and repo_name:
            try:
                update_workspace_state(
                    workspace_path=root,
                    repo_name=repo_name,
                    updates={
                        "serving_collection": collection,
                        "serving_repo_slug": repo_name,
                        "active_repo_slug": repo_name,
                        "qdrant_collection": collection,
                    },
                )
            except Exception:
                pass

        if clear_staging_collection:
            try:
                clear_staging_collection(workspace_path=root, repo_name=repo_name)
            except Exception:
                pass
    finally:
        _cleanup_old_clone(**cleanup_kwargs)


def abort_staging_rebuild(
    *,
    collection: str,
    work_dir: str,
    delete_collection: bool = True,
) -> None:
    if not _staging_enabled():
        raise RuntimeError("Staging is disabled (set CTXCE_STAGING_ENABLED=1 to enable)")
    if not clear_staging_collection:
        raise RuntimeError("clear_staging_collection helper unavailable")
    root, repo_name = resolve_collection_root(collection=collection, work_dir=work_dir)
    if not root:
        raise RuntimeError("No workspace mapping found for collection")
    state = _get_workspace_state_safe(root, repo_name)
    staging = state.get("staging") or {}

    # Idempotent: allow Abort to run even if staging metadata was lost,
    # as long as serving is still pointed at *_old (or the clone artifacts exist).
    staging_active = False
    try:
        if isinstance(staging, dict) and staging.get("collection"):
            staging_active = True
    except Exception:
        staging_active = False
    try:
        if str(state.get("serving_collection") or "").strip() == f"{collection}_old":
            staging_active = True
    except Exception:
        pass
    try:
        if str(state.get("serving_repo_slug") or "").strip().endswith("_old"):
            staging_active = True
    except Exception:
        pass

    if not staging_active:
        # Nothing to abort.
        return

    # Staging rebuild serves traffic from <collection>_old while recreating <collection>.
    # Abort should restore serving state to the primary collection and remove the *_old clone.
    cleanup_kwargs = {
        "collection": collection,
        "repo_name": repo_name,
        "delete_collection": delete_collection,
        "work_dir": work_dir,
        "workspace_root": root,
    }

    try:
        # Reset serving state back to the primary collection.
        if update_workspace_state and repo_name:
            try:
                update_workspace_state(
                    workspace_path=root,
                    repo_name=repo_name,
                    updates={
                        "serving_collection": collection,
                        "serving_repo_slug": repo_name,
                        "active_repo_slug": repo_name,
                        "qdrant_collection": collection,
                    },
                )
            except Exception:
                pass

        clear_staging_collection(workspace_path=root, repo_name=repo_name)
    finally:
        _cleanup_old_clone(**cleanup_kwargs)
