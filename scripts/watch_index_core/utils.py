"""Misc utilities shared across watch_index_core modules."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Type
from watchdog.observers import Observer

import scripts.ingest_code as idx
from .config import LOGGER, ROOT, default_collection_name
from scripts.workspace_state import (
    _extract_repo_name_from_path,
    PLACEHOLDER_COLLECTION_NAMES,
    ensure_logical_repo_id,
    find_collection_for_logical_repo,
    get_collection_name,
    get_workspace_state,
    is_multi_repo_mode,
    logical_repo_reuse_enabled,
    update_workspace_state,
)


def safe_print(*args: Any, **kwargs: Any) -> None:
    """Best-effort print that swallows IO errors."""
    try:
        print(*args, **kwargs)
    except Exception:
        pass


def get_boolean_env(name: str, default: bool = False) -> bool:
    """Parse a boolean environment variable."""
    val = os.environ.get(name)
    if val is None:
        return default
    return str(val).strip().lower() in {"1", "true", "yes", "on"}


def resolve_vector_name_config(
    client,
    collection_name: str,
    model_dim: int,
    model_name: str,
) -> str:
    """Resolve the vector name to use for the collection."""
    try:
        info = client.get_collection(collection_name)
        cfg = info.config.params.vectors
        if isinstance(cfg, dict) and cfg:
            vector_name = None
            for name, params in cfg.items():
                psize = getattr(params, "size", None) or getattr(params, "dim", None)
                if psize and int(psize) == int(model_dim):
                    vector_name = name
                    break
            if vector_name is None and getattr(idx, "LEX_VECTOR_NAME", None) in cfg:
                for name in cfg.keys():
                    if name != idx.LEX_VECTOR_NAME:
                        vector_name = name
                        break
            if vector_name is None:
                vector_name = idx._sanitize_vector_name(model_name)
        else:
            vector_name = idx._sanitize_vector_name(model_name)
    except Exception:
        vector_name = idx._sanitize_vector_name(model_name)
    return vector_name


def create_observer(use_polling: bool, observer_cls: Type[Observer] = Observer) -> Observer:
    """Create a watchdog observer based on configuration."""
    if use_polling:
        try:
            from watchdog.observers.polling import PollingObserver  # type: ignore

            obs = PollingObserver()
            try:
                safe_print("[watch_mode] Using polling observer for filesystem events")
            except Exception:
                pass
            return obs
        except Exception:
            try:
                safe_print(
                    "[watch_mode] Polling observer unavailable, falling back to default Observer"
                )
            except Exception:
                pass
    return observer_cls()


def _detect_repo_for_file(file_path: Path) -> Optional[Path]:
    """Detect repository root for a file under WATCH root."""
    try:
        rel_path = file_path.resolve().relative_to(ROOT.resolve())
    except Exception:
        return None
    if not rel_path.parts:
        return ROOT
    return ROOT / rel_path.parts[0]


def _repo_name_or_none(repo_path: Optional[Path]) -> Optional[str]:
    """Extract repo name from path, or None if path is None."""
    return _extract_repo_name_from_path(str(repo_path)) if repo_path else None


def _get_collection_for_repo(repo_path: Path) -> str:
    """Resolve Qdrant collection for a repo, with logical_repo_id-aware reuse.

    In multi-repo mode, prefer reusing an existing canonical collection that has
    already been associated with this logical repository (same git common dir)
    by consulting workspace_state. Falls back to the legacy per-repo hashed
    collection naming when no mapping exists.
    """

    default_coll = default_collection_name()
    try:
        repo_name = _extract_repo_name_from_path(str(repo_path))
    except Exception:
        repo_name = None

    # Multi-repo: always honor explicit serving/qdrant collection from state when present.
    # This is required for staging/migration workflows (e.g. *_old repos) even when
    # logical-repo reuse is disabled.
    if repo_name and is_multi_repo_mode():
        workspace_root = (
            os.environ.get("WORKSPACE_PATH")
            or os.environ.get("WATCH_ROOT")
            or "/work"
        )
        try:
            ws_root_path = Path(workspace_root).resolve()
        except Exception:
            ws_root_path = Path(workspace_root)
        ws_path = str((ws_root_path / repo_name).resolve())

        state: Dict[str, Any]
        try:
            state = get_workspace_state(ws_path, repo_name) or {}
        except Exception:
            state = {}

        if isinstance(state, dict):
            serving_coll = state.get("serving_collection") or state.get("qdrant_collection")
            if isinstance(serving_coll, str) and serving_coll:
                # In multi-repo mode the watcher must not route per-repo events into
                # a workspace-level placeholder collection (e.g. "codebase").
                # Older refactors could persist this into state.json; treat it as invalid
                # so we re-derive a deterministic per-repo collection.
                invalid = {default_coll}
                try:
                    invalid.update(set(PLACEHOLDER_COLLECTION_NAMES or []))
                except Exception:
                    pass

                if serving_coll not in invalid:
                    return serving_coll

        # Multi-repo: try to reuse a canonical collection based on logical_repo_id
        if logical_repo_reuse_enabled():
            try:
                state = ensure_logical_repo_id(state, ws_path)
            except Exception:
                pass

            lrid = state.get("logical_repo_id")
            if isinstance(lrid, str) and lrid:
                coll: Optional[str]
                try:
                    coll = find_collection_for_logical_repo(
                        lrid, search_root=str(ws_root_path)
                    )
                except Exception:
                    coll = None
                if isinstance(coll, str) and coll:
                    try:
                        update_workspace_state(
                            workspace_path=ws_path,
                            updates={"qdrant_collection": coll, "logical_repo_id": lrid},
                            repo_name=repo_name,
                        )
                    except Exception:
                        pass
                    return coll

        # Legacy behaviour: derive per-repo collection name
        try:
            derived = get_collection_name(repo_name)
            # Best-effort: if state was previously pointing at a placeholder collection,
            # persist the corrected mapping so future runs don't regress.
            try:
                update_workspace_state(
                    workspace_path=ws_path,
                    updates={"qdrant_collection": derived, "serving_collection": derived},
                    repo_name=repo_name,
                )
            except Exception:
                pass
            return derived
        except Exception:
            return default_coll

    # Single-repo mode or repo_name detection failed: use existing helpers/env
    try:
        if repo_name:
            return get_collection_name(repo_name)
    except Exception:
        pass
    return default_coll


def _get_collection_for_file(file_path: Path) -> str:
    if not is_multi_repo_mode():
        return default_collection_name()
    repo_path = _detect_repo_for_file(file_path)
    if repo_path is not None:
        return _get_collection_for_repo(repo_path)
    return default_collection_name()


def safe_log_error(logger, message: str, extra: dict | None = None) -> None:
    """Safely log an error with optional extra context, suppressing any logging failures."""
    try:
        logger.error(message, extra=extra or {}, exc_info=True)
    except Exception:
        pass


__all__ = [
    "safe_print",
    "safe_log_error",
    "get_boolean_env",
    "resolve_vector_name_config",
    "create_observer",
    "_detect_repo_for_file",
    "_get_collection_for_repo",
    "_get_collection_for_file",
    "_repo_name_or_none",
]
