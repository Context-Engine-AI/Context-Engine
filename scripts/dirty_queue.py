from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from qdrant_client import QdrantClient


# TODO(NEON_WARNING): VS Code "force reindex" uses standalone_upload_client.py --force, which treats all files as
# created and can generate repo-wide dirty ops (nullifying the dirty-queue fast-path). Consider separating "reindex"
# vs "force upload", and/or adding bulk-marker/threshold fallback; keep force-upload as an escape hatch when
# client/server state diverges.


def _env_truthy(val: str | None, default: bool) -> bool:
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class DirtyQueueHooks:
    cross_process_lock: Any | None
    get_collection_for_file: Callable[[Path], str] | None
    sanitize_vector_name: Callable[[str], str]
    ensure_collection_and_indexes_once: Callable[[QdrantClient, str, int, str], Any]
    index_single_file: Callable[..., bool]
    delete_points_by_path: Callable[[QdrantClient, str, str], Any]
    remove_cached_file: Callable[[str, str], Any] | None
    remove_cached_symbols: Callable[[str], Any] | None
    lex_vector_name: str
    mini_vector_name: str


def _pick_vector_name_for_collection(
    client: QdrantClient,
    collection: str,
    *,
    dim: int,
    model_name: str,
    hooks: DirtyQueueHooks,
) -> str:
    vn: str | None = None
    try:
        info = client.get_collection(collection)
        cfg = info.config.params.vectors
        if isinstance(cfg, dict) and cfg:
            # Prefer named vector whose size matches current embedding dim
            for name, params in cfg.items():
                psize = getattr(params, "size", None) or getattr(params, "dim", None)
                if psize and int(psize) == int(dim):
                    vn = name
                    break
            # Otherwise, if a lex vector exists, pick a different name as dense
            if vn is None and hooks.lex_vector_name in cfg:
                for name in cfg.keys():
                    if name != hooks.lex_vector_name:
                        vn = name
                        break
    except Exception:
        vn = None
    if vn is None:
        vn = hooks.sanitize_vector_name(model_name)
    return vn


def _dirty_queue_default_enabled(root: Path, *, is_multi_repo: bool) -> bool:
    if not is_multi_repo:
        return False

    work_dir = Path(os.environ.get("WORK_DIR") or os.environ.get("WORKDIR") or "/work")
    repos_dir = work_dir / ".codebase" / "repos"

    try:
        root_res = root.resolve()
    except Exception:
        root_res = root
    try:
        work_res = work_dir.resolve()
    except Exception:
        work_res = work_dir

    try:
        rel = root_res.relative_to(work_res)
    except Exception:
        return False
    if len(rel.parts) != 1:
        return False
    slug = rel.parts[0]
    if not slug:
        return False

    try:
        return (repos_dir / slug / ".ctxce_managed_upload").exists()
    except Exception:
        return False


def _dirty_queue_enabled(root: Path, *, is_multi_repo: bool) -> bool:
    return _env_truthy(
        os.environ.get("INDEX_DIRTY_QUEUE_ENABLED"),
        _dirty_queue_default_enabled(root, is_multi_repo=is_multi_repo),
    )


def _dirty_queue_force_full_scan() -> bool:
    return _env_truthy(os.environ.get("INDEX_DIRTY_QUEUE_FORCE_FULL_SCAN"), False)


def _drain_dirty_ops(root: Path, *, hooks: DirtyQueueHooks) -> list[dict[str, Any]] | None:
    work_dir = Path(os.environ.get("WORK_DIR") or os.environ.get("WORKDIR") or "/work")
    repos_dir = work_dir / ".codebase" / "repos"

    try:
        if not repos_dir.exists():
            return []
    except Exception:
        return None

    try:
        root_res = root.resolve()
    except Exception:
        root_res = root
    try:
        work_res = work_dir.resolve()
    except Exception:
        work_res = work_dir

    try:
        rel = root_res.relative_to(work_res)
    except Exception:
        return None
    if len(rel.parts) != 1:
        return None
    slug = rel.parts[0]
    if not slug:
        return None

    repo_dir = repos_dir / slug
    try:
        if not (repo_dir / ".ctxce_managed_upload").exists():
            return None
    except Exception:
        return None

    queue_path = repo_dir / "dirty_ops.jsonl"
    try:
        if not queue_path.exists():
            return []
    except Exception:
        return None

    lock_path = queue_path.with_suffix(queue_path.suffix + ".lock")

    lines: list[str] = []
    try:
        if hooks.cross_process_lock is not None:
            with hooks.cross_process_lock(lock_path):
                try:
                    with open(queue_path, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                except FileNotFoundError:
                    lines = []
                try:
                    with open(queue_path, "w", encoding="utf-8"):
                        pass
                except Exception:
                    pass
        else:
            try:
                with open(queue_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
            except FileNotFoundError:
                lines = []
            try:
                with open(queue_path, "w", encoding="utf-8"):
                    pass
            except Exception:
                pass
    except Exception:
        return None

    out: list[dict[str, Any]] = []
    for ln in lines:
        s = (ln or "").strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        rec = dict(obj)
        rec.setdefault("repo", slug)
        out.append(rec)

    return out


def _index_dirty_ops(
    ops: list[dict[str, Any]],
    *,
    root: Path,
    qdrant_url: str,
    api_key: str,
    collection: str,
    model_name: str,
    dedupe: bool,
    pseudo_mode: str,
    hooks: DirtyQueueHooks,
) -> None:
    try:
        from scripts.embedder import get_embedding_model, get_model_dimension

        model = get_embedding_model(model_name)
        dim = get_model_dimension(model_name)
    except ImportError:
        try:
            from fastembed import TextEmbedding  # type: ignore
        except Exception:
            try:
                print("[dirty_queue] fastembed not available; cannot index")
            except Exception:
                pass
            return
        model = TextEmbedding(model_name=model_name)
        dim = len(next(model.embed(["dimension probe"])))

    client = QdrantClient(
        url=qdrant_url,
        api_key=api_key or None,
        timeout=int(os.environ.get("QDRANT_TIMEOUT", "20") or 20),
    )

    work_dir = Path(os.environ.get("WORK_DIR") or os.environ.get("WORKDIR") or "/work")
    vector_name_by_collection: dict[str, str] = {}

    processed = 0
    deleted = 0
    moved = 0
    indexed = 0

    for rec in ops:
        processed += 1
        op = (rec.get("op") or "").strip().lower()
        repo = (rec.get("repo") or "").strip()
        rel_path = rec.get("path")
        if not repo or not isinstance(rel_path, str) or not rel_path:
            continue

        repo_root = work_dir / repo
        file_path = repo_root / rel_path

        current_collection = collection
        try:
            if hooks.get_collection_for_file:
                current_collection = hooks.get_collection_for_file(file_path)
        except Exception:
            current_collection = collection
        if not current_collection:
            continue

        if op == "deleted":
            try:
                hooks.delete_points_by_path(client, current_collection, str(file_path))
            except Exception:
                pass
            try:
                if hooks.remove_cached_file:
                    hooks.remove_cached_file(str(file_path), repo)
            except Exception:
                pass
            try:
                if hooks.remove_cached_symbols:
                    hooks.remove_cached_symbols(str(file_path))
            except Exception:
                pass
            deleted += 1
            continue

        if op == "moved":
            source_rel = rec.get("source_path") or rec.get("source_relative_path")
            if isinstance(source_rel, str) and source_rel:
                old_path = repo_root / source_rel
                try:
                    hooks.delete_points_by_path(client, current_collection, str(old_path))
                except Exception:
                    pass
                try:
                    if hooks.remove_cached_file:
                        hooks.remove_cached_file(str(old_path), repo)
                except Exception:
                    pass
                try:
                    if hooks.remove_cached_symbols:
                        hooks.remove_cached_symbols(str(old_path))
                except Exception:
                    pass
            moved += 1

        try:
            if not file_path.exists():
                continue
        except Exception:
            continue

        vn = vector_name_by_collection.get(current_collection)
        if not vn:
            vn = _pick_vector_name_for_collection(
                client,
                current_collection,
                dim=dim,
                model_name=model_name,
                hooks=hooks,
            )
            vector_name_by_collection[current_collection] = vn

        try:
            hooks.ensure_collection_and_indexes_once(client, current_collection, dim, vn)
        except Exception:
            continue

        try:
            ok = hooks.index_single_file(
                client,
                model,
                current_collection,
                vector_name_by_collection[current_collection],
                file_path,
                dedupe=dedupe,
                skip_unchanged=True,
                pseudo_mode=pseudo_mode,
                repo_name_for_cache=repo,
            )
            if ok:
                indexed += 1
        except Exception:
            pass

    try:
        print(
            f"[dirty_queue] processed={processed} indexed={indexed} deleted={deleted} moved={moved}"
        )
    except Exception:
        pass


def maybe_process_dirty_queue(
    *,
    root: Path,
    qdrant_url: str,
    api_key: str,
    collection: str,
    model_name: str,
    recreate: bool,
    dedupe: bool,
    skip_unchanged: bool,
    pseudo_mode: str,
    is_multi_repo: bool,
    hooks: DirtyQueueHooks,
) -> str:
    if recreate or not skip_unchanged:
        return "fallback_scan"
    if _dirty_queue_force_full_scan():
        return "fallback_scan"
    if not _dirty_queue_enabled(root, is_multi_repo=is_multi_repo):
        return "fallback_scan"

    ops = _drain_dirty_ops(root, hooks=hooks)
    if ops is None:
        try:
            print("[dirty_queue] Failed to drain queue; falling back to full scan")
        except Exception:
            pass
        return "fallback_scan"

    if not ops:
        try:
            print("[dirty_queue] No pending ops; skipping full scan")
        except Exception:
            pass
        return "no_op"

    _index_dirty_ops(
        ops,
        root=root,
        qdrant_url=qdrant_url,
        api_key=api_key,
        collection=collection,
        model_name=model_name,
        dedupe=dedupe,
        pseudo_mode=pseudo_mode,
        hooks=hooks,
    )
    return "processed"
