#!/usr/bin/env python3
import os
import time
import threading
import json
import subprocess
from pathlib import Path
from typing import Optional, Set, Dict, List, Any

from qdrant_client import QdrantClient, models

# watcher
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Ensure project root is on sys.path when run as a script (so 'scripts' can be imported)
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.workspace_state import (
    _extract_repo_name_from_path,
    get_collection_name,
    _get_global_state_dir,
    is_multi_repo_mode,
    get_cached_file_hash,
    set_cached_file_hash,
    remove_cached_file,
    update_indexing_status,
    update_workspace_state,
    get_workspace_state,
    ensure_logical_repo_id,
    find_collection_for_logical_repo,
    logical_repo_reuse_enabled,
    _get_repo_state_dir,
    _cross_process_lock,
    get_collection_mappings,
)
import hashlib
from datetime import datetime

import scripts.ingest_code as idx
from scripts.logger import get_logger


try:
    logger = get_logger(__name__)
except Exception:  # pragma: no cover - fallback for logger import issues
    import logging

    logger = logging.getLogger(__name__)

QDRANT_URL = os.environ.get("QDRANT_URL", "http://qdrant:6333")
MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
ROOT = Path(os.environ.get("WATCH_ROOT", "/work")).resolve()

# Back-compat: legacy modules/tests expect a module-level COLLECTION constant.
# It will be updated in main() once the resolved collection is known.
COLLECTION = os.environ.get("COLLECTION_NAME", "my-collection")

# Debounce interval
DELAY_SECS = float(os.environ.get("WATCH_DEBOUNCE_SECS", "1.0"))

# Performance tuning: bulk indexing threshold and batch upsert size
BULK_INDEX_THRESHOLD = int(os.environ.get("WATCH_BULK_INDEX_THRESHOLD", "50"))
BATCH_UPSERT_SIZE = int(os.environ.get("WATCH_BATCH_UPSERT_SIZE", "100"))


def _detect_repo_for_file(file_path: Path) -> Optional[Path]:
    """Detect repository root for a file under WATCH root."""
    try:
        rel_path = file_path.resolve().relative_to(ROOT.resolve())
    except Exception:
        return None
    if not rel_path.parts:
        return ROOT
    return ROOT / rel_path.parts[0]


def _get_collection_for_repo(repo_path: Path) -> str:
    """Resolve Qdrant collection for a repo, with logical_repo_id-aware reuse.

    In multi-repo mode, prefer reusing an existing canonical collection that has
    already been associated with this logical repository (same git common dir)
    by consulting workspace_state. Falls back to the legacy per-repo hashed
    collection naming when no mapping exists.
    """
    default_coll = os.environ.get("COLLECTION_NAME", "my-collection")
    try:
        repo_name = _extract_repo_name_from_path(str(repo_path))
    except Exception:
        repo_name = None

    # Multi-repo: try to reuse a canonical collection based on logical_repo_id
    if repo_name and is_multi_repo_mode() and logical_repo_reuse_enabled():
        workspace_root = os.environ.get("WORKSPACE_PATH") or os.environ.get("WATCH_ROOT") or "/work"
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
            try:
                state = ensure_logical_repo_id(state, ws_path)
            except Exception:
                pass
            lrid = state.get("logical_repo_id")
            if isinstance(lrid, str) and lrid:
                coll: Optional[str]
                try:
                    coll = find_collection_for_logical_repo(lrid, search_root=str(ws_root_path))
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

            # Fallback to any explicit collection stored in state for this repo
            coll2 = state.get("qdrant_collection")
            if isinstance(coll2, str) and coll2:
                return coll2

        # Legacy behaviour: derive per-repo collection name
        try:
            return get_collection_name(repo_name)
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
        return os.environ.get("COLLECTION_NAME", "my-collection")
    repo_path = _detect_repo_for_file(file_path)
    if repo_path is not None:
        return _get_collection_for_repo(repo_path)
    return os.environ.get("COLLECTION_NAME", "my-collection")


class ChangeQueue:
    def __init__(self, process_cb):
        self._lock = threading.Lock()
        self._paths: Set[Path] = set()
        self._pending: Set[Path] = set()
        self._timer: threading.Timer | None = None
        self._process_cb = process_cb
        # Serialize processing to avoid concurrent use of TextEmbedding/QdrantClient
        self._processing_lock = threading.Lock()

    def add(self, p: Path):
        with self._lock:
            self._paths.add(p)
            if self._timer is not None:
                try:
                    self._timer.cancel()
                except Exception as e:
                    logger.error(
                        "Failed to cancel timer in ChangeQueue.add",
                        extra={"error": str(e)},
                    )
            self._timer = threading.Timer(DELAY_SECS, self._flush)
            self._timer.daemon = True
            self._timer.start()

    def _flush(self):
        # Grab current batch
        with self._lock:
            paths = list(self._paths)
            self._paths.clear()
            self._timer = None
        # Try to run the processor exclusively; if busy, queue and return
        if not self._processing_lock.acquire(blocking=False):
            with self._lock:
                self._pending.update(paths)
                if self._timer is None:
                    # schedule a follow-up flush to pick up pending when free
                    self._timer = threading.Timer(DELAY_SECS, self._flush)
                    self._timer.daemon = True
                    self._timer.start()
            return
        try:
            todo = paths
            while True:
                try:
                    self._process_cb(list(todo))
                except Exception as e:
                    try:
                        print(f"[watcher_error] processing batch failed: {e}")
                    except Exception as inner_e:  # pragma: no cover - logging fallback
                        logger.error(
                            "Exception in ChangeQueue._flush during batch processing",
                            extra={"error": str(inner_e)},
                        )
                # drain any pending accumulated during processing
                with self._lock:
                    if not self._pending:
                        break
                    todo = list(self._pending)
                    self._pending.clear()
        finally:
            self._processing_lock.release()


class IndexHandler(FileSystemEventHandler):
    def __init__(
        self,
        root: Path,
        queue: ChangeQueue,
        client: Optional[QdrantClient],
        default_collection: Optional[str] = None,
        *,
        collection: Optional[str] = None,
    ):
        super().__init__()
        self.root = root
        self.queue = queue
        self.client = client
        resolved_collection = collection if collection is not None else default_collection
        self.default_collection = resolved_collection
        # In multi-repo mode, per-file collections are resolved via _get_collection_for_file.
        # Avoid using a root-level default collection (e.g., "/work-<hash>") for data ops.
        if is_multi_repo_mode():
            self.collection = None
        else:
            self.collection = resolved_collection
        self.excl = idx._Excluder(root)
        # Track ignore file for live reloads
        try:
            ig_name = os.environ.get("QDRANT_IGNORE_FILE", ".qdrantignore")
            self._ignore_path = (self.root / ig_name).resolve()
        except (OSError, ValueError) as e:
            try:
                print(f"[ignore_file] Could not resolve ignore file path: {e}")
            except Exception:
                pass
            self._ignore_path = None
        try:
            self._ignore_mtime = (
                self._ignore_path.stat().st_mtime
                if self._ignore_path and self._ignore_path.exists()
                else 0.0
            )
        except Exception:
            self._ignore_mtime = 0.0

    def _maybe_reload_excluder(self):
        try:
            if not self._ignore_path:
                return
            cur = (
                self._ignore_path.stat().st_mtime if self._ignore_path.exists() else 0.0
            )
            if cur != self._ignore_mtime:
                self.excl = idx._Excluder(self.root)
                self._ignore_mtime = cur
                try:
                    print(f"[ignore_reload] reloaded patterns from {self._ignore_path}")
                except Exception:
                    pass
        except Exception:
            pass

    def _maybe_enqueue(self, src_path: str):
        # Refresh ignore patterns if the file changed
        self._maybe_reload_excluder()
        p = Path(src_path)
        try:
            p = p.resolve()
        except Exception:
            return
        # skip directories
        if p.is_dir():
            return
        # ensure file is under root
        try:
            rel = p.resolve().relative_to(self.root.resolve())
        except ValueError:
            return

        try:
            if _get_global_state_dir is not None:
                global_state_dir = _get_global_state_dir()
                if p.is_relative_to(global_state_dir):
                    return
        except (OSError, ValueError):
            pass

        if any(part == ".codebase" for part in p.parts):
            return

        # Git history manifests are handled by a separate ingestion pipeline and should still
        # be processed even when .remote-git is excluded from code indexing.
        if any(part == ".remote-git" for part in p.parts) and p.suffix.lower() == ".json":
            self.queue.add(p)
            return

        # directory-level excludes (parent dir)
        rel_dir = "/" + str(rel.parent).replace(os.sep, "/")
        if rel_dir == "/.":
            rel_dir = "/"
        if self.excl.exclude_dir(rel_dir):
            return
        # only code files
        if p.suffix.lower() not in idx.CODE_EXTS:
            return
        # file-level excludes
        relf = (rel_dir.rstrip("/") + "/" + p.name).replace("//", "/")
        if self.excl.exclude_file(relf):
            return
        self.queue.add(p)

    def on_modified(self, event):
        if not event.is_directory:
            self._maybe_enqueue(event.src_path)

    def on_created(self, event):
        if not event.is_directory:
            self._maybe_enqueue(event.src_path)

    def on_deleted(self, event):
        if event.is_directory:
            return
        try:
            p = Path(event.src_path).resolve()
        except Exception:
            return
        if any(part == ".codebase" for part in p.parts):
            return
        # Only attempt deletion for code files we would have indexed
        if p.suffix.lower() not in idx.CODE_EXTS:
            return
        if self.client is not None:
            try:
                if is_multi_repo_mode():
                    collection = _get_collection_for_file(p)
                else:
                    collection = self.collection or _get_collection_for_file(p)
                idx.delete_points_by_path(self.client, collection, str(p))
                print(f"[deleted] {p} -> {collection}")
            except Exception:
                pass
        else:
            print(f"File deletion detected: {p}")

        try:
            repo_path = _detect_repo_for_file(p)
            if repo_path:
                repo_name = _extract_repo_name_from_path(str(repo_path))
                remove_cached_file(str(p), repo_name)

                # Remove symbol cache entry
                try:
                    from scripts.workspace_state import remove_cached_symbols
                    remove_cached_symbols(str(p))
                    print(f"[deleted_symbol_cache] {p}")
                except Exception as e:
                    print(f"[symbol_cache_delete_error] {p}: {e}")
            else:
                root_repo_name = _extract_repo_name_from_path(str(self.root))
                remove_cached_file(str(p), root_repo_name)

                # Remove symbol cache entry (single repo mode)
                try:
                    from scripts.workspace_state import remove_cached_symbols
                    remove_cached_symbols(str(p))
                    print(f"[deleted_symbol_cache] {p}")
                except Exception as e:
                    print(f"[symbol_cache_delete_error] {p}: {e}")
        except Exception:
            pass

        try:
            repo_path = _detect_repo_for_file(p) or self.root
            _log_activity(str(repo_path), "deleted", p)
        except Exception as e:
            try:
                print(f"[delete_error] {p}: {e}")
            except Exception:
                pass

    def on_moved(self, event):
        if event.is_directory:
            return
        # Attempt optimized rename when content unchanged; else fallback to reindex
        try:
            src = Path(event.src_path).resolve()
            dest = Path(event.dest_path).resolve()
        except Exception:
            return
        # Only react to code files
        if (
            dest.suffix.lower() not in idx.CODE_EXTS
            and src.suffix.lower() not in idx.CODE_EXTS
        ):
            return
        # If destination directory is ignored, treat as simple deletion
        try:
            rel_dir = "/" + str(
                dest.parent.resolve().relative_to(self.root.resolve())
            ).replace(os.sep, "/")
            if rel_dir == "/.":
                rel_dir = "/"
            if self.excl.exclude_dir(rel_dir):
                if src.suffix.lower() in idx.CODE_EXTS:
                    try:
                        if is_multi_repo_mode():
                            coll = _get_collection_for_file(src)
                        else:
                            coll = self.collection or _get_collection_for_file(src)
                        idx.delete_points_by_path(
                            self.client, coll, str(src)
                        )
                        print(f"[moved:ignored_dest_deleted_src] {src} -> {dest}")
                        try:
                            src_repo_path = _detect_repo_for_file(src)
                            src_repo_name = (
                                _extract_repo_name_from_path(str(src_repo_path))
                                if src_repo_path is not None
                                else None
                            )
                            remove_cached_file(str(src), src_repo_name)
                        except Exception:
                            pass

                    except Exception:
                        pass
                return
        except Exception:
            pass
        src_collection = _get_collection_for_file(src)
        dest_collection = _get_collection_for_file(dest)
        is_cross_collection = src_collection != dest_collection
        if is_cross_collection:
            print(f"[cross_collection_move] {src} -> {dest}")

        moved_count = -1
        renamed_hash: str | None = None
        if self.client is not None:
            try:
                moved_count, renamed_hash = _rename_in_store(
                    self.client, src_collection, src, dest, dest_collection
                )
            except Exception:
                moved_count, renamed_hash = -1, None
        if moved_count and moved_count > 0:
            try:
                print(
                    f"[moved] {src} -> {dest} ({moved_count} chunk(s) relinked)"
                )
                src_repo_path = _detect_repo_for_file(src)
                dest_repo_path = _detect_repo_for_file(dest)
                src_repo_name = (
                    _extract_repo_name_from_path(str(src_repo_path))
                    if src_repo_path is not None
                    else None
                )
                dest_repo_name = (
                    _extract_repo_name_from_path(str(dest_repo_path))
                    if dest_repo_path is not None
                    else None
                )
                src_hash = ""
                if src_repo_name:
                    src_hash = get_cached_file_hash(str(src), src_repo_name)
                    remove_cached_file(str(src), src_repo_name)
                if not src_hash and renamed_hash:
                    src_hash = renamed_hash
                if dest_repo_name and src_hash:
                    set_cached_file_hash(
                        str(dest), src_hash, dest_repo_name
                    )
            except Exception:
                pass
            try:
                _log_activity(
                    str(dest_repo_path or self.root),
                    "moved",
                    dest,
                    {"from": str(src), "chunks": int(moved_count)},
                )
            except Exception:
                pass
            return
        if self.client is not None:
            try:
                if src.suffix.lower() in idx.CODE_EXTS:
                    try:
                        idx.delete_points_by_path(self.client, src_collection, str(src))
                    except Exception:
                        # In multi-repo mode, avoid falling back to any root-level collection.
                        if (not is_multi_repo_mode()) and self.collection:
                            idx.delete_points_by_path(
                                self.client,
                                self.collection,
                                str(src),
                            )
                        else:
                            raise
                    print(f"[moved:deleted_src] {src}")
            except Exception:
                pass
        else:
            print(f"[remote_mode] Move detected: {src} -> {dest}")
        try:
            self._maybe_enqueue(str(dest))
        except Exception:
            pass


# --- Workspace state helpers ---
def _set_status_indexing(workspace_path: str, total_files: int) -> None:
    try:
        repo_name = _extract_repo_name_from_path(workspace_path)
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


def _update_progress(
    workspace_path: str,
    started_at: str,
    processed: int,
    total: int,
    current_file: Path | None,
) -> None:
    try:
        repo_name = _extract_repo_name_from_path(workspace_path)
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


def _log_activity(
    workspace_path: str, action: str, file_path: Path, details: dict | None = None
) -> None:
    try:
        repo_name = _extract_repo_name_from_path(workspace_path)
        from scripts.workspace_state import log_activity

        valid_actions = {"indexed", "deleted", "skipped", "scan-completed", "initialized", "moved"}
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


# --- Move/Rename optimization: reuse vectors when file content unchanged ---
def _rename_in_store(
    client: QdrantClient,
    src_collection: str,
    src: Path,
    dest: Path,
    dest_collection: Optional[str] = None,
) -> tuple[int, str | None]:
    """Best-effort: if dest content hash matches previously indexed src hash,
    update points in-place to the new path without re-embedding.

    Returns number of points moved, or -1 if not applicable/failure.
    """
    if dest_collection is None:
        dest_collection = src_collection
    try:
        if not dest.exists() or dest.is_dir():
            return -1
        try:
            text = dest.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return -1
        dest_hash = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()
        prev = idx.get_indexed_file_hash(client, src_collection, str(src))
        logger.debug(
            "rename fast-path candidate src=%s dest=%s prev_hash=%s dest_hash=%s",
            str(src),
            str(dest),
            prev,
            dest_hash,
        )
        if not prev or prev != dest_hash:
            return -1, prev if prev else None

        moved = 0
        next_offset = None
        while True:
            filt = models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.path", match=models.MatchValue(value=str(src))
                    )
                ]
            )
            points, next_offset = client.scroll(
                collection_name=src_collection,
                scroll_filter=filt,
                with_payload=True,
                with_vectors=True,
                limit=256,
                offset=next_offset,
            )
            if not points:
                break
            new_points = []
            for rec in points:
                payload = rec.payload or {}
                md = payload.get("metadata") or {}
                code = md.get("code") or ""
                try:
                    start_line = int(md.get("start_line") or 1)
                    end_line = int(md.get("end_line") or start_line)
                except Exception:
                    start_line, end_line = 1, 1
                new_id = idx.hash_id(code, str(dest), start_line, end_line)

                # Update metadata path fields
                new_md = dict(md)
                new_md["path"] = str(dest)
                new_md["path_prefix"] = str(dest.parent)
                # Recompute dual-path hints
                cur_path = str(dest)
                host_root = (
                    str(os.environ.get("HOST_INDEX_PATH") or "").strip().rstrip("/")
                )
                if ":" in host_root: # Windows drive letter (e.g., "C:")
                    host_root = ""
                host_path = None
                container_path = None
                try:
                    if cur_path.startswith("/work/") and host_root:
                        rel = cur_path[len("/work/") :]
                        host_path = os.path.realpath(os.path.join(host_root, rel))
                        container_path = cur_path
                    else:
                        host_path = cur_path
                        if host_root and cur_path.startswith(host_root + "/"):
                            rel = cur_path[len(host_root) + 1 :]
                            container_path = "/work/" + rel
                except Exception:
                    host_path = cur_path
                    container_path = cur_path if cur_path.startswith("/work/") else None
                new_md["host_path"] = host_path
                new_md["container_path"] = container_path

                new_payload = dict(payload)
                new_payload["metadata"] = new_md

                vec = rec.vector  # Named or unnamed vector(s)
                try:
                    new_points.append(
                        models.PointStruct(id=new_id, vector=vec, payload=new_payload)
                    )
                except Exception:
                    continue
            if new_points:
                logger.debug(
                    "rename fast-path upserting %d chunk(s) %s -> %s into %s",
                    len(new_points),
                    str(src),
                    str(dest),
                    dest_collection,
                )
                idx.upsert_points(client, dest_collection, new_points)
                moved += len(new_points)
            if next_offset is None:
                break

        try:
            idx.delete_points_by_path(client, src_collection, str(src))
        except Exception:
            pass
        return moved, dest_hash
    except Exception as exc:
        try:
            logger.warning(
                "[rename_debug] rename failed for %s -> %s: %s",
                str(src),
                str(dest),
                exc,
            )
        except Exception:
            pass
        return -1, None


def _start_pseudo_backfill_worker(
    client: QdrantClient,
    default_collection: str,
    model_dim: int,
    vector_name: str,
) -> None:
    flag = (os.environ.get("PSEUDO_BACKFILL_ENABLED") or "").strip().lower()
    if flag not in {"1", "true", "yes", "on"}:
        return

    try:
        interval = float(os.environ.get("PSEUDO_BACKFILL_TICK_SECS", "60") or 60.0)
    except Exception:
        interval = 60.0
    if interval <= 0:
        return
    try:
        max_points = int(os.environ.get("PSEUDO_BACKFILL_MAX_POINTS", "256") or 256)
    except Exception:
        max_points = 256
    if max_points <= 0:
        max_points = 1

    def _worker() -> None:
        while True:
            try:
                try:
                    mappings = get_collection_mappings(search_root=str(ROOT))
                except Exception:
                    mappings = []
                if not mappings:
                    mappings = [
                        {"repo_name": None, "collection_name": default_collection},
                    ]
                for mapping in mappings:
                    coll = mapping.get("collection_name") or default_collection
                    repo_name = mapping.get("repo_name")
                    if not coll:
                        continue
                    try:
                        if is_multi_repo_mode() and repo_name:
                            state_dir = _get_repo_state_dir(repo_name)
                        else:
                            state_dir = _get_global_state_dir(str(ROOT))
                        lock_path = state_dir / "pseudo.lock"
                        with _cross_process_lock(lock_path):
                            processed = idx.pseudo_backfill_tick(
                                client,
                                coll,
                                repo_name=repo_name,
                                max_points=max_points,
                                dim=model_dim,
                                vector_name=vector_name,
                            )
                            if processed:
                                try:
                                    print(
                                        f"[pseudo_backfill] repo={repo_name or 'default'} collection={coll} processed={processed}"
                                    )
                                except Exception:
                                    pass
                    except Exception as e:
                        try:
                            print(
                                f"[pseudo_backfill] error repo={repo_name or 'default'} collection={coll}: {e}"
                            )
                        except Exception:
                            pass
            except Exception:
                pass
            time.sleep(interval)

    thread = threading.Thread(target=_worker, name="pseudo-backfill", daemon=True)
    thread.start()


def main():
    # Resolve collection name from workspace state before any client/state ops
    try:
        from scripts.workspace_state import get_collection_name as _get_coll
    except Exception:
        _get_coll = None

    multi_repo_enabled = False
    try:
        multi_repo_enabled = bool(is_multi_repo_mode())
    except Exception:
        multi_repo_enabled = False

    default_collection = os.environ.get("COLLECTION_NAME", "my-collection")
    # In multi-repo mode, per-repo collections are resolved via _get_collection_for_file
    # and workspace_state; avoid deriving a root-level collection like "/work-<hash>".
    if _get_coll and not multi_repo_enabled:
        try:
            resolved = _get_coll(str(ROOT))
            if resolved:
                default_collection = resolved
        except Exception:
            pass
    if multi_repo_enabled:
        print("[multi_repo] Multi-repo mode enabled - per-repo collections in use")
    else:
        print("[single_repo] Single-repo mode enabled - using single collection")

    global COLLECTION
    COLLECTION = default_collection

    print(
        f"Watch mode: root={ROOT} qdrant={QDRANT_URL} collection={default_collection} model={MODEL}"
    )

    # Health check: detect and auto-heal cache/collection sync issues
    try:
        from scripts.collection_health import auto_heal_if_needed

        print("[health_check] Checking collection health...")
        heal_result = auto_heal_if_needed(
            str(ROOT), default_collection, QDRANT_URL, dry_run=False
        )
        if heal_result.get("action_taken") == "cleared_cache":
            print("[health_check] Cache cleared due to sync issue - files will be reindexed")
        elif not heal_result.get("health_check", {}).get("healthy", True):
            print(
                f"[health_check] Issue detected: {heal_result['health_check'].get('issue', 'unknown')}"
            )
        else:
            print("[health_check] Collection health OK")
    except Exception as e:
        print(f"[health_check] Warning: health check failed: {e}")

    client = QdrantClient(
        url=QDRANT_URL, timeout=int(os.environ.get("QDRANT_TIMEOUT", "20") or 20)
    )

    # Use centralized embedder factory if available (supports Qwen3 feature flag)
    try:
        from scripts.embedder import get_embedding_model, get_model_dimension
        model = get_embedding_model(MODEL)
        model_dim = get_model_dimension(MODEL)
    except ImportError:
        # Fallback to direct fastembed initialization
        from fastembed import TextEmbedding
        model = TextEmbedding(model_name=MODEL)
        model_dim = len(next(model.embed(["dimension probe"])))

    try:
        info = client.get_collection(default_collection)
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
                vector_name = idx._sanitize_vector_name(MODEL)
        else:
            vector_name = idx._sanitize_vector_name(MODEL)
    except Exception:
        vector_name = idx._sanitize_vector_name(MODEL)

    try:
        idx.ensure_collection_and_indexes_once(
            client, default_collection, model_dim, vector_name
        )
    except Exception:
        pass

    _start_pseudo_backfill_worker(client, default_collection, model_dim, vector_name)

    try:
        if multi_repo_enabled:
            root_repo_name = _extract_repo_name_from_path(str(ROOT))
            if root_repo_name:
                root_collection = get_collection_name(root_repo_name)
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
            update_workspace_state(
                workspace_path=str(ROOT),
                updates={"qdrant_collection": default_collection},
            )
            update_indexing_status(status={"state": "watching"})
    except Exception as e:
        print(f"[workspace_state] Error initializing workspace state: {e}")

    q = ChangeQueue(
        lambda paths: _process_paths(
            paths, client, model, vector_name, model_dim, str(ROOT)
        )
    )
    handler = IndexHandler(ROOT, q, client, default_collection)

    use_polling = (os.environ.get("WATCH_USE_POLLING") or "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    if use_polling:
        try:
            from watchdog.observers.polling import PollingObserver  # type: ignore

            obs = PollingObserver()
            try:
                print("[watch_mode] Using polling observer for filesystem events")
            except Exception:
                pass
        except Exception:
            obs = Observer()
            try:
                print(
                    "[watch_mode] Polling observer unavailable, falling back to default Observer"
                )
            except Exception:
                pass
    else:
        obs = Observer()
    obs.schedule(handler, str(ROOT), recursive=True)
    obs.start()

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        obs.stop()
        obs.join()


def _process_git_history_manifest(
    p: Path,
    client,
    model,
    collection: str,
    vector_name: str,
    repo_name: Optional[str],
):
    try:
        import sys

        script = ROOT_DIR / "scripts" / "ingest_history.py"
        if not script.exists():
            return
        cmd = [sys.executable or "python3", str(script), "--manifest-json", str(p)]
        env = os.environ.copy()
        if collection:
            env["COLLECTION_NAME"] = collection
        if QDRANT_URL:
            env["QDRANT_URL"] = QDRANT_URL
        if repo_name:
            env["REPO_NAME"] = repo_name
        try:
            print(
                f"[git_history_manifest] launching ingest_history.py for {p} collection={collection} repo={repo_name}"
            )
        except Exception:
            pass
        subprocess.Popen(cmd, env=env)
    except Exception:
        return


def _should_use_bulk_indexer(paths: List[Path], workspace_path: str) -> Dict[str, List[Path]]:
    """Detect if paths belong to new directories that should use bulk indexing.

    Returns a dict of {directory_path: [files]} for directories that exceed
    BULK_INDEX_THRESHOLD and should be indexed via subprocess.
    """
    if BULK_INDEX_THRESHOLD <= 0:
        return {}

    # Group paths by their immediate parent under workspace root
    dir_counts: Dict[str, List[Path]] = {}
    for p in paths:
        try:
            # Find the top-level directory under ROOT for this file
            rel = p.resolve().relative_to(ROOT.resolve())
            if rel.parts:
                top_dir = ROOT / rel.parts[0]
                dir_counts.setdefault(str(top_dir), []).append(p)
        except (ValueError, OSError):
            continue

    # Return directories that exceed the threshold
    bulk_dirs = {}
    for dir_path, files in dir_counts.items():
        if len(files) >= BULK_INDEX_THRESHOLD:
            bulk_dirs[dir_path] = files

    return bulk_dirs


def _run_bulk_indexer(directory: str, collection: str) -> bool:
    """Run ingest_code.py as subprocess for bulk indexing a directory.

    Returns True if subprocess was launched successfully.
    """
    try:
        import sys

        script = ROOT_DIR / "scripts" / "ingest_code.py"
        if not script.exists():
            print(f"[bulk_index] Script not found: {script}")
            return False

        cmd = [
            sys.executable or "python3",
            str(script),
            "--root", directory,
            "--skip-unchanged",  # Use hash-based skip for efficiency
        ]

        env = os.environ.copy()
        env["COLLECTION_NAME"] = collection
        env["QDRANT_URL"] = QDRANT_URL
        # Disable smart reindex in bulk mode for speed
        env["INDEX_SMART_REINDEX"] = "0"

        print(f"[bulk_index] Launching bulk indexer for {directory} -> {collection}")

        # Run synchronously so we know when it's done
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"[bulk_index] Completed {directory}")
            return True
        else:
            print(f"[bulk_index] Failed {directory}: {result.stderr[:500] if result.stderr else 'unknown error'}")
            return False

    except Exception as e:
        print(f"[bulk_index] Error launching indexer for {directory}: {e}")
        return False


def _process_paths(paths, client, model, vector_name: str, model_dim: int, workspace_path: str):
    unique_paths = sorted(set(Path(x) for x in paths))
    if not unique_paths:
        return

    started_at = datetime.now().isoformat()

    # Check if any directories should use bulk indexing
    bulk_dirs = _should_use_bulk_indexer(unique_paths, workspace_path)
    bulk_indexed_files: Set[Path] = set()

    if bulk_dirs:
        for dir_path, dir_files in bulk_dirs.items():
            try:
                collection = _get_collection_for_file(dir_files[0]) if dir_files else COLLECTION
                print(f"[bulk_index] Detected {len(dir_files)} files in {dir_path}, using bulk indexer")
                if _run_bulk_indexer(dir_path, collection):
                    bulk_indexed_files.update(dir_files)
            except Exception as e:
                print(f"[bulk_index] Failed for {dir_path}: {e}")

    # Filter out files that were bulk indexed
    remaining_paths = [p for p in unique_paths if p not in bulk_indexed_files]

    if not remaining_paths:
        print(f"[bulk_index] All {len(unique_paths)} files handled by bulk indexer")
        return

    repo_groups: dict[str, list[Path]] = {}
    for p in remaining_paths:
        repo_path = _detect_repo_for_file(p) or Path(workspace_path)
        repo_groups.setdefault(str(repo_path), []).append(p)

    for repo_path, repo_files in repo_groups.items():
        try:
            repo_name = _extract_repo_name_from_path(repo_path)
            update_indexing_status(
                repo_name=repo_name,
                status={
                    "state": "indexing",
                    "started_at": started_at,
                    "progress": {
                        "files_processed": 0,
                        "total_files": len(repo_files),
                    },
                },
            )
        except Exception:
            pass

    repo_progress: dict[str, int] = {key: 0 for key in repo_groups.keys()}

    for p in remaining_paths:
        repo_path = _detect_repo_for_file(p) or Path(workspace_path)
        repo_key = str(repo_path)
        repo_files = repo_groups.get(repo_key, [])
        repo_name = _extract_repo_name_from_path(repo_key)
        collection = _get_collection_for_file(p)

        if ".remote-git" in p.parts and p.suffix.lower() == ".json":
            try:
                _process_git_history_manifest(p, client, model, collection, vector_name, repo_name)
            except Exception as e:
                try:
                    print(f"[commit_ingest_error] {p}: {e}")
                except Exception:
                    pass
            repo_progress[repo_key] = repo_progress.get(repo_key, 0) + 1
            try:
                _update_progress(
                    repo_key,
                    started_at,
                    repo_progress[repo_key],
                    len(repo_files),
                    p,
                )
            except Exception:
                pass
            continue

        if not p.exists():
            if client is not None:
                try:
                    idx.delete_points_by_path(client, collection, str(p))
                    print(f"[deleted] {p} -> {collection}")
                except Exception:
                    pass
            try:
                remove_cached_file(str(p), repo_name)
            except Exception:
                pass
            _log_activity(repo_key, "deleted", p)
            repo_progress[repo_key] = repo_progress.get(repo_key, 0) + 1
            try:
                _update_progress(
                    repo_key,
                    started_at,
                    repo_progress[repo_key],
                    len(repo_files),
                    p,
                )
            except Exception:
                pass
            continue

        if client is not None and model is not None:
            try:
                idx.ensure_collection_and_indexes_once(client, collection, model_dim, vector_name)
            except Exception:
                pass

            ok = False
            try:
                # Prefer smart symbol-aware reindexing when enabled and cache is available
                try:
                    if getattr(idx, "_smart_symbol_reindexing_enabled", None) and idx._smart_symbol_reindexing_enabled():
                        text: str | None = None
                        try:
                            text = p.read_text(encoding="utf-8", errors="ignore")
                        except Exception:
                            text = None
                        if text is not None:
                            try:
                                language = idx.detect_language(p)
                            except Exception:
                                language = ""
                            try:
                                file_hash = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()
                            except Exception:
                                file_hash = ""
                            if file_hash:
                                try:
                                    use_smart, smart_reason = idx.should_use_smart_reindexing(str(p), file_hash)
                                except Exception:
                                    use_smart, smart_reason = False, "smart_check_failed"

                                # Bootstrap: if we have no symbol cache yet, still run smart path once
                                # to populate the cache for future smart reindexing
                                bootstrap = smart_reason == "no_cached_symbols"
                                if use_smart or bootstrap:
                                    msg_kind = "smart reindexing" if use_smart else "bootstrap (populating symbol cache)"
                                    try:
                                        print(f"[SMART_REINDEX][watcher] Using {msg_kind} for {p} ({smart_reason})")
                                    except Exception:
                                        pass
                                    try:
                                        status = idx.process_file_with_smart_reindexing(
                                            p,
                                            text,
                                            language,
                                            client,
                                            collection,
                                            repo_name,
                                            model,
                                            vector_name,
                                        )
                                        ok = status == "success"
                                    except Exception as se:
                                        try:
                                            print(f"[SMART_REINDEX][watcher] Smart reindexing failed for {p}: {se}")
                                        except Exception:
                                            pass
                                        ok = False
                                else:
                                    try:
                                        print(f"[SMART_REINDEX][watcher] Using full reindexing for {p} ({smart_reason})")
                                    except Exception:
                                        pass
                except Exception as e_smart:
                    try:
                        print(f"[SMART_REINDEX][watcher] Smart reindexing disabled or preview failed for {p}: {e_smart}")
                    except Exception:
                        pass

                # Fallback: full single-file reindex. Pseudo/tags are inlined by default;
                # when PSEUDO_BACKFILL_ENABLED=1 we run base-only and rely on backfill.
                if not ok:
                    flag = (os.environ.get("PSEUDO_BACKFILL_ENABLED") or "").strip().lower()
                    pseudo_mode = "off" if flag in {"1", "true", "yes", "on"} else "full"
                    ok = idx.index_single_file(
                        client,
                        model,
                        collection,
                        vector_name,
                        p,
                        dedupe=True,
                        skip_unchanged=False,
                        pseudo_mode=pseudo_mode,
                        repo_name_for_cache=repo_name,
                    )
            except Exception as e:
                try:
                    print(f"[index_error] {p}: {e}")
                except Exception:
                    pass
                ok = False
            status = "indexed" if ok else "skipped"
            print(f"[{status}] {p} -> {collection}")
            if ok:
                try:
                    size = int(p.stat().st_size)
                except Exception:
                    size = None
                _log_activity(repo_key, "indexed", p, {"file_size": size})
            else:
                _log_activity(
                    repo_key, "skipped", p, {"reason": "no-change-or-error"}
                )
        else:
            print(f"Not processing locally: {p}")
            _log_activity(repo_key, "skipped", p, {"reason": "remote-mode"})

        repo_progress[repo_key] = repo_progress.get(repo_key, 0) + 1
        try:
            _update_progress(
                repo_key,
                started_at,
                repo_progress[repo_key],
                len(repo_files),
                p,
            )
        except Exception:
            pass

    for repo_path in repo_groups.keys():
        try:
            repo_name = _extract_repo_name_from_path(repo_path)
            update_indexing_status(
                repo_name=repo_name,
                status={"state": "watching"},
            )
        except Exception:
            pass


if __name__ == "__main__":
    main()
