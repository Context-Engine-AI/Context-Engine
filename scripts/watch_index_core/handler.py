"""Watchdog event handler responsible for enqueueing file changes."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from qdrant_client import QdrantClient
from watchdog.events import FileSystemEventHandler

import scripts.ingest_code as idx
from scripts.workspace_state import (
    _extract_repo_name_from_path,
    _get_global_state_dir,
    get_cached_file_hash,
    log_watcher_activity as _log_activity,
    remove_cached_file,
    remove_cached_symbols,
    set_cached_file_hash,
)
from .config import LOGGER
from .utils import (
    _detect_repo_for_file, 
    _get_collection_for_file,
    _repo_name_or_none,
    safe_print,
)
from .rename import _rename_in_store


class IndexHandler(FileSystemEventHandler):
    def __init__(
        self,
        root: Path,
        queue,
        client: Optional[QdrantClient],
        default_collection: Optional[str] = None,
        *,
        collection: Optional[str] = None,
    ):
        super().__init__()
        self.root = root
        self.queue = queue
        self.client = client
        # Only pin to a fixed collection when explicitly provided.
        # Otherwise, allow multi-repo mode to resolve per-file collections.
        # In multi-repo mode, per-file collections are resolved via _get_collection_for_file
        # and workspace_state; avoid deriving a root-level collection like "/work-<hash>".
        self.default_collection = default_collection
        self.collection = collection
        self.excl = idx._Excluder(root)
        # Track ignore file for live reloads
        try:
            ig_name = os.environ.get("QDRANT_IGNORE_FILE", ".qdrantignore")
            self._ignore_path = (self.root / ig_name).resolve()
        except (OSError, ValueError) as exc:
            safe_print(f"[ignore_file] Could not resolve ignore file path: {exc}")
            self._ignore_path = None
        try:
            self._ignore_mtime = (
                self._ignore_path.stat().st_mtime
                if self._ignore_path and self._ignore_path.exists()
                else 0.0
            )
        except Exception:
            self._ignore_mtime = 0.0

    def _maybe_reload_excluder(self) -> None:
        try:
            if not self._ignore_path:
                return
            cur = (
                self._ignore_path.stat().st_mtime if self._ignore_path.exists() else 0.0
            )
            # Refresh ignore patterns if the file changed
            if cur != self._ignore_mtime:
                self.excl = idx._Excluder(self.root)
                self._ignore_mtime = cur
                safe_print(f"[ignore_reload] reloaded patterns from {self._ignore_path}")
        except Exception:
            pass

    def _maybe_enqueue(self, src_path: str) -> None:
        self._maybe_reload_excluder()
        p = Path(src_path)
        try:
            p = p.resolve()
        except Exception:
            return
        if p.is_dir():
            return
        try:
            rel = p.resolve().relative_to(self.root.resolve())
        except ValueError:
            return

        try:
            if callable(_get_global_state_dir):
                global_state_dir = _get_global_state_dir()
                if global_state_dir is not None and p.is_relative_to(global_state_dir):
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

        rel_dir = "/" + str(rel.parent).replace(os.sep, "/")
        if rel_dir == "/.":
            rel_dir = "/"
        if self.excl.exclude_dir(rel_dir):
            return
        # only code files (check extension AND extensionless files like Dockerfile)
        if not idx.is_indexable_file(p):
            return
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
        if not idx.is_indexable_file(p):
            return
        collection = self._resolve_collection(p)
        self._delete_points(p, collection)

        try:
            repo_path = _detect_repo_for_file(p) or self.root
            _log_activity(str(repo_path), "deleted", p)
        except Exception as exc:
            safe_print(f"[delete_error] {p}: {exc}")
        self._invalidate_cache(p)

    def on_moved(self, event):
        if event.is_directory:
            return
        try:
            src = Path(event.src_path).resolve()
            dest = Path(event.dest_path).resolve()
        except Exception:
            return
        if not idx.is_indexable_file(dest) and not idx.is_indexable_file(src):
            return
        try:
            rel_dir = "/" + str(
                dest.parent.resolve().relative_to(self.root.resolve())
            ).replace(os.sep, "/")
            if rel_dir == "/.":
                rel_dir = "/"
            if self.excl.exclude_dir(rel_dir):
                if idx.is_indexable_file(src):
                    try:
                        coll = self._resolve_collection(src)
                        if self.client is not None and coll is not None:
                            idx.delete_points_by_path(self.client, coll, str(src))
                        safe_print(f"[moved:ignored_dest_deleted_src] {src} -> {dest}")
                        src_repo_path = _detect_repo_for_file(src)
                        src_repo_name = _repo_name_or_none(src_repo_path)
                        try:
                            if src_repo_name:
                                remove_cached_file(str(src), src_repo_name)
                        except Exception:
                            pass
                    except Exception:
                        pass
                return
        except Exception:
            pass

        src_collection = self._resolve_collection(src)
        dest_collection = self._resolve_collection(dest)
        is_cross_collection = src_collection != dest_collection
        if is_cross_collection:
            safe_print(f"[cross_collection_move] {src} -> {dest}")

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
                safe_print(f"[moved] {src} -> {dest} ({moved_count} chunk(s) relinked)")
                src_repo_path = _detect_repo_for_file(src)
                dest_repo_path = _detect_repo_for_file(dest)
                src_repo_name = _repo_name_or_none(src_repo_path)
                dest_repo_name = _repo_name_or_none(dest_repo_path)
                try:
                    src_hash = get_cached_file_hash(str(src), src_repo_name) if src_repo_name else None
                except Exception:
                    src_hash = None
                self._invalidate_cache(src)
                if not src_hash and renamed_hash:
                    src_hash = renamed_hash
                if dest_repo_name and src_hash:
                    try:
                        set_cached_file_hash(str(dest), src_hash, dest_repo_name)
                    except Exception:
                        pass
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
                if idx.is_indexable_file(src):
                    try:
                        idx.delete_points_by_path(self.client, src_collection, str(src))
                        safe_print(f"[moved:reindex_src] {src} -> {dest} (dest skipped)")
                    except Exception:
                        pass
                else:
                    # Non-indexable source file: use _delete_points for cleanup
                    self._delete_points(src, src_collection)
            except Exception:
                pass
        else:
            safe_print(f"[remote_mode] Move detected: {src} -> {dest}")
        try:
            self._maybe_enqueue(str(dest))
        except Exception:
            pass

    def _resolve_collection(self, path: Path) -> str | None:
        if self.collection is not None:
            return self.collection
        try:
            return _get_collection_for_file(path)
        except Exception:
            return self.default_collection

    def _delete_points(self, path: Path, collection: str | None) -> None:
        if self.client is None or collection is None:
            safe_print(f"[remote_mode] File change detected without client: {path}")
            return
        try:
            idx.delete_points_by_path(self.client, collection, str(path))
            safe_print(f"[deleted] {path} -> {collection}")
        except Exception:
            pass

    def _invalidate_cache(self, path: Path) -> Optional[str]:
        detected_repo_path = _detect_repo_for_file(path)
        repo_path = detected_repo_path or self.root
        repo_name = _extract_repo_name_from_path(str(repo_path)) if repo_path else None
        try:
            if repo_name:
                remove_cached_file(str(path), repo_name)
        except Exception:
            pass
        try:
            remove_cached_symbols(str(path))
        except Exception:
            pass
        return _repo_name_or_none(detected_repo_path)


__all__ = ["IndexHandler"]
