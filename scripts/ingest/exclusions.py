#!/usr/bin/env python3
"""
ingest/exclusions.py - File and directory exclusion logic.

This module provides the _Excluder class and related utilities for determining
which files and directories should be skipped during indexing.
"""
from __future__ import annotations

import os
import fnmatch
from pathlib import Path
from typing import List, Iterable

from scripts.ingest.config import (
    CODE_EXTS,
    EXTENSIONLESS_FILES,
    _DEFAULT_EXCLUDE_DIRS,
    _DEFAULT_EXCLUDE_DIR_GLOBS,
    _DEFAULT_EXCLUDE_FILES,
    _ANY_DEPTH_EXCLUDE_DIR_NAMES,
    _env_truthy,
)


class _Excluder:
    """Handles file and directory exclusion based on patterns."""
    
    def __init__(self, root: Path):
        self.root = root
        self.dir_prefixes: List[str] = []  # absolute like /path/sub
        self.dir_globs: List[str] = []  # fnmatch patterns for directory names
        self.file_globs: List[str] = []  # fnmatch patterns

        # Defaults
        use_defaults = _env_truthy(os.environ.get("QDRANT_DEFAULT_EXCLUDES"), True)
        if use_defaults:
            self.dir_prefixes.extend(_DEFAULT_EXCLUDE_DIRS)
            self.dir_globs.extend(_DEFAULT_EXCLUDE_DIR_GLOBS)
            self.file_globs.extend(_DEFAULT_EXCLUDE_FILES)

        # .qdrantignore
        ignore_file = os.environ.get("QDRANT_IGNORE_FILE", ".qdrantignore")
        ig_path = root / ignore_file
        if ig_path.exists():
            for raw in ig_path.read_text(
                encoding="utf-8", errors="ignore"
            ).splitlines():
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                self._add_pattern(line)

        # Extra excludes via env (comma separated)
        extra = os.environ.get("QDRANT_EXCLUDES", "").strip()
        if extra:
            for pat in [p.strip() for p in extra.split(",") if p.strip()]:
                self._add_pattern(pat)

    def _add_pattern(self, pat: str):
        """Add a pattern to the appropriate exclusion list."""
        # Normalize to leading-slash for prefixes
        has_wild = any(ch in pat for ch in "*?[")
        if pat.startswith("/") and not has_wild:
            # Treat as directory prefix if no wildcard
            self.dir_prefixes.append(pat.rstrip("/"))
        else:
            # Treat as file glob (match against relpath and basename)
            self.file_globs.append(pat.lstrip("/"))

    def exclude_dir(self, rel: str) -> bool:
        """Check if a directory should be excluded."""
        # rel like /a/b
        for pref in self.dir_prefixes:
            if rel == pref or rel.startswith(pref + "/"):
                return True

        base = rel.rsplit("/", 1)[-1]

        # Match directory name against dir_globs (e.g., .venv*)
        for g in self.dir_globs:
            if fnmatch.fnmatch(base, g):
                return True

        # Treat single-segment dir prefixes (e.g. "/.git", "/node_modules") as
        # "exclude this directory name anywhere". This matters when indexing a
        # workspace root that contains multiple repos, e.g. /work/<repo>/.git.
        try:
            if base in _ANY_DEPTH_EXCLUDE_DIR_NAMES and ("/" + base) in self.dir_prefixes:
                return True
        except Exception:
            pass

        # Also allow dir name-only patterns in file_globs (e.g., node_modules)
        for g in self.file_globs:
            # Match bare dir names without wildcards
            if g and all(ch not in g for ch in "*?[") and base == g:
                return True
        return False

    def exclude_file(self, rel: str) -> bool:
        """Check if a file should be excluded."""
        # Try matching whole rel path and basename
        base = rel.rsplit("/", 1)[-1]
        for g in self.file_globs:
            if fnmatch.fnmatch(rel.lstrip("/"), g) or fnmatch.fnmatch(base, g):
                return True
        return False


def is_indexable_file(p: Path) -> bool:
    """Check if a file should be indexed (by extension or name pattern).

    Public API for use by watch_index and other modules.
    """
    # Check by extension first
    if p.suffix.lower() in CODE_EXTS:
        return True
    # Check by filename (for Dockerfile, Makefile, etc.)
    fname_lower = p.name.lower()
    if fname_lower in EXTENSIONLESS_FILES:
        return True
    # Check for Dockerfile.* pattern (e.g., Dockerfile.dev, Dockerfile.prod)
    if fname_lower.startswith("dockerfile"):
        return True
    return False


# Backward-compatible alias
_is_indexable_file = is_indexable_file


def _should_skip_explicit_file_by_excluder(file_path: Path) -> bool:
    """Check if a file should be skipped based on exclusion rules."""
    try:
        p = file_path if isinstance(file_path, Path) else Path(str(file_path))
    except Exception:
        return False

    root = None
    try:
        parts = list(p.parts)
        if ".remote-git" in parts:
            i = parts.index(".remote-git")
            root = Path(*parts[:i]) if i > 0 else Path("/")
    except Exception:
        root = None

    if root is None:
        try:
            s = str(p)
            if s.startswith("/work/"):
                slug = s[len("/work/"):].split("/", 1)[0]
                root = (Path("/work") / slug) if slug else None
        except Exception:
            root = None

    if root is None:
        try:
            ws = (os.environ.get("WATCH_ROOT") or os.environ.get("WORKSPACE_PATH") or "").strip()
            if ws:
                ws_path = Path(ws).resolve()
                pr = p.resolve()
                if pr == ws_path or ws_path in pr.parents:
                    root = ws_path
        except Exception:
            root = None

    if root is None:
        try:
            pr = p.resolve()
            for anc in [pr.parent] + list(pr.parents):
                if (anc / ".codebase").exists():
                    root = anc
                    break
        except Exception:
            root = None

    if not root or str(root) == "/":
        return False

    try:
        rel = p.resolve().relative_to(root.resolve()).as_posix().lstrip("/")
    except Exception:
        return False
    if not rel:
        return False

    try:
        excl = _Excluder(root)
        cur = ""
        for seg in [x for x in rel.split("/") if x][:-1]:
            cur = cur + "/" + seg
            if excl.exclude_dir(cur):
                return True
        return excl.exclude_file(rel)
    except Exception:
        return False


def iter_files(root: Path) -> Iterable[Path]:
    """Iterate over indexable files in a directory tree."""
    # Allow passing a single file
    if root.is_file():
        if is_indexable_file(root) and not _should_skip_explicit_file_by_excluder(root):
            yield root
        return

    excl = _Excluder(root)
    # Use os.walk to prune directories for performance
    # NOTE: avoid Path.resolve()/realpath here; on network filesystems (e.g. CephFS)
    # it can trigger expensive metadata calls during large unchanged indexing runs.
    try:
        root_abs = os.path.abspath(str(root))
    except Exception:
        root_abs = str(root)

    for dirpath, dirnames, filenames in os.walk(root_abs):
        # Compute rel path like /a/b from root without resolving symlinks
        try:
            rel = os.path.relpath(dirpath, root_abs)
        except Exception:
            rel = "."
        if rel in (".", ""):
            rel_dir = "/"
        else:
            rel_dir = "/" + rel.replace(os.sep, "/")
        # Prune excluded directories in-place
        keep = []
        for d in dirnames:
            rel = (rel_dir.rstrip("/") + "/" + d).replace("//", "/")
            if excl.exclude_dir(rel):
                continue
            keep.append(d)
        dirnames[:] = keep

        for f in filenames:
            p = Path(dirpath) / f
            if not is_indexable_file(p):
                continue
            relf = (rel_dir.rstrip("/") + "/" + f).replace("//", "/")
            if excl.exclude_file(relf):
                continue
            yield p
