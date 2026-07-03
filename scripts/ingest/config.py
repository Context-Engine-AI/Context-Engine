#!/usr/bin/env python3
"""
ingest/config.py - Environment-based configuration and constants for code ingestion.

This module centralizes all configuration values, environment variable parsing,
file extension mappings, and exclusion patterns for the indexing subsystem.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Any, Optional

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def _safe_int_env(key: str, default: int) -> int:
    """Safely parse an integer from environment variable."""
    try:
        val = os.environ.get(key)
        return int(val) if val else default
    except (ValueError, TypeError):
        return default


def _env_truthy(val: str | None, default: bool) -> bool:
    """Check if environment value is truthy."""
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


# ---------------------------------------------------------------------------
# Named vector configuration
# ---------------------------------------------------------------------------
LEX_VECTOR_NAME = os.environ.get("LEX_VECTOR_NAME", "lex")
# Legacy default 4096 for existing collections; new users can set LEX_VECTOR_DIM=2048 via .env
LEX_VECTOR_DIM = _safe_int_env("LEX_VECTOR_DIM", 4096)

# Optional mini vector (ReFRAG-style gating); conditionally created by REFRAG_MODE
MINI_VECTOR_NAME = os.environ.get("MINI_VECTOR_NAME", "mini")
MINI_VEC_DIM = int(os.environ.get("MINI_VEC_DIM", "64") or 64)

# Lossless sparse lexical vector (no hash collisions)
LEX_SPARSE_NAME = os.environ.get("LEX_SPARSE_NAME", "lex_sparse")
LEX_SPARSE_MODE = os.environ.get("LEX_SPARSE_MODE", "0").strip().lower() in ("1", "true", "yes", "on")


# ---------------------------------------------------------------------------
# Stop words for lexical tokenization
# ---------------------------------------------------------------------------
_STOP = {
    "the",
    "a",
    "an",
    "of",
    "in",
    "on",
    "for",
    "and",
    "or",
    "to",
    "with",
    "by",
    "is",
    "are",
    "be",
    "this",
    "that",
}


# ---------------------------------------------------------------------------
# File extension to language mapping
# ---------------------------------------------------------------------------
CODE_EXTS: Dict[str, str] = {
    # Core languages
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".jsx": "javascript",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".php": "php",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".csx": "csharp",
    ".kt": "kotlin",
    ".swift": "swift",
    ".scala": "scala",
    # Shell/scripting
    ".sh": "shell",
    ".ps1": "powershell",
    ".psm1": "powershell",
    ".psd1": "powershell",
    ".pl": "perl",
    ".lua": "lua",
    # Data/config
    ".sql": "sql",
    ".md": "markdown",
    ".yml": "yaml",
    ".yaml": "yaml",
    ".toml": "toml",
    ".ini": "ini",
    ".cfg": "ini",
    ".conf": "ini",
    ".json": "json",
    ".xml": "xml",
    ".csproj": "xml",
    ".config": "xml",
    ".resx": "xml",
    # Web
    ".html": "html",
    ".htm": "html",
    ".css": "css",
    ".scss": "scss",
    ".sass": "sass",
    ".less": "less",
    ".vue": "vue",
    ".svelte": "svelte",
    ".cshtml": "razor",
    ".razor": "razor",
    # Infrastructure
    ".tf": "terraform",
    ".tfvars": "terraform",
    ".hcl": "hcl",
    ".dockerfile": "dockerfile",
    # Additional languages
    ".elm": "elm",
    ".dart": "dart",
    ".r": "r",
    ".R": "r",
    ".m": "matlab",
    ".cljs": "clojure",
    ".clj": "clojure",
    ".hs": "haskell",
    ".ml": "ocaml",
    ".zig": "zig",
    ".nim": "nim",
    ".v": "verilog",
    ".sv": "verilog",
    ".vhdl": "vhdl",
    ".asm": "assembly",
    ".s": "assembly",
}

# Files matched by name (no extension or special names)
# Keys are lowercase filename patterns, values are language
# NOTE: .env files are excluded to prevent leaking secrets to LLMs
EXTENSIONLESS_FILES: Dict[str, str] = {
    "dockerfile": "dockerfile",
    "makefile": "makefile",
    "gemfile": "ruby",
    "rakefile": "ruby",
    "procfile": "yaml",
    "vagrantfile": "ruby",
    "jenkinsfile": "groovy",
    ".gitignore": "gitignore",
    ".dockerignore": "dockerignore",
    ".editorconfig": "ini",
}


# ---------------------------------------------------------------------------
# Exclusion patterns (configurable via env / .qdrantignore)
# ---------------------------------------------------------------------------
_DEFAULT_EXCLUDE_DIRS = [
    "/models",
    "/.vs",
    "/.vscode",
    "/.cache",
    "/.codebase",
    "/.remote-git",
    "/node_modules",
    "/dist",
    "/build",
    "/.venv",
    "/venv",
    "/py-venv",
    "/site-packages",
    "/__pycache__",
    "bin",
    "obj",
    "TestResults",
    "/.git",
]

# Glob patterns for directories (matched against basename)
_DEFAULT_EXCLUDE_DIR_GLOBS = [
    ".venv*",  # .venv, .venv311, .venv39, etc.
]

_DEFAULT_EXCLUDE_FILES = [
    "*.onnx",
    "*.bin",
    "*.safetensors",
    "tokenizer.json",
    "*.whl",
    "*.tar.gz",
]

_ANY_DEPTH_EXCLUDE_DIR_NAMES = {
    ".git",
    ".remote-git",
    ".codebase",
    "node_modules",
}


# ---------------------------------------------------------------------------
# Workspace state function imports (optional)
# ---------------------------------------------------------------------------
# These are imported at module load time for convenience, with fallbacks
try:
    from scripts.workspace_state import (
        is_multi_repo_mode,
        get_collection_name,
        logical_repo_reuse_enabled,
    )
except ImportError:
    is_multi_repo_mode = None  # type: ignore
    get_collection_name = None  # type: ignore

    def logical_repo_reuse_enabled() -> bool:  # type: ignore[no-redef]
        return False

# Import watcher's repo detection for surgical fix
try:
    from scripts.watch_index_core.routing import _detect_repo_for_file, _get_collection_for_file
except ImportError:
    _detect_repo_for_file = None  # type: ignore
    _get_collection_for_file = None  # type: ignore

# Import other workspace state functions (optional)
try:
    from scripts.workspace_state import (
        log_activity,
        get_cached_file_hash,
        set_cached_file_hash,
        remove_cached_file,
        update_indexing_status,
        update_workspace_state,
        get_cached_symbols,
        set_cached_symbols,
        remove_cached_symbols,
        compare_symbol_changes,
        get_cached_pseudo,
        set_cached_pseudo,
        update_symbols_with_pseudo,
        get_workspace_state,
        get_cached_file_meta,
        indexing_lock,
        file_indexing_lock,
        is_file_locked,
    )
except ImportError:
    # State integration is optional; continue if not available
    log_activity = None  # type: ignore
    get_cached_file_hash = None  # type: ignore
    set_cached_file_hash = None  # type: ignore
    remove_cached_file = None  # type: ignore
    update_indexing_status = None  # type: ignore
    update_workspace_state = None  # type: ignore
    get_cached_symbols = None  # type: ignore
    set_cached_symbols = None  # type: ignore
    remove_cached_symbols = None  # type: ignore
    get_cached_pseudo = None  # type: ignore
    set_cached_pseudo = None  # type: ignore
    update_symbols_with_pseudo = None  # type: ignore
    compare_symbol_changes = None  # type: ignore
    get_workspace_state = None  # type: ignore
    get_cached_file_meta = None  # type: ignore
    indexing_lock = None  # type: ignore
    file_indexing_lock = None  # type: ignore
    is_file_locked = None  # type: ignore
