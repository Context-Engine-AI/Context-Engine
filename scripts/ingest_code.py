#!/usr/bin/env python3
"""
ingest_code.py - Façade module for code indexing.

This is the stable public entrypoint for the code indexing subsystem.
All internal logic has been refactored into smaller, focused modules in scripts/ingest/:
- config.py: Environment-based configuration and constants
- tree_sitter.py: Tree-sitter setup and language loading
- vectors.py: Vector generation utilities (lex hash, mini projection)
- exclusions.py: File and directory exclusion logic
- chunking.py: Code chunking utilities (line, semantic, token-based)
- symbols.py: Symbol extraction for code analysis
- pseudo.py: ReFRAG pseudo-description and tag generation
- metadata.py: Metadata extraction (git, imports, calls)
- qdrant.py: Qdrant schema and I/O operations
- pipeline.py: Helper functions for indexing
- cli.py: Command-line interface

This façade:
1. Re-exports all public APIs for backwards compatibility
2. Implements main orchestration functions (index_single_file, index_repo, process_file_with_smart_reindexing)
3. Provides the CLI entrypoint (main)
"""
from __future__ import annotations

import os
import sys
import hashlib
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, TYPE_CHECKING

# Ensure project root is on sys.path when run as a script
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from qdrant_client import QdrantClient, models

# ---------------------------------------------------------------------------
# Re-exports from ingest/config.py
# ---------------------------------------------------------------------------
from scripts.ingest.config import (
    ROOT_DIR,
    _safe_int_env,
    _env_truthy,
    LEX_VECTOR_NAME,
    LEX_VECTOR_DIM,
    MINI_VECTOR_NAME,
    MINI_VEC_DIM,
    LEX_SPARSE_NAME,
    LEX_SPARSE_MODE,
    _STOP,
    CODE_EXTS,
    EXTENSIONLESS_FILES,
    _DEFAULT_EXCLUDE_DIRS,
    _DEFAULT_EXCLUDE_DIR_GLOBS,
    _DEFAULT_EXCLUDE_FILES,
    _ANY_DEPTH_EXCLUDE_DIR_NAMES,
    is_multi_repo_mode,
    get_collection_name,
    logical_repo_reuse_enabled,
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
    _detect_repo_for_file,
    _get_collection_for_file,
)

# ---------------------------------------------------------------------------
# Re-exports from ingest/tree_sitter.py
# ---------------------------------------------------------------------------
from scripts.ingest.tree_sitter import (
    _TS_LANGUAGES,
    _TS_AVAILABLE,
    _use_tree_sitter,
    _ts_parser,
    _load_ts_language,
)

try:
    from tree_sitter import Parser, Language
except ImportError:
    Parser = None  # type: ignore
    Language = None  # type: ignore

# ---------------------------------------------------------------------------
# Re-exports from ingest/vectors.py
# ---------------------------------------------------------------------------
from scripts.ingest.vectors import (
    _MINI_PROJ_CACHE,
    _get_mini_proj,
    project_mini,
    _split_ident_lex,
    _lex_hash_vector,
)

# ---------------------------------------------------------------------------
# Re-exports from ingest/exclusions.py
# ---------------------------------------------------------------------------
from scripts.ingest.exclusions import (
    _Excluder,
    is_indexable_file,
    _is_indexable_file,
    _should_skip_explicit_file_by_excluder,
    iter_files,
)

# ---------------------------------------------------------------------------
# Re-exports from ingest/chunking.py
# ---------------------------------------------------------------------------
from scripts.ingest.chunking import (
    chunk_lines,
    chunk_semantic,
    chunk_by_tokens,
)

# ---------------------------------------------------------------------------
# Re-exports from ingest/symbols.py
# ---------------------------------------------------------------------------
from scripts.ingest.symbols import (
    _Sym,
    _extract_symbols_python,
    _extract_symbols_js_like,
    _extract_symbols_go,
    _extract_symbols_java,
    _extract_symbols_csharp,
    _extract_symbols_php,
    _extract_symbols_shell,
    _extract_symbols_yaml,
    _extract_symbols_powershell,
    _extract_symbols_rust,
    _extract_symbols_terraform,
    _ts_extract_symbols_python,
    _ts_extract_symbols_js,
    _ts_extract_symbols_yaml,
    _ts_extract_symbols,
    _extract_symbols,
    _choose_symbol_for_chunk,
    extract_symbols_with_tree_sitter,
)

# ---------------------------------------------------------------------------
# Re-exports from ingest/pseudo.py
# ---------------------------------------------------------------------------
from scripts.ingest.pseudo import (
    _pseudo_describe_enabled,
    _smart_symbol_reindexing_enabled,
    generate_pseudo_tags,
    should_process_pseudo_for_chunk,
    should_use_smart_reindexing,
)

# ---------------------------------------------------------------------------
# Re-exports from ingest/metadata.py
# ---------------------------------------------------------------------------
from scripts.ingest.metadata import (
    _git_metadata,
    _extract_imports,
    _extract_calls,
    _get_imports_calls,
    _get_host_path_from_origin,
    _compute_host_and_container_paths,
)

# ---------------------------------------------------------------------------
# Re-exports from ingest/qdrant.py
# ---------------------------------------------------------------------------
from scripts.ingest.qdrant import (
    ENSURED_COLLECTIONS,
    ENSURED_COLLECTIONS_LAST_CHECK,
    CollectionNeedsRecreateError,
    ensure_collection,
    recreate_collection,
    ensure_payload_indexes,
    ensure_collection_and_indexes_once,
    get_indexed_file_hash,
    delete_points_by_path,
    upsert_points,
    hash_id,
    embed_batch,
)

# ---------------------------------------------------------------------------
# Re-exports from ingest/pipeline.py (helpers only)
# ---------------------------------------------------------------------------

from scripts.ingest.pipeline import (
    _detect_repo_name_from_path,
    detect_language,
    build_information,
    pseudo_backfill_tick,
    # Main orchestration functions - pipeline.py is the single source of truth
    index_single_file,
    _index_single_file_inner,
    index_repo,
    process_file_with_smart_reindexing,
)
# ---------------------------------------------------------------------------
# Re-exports from ingest/cli.py
# ---------------------------------------------------------------------------
from scripts.ingest.cli import (
    parse_args,
)

# ---------------------------------------------------------------------------
# Additional imports for backward compatibility
# ---------------------------------------------------------------------------
try:
    from scripts.embedder import get_embedding_model as _get_embedding_model
    _EMBEDDER_FACTORY = True
except ImportError:
    _EMBEDDER_FACTORY = False

if TYPE_CHECKING:
    from fastembed import TextEmbedding

try:
    from fastembed import TextEmbedding
except ImportError:
    TextEmbedding = None  # type: ignore

from scripts.utils import sanitize_vector_name as _sanitize_vector_name
from scripts.utils import lex_hash_vector_text as _lex_hash_vector_text
from scripts.utils import lex_sparse_vector_text as _lex_sparse_vector_text

try:
    from scripts.ast_analyzer import get_ast_analyzer, chunk_code_semantically
    _AST_ANALYZER_AVAILABLE = True
except ImportError:
    _AST_ANALYZER_AVAILABLE = False

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore

_TS_WARNED = False


# ---------------------------------------------------------------------------
# Main orchestration functions (kept in façade for test monkeypatching)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------
def main():
    """Main entry point for the CLI."""
    from scripts.ingest.cli import main as _cli_main
    _cli_main()


# ---------------------------------------------------------------------------
# Public API (__all__)
# ---------------------------------------------------------------------------
__all__ = [
    # Config
    "ROOT_DIR",
    "LEX_VECTOR_NAME",
    "LEX_VECTOR_DIM",
    "MINI_VECTOR_NAME",
    "MINI_VEC_DIM",
    "LEX_SPARSE_NAME",
    "LEX_SPARSE_MODE",
    "CODE_EXTS",
    "EXTENSIONLESS_FILES",
    # Workspace state
    "is_multi_repo_mode",
    "get_collection_name",
    "logical_repo_reuse_enabled",
    "get_cached_symbols",
    "set_cached_symbols",
    "compare_symbol_changes",
    "get_cached_pseudo",
    "set_cached_pseudo",
    "get_cached_file_hash",
    "set_cached_file_hash",
    # Tree-sitter
    "_TS_AVAILABLE",
    "_TS_LANGUAGES",
    "_use_tree_sitter",
    # Vectors
    "project_mini",
    "_lex_hash_vector",
    # Exclusions
    "iter_files",
    "is_indexable_file",
    "_Excluder",
    # Chunking
    "chunk_lines",
    "chunk_semantic",
    "chunk_by_tokens",
    # Symbols
    "_extract_symbols",
    "extract_symbols_with_tree_sitter",
    "_choose_symbol_for_chunk",
    # Pseudo
    "_pseudo_describe_enabled",
    "_smart_symbol_reindexing_enabled",
    "generate_pseudo_tags",
    "should_process_pseudo_for_chunk",
    "should_use_smart_reindexing",
    # Metadata
    "_git_metadata",
    "_get_imports_calls",
    # Qdrant
    "ensure_collection",
    "recreate_collection",
    "ensure_payload_indexes",
    "ensure_collection_and_indexes_once",
    "get_indexed_file_hash",
    "delete_points_by_path",
    "upsert_points",
    "hash_id",
    "embed_batch",
    # Pipeline
    "_detect_repo_name_from_path",
    "detect_language",
    "build_information",
    "index_single_file",
    "index_repo",
    "process_file_with_smart_reindexing",
    "pseudo_backfill_tick",
    # CLI
    "main",
    # Backward compat
    "TextEmbedding",
    "_EMBEDDER_FACTORY",
]


if __name__ == "__main__":
    main()
