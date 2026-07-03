"""
Ingest package - Code indexing subsystem.

This package contains extracted modules from ingest_code.py:
- config: Environment-based configuration and constants
- tree_sitter: Tree-sitter setup and language loading
- vectors: Vector generation utilities (lex hash, mini projection)
- exclusions: File and directory exclusion logic
- chunking: Code chunking utilities (line, semantic, token-based)
- symbols: Symbol extraction for code analysis
- pseudo: ReFRAG pseudo-description and tag generation
- metadata: Metadata extraction (git, imports, calls)
- qdrant: Qdrant schema and I/O operations
- pipeline: Core indexing pipeline
- cli: Command-line interface

Usage:
    from scripts.ingest import config, pipeline, qdrant
    from scripts.ingest.config import LEX_VECTOR_NAME, LEX_VECTOR_DIM
    from scripts.ingest.pipeline import index_repo, index_single_file
    from scripts.ingest.qdrant import ensure_collection, upsert_points
"""
from scripts.ingest import config
from scripts.ingest import tree_sitter
from scripts.ingest import vectors
from scripts.ingest import exclusions
from scripts.ingest import chunking
from scripts.ingest import symbols
from scripts.ingest import pseudo
from scripts.ingest import metadata
from scripts.ingest import qdrant
from scripts.ingest import pipeline
from scripts.ingest import cli

__all__ = [
    "config",
    "tree_sitter",
    "vectors",
    "exclusions",
    "chunking",
    "symbols",
    "pseudo",
    "metadata",
    "qdrant",
    "pipeline",
    "cli",
]
