#!/usr/bin/env python3
"""
CoSQA Corpus Indexer - Thin adapter over shared benchmark indexing core.

Transforms CoSQA dataset entries into generic BenchmarkDoc format and delegates
to scripts/benchmarks/core_indexer.py for actual indexing.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from scripts.benchmarks.core_indexer import (
    BenchmarkDoc,
    index_benchmark_corpus,
    compute_corpus_fingerprint as _compute_corpus_fingerprint,
    get_collection_fingerprint as _get_collection_fingerprint,
    collection_matches_corpus as _collection_matches_corpus,
)
from scripts.benchmarks.qdrant_utils import (
    get_qdrant_client,
    get_embedding_model,
)

# Default configuration
DEFAULT_COLLECTION = "cosqa-corpus"
DEFAULT_BATCH_SIZE = 100


def _extract_docstring_treesitter(code: str) -> str:
    """Extract docstring from Python code using tree-sitter (preferred)."""
    try:
        from scripts.ingest.tree_sitter import _ts_parser
        
        parser = _ts_parser("python")
        if parser is None:
            return ""
        
        tree = parser.parse(code.encode("utf-8"))
        root = tree.root_node
        
        # Find first function_definition or class_definition
        def find_first_def(node):
            if node.type in ("function_definition", "class_definition"):
                return node
            for child in node.children:
                result = find_first_def(child)
                if result:
                    return result
            return None
        
        def_node = find_first_def(root)
        if def_node is None:
            return ""
        
        # Look for expression_statement with string as first child of block/body
        for child in def_node.children:
            if child.type == "block":
                for stmt in child.children:
                    if stmt.type == "expression_statement":
                        for expr in stmt.children:
                            if expr.type == "string":
                                # Extract string content, strip quotes
                                text = code[expr.start_byte:expr.end_byte]
                                # Remove triple quotes
                                if text.startswith('"""') and text.endswith('"""'):
                                    return text[3:-3].strip()
                                if text.startswith("'''") and text.endswith("'''"):
                                    return text[3:-3].strip()
                                return text.strip('"\'').strip()
                        break  # Only check first statement
                break
        return ""
    except Exception:
        return ""


def _extract_docstring_ast(code: str) -> str:
    """Extract docstring from Python code using ast (fallback, may emit SyntaxWarnings)."""
    import ast
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                docstring = ast.get_docstring(node)
                if docstring:
                    return docstring
        return ast.get_docstring(tree) or ""
    except Exception:
        return ""


def _extract_docstring(code: str) -> str:
    """Extract docstring from Python code. Uses tree-sitter (no warnings), falls back to ast."""
    # Try tree-sitter first (handles invalid escape sequences gracefully)
    result = _extract_docstring_treesitter(code)
    if result:
        return result
    # Fallback to ast with warning suppression
    return _extract_docstring_ast(code)


def cosqa_entry_to_doc(entry: Dict[str, Any]) -> BenchmarkDoc:
    """Convert CoSQA entry (from to_index_payload) to BenchmarkDoc.

    Expects entry from CoSQACorpusEntry.to_index_payload() which includes
    realistic paths, symbol_path, and rerank_text for proper Context-Engine
    pipeline evaluation.
    """
    code_id = entry.get("code_id", "")
    text = entry.get("text", "")
    func_name = entry.get("func_name", "")
    docstring = entry.get("docstring", "")

    # Extract docstring from code if not provided
    if not docstring and text:
        docstring = _extract_docstring(text)

    # Pass through the rich metadata from to_index_payload() - includes path,
    # symbol_path, rerank_text for FNAME_BOOST and symbol boost testing
    inner_metadata = entry.get("metadata", {})

    return BenchmarkDoc(
        doc_id=code_id,
        text=text,
        language=entry.get("language", "python"),
        metadata={
            "code_id": code_id,
            "func_name": func_name,
            "docstring": docstring,
            "source": entry.get("source", "cosqa"),
            "path": inner_metadata.get("path", f"cosqa/{code_id}.py"),
            "symbol": inner_metadata.get("symbol", ""),
            "symbol_path": inner_metadata.get("symbol_path", ""),
        },
    )


def compute_corpus_fingerprint(corpus_entries: List[Dict[str, Any]]) -> str:
    """Compute fingerprint for CoSQA corpus."""
    docs = [cosqa_entry_to_doc(e) for e in corpus_entries]
    return _compute_corpus_fingerprint(docs)


def get_collection_fingerprint(collection: str) -> str | None:
    """Get stored fingerprint from collection."""
    client = get_qdrant_client()
    return _get_collection_fingerprint(client, collection)


def collection_matches_corpus(collection: str, corpus_entries: List[Dict[str, Any]]) -> bool:
    """Check if collection matches corpus fingerprint."""
    client = get_qdrant_client()
    docs = [cosqa_entry_to_doc(e) for e in corpus_entries]
    return _collection_matches_corpus(client, collection, docs)


def index_corpus(
    corpus_entries: List[Dict[str, Any]],
    collection: str = DEFAULT_COLLECTION,
    batch_size: int = DEFAULT_BATCH_SIZE,
    recreate: bool = False,
    resume: bool = True,
    progress_callback: callable | None = None,
    force: bool = False,
) -> Dict[str, Any]:
    """Index CoSQA corpus into Qdrant using shared benchmark indexer.

    Args:
        corpus_entries: List of corpus entries from dataset.get_corpus_for_indexing()
        collection: Qdrant collection name
        batch_size: Batch size for upsert operations
        recreate: Whether to recreate the collection
        resume: Ignored (kept for API compatibility)
        progress_callback: Ignored (kept for API compatibility)
        force: Force reindexing even if collection matches corpus

    Returns:
        Stats dict with indexed count, time, errors, reused
    """
    # CoSQA snippets are atomic units - disable chunking so 1 point = 1 snippet.
    # This ensures points_count ≈ corpus_size for proper rerank/candidate scaling.
    os.environ.setdefault("INDEX_SEMANTIC_CHUNKS", "0")
    os.environ.setdefault("INDEX_CHUNK_LINES", "10000")
    os.environ.setdefault("INDEX_CHUNK_OVERLAP", "0")

    # Convert CoSQA entries to generic BenchmarkDoc format
    docs = [cosqa_entry_to_doc(e) for e in corpus_entries]

    # Get client and model
    client = get_qdrant_client()
    model = get_embedding_model()

    # Delegate to shared benchmark indexer
    if force:
        recreate = True

    result = index_benchmark_corpus(
        docs=docs,
        client=client,
        model=model,
        collection=collection,
        batch_size=batch_size,
        recreate=recreate,
        skip_if_exists=not force,
    )

    # Convert result format to match old API
    return {
        "indexed": result.get("indexed_count", 0),
        "skipped": result.get("skipped_count", 0),
        "errors": 0,
        "reused": result.get("reused", False),
    }


def verify_collection(collection: str = DEFAULT_COLLECTION) -> Dict[str, Any]:
    """Verify collection exists and return stats."""
    client = get_qdrant_client()
    try:
        info = client.get_collection(collection)
        return {
            "exists": True,
            "points_count": info.points_count,
            "status": str(info.status),
        }
    except Exception as e:
        return {"exists": False, "error": str(e)}
