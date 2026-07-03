#!/usr/bin/env python3
"""
ingest/pseudo.py - ReFRAG pseudo-description and tag generation.

This module provides functions for generating pseudo descriptions and tags
for code chunks using LLM decoders (GLM or llama.cpp).
"""
from __future__ import annotations

import os
from typing import Tuple, List

from scripts.ingest.config import (
    get_cached_pseudo,
    set_cached_pseudo,
    compare_symbol_changes,
)


def _pseudo_describe_enabled() -> bool:
    """Check if pseudo description generation is enabled."""
    val = str(os.environ.get("REFRAG_PSEUDO_DESCRIBE", "0")).strip().lower()
    return val in {"1", "true", "yes", "on"}


def _smart_symbol_reindexing_enabled() -> bool:
    """Check if symbol-aware reindexing is enabled."""
    try:
        return str(os.environ.get("SMART_SYMBOL_REINDEXING", "0")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
    except Exception:
        return False


def generate_pseudo_tags(text: str) -> Tuple[str, List[str]]:
    """Best-effort: ask local decoder to produce a short label and 3-6 tags.
    
    Returns (pseudo, tags). On failure returns ("", []).
    """
    pseudo: str = ""
    tags: list[str] = []
    if not _pseudo_describe_enabled() or not text.strip():
        return pseudo, tags
    try:
        from scripts.refrag_llamacpp import (
            LlamaCppRefragClient,
            is_decoder_enabled,
            get_runtime_kind,
        )
        if not is_decoder_enabled():
            return "", []
        runtime = get_runtime_kind()
        # Keep decoding tight/fast – this is only enrichment for retrieval.
        if runtime == "glm":
            prompt = (
                "You are a JSON-only function that labels code spans for search enrichment.\n"
                "Respond with a single JSON object and nothing else (no prose, no markdown).\n"
                "Exact format: {\"pseudo\": string (<=20 tokens), \"tags\": [3-6 short strings]}.\n"
                "Code:\n" + text[:2000]
            )
            from scripts.refrag_glm import GLMRefragClient
            client = GLMRefragClient()
            out = client.generate_with_soft_embeddings(
                prompt=prompt,
                max_tokens=int(os.environ.get("PSEUDO_MAX_TOKENS", "96") or 96),
                temperature=float(os.environ.get("PSEUDO_TEMPERATURE", "0.10") or 0.10),
                top_p=float(os.environ.get("PSEUDO_TOP_P", "0.9") or 0.9),
                stop=["\n\n"],
                force_json=True,
            )
        else:
            prompt = (
                "You label code spans for search enrichment.\n"
                "Return strictly JSON: {\"pseudo\": string (<=20 tokens), \"tags\": [3-6 short strings]}.\n"
                "Code:\n" + text[:2000]
            )
            client = LlamaCppRefragClient()
            out = client.generate_with_soft_embeddings(
                prompt=prompt,
                max_tokens=int(os.environ.get("PSEUDO_MAX_TOKENS", "96") or 96),
                temperature=float(os.environ.get("PSEUDO_TEMPERATURE", "0.10") or 0.10),
                top_k=int(os.environ.get("PSEUDO_TOP_K", "30") or 30),
                top_p=float(os.environ.get("PSEUDO_TOP_P", "0.9") or 0.9),
                stop=["\n\n"],
            )
        import json as _json
        from scripts.llm_utils import strip_markdown_fences
        try:
            obj = _json.loads(strip_markdown_fences(out))
            if isinstance(obj, dict):
                p = obj.get("pseudo")
                t = obj.get("tags")
                if isinstance(p, str):
                    pseudo = p.strip()[:256]
                if isinstance(t, list):
                    tags = [str(x).strip() for x in t if str(x).strip()][:6]
        except Exception:
            pass
    except Exception:
        return "", []
    return pseudo, tags


def should_process_pseudo_for_chunk(
    file_path: str, chunk: dict, changed_symbols: set
) -> Tuple[bool, str, List[str]]:
    """Determine if a chunk needs pseudo processing based on symbol changes AND pseudo cache.

    Uses existing symbol change detection and pseudo cache lookup for optimal performance.

    Args:
        file_path: Path to the file containing this chunk
        chunk: Chunk dict with symbol information
        changed_symbols: Set of symbol IDs that changed (from compare_symbol_changes)

    Returns:
        (needs_processing, cached_pseudo, cached_tags)
    """
    # For chunks without symbol info, process them (fallback - no symbol to reuse from)
    symbol_name = chunk.get("symbol", "")
    if not symbol_name:
        return True, "", []

    # Create symbol ID matching the format used in symbol cache
    kind = chunk.get("kind", "unknown")
    start_line = chunk.get("start", 0)
    symbol_id = f"{kind}_{symbol_name}_{start_line}"

    # If we don't have any change information, best effort: try reusing cached pseudo when present
    if not changed_symbols and get_cached_pseudo:
        try:
            cached_pseudo, cached_tags = get_cached_pseudo(file_path, symbol_id)
            if cached_pseudo or cached_tags:
                return False, cached_pseudo, cached_tags
        except Exception:
            pass
        return True, "", []

    # Unchanged symbol: prefer reuse when cached pseudo/tags exist
    if symbol_id not in changed_symbols:
        if get_cached_pseudo:
            try:
                cached_pseudo, cached_tags = get_cached_pseudo(file_path, symbol_id)
                if cached_pseudo or cached_tags:
                    return False, cached_pseudo, cached_tags
            except Exception:
                pass
        # Unchanged but no cached data yet – process once
        return True, "", []

    # Symbol content changed: always re-run pseudo; do not reuse stale cached values
    return True, "", []


def should_use_smart_reindexing(file_path: str, file_hash: str) -> Tuple[bool, str]:
    """Determine if smart reindexing should be used for a file.

    Returns:
        (use_smart, reason)
    """
    from scripts.ingest.config import get_cached_symbols, compare_symbol_changes
    from scripts.ingest.symbols import extract_symbols_with_tree_sitter
    
    if not _smart_symbol_reindexing_enabled():
        return False, "smart_reindexing_disabled"

    if not get_cached_symbols:
        return False, "symbol_cache_unavailable"

    # Load cached symbols
    cached_symbols = get_cached_symbols(file_path)
    if not cached_symbols:
        return False, "no_cached_symbols"

    # Extract current symbols
    current_symbols = extract_symbols_with_tree_sitter(file_path)
    if not current_symbols:
        return False, "no_current_symbols"

    # Compare symbols
    unchanged_symbols, changed_symbols = compare_symbol_changes(cached_symbols, current_symbols)

    total_symbols = len(current_symbols)
    changed_ratio = len(changed_symbols) / max(total_symbols, 1)

    # Use thresholds to decide strategy
    max_changed_ratio = float(os.environ.get("MAX_CHANGED_SYMBOLS_RATIO", "0.3"))
    if changed_ratio > max_changed_ratio:
        return False, f"too_many_changes_{changed_ratio:.2f}"

    print(f"[SMART_REINDEX] {file_path}: {len(unchanged_symbols)} unchanged, {len(changed_symbols)} changed")
    return True, f"smart_reindex_{len(changed_symbols)}/{total_symbols}"
