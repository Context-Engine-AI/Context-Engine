#!/usr/bin/env python3
"""
mcp/info_request.py - Info request helpers for MCP indexer server.

Extracted from mcp_indexer_server.py for better modularity.
Contains:
- Helper functions for info_request tool
"""

from __future__ import annotations

__all__ = [
    "_extract_symbols_from_query",
    "_extract_related_concepts",
    "_format_information_field",
    "_extract_relationships",
    "_calculate_confidence",
]

import re
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# Import _split_ident for tokenization
from scripts.mcp_impl.utils import _split_ident


def _extract_symbols_from_query(query: str) -> list[str]:
    """Extract potential symbol names from a query string."""
    # Match CamelCase, snake_case, or standalone words that look like identifiers
    patterns = [
        r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b',  # CamelCase
        r'\b[a-z_][a-z0-9_]*(?:_[a-z0-9]+)+\b',  # snake_case
        r'\b(?:def|class|function|method|async)\s+(\w+)',  # explicit mentions
    ]
    symbols = set()
    for pat in patterns:
        for m in re.finditer(pat, query):
            sym = m.group(1) if m.lastindex else m.group(0)
            if len(sym) > 2:
                symbols.add(sym)
    return list(symbols)[:5]  # Limit to top 5


def _extract_related_concepts(query: str, results: list) -> list[str]:
    """Extract related technical concepts dynamically from results (codebase-agnostic)."""
    concepts = set()

    # Extract from results - this works on any codebase
    for r in results[:10]:
        # From symbols: split CamelCase/snake_case into meaningful parts
        sym = r.get("symbol", "") or ""
        if sym and len(sym) > 2:
            parts = [p for p in re.split(r'(?=[A-Z])|_|-', sym) if p and len(p) > 2]
            for part in parts[:3]:
                concepts.add(part.lower())

        # From file paths: extract directory/module names
        path = r.get("path", "") or ""
        if path:
            path_parts = path.replace("\\", "/").split("/")
            for pp in path_parts[-3:]:  # Last 3 path segments
                # Remove extension and split
                name = pp.rsplit(".", 1)[0] if "." in pp else pp
                if name and len(name) > 2 and not name.startswith("_"):
                    concepts.add(name.lower())

        # From kind: function, class, method, etc.
        kind = r.get("kind", "") or ""
        if kind and len(kind) > 2:
            concepts.add(kind.lower())

    # From query: extract significant words (skip common words)
    skip_words = {"the", "is", "are", "how", "does", "what", "where", "find", "get", "set", "for", "and", "with"}
    query_parts = re.split(r'\W+', query.lower())
    for qp in query_parts:
        if qp and len(qp) > 2 and qp not in skip_words:
            concepts.add(qp)

    # Sort by frequency in results for relevance
    return list(concepts)[:10]


def _format_information_field(result: dict) -> str:
    """Generate human-readable information field for a result."""
    path = result.get("path", "")
    symbol = result.get("symbol", "")
    start = result.get("start_line", 0)
    end = result.get("end_line", 0)
    kind = result.get("kind", "")

    # Get just the filename
    filename = path.split("/")[-1] if "/" in path else path

    if symbol and kind:
        return f"Found {kind} '{symbol}' in {filename} (lines {start}-{end})"
    elif symbol:
        return f"Found '{symbol}' in {filename} (lines {start}-{end})"
    else:
        return f"Found match in {filename} (lines {start}-{end})"


def _extract_relationships(result: dict) -> dict:
    """Extract relationship metadata (imports, calls) from a result."""
    relations = result.get("relations") or {}
    # Get from relations object if present
    imports = relations.get("imports") or []
    calls = relations.get("calls") or []
    symbol_path = relations.get("symbol_path") or ""
    # Also check top-level metadata (fallback)
    if not imports:
        imports = result.get("imports") or []
    if not calls:
        calls = result.get("calls") or []
    # Get related paths if available
    related_paths = result.get("related_paths") or []

    return {
        "imports_from": imports[:10] if imports else [],  # Limit to 10
        "calls": calls[:10] if calls else [],
        "symbol_path": symbol_path,
        "related_paths": related_paths[:5] if related_paths else [],
    }


def _calculate_confidence(query: str, results: list) -> dict:
    """Calculate confidence metrics for the search."""
    if not results:
        return {"level": "none", "score": 0.0, "reason": "no_results"}

    avg_score = sum(r.get("score", 0) for r in results) / len(results)
    top_score = results[0].get("score", 0) if results else 0

    # Check if query terms match symbols
    query_tokens = set(_split_ident(query.lower()))
    symbol_matches = sum(
        1 for r in results[:5]
        if any(tok in _split_ident((r.get("symbol", "") or "").lower())
               for tok in query_tokens)
    )

    if top_score > 0.8 and symbol_matches > 0:
        level = "high"
    elif avg_score > 0.6:
        level = "medium"
    elif results:
        level = "low"
    else:
        level = "none"

    return {
        "level": level,
        "score": round(avg_score, 3),
        "top_score": round(top_score, 3),
        "symbol_matches": symbol_matches,
    }

