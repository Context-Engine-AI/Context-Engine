#!/usr/bin/env python3
"""
ingest/chunking.py - Code chunking utilities.

This module provides various chunking strategies: line-based, semantic (AST-aware),
and token-based micro-chunking for the ReFRAG system.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Dict, Any

from scripts.ingest.config import ROOT_DIR
from scripts.ingest.tree_sitter import _use_tree_sitter, _TS_LANGUAGES

# Import AST analyzer for enhanced semantic chunking
try:
    from scripts.ast_analyzer import get_ast_analyzer, chunk_code_semantically
    _AST_ANALYZER_AVAILABLE = True
except ImportError:
    _AST_ANALYZER_AVAILABLE = False


# Cache tokenizers loaded from TOKENIZER_JSON (or default) to avoid repeatedly
# re-reading tokenizer.json from disk during micro-chunking.
_TOKENIZER_CACHE: dict[str, Any] = {}


def chunk_lines(text: str, max_lines: int = 120, overlap: int = 20) -> List[Dict]:
    """Chunk text into overlapping line-based segments."""
    lines = text.splitlines()
    chunks = []
    i = 0
    n = len(lines)
    while i < n:
        j = min(n, i + max_lines)
        chunk = "\n".join(lines[i:j])
        chunks.append({"text": chunk, "start": i + 1, "end": j})
        if j == n:
            break
        i = max(j - overlap, i + 1)
    return chunks


def chunk_semantic(
    text: str, language: str, max_lines: int = 120, overlap: int = 20
) -> List[Dict]:
    """AST-aware chunking that tries to keep complete functions/classes together."""
    # Import here to avoid circular imports
    from scripts.ingest.symbols import _extract_symbols
    
    # Try enhanced AST analyzer first (if available).
    # This is capability-driven: only use it when the analyzer can actually support the
    # requested language given current USE_TREE_SITTER configuration.
    use_enhanced = os.environ.get("INDEX_USE_ENHANCED_AST", "1").lower() in {"1", "true", "yes", "on"}
    _ast_supported = False
    if use_enhanced and _AST_ANALYZER_AVAILABLE:
        try:
            # ast_analyzer internally respects USE_TREE_SITTER when constructing the analyzer
            # (see scripts/ast_analyzer.py:get_ast_analyzer).
            from scripts.ast_analyzer import get_ast_analyzer  # type: ignore

            analyzer = get_ast_analyzer()
            lang_key = str(language or "").strip().lower()
            # Supported either via builtin ast (python) or via tree-sitter when enabled.
            caps = getattr(analyzer, "supported_languages", {}) or {}
            lang_caps = caps.get(lang_key) if isinstance(caps, dict) else None
            if lang_key == "python":
                _ast_supported = True
            elif (
                isinstance(lang_caps, dict)
                and bool(lang_caps.get("tree_sitter"))
                and bool(getattr(analyzer, "use_tree_sitter", False))
            ):
                _ast_supported = True
        except Exception:
            _ast_supported = False

    if use_enhanced and _AST_ANALYZER_AVAILABLE and _ast_supported:
        try:
            chunks = chunk_code_semantically(text, language, max_lines, overlap)
            # Convert to expected format
            return [
                {
                    "text": c["text"],
                    "start": c["start"],
                    "end": c["end"],
                    "is_semantic": c.get("is_semantic", True)
                }
                for c in chunks
            ]
        except Exception as e:
            if os.environ.get("DEBUG_INDEXING"):
                print(f"[DEBUG] Enhanced AST chunking failed, falling back: {e}")
    
    lines = text.splitlines()
    n = len(lines)

    # Extract symbols with line ranges (works for many languages via regex fallbacks,
    # and may optionally use tree-sitter when enabled).
    symbols = _extract_symbols(language, text)
    if not symbols:
        return chunk_lines(text, max_lines, overlap)

    # Sort symbols by start line
    symbols.sort(key=lambda s: s.start)

    chunks = []
    i = 0  # Current line index (0-based)
    sym_idx = 0  # cursor into sorted symbols

    while i < n:
        chunk_start = i + 1  # 1-based for output
        chunk_end = min(n, i + max_lines)  # 1-based

        # Advance symbol cursor to the first symbol that could contain chunk_start
        while sym_idx < len(symbols) and symbols[sym_idx].end < chunk_start:
            sym_idx += 1
        enclosing_symbol = None
        if sym_idx < len(symbols):
            sym = symbols[sym_idx]
            if sym.start <= chunk_start <= sym.end:
                enclosing_symbol = sym

        # Try to find a symbol that starts within our current window
        best_symbol = None
        for sym in symbols[sym_idx:]:
            if sym.start >= chunk_start and sym.start <= chunk_end:
                # Check if the entire symbol fits within max_lines from current position
                symbol_size = sym.end - sym.start + 1
                if symbol_size <= max_lines and sym.end <= i + max_lines:
                    best_symbol = sym
                    break

        if best_symbol:
            # Chunk this complete symbol
            chunk_text = "\n".join(lines[best_symbol.start - 1 : best_symbol.end])
            chunks.append(
                {
                    "text": chunk_text,
                    "start": best_symbol.start,
                    "end": best_symbol.end,
                    "symbol": best_symbol.name,
                    "kind": best_symbol.kind,
                }
            )
            # Move past this symbol with minimal overlap
            i = max(best_symbol.end - overlap, i + 1)
        else:
            # No suitable symbol found, fall back to line-based chunking
            actual_end = min(n, i + max_lines)
            # If we're currently inside a symbol that's too large to fit, avoid
            # crossing the symbol boundary into the next definition.
            if enclosing_symbol is not None:
                actual_end = min(actual_end, enclosing_symbol.end)
            chunk_text = "\n".join(lines[i:actual_end])
            rec = {"text": chunk_text, "start": i + 1, "end": actual_end}
            if enclosing_symbol is not None:
                rec["symbol"] = enclosing_symbol.name
                rec["kind"] = enclosing_symbol.kind
            chunks.append(rec)
            i = max(actual_end - overlap, i + 1)

    return chunks


def chunk_by_tokens(
    text: str, k_tokens: int = None, stride_tokens: int = None
) -> List[Dict]:
    """Token-based micro-chunking (ReFRAG-lite).
    
    Produces tiny fixed-size token windows with stride, maps back to original line ranges.
    """
    try:
        from tokenizers import Tokenizer  # lightweight, already in requirements
    except Exception:
        Tokenizer = None  # type: ignore

    # Prefer explicit function arguments when provided; fall back to env/defaults.
    try:
        if k_tokens is not None:
            k = int(k_tokens)
        else:
            k = int(os.environ.get("MICRO_CHUNK_TOKENS", "16") or 16)
    except Exception:
        k = 16
    try:
        if stride_tokens is not None:
            s = int(stride_tokens)
        else:
            s = int(os.environ.get("MICRO_CHUNK_STRIDE", "") or max(1, k // 2))
    except Exception:
        s = max(1, k // 2)

    # Helper: simple regex-based token offsets when HF tokenizer JSON is unavailable
    def _simple_offsets(txt: str):
        import re
        offs = []
        for m in re.finditer(r"\S+", txt):
            offs.append((m.start(), m.end()))
        return offs

    offsets = []
    # Load tokenizer; default to local model file if present
    tok_path = os.environ.get(
        "TOKENIZER_JSON", str((ROOT_DIR / "models" / "tokenizer.json"))
    )
    if Tokenizer is not None:
        try:
            tokenizer = _TOKENIZER_CACHE.get(tok_path)
            if tokenizer is None:
                tokenizer = Tokenizer.from_file(tok_path)
                _TOKENIZER_CACHE[tok_path] = tokenizer
            try:
                enc = tokenizer.encode(text)
                offsets = getattr(enc, "offsets", None) or []
            except Exception:
                offsets = []
        except Exception:
            offsets = []

    if not offsets:
        # Fallback to simple regex tokenization; avoids degrading to 120-line chunks
        if os.environ.get("DEBUG_CHUNKING"):
            print("[ingest] tokenizers missing/unusable -> using simple regex tokenization")
        offsets = _simple_offsets(text)

    if not offsets:
        return chunk_lines(text, max_lines=120, overlap=20)

    # Precompute line starts for fast char->line mapping
    lines = text.splitlines(keepends=True)
    line_starts = []
    pos = 0
    for ln in lines:
        line_starts.append(pos)
        pos += len(ln)
    total_chars = len(text)

    def char_to_line(c: int) -> int:
        # Binary search line_starts to find 1-based line number
        lo, hi = 0, len(line_starts) - 1
        if c <= 0:
            return 1
        if c >= total_chars:
            return len(lines)
        ans = 0
        while lo <= hi:
            mid = (lo + hi) // 2
            if line_starts[mid] <= c:
                ans = mid
                lo = mid + 1
            else:
                hi = mid - 1
        return ans + 1  # 1-based

    chunks: List[Dict] = []
    i = 0
    n = len(offsets)
    while i < n:
        j = min(n, i + k)
        start_char = offsets[i][0]
        end_char = offsets[j - 1][1] if j - 1 < n else offsets[-1][1]
        start_char = max(0, start_char)
        end_char = min(total_chars, max(start_char, end_char))
        chunk_text = text[start_char:end_char]
        if chunk_text:
            start_line = char_to_line(start_char)
            end_line = (
                char_to_line(end_char - 1) if end_char > start_char else start_line
            )
            chunks.append(
                {
                    "text": chunk_text,
                    "start": start_line,
                    "end": end_line,
                }
            )
        if j == n:
            break
        i = i + s if s > 0 else i + 1
    return chunks
