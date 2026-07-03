"""Pattern search MCP tool implementation.

Single unified tool that handles both code examples and natural language descriptions.
Supports TOON output format for token-efficient responses.
"""
from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Union

# Import logger with fallback
try:
    from scripts.logger import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

# Import pattern detection components (lazy to avoid startup penalty)
_PATTERN_SEARCH_LOADED = False
_pattern_search_fn = None
_search_by_pattern_description_fn = None


def _ensure_pattern_search():
    """Lazy load pattern search module."""
    global _PATTERN_SEARCH_LOADED, _pattern_search_fn, _search_by_pattern_description_fn
    if _PATTERN_SEARCH_LOADED:
        return True
    try:
        from scripts.pattern_detection.search import (
            pattern_search,
            search_by_pattern_description,
        )
        _pattern_search_fn = pattern_search
        _search_by_pattern_description_fn = search_by_pattern_description
        _PATTERN_SEARCH_LOADED = True
        return True
    except ImportError as e:
        logger.warning(f"Pattern search not available: {e}")
        return False


# Supported languages for tree-sitter parsing
_SUPPORTED_LANGUAGES = {
    "python", "javascript", "typescript", "go", "rust", "java", "c", "cpp",
    "ruby", "php", "csharp", "kotlin", "swift", "scala", "bash", "lua",
}

# Fenced code block pattern
_FENCED_CODE = re.compile(r'^```\w*\n.*\n```$', re.DOTALL)

# Universal code syntax patterns (work across all languages)
_CODE_SYNTAX = re.compile(
    r'[{}\[\]();]|'                          # Brackets, braces, parens, semicolons
    r'::|->|=>|:=|'                          # C++/Rust/Go/JS operators
    r'\.\w+\(|'                              # Method call: .foo(
    r'\w+\s*\([^)]*\)|'                      # Function call: foo() or foo(args)
    r'^\s*(def|func|fn|function|class|struct|enum|impl|trait|interface)\s+\w',  # Definitions
    re.MULTILINE
)

# Multi-line code indicators (braces/semicolons at line boundaries)
_MULTILINE_CODE = re.compile(
    r'[{}]\s*$|'                 # Brace at line end
    r'^\s*[{}]|'                 # Brace at line start
    r';\s*$',                    # Semicolon at line end
    re.MULTILINE
)


def _detect_query_mode(text: str, language: str | None) -> str:
    """
    Auto-detect if text is code or natural language description.

    Works across all 16+ supported languages using universal syntax patterns.
    Returns: "code" or "description"
    """
    text = text.strip()
    if not text:
        return "description"

    # 1. Fenced code block → code
    if _FENCED_CODE.match(text):
        return "code"

    # 2. Multi-line with braces/semicolons → code
    if '\n' in text and _MULTILINE_CODE.search(text):
        return "code"

    # 3. Universal code syntax (brackets, operators, calls, definitions)
    if _CODE_SYNTAX.search(text):
        return "code"

    # 4. Language hint is advisory only, but allow a narrow "bare identifiers" case.
    # Some clients send extremely minimal snippets (e.g. "some text") with a language hint.
    # Treat exactly-two identifier tokens as code when a supported language hint is present.
    if language and language.lower() in _SUPPORTED_LANGUAGES:
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*\s+[A-Za-z_][A-Za-z0-9_]*", text):
            return "code"
        return "description"

    # 5. Default to natural language
    return "description"


async def _pattern_search_impl(
    query: Optional[str] = None,
    language: Optional[str] = None,
    limit: Optional[int] = None,
    min_score: Optional[float] = None,
    include_snippet: Optional[bool] = None,
    context_lines: Optional[int] = None,
    hybrid: Optional[bool] = None,
    semantic_weight: Optional[float] = None,
    collection: Optional[str] = None,
    target_languages: Optional[List[str]] = None,
    output_format: Optional[str] = None,
    compact: Optional[bool] = None,
    aroma_rerank: Optional[bool] = None,
    aroma_alpha: Optional[float] = None,
    query_mode: Optional[str] = None,  # "code", "description", or "auto" (default)
    coerce_bool_fn=None,
    coerce_int_fn=None,
    coerce_float_fn=None,
) -> Dict[str, Any]:
    """Unified pattern search - handles both code examples and NL descriptions."""
    if not _ensure_pattern_search():
        return {"ok": False, "error": "Pattern search module not available"}

    if not query or not str(query).strip():
        return {"ok": False, "error": "query parameter is required"}

    query_text = str(query).strip()

    # Coerce parameters - handle string "false"/"0" correctly
    def _default_coerce_bool(v, d):
        if v is None:
            return d
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.strip().lower() in ("1", "true", "yes", "on")
        return bool(v)

    def _safe_coerce_int(v, d):
        if v is None:
            return d
        try:
            return int(v)
        except (ValueError, TypeError):
            return d

    def _safe_coerce_float(v, d):
        if v is None:
            return d
        try:
            return float(v)
        except (ValueError, TypeError):
            return d

    _coerce_bool = coerce_bool_fn or _default_coerce_bool
    _coerce_int = coerce_int_fn or _safe_coerce_int
    _coerce_float = coerce_float_fn or _safe_coerce_float

    # Defaults aligned with core pattern_search API for consistent behavior
    eff_limit = _coerce_int(limit, 10)
    eff_include_snippet = _coerce_bool(include_snippet, True)
    eff_context_lines = _coerce_int(context_lines, 3)
    eff_hybrid = _coerce_bool(hybrid, False)
    eff_semantic_weight = _coerce_float(semantic_weight, 0.3)
    eff_compact = _coerce_bool(compact, False)
    eff_aroma_rerank = _coerce_bool(aroma_rerank, True)  # AROMA enabled by default
    eff_aroma_alpha = _coerce_float(aroma_alpha, 0.6)

    # Determine query mode: explicit override or auto-detect
    eff_language = str(language).strip() if language else None
    eff_query_mode = str(query_mode).strip().lower() if query_mode else "auto"

    if eff_query_mode == "code":
        is_code = True
    elif eff_query_mode == "description":
        is_code = False
    else:  # auto
        detected = _detect_query_mode(query_text, eff_language)
        is_code = (detected == "code")

    # Path-specific min_score defaults:
    # - Code path: 0.5 (vector similarity scores are typically higher)
    # - NL path: 0.0 (keyword overlap scores are often low, don't filter by default)
    eff_min_score = _coerce_float(min_score, 0.5 if is_code else 0.0)

    try:
        if is_code:
            # Structural pattern search using code example
            result = _pattern_search_fn(
                example=query_text,
                language=eff_language or "python",
                limit=eff_limit,
                min_score=eff_min_score,
                include_snippet=eff_include_snippet,
                context_lines=eff_context_lines,
                hybrid=eff_hybrid,
                semantic_weight=eff_semantic_weight,
                collection=collection,
                target_languages=target_languages,
                output_format=output_format,
                compact=eff_compact,
                aroma_rerank=eff_aroma_rerank,
                aroma_alpha=eff_aroma_alpha,
            )
        else:
            # Natural language pattern description search
            result = _search_by_pattern_description_fn(
                description=query_text,
                limit=eff_limit,
                min_score=eff_min_score,
                collection=collection,
                target_languages=target_languages,
                output_format=output_format,
                compact=eff_compact,
            )

        # Convert response object to dict if needed
        if not isinstance(result, dict):
            result = result.to_dict()

        # Preserve upstream ok flag (derived from search_mode) instead of overriding
        # This ensures errors from core search propagate to MCP clients
        result["query_mode"] = "code" if is_code else "description"

        return result
    except Exception as e:
        logger.error(f"Pattern search failed: {e}")
        return {"ok": False, "error": str(e)}
