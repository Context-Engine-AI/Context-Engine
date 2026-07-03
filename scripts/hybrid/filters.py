#!/usr/bin/env python3
"""
Filter and classification logic extracted from hybrid_search.py.

This module provides:
- File pattern constants for core, test, vendor, and language detection
- Classification functions: is_test_file, is_core_file, is_vendor_path, lang_matches_path
- Query DSL parsing: parse_query_dsl
- Tokenization helpers: tokenize_queries, _split_ident, _STOP
- Filter sanitization: _sanitize_filter_obj with caching
"""

__all__ = [
    "CORE_FILE_PATTERNS", "NON_CORE_PATTERNS", "TEST_FILE_PATTERNS", "VENDOR_PATTERNS", "LANG_EXTS",
    "is_test_file", "is_core_file", "is_vendor_path", "lang_matches_path",
    "parse_query_dsl", "_STOP", "_split_ident", "tokenize_queries",
]

import re
import threading
from typing import Any, Dict, List, Tuple

# =============================================================================
# File pattern constants
# =============================================================================

# Core file patterns (prioritize implementation over tests/docs)
CORE_FILE_PATTERNS = [
    r"\.py$",
    r"\.js$",
    r"\.ts$",
    r"\.tsx$",
    r"\.jsx$",
    r"\.go$",
    r"\.rs$",
    r"\.java$",
    r"\.cpp$",
    r"\.c$",
    r"\.h$",
]

NON_CORE_PATTERNS = [
    r"test",
    r"spec",
    r"__test__",
    r"\.test\.",
    r"\.spec\.",
    r"_test\.py$",
    r"_spec\.py$",
    r"docs?/",
    r"documentation/",
    r"\.md$",
    r"\.txt$",
    r"README",
    r"CHANGELOG",
]

# Test file patterns
TEST_FILE_PATTERNS = [
    r"/tests?/",
    r"(^|/)test_",
    r"_test\.",
    r"\.test\.",
    r"\.spec\.",
]

# Vendor/third-party detection
VENDOR_PATTERNS = [
    "vendor/",
    "third_party/",
    "node_modules/",
    "/dist/",
    "/build/",
    ".generated/",
    "generated/",
    "autogen/",
    "target/",
]

# Language extension mapping
LANG_EXTS: Dict[str, List[str]] = {
    "python": [".py"],
    "typescript": [".ts", ".tsx"],
    "javascript": [".js", ".jsx"],
    "go": [".go"],
    "rust": [".rs"],
    "java": [".java"],
    "cpp": [".cpp", ".cc", ".cxx", ".hpp", ".h"],
    "c": [".c", ".h"],
    "csharp": [".cs", ".csx"],
    "razor": [".cshtml", ".razor"],
    "xml": [".csproj", ".resx", ".config"],
}


# =============================================================================
# Classification functions
# =============================================================================

def is_test_file(path: str) -> bool:
    """Check if path matches test file patterns."""
    p = path.lower()
    for pattern in TEST_FILE_PATTERNS:
        if re.search(pattern, p):
            return True
    return False


def is_core_file(path: str) -> bool:
    """Check if file is core implementation (not test/doc)."""
    path_lower = path.lower()
    # Skip non-core files
    for pattern in NON_CORE_PATTERNS:
        if re.search(pattern, path_lower):
            return False
    # Check for core file extensions
    for pattern in CORE_FILE_PATTERNS:
        if re.search(pattern, path_lower):
            return True
    return False


def is_vendor_path(path: str) -> bool:
    """Check if path is in vendor/third-party directories."""
    p = path.lower()
    return any(s in p for s in VENDOR_PATTERNS)


def lang_matches_path(lang: str, path: str) -> bool:
    """Check if language matches file path extension."""
    if not lang:
        return False
    exts = LANG_EXTS.get(lang.lower(), [])
    pl = path.lower()
    return any(pl.endswith(ext) for ext in exts)


# =============================================================================
# Query DSL parsing
# =============================================================================

def parse_query_dsl(queries: List[str]) -> Tuple[List[str], Dict[str, str]]:
    """
    Parse query DSL tokens from query strings.
    
    Supported tokens: lang:, language:, file:, path:, under:, kind:, symbol:, ext:, not:, case:, repo:
    
    Returns:
        Tuple of (clean_queries, extracted_tokens)
    """
    clean: List[str] = []
    extracted: Dict[str, str] = {}
    token_re = re.compile(
        r"\b(?:(lang|language|file|path|under|kind|symbol|ext|not|case|repo))\s*:\s*([^\s]+)",
        re.IGNORECASE,
    )
    for q in queries:
        parts = []
        last = 0
        for m in token_re.finditer(q):
            key = m.group(1).lower()
            val = m.group(2)
            if key in ("file", "path"):
                extracted["under"] = val
            elif key in ("lang", "language"):
                extracted["language"] = val
            elif key in ("ext",):
                extracted["ext"] = val
            elif key in ("not",):
                extracted["not"] = val
            elif key in ("case",):
                extracted["case"] = val
            elif key in ("repo",):
                extracted["repo"] = val
            else:
                extracted[key] = val
            parts.append(q[last : m.start()].strip())
            last = m.end()
        parts.append(q[last:].strip())
        remaining = " ".join([p for p in parts if p])
        if remaining:
            clean.append(remaining)
    # Keep at least an empty query if everything was tokens (filter-only mode)
    # Callers should filter empty strings before embedding
    if not clean and queries:
        clean = [""]
    return clean, extracted


# =============================================================================
# Tokenization helpers
# =============================================================================

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


def _split_ident(s: str) -> List[str]:
    """Split snake_case and camelCase identifiers into tokens."""
    parts = re.split(r"[^A-Za-z0-9]+", s)
    out: List[str] = []
    for p in parts:
        if not p:
            continue
        # camelCase split
        segs = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+", p)
        out.extend([x for x in segs if x])
    return [x.lower() for x in out if x and x.lower() not in _STOP]


def tokenize_queries(phrases: List[str]) -> List[str]:
    """Tokenize query phrases into individual terms, removing stopwords and deduping."""
    toks: List[str] = []
    for ph in phrases:
        toks.extend(_split_ident(ph))
    # de-dup preserving order
    seen = set()
    out: List[str] = []
    for t in toks:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out


# =============================================================================
# Filter sanitization with caching
# =============================================================================

_FILTER_CACHE: Dict[int, Any] = {}
_FILTER_CACHE_LOCK = threading.Lock()
_FILTER_CACHE_MAX = 256


def _sanitize_filter_obj(flt: Any) -> Any:
    """
    Sanitize filter objects for Qdrant queries.
    
    Handles both model-style objects and dict-like filters.
    Uses caching to avoid repeated deep copies.
    
    Args:
        flt: Filter object (models.Filter, dict, or None)
        
    Returns:
        Sanitized filter or None if empty/invalid
    """
    if flt is None:
        return None

    # Try cache first (hash by id for object identity)
    cache_key = id(flt)
    with _FILTER_CACHE_LOCK:
        if cache_key in _FILTER_CACHE:
            return _FILTER_CACHE[cache_key]

    try:
        # Try model-style attributes first
        must = getattr(flt, "must", None)
        should = getattr(flt, "should", None)
        must_not = getattr(flt, "must_not", None)
        if must is None and should is None and must_not is None:
            # Maybe dict-like
            if isinstance(flt, dict):
                m = [c for c in (flt.get("must") or []) if c is not None]
                s = [c for c in (flt.get("should") or []) if c is not None]
                mn = [c for c in (flt.get("must_not") or []) if c is not None]
                result = None if (not m and not s and not mn) else flt
            else:
                # Unknown structure -> drop
                result = None
        else:
            m = [c for c in (must or []) if c is not None]
            s = [c for c in (should or []) if c is not None]
            mn = [c for c in (must_not or []) if c is not None]
            result = None if (not m and not s and not mn) else flt
    except Exception:
        result = None

    # Cache result (with size limit)
    with _FILTER_CACHE_LOCK:
        if len(_FILTER_CACHE) < _FILTER_CACHE_MAX:
            _FILTER_CACHE[cache_key] = result

    return result


def clear_filter_cache() -> None:
    """Clear the filter sanitization cache."""
    with _FILTER_CACHE_LOCK:
        _FILTER_CACHE.clear()
