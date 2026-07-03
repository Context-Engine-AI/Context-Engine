#!/usr/bin/env python3
"""
ingest/tree_sitter.py - Tree-sitter setup and language loading.

This module handles tree-sitter parser initialization, language package loading,
and provides utilities for checking tree-sitter availability.
"""
from __future__ import annotations

import os
from typing import Dict, Any, List

# ---------------------------------------------------------------------------
# Tree-sitter language registry
# ---------------------------------------------------------------------------
_TS_LANGUAGES: Dict[str, Any] = {}
_TS_AVAILABLE = False
_TS_WARNED = False

# Try to import tree-sitter (0.25+ API)
try:
    from tree_sitter import Parser, Language  # type: ignore

    def _load_ts_language(mod: Any, *, preferred: list[str] | None = None) -> Any | None:
        """Return a tree-sitter Language instance from a per-language package.

        Different packages expose different entrypoints (e.g. language(),
        language_typescript(), language_tsx()).
        """
        preferred = preferred or []
        candidates: list[Any] = []
        if getattr(mod, "language", None) is not None and callable(getattr(mod, "language")):
            candidates.append(getattr(mod, "language"))
        for name in preferred:
            fn = getattr(mod, name, None)
            if fn is not None and callable(fn):
                candidates.append(fn)
        # Last resort: scan for any callable language* attribute
        for name in dir(mod):
            if not name.startswith("language"):
                continue
            fn = getattr(mod, name, None)
            if fn is not None and callable(fn):
                candidates.append(fn)

        for fn in candidates:
            try:
                raw_lang = fn()
                return raw_lang if isinstance(raw_lang, Language) else Language(raw_lang)
            except Exception:
                continue
        return None

    # Import all available language packages
    for lang_name, pkg_name in [
        ("python", "tree_sitter_python"),
        ("javascript", "tree_sitter_javascript"),
        ("typescript", "tree_sitter_typescript"),
        ("go", "tree_sitter_go"),
        ("rust", "tree_sitter_rust"),
        ("java", "tree_sitter_java"),
        ("c", "tree_sitter_c"),
        ("cpp", "tree_sitter_cpp"),
        ("ruby", "tree_sitter_ruby"),
        ("c_sharp", "tree_sitter_c_sharp"),
        ("bash", "tree_sitter_bash"),
        ("json", "tree_sitter_json"),
        ("yaml", "tree_sitter_yaml"),
        ("html", "tree_sitter_html"),
        ("css", "tree_sitter_css"),
        ("markdown", "tree_sitter_markdown"),
    ]:
        try:
            mod = __import__(pkg_name)
            preferred: list[str] = []
            if lang_name == "typescript":
                preferred = ["language_typescript"]
            elif lang_name == "c_sharp":
                preferred = ["language_c_sharp", "language_csharp"]
            lang = _load_ts_language(mod, preferred=preferred)
            if lang is not None:
                _TS_LANGUAGES[lang_name] = lang
                # Also load TSX if provided by the typescript package
                if lang_name == "typescript":
                    tsx_lang = _load_ts_language(mod, preferred=["language_tsx"])
                    if tsx_lang is not None:
                        _TS_LANGUAGES["tsx"] = tsx_lang
        except Exception:
            pass  # Language package not installed

    # Add aliases
    if "javascript" in _TS_LANGUAGES:
        _TS_LANGUAGES["jsx"] = _TS_LANGUAGES["javascript"]
    if "c_sharp" in _TS_LANGUAGES:
        _TS_LANGUAGES["csharp"] = _TS_LANGUAGES["c_sharp"]
    if "bash" in _TS_LANGUAGES:
        _TS_LANGUAGES["shell"] = _TS_LANGUAGES["bash"]
        _TS_LANGUAGES["sh"] = _TS_LANGUAGES["bash"]

    _TS_AVAILABLE = len(_TS_LANGUAGES) > 0

except Exception:  # pragma: no cover
    Parser = None  # type: ignore
    Language = None  # type: ignore
    _TS_LANGUAGES = {}
    _TS_AVAILABLE = False

    def _load_ts_language(mod: Any, *, preferred: list[str] | None = None) -> Any | None:
        """Stub when tree-sitter is not available."""
        return None


def _use_tree_sitter() -> bool:
    """Check if tree-sitter should be used for parsing."""
    global _TS_WARNED
    val = os.environ.get("USE_TREE_SITTER")
    # Default ON when libs are available; allow explicit disable via 0/false
    if val is None or str(val).strip() == "":
        want = True
    else:
        want = str(val).strip().lower() in {"1", "true", "yes", "on"}
    if want and not _TS_AVAILABLE and not _TS_WARNED:
        print(
            "[WARN] USE_TREE_SITTER=1 but tree-sitter libs not available; falling back to regex heuristics"
        )
        _TS_WARNED = True
    return _TS_AVAILABLE and want


def _ts_parser(lang_key: str):
    """Return a tree-sitter Parser for the given language key.

    Uses tree-sitter 0.25+ API with pre-loaded Language objects.
    """
    if not _use_tree_sitter():
        return None

    if Parser is None or lang_key not in _TS_LANGUAGES:
        return None

    try:
        lang = _TS_LANGUAGES[lang_key]
        # Support both APIs:
        # - tree_sitter<0.25 (or some builds): Parser(lang)
        # - tree_sitter>=0.25: Parser(); parser.set_language(lang)
        try:
            p = Parser()  # type: ignore[call-arg]
            if hasattr(p, "set_language"):
                p.set_language(lang)  # type: ignore[attr-defined]
                return p
            # Fallback if set_language isn't present but language attribute is
            try:
                setattr(p, "language", lang)
                return p
            except Exception:
                pass
        except Exception:
            pass
        return Parser(lang)  # type: ignore[misc]
    except Exception:
        return None


# Export Parser and Language for type hints
__all__ = [
    "_TS_LANGUAGES",
    "_TS_AVAILABLE",
    "_use_tree_sitter",
    "_ts_parser",
    "_load_ts_language",
    "Parser",
    "Language",
]
