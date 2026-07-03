#!/usr/bin/env python3
"""
mcp/code_signals.py - Code signal detection for intelligent query targeting.

Extracted from mcp_indexer_server.py for better modularity.
Contains:
- Embedding-based code intent detection (centroids)
- Regex-based code pattern detection
- Symbol extraction from queries
- Code keyword matching
"""

from __future__ import annotations

__all__ = [
    # Constants
    "_CODE_INTENT_CACHE",
    "_CODE_INTENT_LOCK",
    "_CODE_QUERY_ARCHETYPES",
    "_PROSE_QUERY_ARCHETYPES",
    "_CODE_SIGNAL_PATTERNS",
    "_CODE_KEYWORDS",
    # Functions
    "_init_code_intent_centroids",
    "_detect_code_intent_embedding",
    "_detect_code_signals",
]

import logging
import os
import re
import threading
from typing import Any, Dict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Embedding-based code intent detection
# ---------------------------------------------------------------------------
_CODE_INTENT_CACHE: Dict[str, Any] = {"code_centroid": None, "prose_centroid": None, "initialized": False}
_CODE_INTENT_LOCK = threading.Lock()

# Archetypal queries for each category
_CODE_QUERY_ARCHETYPES = [
    "getUserById function implementation",
    "class UserService methods",
    "handleRequest error handling",
    "database connection pool configuration",
    "async function fetchData",
    "struct Config fields",
    "impl Trait for MyType",
    "std::vector push_back",
    "interface IRepository methods",
    "def process_data parameters",
    "React component useState hook",
    "SQL query optimization",
    "API endpoint authentication",
    "constructor initialization",
    "enum Status values",
]

_PROSE_QUERY_ARCHETYPES = [
    "how does the authentication work",
    "explain the data flow",
    "what is the purpose of this module",
    "where is the main entry point",
    "how to configure the database",
    "why was this approach chosen",
    "what are the dependencies",
    "describe the architecture",
    "when should I use this feature",
    "how to run the tests",
]


def _init_code_intent_centroids():
    """Initialize embedding centroids for code vs prose query detection."""
    global _CODE_INTENT_CACHE
    
    # Import here to avoid circular dependency
    from scripts.mcp_impl.admin_tools import _get_embedding_model
    import numpy as np
    
    with _CODE_INTENT_LOCK:
        if _CODE_INTENT_CACHE.get("initialized"):
            return
        try:
            model_name = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
            model = _get_embedding_model(model_name)
            if model is None:
                _CODE_INTENT_CACHE["initialized"] = False
                return

            # Embed archetypes
            code_embeddings = list(model.embed(_CODE_QUERY_ARCHETYPES))
            prose_embeddings = list(model.embed(_PROSE_QUERY_ARCHETYPES))

            # Compute centroids
            code_centroid = np.mean(code_embeddings, axis=0)
            prose_centroid = np.mean(prose_embeddings, axis=0)

            # Normalize
            code_centroid = code_centroid / (np.linalg.norm(code_centroid) + 1e-9)
            prose_centroid = prose_centroid / (np.linalg.norm(prose_centroid) + 1e-9)

            _CODE_INTENT_CACHE["code_centroid"] = code_centroid
            _CODE_INTENT_CACHE["prose_centroid"] = prose_centroid
            _CODE_INTENT_CACHE["initialized"] = True
        except Exception as e:
            if os.environ.get("DEBUG_CODE_SIGNALS"):
                print(f"[DEBUG] Failed to init code intent centroids: {e}")
            _CODE_INTENT_CACHE["initialized"] = False


def _detect_code_intent_embedding(query: str) -> float:
    """Detect code intent using embedding similarity to pre-computed centroids.

    Returns:
        float: 0.0-1.0 indicating code-likeness (1.0 = very code-like)
    """
    # Import here to avoid circular dependency
    from scripts.mcp_impl.admin_tools import _get_embedding_model
    import numpy as np
    
    if not _CODE_INTENT_CACHE.get("initialized"):
        _init_code_intent_centroids()

    if not _CODE_INTENT_CACHE.get("initialized"):
        return 0.5  # Neutral if init failed

    try:
        model_name = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
        model = _get_embedding_model(model_name)
        if model is None:
            return 0.5

        query_embedding = next(model.embed([query]))
        query_embedding = query_embedding / (np.linalg.norm(query_embedding) + 1e-9)

        code_centroid = _CODE_INTENT_CACHE["code_centroid"]
        prose_centroid = _CODE_INTENT_CACHE["prose_centroid"]

        # Cosine similarity
        code_sim = float(np.dot(query_embedding, code_centroid))
        prose_sim = float(np.dot(query_embedding, prose_centroid))

        # Convert to 0-1 score (softmax-ish)
        diff = code_sim - prose_sim
        score = 1.0 / (1.0 + np.exp(-diff * 5))  # Sigmoid with scaling

        return float(score)
    except Exception:
        return 0.5


# ---------------------------------------------------------------------------
# Regex-based code pattern detection
# ---------------------------------------------------------------------------
_CODE_SIGNAL_PATTERNS = {
    "backticks": re.compile(r'`([^`]+)`'),  # `functionName`
    "camelCase": re.compile(r'\b[a-z]+(?:[A-Z][a-z0-9]*)+\b'),  # getUserData
    "PascalCase": re.compile(r'\b[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+\b'),  # MyClassName
    "snake_case": re.compile(r'\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b'),  # my_function
    "SCREAMING_SNAKE": re.compile(r'\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b'),  # MAX_SIZE
    "parentheses": re.compile(r'\b(\w+)\s*\(\)'),  # function()
    "dot_path": re.compile(r'\b\w+(?:\.\w+){2,}\b'),  # module.submodule.func
    "namespace_path": re.compile(r'\b\w+(?:::\w+)+\b'),  # std::vector
    "file_ext": re.compile(
        r'\b[\w/\\-]+\.(?:'
        r'py|pyi|pyx|pxd|js|jsx|ts|tsx|mjs|cjs|go|rs|java|kt|kts|scala|'
        r'c|h|cpp|hpp|cc|hh|cxx|hxx|cs|fs|fsx|vb|swift|m|mm|rb|rake|php|'
        r'lua|pl|pm|sh|bash|zsh|hs|erl|ex|exs|clj|cljs|cljc|lisp|el|'
        r'zig|nim|v|asm|s|sql|tf|hcl|dart|r|jl|ml|mli|proto|thrift|graphql|gql'
        r')\b',
        re.IGNORECASE
    ),
    "path_like": re.compile(r'(?:^|[\s"])(?:\.?\.?/)?(?:[\w-]+/)+[\w.-]+'),
    "generic_type": re.compile(r'\b([A-Z][a-zA-Z0-9]*)<[^>]+>'),  # List<String>
    "interface_I": re.compile(r'\bI[A-Z][a-zA-Z0-9]+\b'),  # IUserRepository
    "type_name": re.compile(r'\b[A-Z][a-z][a-zA-Z0-9]{2,}\b'),  # Vec, String
}

# Keywords that suggest code intent
_CODE_KEYWORDS = frozenset([
    # Universal
    "class", "function", "method", "def", "async", "import", "from",
    "module", "package", "interface", "struct", "enum", "type",
    "variable", "const", "constant", "property", "attribute",
    "constructor", "destructor", "handler", "callback", "hook",
    "implementation", "definition", "declaration", "signature",
    # C/C++
    "namespace", "template", "typedef", "pragma", "inline", "virtual",
    "override", "static", "extern", "sizeof", "nullptr", "constexpr",
    # Rust
    "trait", "impl", "fn", "pub", "mod", "crate", "use", "mut", "ref",
    "where", "dyn", "unsafe", "await", "macro", "derive",
    # Go
    "func", "chan", "defer", "goroutine", "select",
    # Java/Kotlin/Scala
    "abstract", "extends", "implements", "throws", "synchronized",
    "companion", "object", "sealed", "data", "suspend", "lateinit",
    # C#/.NET
    "delegate", "event", "partial", "readonly", "yield",
    "linq", "nullable", "record", "init",
    # Swift
    "protocol", "extension", "guard", "lazy", "weak", "unowned",
    # Ruby
    "attr_accessor", "attr_reader", "require", "include",
    # General
    "api", "endpoint", "route", "controller", "service", "repository",
    "factory", "singleton", "builder", "adapter", "decorator", "facade",
])


def _detect_code_signals(query: str) -> dict:
    """Detect code-like patterns in query and extract potential symbols.

    Returns:
        {
            "has_code_signals": bool,
            "signal_strength": float (0.0-1.0),
            "extracted_symbols": list[str],
            "detected_patterns": list[str],
            "suggested_boosts": dict
        }
    """
    if not query or not isinstance(query, str):
        return {"has_code_signals": False, "signal_strength": 0.0, "extracted_symbols": [], "detected_patterns": [], "suggested_boosts": {}}

    query_lower = query.lower()
    detected_patterns = []
    extracted_symbols = set()
    signal_score = 0.0

    # Check for backtick-wrapped code (highest signal)
    backtick_matches = _CODE_SIGNAL_PATTERNS["backticks"].findall(query)
    if backtick_matches:
        detected_patterns.append("backticks")
        signal_score += 0.4
        for m in backtick_matches:
            if len(m) > 1:
                extracted_symbols.add(m.strip())

    # Check CamelCase/PascalCase
    for name, pattern in [("PascalCase", _CODE_SIGNAL_PATTERNS["PascalCase"]),
                          ("camelCase", _CODE_SIGNAL_PATTERNS["camelCase"])]:
        matches = pattern.findall(query)
        if matches:
            detected_patterns.append(name)
            signal_score += 0.3
            for m in matches:
                if len(m) > 2 and m.lower() not in {"the", "and", "for", "with"}:
                    extracted_symbols.add(m)

    # Check snake_case
    snake_matches = _CODE_SIGNAL_PATTERNS["snake_case"].findall(query)
    if snake_matches:
        detected_patterns.append("snake_case")
        signal_score += 0.3
        for m in snake_matches:
            if len(m) > 3:
                extracted_symbols.add(m)

    # Check SCREAMING_SNAKE_CASE
    screaming_matches = _CODE_SIGNAL_PATTERNS["SCREAMING_SNAKE"].findall(query)
    if screaming_matches:
        detected_patterns.append("SCREAMING_SNAKE")
        signal_score += 0.2
        for m in screaming_matches:
            if len(m) > 3:
                extracted_symbols.add(m)

    # Check for function call syntax: name()
    paren_matches = _CODE_SIGNAL_PATTERNS["parentheses"].findall(query)
    if paren_matches:
        detected_patterns.append("parentheses")
        signal_score += 0.3
        for m in paren_matches:
            if len(m) > 1:
                extracted_symbols.add(m)

    # Check for module.path.syntax
    dot_matches = _CODE_SIGNAL_PATTERNS["dot_path"].findall(query)
    if dot_matches:
        detected_patterns.append("dot_path")
        signal_score += 0.2
        for m in dot_matches:
            parts = m.split(".")
            if parts:
                extracted_symbols.add(parts[-1])
                extracted_symbols.add(m)

    # Check for namespace::path syntax
    namespace_matches = _CODE_SIGNAL_PATTERNS["namespace_path"].findall(query)
    if namespace_matches:
        detected_patterns.append("namespace_path")
        signal_score += 0.3
        for m in namespace_matches:
            parts = m.split("::")
            if parts:
                extracted_symbols.add(parts[-1])
                if len(parts) >= 2:
                    extracted_symbols.add("::".join(parts[-2:]))
                extracted_symbols.add(m)

    # Check for generic types
    generic_matches = _CODE_SIGNAL_PATTERNS["generic_type"].findall(query)
    if generic_matches:
        detected_patterns.append("generic_type")
        signal_score += 0.25
        for m in generic_matches:
            base = m.split("<")[0]
            if base and len(base) > 1:
                extracted_symbols.add(base)

    # Check for file paths/extensions
    if _CODE_SIGNAL_PATTERNS["file_ext"].search(query):
        detected_patterns.append("file_ext")
        signal_score += 0.2

    if _CODE_SIGNAL_PATTERNS["path_like"].search(query):
        detected_patterns.append("path_like")
        signal_score += 0.15

    # Check for interface naming (IFoo)
    interface_matches = _CODE_SIGNAL_PATTERNS["interface_I"].findall(query)
    if interface_matches:
        detected_patterns.append("interface_I")
        signal_score += 0.3
        for m in interface_matches:
            if len(m) > 2:
                extracted_symbols.add(m)

    # Check for type names
    type_matches = _CODE_SIGNAL_PATTERNS["type_name"].findall(query)
    if type_matches:
        for m in type_matches:
            if m.lower() not in {"the", "and", "for", "with", "from", "this", "that", "have", "been", "were", "they"}:
                if len(m) > 2:
                    extracted_symbols.add(m)
        if type_matches and not detected_patterns:
            signal_score += 0.1

    # Check for code keywords
    words = set(query_lower.split())
    keyword_matches = words & _CODE_KEYWORDS
    if keyword_matches:
        detected_patterns.append("code_keywords")
        signal_score += 0.1 * min(len(keyword_matches), 3)

    # Embedding-based intent detection (blended with regex)
    embedding_score = 0.0
    if 0.1 <= signal_score <= 0.5:
        try:
            embedding_score = _detect_code_intent_embedding(query)
            if embedding_score > 0.6:
                detected_patterns.append("embedding_code_intent")
                blend_weight = 0.4 if signal_score < 0.3 else 0.25
                signal_score = signal_score * (1 - blend_weight) + embedding_score * blend_weight
        except Exception:
            pass
    elif str(os.environ.get("CODE_SIGNAL_EMBEDDING", "")).lower() in {"1", "true", "yes"} and signal_score < 0.1:
        try:
            embedding_score = _detect_code_intent_embedding(query)
            if embedding_score > 0.55:
                detected_patterns.append("embedding_code_intent")
                signal_score = embedding_score * 0.6
        except Exception:
            pass

    # Cap at 1.0
    signal_score = min(1.0, signal_score)

    # Build suggested boosts
    suggested_boosts = {}
    if signal_score >= 0.25:
        suggested_boosts["symbol_boost_multiplier"] = 1.0 + signal_score
        suggested_boosts["impl_boost_multiplier"] = 1.0 + (signal_score * 0.5)

    return {
        "has_code_signals": signal_score >= 0.2,
        "signal_strength": round(signal_score, 2),
        "embedding_score": round(embedding_score, 2) if embedding_score else None,
        "extracted_symbols": sorted(extracted_symbols)[:10],
        "detected_patterns": detected_patterns,
        "suggested_boosts": suggested_boosts,
    }

