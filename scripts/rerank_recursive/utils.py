"""
Utility functions for the recursive reranker.

Contains:
- Filename boost computation
- Identifier tokenization
- Embedding cache management
"""
import re
import hashlib
import threading
from typing import Any, Dict, List, Optional, Set

import numpy as np


# Very common tokens that appear everywhere - reduce their weight
_COMMON_TOKENS = frozenset({
    "index", "main", "app", "utils", "util", "helper", "helpers", "common",
    "base", "core", "lib", "src", "test", "tests", "spec", "specs",
    "internal", "public", "private", "static", "default", "new", "old",
    "data", "type", "types", "model", "models", "view", "views",
    "the", "and", "for", "with", "from", "that", "this", "have", "are",
})


def _split_identifier(s: str) -> List[str]:
    """Split any identifier into tokens, handling all common conventions.

    Handles: snake_case, kebab-case, camelCase, PascalCase, SCREAMING_CASE,
    numbers, acronyms (XMLParser -> xml, parser), dot.notation, and mixed styles.

    Special handling:
    - Preserves meaningful acronyms (API, HTTP, JSON, XML, URL, etc.)
    - Strips common prefixes (I for interface, _ for private)
    - Handles version suffixes (v2, 2.0)
    """
    if not s:
        return []

    # Strip common prefixes that don't add meaning
    if len(s) > 1:
        # Interface prefix (IUserService -> UserService)
        if s[0] == 'I' and s[1].isupper():
            s = s[1:]
        # Private prefix (_private -> private)
        elif s[0] == '_':
            s = s.lstrip('_')
        # Dollar prefix ($scope -> scope)
        elif s[0] == '$':
            s = s[1:]

    # Insert space before uppercase letters that follow lowercase (camelCase)
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)
    # Insert space before uppercase letters followed by lowercase (acronyms: XMLParser -> XML Parser)
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", s)
    # Insert space around digit sequences (handler2 -> handler 2, v2 -> v 2)
    s = re.sub(r"([a-zA-Z])(\d)", r"\1 \2", s)
    s = re.sub(r"(\d)([a-zA-Z])", r"\1 \2", s)

    # Split on separators: underscore, hyphen, dot, space
    parts = re.split(r"[_\-.\s]+", s)
    tokens = []
    for part in parts:
        part = part.strip().lower()
        # Skip pure numbers and single chars (except meaningful ones)
        if not part:
            continue
        if part.isdigit():
            continue  # Skip version numbers like "2", "18"
        if len(part) < 2:
            continue
        tokens.append(part)

    return tokens


def _normalize_token(tok: str) -> Set[str]:
    """Return the token plus simple morphological variants."""
    forms = {tok}
    # Simple plural/singular normalization
    if tok.endswith('s') and len(tok) > 3:
        forms.add(tok[:-1])  # services -> service
    elif tok.endswith('es') and len(tok) > 4:
        forms.add(tok[:-2])  # processes -> process
    elif tok.endswith('ies') and len(tok) > 4:
        forms.add(tok[:-3] + 'y')  # utilities -> utility
    # Add singular -> plural
    if not tok.endswith('s') and len(tok) > 2:
        forms.add(tok + 's')
    return forms


def _tokenize_for_fname_boost(text: Any) -> Set[str]:
    """Robust tokenization for filename boosts.

    Some MCP/IDE clients pass query strings that include quotes/brackets
    or list-like wrappers. Regex tokenization is resilient to that.
    """
    if not text:
        return set()
    try:
        s = str(text)
    except Exception:
        return set()

    # Split on any non-alphanumeric
    raw_parts = re.split(r"[^a-zA-Z0-9]+", s)
    tokens = set()
    for part in raw_parts:
        for tok in _split_identifier(part):
            if len(tok) >= 3:  # Query tokens need 3+ chars
                tokens.add(tok)
    return tokens


def _candidate_path_for_fname_boost(candidate: Dict[str, Any]) -> str:
    """Best-effort extraction of a path/filename from candidate objects."""
    for key in ("path", "rel_path", "host_path", "container_path", "client_path"):
        try:
            val = candidate.get(key)
        except Exception:
            val = None
        if isinstance(val, str) and val.strip():
            return val

    try:
        md = candidate.get("metadata") or {}
        if isinstance(md, dict):
            for key in ("path", "rel_path", "host_path", "container_path", "client_path"):
                val = md.get(key)
                if isinstance(val, str) and val.strip():
                    return val
    except Exception:
        pass

    return ""


def _compute_fname_boost(query: Any, candidate: Dict[str, Any], factor: float) -> float:
    """Compute filename/query correlation boost for a candidate.

    Production-grade matching for real-world codebases at scale:

    **Naming convention support:**
    - snake_case, camelCase, PascalCase, kebab-case, SCREAMING_CASE
    - Dot notation (com.company.auth.service)
    - Mixed styles (legacy codebases)

    **Smart tokenization:**
    - Acronyms: XMLParser -> xml, parser; HTTPClient -> http, client
    - Prefixes stripped: IService -> service, _private -> private
    - Numbers separated: handler2 -> handler, React18 -> react

    **Normalization:**
    - Simple plural/singular normalization (services <-> service)

    **Position-aware scoring:**
    - Filename matches weighted higher than directory matches
    - Deeper directories weighted less (noise reduction)

    **Specificity weighting:**
    - Common tokens (index, main, utils) weighted less
    - Rare/specific tokens weighted more

    **Scoring tiers:**
    - Exact match: 1.0 × factor
    - Normalized match (morphology): 0.8 × factor
    - Substring containment: 0.4 × factor
    - Common token penalty: 0.5× multiplier
    - Filename bonus: 1.5× multiplier for filename matches

    Requires 2+ quality matches to trigger (prevents noise).
    """
    if not factor or factor <= 0:
        return 0.0

    query_tokens = _tokenize_for_fname_boost(query)
    if not query_tokens:
        return 0.0

    path = _candidate_path_for_fname_boost(candidate)
    path = str(path or "")
    if not path:
        return 0.0

    # Strip common prefixes that add noise (preserve case for splitting)
    path_clean = path
    path_lower = path.lower()
    for prefix in ("/work/", "/app/", "/src/", "/home/", "/var/", "/opt/", "/usr/"):
        if path_lower.startswith(prefix):
            path_clean = path[len(prefix):]
            break

    # Split path into segments, track position for weighting
    path_segments = re.split(r"[/\\]", path_clean)
    path_segments = [s for s in path_segments if s]  # Remove empty

    if not path_segments:
        return 0.0

    # Tokenize with position info: (token, is_filename, depth)
    # Filename = last segment, depth = 0 for filename, 1 for parent, etc.
    path_token_info: Dict[str, Dict[str, Any]] = {}  # token -> {is_filename, min_depth}

    for i, segment in enumerate(reversed(path_segments)):
        is_filename = (i == 0)
        depth = i

        # Strip extension from filename
        if is_filename and "." in segment:
            segment = segment.rsplit(".", 1)[0]

        for tok in _split_identifier(segment):
            if len(tok) >= 2:
                if tok not in path_token_info:
                    path_token_info[tok] = {"is_filename": is_filename, "depth": depth}
                # Keep the most important occurrence (filename > dir, shallow > deep)
                elif is_filename and not path_token_info[tok]["is_filename"]:
                    path_token_info[tok] = {"is_filename": True, "depth": depth}

    if not path_token_info:
        return 0.0

    path_tokens = set(path_token_info.keys())

    # Build normalized lookup for path tokens
    path_normalized: Dict[str, str] = {}  # normalized_form -> original_token
    for ptok in path_tokens:
        for form in _normalize_token(ptok):
            if form not in path_normalized:
                path_normalized[form] = ptok

    # Score matches with quality tiers
    score = 0.0
    matched_query_tokens = set()

    for qtok in query_tokens:
        qtok_forms = _normalize_token(qtok)
        match_score = 0.0
        matched_ptok = None

        # Tier 1: Exact match
        if qtok in path_tokens:
            match_score = 1.0
            matched_ptok = qtok
        else:
            # Tier 2: Normalized match (plural/singular)
            for qform in qtok_forms:
                if qform in path_normalized:
                    match_score = 0.8
                    matched_ptok = path_normalized[qform]
                    break

            # Tier 3: Substring containment (if no normalized match)
            if match_score == 0.0:
                for ptok in path_tokens:
                    if len(qtok) >= 4 and len(ptok) >= 4:
                        if qtok in ptok or ptok in qtok:
                            overlap = min(len(qtok), len(ptok))
                            if overlap >= 4:
                                match_score = 0.4
                                matched_ptok = ptok
                                break

        if match_score > 0 and matched_ptok:
            matched_query_tokens.add(qtok)

            # Apply position bonus (filename matches worth more)
            info = path_token_info.get(matched_ptok, {})
            if info.get("is_filename"):
                match_score *= 1.5  # 50% bonus for filename match
            else:
                # Depth penalty for deep directories
                depth = info.get("depth", 0)
                if depth > 2:
                    match_score *= 0.8  # Slight penalty for deep paths

            # Common token penalty
            if qtok in _COMMON_TOKENS or matched_ptok in _COMMON_TOKENS:
                match_score *= 0.5

            score += match_score

    # Require 2+ quality matches to trigger (prevents noise from single common word)
    if len(matched_query_tokens) < 2:
        return 0.0

    return float(score * factor)


# ---------------------------------------------------------------------------
# Embedding Cache
# ---------------------------------------------------------------------------

# Global embedding cache for efficiency
# Key is sha256 hex digest (deterministic, collision-resistant)
_EMBEDDING_CACHE: Dict[str, np.ndarray] = {}
_EMBEDDING_CACHE_MAX_SIZE = 10000
_EMBEDDING_CACHE_LOCK = threading.Lock()


def _cache_key(text: str) -> str:
    """Generate deterministic cache key from text (process-stable, collision-resistant)."""
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _get_cached_embedding(text: str) -> Optional[np.ndarray]:
    """Get embedding from cache if exists."""
    key = _cache_key(text)
    with _EMBEDDING_CACHE_LOCK:
        return _EMBEDDING_CACHE.get(key)


def _cache_embedding(text: str, embedding: np.ndarray):
    """Cache embedding for text."""
    key = _cache_key(text)
    with _EMBEDDING_CACHE_LOCK:
        if len(_EMBEDDING_CACHE) >= _EMBEDDING_CACHE_MAX_SIZE:
            # Evict oldest 10%
            keys_to_remove = list(_EMBEDDING_CACHE.keys())[:_EMBEDDING_CACHE_MAX_SIZE // 10]
            for k in keys_to_remove:
                del _EMBEDDING_CACHE[k]
        _EMBEDDING_CACHE[key] = embedding
