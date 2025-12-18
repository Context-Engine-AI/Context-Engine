"""
String similarity functions for fuzzy symbol matching.

Jaro-Winkler gives higher scores to strings with matching prefixes,
useful for matching getUserInfo to get_user_info.
"""

from __future__ import annotations


def jaro_similarity(s1: str, s2: str) -> float:
    """Jaro similarity between two strings. Returns 0.0 to 1.0."""
    if s1 == s2:
        return 1.0
    
    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 0.0
    
    # Match window
    match_dist = max(len1, len2) // 2 - 1
    if match_dist < 0:
        match_dist = 0
    
    s1_matches = [False] * len1
    s2_matches = [False] * len2
    matches = 0
    transpositions = 0
    
    # Find matches
    for i in range(len1):
        start = max(0, i - match_dist)
        end = min(i + match_dist + 1, len2)
        
        for j in range(start, end):
            if s2_matches[j] or s1[i] != s2[j]:
                continue
            s1_matches[i] = True
            s2_matches[j] = True
            matches += 1
            break
    
    if matches == 0:
        return 0.0
    
    # Count transpositions
    k = 0
    for i in range(len1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1
    
    jaro = (
        matches / len1 +
        matches / len2 +
        (matches - transpositions / 2) / matches
    ) / 3
    
    return jaro


def jaro_winkler(s1: str, s2: str, prefix_weight: float = 0.1) -> float:
    """Jaro-Winkler similarity. Boosts score for common prefixes.
    
    Args:
        s1, s2: Strings to compare
        prefix_weight: Weight for prefix bonus (default 0.1, max effective 0.25)
    
    Returns:
        Similarity score from 0.0 to 1.0
    """
    jaro = jaro_similarity(s1, s2)
    
    # Common prefix (up to 4 chars)
    prefix_len = 0
    for c1, c2 in zip(s1[:4], s2[:4]):
        if c1 == c2:
            prefix_len += 1
        else:
            break
    
    return jaro + prefix_len * prefix_weight * (1 - jaro)


def normalize_identifier(name: str) -> str:
    """Normalize identifier for comparison.
    
    Converts camelCase and snake_case to lowercase words.
    getUserInfo -> get user info
    get_user_info -> get user info
    """
    import re
    
    # Insert space before capitals in camelCase
    name = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    # Replace underscores/hyphens with spaces
    name = re.sub(r"[_\-]+", " ", name)
    # Lowercase and collapse whitespace
    return " ".join(name.lower().split())


def fuzzy_symbol_match(query: str, symbol: str, threshold: float = 0.8) -> float:
    """Match a query against a symbol name with fuzzy matching.
    
    Normalizes both strings before comparison.
    
    Args:
        query: Search query (e.g., "getUserInfo")
        symbol: Symbol name from code (e.g., "get_user_info")
        threshold: Minimum score to consider a match (default 0.8)
    
    Returns:
        Similarity score if >= threshold, else 0.0
    """
    q_norm = normalize_identifier(query)
    s_norm = normalize_identifier(symbol)
    
    score = jaro_winkler(q_norm, s_norm)
    return score if score >= threshold else 0.0


def best_fuzzy_matches(
    query: str,
    symbols: list[str],
    top_k: int = 5,
    threshold: float = 0.7,
) -> list[tuple[str, float]]:
    """Find best fuzzy matches for a query among symbols.
    
    Args:
        query: Search query
        symbols: List of symbol names to search
        top_k: Number of results to return
        threshold: Minimum similarity score
    
    Returns:
        List of (symbol, score) tuples, sorted by score descending
    """
    scored = []
    q_norm = normalize_identifier(query)
    
    for sym in symbols:
        s_norm = normalize_identifier(sym)
        score = jaro_winkler(q_norm, s_norm)
        if score >= threshold:
            scored.append((sym, score))
    
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]

