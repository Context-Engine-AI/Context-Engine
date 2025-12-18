"""
Symbol-level diff using Longest Common Subsequence (LCS).

Detects moved/renamed symbols between file versions to reuse embeddings
instead of re-computing them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class SymbolInfo:
    """Represents a code symbol with its content hash."""
    name: str
    kind: str  # function, class, method, etc.
    content_hash: str  # Hash of symbol body
    start_line: int
    end_line: int


@dataclass
class SymbolChange:
    """Describes a change to a symbol."""
    old: Optional[SymbolInfo]
    new: Optional[SymbolInfo]
    change_type: str  # "added", "removed", "modified", "moved", "unchanged"


def lcs_length(seq1: list, seq2: list) -> int:
    """Compute length of longest common subsequence."""
    m, n = len(seq1), len(seq2)
    if m == 0 or n == 0:
        return 0
    
    # Space-optimized: only keep two rows
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, prev
    
    return prev[n]


def lcs_sequence(seq1: list, seq2: list) -> list:
    """Compute the actual longest common subsequence."""
    m, n = len(seq1), len(seq2)
    if m == 0 or n == 0:
        return []
    
    # Build DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
    # Backtrack to find sequence
    result = []
    i, j = m, n
    while i > 0 and j > 0:
        if seq1[i - 1] == seq2[j - 1]:
            result.append(seq1[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    
    return result[::-1]


def diff_symbols(
    old_symbols: list[SymbolInfo],
    new_symbols: list[SymbolInfo],
) -> list[SymbolChange]:
    """Diff two lists of symbols, detecting moves by content hash.
    
    Uses content_hash to identify symbols that moved without changing.
    """
    changes: list[SymbolChange] = []
    
    # Index by content hash for move detection
    old_by_hash: dict[str, SymbolInfo] = {s.content_hash: s for s in old_symbols}
    new_by_hash: dict[str, SymbolInfo] = {s.content_hash: s for s in new_symbols}
    
    # Index by name for modification detection
    old_by_name: dict[str, SymbolInfo] = {s.name: s for s in old_symbols}
    new_by_name: dict[str, SymbolInfo] = {s.name: s for s in new_symbols}
    
    matched_old: set[str] = set()
    matched_new: set[str] = set()
    
    # Pass 1: Find unchanged (same name and hash)
    for sym in new_symbols:
        old = old_by_name.get(sym.name)
        if old and old.content_hash == sym.content_hash:
            changes.append(SymbolChange(old, sym, "unchanged"))
            matched_old.add(old.content_hash)
            matched_new.add(sym.content_hash)
    
    # Pass 2: Find moved (same hash, different name or location)
    for sym in new_symbols:
        if sym.content_hash in matched_new:
            continue
        old = old_by_hash.get(sym.content_hash)
        if old and old.content_hash not in matched_old:
            changes.append(SymbolChange(old, sym, "moved"))
            matched_old.add(old.content_hash)
            matched_new.add(sym.content_hash)
    
    # Pass 3: Find modified (same name, different hash)
    for sym in new_symbols:
        if sym.content_hash in matched_new:
            continue
        old = old_by_name.get(sym.name)
        if old and old.content_hash not in matched_old:
            changes.append(SymbolChange(old, sym, "modified"))
            matched_old.add(old.content_hash)
            matched_new.add(sym.content_hash)
    
    # Pass 4: Remaining are added/removed
    for sym in new_symbols:
        if sym.content_hash not in matched_new:
            changes.append(SymbolChange(None, sym, "added"))
    
    for sym in old_symbols:
        if sym.content_hash not in matched_old:
            changes.append(SymbolChange(sym, None, "removed"))
    
    return changes


def symbols_needing_reembed(changes: list[SymbolChange]) -> list[SymbolInfo]:
    """Return symbols that need new embeddings (added or modified)."""
    result = []
    for c in changes:
        if c.change_type in ("added", "modified") and c.new:
            result.append(c.new)
    return result


def symbols_reusable(changes: list[SymbolChange]) -> list[tuple[SymbolInfo, SymbolInfo]]:
    """Return (old, new) pairs where embeddings can be reused (unchanged or moved)."""
    result = []
    for c in changes:
        if c.change_type in ("unchanged", "moved") and c.old and c.new:
            result.append((c.old, c.new))
    return result

