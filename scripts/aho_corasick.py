"""
Aho-Corasick automaton for multi-pattern string matching.

Finds all occurrences of multiple patterns in text in O(n + m + z) time,
where n = text length, m = total pattern length, z = number of matches.
Single pass instead of one pass per pattern.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterator, Optional


@dataclass(frozen=True)
class Match:
    """A pattern match result."""
    pattern: str
    start: int
    end: int

    __slots__ = ("pattern", "start", "end")

    @property
    def length(self) -> int:
        return self.end - self.start


class _ACNode:
    """Trie node for Aho-Corasick automaton."""
    
    __slots__ = ("children", "fail", "output", "depth")
    
    def __init__(self, depth: int = 0) -> None:
        self.children: dict[str, _ACNode] = {}
        self.fail: Optional[_ACNode] = None
        self.output: list[str] = []  # Patterns ending at this node
        self.depth = depth


class AhoCorasick:
    """Aho-Corasick automaton for efficient multi-pattern matching."""
    
    __slots__ = ("_root", "_patterns", "_case_sensitive", "_built")
    
    def __init__(self, patterns: list[str], case_sensitive: bool = False) -> None:
        """Build automaton from patterns.
        
        Args:
            patterns: List of strings to search for
            case_sensitive: Whether matching is case-sensitive (default: False)
        """
        self._root = _ACNode()
        self._patterns = patterns
        self._case_sensitive = case_sensitive
        self._built = False
        
        # Build trie
        for pattern in patterns:
            self._add_pattern(pattern)
        
        # Build failure links
        self._build_failure_links()
        self._built = True
    
    def _normalize(self, text: str) -> str:
        """Normalize text for matching."""
        return text if self._case_sensitive else text.lower()
    
    def _add_pattern(self, pattern: str) -> None:
        """Add a pattern to the trie."""
        if not pattern:
            return
        
        normalized = self._normalize(pattern)
        node = self._root
        
        for char in normalized:
            if char not in node.children:
                node.children[char] = _ACNode(node.depth + 1)
            node = node.children[char]
        
        # Store original pattern for output
        node.output.append(pattern)
    
    def _build_failure_links(self) -> None:
        """Build failure links using BFS."""
        queue: deque[_ACNode] = deque()
        
        # Initialize depth-1 nodes
        for child in self._root.children.values():
            child.fail = self._root
            queue.append(child)
        
        # BFS to build failure links
        while queue:
            current = queue.popleft()
            
            for char, child in current.children.items():
                queue.append(child)
                
                # Find failure link
                fail_node = current.fail
                while fail_node is not None and char not in fail_node.children:
                    fail_node = fail_node.fail
                
                child.fail = fail_node.children[char] if fail_node else self._root
                
                # Merge outputs from failure chain
                if child.fail and child.fail.output:
                    child.output = child.output + child.fail.output
    
    def find_all(self, text: str) -> list[Match]:
        """Find all pattern matches in text.
        
        Args:
            text: Text to search in
            
        Returns:
            List of Match objects with pattern, start, and end positions
        """
        if not text or not self._patterns:
            return []
        
        matches: list[Match] = []
        normalized = self._normalize(text)
        node = self._root
        
        for i, char in enumerate(normalized):
            # Follow failure links until we find a match or reach root
            while node is not None and char not in node.children:
                node = node.fail
            
            if node is None:
                node = self._root
                continue
            
            node = node.children[char]
            
            # Collect all outputs at this node
            for pattern in node.output:
                start = i - len(pattern) + 1
                matches.append(Match(pattern=pattern, start=start, end=i + 1))
        
        return matches
    
    def find_first(self, text: str) -> Optional[Match]:
        """Find first pattern match in text."""
        for match in self._iter_matches(text):
            return match
        return None

    def _iter_matches(self, text: str) -> Iterator[Match]:
        """Iterate matches lazily (for find_first optimization)."""
        if not text or not self._patterns:
            return

        normalized = self._normalize(text)
        node = self._root

        for i, char in enumerate(normalized):
            while node is not None and char not in node.children:
                node = node.fail

            if node is None:
                node = self._root
                continue

            node = node.children[char]

            for pattern in node.output:
                start = i - len(pattern) + 1
                yield Match(pattern=pattern, start=start, end=i + 1)

    def contains_any(self, text: str) -> bool:
        """Check if text contains any pattern (fast short-circuit)."""
        return self.find_first(text) is not None

    def count_matches(self, text: str) -> int:
        """Count total matches (including overlapping)."""
        return len(self.find_all(text))

    def count_unique_patterns(self, text: str) -> int:
        """Count unique patterns found (for scoring)."""
        return len({m.pattern for m in self.find_all(text)})

    @property
    def patterns(self) -> list[str]:
        """Return the patterns this automaton searches for."""
        return self._patterns.copy()

    def __len__(self) -> int:
        return len(self._patterns)


# Convenience factory for lexical search integration
def build_lexical_automaton(query_terms: list[str], case_sensitive: bool = False) -> AhoCorasick:
    """Build automaton from query terms for lexical scoring.

    This is the main integration point for hybrid_search.py lexical_score().

    Args:
        query_terms: Tokenized query terms (e.g., from tokenize_queries)
        case_sensitive: Whether to match case-sensitively

    Returns:
        AhoCorasick automaton ready for find_all() calls
    """
    # Filter empty terms
    terms = [t for t in query_terms if t and len(t) >= 2]
    return AhoCorasick(terms, case_sensitive=case_sensitive)


def lexical_match_score(
    automaton: AhoCorasick,
    text: str,
    *,
    unique_only: bool = True,
    position_weight: bool = False,
) -> float:
    """Score text based on pattern matches.

    Args:
        automaton: Pre-built AhoCorasick automaton
        text: Text to score
        unique_only: Count unique patterns only (default: True)
        position_weight: Boost matches near start of text (default: False)

    Returns:
        Match score (higher = more matches)
    """
    if not text:
        return 0.0

    matches = automaton.find_all(text)
    if not matches:
        return 0.0

    if unique_only:
        score = float(len({m.pattern for m in matches}))
    else:
        score = float(len(matches))

    if position_weight and matches:
        # Boost for matches in first 200 chars (likely symbol/signature)
        early_matches = sum(1 for m in matches if m.start < 200)
        score += early_matches * 0.5

    return score
