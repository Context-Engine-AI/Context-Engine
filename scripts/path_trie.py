"""
Trie for O(k) path prefix lookups where k = prefix length.

Instead of iterating all paths to find matches for `under="src/components/"`,
traverse trie to the prefix node and collect all descendants.
"""

from __future__ import annotations

from typing import Iterator, Optional


class PathTrieNode:
    """Node in the path trie. Each node represents one path segment."""
    
    __slots__ = ("children", "is_terminal", "full_path")
    
    def __init__(self) -> None:
        self.children: dict[str, PathTrieNode] = {}
        self.is_terminal: bool = False
        self.full_path: Optional[str] = None


class PathTrie:
    """Trie structure for file paths. Splits on '/' separator."""
    
    __slots__ = ("_root", "_count")
    
    def __init__(self) -> None:
        self._root = PathTrieNode()
        self._count = 0
    
    def add(self, path: str) -> None:
        """Add a path to the trie."""
        if not path:
            return
        
        # Normalize: remove leading/trailing slashes
        path = path.strip("/")
        segments = path.split("/")
        
        node = self._root
        for seg in segments:
            if seg not in node.children:
                node.children[seg] = PathTrieNode()
            node = node.children[seg]
        
        if not node.is_terminal:
            node.is_terminal = True
            node.full_path = path
            self._count += 1
    
    def contains(self, path: str) -> bool:
        """Check if exact path exists."""
        node = self._find_node(path)
        return node is not None and node.is_terminal
    
    def _find_node(self, prefix: str) -> Optional[PathTrieNode]:
        """Find the node for a given prefix."""
        if not prefix:
            return self._root
        
        prefix = prefix.strip("/")
        segments = prefix.split("/")
        
        node = self._root
        for seg in segments:
            if seg not in node.children:
                return None
            node = node.children[seg]
        return node
    
    def find_by_prefix(self, prefix: str) -> list[str]:
        """Return all paths matching the prefix."""
        return list(self.iter_prefix(prefix))
    
    def iter_prefix(self, prefix: str) -> Iterator[str]:
        """Iterate all paths under prefix."""
        node = self._find_node(prefix)
        if node is None:
            return
        yield from self._collect_paths(node)
    
    def _collect_paths(self, node: PathTrieNode) -> Iterator[str]:
        """DFS to collect all terminal paths under a node."""
        if node.is_terminal and node.full_path:
            yield node.full_path
        for child in node.children.values():
            yield from self._collect_paths(child)
    
    def find_by_glob(self, pattern: str) -> list[str]:
        """Simple glob matching: supports * for single segment, ** for multiple.
        
        Examples:
            "src/*.py" - matches src/foo.py, not src/bar/foo.py
            "src/**/*.py" - matches src/foo.py, src/bar/foo.py, src/a/b/c.py
        """
        results: list[str] = []
        pattern = pattern.strip("/")
        segments = pattern.split("/")
        self._glob_match(self._root, segments, 0, results)
        return results
    
    def _glob_match(
        self,
        node: PathTrieNode,
        segments: list[str],
        idx: int,
        results: list[str],
    ) -> None:
        """Recursive glob matching."""
        if idx >= len(segments):
            if node.is_terminal and node.full_path:
                results.append(node.full_path)
            return
        
        seg = segments[idx]
        
        if seg == "**":
            # Match zero or more segments
            self._glob_match(node, segments, idx + 1, results)
            for child in node.children.values():
                self._glob_match(child, segments, idx, results)
        elif seg == "*":
            # Match exactly one segment
            for child in node.children.values():
                self._glob_match(child, segments, idx + 1, results)
        elif "*" in seg:
            # Wildcard within segment (e.g., "*.py")
            import fnmatch
            for name, child in node.children.items():
                if fnmatch.fnmatch(name, seg):
                    self._glob_match(child, segments, idx + 1, results)
        else:
            # Exact match
            if seg in node.children:
                self._glob_match(node.children[seg], segments, idx + 1, results)
    
    def __len__(self) -> int:
        return self._count
    
    def __contains__(self, path: str) -> bool:
        return self.contains(path)
    
    @classmethod
    def from_paths(cls, paths: list[str]) -> "PathTrie":
        """Build trie from list of paths."""
        trie = cls()
        for p in paths:
            trie.add(p)
        return trie

