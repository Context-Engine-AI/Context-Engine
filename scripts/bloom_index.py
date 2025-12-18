"""
Bloom Filter for file hash membership testing.

Probabilistic set with O(1) lookup. False positives possible, false negatives not.
Use to skip expensive Qdrant hash checks when file is definitely new.
"""

from __future__ import annotations

import hashlib
import math
import os
import pickle
import threading
from pathlib import Path
from typing import Optional

# Optimal parameters for 100K items with 0.1% false positive rate
DEFAULT_SIZE_BITS = 1_437_759  # ~175KB for 100K items at 0.1% FP
DEFAULT_HASH_COUNT = 10
MAX_ITEMS_ESTIMATE = 100_000


class BloomFilter:
    """Space-efficient probabilistic set for membership testing."""

    __slots__ = ("_bits", "_size", "_hash_count", "_count", "_lock")

    def __init__(
        self,
        size_bits: int = DEFAULT_SIZE_BITS,
        hash_count: int = DEFAULT_HASH_COUNT,
    ) -> None:
        self._size = size_bits
        self._hash_count = hash_count
        self._bits = bytearray((size_bits + 7) // 8)
        self._count = 0
        self._lock = threading.Lock()

    def __getstate__(self) -> dict:
        """Pickle support: exclude unpicklable lock."""
        return {
            "_bits": bytes(self._bits),
            "_size": self._size,
            "_hash_count": self._hash_count,
            "_count": self._count,
        }

    def __setstate__(self, state: dict) -> None:
        """Pickle support: restore from state and recreate lock."""
        self._bits = bytearray(state["_bits"])
        self._size = state["_size"]
        self._hash_count = state["_hash_count"]
        self._count = state["_count"]
        self._lock = threading.Lock()
    
    @classmethod
    def optimal_params(cls, n_items: int, fp_rate: float = 0.001) -> tuple[int, int]:
        """Calculate optimal size and hash count for given parameters.
        
        Args:
            n_items: Expected number of items
            fp_rate: Desired false positive rate (default 0.1%)
        
        Returns:
            (size_bits, hash_count)
        """
        if n_items <= 0:
            n_items = 1
        # m = -n * ln(p) / (ln(2)^2)
        m = int(-n_items * math.log(fp_rate) / (math.log(2) ** 2))
        # k = (m/n) * ln(2)
        k = max(1, int((m / n_items) * math.log(2)))
        return m, k
    
    def _get_hash_positions(self, item: str) -> list[int]:
        """Generate k hash positions using double hashing technique."""
        # Use SHA256 for first hash, MD5 for second (both fast in Python)
        h1 = int(hashlib.sha256(item.encode()).hexdigest()[:16], 16)
        h2 = int(hashlib.md5(item.encode()).hexdigest(), 16)
        
        positions = []
        for i in range(self._hash_count):
            # Double hashing: h(i) = (h1 + i*h2) mod m
            pos = (h1 + i * h2) % self._size
            positions.append(pos)
        return positions
    
    def add(self, item: str) -> None:
        """Add item to the filter."""
        positions = self._get_hash_positions(item)
        with self._lock:
            for pos in positions:
                byte_idx = pos // 8
                bit_idx = pos % 8
                self._bits[byte_idx] |= (1 << bit_idx)
            self._count += 1
    
    def might_contain(self, item: str) -> bool:
        """Test if item might be in the set.
        
        Returns:
            False = definitely not in set
            True = probably in set (with fp_rate probability of false positive)
        """
        positions = self._get_hash_positions(item)
        with self._lock:
            for pos in positions:
                byte_idx = pos // 8
                bit_idx = pos % 8
                if not (self._bits[byte_idx] & (1 << bit_idx)):
                    return False
        return True
    
    def __contains__(self, item: str) -> bool:
        return self.might_contain(item)
    
    @property
    def count(self) -> int:
        """Approximate number of items added."""
        return self._count
    
    @property
    def estimated_fp_rate(self) -> float:
        """Estimate current false positive rate based on bit saturation."""
        with self._lock:
            ones = sum(bin(b).count("1") for b in self._bits)
        saturation = ones / self._size
        return saturation ** self._hash_count
    
    def clear(self) -> None:
        """Reset the filter."""
        with self._lock:
            self._bits = bytearray((self._size + 7) // 8)
            self._count = 0


class BloomIndex:
    """Workspace-aware Bloom filter for file hash indexing."""

    FILENAME = "bloom_index.pkl"

    def __init__(self, workspace_path: str | Path, bloom: Optional[BloomFilter] = None) -> None:
        self.workspace_path = Path(workspace_path)
        self._bloom = bloom or BloomFilter()
        self._dirty = False

    @property
    def _state_dir(self) -> Path:
        return self.workspace_path / ".codebase"

    @property
    def _bloom_path(self) -> Path:
        return self._state_dir / self.FILENAME

    @classmethod
    def load_or_create(cls, workspace_path: str | Path) -> "BloomIndex":
        """Load existing bloom index or create new one."""
        idx = cls(workspace_path)
        if idx._bloom_path.exists():
            try:
                with open(idx._bloom_path, "rb") as f:
                    idx._bloom = pickle.load(f)
            except Exception:
                # Corrupted - start fresh
                idx._bloom = BloomFilter()
        return idx

    def add(self, file_hash: str) -> None:
        """Add file hash to the index."""
        if not file_hash:
            return
        self._bloom.add(file_hash)
        self._dirty = True

    def might_contain(self, file_hash: str) -> bool:
        """Check if file hash might be indexed.

        Returns:
            False = definitely not indexed (safe to skip Qdrant check)
            True = possibly indexed (need Qdrant check to confirm)
        """
        if not file_hash:
            return False
        return self._bloom.might_contain(file_hash)

    def __contains__(self, file_hash: str) -> bool:
        return self.might_contain(file_hash)

    def save(self) -> None:
        """Persist bloom filter to disk."""
        if not self._dirty:
            return
        self._state_dir.mkdir(parents=True, exist_ok=True)
        try:
            with open(self._bloom_path, "wb") as f:
                pickle.dump(self._bloom, f)
            self._dirty = False
        except Exception:
            pass  # Best effort

    def clear(self) -> None:
        """Reset the bloom filter."""
        self._bloom.clear()
        self._dirty = True

    @property
    def count(self) -> int:
        return self._bloom.count

    @property
    def estimated_fp_rate(self) -> float:
        return self._bloom.estimated_fp_rate

    def rebuild_from_cache(self, file_hashes: dict[str, str]) -> None:
        """Rebuild bloom filter from existing cache of file hashes.

        Args:
            file_hashes: Dict mapping file paths to their hashes
        """
        self._bloom.clear()
        for file_hash in file_hashes.values():
            if isinstance(file_hash, dict):
                file_hash = file_hash.get("hash", "")
            if file_hash:
                self._bloom.add(file_hash)
        self._dirty = True


# Module-level singleton for the current workspace
_current_bloom: Optional[BloomIndex] = None
_bloom_lock = threading.Lock()


def get_bloom_index(workspace_path: Optional[str] = None) -> BloomIndex:
    """Get or create bloom index for workspace (singleton per workspace)."""
    global _current_bloom

    if workspace_path is None:
        workspace_path = os.environ.get("WORKSPACE_PATH") or os.environ.get("WATCH_ROOT") or "/work"

    with _bloom_lock:
        if _current_bloom is None or str(_current_bloom.workspace_path) != workspace_path:
            _current_bloom = BloomIndex.load_or_create(workspace_path)
        return _current_bloom


def bloom_might_contain(file_hash: str, workspace_path: Optional[str] = None) -> bool:
    """Quick check if file hash might be in index."""
    return get_bloom_index(workspace_path).might_contain(file_hash)


def bloom_add(file_hash: str, workspace_path: Optional[str] = None) -> None:
    """Add file hash to bloom index."""
    get_bloom_index(workspace_path).add(file_hash)


def bloom_save(workspace_path: Optional[str] = None) -> None:
    """Persist bloom index to disk."""
    get_bloom_index(workspace_path).save()

