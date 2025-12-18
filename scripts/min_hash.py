"""
MinHash for approximate Jaccard similarity (near-duplicate detection).

Use to quickly identify similar code chunks without computing full pairwise comparisons.
LSH (Locality Sensitive Hashing) bands enable sub-linear similarity search.
"""

from __future__ import annotations

import hashlib
import struct
from typing import Iterable, Iterator, Optional

# Large prime for hash functions
_PRIME = (1 << 61) - 1
_MAX_HASH = (1 << 32) - 1


class MinHash:
    """MinHash signature for approximate set similarity."""
    
    __slots__ = ("_num_perm", "_signature", "_a", "_b")
    
    def __init__(self, num_perm: int = 128, seed: int = 42) -> None:
        """
        Initialize MinHash with given number of permutations.
        
        Args:
            num_perm: Number of hash functions (higher = more accurate, more memory)
            seed: Random seed for reproducible hash functions
        """
        self._num_perm = num_perm
        self._signature = [_MAX_HASH] * num_perm
        
        # Generate hash function parameters using seed
        import random
        rng = random.Random(seed)
        self._a = tuple(rng.randint(1, _PRIME - 1) for _ in range(num_perm))
        self._b = tuple(rng.randint(0, _PRIME - 1) for _ in range(num_perm))
    
    def _hash_value(self, value: int, idx: int) -> int:
        """Apply idx-th hash function to value."""
        return ((self._a[idx] * value + self._b[idx]) % _PRIME) & _MAX_HASH
    
    def update(self, item: str | bytes) -> None:
        """Add an item (shingle) to the MinHash signature."""
        if isinstance(item, str):
            item = item.encode('utf-8')
        h = struct.unpack('<I', hashlib.md5(item).digest()[:4])[0]
        
        for i in range(self._num_perm):
            hv = self._hash_value(h, i)
            if hv < self._signature[i]:
                self._signature[i] = hv
    
    def update_batch(self, items: Iterable[str | bytes]) -> None:
        """Add multiple items efficiently."""
        for item in items:
            self.update(item)
    
    @property
    def signature(self) -> tuple[int, ...]:
        """Get the MinHash signature as a tuple."""
        return tuple(self._signature)
    
    def jaccard(self, other: "MinHash") -> float:
        """
        Estimate Jaccard similarity with another MinHash.
        
        Returns value in [0, 1] where 1 means identical sets.
        """
        if self._num_perm != other._num_perm:
            raise ValueError("MinHash signatures must have same num_perm")
        
        matches = sum(1 for a, b in zip(self._signature, other._signature) if a == b)
        return matches / self._num_perm
    
    def merge(self, other: "MinHash") -> None:
        """Merge another MinHash into this one (union operation)."""
        if self._num_perm != other._num_perm:
            raise ValueError("MinHash signatures must have same num_perm")
        
        for i in range(self._num_perm):
            if other._signature[i] < self._signature[i]:
                self._signature[i] = other._signature[i]
    
    def copy(self) -> "MinHash":
        """Create a copy of this MinHash."""
        new = MinHash.__new__(MinHash)
        new._num_perm = self._num_perm
        new._signature = self._signature.copy()
        new._a = self._a
        new._b = self._b
        return new


class MinHashLSH:
    """Locality Sensitive Hashing index for MinHash signatures."""
    
    __slots__ = ("_num_bands", "_rows_per_band", "_buckets", "_signatures")
    
    def __init__(self, num_perm: int = 128, threshold: float = 0.5) -> None:
        """
        Initialize LSH index.
        
        Args:
            num_perm: Number of permutations in MinHash signatures
            threshold: Similarity threshold for candidate pairs
        """
        # Calculate optimal bands/rows for threshold
        # P(candidate) ≈ 1 - (1 - s^r)^b where s=similarity, r=rows, b=bands
        self._rows_per_band, self._num_bands = self._optimal_params(num_perm, threshold)
        self._buckets: list[dict[tuple, list[str]]] = [{} for _ in range(self._num_bands)]
        self._signatures: dict[str, tuple[int, ...]] = {}
    
    @staticmethod
    def _optimal_params(num_perm: int, threshold: float) -> tuple[int, int]:
        """Find optimal (rows, bands) for given threshold."""
        best = (1, num_perm)
        min_err = float('inf')
        
        for b in range(1, num_perm + 1):
            if num_perm % b != 0:
                continue
            r = num_perm // b
            # Probability of becoming candidate at threshold
            fp = (1 - threshold ** r) ** b
            fn = 1 - (1 - (1 - threshold) ** r) ** b
            err = fp + fn
            if err < min_err:
                min_err = err
                best = (r, b)
        
        return best
    
    def insert(self, key: str, minhash: MinHash) -> None:
        """Insert a MinHash signature into the index."""
        sig = minhash.signature
        self._signatures[key] = sig
        
        for band_idx in range(self._num_bands):
            start = band_idx * self._rows_per_band
            end = start + self._rows_per_band
            band_hash = sig[start:end]
            
            if band_hash not in self._buckets[band_idx]:
                self._buckets[band_idx][band_hash] = []
            self._buckets[band_idx][band_hash].append(key)
    
    def query(self, minhash: MinHash) -> Iterator[str]:
        """Find candidate similar items (may have false positives)."""
        sig = minhash.signature
        seen: set[str] = set()
        
        for band_idx in range(self._num_bands):
            start = band_idx * self._rows_per_band
            end = start + self._rows_per_band
            band_hash = sig[start:end]
            
            for key in self._buckets[band_idx].get(band_hash, []):
                if key not in seen:
                    seen.add(key)
                    yield key

