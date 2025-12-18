"""
HyperLogLog for cardinality estimation.

Estimates count of unique items using O(log log n) space.
Use for tracking unique files, chunks, or queries without storing all values.
"""

from __future__ import annotations

import hashlib
import struct
import math
from typing import Iterable


class HyperLogLog:
    """HyperLogLog cardinality estimator."""
    
    __slots__ = ("_p", "_m", "_registers", "_alpha")
    
    def __init__(self, precision: int = 14) -> None:
        """
        Initialize HyperLogLog with given precision.
        
        Args:
            precision: Number of bits for register addressing (4-18).
                      Higher = more accurate, more memory.
                      p=14 uses 16KB, error ~0.8%
        """
        if not 4 <= precision <= 18:
            raise ValueError("precision must be between 4 and 18")
        
        self._p = precision
        self._m = 1 << precision  # 2^p registers
        self._registers = bytearray(self._m)
        
        # Alpha correction factor
        if self._m == 16:
            self._alpha = 0.673
        elif self._m == 32:
            self._alpha = 0.697
        elif self._m == 64:
            self._alpha = 0.709
        else:
            self._alpha = 0.7213 / (1 + 1.079 / self._m)
    
    def _hash(self, item: str | bytes) -> int:
        """Hash item to 64-bit integer."""
        if isinstance(item, str):
            item = item.encode('utf-8')
        return struct.unpack('<Q', hashlib.md5(item).digest()[:8])[0]
    
    @staticmethod
    def _leading_zeros(value: int, max_bits: int = 64) -> int:
        """Count leading zeros after the register bits."""
        if value == 0:
            return max_bits
        count = 0
        mask = 1 << (max_bits - 1)
        while (value & mask) == 0 and count < max_bits:
            count += 1
            mask >>= 1
        return count
    
    def add(self, item: str | bytes) -> None:
        """Add an item to the HyperLogLog."""
        h = self._hash(item)
        
        # Use first p bits for register index
        idx = h & (self._m - 1)
        
        # Count leading zeros in remaining bits + 1
        remaining = h >> self._p
        rho = self._leading_zeros(remaining, 64 - self._p) + 1
        
        # Update register with max
        if rho > self._registers[idx]:
            self._registers[idx] = rho
    
    def add_batch(self, items: Iterable[str | bytes]) -> None:
        """Add multiple items efficiently."""
        for item in items:
            self.add(item)
    
    def count(self) -> int:
        """Estimate the cardinality (number of unique items)."""
        # Raw estimate using harmonic mean
        indicator = sum(2.0 ** (-r) for r in self._registers)
        estimate = self._alpha * self._m * self._m / indicator
        
        # Small range correction (linear counting)
        if estimate <= 2.5 * self._m:
            zeros = self._registers.count(0)
            if zeros > 0:
                estimate = self._m * math.log(self._m / zeros)
        
        # Large range correction (not needed for 64-bit hashes)
        
        return int(estimate + 0.5)
    
    def merge(self, other: "HyperLogLog") -> None:
        """Merge another HyperLogLog into this one (union)."""
        if self._p != other._p:
            raise ValueError("HyperLogLog precision must match for merge")
        
        for i in range(self._m):
            if other._registers[i] > self._registers[i]:
                self._registers[i] = other._registers[i]
    
    def copy(self) -> "HyperLogLog":
        """Create a copy of this HyperLogLog."""
        new = HyperLogLog.__new__(HyperLogLog)
        new._p = self._p
        new._m = self._m
        new._registers = bytearray(self._registers)
        new._alpha = self._alpha
        return new
    
    @property
    def precision(self) -> int:
        """Get the precision (p) value."""
        return self._p
    
    @property
    def memory_bytes(self) -> int:
        """Get memory usage in bytes."""
        return self._m
    
    def relative_error(self) -> float:
        """Get the expected relative error."""
        return 1.04 / math.sqrt(self._m)
    
    def clear(self) -> None:
        """Reset all registers to zero."""
        for i in range(self._m):
            self._registers[i] = 0

