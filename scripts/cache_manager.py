#!/usr/bin/env python3
"""
Unified caching system for Context-Engine.

This module provides a centralized, configurable caching system with multiple
eviction policies and statistics tracking to replace scattered caching mechanisms.
"""

import os
import time
import hashlib
import json
import threading
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from collections import OrderedDict
from enum import Enum
import logging

logger = logging.getLogger("cache_manager")


class EvictionPolicy(Enum):
    """Cache eviction policies."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live
    FIFO = "fifo"  # First In First Out


class CacheEntry:
    """Individual cache entry with metadata."""
    
    def __init__(self, value: Any, ttl: Optional[float] = None):
        self.value = value
        self.created_at = time.time()
        self.last_accessed = self.created_at
        self.access_count = 1
        self.ttl = ttl  # TTL in seconds
        self.size = self._calculate_size(value)
    
    def _calculate_size(self, value: Any) -> int:
        """Estimate memory size of cached value."""
        try:
            if isinstance(value, str):
                return len(value.encode('utf-8'))
            elif isinstance(value, (list, tuple)):
                return sum(self._calculate_size(item) for item in value)
            elif isinstance(value, dict):
                return sum(self._calculate_size(k) + self._calculate_size(v) for k, v in value.items())
            else:
                return len(str(value).encode('utf-8'))
        except Exception:
            return 100  # Default size estimation
    
    def is_expired(self) -> bool:
        """Check if entry has expired based on TTL."""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl
    
    def access(self) -> Any:
        """Record access and return value."""
        self.last_accessed = time.time()
        self.access_count += 1
        return self.value


class UnifiedCache:
    """
    Unified cache system with configurable eviction policies and statistics.
    
    Features:
    - Multiple eviction policies (LRU, LFU, TTL, FIFO)
    - Thread-safe operations
    - Statistics tracking
    - Configurable size limits
    - Automatic cleanup of expired entries
    """
    
    def __init__(
        self,
        name: str,
        max_size: int = 1000,
        max_memory_mb: int = 100,
        eviction_policy: EvictionPolicy = EvictionPolicy.LRU,
        default_ttl: Optional[float] = None,
        cleanup_interval: float = 60.0
    ):
        self.name = name
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.eviction_policy = eviction_policy
        self.default_ttl = default_ttl
        self.cleanup_interval = cleanup_interval
        
        # Storage
        self._cache: Dict[str, CacheEntry] = {}
        self._access_order = OrderedDict()  # For LRU/FIFO
        self._frequency = {}  # For LFU
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Statistics
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'expirations': 0,
            'current_size': 0,
            'current_memory_bytes': 0,
            'total_requests': 0
        }
        
        # Start cleanup thread
        self._cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        self._cleanup_thread.start()
        
        logger.debug(f"Initialized cache '{name}' with policy {eviction_policy.value}, "
                    f"max_size={max_size}, max_memory={max_memory_mb}MB")
    
    def _generate_key(self, key: Union[str, Tuple, List, Dict]) -> str:
        """Generate consistent cache key from various input types."""
        try:
            if isinstance(key, str):
                return key
            elif isinstance(key, (tuple, list)):
                key_str = json.dumps(list(key), sort_keys=True)
            elif isinstance(key, dict):
                key_str = json.dumps(key, sort_keys=True)
            else:
                key_str = str(key)
            
            # Hash long keys to avoid memory issues
            if len(key_str) > 200:
                return hashlib.md5(key_str.encode()).hexdigest()
            return key_str
        except Exception:
            return str(key)
    
    def _should_evict(self) -> bool:
        """Check if cache should trigger eviction."""
        return (
            len(self._cache) >= self.max_size or
            self._stats['current_memory_bytes'] >= self.max_memory_bytes
        )
    
    def _select_victim(self) -> Optional[str]:
        """Select entry for eviction based on policy."""
        if not self._cache:
            return None
        
        if self.eviction_policy == EvictionPolicy.LRU:
            # Least Recently Used
            return next(iter(self._access_order))
        
        elif self.eviction_policy == EvictionPolicy.FIFO:
            # First In First Out
            return next(iter(self._access_order))
        
        elif self.eviction_policy == EvictionPolicy.LFU:
            # Least Frequently Used
            return min(self._frequency.keys(), key=lambda k: self._frequency[k])
        
        elif self.eviction_policy == EvictionPolicy.TTL:
            # Time To Live (expired entries first)
            for key, entry in self._cache.items():
                if entry.is_expired():
                    return key
            # Fallback to LRU if no expired entries
            return next(iter(self._access_order))
        
        return None
    
    def _evict_entry(self, key: str) -> None:
        """Evict a specific entry from cache."""
        if key not in self._cache:
            return
        
        entry = self._cache[key]
        
        # Update statistics
        self._stats['evictions'] += 1
        self._stats['current_size'] -= 1
        self._stats['current_memory_bytes'] -= entry.size
        
        # Remove from all data structures
        del self._cache[key]
        self._access_order.pop(key, None)
        self._frequency.pop(key, None)
        
        logger.debug(f"Evicted cache entry '{key}' from cache '{self.name}'")
    
    def _cleanup_expired(self) -> int:
        """Clean up expired entries and return count cleaned."""
        if self.eviction_policy != EvictionPolicy.TTL and self.default_ttl is None:
            return 0
        
        expired_keys = []
        current_time = time.time()
        
        for key, entry in self._cache.items():
            if entry.is_expired():
                expired_keys.append(key)
        
        for key in expired_keys:
            entry = self._cache[key]
            del self._cache[key]
            self._access_order.pop(key, None)
            self._frequency.pop(key, None)
            
            # Update statistics
            self._stats['expirations'] += 1
            self._stats['current_size'] -= 1
            self._stats['current_memory_bytes'] -= entry.size
        
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired entries from cache '{self.name}'")
        
        return len(expired_keys)
    
    def _cleanup_worker(self) -> None:
        """Background worker for periodic cleanup."""
        while True:
            try:
                time.sleep(self.cleanup_interval)
                with self._lock:
                    self._cleanup_expired()
            except Exception as e:
                logger.error(f"Cache cleanup error in '{self.name}': {e}")
    
    def get(self, key: Union[str, Tuple, List, Dict]) -> Optional[Any]:
        """Get value from cache."""
        cache_key = self._generate_key(key)
        
        with self._lock:
            self._stats['total_requests'] += 1
            
            if cache_key not in self._cache:
                self._stats['misses'] += 1
                return None
            
            entry = self._cache[cache_key]
            
            # Check expiration
            if entry.is_expired():
                del self._cache[cache_key]
                self._access_order.pop(cache_key, None)
                self._frequency.pop(cache_key, None)
                
                self._stats['misses'] += 1
                self._stats['expirations'] += 1
                self._stats['current_size'] -= 1
                self._stats['current_memory_bytes'] -= entry.size
                return None
            
            # Update access tracking
            value = entry.access()
            
            # Update access order for LRU/FIFO
            if cache_key in self._access_order:
                self._access_order.move_to_end(cache_key)
            
            # Update frequency for LFU
            self._frequency[cache_key] = self._frequency.get(cache_key, 0) + 1
            
            self._stats['hits'] += 1
            
            logger.debug(f"Cache hit for key '{cache_key}' in cache '{self.name}'")
            return value
    
    def set(
        self,
        key: Union[str, Tuple, List, Dict],
        value: Any,
        ttl: Optional[float] = None
    ) -> bool:
        """Set value in cache."""
        cache_key = self._generate_key(key)
        ttl = ttl or self.default_ttl

        with self._lock:
            # Track previous entry so we can restore on failure
            old_entry = self._cache.get(cache_key)
            prev_freq = self._frequency.get(cache_key, 0)

            entry = CacheEntry(value, ttl)

            if old_entry is not None:
                # Remove old entry bookkeeping while we attempt to insert the replacement
                self._cache.pop(cache_key, None)
                self._access_order.pop(cache_key, None)
                self._frequency.pop(cache_key, None)
                self._stats['current_size'] -= 1
                self._stats['current_memory_bytes'] -= old_entry.size

            def _needs_evict() -> bool:
                if self.max_size > 0 and (len(self._cache) + 1) > self.max_size:
                    return True
                if self.max_memory_bytes > 0 and (self._stats['current_memory_bytes'] + entry.size) > self.max_memory_bytes:
                    return len(self._cache) > 0
                return False

            # Evict least valuable entries until the new item fits or we run out of victims
            while _needs_evict():
                victim_key = self._select_victim()
                if not victim_key:
                    break
                self._evict_entry(victim_key)

            # If the new entry still does not fit, restore prior state (for updates) and abort
            if (self.max_size > 0 and (len(self._cache) + 1) > self.max_size) or (
                self.max_memory_bytes > 0 and (self._stats['current_memory_bytes'] + entry.size) > self.max_memory_bytes
            ):
                if old_entry is not None:
                    self._cache[cache_key] = old_entry
                    self._access_order[cache_key] = True
                    self._frequency[cache_key] = max(1, prev_freq)
                    self._stats['current_size'] += 1
                    self._stats['current_memory_bytes'] += old_entry.size
                logger.debug(
                    f"Entry '{cache_key}' exceeds cache constraints; not retained in '{self.name}'"
                )
                return False

            # Insert new entry
            self._cache[cache_key] = entry
            self._access_order[cache_key] = True
            self._frequency[cache_key] = (prev_freq + 1) if prev_freq else 1
            self._stats['current_size'] += 1
            self._stats['current_memory_bytes'] += entry.size

            logger.debug(f"Cache set for key '{cache_key}' in cache '{self.name}'")
            return True

    def delete(self, key: Union[str, Tuple, List, Dict]) -> bool:
        """Delete value from cache."""
        cache_key = self._generate_key(key)
        
        with self._lock:
            if cache_key not in self._cache:
                return False
            
            entry = self._cache[cache_key]
            del self._cache[cache_key]
            self._access_order.pop(cache_key, None)
            self._frequency.pop(cache_key, None)
            
            # Update statistics
            self._stats['current_size'] -= 1
            self._stats['current_memory_bytes'] -= entry.size
            
            logger.debug(f"Cache delete for key '{cache_key}' in cache '{self.name}'")
            return True
    
    def clear(self) -> None:
        """Clear all entries from cache."""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
            self._frequency.clear()
            
            # Reset statistics
            self._stats['current_size'] = 0
            self._stats['current_memory_bytes'] = 0
            
            logger.debug(f"Cleared cache '{self.name}'")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            stats = self._stats.copy()
            stats['hit_rate'] = (
                stats['hits'] / max(1, stats['total_requests']) * 100
            )
            stats['memory_usage_mb'] = stats['current_memory_bytes'] / (1024 * 1024)
            stats['cache_name'] = self.name
            stats['eviction_policy'] = self.eviction_policy.value
            return stats
    
    def get_keys(self) -> List[str]:
        """Get all cache keys."""
        with self._lock:
            return list(self._cache.keys())
    
    def size(self) -> int:
        """Get current cache size."""
        with self._lock:
            return len(self._cache)
    
    def __len__(self) -> int:
        """Get current cache size."""
        return self.size()
    
    def __contains__(self, key: Union[str, Tuple, List, Dict]) -> bool:
        """Check if key exists in cache."""
        cache_key = self._generate_key(key)
        with self._lock:
            return cache_key in self._cache
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        try:
            self.clear()
        except Exception:
            pass


# Global cache registry
_cache_registry: Dict[str, UnifiedCache] = {}
_registry_lock = threading.Lock()


def get_cache(
    name: str,
    max_size: Optional[int] = None,
    max_memory_mb: Optional[int] = None,
    eviction_policy: Optional[EvictionPolicy] = None,
    default_ttl: Optional[float] = None
) -> UnifiedCache:
    """Get or create a cache instance with specified configuration."""
    with _registry_lock:
        if name not in _cache_registry:
            # Get configuration from environment or defaults
            env_prefix = f"CACHE_{name.upper()}_"
            
            max_size = max_size or int(os.environ.get(f"{env_prefix}MAX_SIZE", "1000"))
            max_memory_mb = max_memory_mb or int(os.environ.get(f"{env_prefix}MAX_MEMORY_MB", "100"))
            
            policy_str = os.environ.get(f"{env_prefix}EVICT_POLICY", "lru").lower()
            eviction_policy = eviction_policy or EvictionPolicy(policy_str)
            
            default_ttl = default_ttl or (
                float(os.environ.get(f"{env_prefix}DEFAULT_TTL")) 
                if os.environ.get(f"{env_prefix}DEFAULT_TTL") else None
            )
            
            cleanup_interval = float(os.environ.get(f"{env_prefix}CLEANUP_INTERVAL", "60"))
            
            _cache_registry[name] = UnifiedCache(
                name=name,
                max_size=max_size,
                max_memory_mb=max_memory_mb,
                eviction_policy=eviction_policy,
                default_ttl=default_ttl,
                cleanup_interval=cleanup_interval
            )
        
        return _cache_registry[name]


def clear_all_caches() -> None:
    """Clear all registered caches."""
    with _registry_lock:
        for cache in _cache_registry.values():
            try:
                cache.clear()
            except Exception as e:
                logger.error(f"Error clearing cache '{cache.name}': {e}")


def get_all_cache_stats() -> Dict[str, Dict[str, Any]]:
    """Get statistics for all registered caches.
    Ensures predefined caches are present so callers can rely on standard keys.
    """
    # Ensure predefined caches are instantiated OUTSIDE the registry lock to avoid deadlocks
    try:
        get_embedding_cache()
        get_search_cache()
        get_expansion_cache()
    except Exception:
        pass

    with _registry_lock:
        return {name: cache.get_stats() for name, cache in _cache_registry.items()}


# Decorator for memoization
def cached(
    cache_name: str = "default",
    ttl: Optional[float] = None,
    key_func: Optional[Callable] = None
):
    """Decorator to cache function results."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = (args, tuple(sorted(kwargs.items())))
            
            # Get cache
            cache = get_cache(cache_name)
            
            # Try to get from cache
            result = cache.get(cache_key)
            if result is not None:
                return result
            
            # Compute and cache result
            result = func(*args, **kwargs)
            cache.set(cache_key, result, ttl=ttl)
            
            return result
        
        wrapper._cached = True
        wrapper._cache_name = cache_name
        return wrapper
    
    return decorator


# Predefined cache configurations
def get_embedding_cache() -> UnifiedCache:
    """Get cache for query embeddings."""
    return get_cache(
        "embeddings",
        max_size=int(os.environ.get("EMBED_CACHE_MAX_SIZE", "8192")),
        max_memory_mb=int(os.environ.get("EMBED_CACHE_MAX_MEMORY_MB", "50")),
        eviction_policy=EvictionPolicy.LRU,
        default_ttl=None  # Embeddings don't expire
    )


def get_search_cache() -> UnifiedCache:
    """Get cache for search results."""
    return get_cache(
        "search_results",
        max_size=int(os.environ.get("SEARCH_CACHE_MAX_SIZE", "1000")),
        max_memory_mb=int(os.environ.get("SEARCH_CACHE_MAX_MEMORY_MB", "100")),
        eviction_policy=EvictionPolicy.TTL,
        default_ttl=float(os.environ.get("SEARCH_CACHE_TTL", "300"))  # 5 minutes
    )


def get_expansion_cache() -> UnifiedCache:
    """Get cache for query expansion results."""
    return get_cache(
        "expansions",
        max_size=int(os.environ.get("EXPANSION_CACHE_MAX_SIZE", "1000")),
        max_memory_mb=int(os.environ.get("EXPANSION_CACHE_MAX_MEMORY_MB", "20")),
        eviction_policy=EvictionPolicy.LRU,
        default_ttl=float(os.environ.get("EXPANSION_CACHE_TTL", "1800"))  # 30 minutes
    )
