#!/usr/bin/env python3
"""
Request deduplication system for Context-Engine.

This module provides intelligent request deduplication to eliminate
redundant processing and improve overall system efficiency.
"""

import os
import time
import hashlib
import threading
import json
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import defaultdict, deque
import logging

logger = logging.getLogger("deduplication")

# MinHash LSH for fast approximate similarity (feature flag: DEDUP_MINHASH_ENABLED=1)
_MINHASH_ENABLED = os.environ.get("DEDUP_MINHASH_ENABLED", "0").lower() in {"1", "true", "yes", "on"}
_MinHash = None
_MinHashLSH = None

def _get_minhash_classes():
    """Lazy-load MinHash classes."""
    global _MinHash, _MinHashLSH
    if _MinHash is None and _MINHASH_ENABLED:
        try:
            from scripts.min_hash import MinHash, MinHashLSH
            _MinHash = MinHash
            _MinHashLSH = MinHashLSH
        except ImportError:
            pass
    return _MinHash, _MinHashLSH


class RequestFingerprint:
    """Represents a unique fingerprint for a request."""
    
    def __init__(self, request_data: Dict[str, Any]):
        self.request_data = request_data
        self.fingerprint = self._generate_fingerprint(request_data)
        self.created_at = time.time()
        self.access_count = 1
        self.last_accessed = self.created_at
    
    def _generate_fingerprint(self, request_data: Dict[str, Any]) -> str:
        """Generate a consistent fingerprint from request data."""
        try:
            # Normalize request data for consistent fingerprinting
            normalized = {}
            
            # Key fields to include in fingerprint (order matters)
            key_fields = [
                'queries',
                'limit', 
                'per_path',
                'language',
                'under',
                'kind',
                'symbol',
                'ext',
                'not',
                'case',
                'path_regex',
                'path_glob',
                'not_glob',
                'expand',
                'collection',
                'vector_name'
            ]
            
            for field in key_fields:
                if field in request_data:
                    value = request_data[field]
                    
                    # Normalize different types
                    if isinstance(value, (list, tuple)):
                        # Sort lists for consistent ordering
                        if isinstance(value, list):
                            normalized[field] = sorted([str(v).lower().strip() for v in value])
                        else:
                            normalized[field] = tuple(sorted([str(v).lower().strip() for v in value]))
                    elif isinstance(value, dict):
                        # Sort dict keys and normalize values
                        normalized[field] = {
                            str(k).lower(): str(v).lower() 
                            for k, v in sorted(value.items())
                        }
                    elif isinstance(value, bool):
                        normalized[field] = value
                    else:
                        # Normalize strings
                        normalized[field] = str(value).lower().strip()
            
            # Create fingerprint from normalized data
            fingerprint_data = json.dumps(normalized, sort_keys=True, separators=(',', ':'))
            return hashlib.sha256(fingerprint_data.encode('utf-8')).hexdigest()
            
        except Exception as e:
            logger.debug(f"Error generating fingerprint: {e}")
            # Fallback to simple hash of raw data
            return hashlib.md5(str(request_data).encode()).hexdigest()
    
    def access(self) -> None:
        """Record access to this request."""
        self.access_count += 1
        self.last_accessed = time.time()
    
    def is_expired(self, ttl: float) -> bool:
        """Check if request fingerprint has expired."""
        return time.time() - self.created_at > ttl
    
    def get_age_seconds(self) -> float:
        """Get age of this request fingerprint in seconds."""
        return time.time() - self.created_at


class RequestDeduplicator:
    """
    Intelligent request deduplication system.

    Features:
    - Configurable deduplication windows
    - LRU eviction of old requests
    - Statistics tracking
    - Thread-safe operations
    - Multiple deduplication strategies
    - Optional MinHash LSH for fast approximate similarity (DEDUP_MINHASH_ENABLED=1)
    """

    def __init__(
        self,
        name: str = "default",
        dedup_window_seconds: int = 60,
        max_cache_size: int = 10000,
        cleanup_interval: float = 30.0,
        exact_match: bool = True,
        similarity_threshold: float = 0.9
    ):
        self.name = name
        self.dedup_window_seconds = dedup_window_seconds
        self.max_cache_size = max_cache_size
        self.cleanup_interval = cleanup_interval
        self.exact_match = exact_match
        self.similarity_threshold = similarity_threshold

        # Storage for request fingerprints
        self._fingerprints: Dict[str, RequestFingerprint] = {}
        self._access_order = deque()  # For LRU tracking

        # MinHash LSH index for fast similarity search (optional)
        self._minhash_lsh = None
        self._minhashes: Dict[str, Any] = {}  # fingerprint -> MinHash
        if _MINHASH_ENABLED and not exact_match:
            MinHash, MinHashLSH = _get_minhash_classes()
            if MinHashLSH:
                self._minhash_lsh = MinHashLSH(num_perm=128, threshold=similarity_threshold)

        # Thread safety
        self._lock = threading.RLock()

        # Statistics
        self._stats = {
            'total_requests': 0,
            'deduped_requests': 0,
            'unique_requests': 0,
            'cache_hits': 0,
            'cache_size': 0,
            'dedup_rate': 0.0,
            'minhash_enabled': self._minhash_lsh is not None
        }

        # Start cleanup thread
        self._cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        self._cleanup_thread.start()

        logger.debug(f"Initialized request deduplicator with window={dedup_window_seconds}s, "
                    f"max_size={max_cache_size}, exact_match={exact_match}, "
                    f"minhash_lsh={'enabled' if self._minhash_lsh else 'disabled'}")
    
    def _normalize_request_data(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize request data for consistent deduplication (case/whitespace + light stemming)."""
        def _stem_token(tok: str) -> str:
            t = tok.lower().strip()
            # minimal stemming: queries -> query, classes -> class, etc.
            if t.endswith("ies") and len(t) > 4:
                t = t[:-3] + "y"
            elif t.endswith("es") and len(t) > 3:
                t = t[:-2]
            elif t.endswith("s") and len(t) > 3:
                t = t[:-1]
            return t

        normalized: Dict[str, Any] = {}

        # Handle different input types for queries
        if 'queries' in request_data:
            queries = request_data['queries']
            if isinstance(queries, str):
                normalized['queries'] = [_stem_token(queries)]
            elif isinstance(queries, (list, tuple)):
                normalized['queries'] = [_stem_token(str(q)) for q in queries]
            else:
                normalized['queries'] = [_stem_token(str(queries))]

        # Normalize other common fields
        string_fields = ['language', 'under', 'kind', 'symbol', 'ext', 'not', 'case',
                        'path_regex', 'collection', 'vector_name']
        for field in string_fields:
            if field in request_data:
                normalized[field] = _stem_token(str(request_data[field]))

        # Normalize list fields
        list_fields = ['path_glob', 'not_glob']
        for field in list_fields:
            if field in request_data:
                value = request_data[field]
                if isinstance(value, str):
                    normalized[field] = [_stem_token(value)]
                elif isinstance(value, (list, tuple)):
                    normalized[field] = [_stem_token(str(v)) for v in value]
                else:
                    normalized[field] = [_stem_token(str(value))]

        # Copy numeric and boolean fields as-is
        numeric_fields = ['limit', 'per_path']
        for field in numeric_fields:
            if field in request_data:
                try:
                    normalized[field] = int(request_data[field])
                except (ValueError, TypeError):
                    normalized[field] = request_data[field]

        bool_fields = ['expand']
        for field in bool_fields:
            if field in request_data:
                normalized[field] = bool(request_data[field])

        return normalized
    
    def _calculate_similarity(self, cand_norm: Dict[str, Any], exist_norm: Dict[str, Any]) -> float:
        """Calculate similarity between two normalized request dicts.
        Uses Jaccard over flattened key:value tokens. For exact_match, only exact equality returns 1.0.
        """
        if self.exact_match:
            return 1.0 if cand_norm == exist_norm else 0.0

        def _flatten(d: Dict[str, Any]) -> set:
            toks = set()
            for k, v in d.items():
                if isinstance(v, list):
                    toks.update(f"{k}:{str(it)}" for it in v)
                else:
                    toks.add(f"{k}:{str(v)}")
            return toks
        try:
            s1, s2 = _flatten(cand_norm), _flatten(exist_norm)
            if not s1 and not s2:
                return 1.0
            if not s1 or not s2:
                return 0.0
            inter = len(s1 & s2)
            union = len(s1 | s2)
            return inter / union if union else 0.0
        except Exception:
            return 0.0
    
    def _build_minhash(self, normalized_data: Dict[str, Any]) -> Optional[Any]:
        """Build MinHash signature from normalized request data."""
        if not self._minhash_lsh:
            return None
        MinHash, _ = _get_minhash_classes()
        if not MinHash:
            return None

        mh = MinHash(num_perm=128)
        for k, v in normalized_data.items():
            if isinstance(v, list):
                for item in v:
                    mh.update(f"{k}:{item}")
            else:
                mh.update(f"{k}:{v}")
        return mh

    def _find_similar_requests(self, candidate_fp: str, candidate_norm: Dict[str, Any]) -> List[str]:
        """Find requests similar to the given normalized candidate."""
        if self.exact_match:
            return [candidate_fp] if candidate_fp in self._fingerprints else []

        # Use MinHash LSH for fast approximate search if available
        if self._minhash_lsh and len(self._fingerprints) > 10:
            candidate_mh = self._build_minhash(candidate_norm)
            if candidate_mh:
                # Get candidates from LSH (sub-linear search)
                lsh_candidates = list(self._minhash_lsh.query(candidate_mh))
                # Verify with exact Jaccard for candidates only
                similar: List[str] = []
                for fp in lsh_candidates:
                    if fp in self._fingerprints:
                        try:
                            sim = self._calculate_similarity(candidate_norm, self._fingerprints[fp].request_data)
                        except Exception:
                            sim = 0.0
                        if sim >= self.similarity_threshold:
                            similar.append(fp)
                return similar

        # Fallback: linear scan for small caches or when MinHash unavailable
        similar: List[str] = []
        for existing_fp, obj in self._fingerprints.items():
            try:
                sim = self._calculate_similarity(candidate_norm, obj.request_data)
            except Exception:
                sim = 0.0
            if sim >= self.similarity_threshold:
                similar.append(existing_fp)

        return similar
    
    def _cleanup_worker(self) -> None:
        """Background worker for periodic cleanup."""
        while True:
            try:
                time.sleep(self.cleanup_interval)
                self._cleanup_expired()
            except Exception as e:
                logger.error(f"Deduplication cleanup error: {e}")
    
    def _cleanup_expired(self) -> None:
        """Clean up expired request fingerprints."""
        current_time = time.time()
        expired_keys = []
        
        for key, fp in self._fingerprints.items():
            if fp.is_expired(self.dedup_window_seconds):
                expired_keys.append(key)
        
        with self._lock:
            for key in expired_keys:
                del self._fingerprints[key]
                # Remove from access order
                try:
                    self._access_order.remove(key)
                except ValueError:
                    pass
        
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired request fingerprints")
    
    def _evict_if_needed(self) -> None:
        """Evict oldest entries if cache is full."""
        with self._lock:
            while len(self._fingerprints) >= self.max_cache_size:
                try:
                    oldest_key = self._access_order.popleft()
                    if oldest_key in self._fingerprints:
                        del self._fingerprints[oldest_key]
                except (IndexError, KeyError):
                    break
    
    def is_duplicate(self, request_data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Check if request is a duplicate.

        Returns:
            Tuple of (is_duplicate, similar_fingerprint)
        """
        # Normalize request data (no lock needed)
        normalized_data = self._normalize_request_data(request_data)

        # Generate candidate fingerprint (not inserted yet)
        candidate_obj = RequestFingerprint(normalized_data)
        candidate_fp = candidate_obj.fingerprint

        with self._lock:
            # Synchronous TTL cleanup to avoid stale matches in tight loops
            expired_keys = [k for k, fp in self._fingerprints.items() if fp.is_expired(self.dedup_window_seconds)]
            for k in expired_keys:
                try:
                    del self._fingerprints[k]
                    try:
                        self._access_order.remove(k)
                    except ValueError:
                        pass
                except Exception:
                    pass
            if expired_keys:
                # Keep cache_size accurate after purge
                self._stats['cache_size'] = len(self._fingerprints)

            self._stats['total_requests'] += 1

            # Find similar requests using normalized representation
            similar_fingerprints = self._find_similar_requests(candidate_fp, normalized_data)

            if similar_fingerprints:
                # Found similar request(s)
                similar_fp = similar_fingerprints[0]  # Use first match

                # Update access statistics
                self._fingerprints[similar_fp].access()

                # Update access order for LRU
                try:
                    self._access_order.remove(similar_fp)
                    self._access_order.append(similar_fp)
                except ValueError:
                    self._access_order.append(similar_fp)

                self._stats['deduped_requests'] += 1
                self._stats['cache_hits'] += 1

                logger.debug(f"Request deduplicated: {candidate_fp[:8]}... matches {similar_fp[:8]}...")
                return True, similar_fp
            else:
                # New unique request
                self._evict_if_needed()

                # Store new fingerprint
                self._fingerprints[candidate_fp] = candidate_obj
                self._access_order.append(candidate_fp)

                self._stats['unique_requests'] += 1
                self._stats['cache_size'] = len(self._fingerprints)

                logger.debug(f"New unique request: {candidate_fp[:8]}...")
                return False, None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get deduplication statistics."""
        with self._lock:
            stats = self._stats.copy()
            total = stats['total_requests']
            if total > 0:
                stats['dedup_rate'] = round((stats['deduped_requests'] / total) * 100, 2)
            else:
                stats['dedup_rate'] = 0.0

            stats['cache_utilization'] = (stats['cache_size'] / max(1, self.max_cache_size)) * 100

            return stats
    
    def clear_cache(self) -> None:
        """Clear all request fingerprints and reset statistics."""
        with self._lock:
            self._fingerprints.clear()
            self._access_order.clear()
            # Reset statistics
            self._stats = {
                'total_requests': 0,
                'deduped_requests': 0,
                'unique_requests': 0,
                'cache_hits': 0,
                'cache_size': 0,
                'dedup_rate': 0.0
            }

        logger.debug("Cleared request deduplication cache")
    
    def get_cache_size(self) -> int:
        """Get current cache size."""
        with self._lock:
            return len(self._fingerprints)
    
    def __len__(self) -> int:
        """Get current cache size."""
        return self.get_cache_size()


# Global deduplicator instance
_deduplicator: Optional[RequestDeduplicator] = None
_deduplicator_lock = threading.Lock()


def get_deduplicator() -> RequestDeduplicator:
    """Get or create the global request deduplicator."""
    global _deduplicator
    
    with _deduplicator_lock:
        if _deduplicator is None:
            # Configure from environment
            dedup_window = int(os.environ.get("DEDUP_WINDOW_SECONDS", "60"))
            max_cache_size = int(os.environ.get("DEDUP_MAX_CACHE_SIZE", "10000"))
            cleanup_interval = float(os.environ.get("DEDUP_CLEANUP_INTERVAL", "30"))
            exact_match = os.environ.get("DEDUP_EXACT_MATCH", "1").lower() in {"1", "true", "yes"}
            similarity_threshold = float(os.environ.get("DEDUP_SIMILARITY_THRESHOLD", "0.9"))
            
            _deduplicator = RequestDeduplicator(
                dedup_window_seconds=dedup_window,
                max_cache_size=max_cache_size,
                cleanup_interval=cleanup_interval,
                exact_match=exact_match,
                similarity_threshold=similarity_threshold
            )
    
    return _deduplicator


def is_duplicate_request(request_data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Check if a request is a duplicate using the global deduplicator.
    
    Args:
        request_data: Dictionary containing request parameters
        
    Returns:
        Tuple of (is_duplicate, similar_fingerprint)
    """
    deduplicator = get_deduplicator()
    return deduplicator.is_duplicate(request_data)


def get_deduplication_stats() -> Dict[str, Any]:
    """Get deduplication statistics."""
    deduplicator = get_deduplicator()
    return deduplicator.get_stats()


def clear_deduplication_cache() -> None:
    """Clear the deduplication cache."""
    deduplicator = get_deduplicator()
    deduplicator.clear_cache()


# Decorator for automatic request deduplication
def deduplicate_request(ttl: Optional[float] = None):
    """
    Decorator to automatically deduplicate function calls.

    Args:
        ttl: Time-to-live for deduplication in seconds (scoped to this function)
    """
    def decorator(func):
        # Per-function deduplicator so TTL applies locally and predictably in tests/usages
        window = int(ttl) if ttl is not None else int(os.environ.get("DEDUP_WINDOW_SECONDS", "60"))
        local_dedup = RequestDeduplicator(name=f"func:{func.__name__}", dedup_window_seconds=window)

        def wrapper(*args, **kwargs):
            # Build request data from function arguments
            request_data = {
                'function': func.__name__,
                'args': list(args),
                'kwargs': {k: v for k, v in kwargs.items()}
            }

            # Check for duplicates using the local deduplicator
            is_dup, _ = local_dedup.is_duplicate(request_data)
            if is_dup:
                logger.debug(f"Function call deduplicated: {func.__name__}")
                # Return None for duplicates (callers can treat as no-op)
                return None

            # Execute function
            return func(*args, **kwargs)

        wrapper._deduplicated = True
        return wrapper

    return decorator