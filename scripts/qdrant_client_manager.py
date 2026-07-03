#!/usr/bin/env python3
"""
Qdrant client lifecycle management to prevent socket leaks.
Provides connection pooling and singleton client management.
"""
import atexit
import os
import threading
import time
import weakref
from typing import Optional, Dict, List
from contextlib import contextmanager
from qdrant_client import QdrantClient


# Connection pool implementation
class QdrantConnectionPool:
    """Thread-safe connection pool for QdrantClient instances."""
    
    def __init__(self, max_size: int = 10, max_lifetime: float = 300.0):
        self.max_size = max_size
        self.max_lifetime = max_lifetime  # seconds
        self._pool: List[Dict] = []
        self._pool_lock = threading.Lock()
        self._created_count = 0
        self._hits = 0
        self._misses = 0
    
    def get_client(self, url: str, api_key: Optional[str] = None) -> QdrantClient:
        """Get a client from pool or create a new one."""
        with self._pool_lock:
            # Clean up expired connections
            self._cleanup_expired()
            
            # Try to find a matching client in pool
            for i, conn in enumerate(self._pool):
                if (conn['url'] == url and 
                    conn['api_key'] == api_key and 
                    conn['in_use'] == False):
                    conn['in_use'] = True
                    conn['last_used'] = time.time()
                    self._hits += 1
                    return conn['client']
            
            # No suitable client found, create a new one
            if self._created_count < self.max_size:
                client = QdrantClient(url=url, api_key=api_key)
                pool_entry = {
                    'client': client,
                    'url': url,
                    'api_key': api_key,
                    'created_at': time.time(),
                    'last_used': time.time(),
                    'in_use': True
                }
                self._pool.append(pool_entry)
                self._created_count += 1
                self._misses += 1
                return client
            else:
                # Pool is full, create a temporary client (not pooled)
                self._misses += 1
                return QdrantClient(url=url, api_key=api_key)
    
    def return_client(self, client: QdrantClient):
        """Return a client to the pool."""
        with self._pool_lock:
            for conn in self._pool:
                if conn['client'] is client:
                    conn['in_use'] = False
                    conn['last_used'] = time.time()
                    break
    
    def _cleanup_expired(self):
        """Remove expired connections from the pool."""
        current_time = time.time()
        expired_indices = []
        
        for i, conn in enumerate(self._pool):
            if (not conn['in_use'] and 
                current_time - conn['created_at'] > self.max_lifetime):
                expired_indices.append(i)
        
        # Remove expired connections (in reverse order to maintain indices)
        for i in reversed(expired_indices):
            try:
                self._pool[i]['client'].close()
            except Exception:
                pass
            del self._pool[i]
            self._created_count -= 1
    
    def close_all(self):
        """Close all connections in the pool."""
        with self._pool_lock:
            for conn in self._pool:
                try:
                    conn['client'].close()
                except Exception:
                    pass
            self._pool.clear()
            self._created_count = 0
    
    def get_stats(self) -> Dict[str, int]:
        """Get pool statistics."""
        with self._pool_lock:
            return {
                'pool_size': len(self._pool),
                'created_count': self._created_count,
                'hits': self._hits,
                'misses': self._misses,
                'in_use': sum(1 for conn in self._pool if conn['in_use'])
            }


# Global connection pool
_connection_pool: Optional[QdrantConnectionPool] = None
_pool_lock = threading.Lock()

# Legacy singleton support
_client: Optional[QdrantClient] = None
_client_lock = threading.Lock()


def _get_connection_pool() -> QdrantConnectionPool:
    """Get or create the global connection pool."""
    global _connection_pool
    
    with _pool_lock:
        if _connection_pool is None:
            max_size = int(os.environ.get("QDRANT_POOL_MAX_SIZE", "10"))
            max_lifetime = float(os.environ.get("QDRANT_POOL_MAX_LIFETIME", "300"))
            _connection_pool = QdrantConnectionPool(max_size, max_lifetime)
        
        return _connection_pool


def get_qdrant_client(
    url: Optional[str] = None,
    api_key: Optional[str] = None,
    force_new: bool = False,
    use_pool: bool = True
) -> QdrantClient:
    """
    Get or create a Qdrant client with connection pooling support.
    
    Args:
        url: Qdrant URL (defaults to QDRANT_URL env var)
        api_key: API key (defaults to QDRANT_API_KEY env var)
        force_new: Create a new client instead of reusing singleton
        use_pool: Use connection pooling (defaults to True)
    
    Returns:
        QdrantClient instance
    
    Note: When use_pool=True, clients are automatically managed in a thread-safe pool.
    For write-heavy operations, consider using force_new=True.
    """
    url = url or os.environ.get("QDRANT_URL", "http://qdrant:6333")
    api_key = api_key or os.environ.get("QDRANT_API_KEY")
    
    # Use connection pooling if enabled and not forcing new client
    if use_pool and not force_new:
        pool = _get_connection_pool()
        return pool.get_client(url, api_key)
    
    # Fallback to singleton pattern for backward compatibility
    if force_new:
        return QdrantClient(url=url, api_key=api_key if api_key else None)
    
    global _client
    
    with _client_lock:
        if _client is None:
            _client = QdrantClient(url=url, api_key=api_key if api_key else None)
        return _client


def return_qdrant_client(client: QdrantClient):
    """
    Return a client to the connection pool.
    Should be called when done with a pooled client.
    
    Args:
        client: The QdrantClient instance to return to the pool
    """
    if client is None:
        return
    
    pool = _get_connection_pool()
    pool.return_client(client)


@contextmanager
def pooled_qdrant_client(
    url: Optional[str] = None,
    api_key: Optional[str] = None,
    force_new: bool = False
):
    """
    Context manager for getting and returning a pooled Qdrant client.
    
    Usage:
        with pooled_qdrant_client(url="http://localhost:6333") as client:
            # Use client for operations
            client.search(...)
        # Client is automatically returned to pool
    
    Args:
        url: Qdrant URL (defaults to QDRANT_URL env var)
        api_key: API key (defaults to QDRANT_API_KEY env var)
        force_new: Create a new client instead of using pool
    """
    client = get_qdrant_client(url=url, api_key=api_key, force_new=force_new, use_pool=True)
    try:
        yield client
    finally:
        return_qdrant_client(client)


def close_qdrant_client():
    """
    Close singleton Qdrant client and connection pool.
    Should be called at application shutdown or when switching configurations.
    """
    global _client
    
    # Close singleton client
    with _client_lock:
        if _client is not None:
            try:
                _client.close()
            except Exception:
                pass
            _client = None
    
    # Close connection pool
    with _pool_lock:
        if _connection_pool is not None:
            _connection_pool.close_all()


def reset_qdrant_client(url: Optional[str] = None, api_key: Optional[str] = None):
    """
    Reset singleton client with new configuration.
    Useful when switching between different Qdrant instances.
    """
    close_qdrant_client()
    return get_qdrant_client(url=url, api_key=api_key)


def get_qdrant_pool_stats() -> Dict[str, int]:
    """
    Get connection pool statistics.

    Returns:
        Dictionary with pool statistics including size, hits, misses, etc.
    """
    pool = _get_connection_pool()
    return pool.get_stats()


# ---------------------------------------------------------------------------
# Process Cleanup
# ---------------------------------------------------------------------------
# Register atexit handler to close all connections on process shutdown.
# This prevents socket leaks in long-running servers and ensures clean termination.

def _atexit_cleanup():
    """Cleanup handler called on process exit."""
    try:
        close_qdrant_client()
    except Exception:
        pass  # Best effort cleanup on exit


# Register the cleanup handler
atexit.register(_atexit_cleanup)
