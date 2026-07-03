#!/usr/bin/env python3
"""
Test script for unified caching and request deduplication systems.
"""

import os
import sys
import unittest
from unittest.mock import Mock, patch, MagicMock
import time

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import modules to test
from scripts.cache_manager import (
    UnifiedCache, EvictionPolicy, get_cache, get_all_cache_stats, clear_all_caches,
    get_embedding_cache, get_search_cache, get_expansion_cache, cached
)
from scripts.deduplication import (
    RequestDeduplicator, RequestFingerprint, get_deduplicator,
    is_duplicate_request, get_deduplication_stats, clear_deduplication_cache,
    deduplicate_request
)


class TestUnifiedCache(unittest.TestCase):
    """Test unified caching functionality."""

    def setUp(self):
        """Set up test environment."""
        clear_all_caches()

    def test_cache_creation(self):
        """Test cache creation with different policies."""
        # Test LRU cache
        lru_cache = UnifiedCache("test_lru", max_size=10, eviction_policy=EvictionPolicy.LRU)
        self.assertEqual(lru_cache.name, "test_lru")
        self.assertEqual(lru_cache.eviction_policy, EvictionPolicy.LRU)

        # Test LFU cache
        lfu_cache = UnifiedCache("test_lfu", max_size=10, eviction_policy=EvictionPolicy.LFU)
        self.assertEqual(lfu_cache.eviction_policy, EvictionPolicy.LFU)

        # Test TTL cache
        ttl_cache = UnifiedCache("test_ttl", max_size=10, eviction_policy=EvictionPolicy.TTL, default_ttl=1.0)
        self.assertEqual(ttl_cache.eviction_policy, EvictionPolicy.TTL)

    def test_cache_basic_operations(self):
        """Test basic cache set/get operations."""
        cache = UnifiedCache("test_basic", max_size=5)

        # Test set and get
        self.assertTrue(cache.set("key1", "value1"))
        self.assertEqual(cache.get("key1"), "value1")

        # Test non-existent key
        self.assertIsNone(cache.get("nonexistent"))

        # Test cache contains
        self.assertIn("key1", cache)
        self.assertNotIn("nonexistent", cache)

        # Test cache size
        self.assertEqual(len(cache), 1)
        self.assertEqual(cache.size(), 1)

    def test_cache_eviction(self):
        """Test cache eviction policies."""
        cache = UnifiedCache("test_eviction", max_size=3, eviction_policy=EvictionPolicy.LRU)

        # Fill cache beyond capacity
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")
        self.assertEqual(len(cache), 3)

        # Add one more to trigger eviction
        cache.set("key4", "value4")
        self.assertEqual(len(cache), 3)  # Should still be at max size

        # Check that oldest was evicted (key1)
        self.assertIsNone(cache.get("key1"))
        self.assertEqual(cache.get("key4"), "value4")

    def test_cache_ttl_expiration(self):
        """Test TTL-based expiration."""
        cache = UnifiedCache("test_ttl", max_size=10, eviction_policy=EvictionPolicy.TTL, default_ttl=0.1)

        # Set value with short TTL
        cache.set("key1", "value1", ttl=0.1)
        self.assertEqual(cache.get("key1"), "value1")

        # Wait for expiration
        time.sleep(0.2)
        self.assertIsNone(cache.get("key1"))  # Should be expired

    def test_cache_statistics(self):
        """Test cache statistics tracking."""
        cache = UnifiedCache("test_stats", max_size=5)

        # Perform operations
        cache.set("key1", "value1")
        cache.get("key1")  # Hit
        cache.get("key2")  # Miss

        stats = cache.get_stats()
        self.assertEqual(stats['hits'], 1)
        self.assertEqual(stats['misses'], 1)
        self.assertEqual(stats['current_size'], 1)
        self.assertGreater(stats['hit_rate'], 0)

    def test_cache_complex_keys(self):
        """Test caching with complex key types."""
        cache = UnifiedCache("test_complex", max_size=10)

        # Test different key types
        self.assertTrue(cache.set("string_key", "value1"))
        self.assertTrue(cache.set(("tuple", "key"), "value2"))
        self.assertTrue(cache.set(["list", "key"], "value3"))
        self.assertTrue(cache.set({"dict": "key"}, "value4"))

        # Verify retrieval
        self.assertEqual(cache.get("string_key"), "value1")
        self.assertEqual(cache.get(("tuple", "key")), "value2")
        self.assertEqual(cache.get(["list", "key"]), "value3")
        self.assertEqual(cache.get({"dict": "key"}), "value4")

    def test_cache_memory_management(self):
        """Test cache memory limits."""
        cache = UnifiedCache("test_memory", max_size=5, max_memory_mb=0.001)  # 1KB limit

        # Add values until memory limit is reached
        large_value = "x" * 100  # ~100 bytes
        cache.set("key1", large_value)
        cache.set("key2", large_value)
        cache.set("key3", large_value)
        cache.set("key4", large_value)
        cache.set("key5", large_value)

        # Should trigger eviction due to memory limit
        cache.set("key6", large_value)

        stats = cache.get_stats()
        self.assertLessEqual(stats['current_memory_bytes'], 1024)  # Should be under 1KB
        self.assertLessEqual(len(cache), 5)  # Should be at max size

    def test_cached_decorator(self):
        """Test the cached decorator."""
        call_count = 0

        @cached("test_decorator", ttl=1.0)
        def expensive_function(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        # First call should compute
        result1 = expensive_function(5)
        self.assertEqual(result1, 10)
        self.assertEqual(call_count, 1)

        # Second call should use cache
        result2 = expensive_function(5)
        self.assertEqual(result2, 10)
        self.assertEqual(call_count, 1)  # Should not increase

        # Wait for expiration and call again
        time.sleep(1.1)
        result3 = expensive_function(5)
        self.assertEqual(result3, 10)
        self.assertEqual(call_count, 2)  # Should recompute


class TestRequestDeduplication(unittest.TestCase):
    """Test request deduplication functionality."""

    def setUp(self):
        """Set up test environment."""
        clear_deduplication_cache()

    def test_fingerprint_generation(self):
        """Test request fingerprint generation."""
        request1 = {
            'queries': ['test', 'query'],
            'limit': 10,
            'language': 'python'
        }

        request2 = {
            'queries': ['query', 'test'],  # Different order
            'limit': 10,
            'language': 'python'
        }

        request3 = {
            'queries': ['different', 'query'],
            'limit': 10,
            'language': 'python'
        }

        fp1 = RequestFingerprint(request1)
        fp2 = RequestFingerprint(request2)
        fp3 = RequestFingerprint(request3)

        # Same requests should have same fingerprint
        self.assertEqual(fp1.fingerprint, fp2.fingerprint)

        # Different requests should have different fingerprints
        self.assertNotEqual(fp1.fingerprint, fp3.fingerprint)

        # Fingerprints should be consistent
        self.assertEqual(len(fp1.fingerprint), 64)  # SHA256 hash length

    def test_exact_match_deduplication(self):
        """Test exact match deduplication."""
        deduplicator = RequestDeduplicator(
            "test_exact",
            max_cache_size=10,
            exact_match=True,
            similarity_threshold=1.0
        )

        request = {
            'queries': ['test', 'query'],
            'limit': 10
        }

        # First request should be unique
        is_dup1, fp1 = deduplicator.is_duplicate(request)
        self.assertFalse(is_dup1)
        self.assertIsNone(fp1)

        # Second identical request should be duplicate
        is_dup2, fp2 = deduplicator.is_duplicate(request)
        self.assertTrue(is_dup2)
        self.assertIsNotNone(fp2)

    def test_similarity_match_deduplication(self):
        """Test similarity-based deduplication."""
        deduplicator = RequestDeduplicator(
            "test_similarity",
            max_cache_size=10,
            exact_match=False,
            similarity_threshold=0.8
        )

        request1 = {
            'queries': ['test', 'query'],
            'limit': 10
        }

        request2 = {
            'queries': ['test', 'queries'],  # Very similar
            'limit': 10
        }

        request3 = {
            'queries': ['completely', 'different'],
            'limit': 10
        }

        # First request should be unique
        is_dup1, fp1 = deduplicator.is_duplicate(request1)
        self.assertFalse(is_dup1)
        self.assertIsNone(fp1)

        # Second similar request should be duplicate
        is_dup2, fp2 = deduplicator.is_duplicate(request2)
        self.assertTrue(is_dup2)
        self.assertIsNotNone(fp2)

        # Third different request should be unique
        is_dup3, fp3 = deduplicator.is_duplicate(request3)
        self.assertFalse(is_dup3)
        self.assertIsNone(fp3)

    def test_deduplication_statistics(self):
        """Test deduplication statistics tracking."""
        deduplicator = RequestDeduplicator("test_stats", max_cache_size=5)

        request1 = {'queries': ['test'], 'limit': 10}
        request2 = {'queries': ['test'], 'limit': 10}
        request3 = {'queries': ['different'], 'limit': 10}

        # Process requests
        deduplicator.is_duplicate(request1)  # Unique
        deduplicator.is_duplicate(request2)  # Duplicate
        deduplicator.is_duplicate(request3)  # Unique

        stats = deduplicator.get_stats()
        self.assertEqual(stats['total_requests'], 3)
        self.assertEqual(stats['unique_requests'], 2)
        self.assertEqual(stats['deduped_requests'], 1)
        self.assertEqual(stats['dedup_rate'], 33.33)  # 1/3 * 100

    def test_deduplication_ttl(self):
        """Test TTL-based expiration in deduplication."""
        deduplicator = RequestDeduplicator(
            "test_ttl",
            max_cache_size=5,
            dedup_window_seconds=0.1
        )

        request = {'queries': ['test'], 'limit': 10}

        # First request should be unique
        is_dup1, fp1 = deduplicator.is_duplicate(request)
        self.assertFalse(is_dup1)

        # Wait for expiration
        time.sleep(0.2)

        # Same request should be unique again after expiration
        is_dup2, fp2 = deduplicator.is_duplicate(request)
        self.assertFalse(is_dup2)

    def test_deduplicate_request_decorator(self):
        """Test the deduplicate_request decorator."""
        call_count = 0

        @deduplicate_request(ttl=1.0)
        def expensive_search(query):
            nonlocal call_count
            call_count += 1
            return f"search_result_for_{query}"

        # First call should execute
        result1 = expensive_search("test")
        self.assertEqual(result1, "search_result_for_test")
        self.assertEqual(call_count, 1)

        # Second identical call should be deduplicated
        result2 = expensive_search("test")
        self.assertIsNone(result2)  # Decorator returns None for duplicates
        self.assertEqual(call_count, 1)  # Should not increase

        # Wait for expiration and call again
        time.sleep(1.1)
        result3 = expensive_search("test")
        self.assertEqual(result3, "search_result_for_test")
        self.assertEqual(call_count, 2)


class TestCacheIntegration(unittest.TestCase):
    """Test integration between caching systems."""

    def setUp(self):
        """Set up test environment."""
        clear_all_caches()
        clear_deduplication_cache()

    def test_predefined_caches(self):
        """Test predefined cache configurations."""
        # Test embedding cache
        embed_cache = get_embedding_cache()
        self.assertEqual(embed_cache.name, "embeddings")

        # Test search cache
        search_cache = get_search_cache()
        self.assertEqual(search_cache.name, "search_results")

        # Test expansion cache
        expansion_cache = get_expansion_cache()
        self.assertEqual(expansion_cache.name, "expansions")

    def test_global_cache_operations(self):
        """Test global cache management functions."""
        # Test getting all stats
        all_stats = get_all_cache_stats()
        self.assertIsInstance(all_stats, dict)
        self.assertIn("embeddings", all_stats)
        self.assertIn("search_results", all_stats)
        self.assertIn("expansions", all_stats)

        # Test clearing all caches
        embed_cache = get_embedding_cache()
        embed_cache.set("test_key", "test_value")

        self.assertEqual(len(embed_cache), 1)

        clear_all_caches()

        # Caches should be empty after clear
        self.assertEqual(len(embed_cache), 0)

    def test_deduplicator_global(self):
        """Test global deduplicator functions."""
        # Test getting deduplicator
        deduplicator = get_deduplicator()
        self.assertIsInstance(deduplicator, RequestDeduplicator)

        # Test global deduplication functions
        request = {'queries': ['test'], 'limit': 10}

        is_dup1, fp1 = is_duplicate_request(request)
        self.assertFalse(is_dup1)

        is_dup2, fp2 = is_duplicate_request(request)
        self.assertTrue(is_dup2)  # Should be duplicate now

        # Test getting stats
        stats = get_deduplication_stats()
        self.assertGreater(stats['total_requests'], 0)

        # Test clearing cache
        clear_deduplication_cache()

        stats_after_clear = get_deduplication_stats()
        self.assertEqual(stats_after_clear['total_requests'], 0)  # Should reset


class TestPerformanceCharacteristics(unittest.TestCase):
    """Test performance characteristics of caching and deduplication."""

    def test_cache_performance(self):
        """Test cache performance with large datasets."""
        cache = UnifiedCache("perf_test", max_size=1000)

        # Measure time for bulk operations
        start_time = time.time()

        # Bulk insert
        for i in range(500):
            cache.set(f"key_{i}", f"value_{i}")

        insert_time = time.time() - start_time

        # Measure retrieval time
        start_time = time.time()
        for i in range(500):
            cache.get(f"key_{i}")

        retrieval_time = time.time() - start_time

        # Performance should be reasonable
        self.assertLess(insert_time, 1.0)  # Should complete within 1 second
        self.assertLess(retrieval_time, 0.5)  # Should complete within 0.5 seconds

        stats = cache.get_stats()
        self.assertEqual(stats['current_size'], 500)

    def test_deduplication_performance(self):
        """Test deduplication performance with many requests."""
        deduplicator = RequestDeduplicator("perf_test", max_cache_size=1000)

        # Measure time for bulk duplicate checking
        start_time = time.time()

        # Create a mix of unique and duplicate requests (50 unique repeated twice)
        requests = [
            {'queries': [f'query_{i % 50}'], 'limit': 10}
            for i in range(100)
        ]

        duplicate_count = 0
        for request in requests:
            is_dup, _ = deduplicator.is_duplicate(request)
            if is_dup:
                duplicate_count += 1

        processing_time = time.time() - start_time

        # Performance should be reasonable
        self.assertLess(processing_time, 1.0)  # Should complete within 1 second
        self.assertGreater(duplicate_count, 0)  # Should detect duplicates

        stats = deduplicator.get_stats()
        self.assertEqual(stats['total_requests'], 100)


if __name__ == '__main__':
    # Configure test environment
    os.environ['DEBUG_CACHE_MANAGER'] = '0'  # Disable debug output during tests
    os.environ['DEBUG_DEDUPLICATION'] = '0'

    # Run tests
    unittest.main(verbosity=2)


def test_cache_update_respects_memory_cap():
    """Updating an existing key with a much larger value should not exceed memory cap.
    If the updated entry alone violates the cap, the set should fail and the key be removed.
    """
    cache = UnifiedCache("test_update_mem", max_size=5, max_memory_mb=0.001)  # ~1KB

    small = "x" * 100  # ~100 bytes
    assert cache.set("k", small)

    large = "y" * 2000  # ~2KB, exceeds cap on its own
    ok = cache.set("k", large)

    # Should fail to retain the oversized update and not exceed memory cap
    assert ok is False
    # Prior small value should be retained on failed oversized update
    assert cache.get("k") == small

    stats = cache.get_stats()
    assert stats["current_memory_bytes"] <= 1024
