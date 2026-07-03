#!/usr/bin/env python3
"""
Tests for scripts/deduplication.py - Request deduplication system.

Tests cover:
- RequestFingerprint generation and expiration
- RequestDeduplicator duplicate detection
- Statistics tracking
- Cache eviction
"""
import time
import pytest

pytestmark = pytest.mark.unit


# ============================================================================
# Fixture: Import deduplication module
# ============================================================================
@pytest.fixture
def dedup_module():
    """Import deduplication module."""
    import importlib
    dedup = importlib.import_module("scripts.deduplication")
    return dedup


@pytest.fixture
def request_fingerprint(dedup_module):
    """Create a RequestFingerprint instance."""
    return dedup_module.RequestFingerprint


@pytest.fixture
def request_deduplicator(dedup_module):
    """Create a fresh RequestDeduplicator instance."""
    return dedup_module.RequestDeduplicator(
        name="test",
        dedup_window_seconds=60,
        max_cache_size=100,
    )


# ============================================================================
# Tests: RequestFingerprint
# ============================================================================
class TestRequestFingerprint:
    """Tests for RequestFingerprint class."""

    def test_fingerprint_generation(self, request_fingerprint):
        """Fingerprint is generated from request data."""
        fp = request_fingerprint({"queries": ["find function foo"]})
        assert fp.fingerprint is not None
        assert isinstance(fp.fingerprint, str)
        assert len(fp.fingerprint) > 0

    def test_same_data_same_fingerprint(self, request_fingerprint):
        """Same request data produces same fingerprint."""
        data = {"queries": ["find function foo"], "limit": 10}
        fp1 = request_fingerprint(data)
        fp2 = request_fingerprint(data)
        assert fp1.fingerprint == fp2.fingerprint

    def test_different_queries_different_fingerprint(self, request_fingerprint):
        """Different queries produce different fingerprint."""
        fp1 = request_fingerprint({"queries": ["foo"]})
        fp2 = request_fingerprint({"queries": ["bar"]})
        assert fp1.fingerprint != fp2.fingerprint

    def test_access_updates_last_accessed(self, request_fingerprint):
        """access() updates last_accessed timestamp."""
        fp = request_fingerprint({"queries": ["test"]})
        original = fp.last_accessed
        time.sleep(0.01)
        fp.access()
        assert fp.last_accessed >= original

    def test_is_expired(self, request_fingerprint):
        """is_expired returns True when TTL exceeded."""
        fp = request_fingerprint({"queries": ["test"]})
        # Should not be expired with 60s TTL
        assert fp.is_expired(60) is False
        # Should be expired with 0s TTL
        assert fp.is_expired(0) is True

    def test_get_age_seconds(self, request_fingerprint):
        """get_age_seconds returns age in seconds."""
        fp = request_fingerprint({"queries": ["test"]})
        age = fp.get_age_seconds()
        assert age >= 0
        assert age < 1  # Should be less than 1 second old


# ============================================================================
# Tests: RequestDeduplicator
# ============================================================================
class TestRequestDeduplicator:
    """Tests for RequestDeduplicator class."""

    def test_first_request_not_duplicate(self, request_deduplicator):
        """First request is never a duplicate."""
        is_dup, similar = request_deduplicator.is_duplicate({"queries": ["find foo"]})
        assert is_dup is False
        assert similar is None

    def test_exact_duplicate_detected(self, request_deduplicator):
        """Exact duplicate request is detected."""
        data = {"queries": ["find function foo"], "limit": 10}
        
        # First request
        is_dup1, _ = request_deduplicator.is_duplicate(data)
        assert is_dup1 is False
        
        # Second identical request
        is_dup2, similar = request_deduplicator.is_duplicate(data)
        assert is_dup2 is True
        assert similar is not None

    def test_different_queries_not_duplicate(self, request_deduplicator):
        """Different queries is not a duplicate."""
        request_deduplicator.is_duplicate({"queries": ["foo"]})
        is_dup, _ = request_deduplicator.is_duplicate({"queries": ["bar"]})
        assert is_dup is False

    def test_clear_cache(self, request_deduplicator):
        """clear_cache removes all entries."""
        request_deduplicator.is_duplicate({"queries": ["test1"]})
        request_deduplicator.is_duplicate({"queries": ["test2"]})
        
        request_deduplicator.clear_cache()
        assert len(request_deduplicator) == 0

    def test_stats_tracking(self, request_deduplicator):
        """Statistics are tracked correctly."""
        # Initial stats
        stats = request_deduplicator.get_stats()
        assert stats["total_requests"] == 0
        assert stats["deduped_requests"] == 0
        
        # Make some requests
        request_deduplicator.is_duplicate({"queries": ["foo"]})
        request_deduplicator.is_duplicate({"queries": ["foo"]})  # duplicate
        request_deduplicator.is_duplicate({"queries": ["bar"]})
        
        stats = request_deduplicator.get_stats()
        assert stats["total_requests"] == 3
        assert stats["deduped_requests"] == 1


# ============================================================================
# Tests: Global Functions
# ============================================================================
class TestGlobalFunctions:
    """Tests for module-level convenience functions."""

    def test_get_deduplicator_returns_instance(self, dedup_module):
        """get_deduplicator returns a RequestDeduplicator instance."""
        deduplicator = dedup_module.get_deduplicator()
        assert deduplicator is not None
        assert isinstance(deduplicator, dedup_module.RequestDeduplicator)

    def test_get_deduplicator_singleton(self, dedup_module):
        """get_deduplicator returns same instance."""
        d1 = dedup_module.get_deduplicator()
        d2 = dedup_module.get_deduplicator()
        assert d1 is d2

    def test_is_duplicate_request_function(self, dedup_module):
        """is_duplicate_request works at module level."""
        # Clear any existing state
        dedup_module.clear_deduplication_cache()
        
        result = dedup_module.is_duplicate_request({"queries": ["test"]})
        assert isinstance(result, tuple)
        assert len(result) == 2
        is_dup, similar = result
        assert isinstance(is_dup, bool)

    def test_get_deduplication_stats(self, dedup_module):
        """get_deduplication_stats returns stats dict."""
        stats = dedup_module.get_deduplication_stats()
        assert isinstance(stats, dict)
        assert "total_requests" in stats
        # Uses 'deduped_requests' not 'duplicates_detected'
        assert "deduped_requests" in stats
