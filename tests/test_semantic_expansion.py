#!/usr/bin/env python3
"""
Test script for semantic expansion functionality.
"""

import os
import sys
import unittest
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import modules to test
from scripts.semantic_expansion import (
    _cosine_similarity,
    _extract_code_tokens,
    _get_expansion_cache_key,
    _coerce_embedding_vector,
    _cache_expansion,
    _get_cached_expansion,
    expand_queries_semantically,
    expand_queries_with_prf,
    get_expansion_stats,
    clear_expansion_cache
)

# Import for testing
try:
    from scripts.hybrid_search import expand_queries_enhanced
    HYBRID_SEARCH_AVAILABLE = True
except ImportError:
    HYBRID_SEARCH_AVAILABLE = False


class TestSemanticExpansion(unittest.TestCase):
    """Test semantic expansion functionality."""
    
    def setUp(self):
        """Set up test environment."""
        # Clear cache before each test
        clear_expansion_cache()
    
    def test_cosine_similarity(self):
        """Test cosine similarity calculation."""
        # Identical vectors
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [1.0, 0.0, 0.0]
        self.assertAlmostEqual(_cosine_similarity(vec1, vec2), 1.0, places=5)
        
        # Orthogonal vectors
        vec3 = [1.0, 0.0, 0.0]
        vec4 = [0.0, 1.0, 0.0]
        self.assertAlmostEqual(_cosine_similarity(vec3, vec4), 0.0, places=5)
        
        # Similar vectors
        vec5 = [1.0, 1.0, 0.0]
        vec6 = [1.0, 0.5, 0.0]
        similarity = _cosine_similarity(vec5, vec6)
        self.assertGreater(similarity, 0.8)
        self.assertLess(similarity, 1.0)
        
        # Empty vectors
        self.assertEqual(_cosine_similarity([], []), 0.0)
        self.assertEqual(_cosine_similarity([1.0], []), 0.0)
    
    def test_extract_code_tokens(self):
        """Test code token extraction."""
        text = "function calculate_sum(numbers) { return numbers.reduce((a, b) => a + b, 0); }"
        tokens = _extract_code_tokens(text)
        
        # Should extract meaningful tokens (note: calculate_sum is kept as one token)
        expected_tokens = ['function', 'calculate_sum', 'numbers', 'return', 'reduce']
        for token in expected_tokens:
            self.assertIn(token, tokens)
        
        # Should filter out common words
        self.assertNotIn('the', tokens)
        self.assertNotIn('and', tokens)
        self.assertNotIn('a', tokens)
    
    def test_cache_key_generation(self):
        """Test cache key generation."""
        queries1 = ["function", "calculate", "sum"]
        queries2 = ["sum", "calculate", "function"]  # Same queries, different order
        lang1 = "python"
        lang2 = None
        
        key1 = _get_expansion_cache_key(queries1, lang1)
        key2 = _get_expansion_cache_key(queries2, lang1)  # Should be same after sorting
        key3 = _get_expansion_cache_key(queries1, lang2)  # Different language
        
        # Same queries should generate same key regardless of order
        self.assertEqual(key1, key2)
        
        # Different language should generate different key
        self.assertNotEqual(key1, key3)
        
        # Keys should contain query content
        self.assertIn("function", key1)
        self.assertIn("calculate", key1)
        self.assertIn("sum", key1)
        self.assertIn("lang:python", key1)

    def test_coerce_embedding_vector_handles_varied_shapes(self):
        """_coerce_embedding_vector should flatten nested lists and tolist() outputs."""

        class DummyVec:
            def __init__(self, data):
                self._data = data

            def tolist(self):
                return self._data

        # Nested list
        nested = [[0.1, 0.2, 0.3]]
        self.assertEqual(_coerce_embedding_vector(nested), [0.1, 0.2, 0.3])

        # Object with tolist()
        obj = DummyVec((0.4, 0.5))
        self.assertEqual(_coerce_embedding_vector(obj), [0.4, 0.5])

    def test_coerce_embedding_vector_bad_input(self):
        """_coerce_embedding_vector should return None on invalid input."""

        class BadVec:
            def tolist(self):
                raise ValueError("nope")

        self.assertIsNone(_coerce_embedding_vector(BadVec()))
        self.assertIsNone(_coerce_embedding_vector(None))

    def test_cache_stats_hit_and_miss(self):
        """Cache stats should increment misses on set and hits on retrieval."""
        clear_expansion_cache()
        key = _get_expansion_cache_key(["foo"], "py")

        stats = get_expansion_stats()
        self.assertEqual(stats["cache_hits"], 0)
        self.assertEqual(stats["cache_misses"], 0)

        _cache_expansion(key, ["bar"])
        stats = get_expansion_stats()
        self.assertEqual(stats["cache_hits"], 0)
        self.assertEqual(stats["cache_misses"], 1)

        self.assertEqual(_get_cached_expansion(key), ["bar"])
        stats = get_expansion_stats()
        self.assertEqual(stats["cache_hits"], 1)
        self.assertEqual(stats["cache_misses"], 1)
    
    @patch('scripts.semantic_expansion.FASTEMBED_AVAILABLE', False)
    def test_semantic_expansion_fallback(self):
        """Test semantic expansion falls back gracefully when dependencies unavailable."""
        queries = ["function", "calculate"]
        expansions = expand_queries_semantically(queries)
        
        # Should return empty list when dependencies unavailable
        self.assertEqual(expansions, [])
    
    def test_expand_queries_with_prf_fallback(self):
        """Test PRF expansion with lexical fallback."""
        queries = ["function", "calculate"]
        
        # Mock result objects
        mock_result1 = Mock()
        mock_result1.payload = {
            'metadata': {
                'text': 'function calculate_total(items) { return items.reduce((a, b) => a + b, 0); }',
                'symbol': 'calculate_total',
                'symbol_path': 'utils.calculate_total'
            }
        }
        
        mock_result2 = Mock()
        mock_result2.payload = {
            'metadata': {
                'code': 'def sum_numbers(numbers): return sum(numbers)',
                'symbol': 'sum_numbers',
                'path': '/utils/math.py'
            }
        }
        
        results = [mock_result1, mock_result2]
        
        # Test without embedding model (lexical fallback)
        expansions = expand_queries_with_prf(queries, results, model=None)
        
        # Should return some expansions based on lexical similarity
        self.assertIsInstance(expansions, list)
        # Might be empty if no lexical similarity found, but should not crash
    
    def test_expansion_stats(self):
        """Test expansion statistics tracking."""
        # Initially should be empty
        stats = get_expansion_stats()
        self.assertEqual(stats['cache_hits'], 0)
        self.assertEqual(stats['cache_misses'], 0)
        self.assertEqual(stats['cache_size'], 0)
        
        # Clear cache should reset stats
        clear_expansion_cache()
        stats = get_expansion_stats()
        self.assertEqual(stats['cache_hits'], 0)
        self.assertEqual(stats['cache_misses'], 0)
        self.assertEqual(stats['cache_size'], 0)
    
    @unittest.skipUnless(HYBRID_SEARCH_AVAILABLE, "hybrid_search module not available")
    def test_expand_queries_enhanced(self):
        """Test enhanced query expansion integration."""
        queries = ["function", "calculate"]
        
        # Mock dependencies to test integration
        with patch('scripts.hybrid_search.SEMANTIC_EXPANSION_AVAILABLE', False):
            # Should fall back to basic expansion
            expansions = expand_queries_enhanced(queries, language="python")
            
            # Should include original queries
            self.assertIn("function", expansions)
            self.assertIn("calculate", expansions)
            
            # Should include some synonym expansions
            self.assertGreater(len(expansions), len(queries))


class TestSemanticExpansionIntegration(unittest.TestCase):
    """Integration tests for semantic expansion."""
    
    def setUp(self):
        """Set up integration test environment."""
        # Set test environment variables
        os.environ['SEMANTIC_EXPANSION_ENABLED'] = '1'
        os.environ['SEMANTIC_EXPANSION_TOP_K'] = '5'
        os.environ['SEMANTIC_EXPANSION_SIMILARITY_THRESHOLD'] = '0.7'
        os.environ['SEMANTIC_EXPANSION_MAX_TERMS'] = '3'
        clear_expansion_cache()
    
    def tearDown(self):
        """Clean up test environment."""
        # Restore default environment
        os.environ.pop('SEMANTIC_EXPANSION_ENABLED', None)
        os.environ.pop('SEMANTIC_EXPANSION_TOP_K', None)
        os.environ.pop('SEMANTIC_EXPANSION_SIMILARITY_THRESHOLD', None)
        os.environ.pop('SEMANTIC_EXPANSION_MAX_TERMS', None)
    
    @unittest.skipUnless(
        os.environ.get('RUN_INTEGRATION_TESTS') == '1',
        "Integration tests disabled by default"
    )
    def test_end_to_end_semantic_expansion(self):
        """End-to-end test of semantic expansion (requires real services)."""
        # This test requires actual Qdrant and embedding model
        # Only run when explicitly enabled
        queries = ["python function to calculate sum"]
        
        # Test with mocked client and model
        mock_client = Mock()
        mock_model = Mock()
        
        # Mock search results
        mock_results = [
            Mock(payload={'metadata': {'text': 'def calculate_sum(numbers): return sum(numbers)', 'symbol': 'calculate_sum'}}),
            Mock(payload={'metadata': {'code': 'function total(arr) { return arr.reduce((a,b) => a+b, 0); }', 'symbol': 'total'}})
        ]
        
        mock_client.search.return_value = mock_results
        mock_model.embed.return_value = iter([[[0.1, 0.2, 0.3]]])  # Mock embedding
        
        expansions = expand_queries_semantically(
            queries, 
            client=mock_client, 
            model=mock_model,
            collection="test-collection"
        )
        
        # Should return some expansions
        self.assertIsInstance(expansions, list)
        # Verify client.search was called
        mock_client.search.assert_called_once()


if __name__ == '__main__':
    # Configure test environment
    os.environ['DEBUG_HYBRID_SEARCH'] = '0'  # Disable debug output during tests
    
    # Run tests
    unittest.main(verbosity=2)
