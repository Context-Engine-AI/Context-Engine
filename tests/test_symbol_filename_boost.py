"""Tests for symbol and filename boosting logic in hybrid_search.

Tests token-level matching for symbols (camelCase/snake_case splitting) and the
filename boost helper used by hybrid_search.
"""
import pytest
from scripts.hybrid_search import _split_ident
from scripts.rerank_recursive.utils import _compute_fname_boost


class TestSplitIdent:
    """Test identifier splitting for camelCase/snake_case."""

    def test_snake_case(self):
        assert _split_ident("user_auth_handler") == ["user", "auth", "handler"]

    def test_camel_case(self):
        assert _split_ident("userAuthHandler") == ["user", "auth", "handler"]

    def test_pascal_case(self):
        assert _split_ident("UserAuthHandler") == ["user", "auth", "handler"]

    def test_acronym_handling(self):
        # XMLParser -> xml, parser
        result = _split_ident("XMLParser")
        assert "xml" in result
        assert "parser" in result

    def test_mixed_separators(self):
        result = _split_ident("user_authHandler")
        assert "user" in result
        assert "auth" in result
        assert "handler" in result

    def test_filters_stopwords(self):
        # 'the' should be filtered
        result = _split_ident("getTheValue")
        assert "the" not in result
        assert "get" in result
        assert "value" in result

    def test_empty_string(self):
        assert _split_ident("") == []

    def test_single_word(self):
        assert _split_ident("handler") == ["handler"]


class TestSymbolPartMatching:
    """Test that symbol parts are correctly matched against queries."""

    def test_symbol_parts_extracted(self):
        """Verify split parts can match query tokens."""
        symbol = "getUserById"
        parts = set(p.lower() for p in _split_ident(symbol) if len(p) >= 2)
        assert "get" in parts
        assert "user" in parts
        # "by" is filtered as stopword, "id" should remain
        assert "id" in parts

    def test_symbol_parts_from_snake_case(self):
        symbol = "calculate_total_price"
        parts = set(p.lower() for p in _split_ident(symbol) if len(p) >= 2)
        assert "calculate" in parts
        assert "total" in parts
        assert "price" in parts


class TestFilenameBoostLogic:
    """Test filename boost behavior."""

    def test_fname_boost_matches_filename_tokens(self):
        """Two token matches should trigger a boost."""
        q = "authentication service"
        cand = {"path": "services/AuthenticationService.ts"}
        assert _compute_fname_boost(q, cand, 0.1) > 0

    def test_fname_boost_requires_two_matches(self):
        """Single-token queries should not trigger (noise control)."""
        q = "db"
        cand = {"path": "lib/db.py"}
        assert _compute_fname_boost(q, cand, 0.1) == 0.0


class TestBoostIntegration:
    """Integration tests for the boost logic patterns."""

    def test_pattern_symbol_equality_boost(self):
        """Test the pattern used in hybrid_search for symbol equality."""
        symbol = "processUserData"
        sym = symbol.lower()
        sym_parts = set(p.lower() for p in _split_ident(symbol) if len(p) >= 2)
        
        # Query "user" should match via sym_parts
        query = "user"
        ql = query.lower()
        matches = ql == sym or ql in sym_parts
        assert matches, "Query 'user' should match symbol part"

    def test_pattern_filename_boost(self):
        """Filename boost uses the production-grade matcher."""
        q = "authentication service"
        cand = {"path": "services/AuthenticationService.ts"}
        assert _compute_fname_boost(q, cand, 0.1) > 0
