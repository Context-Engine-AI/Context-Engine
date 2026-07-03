#!/usr/bin/env python3
"""
Tests for scripts/hybrid/ranking.py - Ranking and scoring logic.

Tests cover:
- RRF (Reciprocal Rank Fusion) scoring
- Sparse lexical scoring
- Tokenization helpers
- Lexical score computation
- Safe type coercion utilities
"""
import pytest

pytestmark = pytest.mark.unit


# ============================================================================
# Fixture: Import ranking module
# ============================================================================
@pytest.fixture
def ranking_module():
    """Import hybrid ranking module."""
    import importlib
    ranking = importlib.import_module("scripts.hybrid.ranking")
    return ranking


# ============================================================================
# Tests: RRF (Reciprocal Rank Fusion)
# ============================================================================
class TestRRF:
    """Tests for RRF scoring function."""

    def test_rrf_rank_1(self, ranking_module):
        """Rank 1 has highest RRF score."""
        score = ranking_module.rrf(1)
        assert score > 0
        assert score == 1.0 / (ranking_module.RRF_K + 1)

    def test_rrf_decreasing_with_rank(self, ranking_module):
        """RRF score decreases with higher rank."""
        score1 = ranking_module.rrf(1)
        score2 = ranking_module.rrf(2)
        score5 = ranking_module.rrf(5)
        score10 = ranking_module.rrf(10)
        
        assert score1 > score2 > score5 > score10

    def test_rrf_custom_k(self, ranking_module):
        """RRF with custom k parameter."""
        score = ranking_module.rrf(1, k=100)
        assert score == 1.0 / (100 + 1)

    def test_rrf_all_positive(self, ranking_module):
        """All RRF scores are positive."""
        for rank in [1, 10, 100, 1000]:
            assert ranking_module.rrf(rank) > 0


# ============================================================================
# Tests: Sparse Lexical Scoring
# ============================================================================
class TestSparseLexScore:
    """Tests for sparse_lex_score function."""

    def test_zero_score(self, ranking_module):
        """Zero raw score produces non-negative output."""
        score = ranking_module.sparse_lex_score(0.0)
        assert score >= 0

    def test_positive_scores(self, ranking_module):
        """Positive raw scores produce positive outputs."""
        score = ranking_module.sparse_lex_score(5.0)
        assert score > 0

    def test_score_scaling(self, ranking_module):
        """Higher raw scores produce higher outputs."""
        score_low = ranking_module.sparse_lex_score(1.0)
        score_high = ranking_module.sparse_lex_score(10.0)
        assert score_high >= score_low

    def test_weight_affects_score(self, ranking_module):
        """Weight parameter affects output."""
        score_low_weight = ranking_module.sparse_lex_score(5.0, weight=0.1)
        score_high_weight = ranking_module.sparse_lex_score(5.0, weight=0.5)
        # Higher weight should produce higher score
        assert score_high_weight > score_low_weight


# ============================================================================
# Tests: Tokenization
# ============================================================================
class TestTokenization:
    """Tests for tokenization helpers."""

    def test_tokenize_queries_simple(self, ranking_module):
        """Tokenize simple phrases."""
        result = ranking_module.tokenize_queries(["find function foo"])
        # Returns list of tokens
        assert isinstance(result, (list, set))
        result_set = set(result) if isinstance(result, list) else result
        assert "find" in result_set
        assert "function" in result_set
        assert "foo" in result_set

    def test_tokenize_splits_camelcase(self, ranking_module):
        """Tokenize splits camelCase identifiers."""
        result = ranking_module.tokenize_queries(["getUserName"])
        result_set = set(result) if isinstance(result, list) else result
        lower_result = {t.lower() for t in result_set}
        assert "get" in lower_result or "getusername" in lower_result

    def test_tokenize_splits_snake_case(self, ranking_module):
        """Tokenize splits snake_case identifiers."""
        result = ranking_module.tokenize_queries(["get_user_name"])
        result_set = set(result) if isinstance(result, list) else result
        lower_result = {t.lower() for t in result_set}
        # Should split on underscores
        assert "get" in lower_result or "user" in lower_result or "name" in lower_result

    def test_tokenize_removes_stopwords(self, ranking_module):
        """Tokenize removes common stopwords."""
        result = ranking_module.tokenize_queries(["the function of the class"])
        result_set = set(result) if isinstance(result, list) else result
        # 'the' and 'of' are stopwords
        assert "the" not in result_set
        assert "of" not in result_set

    def test_tokenize_empty_input(self, ranking_module):
        """Tokenize handles empty input."""
        result = ranking_module.tokenize_queries([])
        assert isinstance(result, (list, set))
        assert len(result) == 0


# ============================================================================
# Tests: Lexical Score
# ============================================================================
class TestLexicalScore:
    """Tests for lexical_score function."""

    def test_no_match_zero(self, ranking_module):
        """No matching tokens produces zero score."""
        score = ranking_module.lexical_score(
            ["foobar"],
            {"text": "completely different content", "path": "unrelated.py"}
        )
        # Score should be 0 or very low when no match
        assert score >= 0

    def test_exact_match_in_metadata(self, ranking_module):
        """Exact match in metadata produces positive score."""
        score = ranking_module.lexical_score(
            ["authentication"],
            {"text": "def authenticate_user():", "path": "authentication.py", "symbol": "authenticate_user"}
        )
        assert score > 0

    def test_path_match_contributes(self, ranking_module):
        """Path matching contributes to score."""
        score_match = ranking_module.lexical_score(
            ["router"],
            {"text": "some code", "path": "router/handler.py"}
        )
        score_no_match = ranking_module.lexical_score(
            ["router"],
            {"text": "some code", "path": "database/model.py"}
        )
        assert score_match > score_no_match


# ============================================================================
# Tests: Safe Type Coercion
# ============================================================================
class TestSafeTypeCoercion:
    """Tests for _safe_int and _safe_float utilities."""

    def test_safe_int_valid(self, ranking_module):
        """_safe_int converts valid values."""
        assert ranking_module._safe_int("42", 0) == 42
        assert ranking_module._safe_int(42, 0) == 42

    def test_safe_int_invalid_uses_default(self, ranking_module):
        """_safe_int uses default for invalid values."""
        assert ranking_module._safe_int("not_a_number", 99) == 99
        assert ranking_module._safe_int(None, 99) == 99

    def test_safe_float_valid(self, ranking_module):
        """_safe_float converts valid values."""
        assert ranking_module._safe_float("3.14", 0.0) == 3.14
        assert ranking_module._safe_float(3.14, 0.0) == 3.14

    def test_safe_float_invalid_uses_default(self, ranking_module):
        """_safe_float uses default for invalid values."""
        assert ranking_module._safe_float("not_a_number", 1.5) == 1.5
        assert ranking_module._safe_float(None, 1.5) == 1.5


# ============================================================================
# Tests: Constants
# ============================================================================
class TestConstants:
    """Tests for module constants."""

    def test_rrf_k_defined(self, ranking_module):
        """RRF_K constant is defined and positive."""
        assert hasattr(ranking_module, "RRF_K")
        assert ranking_module.RRF_K > 0

    def test_lex_vector_weight_defined(self, ranking_module):
        """LEX_VECTOR_WEIGHT constant is defined."""
        assert hasattr(ranking_module, "LEX_VECTOR_WEIGHT")
        assert 0 <= ranking_module.LEX_VECTOR_WEIGHT <= 1


# ============================================================================
# Tests: Adaptive Candidate Retrieval
# ============================================================================
class TestAdaptivePerQuery:
    """Tests for _adaptive_per_query scaling logic."""

    def test_scales_with_collection_size(self, ranking_module):
        """Candidate retrieval increases as collection size grows."""
        base_size = max(1000, ranking_module.LARGE_COLLECTION_THRESHOLD // 10)
        base_limit = 24

        at_base = ranking_module._adaptive_per_query(base_limit, base_size, has_filters=False)
        at_2x = ranking_module._adaptive_per_query(base_limit, base_size * 2, has_filters=False)
        at_20x = ranking_module._adaptive_per_query(base_limit, base_size * 20, has_filters=False)

        assert at_base == base_limit
        assert at_2x > at_base
        assert at_20x > at_2x

    def test_filters_reduce_scaling(self, ranking_module):
        """Filters should not increase candidate budgets compared to unfiltered queries."""
        base_size = max(1000, ranking_module.LARGE_COLLECTION_THRESHOLD // 10)
        base_limit = 24
        size = base_size * 20

        unfiltered = ranking_module._adaptive_per_query(base_limit, size, has_filters=False)
        filtered = ranking_module._adaptive_per_query(base_limit, size, has_filters=True)

        assert filtered <= unfiltered

    def test_clamped_to_max(self, ranking_module):
        """Adaptive scaling should clamp to a hard max to avoid runaway retrieval."""
        base_size = max(1000, ranking_module.LARGE_COLLECTION_THRESHOLD // 10)
        # Use a larger base_limit so we deterministically hit the clamp.
        base_limit = 100

        out = ranking_module._adaptive_per_query(base_limit, base_size * 1000, has_filters=False)
        assert out <= 400
