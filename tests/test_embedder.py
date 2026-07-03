#!/usr/bin/env python3
"""
Tests for scripts/embedder.py - Centralized embedding model factory.

Tests cover:
- Model dimension detection
- Query prefixing for Qwen3 models
- Qwen3 feature detection
- Model caching behavior
"""
import os
import pytest

pytestmark = pytest.mark.unit


# ============================================================================
# Fixture: Import embedder with isolated environment
# ============================================================================
@pytest.fixture
def embedder_module(monkeypatch):
    """Import embedder with clean environment."""
    import importlib
    
    # Clear relevant env vars for isolated testing
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("QWEN3_EMBEDDING_ENABLED", raising=False)
    monkeypatch.delenv("QWEN3_QUERY_INSTRUCTION", raising=False)
    
    embedder = importlib.import_module("scripts.embedder")
    return importlib.reload(embedder)


# ============================================================================
# Tests: Model Dimension Detection
# ============================================================================
class TestGetModelDimension:
    """Tests for get_model_dimension function."""

    def test_default_bge_base_768(self, embedder_module):
        """BGE-base models return 768 dimensions."""
        dim = embedder_module.get_model_dimension("BAAI/bge-base-en-v1.5")
        assert dim == 768

    def test_bge_small_384(self, embedder_module):
        """BGE-small models return 384 dimensions."""
        dim = embedder_module.get_model_dimension("BAAI/bge-small-en-v1.5")
        assert dim == 384

    def test_bge_large_1024(self, embedder_module):
        """BGE-large models return 1024 dimensions."""
        dim = embedder_module.get_model_dimension("BAAI/bge-large-en-v1.5")
        assert dim == 1024

    def test_minilm_384(self, embedder_module):
        """MiniLM models return 384 dimensions."""
        dim = embedder_module.get_model_dimension("sentence-transformers/all-MiniLM-L6-v2")
        assert dim == 384

    def test_e5_small_384(self, embedder_module):
        """E5-small models return 384 dimensions."""
        dim = embedder_module.get_model_dimension("intfloat/e5-small")
        assert dim == 384

    def test_e5_base_768(self, embedder_module):
        """E5-base models return 768 dimensions."""
        dim = embedder_module.get_model_dimension("intfloat/e5-base")
        assert dim == 768

    def test_e5_large_1024(self, embedder_module):
        """E5-large models return 1024 dimensions."""
        dim = embedder_module.get_model_dimension("intfloat/e5-large")
        assert dim == 1024

    def test_qwen3_1024(self, embedder_module):
        """Qwen3 models return 1024 dimensions."""
        dim = embedder_module.get_model_dimension("electroglyph/Qwen3-Embedding-0.6B-onnx-uint8")
        assert dim == 1024

    def test_unknown_model_defaults_768(self, embedder_module):
        """Unknown models default to 768 dimensions."""
        dim = embedder_module.get_model_dimension("some-unknown/model-name")
        assert dim == 768

    def test_none_uses_default(self, embedder_module, monkeypatch):
        """None model_name uses EMBEDDING_MODEL env or DEFAULT_MODEL."""
        # No env var set, should use DEFAULT_MODEL (bge-base = 768)
        dim = embedder_module.get_model_dimension(None)
        assert dim == 768


# ============================================================================
# Tests: Qwen3 Detection
# ============================================================================
class TestIsQwen3Model:
    """Tests for is_qwen3_model function."""

    def test_detects_qwen3_in_name(self, embedder_module):
        """Detects Qwen3 in model name (case-insensitive)."""
        assert embedder_module.is_qwen3_model("electroglyph/Qwen3-Embedding-0.6B") is True
        assert embedder_module.is_qwen3_model("some/QWEN3-model") is True
        assert embedder_module.is_qwen3_model("qwen3-test") is True

    def test_non_qwen3_models(self, embedder_module):
        """Non-Qwen3 models return False."""
        assert embedder_module.is_qwen3_model("BAAI/bge-base-en-v1.5") is False
        assert embedder_module.is_qwen3_model("sentence-transformers/all-MiniLM") is False


# ============================================================================
# Tests: Query Prefixing
# ============================================================================
class TestPrefixQuery:
    """Tests for prefix_query and prefix_queries functions."""

    def test_non_qwen3_returns_original(self, embedder_module):
        """Non-Qwen3 models don't prefix queries."""
        query = "find function foo"
        result = embedder_module.prefix_query(query, "BAAI/bge-base-en-v1.5")
        assert result == query

    def test_prefix_queries_list(self, embedder_module):
        """prefix_queries handles list of queries for non-Qwen3."""
        queries = ["query1", "query2", "query3"]
        result = embedder_module.prefix_queries(queries, "BAAI/bge-base-en-v1.5")
        assert result == queries


# ============================================================================
# Tests: Constants
# ============================================================================
class TestConstants:
    """Tests for module constants."""

    def test_default_model_defined(self, embedder_module):
        """DEFAULT_MODEL constant is defined."""
        assert hasattr(embedder_module, "DEFAULT_MODEL")
        assert embedder_module.DEFAULT_MODEL == "BAAI/bge-base-en-v1.5"

    def test_qwen3_model_defined(self, embedder_module):
        """QWEN3_MODEL constant is defined."""
        assert hasattr(embedder_module, "QWEN3_MODEL")
        assert "qwen3" in embedder_module.QWEN3_MODEL.lower()

    def test_qwen3_dim_is_1024(self, embedder_module):
        """QWEN3_DIM is 1024."""
        assert embedder_module.QWEN3_DIM == 1024
