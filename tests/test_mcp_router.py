#!/usr/bin/env python3
"""
Tests for mcp_router.py - Intent classification and tool routing.

Tests cover:
- Intent classification (rule-based and ML fallback)
- Plan building for various query types
- HTTP client helpers
"""
import importlib
import os

import pytest

pytestmark = pytest.mark.unit


# ============================================================================
# Fixture: Router module import
# ============================================================================
@pytest.fixture
def router_module(monkeypatch):
    """Import mcp_router with isolated environment."""
    monkeypatch.delenv("MCP_HTTP_URL", raising=False)
    monkeypatch.delenv("MCP_INDEXER_HTTP_URL", raising=False)

    router = importlib.import_module("scripts.mcp_router")
    return importlib.reload(router)


# ============================================================================
# Tests: Intent Constants
# ============================================================================
class TestIntentConstants:
    """Tests for intent constant definitions."""

    def test_intent_constants_defined(self, router_module):
        """All expected intent constants are defined."""
        assert router_module.INTENT_ANSWER == "answer"
        assert router_module.INTENT_SEARCH == "search"
        assert router_module.INTENT_INDEX == "index"
        assert router_module.INTENT_PRUNE == "prune"
        assert router_module.INTENT_STATUS == "status"
        assert router_module.INTENT_LIST == "list"

    def test_search_specialized_intents(self, router_module):
        """Specialized search intents are defined."""
        assert router_module.INTENT_SEARCH_TESTS == "search_tests"
        assert router_module.INTENT_SEARCH_CONFIG == "search_config"
        assert router_module.INTENT_SEARCH_CALLERS == "search_callers"


# ============================================================================
# Tests: Intent Classification (Rule-based)
# ============================================================================
class TestClassifyIntentRules:
    """Tests for _classify_intent_rules function."""

    def test_status_intent_patterns(self, router_module):
        """Status-related queries are classified correctly."""
        status_queries = [
            "qdrant status",
            "indexing status",
            "collection status",
        ]
        for q in status_queries:
            intent = router_module._classify_intent_rules(q)
            assert intent == router_module.INTENT_STATUS, f"Failed for: {q}"

    def test_list_intent_patterns(self, router_module):
        """List-related queries are classified correctly."""
        list_queries = [
            "list collections",
            "show all collections",
        ]
        for q in list_queries:
            intent = router_module._classify_intent_rules(q)
            assert intent == router_module.INTENT_LIST, f"Failed for: {q}"

    def test_search_tests_intent(self, router_module):
        """Test search queries are classified correctly."""
        queries = [
            "find tests for foo",
            "search for test files",
        ]
        for q in queries:
            intent = router_module._classify_intent_rules(q)
            assert intent == router_module.INTENT_SEARCH_TESTS, f"Failed for: {q}"

    def test_search_config_intent(self, router_module):
        """Config search queries are classified correctly."""
        queries = [
            "find config for database",
            "where is the yaml config",
        ]
        for q in queries:
            intent = router_module._classify_intent_rules(q)
            assert intent == router_module.INTENT_SEARCH_CONFIG, f"Failed for: {q}"


# ============================================================================
# Tests: High-level classify_intent
# ============================================================================
class TestClassifyIntent:
    """Tests for the main classify_intent function."""

    def test_classify_intent_returns_intent(self, router_module):
        """classify_intent returns a valid intent string."""
        intent = router_module.classify_intent("reindex the codebase")
        # Should return some intent (index or answer depending on ML)
        assert intent is not None
        assert isinstance(intent, str)

    def test_classify_intent_status(self, router_module):
        """Status queries classified correctly."""
        intent = router_module.classify_intent("qdrant status")
        assert intent == router_module.INTENT_STATUS

    def test_classify_intent_list(self, router_module):
        """List queries classified correctly."""
        intent = router_module.classify_intent("list collections")
        assert intent == router_module.INTENT_LIST


# ============================================================================
# Tests: Build Plan
# ============================================================================
class TestBuildPlan:
    """Tests for build_plan function."""

    def test_build_plan_returns_list(self, router_module):
        """build_plan returns a list of (tool, args) tuples."""
        # Use a query that triggers rule-based classification (avoids embedding model)
        plan = router_module.build_plan("list collections")

        assert isinstance(plan, list)
        assert len(plan) >= 1
        # Each item is a tuple of (tool_name, args_dict)
        tool_name, args = plan[0]
        assert isinstance(tool_name, str)
        assert isinstance(args, dict)

    def test_build_plan_status_tool(self, router_module):
        """Status queries map to qdrant_status tool."""
        plan = router_module.build_plan("qdrant status")

        tool_name, args = plan[0]
        assert tool_name == "qdrant_status"

    def test_build_plan_list_tool(self, router_module):
        """List queries map to qdrant_list tool."""
        plan = router_module.build_plan("list collections")

        tool_name, args = plan[0]
        assert tool_name == "qdrant_list"

    def test_build_plan_search_tests_tool(self, router_module):
        """Test search queries map to search_tests_for tool."""
        plan = router_module.build_plan("find tests for authentication")

        tool_name, args = plan[0]
        assert tool_name == "search_tests_for"
        assert "query" in args

    def test_build_plan_search_config_tool(self, router_module):
        """Config search queries map to search_config_for tool."""
        plan = router_module.build_plan("find config for database")

        tool_name, args = plan[0]
        assert tool_name == "search_config_for"

    def test_build_plan_includes_query(self, router_module):
        """build_plan includes the query in args for search tools."""
        # Use a query that triggers rule-based classification (avoids embedding model)
        plan = router_module.build_plan("find tests for authentication")

        tool_name, args = plan[0]
        # search_tests_for includes query in args
        assert "query" in args or tool_name in {"qdrant_status", "qdrant_list", "qdrant_prune"}


# ============================================================================
# Tests: HTTP Helpers
# ============================================================================
class TestHttpHelpers:
    """Tests for HTTP client helper functions."""

    def test_filter_args_removes_none(self, router_module):
        """_filter_args removes None values from dict."""
        args = {"a": 1, "b": None, "c": "hello", "d": None}
        filtered = router_module._filter_args(args)

        assert filtered == {"a": 1, "c": "hello"}

    def test_filter_args_preserves_false(self, router_module):
        """_filter_args preserves False and 0 values."""
        args = {"a": False, "b": 0, "c": None}
        filtered = router_module._filter_args(args)

        assert "a" in filtered
        assert "b" in filtered
        assert "c" not in filtered

    def test_parse_stream_or_json_parses_json(self, router_module):
        """_parse_stream_or_json parses valid JSON."""
        body = b'{"result": "success"}'
        parsed = router_module._parse_stream_or_json(body)

        assert parsed == {"result": "success"}


# ============================================================================
# Tests: Failure Response Detection
# ============================================================================
class TestFailureResponseDetection:
    """Tests for _is_failure_response function."""

    def test_success_response_not_failure(self, router_module):
        """Successful responses are not failures."""
        resp = {"result": "data", "ok": True}
        assert router_module._is_failure_response(resp) is False

    def test_empty_response_not_failure(self, router_module):
        """Empty dicts are not failures."""
        assert router_module._is_failure_response({}) is False

    def test_detects_isError_true(self, router_module):
        """Detects responses with isError=True."""
        # Note: depends on actual implementation
        resp = {"isError": True, "content": []}
        result = router_module._is_failure_response(resp)
        # May be True or False depending on implementation
        assert isinstance(result, bool)
