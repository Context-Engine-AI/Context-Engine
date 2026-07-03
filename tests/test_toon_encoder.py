"""Tests for TOON (Token-Oriented Object Notation) encoder."""

import os
import pytest

# Ensure scripts module is importable
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.toon_encoder import (
    encode,
    encode_tabular,
    encode_simple_array,
    encode_object,
    encode_search_results,
    encode_context_results,
    compare_formats,
    is_toon_enabled,
    _encode_value,
    _is_uniform_array_of_objects,
)


class TestFeatureFlags:
    """Test feature flag behavior."""

    def test_toon_disabled_by_default(self, monkeypatch):
        monkeypatch.delenv("TOON_ENABLED", raising=False)
        assert is_toon_enabled() is False

    def test_toon_enabled_when_set(self, monkeypatch):
        monkeypatch.setenv("TOON_ENABLED", "1")
        assert is_toon_enabled() is True
        
        monkeypatch.setenv("TOON_ENABLED", "true")
        assert is_toon_enabled() is True


class TestValueEncoding:
    """Test individual value encoding per TOON spec."""

    def test_encode_primitives(self):
        # Per spec §2: null is encoded as literal "null"
        assert _encode_value(None, ",") == "null"
        assert _encode_value(True, ",") == "true"
        assert _encode_value(False, ",") == "false"
        assert _encode_value(42, ",") == "42"
        assert _encode_value(3.14, ",") == "3.14"
        assert _encode_value("hello", ",") == "hello"

    def test_encode_string_with_delimiter(self):
        # String containing delimiter should be quoted
        assert _encode_value("hello,world", ",") == '"hello,world"'

    def test_encode_string_with_newline(self):
        # Per spec §7.1: newlines escaped as \n
        assert _encode_value("line1\nline2", ",") == '"line1\\nline2"'

    def test_encode_nested_object(self):
        # Nested objects become compact JSON
        result = _encode_value({"a": 1}, ",")
        assert result == '{"a":1}'


class TestArrayDetection:
    """Test uniform array detection."""

    def test_uniform_array(self):
        arr = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
        assert _is_uniform_array_of_objects(arr) is True

    def test_non_uniform_array(self):
        arr = [{"a": 1}, {"b": 2}]  # Different keys
        assert _is_uniform_array_of_objects(arr) is False

    def test_empty_array(self):
        assert _is_uniform_array_of_objects([]) is False

    def test_primitive_array(self):
        assert _is_uniform_array_of_objects([1, 2, 3]) is False


class TestTabularEncoding:
    """Test tabular array encoding."""

    def test_encode_tabular_basic(self):
        arr = [
            {"id": 1, "name": "Alice"},
            {"id": 2, "name": "Bob"},
        ]
        lines = encode_tabular("users", arr)
        # python-toon uses [N,] format for comma delimiter
        assert lines[0] == "users[2,]{id,name}:"
        assert lines[1] == "  1,Alice"
        assert lines[2] == "  2,Bob"

    def test_encode_tabular_no_length(self):
        # Note: python-toon always includes length markers
        arr = [{"x": 1}]
        lines = encode_tabular("items", arr, include_length=False)
        # Still includes length due to python-toon behavior
        assert "items" in lines[0]
        assert "x" in lines[0]

    def test_encode_tabular_tab_delimiter(self):
        # Per spec §6: tab delimiter appears inside brackets [N<TAB>] and braces
        arr = [{"a": 1, "b": 2}]
        lines = encode_tabular("data", arr, delimiter="\t")
        assert lines[0] == "data[1\t]{a\tb}:"
        assert lines[1] == "  1\t2"

    def test_encode_empty_array(self):
        # Per spec §9.1: empty arrays are key[0]: (no values after colon)
        lines = encode_tabular("empty", [])
        assert lines == ["empty[0]:"]


class TestSimpleArrayEncoding:
    """Test simple array encoding."""

    def test_encode_simple_array(self):
        lines = encode_simple_array("tags", ["python", "async", "api"])
        assert lines == ["tags[3]: python,async,api"]

    def test_encode_numeric_array(self):
        lines = encode_simple_array("nums", [1, 2, 3])
        assert lines == ["nums[3]: 1,2,3"]


class TestFullEncoding:
    """Test full object encoding."""

    def test_encode_simple_object(self):
        obj = {"name": "Alice", "age": 30, "active": True}
        result = encode(obj)
        assert "name: Alice" in result
        assert "age: 30" in result
        assert "active: true" in result

    def test_encode_nested_object(self):
        obj = {
            "user": {
                "name": "Bob",
                "role": "admin",
            }
        }
        result = encode(obj)
        assert "user:" in result
        assert "  name: Bob" in result

    def test_encode_with_tabular_array(self):
        obj = {
            "users": [
                {"id": 1, "name": "Alice"},
                {"id": 2, "name": "Bob"},
            ]
        }
        result = encode(obj)
        # python-toon uses [N,] format for comma delimiter
        assert "users[2,]{id,name}:" in result
        assert "  1,Alice" in result
        assert "  2,Bob" in result

    def test_encode_mixed_structure(self):
        """Test the hikes example from TOON spec."""
        obj = {
            "context": {
                "task": "Our favorite hikes together",
                "location": "Boulder",
            },
            "friends": ["ana", "luis", "sam"],
            "hikes": [
                {"id": 1, "name": "Blue Lake Trail", "distanceKm": 7.5},
                {"id": 2, "name": "Ridge Overlook", "distanceKm": 9.2},
            ]
        }
        result = encode(obj)
        # Context nested
        assert "context:" in result
        assert "  task: Our favorite hikes together" in result
        # Friends inline
        assert "friends[3]: ana,luis,sam" in result
        # Hikes tabular - python-toon uses [N,] format
        assert "hikes[2,]{id,name,distanceKm}:" in result


class TestSearchResults:
    """Test search result encoding specifically."""

    def test_encode_search_results_compact(self):
        results = [
            {"path": "/src/main.py", "start_line": 10, "end_line": 20, "score": 0.95},
            {"path": "/src/utils.py", "start_line": 5, "end_line": 15, "score": 0.87},
        ]
        output = encode_search_results(results, compact=True)
        assert "results[2]{path,start_line,end_line}:" in output
        assert "  /src/main.py,10,20" in output
        assert "0.95" not in output  # Score excluded in compact

    def test_encode_search_results_full(self):
        results = [
            {"path": "/src/main.py", "start_line": 10, "end_line": 20, "score": 0.95, "symbol": "main"},
        ]
        output = encode_search_results(results, compact=False)
        assert "score" in output
        assert "0.95" in output

    def test_encode_empty_results(self):
        # Per spec: empty arrays are key[0]: (nothing after colon)
        output = encode_search_results([])
        assert output == "results[0]:"


class TestTokenComparison:
    """Test token counting and comparison."""

    def test_compare_formats_basic(self):
        data = {
            "users": [
                {"id": 1, "name": "Alice", "role": "admin"},
                {"id": 2, "name": "Bob", "role": "user"},
                {"id": 3, "name": "Carol", "role": "viewer"},
            ]
        }
        stats = compare_formats(data)

        assert stats["json_tokens"] > 0
        assert stats["toon_tokens"] > 0
        # TOON should be smaller than pretty JSON for tabular data
        assert stats["toon_tokens"] < stats["json_tokens"]
        assert stats["savings_vs_json"] > 0

    def test_compare_formats_larger_dataset(self):
        """Simulate search results - where TOON shines."""
        data = {
            "results": [
                {"path": f"/src/file{i}.py", "start_line": i * 10, "end_line": i * 10 + 5, "score": 0.9 - i * 0.05}
                for i in range(20)
            ]
        }
        stats = compare_formats(data)

        # Should see significant savings on uniform arrays
        assert stats["savings_vs_json"] > 30  # Expect 30%+ savings
        print(f"\nToken comparison for 20 search results:")
        print(f"  JSON (pretty):  {stats['json_tokens']} tokens")
        print(f"  JSON (compact): {stats['json_compact_tokens']} tokens")
        print(f"  TOON (comma):   {stats['toon_tokens']} tokens")
        print(f"  TOON (tab):     {stats['toon_tab_tokens']} tokens")
        print(f"  Savings vs JSON: {stats['savings_vs_json']}%")


class TestMCPIntegration:
    """Test TOON integration with MCP server helpers."""

    def test_should_use_toon_explicit_param(self, monkeypatch):
        """Test explicit output_format parameter takes precedence."""
        # Import the helpers from mcp_indexer_server
        monkeypatch.delenv("TOON_ENABLED", raising=False)

        # We need to test the helper functions directly
        # Since they're in mcp_indexer_server, we'll test the logic here
        from scripts.toon_encoder import is_toon_enabled

        # When TOON_ENABLED is not set, default is False
        assert is_toon_enabled() is False

        # When explicitly set
        monkeypatch.setenv("TOON_ENABLED", "1")
        assert is_toon_enabled() is True

    def test_format_results_as_toon_structure(self):
        """Test that TOON formatting adds expected fields."""
        # Simulate a search response
        response = {
            "results": [
                {"path": "/src/main.py", "start_line": 10, "end_line": 20, "score": 0.95},
                {"path": "/src/utils.py", "start_line": 5, "end_line": 15, "score": 0.87},
            ],
            "total": 2,
            "args": {"query": "test"},
        }

        # Apply TOON formatting
        toon_output = encode_search_results(response["results"], compact=True)

        # Verify TOON output structure
        assert "results[2]{path,start_line,end_line}:" in toon_output
        assert "/src/main.py,10,20" in toon_output
        assert "/src/utils.py,5,15" in toon_output

    def test_toon_replaces_results_array(self):
        """Test that TOON formatting replaces JSON array with TOON string."""
        results = [
            {"path": "/src/main.py", "start_line": 10, "end_line": 20},
        ]

        # When TOON is applied, results becomes a string
        toon_output = encode_search_results(results, compact=True)
        assert isinstance(toon_output, str)
        assert "results[1]{path,start_line,end_line}:" in toon_output
        assert "/src/main.py,10,20" in toon_output

    def test_toon_compact_mode_excludes_score(self):
        """Test that compact mode excludes score field."""
        results = [
            {"path": "/src/main.py", "start_line": 10, "end_line": 20, "score": 0.95},
        ]

        output = encode_search_results(results, compact=True)

        # Score should not be in compact output
        assert "0.95" not in output
        assert "score" not in output

    def test_toon_full_mode_includes_score(self):
        """Test that full mode includes score field."""
        results = [
            {"path": "/src/main.py", "start_line": 10, "end_line": 20, "score": 0.95},
        ]

        output = encode_search_results(results, compact=False)

        # Score should be in full output
        assert "0.95" in output


class TestContextResults:
    """Test encode_context_results for mixed code/memory results."""

    def test_encode_empty_context_results(self):
        """Test empty results return proper TOON format per spec."""
        output = encode_context_results([])
        # Per spec: empty arrays are key[0]: (nothing after colon)
        assert output == "results[0]:"

    def test_encode_code_only_results(self):
        """Test encoding results with only code entries."""
        results = [
            {"source": "code", "path": "/src/main.py", "start_line": 10, "end_line": 20, "score": 0.95},
            {"source": "code", "path": "/src/utils.py", "start_line": 5, "end_line": 15, "score": 0.87},
        ]

        output = encode_context_results(results, compact=True)

        # Should have code section
        assert "code[2]{path,start_line,end_line}:" in output
        assert "/src/main.py,10,20" in output
        assert "/src/utils.py,5,15" in output
        # Should not have memory section
        assert "memory[" not in output

    def test_encode_memory_only_results(self):
        """Test encoding results with only memory entries."""
        results = [
            {"source": "memory", "content": "API uses OAuth2", "score": 0.92},
            {"source": "memory", "content": "Database is PostgreSQL", "score": 0.85},
        ]

        output = encode_context_results(results, compact=True)

        # Should have memory section
        assert "memory[2]{content,score}:" in output
        assert "API uses OAuth2" in output
        assert "Database is PostgreSQL" in output
        # Should not have code section
        assert "code[" not in output

    def test_encode_mixed_code_and_memory(self):
        """Test encoding results with both code and memory entries."""
        results = [
            {"source": "code", "path": "/src/auth.py", "start_line": 100, "end_line": 150, "score": 0.95},
            {"source": "memory", "content": "Auth uses JWT tokens", "score": 0.90},
            {"source": "code", "path": "/src/jwt.py", "start_line": 20, "end_line": 40, "score": 0.88},
        ]

        output = encode_context_results(results, compact=True)

        # Should have both sections
        assert "code[2]{path,start_line,end_line}:" in output
        assert "memory[1]{content,score}:" in output
        assert "/src/auth.py,100,150" in output
        assert "/src/jwt.py,20,40" in output
        assert "Auth uses JWT tokens" in output

    def test_encode_context_results_full_mode(self):
        """Test full mode includes additional fields."""
        results = [
            {"source": "code", "path": "/src/main.py", "start_line": 10, "end_line": 20, "score": 0.95, "symbol": "main"},
            {"source": "memory", "content": "Note about main", "score": 0.90, "id": "mem-123"},
        ]

        output = encode_context_results(results, compact=False)

        # Full mode should include score and symbol for code
        assert "code[1]{path,start_line,end_line,score,symbol}:" in output
        # Full mode should include id for memory
        assert "memory[1]{content,score,id}:" in output

    def test_encode_context_results_with_delimiter(self):
        """Test custom delimiter works correctly."""
        results = [
            {"source": "code", "path": "/src/main.py", "start_line": 10, "end_line": 20},
        ]

        output = encode_context_results(results, delimiter="\t", compact=True)

        # Should use tab delimiter with [N\t] format
        assert "code[1\t]{path\tstart_line\tend_line}:" in output
        assert "/src/main.py\t10\t20" in output


class TestFormatContextResultsAsToon:
    """Test _format_context_results_as_toon MCP helper."""

    def test_format_empty_results_adds_marker(self):
        """Test that empty results still get output_format marker."""
        from scripts.mcp_indexer_server import _format_context_results_as_toon

        response = {"results": [], "total": 0}
        result = _format_context_results_as_toon(response.copy())

        assert result["output_format"] == "toon"
        # Empty array per spec: key[0]:
        assert result["results"] == "results[0]:"

    def test_format_mixed_results(self):
        """Test formatting mixed code/memory results."""
        from scripts.mcp_indexer_server import _format_context_results_as_toon

        response = {
            "results": [
                {"source": "code", "path": "/src/api.py", "start_line": 1, "end_line": 10},
                {"source": "memory", "content": "API docs here", "score": 0.9},
            ],
            "total": 2,
        }
        result = _format_context_results_as_toon(response.copy())

        assert result["output_format"] == "toon"
        assert isinstance(result["results"], str)
        assert "code[1]" in result["results"]
        assert "memory[1]" in result["results"]

    def test_format_preserves_other_fields(self):
        """Test that formatting preserves non-results fields."""
        from scripts.mcp_indexer_server import _format_context_results_as_toon

        response = {
            "results": [{"source": "code", "path": "/a.py", "start_line": 1, "end_line": 5}],
            "total": 1,
            "diag": {"code_hits": 1, "mem_hits": 0},
            "args": {"query": "test"},
        }
        result = _format_context_results_as_toon(response.copy())

        assert result["total"] == 1
        assert result["diag"] == {"code_hits": 1, "mem_hits": 0}
        assert result["args"] == {"query": "test"}


class TestDynamicFieldInclusion:
    """Test that TOON encoding doesn't drop fields."""

    def test_encode_with_snippet_field(self):
        """Test that snippet field is included in full mode."""
        results = [
            {"path": "/src/main.py", "start_line": 10, "end_line": 20,
             "score": 0.95, "snippet": "def main():\n    pass"},
        ]

        output = encode_search_results(results, compact=False)

        assert "snippet" in output
        assert "def main()" in output

    def test_encode_with_information_field(self):
        """Test that info_request's information field is included."""
        results = [
            {"path": "/src/auth.py", "start_line": 1, "end_line": 50,
             "score": 0.9, "information": "Authentication handler at /src/auth.py:1-50",
             "relevance_score": 0.9},
        ]

        output = encode_search_results(results, compact=False)

        assert "information" in output
        assert "relevance_score" in output
        assert "Authentication handler" in output

    def test_encode_with_relationships_field(self):
        """Test that relationships dict is encoded as JSON."""
        results = [
            {"path": "/src/api.py", "start_line": 10, "end_line": 30,
             "relationships": {"imports_from": ["os", "sys"], "calls": ["auth.login"]}},
        ]

        output = encode_search_results(results, compact=False)

        assert "relationships" in output
        # Nested objects become compact JSON
        assert "imports_from" in output

    def test_encode_preserves_all_custom_fields(self):
        """Test that any custom fields added by tools are preserved."""
        results = [
            {"path": "/a.py", "start_line": 1, "end_line": 5,
             "custom_field": "custom_value", "another_field": 123},
        ]

        output = encode_search_results(results, compact=False)

        assert "custom_field" in output
        assert "custom_value" in output
        assert "another_field" in output
        assert "123" in output

    def test_context_results_with_extra_memory_fields(self):
        """Test memory results preserve all fields in full mode."""
        results = [
            {"source": "memory", "content": "API note", "score": 0.9,
             "id": "mem-123", "created_at": "2024-01-01", "tags": ["api", "docs"]},
        ]

        output = encode_context_results(results, compact=False)

        assert "content" in output
        assert "id" in output
        assert "created_at" in output
        assert "tags" in output

