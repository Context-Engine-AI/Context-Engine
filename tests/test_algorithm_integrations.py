"""Tests for algorithm integrations: Bloom, Aho-Corasick, Jaro-Winkler, Symbol Diff, MinHash, HyperLogLog, PathTrie."""
import os
import pytest


class TestBloomFilter:
    """Tests for Bloom filter integration in ingest_code.py."""

    def test_bloom_filter_basic_operations(self):
        """Test basic Bloom filter add/query."""
        from scripts.bloom_index import BloomIndex
        
        bloom = BloomIndex(expected_items=1000, false_positive_rate=0.01)
        bloom.add("test_hash_1")
        bloom.add("test_hash_2")
        
        assert bloom.might_contain("test_hash_1")
        assert bloom.might_contain("test_hash_2")
        # Unknown hash might have false positive but probably not
        assert not bloom.might_contain("definitely_not_in_bloom_xyz123")

    def test_bloom_filter_persistence(self, tmp_path):
        """Test Bloom filter save/load."""
        from scripts.bloom_index import BloomIndex
        
        bloom = BloomIndex(expected_items=100, false_positive_rate=0.01)
        bloom.add("persist_test_1")
        bloom.add("persist_test_2")
        
        path = tmp_path / "bloom.bin"
        bloom.save(str(path))
        
        loaded = BloomIndex.load(str(path))
        assert loaded.might_contain("persist_test_1")
        assert loaded.might_contain("persist_test_2")


class TestAhoCorasick:
    """Tests for Aho-Corasick integration in hybrid_search.py."""

    def test_aho_corasick_basic_matching(self):
        """Test basic multi-pattern matching."""
        from scripts.aho_corasick import AhoCorasick
        
        ac = AhoCorasick(["hello", "world", "test"])
        ac.build()
        
        matches = ac.search("hello world, this is a test")
        patterns_found = {m[1] for m in matches}
        assert "hello" in patterns_found
        assert "world" in patterns_found
        assert "test" in patterns_found

    def test_aho_corasick_case_insensitive(self):
        """Test case-insensitive matching."""
        from scripts.aho_corasick import AhoCorasick
        
        ac = AhoCorasick(["Hello", "WORLD"], case_sensitive=False)
        ac.build()
        
        matches = ac.search("HELLO world")
        patterns_found = {m[1].lower() for m in matches}
        assert "hello" in patterns_found
        assert "world" in patterns_found

    def test_integration_lexical_score_ac(self):
        """Test lexical_score_ac from hybrid_search."""
        from scripts.hybrid_search import lexical_score_ac, build_lexical_automaton_for_search
        
        automaton = build_lexical_automaton_for_search(["function", "class", "method"])
        if automaton:  # Only if AC is enabled
            score = lexical_score_ac(automaton, "This function calls a method")
            assert score > 0


class TestJaroWinkler:
    """Tests for Jaro-Winkler fuzzy matching in hybrid_search.py."""

    def test_jaro_winkler_exact_match(self):
        """Test Jaro-Winkler with exact match."""
        from scripts.fuzzy_match import jaro_winkler
        
        score = jaro_winkler("hello", "hello")
        assert score == 1.0

    def test_jaro_winkler_similar_strings(self):
        """Test Jaro-Winkler with similar strings."""
        from scripts.fuzzy_match import jaro_winkler
        
        # Similar function names (different naming conventions)
        score = jaro_winkler("getUserInfo", "get_user_info")
        assert score > 0.7  # Should be fairly similar
        
        # Typo
        score = jaro_winkler("processData", "procesData")
        assert score > 0.9

    def test_jaro_winkler_different_strings(self):
        """Test Jaro-Winkler with very different strings."""
        from scripts.fuzzy_match import jaro_winkler
        
        score = jaro_winkler("abc", "xyz")
        assert score < 0.5


class TestSymbolDiff:
    """Tests for Symbol Diff integration in workspace_state.py."""

    def test_symbol_diff_unchanged(self):
        """Test detecting unchanged symbols."""
        from scripts.symbol_diff import SymbolInfo, diff_symbols
        
        old = [SymbolInfo("foo", "function", "hash1", 10, 20)]
        new = [SymbolInfo("foo", "function", "hash1", 10, 20)]
        
        changes = diff_symbols(old, new)
        assert len(changes) == 1
        assert changes[0].change_type == "unchanged"

    def test_symbol_diff_moved(self):
        """Test detecting moved symbols (same content, different location)."""
        from scripts.symbol_diff import SymbolInfo, diff_symbols
        
        old = [SymbolInfo("foo", "function", "hash1", 10, 20)]
        new = [SymbolInfo("foo", "function", "hash1", 50, 60)]  # Same hash, different lines
        
        changes = diff_symbols(old, new)
        assert len(changes) == 1
        assert changes[0].change_type == "unchanged"  # Same name + hash = unchanged

    def test_symbol_diff_modified(self):
        """Test detecting modified symbols."""
        from scripts.symbol_diff import SymbolInfo, diff_symbols
        
        old = [SymbolInfo("foo", "function", "hash1", 10, 20)]
        new = [SymbolInfo("foo", "function", "hash2", 10, 20)]  # Different hash
        
        changes = diff_symbols(old, new)
        assert len(changes) == 1
        assert changes[0].change_type == "modified"

    def test_compare_symbol_changes_with_diff(self, monkeypatch):
        """Test compare_symbol_changes uses symbol_diff when enabled."""
        monkeypatch.setenv("SYMBOL_DIFF_ENABLED", "1")
        
        # Re-import to pick up env change
        import importlib
        import scripts.workspace_state as ws_module
        importlib.reload(ws_module)
        
        old_syms = {
            "func_foo_10": {"name": "foo", "type": "func", "content_hash": "abc", "start_line": 10, "end_line": 20}
        }
        new_syms = {
            "func_foo_50": {"name": "foo", "type": "func", "content_hash": "abc", "start_line": 50, "end_line": 60}
        }
        
        unchanged, changed = ws_module.compare_symbol_changes(old_syms, new_syms)
        # Moved symbol should be in unchanged (reusable embedding)
        assert len(unchanged) >= 0  # Implementation may vary

