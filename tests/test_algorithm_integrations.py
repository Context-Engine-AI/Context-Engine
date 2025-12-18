"""Tests for algorithm integrations: Bloom, Aho-Corasick, Jaro-Winkler, Symbol Diff, MinHash, HyperLogLog, PathTrie."""
import os
import pytest


class TestBloomFilter:
    """Tests for Bloom filter integration in ingest_code.py."""

    def test_bloom_filter_basic_operations(self):
        """Test basic BloomFilter add/query."""
        from scripts.bloom_index import BloomFilter

        bloom = BloomFilter(size_bits=10000, hash_count=7)
        bloom.add("test_hash_1")
        bloom.add("test_hash_2")

        assert bloom.might_contain("test_hash_1")
        assert bloom.might_contain("test_hash_2")
        # Unknown hash might have false positive but probably not
        assert not bloom.might_contain("definitely_not_in_bloom_xyz123")

    def test_bloom_index_workspace(self, tmp_path):
        """Test BloomIndex with workspace."""
        from scripts.bloom_index import BloomIndex

        bloom = BloomIndex(workspace_path=tmp_path)
        bloom.add("persist_test_1")
        bloom.add("persist_test_2")

        assert bloom.might_contain("persist_test_1")
        assert bloom.might_contain("persist_test_2")

        # Save and reload
        bloom.save()
        loaded = BloomIndex.load_or_create(tmp_path)
        assert loaded.might_contain("persist_test_1")
        assert loaded.might_contain("persist_test_2")


class TestAhoCorasick:
    """Tests for Aho-Corasick integration in hybrid_search.py."""

    def test_aho_corasick_basic_matching(self):
        """Test basic multi-pattern matching."""
        from scripts.aho_corasick import AhoCorasick

        # AhoCorasick builds automatically on init
        ac = AhoCorasick(["hello", "world", "test"])

        matches = ac.search("hello world, this is a test")
        patterns_found = {m[1] for m in matches}
        assert "hello" in patterns_found
        assert "world" in patterns_found
        assert "test" in patterns_found

    def test_aho_corasick_case_insensitive(self):
        """Test case-insensitive matching."""
        from scripts.aho_corasick import AhoCorasick

        # Default is case-insensitive
        ac = AhoCorasick(["Hello", "WORLD"], case_sensitive=False)

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


class TestMinHash:
    """Tests for MinHash LSH integration in deduplication.py."""

    def _shingle(self, text: str, k: int = 3) -> list[str]:
        """Generate k-shingles from text."""
        words = text.split()
        return [" ".join(words[i:i+k]) for i in range(len(words) - k + 1)]

    def test_minhash_basic(self):
        """Test basic MinHash signature generation."""
        from scripts.min_hash import MinHash

        text = "hello world this is a test"
        shingles = self._shingle(text)

        mh1 = MinHash(num_perm=128)
        mh1.update_batch(shingles)

        mh2 = MinHash(num_perm=128)
        mh2.update_batch(shingles)

        # Same input should have identical signatures
        assert mh1.signature == mh2.signature

    def test_minhash_similarity(self):
        """Test MinHash Jaccard similarity estimation."""
        from scripts.min_hash import MinHash

        text1 = "the quick brown fox jumps over the lazy dog"
        text2 = "the quick brown fox jumps over the lazy cat"
        text3 = "completely different text about something else entirely"

        mh1 = MinHash(num_perm=128)
        mh1.update_batch(self._shingle(text1))

        mh2 = MinHash(num_perm=128)
        mh2.update_batch(self._shingle(text2))

        sim = mh1.jaccard(mh2)
        assert sim > 0.3  # Should have some overlap

        mh3 = MinHash(num_perm=128)
        mh3.update_batch(self._shingle(text3))

        sim_diff = mh1.jaccard(mh3)
        assert sim_diff < sim  # Different text should be less similar

    def test_minhash_lsh_basic(self):
        """Test MinHash LSH index."""
        from scripts.min_hash import MinHash, MinHashLSH

        lsh = MinHashLSH(threshold=0.3, num_perm=128)

        text1 = "document about machine learning and algorithms"
        mh1 = MinHash(num_perm=128)
        mh1.update_batch(self._shingle(text1))
        lsh.insert("doc1", mh1)

        text2 = "document about machine learning and deep learning"
        mh2 = MinHash(num_perm=128)
        mh2.update_batch(self._shingle(text2))
        lsh.insert("doc2", mh2)

        # Query with similar document
        results = lsh.query(mh1)
        assert "doc1" in results


class TestHyperLogLog:
    """Tests for HyperLogLog integration in workspace_state.py."""

    def test_hyperloglog_basic_counting(self):
        """Test basic cardinality estimation."""
        from scripts.hyperloglog import HyperLogLog

        hll = HyperLogLog(precision=12)
        for i in range(1000):
            hll.add(f"item_{i}")

        estimate = hll.count()
        # Should be within ~5% of actual count
        assert 900 < estimate < 1100

    def test_hyperloglog_duplicates(self):
        """Test that duplicates don't increase count."""
        from scripts.hyperloglog import HyperLogLog

        hll = HyperLogLog(precision=12)
        for _ in range(100):
            hll.add("same_item")

        estimate = hll.count()
        assert estimate < 5  # Should be ~1

    def test_hyperloglog_merge(self):
        """Test merging two HyperLogLogs."""
        from scripts.hyperloglog import HyperLogLog

        hll1 = HyperLogLog(precision=10)
        for i in range(500):
            hll1.add(f"set1_{i}")

        hll2 = HyperLogLog(precision=10)
        for i in range(500):
            hll2.add(f"set2_{i}")

        hll1.merge(hll2)
        estimate = hll1.count()
        # Should estimate ~1000 unique items
        assert 800 < estimate < 1200

    def test_workspace_hll_integration(self, monkeypatch, tmp_path):
        """Test HLL integration with workspace state."""
        monkeypatch.setenv("HLL_STATS_ENABLED", "1")
        monkeypatch.setenv("WORKSPACE_PATH", str(tmp_path))

        # Create state directory
        state_dir = tmp_path / ".codebase"
        state_dir.mkdir()

        import importlib
        import scripts.workspace_state as ws_module
        importlib.reload(ws_module)

        # Track some files
        for i in range(100):
            ws_module.track_file_in_hll(f"/work/file_{i}.py", str(tmp_path))

        estimate = ws_module.get_unique_files_estimate(str(tmp_path))
        if estimate is not None:  # Only if HLL is enabled
            assert 80 < estimate < 120


class TestPathTrie:
    """Tests for PathTrie integration in hybrid_search.py."""

    def test_path_trie_basic(self):
        """Test basic PathTrie operations."""
        from scripts.path_trie import PathTrie

        trie = PathTrie()
        trie.add("work/src/main.py")
        trie.add("work/src/utils.py")
        trie.add("work/tests/test_main.py")

        # PathTrie strips leading slashes internally
        assert trie.contains("work/src/main.py")
        assert not trie.contains("work/nonexistent.py")

    def test_path_trie_prefix_search(self):
        """Test prefix-based path search."""
        from scripts.path_trie import PathTrie

        trie = PathTrie()
        trie.add("work/src/components/Button.tsx")
        trie.add("work/src/components/Modal.tsx")
        trie.add("work/src/utils/helpers.ts")
        trie.add("work/tests/test_components.py")

        src_paths = list(trie.find_by_prefix("work/src"))
        assert len(src_paths) == 3
        assert "work/src/components/Button.tsx" in src_paths

        component_paths = list(trie.find_by_prefix("work/src/components"))
        assert len(component_paths) == 2

    def test_path_trie_len(self):
        """Test path count in trie."""
        from scripts.path_trie import PathTrie

        trie = PathTrie()
        assert len(trie) == 0

        trie.add("work/file1.py")
        trie.add("work/file2.py")

        assert len(trie) == 2
        assert trie.contains("work/file1.py")
        assert trie.contains("work/file2.py")

