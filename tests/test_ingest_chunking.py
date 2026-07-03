import os
import textwrap
import importlib

ing = importlib.import_module("scripts.ingest_code")


def test_chunk_lines_basic_overlap():
    text = "\n".join(f"L{i}" for i in range(1, 31))
    chunks = ing.chunk_lines(text, max_lines=10, overlap=2)
    assert chunks[0]["start"] == 1 and chunks[0]["end"] == 10
    assert chunks[1]["start"] == 9  # 10 - overlap + 1
    assert chunks[-1]["end"] == 30


def test_chunk_lines_empty():
    chunks = ing.chunk_lines("", max_lines=10, overlap=2)
    assert chunks == []


def test_chunk_semantic_fallback_no_ts(monkeypatch):
    monkeypatch.setenv("USE_TREE_SITTER", "0")
    text = "\n".join(f"L{i}" for i in range(1, 26))
    chunks = ing.chunk_semantic(text, language="python", max_lines=8, overlap=3)
    # Should behave like chunk_lines because there are no symbols in this text,
    # regardless of tree-sitter availability.
    chunks2 = ing.chunk_lines(text, max_lines=8, overlap=3)
    # Compare ignoring the is_semantic key (added by chunk_semantic wrapper)
    for c in chunks:
        c.pop("is_semantic", None)
    assert chunks == chunks2


def test_chunk_semantic_does_not_cross_large_symbol_boundary(monkeypatch):
    monkeypatch.setenv("USE_TREE_SITTER", "0")
    monkeypatch.setenv("INDEX_USE_ENHANCED_AST", "0")  # Use regex fallback, not AST analyzer
    # Build a large function with proper indentation
    big_body_lines = ["    x = 1" for _ in range(120)]
    text = "def big():\n" + "\n".join(big_body_lines) + "\n    return x\n\ndef small():\n    return 2\n"
    # big(): 1 (def) + 120 body + 1 return = lines 1-122
    big_start = 1
    big_end = 122

    chunks = ing.chunk_semantic(text, language="python", max_lines=30, overlap=5)
    assert any(c.get("symbol") == "big" for c in chunks)
    assert any(c.get("symbol") == "small" for c in chunks)

    for c in chunks:
        if big_start <= c["start"] <= big_end:
            assert c["end"] <= big_end
