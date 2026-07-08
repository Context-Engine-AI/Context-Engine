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


def test_chunk_semantic_regex_fallback_does_not_drop_gaps(monkeypatch):
    monkeypatch.setenv("USE_TREE_SITTER", "0")
    monkeypatch.setenv("INDEX_USE_ENHANCED_AST", "0")  # force regex/builtin-ast fallback, not the AST analyzer
    lines = [
        '"""Module docstring."""',  # 1
        "import os",  # 2
        "import sys",  # 3
        "",  # 4
        "",  # 5
        "def foo():",  # 6
        "    return 1",  # 7
        "",  # 8
        "",  # 9
        "x = 1",  # 10  -- top-level gap between foo() and bar()
        "y = 2",  # 11
        "z = 3",  # 12
        "",  # 13
        "",  # 14
        "def bar():",  # 15
        "    return 2",  # 16
    ]
    text = "\n".join(lines)
    total_lines = len(lines)

    chunks = ing.chunk_semantic(text, language="python", max_lines=50, overlap=0)

    # Every input line must appear in exactly one emitted chunk: the leading
    # docstring/imports and the top-level code between foo() and bar() must not
    # be silently dropped when the regex-fallback jumps straight to a symbol.
    covered = []
    for c in chunks:
        covered.extend(range(c["start"], c["end"] + 1))
    assert covered == list(range(1, total_lines + 1))

    assert any(c.get("symbol") == "foo" for c in chunks)
    assert any(c.get("symbol") == "bar" for c in chunks)
