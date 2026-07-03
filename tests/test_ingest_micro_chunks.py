import os
import pytest

import sys, types, importlib

# Stub heavy optional deps to allow importing ingest_code in a minimal test env
sys.modules.setdefault(
    "qdrant_client",
    types.SimpleNamespace(QdrantClient=object, models=types.SimpleNamespace()),
)
sys.modules.setdefault("fastembed", types.SimpleNamespace(TextEmbedding=object))
from scripts.ingest_code import chunk_by_tokens


def test_chunk_by_tokens_basic_overlap_and_bounds(monkeypatch):
    # Force small token windows for the test
    monkeypatch.setenv("MICRO_CHUNK_TOKENS", "8")
    monkeypatch.setenv("MICRO_CHUNK_STRIDE", "4")

    # Synthetic text with multiple identifiers and varying line lengths
    text = (
        "def foo(x):\n"  # 1
        "    return x + 1\n"  # 2
        "\n"  # 3
        "class Bar:\n"  # 4
        "    def baz(self, y):\n"  # 5
        "        v = y * 2\n"  # 6
        "        return v\n"  # 7
        "\n"  # 8
        "# some comment and extra text to increase token count\n"  # 9
        "for i in range(10):\n"  # 10
        "    print(i)\n"  # 11
    )

    chunks = chunk_by_tokens(text)

    # Should generate multiple micro-chunks
    assert isinstance(chunks, list)
    assert len(chunks) >= 2

    lines = text.splitlines()
    total_lines = len(lines)

    # Each chunk must have non-empty text and valid line bounds
    for ch in chunks:
        assert ch["text"], "chunk text should not be empty"
        assert 1 <= ch["start"] <= total_lines
        assert 1 <= ch["end"] <= total_lines
        assert ch["end"] >= ch["start"]

    # Starts should be non-decreasing and there should be overlap due to stride < k
    starts = [c["start"] for c in chunks]
    assert starts == sorted(starts), "chunk starts should be non-decreasing"

    # There should be at least one pair of chunks with an overlapping line range
    overlaps = 0
    for a, b in zip(chunks, chunks[1:]):
        if a["end"] >= b["start"]:
            overlaps += 1
    assert overlaps >= 1
