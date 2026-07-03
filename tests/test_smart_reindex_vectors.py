import os
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest


@pytest.mark.usefixtures("monkeypatch")
def test_smart_reindex_refreshes_lex_vector_for_reused_chunks(tmp_path, monkeypatch):
    """When reusing an existing dense embedding, smart reindex must refresh LEX vector.

    Otherwise pseudo/tags changes can drift from the stored lexical vector.
    """
    # The smart reindex logic we test doesn't require the real library.
    monkeypatch.setitem(sys.modules, "fastembed", SimpleNamespace(TextEmbedding=object))

    from scripts import ingest_code

    # Deterministic pseudo/tags so we can predict lexical vector.
    monkeypatch.setattr(
        ingest_code,
        "should_process_pseudo_for_chunk",
        lambda fp, ch, changed: (False, "pseudo", ["tag"]),
    )

    # Avoid touching any caches.
    monkeypatch.setattr(ingest_code, "get_cached_symbols", lambda fp: {})
    monkeypatch.setattr(ingest_code, "compare_symbol_changes", lambda a, b: ([], []))
    monkeypatch.setattr(ingest_code, "set_cached_pseudo", None)
    monkeypatch.setattr(ingest_code, "set_cached_symbols", None)
    monkeypatch.setattr(ingest_code, "set_cached_file_hash", None)

    # Force simple line chunking.
    monkeypatch.setenv("INDEX_MICRO_CHUNKS", "0")
    monkeypatch.setenv("INDEX_SEMANTIC_CHUNKS", "0")
    monkeypatch.setenv("USE_TREE_SITTER", "0")
    monkeypatch.setenv("REFRAG_MODE", "0")

    code = "def add(a, b):\n    return a + b\n"
    fp = tmp_path / "x.py"
    fp.write_text(code, encoding="utf-8")

    # Compute the exact chunk text the indexer will use.
    chunk = ingest_code.chunk_lines(code, max_lines=120, overlap=20)[0]
    code_text = chunk["text"]
    info_text = ingest_code.build_information(
        "python",
        Path(fp),
        chunk["start"],
        chunk["end"],
        code_text.splitlines()[0] if code_text else "",
    )

    dense_key = "dense"
    old_lex = [0.0] * ingest_code.LEX_VECTOR_DIM
    old_lex[0] = 1.0

    existing_record = SimpleNamespace(
        payload={
            "document": info_text,
            "information": info_text,
            "metadata": {
                "path": str(fp),
                "code": code_text,
                "kind": "function",
                "symbol": "add",
                "start_line": 1,
            }
        },
        vector={dense_key: [0.1, 0.2, 0.3], ingest_code.LEX_VECTOR_NAME: old_lex},
    )

    class FakeClient:
        def scroll(self, **kwargs):
            # Return one existing point and then stop.
            if getattr(self, "_done", False):
                return ([], None)
            self._done = True
            return ([existing_record], None)

    captured = {}

    def fake_upsert_points(_client, _collection, points):
        captured["points"] = points

    monkeypatch.setattr(ingest_code, "upsert_points", fake_upsert_points)
    monkeypatch.setattr(ingest_code, "delete_points_by_path", lambda *a, **k: None)

    # Mock embed_batch since dense enrichment may trigger re-embedding
    reused_dense = [0.1, 0.2, 0.3]
    monkeypatch.setattr(ingest_code, "embed_batch", lambda _model, texts: [reused_dense for _ in texts])

    # Model is unused when embeddings are mocked.
    dummy_model = object()

    status = ingest_code.process_file_with_smart_reindexing(
        file_path=Path(fp),
        text=code,
        language="python",
        client=FakeClient(),
        current_collection="c",
        per_file_repo="r",
        model=dummy_model,
        vector_name=dense_key,
    )

    assert status == "success"
    assert "points" in captured and len(captured["points"]) == 1

    out_vec = captured["points"][0].vector
    assert isinstance(out_vec, dict)
    assert ingest_code.LEX_VECTOR_NAME in out_vec

    expected_aug = (code_text or "") + " pseudo" + " tag"
    expected_lex = ingest_code._lex_hash_vector_text(expected_aug)
    assert out_vec[ingest_code.LEX_VECTOR_NAME] == expected_lex
    # Make sure we didn't keep the old lex vector.
    assert out_vec[ingest_code.LEX_VECTOR_NAME] != old_lex


def test_smart_reindex_does_not_reuse_when_info_changes(tmp_path, monkeypatch):
    """Dense embeddings must not be reused if `information` differs."""

    monkeypatch.setitem(sys.modules, "fastembed", SimpleNamespace(TextEmbedding=object))

    from scripts import ingest_code

    # Avoid touching any caches.
    monkeypatch.setattr(ingest_code, "get_cached_symbols", lambda fp: {})
    monkeypatch.setattr(ingest_code, "compare_symbol_changes", lambda a, b: ([], []))
    monkeypatch.setattr(ingest_code, "set_cached_pseudo", None)
    monkeypatch.setattr(ingest_code, "set_cached_symbols", None)
    monkeypatch.setattr(ingest_code, "set_cached_file_hash", None)

    # Force simple line chunking.
    monkeypatch.setenv("INDEX_MICRO_CHUNKS", "0")
    monkeypatch.setenv("INDEX_SEMANTIC_CHUNKS", "0")
    monkeypatch.setenv("USE_TREE_SITTER", "0")
    monkeypatch.setenv("REFRAG_MODE", "0")

    # Make build_information return a value that won't match the stored record.
    old_info = "old-info"
    new_info = "new-info"
    monkeypatch.setattr(ingest_code, "build_information", lambda *a, **k: new_info)

    code = "def hi():\n    return 1\n"
    fp = tmp_path / "x.py"
    fp.write_text(code, encoding="utf-8")

    chunk = ingest_code.chunk_lines(code, max_lines=120, overlap=20)[0]
    code_text = chunk["text"]

    dense_key = "dense"
    reused_dense = [0.123, 0.456]

    existing_record = SimpleNamespace(
        payload={
            "document": old_info,
            "information": old_info,
            "metadata": {
                "path": str(fp),
                "code": code_text,
                "kind": "function",
                "symbol": "hi",
                "start_line": 1,
            },
        },
        vector={dense_key: reused_dense, ingest_code.LEX_VECTOR_NAME: [0.0] * ingest_code.LEX_VECTOR_DIM},
    )

    class FakeClient:
        def scroll(self, **kwargs):
            if getattr(self, "_done", False):
                return ([], None)
            self._done = True
            return ([existing_record], None)

    captured = {}

    def fake_upsert_points(_client, _collection, points):
        captured["points"] = points

    monkeypatch.setattr(ingest_code, "upsert_points", fake_upsert_points)
    monkeypatch.setattr(ingest_code, "delete_points_by_path", lambda *a, **k: None)

    embedded_vec = [9.9, 8.8]
    monkeypatch.setattr(ingest_code, "embed_batch", lambda _model, texts: [embedded_vec for _ in texts])

    status = ingest_code.process_file_with_smart_reindexing(
        file_path=Path(fp),
        text=code,
        language="python",
        client=FakeClient(),
        current_collection="c",
        per_file_repo="r",
        model=object(),
        vector_name=dense_key,
    )

    assert status == "success"
    assert len(captured["points"]) == 1
    out_vec = captured["points"][0].vector
    assert out_vec[dense_key] == embedded_vec
    assert out_vec[dense_key] != reused_dense


def test_smart_reindex_unnamed_reuse_requires_dense_vector(tmp_path, monkeypatch):
    """If an existing unnamed-vector point has only lex/mini, re-embed instead of reusing []."""

    monkeypatch.setitem(sys.modules, "fastembed", SimpleNamespace(TextEmbedding=object))

    from scripts import ingest_code

    # Avoid touching any caches.
    monkeypatch.setattr(ingest_code, "get_cached_symbols", lambda fp: {})
    monkeypatch.setattr(ingest_code, "compare_symbol_changes", lambda a, b: ([], []))
    monkeypatch.setattr(ingest_code, "set_cached_pseudo", None)
    monkeypatch.setattr(ingest_code, "set_cached_symbols", None)
    monkeypatch.setattr(ingest_code, "set_cached_file_hash", None)

    # Force simple line chunking.
    monkeypatch.setenv("INDEX_MICRO_CHUNKS", "0")
    monkeypatch.setenv("INDEX_SEMANTIC_CHUNKS", "0")
    monkeypatch.setenv("USE_TREE_SITTER", "0")
    monkeypatch.setenv("REFRAG_MODE", "0")

    code = "def hi():\n    return 1\n"
    fp = tmp_path / "x.py"
    fp.write_text(code, encoding="utf-8")

    chunk = ingest_code.chunk_lines(code, max_lines=120, overlap=20)[0]
    code_text = chunk["text"]
    info_text = ingest_code.build_information(
        "python",
        Path(fp),
        chunk["start"],
        chunk["end"],
        code_text.splitlines()[0] if code_text else "",
    )

    existing_record = SimpleNamespace(
        payload={
            "document": info_text,
            "information": info_text,
            "metadata": {
                "path": str(fp),
                "code": code_text,
                "kind": "function",
                "symbol": "hi",
                "start_line": 1,
            },
        },
        # Only lex/mini present: should not be reused as dense.
        vector={
            ingest_code.LEX_VECTOR_NAME: [0.0] * ingest_code.LEX_VECTOR_DIM,
            ingest_code.MINI_VECTOR_NAME: [0.0] * ingest_code.MINI_VEC_DIM,
        },
    )

    class FakeClient:
        def scroll(self, **kwargs):
            if getattr(self, "_done", False):
                return ([], None)
            self._done = True
            return ([existing_record], None)

    captured = {}

    def fake_upsert_points(_client, _collection, points):
        captured["points"] = points

    monkeypatch.setattr(ingest_code, "upsert_points", fake_upsert_points)
    monkeypatch.setattr(ingest_code, "delete_points_by_path", lambda *a, **k: None)

    embedded_vec = [7.7, 6.6]
    monkeypatch.setattr(ingest_code, "embed_batch", lambda _model, texts: [embedded_vec for _ in texts])

    status = ingest_code.process_file_with_smart_reindexing(
        file_path=Path(fp),
        text=code,
        language="python",
        client=FakeClient(),
        current_collection="c",
        per_file_repo="r",
        model=object(),
        vector_name=None,
    )

    assert status == "success"
    assert len(captured["points"]) == 1
    out_vec = captured["points"][0].vector
    assert out_vec == embedded_vec
