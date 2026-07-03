import json
import sys
from types import SimpleNamespace

import importlib


def test_hybrid_cli_json_output(monkeypatch, capsys):
    hy = importlib.import_module("scripts.hybrid_search")
    embedder = importlib.import_module("scripts.embedder")

    class DummyVec:
        def __init__(self):
            self._data = [0.1, 0.2]

        def tolist(self):
            return list(self._data)

    class DummyEmbedding:
        def __init__(self, model_name):
            self.model_name = model_name

        def embed(self, texts):
            for _ in texts:
                yield DummyVec()

    class DummyClient:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    def fake_dense_query(client, vec_name, vector, flt, per_query, collection_name=None, query_text=None):
        md = {
            "path": "/work/pkg/a.py",
            "symbol": "foo",
            "symbol_path": "pkg.a:foo",
            "path_prefix": "/work/pkg",
            "start_line": 1,
            "end_line": 2,
            "code": "def foo():\n    return 1\n",
            "imports": ["pkg.b"],
            "calls": ["pkg.bar"],
        }
        return [SimpleNamespace(id="1", payload={"metadata": md})]

    monkeypatch.setenv("COLLECTION_NAME", "test-collection")
    monkeypatch.setenv("QDRANT_URL", "http://example.invalid:6333")
    monkeypatch.setenv("EMBEDDING_MODEL", "stub-model")
    monkeypatch.setattr(hy, "TextEmbedding", DummyEmbedding)
    monkeypatch.setattr(hy, "_get_embedding_model", lambda *a, **k: DummyEmbedding("stub-model"))
    monkeypatch.setattr(embedder, "get_embedding_model", lambda *a, **k: DummyEmbedding("stub-model"))
    monkeypatch.setattr(hy, "QdrantClient", DummyClient)
    monkeypatch.setattr(hy, "_ensure_collection", lambda *a, **k: None)
    monkeypatch.setattr(hy, "lex_hash_vector", lambda *a, **k: [])
    monkeypatch.setattr(hy, "lex_query", lambda *a, **k: [])
    monkeypatch.setattr(hy, "expand_queries", lambda queries, lang=None: queries)
    monkeypatch.setattr(hy, "_embed_queries_cached", lambda *a, **k: [[0.1, 0.2]])
    monkeypatch.setattr(hy, "dense_query", fake_dense_query)
    monkeypatch.setattr(hy, "lexical_score", lambda *a, **k: 0.0)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "hybrid_search.py",
            "--json",
            "--query",
            "foo",
            "--limit",
            "1",
        ],
    )

    hy.main()

    captured = capsys.readouterr()
    lines = [ln for ln in captured.out.splitlines() if ln.strip()]
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["path"] == "/work/pkg/a.py"
    assert payload["components"]["dense_rrf"] > 0
    # "why" is optional (disabled by default via INCLUDE_WHY=0)
    # If present, verify it contains dense_rrf; otherwise just check components
    if "why" in payload:
        assert any("dense_rrf" in entry for entry in payload["why"])
    assert payload["relations"]["imports"] == ["pkg.b"]
