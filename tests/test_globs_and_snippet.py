import importlib
import builtins
import json
import types
from pathlib import Path

import pytest

# Import targets
hyb = importlib.import_module("scripts.hybrid_search")
srv = importlib.import_module("scripts.mcp_indexer_server")


class _Pt:
    def __init__(self, pid, path):
        self.id = pid
        # minimal payload structure used by hybrid filters
        self.payload = {
            "metadata": {
                "path": path,
                "start_line": 1,
                "end_line": 2,
            }
        }


class _QP:
    def __init__(self, points):
        self.points = points


class FakeQdrant:
    def __init__(self, points):
        self._points = points

    # dense_query tries query_points first, then search on exception
    def query_points(self, **kwargs):
        return _QP(self._points)

    def search(self, **kwargs):
        return self._points


class FakeEmbed:
    class _Vec:
        def __init__(self):
            self._v = [0.01] * 8

        def tolist(self):
            return self._v

    def embed(self, texts):
        for _ in texts:
            yield FakeEmbed._Vec()


@pytest.mark.unit
def test_run_hybrid_search_list_globs(monkeypatch):
    # Prepare fake points across different folders to exercise path_glob/not_glob
    pts = [
        _Pt("1", "src/a.py"),
        _Pt("2", "tests/b_test.py"),
        _Pt("3", "docs/readme.md"),
    ]
    # Patch client factory used inside hybrid_search so we don't hit real Qdrant
    monkeypatch.setattr(hyb, "get_qdrant_client", lambda *a, **k: FakeQdrant(pts))
    monkeypatch.setattr(hyb, "return_qdrant_client", lambda *a, **k: None)
    monkeypatch.setenv("EMBEDDING_MODEL", "unit-test")
    monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")

    # Use fake embedder to avoid model init
    monkeypatch.setattr(hyb, "TextEmbedding", lambda *a, **k: FakeEmbed())
    monkeypatch.setattr(hyb, "_get_embedding_model", lambda *a, **k: FakeEmbed())

    # path_glob supports list: keep src/*.py and tests/*, filter out docs/*
    items = hyb.run_hybrid_search(
        queries=["foo"],
        limit=10,
        per_path=2,
        path_glob=["src/*.py", "tests/*"],
        not_glob=None,
        expand=False,
        model=FakeEmbed(),
    )
    paths = {it.get("path") for it in items}
    assert "src/a.py" in paths
    assert "tests/b_test.py" in paths
    assert "docs/readme.md" not in paths


@pytest.mark.unit
def test_run_hybrid_search_slugged_path_globs(monkeypatch):
    # Ensure path_glob matches slugged /work/<slug>/ paths
    pts = [
        _Pt("1", "/work/repo-slug/src/a.py"),
        _Pt("2", "/work/other/docs/readme.md"),
    ]
    monkeypatch.setattr(hyb, "get_qdrant_client", lambda *a, **k: FakeQdrant(pts))
    monkeypatch.setattr(hyb, "return_qdrant_client", lambda *a, **k: None)
    monkeypatch.setenv("EMBEDDING_MODEL", "unit-test")
    monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
    monkeypatch.setattr(hyb, "TextEmbedding", lambda *a, **k: FakeEmbed())
    monkeypatch.setattr(hyb, "_get_embedding_model", lambda *a, **k: FakeEmbed())

    items = hyb.run_hybrid_search(
        queries=["foo"],
        limit=10,
        per_path=2,
        path_glob=["src/*.py"],  # repo-relative glob
        not_glob=None,
        expand=False,
        model=FakeEmbed(),
    )
    paths = {it.get("path") for it in items}
    assert "/work/repo-slug/src/a.py" in paths
    assert "/work/other/docs/readme.md" not in paths


@pytest.mark.unit
def test_dense_query_preserves_collection_on_filter_drop(monkeypatch):
    calls = []

    class RecordingClient:
        def __init__(self):
            self._fail_once = True

        def query_points(self, **kwargs):
            calls.append(kwargs)
            if self._fail_once:
                self._fail_once = False
                raise Exception("boom")
            return types.SimpleNamespace(points=["ok"])

    # Avoid pulling real qdrant models
    monkeypatch.setattr(
        hyb,
        "models",
        types.SimpleNamespace(SearchParams=lambda hnsw_ef: {"hnsw_ef": hnsw_ef}),
    )

    client = RecordingClient()
    out = hyb.dense_query(client, "vec", [0.1], flt=None, per_query=1, collection_name="explicit-coll")

    assert out == ["ok"]
    assert len(calls) == 2  # first fails, second succeeds after filter drop
    assert calls[1]["collection_name"] == "explicit-coll"
    assert calls[1].get("query_filter") is None or calls[1].get("filter") is None


@pytest.mark.unit
def test_collection_prefers_env_over_state(monkeypatch, tmp_path):
    # State file should be ignored when COLLECTION_NAME env var is set
    state_dir = tmp_path / ".codebase"
    state_dir.mkdir()
    (state_dir / "state.json").write_text(json.dumps({"qdrant_collection": "state-coll"}), encoding="utf-8")

    monkeypatch.setenv("WORKSPACE_PATH", str(tmp_path))
    monkeypatch.setenv("COLLECTION_NAME", "env-coll")

    assert hyb._collection(None) == "env-coll"


@pytest.mark.unit
def test_repo_search_snippet_strict_cap_after_highlight(monkeypatch):
    # Stub run_hybrid_search to emit a single result with a known path and range
    async def fake_run(**kwargs):
        return {"results": [{"path": "/work/f.txt", "start_line": 1, "end_line": 1}]}

    # Force in-process shaping to trigger snippet code path
    monkeypatch.setenv("HYBRID_IN_PROCESS", "1")

    # Monkeypatch srv.hybrid_search.run_hybrid_search result pathing via repo_search flow
    monkeypatch.setattr(
        srv, "_tokens_from_queries", lambda q: ["foo"]
    )  # ensure highlight runs

    # Fake open for the specific /work path
    big_line = "foo " * 1000  # large content to exceed cap
    _orig_open = builtins.open

    def fake_open(path, *a, **k):
        if path == "/work/f.txt":
            return types.SimpleNamespace(readlines=lambda: [big_line])
        return _orig_open(path, *a, **k)  # pragma: no cover

    # Ensure sandbox passes
    monkeypatch.setattr(srv.os.path, "isabs", lambda p: True)
    monkeypatch.setattr(srv.os.path, "realpath", lambda p: "/work/f.txt")
    monkeypatch.setenv("MCP_SNIPPET_MAX_BYTES", "64")

    # Stub hybrid_search.run_hybrid_search to return a single item
    import sys

    class HS:
        @staticmethod
        def run_hybrid_search(**kwargs):
            return [
                {"path": "/work/f.txt", "start_line": 1, "end_line": 1, "score": 1.0}
            ]

    monkeypatch.setitem(sys.modules, "scripts.hybrid_search", HS)

    import io

    # Patch open builtin used by server
    monkeypatch.setattr(builtins, "open", fake_open)

    # Execute
    res = srv.asyncio.get_event_loop().run_until_complete(
        srv.repo_search(
            query="foo", include_snippet=True, highlight_snippet=True, context_lines=0
        )
    )
    snip = res["results"][0].get("snippet", "")
    # Strict cap: final length must be <= cap (64)
    assert len(snip) <= 64


@pytest.mark.unit
def test_repo_search_docstring_clean():
    doc = srv.repo_search.__doc__
    assert doc and "Zero-config code search" in doc
    # Ensure stray inline pseudo-code is not embedded in docstring
    assert "Accept common alias keys from clients" not in doc
