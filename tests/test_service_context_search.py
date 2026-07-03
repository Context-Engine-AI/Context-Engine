import importlib
import json
import pytest

srv = importlib.import_module("scripts.mcp_indexer_server")


class FakePoint:
    def __init__(self, score, payload):
        self.score = score
        self.payload = payload


class FakeQdrantMem:
    def __init__(self, items):
        self._items = items

    def search(self, **kwargs):
        return self._items

    def scroll(self, **kwargs):
        return (self._items, None)


class FakeEmbed:
    class _Vec:
        def tolist(self):
            return [0.1] * 8

    def embed(self, texts):
        # return an iterator of vector-like objects with .tolist()
        for _ in texts:
            yield self._Vec()


@pytest.mark.service
@pytest.mark.asyncio
async def test_context_search_blend_compact(monkeypatch):
    # repo_search returns two code hits (async stub)
    async def fake_repo_search(**kwargs):
        return {
            "results": [
                {"score": 0.8, "path": "/x/a.py", "start_line": 1, "end_line": 3},
                {"score": 0.6, "path": "/x/b.py", "start_line": 5, "end_line": 9},
            ]
        }

    monkeypatch.setattr(srv, "repo_search", fake_repo_search)
    monkeypatch.setattr(srv, "_get_embedding_model", lambda *a, **k: FakeEmbed())

    # Memory fallback via Qdrant: two memory-like points (no path in metadata)
    mem_items = [
        FakePoint(0.9, {"content": "foo note one", "metadata": {}}),
        FakePoint(0.2, {"content": "bar note two", "metadata": {}}),
    ]
    import qdrant_client

    monkeypatch.setattr(
        qdrant_client, "QdrantClient", lambda *a, **k: FakeQdrantMem(mem_items)
    )

    res = await srv.context_search(
        query="foo bar",
        limit=3,
        per_path=1,
        include_memories=True,
        memory_weight=0.5,
        compact=True,
    )

    assert "results" in res
    # Compact shape: code entries have path+lines; memory entries have content only
    for it in res["results"]:
        if it.get("source") == "code":
            assert "path" in it and "start_line" in it and "end_line" in it
        else:
            assert "content" in it and len(it["content"]) > 0


@pytest.mark.service
@pytest.mark.asyncio
async def test_context_search_weight_scaling(monkeypatch):
    # repo_search returns one code hit (async stub)
    async def fake_repo_search(**kwargs):
        return {
            "results": [
                {"score": 0.5, "path": "/x/a.py", "start_line": 1, "end_line": 3}
            ]
        }

    monkeypatch.setattr(srv, "repo_search", fake_repo_search)

    # Force SSE memory path with a fake FastMCP client
    monkeypatch.setenv("MEMORY_SSE_ENABLED", "1")

    class T:
        def __init__(self, name):
            self.name = name

    class Item:
        def __init__(self, text):
            self.text = text

    class Resp:
        def __init__(self):
            self.content = [Item("foo note")]

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def list_tools(self):
            return [T("find")]

        async def call_tool(self, *a, **k):
            return Resp()

    # Import fastmcp inside test to avoid module-level import conflicts
    # Clear any broken mcp modules from sys.modules first
    import sys
    mcp_modules = [k for k in sys.modules.keys() if k == 'mcp' or k.startswith('mcp.')]
    for mod in mcp_modules:
        if mod in sys.modules and not hasattr(sys.modules.get(mod, object()), 'types'):
            del sys.modules[mod]
    import fastmcp
    monkeypatch.setattr(fastmcp, "Client", lambda *a, **k: FakeClient())

    res = await srv.context_search(
        query="foo",
        limit=2,
        per_path=1,
        include_memories=True,
        memory_weight=2.0,
        compact=False,
    )

    mem_scores = [r["score"] for r in res["results"] if r.get("source") == "memory"]
    code_scores = [r["score"] for r in res["results"] if r.get("source") == "code"]
    assert mem_scores, "expected at least one memory result"
    assert code_scores, "expected at least one code result"
    assert max(mem_scores) > max(code_scores)


@pytest.mark.service
@pytest.mark.asyncio
async def test_context_search_per_source_limits(monkeypatch):
    # repo_search returns three code hits
    async def fake_repo_search(**kwargs):
        return {
            "results": [
                {"score": 0.9, "path": "/x/a.py", "start_line": 1, "end_line": 3},
                {"score": 0.8, "path": "/x/b.py", "start_line": 4, "end_line": 6},
                {"score": 0.7, "path": "/x/c.py", "start_line": 7, "end_line": 9},
            ]
        }

    monkeypatch.setattr(srv, "repo_search", fake_repo_search)
    monkeypatch.setattr(srv, "_get_embedding_model", lambda *a, **k: FakeEmbed())

    # Drive memory hits via SSE path with a fake FastMCP client yielding 3 notes
    monkeypatch.setenv("MEMORY_SSE_ENABLED", "1")

    class T:
        def __init__(self, name):
            self.name = name

    class Item:
        def __init__(self, text):
            self.text = text

    class Resp:
        def __init__(self):
            self.content = [Item("m1"), Item("m2"), Item("m3")]

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def list_tools(self):
            return [T("find")]

        async def call_tool(self, *a, **k):
            return Resp()

    # Import fastmcp inside test to avoid module-level import conflicts
    # Clear any broken mcp modules from sys.modules first
    import sys
    mcp_modules = [k for k in sys.modules.keys() if k == 'mcp' or k.startswith('mcp.')]
    for mod in mcp_modules:
        if mod in sys.modules and not hasattr(sys.modules.get(mod, object()), 'types'):
            del sys.modules[mod]
    import fastmcp
    monkeypatch.setattr(fastmcp, "Client", lambda *a, **k: FakeClient())

    res = await srv.context_search(
        query="foo",
        limit=5,
        per_path=1,
        include_memories=True,
        per_source_limits=json.dumps({"code": 1, "memory": 2}),
        compact=True,
    )

    kinds = [r.get("source") for r in res.get("results", [])]
    assert kinds.count("code") <= 1
    assert kinds.count("memory") <= 2
    # Ensure at least one of each (since both sources available)
    assert "code" in kinds and "memory" in kinds
