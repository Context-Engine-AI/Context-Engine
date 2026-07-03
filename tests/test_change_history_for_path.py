import importlib
import sys
import types
import pytest

# Provide a minimal stub for mcp.server.fastmcp.FastMCP so importing the server doesn't exit
mcp_pkg = types.ModuleType("mcp")
server_pkg = types.ModuleType("mcp.server")
fastmcp_pkg = types.ModuleType("mcp.server.fastmcp")

class _FastMCP:
    def __init__(self, *args, **kwargs):
        pass
    def tool(self, *args, **kwargs):
        def _decorator(fn):
            return fn
        return _decorator

class _Context:
    def __init__(self, *args, **kwargs):
        # Tests only access .session when present; keep permissive defaults
        self.session = kwargs.get("session")

setattr(fastmcp_pkg, "FastMCP", _FastMCP)
setattr(fastmcp_pkg, "Context", _Context)
sys.modules.setdefault("mcp", mcp_pkg)
sys.modules.setdefault("mcp.server", server_pkg)
sys.modules.setdefault("mcp.server.fastmcp", fastmcp_pkg)

srv = importlib.import_module("scripts.mcp_indexer_server")


class FakePoint:
    def __init__(self, payload):
        self.payload = payload


class FakeQdrant:
    def __init__(self, pages):
        # pages: list[list[FakePoint]] to simulate pagination
        self._pages = pages
        self._i = 0

    def scroll(self, **kwargs):
        if self._i >= len(self._pages):
            return ([], None)
        page = self._pages[self._i]
        self._i += 1
        # Return next_page offset as None to stop after last page
        return (page, None if self._i >= len(self._pages) else object())


@pytest.mark.service
@pytest.mark.anyio
async def test_change_history_strict_match_under_work(monkeypatch):
    import qdrant_client

    pts = [
        FakePoint({
            "metadata": {
                "path": "/work/a.py",
                "file_hash": "h1",
                "last_modified_at": 100,
                "ingested_at": 90,
                "churn_count": 2,
            }
        }),
        FakePoint({
            "metadata": {
                "path": "/work/a.py",
                "file_hash": "h2",
                "last_modified_at": 120,
                "ingested_at": 110,
                "churn_count": 3,
            }
        }),
        FakePoint({
            "metadata": {
                "path": "/work/a.py",
                "file_hash": "h1",
                "last_modified_at": 130,
                "ingested_at": 115,
                "churn_count": 5,
            }
        }),
    ]

    monkeypatch.setattr(qdrant_client, "QdrantClient", lambda *a, **k: FakeQdrant([pts]))

    res = await srv.change_history_for_path(path="/work/a.py", max_points=100)
    assert res.get("ok") is True
    summary = res.get("summary") or {}
    assert summary.get("path") == "/work/a.py"
    assert summary.get("points_scanned") == 3
    assert summary.get("distinct_hashes") == 2  # h1,h2
    assert summary.get("last_modified_min") == 100
    assert summary.get("last_modified_max") == 130
    assert summary.get("ingested_min") == 90
    assert summary.get("ingested_max") == 115
    assert summary.get("churn_count_max") == 5

