import types
import importlib
import pytest

srv = importlib.import_module("scripts.mcp_indexer_server")


class FakeQdrant:
    def __init__(self):
        self._collections = {"test": {"vectors_count": 123, "points_count": 456}}

    def get_collections(self):
        return types.SimpleNamespace(collections=[types.SimpleNamespace(name="test")])

    def get_collection(self, collection_name):
        return types.SimpleNamespace(points_count=456, vectors_count=123)

    def count(self, collection_name, exact=True):
        return types.SimpleNamespace(count=456)

    def scroll(
        self, collection_name, limit, offset=None, with_payload=True, with_vectors=False
    ):
        return ([], None)


@pytest.mark.service
def test_qdrant_status_mocked(monkeypatch):
    # Patch the import site inside the function body by targeting the real module
    import qdrant_client

    monkeypatch.setattr(qdrant_client, "QdrantClient", lambda *a, **k: FakeQdrant())

    out = srv.asyncio.get_event_loop().run_until_complete(
        srv.qdrant_status(collection="test")
    )
    # qdrant_status returns a summary shape without an 'ok' key
    assert out.get("collection") == "test"
    assert "count" in out and "last_ingested_at" in out
