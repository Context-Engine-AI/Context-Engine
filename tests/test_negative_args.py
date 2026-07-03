import os
import pytest

import scripts.mcp_indexer_server as srv


@pytest.mark.service
def test_repo_search_conflicting_filters_empty_ok(monkeypatch):
    # In-process, but no results due to conflicting filters (simulate by returning [])
    monkeypatch.setenv("HYBRID_IN_PROCESS", "1")
    monkeypatch.setattr(srv, "_get_embedding_model", lambda *a, **k: object())

    import scripts.hybrid_search as hy

    monkeypatch.setattr(hy, "run_hybrid_search", lambda *a, **k: [])

    res = srv.asyncio.get_event_loop().run_until_complete(
        srv.repo_search(queries=["foo"], limit=3, ext="cpp", compact=True)
    )

    assert res.get("ok") is True
    assert res.get("results") == []
