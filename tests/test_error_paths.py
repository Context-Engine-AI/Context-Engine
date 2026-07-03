import os
import asyncio
import types
import pytest

import scripts.mcp_indexer_server as srv


@pytest.mark.service
def test_repo_search_malformed_jsonl_subprocess(monkeypatch):
    # Force subprocess path and simulate malformed JSONL stdout
    monkeypatch.setenv("HYBRID_IN_PROCESS", "0")

    async def fake_run(cmd, **kwargs):
        # Simulate subprocess failure with malformed output
        return {"ok": False, "code": 1, "stdout": "not-json\n", "stderr": "malformed"}

    monkeypatch.setattr(srv, "_run_async", fake_run)

    res = srv.asyncio.get_event_loop().run_until_complete(
        srv.repo_search(queries=["x"], limit=1, compact=False)
    )

    assert res.get("ok") is False
    assert res.get("code", 1) != 0


@pytest.mark.service
def test_repo_search_inproc_qdrant_failure_fallback_and_fail(monkeypatch):
    # In-process hybrid raises (simulating Qdrant connectivity failure),
    # subprocess fallback also fails.
    monkeypatch.setenv("HYBRID_IN_PROCESS", "1")

    # Avoid real model load
    monkeypatch.setattr(srv, "_get_embedding_model", lambda *a, **k: object())

    # Cause in-process path to fail
    import scripts.hybrid_search as hy

    def boom(*a, **k):
        raise ConnectionError("qdrant down")

    monkeypatch.setattr(hy, "run_hybrid_search", boom)

    # And make the subprocess fallback fail too
    async def fake_run(cmd, **kwargs):
        return {"ok": False, "code": 1, "stdout": "", "stderr": "qdrant unreachable"}

    monkeypatch.setattr(srv, "_run_async", fake_run)

    res = srv.asyncio.get_event_loop().run_until_complete(
        srv.repo_search(queries=["x"], limit=1, compact=True)
    )

    assert res.get("ok") is False
    assert res.get("code", 0) != 0
    assert "stderr" in res or res.get("error")
