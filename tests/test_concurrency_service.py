import os
import asyncio
import pytest

import scripts.mcp_indexer_server as srv


@pytest.mark.service
@pytest.mark.asyncio
async def test_repo_search_concurrent(monkeypatch):
    # In-process, fast stubbed hybrid search and model
    monkeypatch.setenv("HYBRID_IN_PROCESS", "1")
    monkeypatch.setattr(srv, "_get_embedding_model", lambda *a, **k: object())

    import scripts.hybrid_search as hy

    def fast_stub(*a, **k):
        # Return tiny deterministic list
        return [
            {"score": 0.5, "path": "/x/a.py", "start_line": 1, "end_line": 3},
            {"score": 0.4, "path": "/x/b.py", "start_line": 5, "end_line": 9},
        ]

    monkeypatch.setattr(hy, "run_hybrid_search", fast_stub)

    async def one():
        return await srv.repo_search(queries=["def"], limit=2, compact=True)

    # Ensure no event-loop blocking under moderate concurrency
    await asyncio.wait_for(asyncio.gather(*[one() for _ in range(25)]), timeout=30)
