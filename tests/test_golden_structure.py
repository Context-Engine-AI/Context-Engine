import os
import json
from pathlib import Path
import pytest

import scripts.mcp_indexer_server as srv


@pytest.mark.service
def test_repo_search_compact_golden_subset(monkeypatch):
    # Use in-process with a deterministic stub
    monkeypatch.setenv("HYBRID_IN_PROCESS", "1")
    monkeypatch.setattr(srv, "_get_embedding_model", lambda *a, **k: object())

    import scripts.hybrid_search as hy

    def stub(*a, **k):
        return [
            {"score": 0.91, "path": "/x/a.py", "start_line": 1, "end_line": 3},
            {"score": 0.42, "path": "/x/b.py", "start_line": 10, "end_line": 20},
        ]

    monkeypatch.setattr(hy, "run_hybrid_search", stub)

    res = srv.asyncio.get_event_loop().run_until_complete(
        srv.repo_search(queries=["q"], limit=2, compact=True)
    )

    # Normalize subset: path/start_line/end_line/symbol only
    got = [
        {
            "path": r.get("path"),
            "start_line": r.get("start_line"),
            "end_line": r.get("end_line"),
        }
        for r in res.get("results", [])
    ]

    golden_path = Path(__file__).parent / "data" / "golden_compact.json"
    with golden_path.open("r", encoding="utf-8") as f:
        want = json.load(f)

    assert got == want
