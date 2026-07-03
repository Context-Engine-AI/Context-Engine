import importlib
import types
import os
import pytest

srv = importlib.import_module("scripts.mcp_indexer_server")


@pytest.mark.service
def test_rerank_timeout_floor_and_env_defaults(monkeypatch):
    # Force rerank via env default when arg not provided
    monkeypatch.setenv("RERANKER_ENABLED", "1")
    monkeypatch.setenv("RERANK_IN_PROCESS", "0")

    # Floor 1500ms; client asks 200ms -> effective >= 1500ms -> 1.5s
    monkeypatch.setenv("RERANK_TIMEOUT_FLOOR_MS", "1500")
    # Fix default timeout for test determinism (CI may set a higher value)
    monkeypatch.setenv("RERANKER_TIMEOUT_MS", "200")

    # Fake _run_async to capture calls
    calls = []

    async def fake_run(cmd, env=None, timeout=None):
        calls.append({"cmd": cmd, "timeout": timeout})
        # Distinguish hybrid vs rerank by script name
        if any("rerank_local.py" in str(x) for x in cmd):
            # Return something that looks like rerank stdout
            return {
                "ok": True,
                "stdout": "0.9\t/path.py\t\t1-3\n",
                "stderr": "",
                "code": 0,
            }
        else:
            # Hybrid JSONL minimal
            return {
                "ok": True,
                "stdout": '{"score": 0.1, "path": "/p", "start_line": 1, "end_line": 2}\n',
                "stderr": "",
                "code": 0,
            }

    monkeypatch.setattr(srv, "_run_async", fake_run)

    # Call repo_search with no rerank_enabled arg to pick env default
    res = srv.asyncio.get_event_loop().run_until_complete(
        srv.repo_search(query="foo", limit=3, per_path=1)
    )

    assert any(
        "rerank_local.py" in " ".join(map(str, c["cmd"])) for c in calls
    ), "rerank subprocess should be invoked"
    # find rerank call
    rc = next(c for c in calls if any("rerank_local.py" in str(x) for x in c["cmd"]))
    assert rc["timeout"] >= 1.5 and rc["timeout"] <= 2.0
    assert res["used_rerank"] is True
