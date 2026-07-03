import importlib

import pytest
from qdrant_client import models


ih = importlib.import_module("scripts.ingest_history")


class FakeClient:
    def __init__(self):
        self.calls = []

    def delete(self, **kwargs):
        self.calls.append(kwargs)


def _dump(obj):
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    return obj


def _assert_has_match(conds, *, key: str, value: str) -> None:
    for c in conds:
        cd = _dump(c)
        if cd.get("key") != key:
            continue
        match = cd.get("match") or {}
        if isinstance(match, dict) and match.get("value") == value:
            return
    raise AssertionError(f"missing condition key={key} value={value}; got={conds}")


@pytest.mark.unit
def test_prune_skipped_on_delta(monkeypatch):
    client = FakeClient()
    monkeypatch.setenv("GIT_HISTORY_PRUNE", "1")
    monkeypatch.setattr(ih, "REPO_NAME", "repo")
    monkeypatch.setattr(ih, "COLLECTION", "coll")

    ih._prune_old_commit_points(client, "run-1", mode="delta")
    assert client.calls == []


@pytest.mark.unit
def test_prune_skipped_when_disabled(monkeypatch):
    client = FakeClient()
    monkeypatch.setenv("GIT_HISTORY_PRUNE", "0")
    monkeypatch.setattr(ih, "REPO_NAME", "repo")
    monkeypatch.setattr(ih, "COLLECTION", "coll")

    ih._prune_old_commit_points(client, "run-1", mode="snapshot")
    assert client.calls == []


@pytest.mark.unit
def test_prune_called_on_snapshot(monkeypatch):
    client = FakeClient()
    monkeypatch.setenv("GIT_HISTORY_PRUNE", "1")
    monkeypatch.setattr(ih, "REPO_NAME", "repo")
    monkeypatch.setattr(ih, "COLLECTION", "coll")

    run_id = "git_history_test_run"
    ih._prune_old_commit_points(client, run_id, mode="snapshot")

    assert len(client.calls) == 1
    call = client.calls[0]

    assert call.get("collection_name") == "coll"
    assert call.get("wait") is True

    selector = call.get("points_selector")
    assert isinstance(selector, models.FilterSelector)

    flt = selector.filter
    assert isinstance(flt, models.Filter)

    flt_d = _dump(flt)
    must = flt_d.get("must") or []
    must_not = flt_d.get("must_not") or []

    _assert_has_match(must, key="metadata.kind", value="git_message")
    _assert_has_match(must, key="metadata.repo", value="repo")
    _assert_has_match(must_not, key="metadata.git_history_run_id", value=run_id)


@pytest.mark.unit
def test_prune_swallow_delete_errors(monkeypatch):
    class BoomClient:
        def delete(self, **kwargs):
            raise RuntimeError("boom")

    monkeypatch.setenv("GIT_HISTORY_PRUNE", "1")
    monkeypatch.setattr(ih, "REPO_NAME", "repo")
    monkeypatch.setattr(ih, "COLLECTION", "coll")

    ih._prune_old_commit_points(BoomClient(), "run-1", mode="snapshot")
