import importlib
from pathlib import Path
import os
import time
import types
import pytest

wi = importlib.import_module("scripts.watch_index")
idx = importlib.import_module("scripts.ingest_code")


class FakeQueue:
    def __init__(self):
        self.added = []

    def add(self, p: Path):
        self.added.append(str(p))


class FakeClient:
    pass


class E:
    def __init__(self, src, dest=None, is_dir=False):
        self.src_path = str(src)
        self.dest_path = str(dest) if dest else None
        self.is_directory = is_dir


@pytest.mark.unit
def test_on_deleted_calls_delete_points(monkeypatch, tmp_path):
    monkeypatch.setenv("MULTI_REPO_MODE", "0")
    q = FakeQueue()
    handler = wi.IndexHandler(root=tmp_path, queue=q, client=FakeClient(), collection="c")

    called = {}

    def fake_delete(client, collection, p):
        called["args"] = (collection, p)

    monkeypatch.setattr(idx, "delete_points_by_path", fake_delete)

    # Create a code-like file to pass suffix filter; then delete event on it
    f = tmp_path / "a.py"
    f.write_text("print('x')\n")
    handler.on_deleted(E(f))

    assert "args" in called
    assert called["args"][0] == "c"
    assert called["args"][1].endswith("/a.py")


@pytest.mark.unit
def test_on_moved_enqueues_new_dest(monkeypatch, tmp_path):
    monkeypatch.setenv("MULTI_REPO_MODE", "0")
    q = FakeQueue()
    handler = wi.IndexHandler(root=tmp_path, queue=q, client=FakeClient(), collection="c")

    # Ensure .py suffix passes filters
    src = tmp_path / "a.py"
    dst = tmp_path / "b.py"
    src.write_text("print('a')\n")
    dst.write_text("print('b')\n")

    # Monkeypatch excluder to allow all; and code extensions are already .py
    handler.on_moved(E(src, dest=dst))

    # Debounced queue adds; but our FakeQueue records immediately via add()
    assert q.added, "expected destination to be enqueued"
    assert any(s.endswith("/b.py") for s in q.added)


@pytest.mark.unit
def test_ignore_reload_rebuilds_excluder(monkeypatch, tmp_path):
    monkeypatch.setenv("MULTI_REPO_MODE", "0")
    # Place .qdrantignore; construct handler (captures mtime)
    ign = tmp_path / ".qdrantignore"
    ign.write_text("# initial\n")
    q = FakeQueue()
    handler = wi.IndexHandler(root=tmp_path, queue=q, client=FakeClient(), collection="c")
    old = handler.excl

    # Touch ignore file to bump mtime and trigger reload
    time.sleep(0.01)
    ign.write_text("*.gen\n")
    handler._maybe_reload_excluder()

    assert handler.excl is not old, "excluder should be rebuilt after ignore file change"


@pytest.mark.unit
def test_remote_git_manifest_is_enqueued_even_if_excluded(monkeypatch, tmp_path):
    monkeypatch.setenv("MULTI_REPO_MODE", "0")
    q = FakeQueue()
    handler = wi.IndexHandler(root=tmp_path, queue=q, client=FakeClient(), collection="c")

    # .remote-git should be excluded by default excluder rules for code indexing
    assert handler.excl.exclude_dir("/.remote-git")

    manifest_dir = tmp_path / ".remote-git"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest = manifest_dir / "git_history_test.json"
    manifest.write_text("{}\n")

    handler.on_created(E(manifest))
    assert any(p.endswith("/.remote-git/git_history_test.json") for p in q.added)

