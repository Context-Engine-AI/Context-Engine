import importlib
import types
import os
import pytest

pytestmark = pytest.mark.unit

def test_main_resolves_collection_from_state(monkeypatch, tmp_path):
    # Env setup: placeholder collection name at startup
    monkeypatch.setenv("WATCH_ROOT", str(tmp_path))
    monkeypatch.setenv("COLLECTION_NAME", "my-collection")
    monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
    monkeypatch.setenv("EMBEDDING_MODEL", "fake")

    # Single-repo mode: watch_index.main should keep COLLECTION_NAME from env.

    wi = importlib.import_module("scripts.watch_index")
    # Reload to re-read env defaults (COLLECTION) in module globals
    wi = importlib.reload(wi)

    # Fake QdrantClient: force get_collection to raise so code chooses sanitized vector name path
    class FakeQdrant:
        def __init__(self, *a, **k):
            pass
        def get_collection(self, *a, **k):
            raise RuntimeError("not available in unit test")
    monkeypatch.setattr(wi, "QdrantClient", FakeQdrant, raising=True)

    # Fake embedding model: just yield a fixed-dim vector on probe
    class FakeModel:
        def __init__(self, model_name: str = "fake"):
            self.model_name = model_name
        def embed(self, texts):
            # Next() will take the first yielded item and len() is the dim
            yield [0.0] * 32

    # Patch the embedder module that watch_index imports from at runtime
    embedder = importlib.import_module("scripts.embedder")
    monkeypatch.setattr(embedder, "get_embedding_model", lambda m: FakeModel(m), raising=True)
    monkeypatch.setattr(embedder, "get_model_dimension", lambda m: 32, raising=True)

    # No-op ensure calls
    idx = importlib.import_module("scripts.ingest_code")
    monkeypatch.setattr(idx, "ensure_collection", lambda *a, **k: None, raising=True)
    monkeypatch.setattr(idx, "ensure_payload_indexes", lambda *a, **k: None, raising=True)

    # Fake Observer to avoid starting real threads
    class FakeObserver:
        def schedule(self, *a, **k):
            pass
        def start(self):
            pass
        def stop(self):
            pass
        def join(self):
            pass
    monkeypatch.setattr(wi, "Observer", FakeObserver, raising=True)

    # Mock collection_health to avoid FS access
    heal_mod = importlib.import_module("scripts.collection_health")
    monkeypatch.setattr(heal_mod, "auto_heal_if_needed", lambda *a, **k: {}, raising=True)
    monkeypatch.setattr(heal_mod, "auto_heal_multi_repo", lambda *a, **k: {}, raising=True)

    # Make the main loop exit immediately by raising KeyboardInterrupt on sleep
    def _raise_kb(_):
        raise KeyboardInterrupt()
    monkeypatch.setattr(wi.time, "sleep", _raise_kb, raising=True)

    # Precondition: module-level COLLECTION should reflect placeholder at import time
    assert wi.COLLECTION == os.environ.get("COLLECTION_NAME") == "my-collection"

    # Run main(); in single-repo mode it should keep the env-provided COLLECTION_NAME
    wi.main()

    # Postcondition: global COLLECTION remains the env-provided name
    assert wi.COLLECTION == "my-collection"


def test_multi_repo_ignores_placeholder_collection_in_state(monkeypatch, tmp_path):
    # Multi-repo mode should not route per-repo events into a workspace-level default
    # collection (e.g. "codebase") even if stale state.json claims that.
    monkeypatch.setenv("MULTI_REPO_MODE", "1")
    monkeypatch.setenv("WATCH_ROOT", str(tmp_path))
    monkeypatch.setenv("COLLECTION_NAME", "codebase")

    utils = importlib.import_module("scripts.watch_index_core.utils")
    utils = importlib.reload(utils)
    monkeypatch.setattr(utils, "ROOT", tmp_path, raising=False)
    monkeypatch.setattr(utils, "is_multi_repo_mode", lambda: True, raising=True)

    repo_slug = "Pirate Survivors-2b23a7e45f2c4b9f"
    repo_path = tmp_path / repo_slug
    repo_path.mkdir(parents=True, exist_ok=True)
    # Provide a file under the repo so repo detection works
    target = repo_path / ".gitignore"
    target.write_text("# test\n")

    # Force state to claim it should use the placeholder collection.
    def _fake_get_workspace_state(ws_path: str, repo_name: str | None = None):
        return {"qdrant_collection": "codebase", "serving_collection": "codebase"}

    monkeypatch.setattr(utils, "get_workspace_state", _fake_get_workspace_state, raising=True)

    # Derivation should win over stale placeholder mapping.
    monkeypatch.setattr(utils, "get_collection_name", lambda rn: f"derived-{rn}", raising=True)
    # Avoid touching real state persistence in unit test
    monkeypatch.setattr(utils, "update_workspace_state", lambda *a, **k: None, raising=True)

    # Sanity: still consider codebase a placeholder/default
    assert utils.default_collection_name() == "codebase"

    resolved = utils._get_collection_for_file(target)
    assert resolved == f"derived-{repo_slug}"

