import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def staging_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # Make workspace_state treat this temp dir as the workspace root.
    monkeypatch.setenv("MULTI_REPO_MODE", "1")
    monkeypatch.setenv("WORKSPACE_PATH", str(tmp_path))
    monkeypatch.setenv("WATCH_ROOT", str(tmp_path))
    monkeypatch.setenv("WORK_DIR", str(tmp_path))
    monkeypatch.setenv("WORKDIR", str(tmp_path))
    monkeypatch.setenv("CTXCE_STAGING_ENABLED", "1")

    # Avoid any accidental external calls.
    monkeypatch.setenv("HF_HUB_DISABLE_XET", "1")
    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "false")

    repo_name = "repo1"
    collection = "coll1"

    repo_ws = tmp_path / repo_name
    repo_ws.mkdir(parents=True)
    (repo_ws / "hello.txt").write_text("hello", encoding="utf-8")

    return {
        "root": tmp_path,
        "repo_name": repo_name,
        "collection": collection,
        "repo_ws": repo_ws,
    }


def _read_repo_state(root: Path, repo_name: str) -> dict:
    p = root / ".codebase" / "repos" / repo_name / "state.json"
    return json.loads(p.read_text(encoding="utf-8"))


def _repo_state_path(root: Path, repo_name: str) -> Path:
    return root / ".codebase" / "repos" / repo_name / "state.json"


def _seed_repo_state(
    workspace_state_module,
    *,
    repo_ws: Path,
    repo_name: str,
    collection: str,
    active_env: Optional[Dict[str, str]] = None,
    pending_env: Optional[Dict[str, str]] = None,
) -> Dict:
    active_env = active_env or {"CTXCE_E2E_MARKER": "active"}
    pending_env = pending_env or {"CTXCE_E2E_MARKER": "pending"}
    workspace_state_module.update_workspace_state(
        workspace_path=str(repo_ws),
        repo_name=repo_name,
        updates={
            "qdrant_collection": collection,
            "serving_collection": collection,
            "serving_repo_slug": repo_name,
            "active_repo_slug": repo_name,
            "indexing_env": active_env,
            "indexing_env_pending": pending_env,
        },
    )
    return {"active": active_env, "pending": pending_env}


def test_staging_start_promote_activate_and_abort_are_consistent(
    staging_workspace: dict, monkeypatch: pytest.MonkeyPatch
):
    from scripts import indexing_admin
    from scripts import workspace_state

    root: Path = staging_workspace["root"]
    repo_name: str = staging_workspace["repo_name"]
    collection: str = staging_workspace["collection"]
    repo_ws: Path = staging_workspace["repo_ws"]

    envs = _seed_repo_state(
        workspace_state_module=workspace_state,
        repo_ws=repo_ws,
        repo_name=repo_name,
        collection=collection,
        active_env={"CTXCE_ENV_ROLE": "active", "CTXCE_E2E_MARKER": "active"},
        pending_env={"CTXCE_ENV_ROLE": "pending", "CTXCE_E2E_MARKER": "pending"},
    )
    pending_env = envs["pending"]
    # Seed cache.json so staging clone should copy + retarget paths.
    src_cache_dir = root / ".codebase" / "repos" / repo_name
    src_cache_dir.mkdir(parents=True, exist_ok=True)
    original_path = (repo_ws / "hello.txt").resolve().as_posix()
    (src_cache_dir / "cache.json").write_text(
        json.dumps(
            {
                "file_hashes": {
                    original_path: {"hash": "abc123", "size": 5, "mtime": 123456},
                }
            }
        ),
        encoding="utf-8",
    )

    # Patch boundaries (no real Qdrant, no subprocess indexing).
    calls = {
        "copy": [],
        "recreate": [],
        "spawn": [],
        "delete": [],
        "wait_clone": 0,
        "normalize_schema": 0,
    }

    def fake_mapping_index(*, work_dir: str):
        return {
            collection: [
                {
                    "repo_name": repo_name,
                    "container_path": str(repo_ws),
                }
            ]
        }

    monkeypatch.setattr(indexing_admin, "collection_mapping_index", fake_mapping_index)
    monkeypatch.setattr(indexing_admin, "_get_collection_point_count", lambda **_: 0)

    def fake_copy_collection_qdrant(**kwargs):
        calls["copy"].append(kwargs)
        return kwargs.get("target")

    def fake_recreate_collection_qdrant(**kwargs):
        calls["recreate"].append(kwargs)

    def fake_spawn_ingest_code(**kwargs):
        calls["spawn"].append(kwargs)

    def fake_delete_collection_qdrant(**kwargs):
        calls["delete"].append(kwargs)

    def fake_wait_for_clone_points(**kwargs):
        calls["wait_clone"] += 1

    def fake_normalize_cloned_collection_schema(**kwargs):
        calls["normalize_schema"] += 1

    monkeypatch.setattr(indexing_admin, "copy_collection_qdrant", fake_copy_collection_qdrant)
    monkeypatch.setattr(indexing_admin, "recreate_collection_qdrant", fake_recreate_collection_qdrant)
    monkeypatch.setattr(indexing_admin, "spawn_ingest_code", fake_spawn_ingest_code)
    monkeypatch.setattr(indexing_admin, "delete_collection_qdrant", fake_delete_collection_qdrant)
    monkeypatch.setattr(indexing_admin, "_wait_for_clone_points", fake_wait_for_clone_points)
    monkeypatch.setattr(
        indexing_admin, "_normalize_cloned_collection_schema", fake_normalize_cloned_collection_schema
    )

    # === START staging ===
    old_collection = f"{collection}_old"
    indexing_admin.start_staging_rebuild(collection=collection, work_dir=str(root))

    # Contract: copy collection called for primary -> _old.
    assert calls["copy"], "Expected copy_collection_qdrant to be called"
    assert calls["copy"][0]["source"] == collection
    assert calls["copy"][0]["target"] == old_collection

    # Contract: workspace clone dir exists.
    assert (root / f"{repo_name}_old").exists()

    # Contract: staging metadata present and serving switched to _old.
    st = _read_repo_state(root, repo_name)
    assert st.get("serving_collection") == old_collection
    assert st.get("qdrant_collection") == collection
    assert isinstance(st.get("staging"), dict)
    assert st["staging"].get("collection") == old_collection
    assert st["staging"].get("environment") == pending_env
    assert (st.get("indexing_env") or {}) == envs["active"]
    assert (st.get("indexing_env_pending") or {}) == envs["pending"]

    # Contract: spawn_ingest_code called with env_overrides matching pending env and forced collection.
    assert calls["spawn"], "Expected spawn_ingest_code to be called"
    spawn = calls["spawn"][0]
    assert spawn["collection"] == collection
    assert spawn["repo_name"] == repo_name
    assert spawn["env_overrides"] == pending_env
    assert calls["recreate"], "Expected recreate_collection_qdrant to be invoked"
    assert calls["recreate"][0]["collection"] == collection

    # Contract: _old repo state exists and preserves serving env (pending cleared).
    old_state = _read_repo_state(root, f"{repo_name}_old")
    assert old_state.get("qdrant_collection") == old_collection
    assert old_state.get("serving_collection") == old_collection
    assert old_state.get("indexing_env_pending") in (None, {})
    assert (old_state.get("indexing_env") or {}) == envs["active"]
    # Cached file hashes should be copied and retargeted to the cloned workspace.
    clone_cache_path = root / ".codebase" / "repos" / f"{repo_name}_old" / "cache.json"
    assert clone_cache_path.exists(), "expected cloned cache.json"
    clone_cache = json.loads(clone_cache_path.read_text(encoding="utf-8"))
    clone_path = (root / f"{repo_name}_old" / "hello.txt").resolve().as_posix()
    assert clone_path in clone_cache.get("file_hashes", {}), "clone cache missing retargeted path"
    assert original_path not in clone_cache.get("file_hashes", {}), "original path leaked into clone cache"

    # === PROMOTE pending env ===
    workspace_state.promote_pending_indexing_config(workspace_path=str(repo_ws), repo_name=repo_name)
    st = _read_repo_state(root, repo_name)
    assert (st.get("indexing_env") or {}).get("CTXCE_E2E_MARKER") == "pending"
    assert st.get("indexing_env_pending") in (None, {})

    # === ACTIVATE ===
    indexing_admin.activate_staging_rebuild(collection=collection, work_dir=str(root))
    st = _read_repo_state(root, repo_name)
    assert st.get("staging") is None
    assert st.get("serving_collection") == collection
    assert st.get("serving_repo_slug") == repo_name

    # Cleanup: old workspace + old repo meta removed.
    assert not (root / f"{repo_name}_old").exists()
    assert not (root / ".codebase" / "repos" / f"{repo_name}_old").exists()

    # Contract: _old collection delete invoked.
    assert any(d.get("collection") == old_collection for d in calls["delete"])

    # Idempotent activate
    indexing_admin.activate_staging_rebuild(collection=collection, work_dir=str(root))

    # === ABORT scenario ===
    # Recreate old artifacts to ensure abort cleans them.
    workspace_state.update_workspace_state(
        workspace_path=str(repo_ws),
        repo_name=repo_name,
        updates={
            "indexing_env_pending": {"CTXCE_E2E_MARKER": "pending2"},
        },
    )

    indexing_admin.start_staging_rebuild(collection=collection, work_dir=str(root))
    assert (root / f"{repo_name}_old").exists()
    assert (root / ".codebase" / "repos" / f"{repo_name}_old").exists()

    indexing_admin.abort_staging_rebuild(collection=collection, work_dir=str(root), delete_collection=True)

    # Abort restores serving to primary, clears staging, and cleans up.
    st = _read_repo_state(root, repo_name)
    assert st.get("staging") is None
    assert st.get("serving_collection") == collection
    assert st.get("serving_repo_slug") == repo_name
    assert not (root / f"{repo_name}_old").exists()
    assert not (root / ".codebase" / "repos" / f"{repo_name}_old").exists()

    # Idempotent abort
    indexing_admin.abort_staging_rebuild(collection=collection, work_dir=str(root), delete_collection=True)


def test_update_workspace_state_allows_repo_state_updates_without_workspace_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from scripts import workspace_state

    monkeypatch.delenv("WORKSPACE_PATH", raising=False)
    monkeypatch.delenv("WATCH_ROOT", raising=False)
    monkeypatch.delenv("WORK_DIR", raising=False)
    monkeypatch.delenv("WORKDIR", raising=False)
    monkeypatch.setenv("MULTI_REPO_MODE", "1")
    monkeypatch.setenv("WORKSPACE_PATH", str(tmp_path))
    monkeypatch.setenv("WATCH_ROOT", str(tmp_path))

    repo_name = "missing_ws_repo"
    repo_state_dir = tmp_path / ".codebase" / "repos" / repo_name
    repo_state_dir.mkdir(parents=True)

    # Ensure the repo workspace dir does NOT exist.
    assert not (tmp_path / repo_name).exists()

    # Should still write state because state dir exists.
    workspace_state.update_workspace_state(
        workspace_path=str(tmp_path),
        repo_name=repo_name,
        updates={"indexing_env_pending": {"CTXCE_E2E_MARKER": "ok"}},
    )

    assert _repo_state_path(tmp_path, repo_name).exists()
    st = _read_repo_state(tmp_path, repo_name)
    assert (st.get("indexing_env_pending") or {}).get("CTXCE_E2E_MARKER") == "ok"


def test_cleanup_handles_read_only_dirs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    from scripts import indexing_admin

    deleted = []

    def fake_chmod(path, mode):
        deleted.append(("chmod", path, mode))
        original_chmod(path, mode)

    target_dir = tmp_path / "repo_old"
    target_dir.mkdir()
    (target_dir / "file.txt").write_text("hello", encoding="utf-8")

    # Patch shutil.rmtree to fail once, then succeed after chmod.
    call_count = {"attempt": 0}

    def patched_rmtree(path):
        if call_count["attempt"] == 0:
            call_count["attempt"] += 1
            deleted.append(("rmtree_fail", path))
            raise PermissionError("nope")
        deleted.append(("rmtree_success", path))
        return original_rmtree(path)

    original_rmtree = indexing_admin.shutil.rmtree
    original_chmod = indexing_admin.os.chmod
    monkeypatch.setattr(indexing_admin.shutil, "rmtree", patched_rmtree)
    monkeypatch.setattr(indexing_admin.os, "chmod", fake_chmod)

    result = indexing_admin._delete_path_tree(target_dir)
    assert result is True
    assert not target_dir.exists()
    # Ensure chmod was attempted before the second rmtree.
    assert any(entry[0] == "chmod" for entry in deleted)


def test_start_recreates_primary_before_spawn(staging_workspace: dict, monkeypatch: pytest.MonkeyPatch):
    from scripts import indexing_admin
    from scripts import workspace_state

    root: Path = staging_workspace["root"]
    repo_name: str = staging_workspace["repo_name"]
    collection: str = staging_workspace["collection"]
    repo_ws: Path = staging_workspace["repo_ws"]

    _seed_repo_state(
        workspace_state_module=workspace_state,
        repo_ws=repo_ws,
        repo_name=repo_name,
        collection=collection,
        active_env={"CTXCE_ENV_ROLE": "active"},
        pending_env={"CTXCE_ENV_ROLE": "pending"},
    )

    events: list[str] = []

    monkeypatch.setattr(
        indexing_admin,
        "collection_mapping_index",
        lambda *, work_dir: {
            collection: [{"repo_name": repo_name, "container_path": str(repo_ws)}]
        },
    )
    monkeypatch.setattr(indexing_admin, "_get_collection_point_count", lambda **_: 0)
    monkeypatch.setattr(indexing_admin, "_wait_for_clone_points", lambda **_: events.append("wait"))
    monkeypatch.setattr(indexing_admin, "_normalize_cloned_collection_schema", lambda **_: events.append("normalize"))
    monkeypatch.setattr(
        indexing_admin,
        "copy_collection_qdrant",
        lambda **kwargs: events.append("copy"),
    )

    def fake_recreate(**kwargs):
        events.append("recreate")

    def fake_spawn(**kwargs):
        events.append("spawn")

    monkeypatch.setattr(indexing_admin, "recreate_collection_qdrant", fake_recreate)
    monkeypatch.setattr(indexing_admin, "spawn_ingest_code", fake_spawn)

    indexing_admin.start_staging_rebuild(collection=collection, work_dir=str(root))
    assert "recreate" in events and "spawn" in events
    assert events.index("recreate") < events.index("spawn")


def test_start_handles_copy_failure_without_mutation(staging_workspace: dict, monkeypatch: pytest.MonkeyPatch):
    from scripts import indexing_admin
    from scripts import workspace_state

    root: Path = staging_workspace["root"]
    repo_name: str = staging_workspace["repo_name"]
    collection: str = staging_workspace["collection"]
    repo_ws: Path = staging_workspace["repo_ws"]

    _seed_repo_state(
        workspace_state_module=workspace_state,
        repo_ws=repo_ws,
        repo_name=repo_name,
        collection=collection,
    )

    initial_state = _read_repo_state(root, repo_name)

    monkeypatch.setattr(
        indexing_admin,
        "collection_mapping_index",
        lambda *, work_dir: {
            collection: [{"repo_name": repo_name, "container_path": str(repo_ws)}]
        },
    )
    monkeypatch.setattr(indexing_admin, "_get_collection_point_count", lambda **_: 0)

    def fake_copy(**kwargs):
        raise RuntimeError("copy failed")

    monkeypatch.setattr(indexing_admin, "copy_collection_qdrant", fake_copy)

    with pytest.raises(RuntimeError, match="copy failed"):
        indexing_admin.start_staging_rebuild(collection=collection, work_dir=str(root))

    # State should remain unchanged.
    assert _read_repo_state(root, repo_name) == initial_state
    # No *_old artifacts should exist.
    assert not (root / f"{repo_name}_old").exists()
    assert not (root / ".codebase" / "repos" / f"{repo_name}_old").exists()


def test_start_handles_clone_wait_failure_without_mutation(staging_workspace: dict, monkeypatch: pytest.MonkeyPatch):
    from scripts import indexing_admin
    from scripts import workspace_state

    root: Path = staging_workspace["root"]
    repo_name: str = staging_workspace["repo_name"]
    collection: str = staging_workspace["collection"]
    repo_ws: Path = staging_workspace["repo_ws"]

    _seed_repo_state(
        workspace_state_module=workspace_state,
        repo_ws=repo_ws,
        repo_name=repo_name,
        collection=collection,
    )

    initial_state = _read_repo_state(root, repo_name)

    monkeypatch.setattr(
        indexing_admin,
        "collection_mapping_index",
        lambda *, work_dir: {
            collection: [{"repo_name": repo_name, "container_path": str(repo_ws)}]
        },
    )
    monkeypatch.setattr(indexing_admin, "_get_collection_point_count", lambda **_: 123)
    monkeypatch.setattr(indexing_admin, "copy_collection_qdrant", lambda **kwargs: None)

    def fake_wait(**kwargs):
        raise RuntimeError("wait failed")

    monkeypatch.setattr(indexing_admin, "_wait_for_clone_points", fake_wait)

    with pytest.raises(RuntimeError, match="wait failed"):
        indexing_admin.start_staging_rebuild(collection=collection, work_dir=str(root))

    assert _read_repo_state(root, repo_name) == initial_state
    assert not (root / f"{repo_name}_old").exists()
    assert not (root / ".codebase" / "repos" / f"{repo_name}_old").exists()


def test_start_handles_spawn_failure_but_leaves_staging_state(staging_workspace: dict, monkeypatch: pytest.MonkeyPatch):
    from scripts import indexing_admin
    from scripts import workspace_state

    root: Path = staging_workspace["root"]
    repo_name: str = staging_workspace["repo_name"]
    collection: str = staging_workspace["collection"]
    repo_ws: Path = staging_workspace["repo_ws"]

    _seed_repo_state(
        workspace_state_module=workspace_state,
        repo_ws=repo_ws,
        repo_name=repo_name,
        collection=collection,
    )

    monkeypatch.setattr(
        indexing_admin,
        "collection_mapping_index",
        lambda *, work_dir: {
            collection: [{"repo_name": repo_name, "container_path": str(repo_ws)}]
        },
    )
    monkeypatch.setattr(indexing_admin, "_get_collection_point_count", lambda **_: 0)
    monkeypatch.setattr(indexing_admin, "copy_collection_qdrant", lambda **kwargs: None)
    monkeypatch.setattr(indexing_admin, "_wait_for_clone_points", lambda **kwargs: None)
    monkeypatch.setattr(indexing_admin, "_normalize_cloned_collection_schema", lambda **kwargs: None)
    monkeypatch.setattr(indexing_admin, "recreate_collection_qdrant", lambda **kwargs: None)

    def failing_spawn(**kwargs):
        raise RuntimeError("spawn failed")

    monkeypatch.setattr(indexing_admin, "spawn_ingest_code", failing_spawn)

    with pytest.raises(RuntimeError, match="spawn failed"):
        indexing_admin.start_staging_rebuild(collection=collection, work_dir=str(root))

    st = _read_repo_state(root, repo_name)
    assert (root / f"{repo_name}_old").exists()
    assert (root / ".codebase" / "repos" / f"{repo_name}_old").exists()
    assert st.get("serving_collection") == f"{collection}_old"
    assert isinstance(st.get("staging"), dict)


def test_admin_staging_endpoints_exercise_http_layer(monkeypatch: pytest.MonkeyPatch):
    from scripts import upload_service

    calls = {"start": 0, "activate": 0, "abort": 0}

    monkeypatch.setattr(upload_service, "AUTH_ENABLED", True)
    monkeypatch.setattr(upload_service, "_require_admin_session", lambda request: {"user_id": "admin"})
    monkeypatch.setattr(upload_service, "is_staging_enabled", lambda: True)
    monkeypatch.setattr(upload_service, "WORK_DIR", "/fake/work")
    monkeypatch.setenv("WORK_DIR", "/fake/work")
    monkeypatch.setenv("CTXCE_STAGING_ENABLED", "1")

    def fake_start(**kwargs):
        calls["start"] += 1
        return f"{kwargs['collection']}_old"

    def fake_activate(**kwargs):
        calls["activate"] += 1

    def fake_abort(**kwargs):
        calls["abort"] += 1

    monkeypatch.setattr(upload_service, "start_staging_rebuild", fake_start)
    monkeypatch.setattr(upload_service, "activate_staging_rebuild", fake_activate)
    monkeypatch.setattr(upload_service, "abort_staging_rebuild", fake_abort)
    monkeypatch.setattr(
        upload_service,
        "resolve_collection_root",
        lambda **kwargs: ("/fake/root", "repo1"),
    )

    client = TestClient(upload_service.app)

    resp = client.post("/admin/staging/start", data={"collection": "coll1"}, follow_redirects=False)
    assert resp.status_code == 302
    assert calls["start"] == 1

    resp = client.post("/admin/staging/activate", data={"collection": "coll1"}, follow_redirects=False)
    assert resp.status_code == 302
    assert calls["activate"] == 1

    resp = client.post("/admin/staging/abort", data={"collection": "coll1"}, follow_redirects=False)
    assert resp.status_code == 302
    assert calls["abort"] == 1


def test_watcher_collection_resolution_prefers_serving_state_when_staging_enabled(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    from scripts.watch_index_core import utils as watch_utils

    repo_path = tmp_path / "repo1"
    repo_path.mkdir()

    monkeypatch.setenv("MULTI_REPO_MODE", "1")
    monkeypatch.setenv("WORKSPACE_PATH", str(tmp_path))
    monkeypatch.setenv("WATCH_ROOT", str(tmp_path))

    monkeypatch.setattr(watch_utils, "_extract_repo_name_from_path", lambda *_: "repo1")
    monkeypatch.setattr(watch_utils, "is_multi_repo_mode", lambda: True)
    monkeypatch.setattr(
        watch_utils,
        "get_workspace_state",
        lambda workspace_path, repo_name: {"serving_collection": "coll1_old"},
    )

    coll = watch_utils._get_collection_for_repo(repo_path)
    assert coll == "coll1_old"


def test_watcher_collection_resolution_falls_back_to_env_when_staging_disabled(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    from scripts.watch_index_core import utils as watch_utils

    repo_path = tmp_path / "repo1"
    repo_path.mkdir()

    monkeypatch.setenv("MULTI_REPO_MODE", "0")
    monkeypatch.setenv("COLLECTION_NAME", "env-coll")

    monkeypatch.setattr(watch_utils, "_extract_repo_name_from_path", lambda *_: "repo1")
    monkeypatch.setattr(watch_utils, "is_multi_repo_mode", lambda: False)

    def _fail_get_collection(repo_name: str) -> str:
        raise RuntimeError("no mapping")

    monkeypatch.setattr(watch_utils, "get_collection_name", _fail_get_collection)

    coll = watch_utils._get_collection_for_repo(repo_path)
    assert coll == "env-coll"


def test_spawn_ingest_code_applies_env_overrides_for_staging(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    from scripts import indexing_admin

    monkeypatch.setenv("BASE_ONLY", "system")

    captured: Dict[str, Any] = {}

    def fake_popen(cmd, env):
        captured["cmd"] = cmd
        captured["env"] = env
        class _Proc:
            pass
        return _Proc()

    monkeypatch.setattr(indexing_admin.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(indexing_admin, "update_indexing_status", lambda **_: None)

    overrides = {"PENDING_KEY": "pending", "BASE_ONLY": "override"}

    indexing_admin.spawn_ingest_code(
        root=str(tmp_path / "repo1"),
        work_dir=str(tmp_path),
        collection="primary-coll",
        recreate=True,
        repo_name="repo1",
        env_overrides=overrides,
    )

    env = captured["env"]
    assert env["PENDING_KEY"] == "pending"
    assert env["BASE_ONLY"] == "override"
    assert env["COLLECTION_NAME"] == "primary-coll"
    assert env["CTXCE_FORCE_COLLECTION_NAME"] == "1"
    assert env["WATCH_ROOT"] == str(tmp_path)
    assert env["WORKSPACE_PATH"] == str(tmp_path)


def test_spawn_ingest_code_without_overrides_uses_system_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    from scripts import indexing_admin

    monkeypatch.setenv("BASE_ONLY", "system")

    captured: Dict[str, Any] = {}

    def fake_popen(cmd, env):
        captured["env"] = env
        class _Proc:
            pass
        return _Proc()

    monkeypatch.setattr(indexing_admin.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(indexing_admin, "update_indexing_status", lambda **_: None)

    indexing_admin.spawn_ingest_code(
        root=str(tmp_path / "repo1"),
        work_dir=str(tmp_path),
        collection="primary-coll",
        recreate=False,
        repo_name="repo1",
        env_overrides=None,
    )

    env = captured["env"]
    assert env["BASE_ONLY"] == "system"
    assert env["COLLECTION_NAME"] == "primary-coll"
    assert "CTXCE_FORCE_COLLECTION_NAME" not in env


def test_promote_pending_env_without_pending_config(staging_workspace: dict):
    from scripts import workspace_state

    root: Path = staging_workspace["root"]
    repo_name: str = staging_workspace["repo_name"]
    repo_ws: Path = staging_workspace["repo_ws"]

    workspace_state.update_workspace_state(
        workspace_path=str(repo_ws),
        repo_name=repo_name,
        updates={
            "indexing_env": {"ROLE": "active"},
            "indexing_env_pending": {"ROLE": "pending"},
            "indexing_config": {"chunk": "current"},
            "indexing_config_pending": None,
        },
    )

    workspace_state.promote_pending_indexing_config(workspace_path=str(repo_ws), repo_name=repo_name)

    st = _read_repo_state(root, repo_name)
    assert st.get("indexing_env") == {"ROLE": "pending"}
    assert st.get("indexing_config") == {"chunk": "current"}
    assert st.get("indexing_env_pending") in (None, {})


def test_promote_pending_config_without_pending_env(staging_workspace: dict):
    from scripts import workspace_state

    root: Path = staging_workspace["root"]
    repo_name: str = staging_workspace["repo_name"]
    repo_ws: Path = staging_workspace["repo_ws"]

    workspace_state.update_workspace_state(
        workspace_path=str(repo_ws),
        repo_name=repo_name,
        updates={
            "indexing_env": {"ROLE": "active"},
            "indexing_env_pending": None,
            "indexing_config": {"chunk": "current"},
            "indexing_config_pending": {"chunk": "pending"},
        },
    )

    workspace_state.promote_pending_indexing_config(workspace_path=str(repo_ws), repo_name=repo_name)

    st = _read_repo_state(root, repo_name)
    assert st.get("indexing_env") == {"ROLE": "active"}
    assert st.get("indexing_config") == {"chunk": "pending"}
    assert st.get("indexing_config_pending") in (None, {})


def test_resolve_codebase_root_prefers_env_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    from scripts import indexing_admin

    work_root = tmp_path / "work" / "repo"
    work_root.mkdir(parents=True)
    env_root = tmp_path / "codebase_home"
    (env_root / ".codebase" / "repos").mkdir(parents=True)

    monkeypatch.setenv("CTXCE_CODEBASE_ROOT", str(env_root))

    resolved = indexing_admin._resolve_codebase_root(work_root)
    assert resolved == env_root


def test_resolve_codebase_root_fallbacks_to_parent(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    from scripts import indexing_admin

    parent = tmp_path / "workspace"
    repo = parent / "repo"
    repo.mkdir(parents=True)
    codebase_root = parent
    codebase_dir = codebase_root / ".codebase" / "repos"
    codebase_dir.mkdir(parents=True)

    monkeypatch.delenv("CTXCE_CODEBASE_ROOT", raising=False)

    resolved = indexing_admin._resolve_codebase_root(parent)
    assert resolved == codebase_root

    resolved_parent = indexing_admin._resolve_codebase_root(repo)
    assert resolved_parent == codebase_root


def test_admin_abort_endpoint_falls_back_to_clear_when_abort_helper_missing(monkeypatch: pytest.MonkeyPatch):
    from scripts import upload_service

    calls = {"clear": []}

    monkeypatch.setattr(upload_service, "_require_admin_session", lambda request: {"user_id": "admin"})
    monkeypatch.setattr(upload_service, "abort_staging_rebuild", None)
    monkeypatch.setattr(
        upload_service,
        "clear_staging_collection",
        lambda workspace_path, repo_name: calls["clear"].append((workspace_path, repo_name)),
    )
    monkeypatch.setattr(
        upload_service,
        "resolve_collection_root",
        lambda **kwargs: ("/fake/root", "repo1"),
    )

    client = TestClient(upload_service.app)

    resp = client.post("/admin/staging/abort", data={"collection": "coll1"}, follow_redirects=False)
    assert resp.status_code == 302
    assert calls["clear"] == [("/fake/root", "repo1")]


def test_watcher_collection_reuse_logical_repo(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    from scripts.watch_index_core import utils as watch_utils

    repo_path = tmp_path / "repoA"
    repo_path.mkdir()

    monkeypatch.setenv("MULTI_REPO_MODE", "1")
    monkeypatch.setenv("WORKSPACE_PATH", str(tmp_path))
    monkeypatch.setattr(watch_utils, "_extract_repo_name_from_path", lambda *_: "repoA")
    monkeypatch.setattr(watch_utils, "is_multi_repo_mode", lambda: True)
    monkeypatch.setattr(watch_utils, "logical_repo_reuse_enabled", lambda: True)

    def fake_get_state(ws_path, repo_name):
        return {}

    monkeypatch.setattr(watch_utils, "get_workspace_state", fake_get_state)

    def fake_ensure(state, ws_path):
        new_state = dict(state or {})
        new_state["logical_repo_id"] = "lrid"
        return new_state

    monkeypatch.setattr(watch_utils, "ensure_logical_repo_id", fake_ensure)
    monkeypatch.setattr(
        watch_utils,
        "find_collection_for_logical_repo",
        lambda lrid, search_root: "reuse-coll",
    )

    updates = []
    monkeypatch.setattr(
        watch_utils,
        "update_workspace_state",
        lambda **kwargs: updates.append(kwargs),
    )

    coll = watch_utils._get_collection_for_repo(repo_path)
    assert coll == "reuse-coll"
    assert updates and updates[0]["updates"]["qdrant_collection"] == "reuse-coll"
