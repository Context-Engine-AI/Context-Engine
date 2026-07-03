import importlib

import pytest
from fastapi.testclient import TestClient
from starlette.responses import Response


@pytest.mark.unit
def test_env_gate_blocks_delete_endpoint(monkeypatch):
    monkeypatch.setenv("CTXCE_ADMIN_COLLECTION_DELETE_ENABLED", "0")

    srv = importlib.import_module("scripts.upload_service")
    srv = importlib.reload(srv)

    def _noop_admin(_request):
        return {"user_id": "admin"}

    monkeypatch.setattr(srv, "_require_admin_session", _noop_admin)

    def _fake_render_admin_error(_request, title, message, back_href="/admin", status_code=400):
        return Response(content=f"{title}: {message}", status_code=status_code)

    monkeypatch.setattr(srv, "render_admin_error", _fake_render_admin_error)

    def _should_not_be_called(**_kwargs):
        raise AssertionError("delete_collection_everywhere should not be called when env gate is off")

    monkeypatch.setattr(srv, "delete_collection_everywhere", _should_not_be_called)

    client = TestClient(srv.app)
    resp = client.post("/admin/collections/delete", data={"collection": "c1", "delete_fs": ""})
    assert resp.status_code == 403


@pytest.mark.unit
def test_admin_role_gate_blocks_non_admin(monkeypatch):
    monkeypatch.setenv("CTXCE_AUTH_ENABLED", "1")
    monkeypatch.setenv("CTXCE_ADMIN_COLLECTION_DELETE_ENABLED", "1")

    auth = importlib.import_module("scripts.auth_backend")
    importlib.reload(auth)

    srv = importlib.import_module("scripts.upload_service")
    srv = importlib.reload(srv)

    monkeypatch.setattr(srv, "_get_valid_session_record", lambda _req: {"user_id": "u1"})
    monkeypatch.setattr(srv, "is_admin_user", lambda _uid: False)

    client = TestClient(srv.app)
    resp = client.post("/admin/collections/delete", data={"collection": "c1"})
    assert resp.status_code == 403
    assert resp.json().get("detail") == "Admin required"


@pytest.mark.unit
def test_collection_admin_refuses_when_env_disabled(monkeypatch):
    monkeypatch.setenv("CTXCE_ADMIN_COLLECTION_DELETE_ENABLED", "0")
    ca = importlib.import_module("scripts.collection_admin")
    ca = importlib.reload(ca)

    with pytest.raises(PermissionError):
        ca.delete_collection_everywhere(collection="c1", cleanup_fs=False)


@pytest.mark.unit
def test_managed_upload_marker_gates_workspace_deletion(tmp_path):
    ca = importlib.import_module("scripts.collection_admin")
    ca = importlib.reload(ca)

    work_root = tmp_path / "work"
    work_root.mkdir(parents=True, exist_ok=True)

    slug = work_root / "repo-0123456789abcdef"
    slug.mkdir(parents=True, exist_ok=True)

    assert ca._is_managed_upload_workspace_dir(slug, work_root=work_root) is False

    marker = work_root / ".codebase" / "repos" / slug.name
    marker.mkdir(parents=True, exist_ok=True)
    (marker / ".ctxce_managed_upload").write_text("1\n")
    assert ca._is_managed_upload_workspace_dir(slug, work_root=work_root) is True


@pytest.mark.unit
def test_registry_undelete_on_discovery(monkeypatch, tmp_path):
    db = tmp_path / "auth.sqlite"

    monkeypatch.setenv("CTXCE_AUTH_ENABLED", "1")
    monkeypatch.setenv("CTXCE_AUTH_DB_URL", f"sqlite:///{db}")
    monkeypatch.setenv("CTXCE_COLLECTION_REGISTRY_UNDELETE_ON_DISCOVERY", "1")

    ab = importlib.import_module("scripts.auth_backend")
    ab = importlib.reload(ab)

    ab._ensure_db()
    with ab._db_connection() as conn:
        with conn:
            conn.execute(
                "INSERT INTO collections (id, qdrant_collection, created_at, metadata_json, is_deleted) VALUES (?, ?, ?, ?, 1)",
                ("c1", "c1", 123, None),
            )

    got = ab.ensure_collection("c1")
    assert got.get("is_deleted") == 0

    with ab._db_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT is_deleted FROM collections WHERE qdrant_collection = ?", ("c1",))
        row = cur.fetchone()
    assert int(row[0] or 0) == 0


@pytest.mark.unit
def test_cleanup_state_files_deletes_symbol_caches_single_repo(monkeypatch, tmp_path):
    monkeypatch.setenv("WORK_DIR", str(tmp_path / "work"))
    ca = importlib.import_module("scripts.collection_admin")
    ca = importlib.reload(ca)

    ws = tmp_path / "ws"
    st_dir = ws / ".codebase"
    st_dir.mkdir(parents=True, exist_ok=True)

    state_file = st_dir / "state.json"
    state_file.write_text("{}")
    (st_dir / "cache.json").write_text("{}")
    (st_dir / "symbols_aaa.json").write_text("{}")
    (st_dir / "symbols_bbb.json").write_text("{}")

    removed = ca._cleanup_state_files_for_mapping({"state_file": str(state_file)})
    assert removed >= 1
    assert not state_file.exists()
    assert not (st_dir / "cache.json").exists()
    assert not (st_dir / "symbols_aaa.json").exists()
    assert not (st_dir / "symbols_bbb.json").exists()


@pytest.mark.unit
def test_cleanup_state_files_deletes_repo_dir_multi_repo(monkeypatch, tmp_path):
    monkeypatch.setenv("MULTI_REPO_MODE", "1")
    monkeypatch.setenv("WORK_DIR", str(tmp_path / "work"))
    monkeypatch.setenv("WORKSPACE_PATH", str(tmp_path / "work"))

    ca = importlib.import_module("scripts.collection_admin")
    ca = importlib.reload(ca)

    work_root = tmp_path / "work"
    repo_name = "repo-0123456789abcdef"

    repo_dir = work_root / ".codebase" / "repos" / repo_name
    repo_dir.mkdir(parents=True, exist_ok=True)
    (repo_dir / "state.json").write_text("{}")
    (repo_dir / "cache.json").write_text("{}")
    (repo_dir / "symbols_aaa.json").write_text("{}")

    removed = ca._cleanup_state_files_for_mapping(
        {"state_file": str(repo_dir / "state.json")},
        work_root=work_root,
    )
    assert removed >= 1
    assert not repo_dir.exists()


@pytest.mark.unit
def test_workspace_state_does_not_recreate_repo_metadata_when_workspace_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("MULTI_REPO_MODE", "1")
    monkeypatch.setenv("WORKSPACE_PATH", str(tmp_path / "work"))

    ws_root = tmp_path / "work"
    ws_root.mkdir(parents=True, exist_ok=True)

    repo_name = "repo-0123456789abcdef"
    state_dir = ws_root / ".codebase" / "repos" / repo_name
    assert not state_dir.exists()

    ws = importlib.import_module("scripts.workspace_state")
    ws = importlib.reload(ws)

    ws.log_activity(repo_name=repo_name, action="deleted", workspace_path=str(ws_root))
    assert not state_dir.exists()
