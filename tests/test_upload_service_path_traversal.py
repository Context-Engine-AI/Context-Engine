import io
import json
import tarfile
from pathlib import Path

import pytest


def _write_bundle(tmp_path: Path, operations: list[dict]) -> Path:
    bundle_path = tmp_path / "bundle.tar.gz"
    payload = json.dumps({"operations": operations}).encode("utf-8")

    with tarfile.open(bundle_path, "w:gz") as tar:
        info = tarfile.TarInfo(name="metadata/operations.json")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))

    return bundle_path


def _write_bundle_with_moved_file(tmp_path: Path, dest_path: str, content: bytes) -> Path:
    bundle_path = tmp_path / "bundle.tar.gz"
    operations = [{"operation": "moved", "path": dest_path, "source_path": "missing_src.txt"}]
    payload = json.dumps({"operations": operations}).encode("utf-8")

    with tarfile.open(bundle_path, "w:gz") as tar:
        info = tarfile.TarInfo(name="metadata/operations.json")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))

        file_info = tarfile.TarInfo(name=f"files/moved/{dest_path}")
        file_info.size = len(content)
        tar.addfile(file_info, io.BytesIO(content))

    return bundle_path


def _write_bundle_with_created_file(tmp_path: Path, rel_path: str, content: bytes) -> Path:
    bundle_path = tmp_path / "bundle.tar.gz"
    operations = [{"operation": "created", "path": rel_path}]
    payload = json.dumps({"operations": operations}).encode("utf-8")

    with tarfile.open(bundle_path, "w:gz") as tar:
        info = tarfile.TarInfo(name="metadata/operations.json")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))

        file_info = tarfile.TarInfo(name=f"files/created/{rel_path}")
        file_info.size = len(content)
        tar.addfile(file_info, io.BytesIO(content))

    return bundle_path


def test_process_delta_bundle_rejects_traversal_created(tmp_path, monkeypatch):
    import scripts.upload_delta_bundle as us

    work_dir = tmp_path / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(us, "WORK_DIR", str(work_dir))

    bundle = _write_bundle(
        tmp_path,
        [{"operation": "created", "path": "../../evil.txt"}],
    )

    with pytest.raises(ValueError, match="escapes workspace"):
        us.process_delta_bundle(
            workspace_path="/home/user/repo",
            bundle_path=bundle,
            manifest={"bundle_id": "b1"},
        )


def test_process_delta_bundle_moved_falls_back_to_tar_payload_when_source_missing(tmp_path, monkeypatch):
    import scripts.upload_delta_bundle as us

    work_dir = tmp_path / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(us, "WORK_DIR", str(work_dir))

    slug = "repo-0123456789abcdef"
    bundle = _write_bundle_with_moved_file(tmp_path, "dst.txt", b"moved-payload")

    counts = us.process_delta_bundle(
        workspace_path=f"/work/{slug}",
        bundle_path=bundle,
        manifest={"bundle_id": "b-moved"},
    )

    assert counts.get("moved") == 1
    assert (work_dir / slug / "dst.txt").read_bytes() == b"moved-payload"


def test_process_delta_bundle_slugged_workspace_creates_marker(tmp_path, monkeypatch):
    import scripts.upload_delta_bundle as us

    work_dir = tmp_path / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(us, "WORK_DIR", str(work_dir))

    slug = "repo-0123456789abcdef"
    bundle = _write_bundle_with_created_file(tmp_path, "a.txt", b"hello")

    counts = us.process_delta_bundle(
        workspace_path=f"/work/{slug}",
        bundle_path=bundle,
        manifest={"bundle_id": "b1"},
    )

    assert counts.get("created") == 1
    assert (work_dir / slug / "a.txt").exists()
    assert not (work_dir / slug / slug / "a.txt").exists()
    assert (work_dir / ".codebase" / "repos" / slug / ".ctxce_managed_upload").exists()


def test_process_delta_bundle_mirrors_to_old_slug_when_staging_active(tmp_path, monkeypatch):
    import scripts.upload_delta_bundle as us

    work_dir = tmp_path / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(us, "WORK_DIR", str(work_dir))

    canonical_slug = "repo1-0123456789abcdef"
    old_slug = f"{canonical_slug}_old"

    monkeypatch.setenv("CTXCE_STAGING_ENABLED", "1")
    monkeypatch.setattr(us, "is_staging_enabled", lambda: True)
    monkeypatch.setattr(us, "_extract_repo_name_from_path", lambda path: canonical_slug)
    monkeypatch.setattr(
        us,
        "get_collection_state_snapshot",
        lambda workspace_path=None, repo_name=None: {
            "serving_repo_slug": old_slug,
            "active_repo_slug": canonical_slug,
        },
    )

    bundle = _write_bundle_with_created_file(tmp_path, "src/file.txt", b"payload")

    counts = us.process_delta_bundle(
        workspace_path="/work/random",
        bundle_path=bundle,
        manifest={"bundle_id": "b-dual"},
    )

    assert counts.get("created") == 1

    for slug in (canonical_slug, old_slug):
        target = work_dir / slug / "src" / "file.txt"
        assert target.exists(), f"expected write for {slug}"
        marker = work_dir / ".codebase" / "repos" / slug / ".ctxce_managed_upload"
        assert marker.exists(), f"expected marker for {slug}"


def test_process_delta_bundle_rejects_absolute_paths(tmp_path, monkeypatch):
    import scripts.upload_delta_bundle as us

    work_dir = tmp_path / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(us, "WORK_DIR", str(work_dir))

    bundle = _write_bundle(
        tmp_path,
        [{"operation": "created", "path": "/etc/passwd"}],
    )

    with pytest.raises(ValueError, match="Absolute paths"):
        us.process_delta_bundle(
            workspace_path="/home/user/repo",
            bundle_path=bundle,
            manifest={"bundle_id": "b1"},
        )


def test_process_delta_bundle_rejects_traversal_moved_source(tmp_path, monkeypatch):
    import scripts.upload_delta_bundle as us

    work_dir = tmp_path / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(us, "WORK_DIR", str(work_dir))

    bundle = _write_bundle(
        tmp_path,
        [
            {
                "operation": "moved",
                "path": "dst.txt",
                "source_path": "../../escape.txt",
            }
        ],
    )

    with pytest.raises(ValueError, match="escapes workspace"):
        us.process_delta_bundle(
            workspace_path="/home/user/repo",
            bundle_path=bundle,
            manifest={"bundle_id": "b1"},
        )
