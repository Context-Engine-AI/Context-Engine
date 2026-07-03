import os
import uuid
import importlib
import subprocess
from types import SimpleNamespace

import pytest
from qdrant_client import QdrantClient, models


# qdrant_container fixture is now provided by conftest.py
# It uses CI Qdrant service (localhost:6333) or testcontainers (local dev)


ing = importlib.import_module("scripts.ingest_code")
mem_backup = importlib.import_module("scripts.memory_backup")
mem_restore = importlib.import_module("scripts.memory_restore")

pytestmark = pytest.mark.integration


def _create_collection_with_memory(qdrant_url: str, name: str, dim: int = 8) -> QdrantClient:
    """Create a collection with dense+lex vectors and a single memory point.

    The collection is intentionally created without the ReFRAG mini vector so that
    ensure_collection(..., REFRAG_MODE=1) must add it, exercising the
    backup/recreate/restore path.
    """
    client = QdrantClient(url=qdrant_url)

    vectors_cfg = {
        "code": models.VectorParams(size=dim, distance=models.Distance.COSINE),
        ing.LEX_VECTOR_NAME: models.VectorParams(
            size=ing.LEX_VECTOR_DIM, distance=models.Distance.COSINE
        ),
    }
    client.create_collection(collection_name=name, vectors_config=vectors_cfg)

    # One "memory" point (no metadata.path) and one code point (with path).
    # Use integer point IDs to match Qdrant's accepted ID types.
    points = [
        models.PointStruct(
            id=1,
            vector={"code": [0.1] * dim},
            payload={"information": "test memory", "metadata": {}},
        ),
        models.PointStruct(
            id=2,
            vector={"code": [0.2] * dim},
            payload={
                "information": "code chunk",
                # Mark as real code: has a path and language/kind so is_memory_point() returns False
                "metadata": {"path": "/tmp/example.py", "language": "python", "kind": "code"},
            },
        ),
    ]
    client.upsert(collection_name=name, points=points)
    return client


def _get_point_ids(client: QdrantClient, collection_name: str) -> set[str]:
    pts, _ = client.scroll(
        collection_name=collection_name,
        limit=None,
        with_payload=False,
        with_vectors=False,
    )
    return {str(p.id) for p in pts}


def test_memory_backup_restore_happy_path(qdrant_container, monkeypatch):
    """ensure_collection should avoid destructive changes by default.

    Scenario:
    - Start with a collection that has dense+lex vectors and at least one
      "memory" point.
    - Enable REFRAG_MODE so ensure_collection wants to add the mini vector.
    - The collection should be updated (if possible) without recreation.
    - Existing points should remain intact.
    """
    os.environ["QDRANT_URL"] = qdrant_container
    collection = f"test-mem-{uuid.uuid4().hex[:8]}"

    client = _create_collection_with_memory(qdrant_container, collection, dim=8)

    # Force ReFRAG on so ensure_collection tries to add MINI_VECTOR_NAME
    os.environ["REFRAG_MODE"] = "1"
    os.environ.pop("STRICT_MEMORY_RESTORE", None)

    # Run ensure_collection: this should trigger backup + recreate + restore
    ing.ensure_collection(client, collection, dim=8, vector_name="code")

    info = client.get_collection(collection)
    cfg = info.config.params.vectors

    # Dense + lex must be present
    assert "code" in cfg
    assert ing.LEX_VECTOR_NAME in cfg

    # When REFRAG_MODE is on, mini vector may be added if supported by the server/client

    # Existing points should still exist (no destructive recreate)
    ids = _get_point_ids(client, collection)
    assert "1" in ids
    assert "2" in ids


def test_memory_restore_strict_mode_no_recreate(qdrant_container, monkeypatch):
    """STRICT_MEMORY_RESTORE should not trigger errors when no recreate occurs."""
    os.environ["QDRANT_URL"] = qdrant_container
    collection = f"test-mem-strict-{uuid.uuid4().hex[:8]}"

    client = _create_collection_with_memory(qdrant_container, collection, dim=8)

    os.environ["REFRAG_MODE"] = "1"
    os.environ["STRICT_MEMORY_RESTORE"] = "1"

    # Patch subprocess.run to:
    # - allow the real memory_backup.py to run
    # - force memory_restore.py to fail with non-zero exit
    orig_run = subprocess.run

    def fake_run(args, **kwargs):  # type: ignore[override]
        cmd_str = " ".join(map(str, args))
        if "memory_backup.py" in cmd_str:
            return orig_run(args, **kwargs)
        if "memory_restore.py" in cmd_str:
            return SimpleNamespace(returncode=1, stdout="", stderr="simulated restore failure")
        return orig_run(args, **kwargs)

    monkeypatch.setattr(subprocess, "run", fake_run)

    ing.ensure_collection(client, collection, dim=8, vector_name="code")

    ids = _get_point_ids(client, collection)
    assert "1" in ids
    assert "2" in ids


def test_memory_backup_failure_tolerant_mode_no_recreate(qdrant_container, monkeypatch):
    """If backup fails but STRICT_MEMORY_RESTORE is not set, ensure_collection
    should still proceed without destructive recreation.
    """
    os.environ["QDRANT_URL"] = qdrant_container
    collection = f"test-mem-backup-fail-{uuid.uuid4().hex[:8]}"

    client = _create_collection_with_memory(qdrant_container, collection, dim=8)

    os.environ["REFRAG_MODE"] = "1"
    os.environ.pop("STRICT_MEMORY_RESTORE", None)

    # Patch subprocess.run so memory_backup.py fails, but everything else runs normally
    orig_run = subprocess.run

    def fake_run(args, **kwargs):  # type: ignore[override]
        cmd_str = " ".join(map(str, args))
        if "memory_backup.py" in cmd_str:
            return SimpleNamespace(returncode=1, stdout="", stderr="simulated backup failure")
        return orig_run(args, **kwargs)

    monkeypatch.setattr(subprocess, "run", fake_run)

    # Should not raise even though backup fails
    ing.ensure_collection(client, collection, dim=8, vector_name="code")

    info = client.get_collection(collection)
    cfg = info.config.params.vectors

    # Collection should still have the expected vectors
    assert "code" in cfg
    assert ing.LEX_VECTOR_NAME in cfg

    # Backup failure should not delete existing points
    ids = _get_point_ids(client, collection)
    assert "1" in ids
    assert "2" in ids


def test_memory_backup_and_restore_scripts_roundtrip(qdrant_container, tmp_path):
    """Directly exercise memory_backup.export_memories and
    memory_restore.restore_memories without going through ensure_collection.

    This confirms that the backup file contains the expected memory and that
    restore_memories can recreate it in a fresh collection.
    """
    os.environ["QDRANT_URL"] = qdrant_container
    collection = f"test-mem-scripts-{uuid.uuid4().hex[:8]}"

    client = _create_collection_with_memory(qdrant_container, collection, dim=8)

    # Backup memories from the collection
    backup_file = tmp_path / "memories_backup.json"
    result = mem_backup.export_memories(
        collection_name=collection,
        output_file=str(backup_file),
        client=client,
        include_vectors=True,
        batch_size=100,
    )

    assert result["success"] is True
    assert result["memory_count"] == 1
    assert backup_file.exists()

    # Drop the original collection entirely
    client.delete_collection(collection)

    # Restore into a fresh collection; let restore_memories create it
    restore_result = mem_restore.restore_memories(
        backup_file=str(backup_file),
        collection_name=collection,
        client=client,
        embedding_model_name=None,
        vector_name="code",
        batch_size=50,
        skip_existing=True,
        skip_collection_creation=False,
    )

    assert restore_result["success"] is True

    # After restore, there should be exactly one memory point (id 1) and no code point (id 2)
    ids = _get_point_ids(client, collection)
    assert "1" in ids
    assert "2" not in ids
