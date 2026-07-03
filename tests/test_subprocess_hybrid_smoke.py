import os
import uuid
import subprocess
import sys
import pytest
import importlib

pytestmark = pytest.mark.integration


# Always enabled; smoke runs as part of full suite
# qdrant_container fixture is now provided by conftest.py
# It uses CI Qdrant service (localhost:6333) or testcontainers (local dev)


@pytest.mark.integration
def test_hybrid_cli_runs_basic(tmp_path, qdrant_container):
    env = os.environ.copy()
    env["QDRANT_URL"] = qdrant_container
    collection_name = f"test-{uuid.uuid4().hex[:8]}"
    env["COLLECTION_NAME"] = collection_name

    # Warm the FastEmbed model cache to avoid long first-run download inside subprocess
    try:
        from fastembed import TextEmbedding

        TextEmbedding(model_name="BAAI/bge-base-en-v1.5")
    except Exception:
        pass

    # Create a tiny repo and index it so the CLI has something to return
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "a.py").write_text(
        "def test():\n    return 1\n", encoding="utf-8"
    )
    ing = importlib.import_module("scripts.ingest_code")
    prev_collection = os.environ.get("COLLECTION_NAME")
    os.environ["COLLECTION_NAME"] = collection_name
    try:
        ing.index_repo(
            root=tmp_path,
            qdrant_url=qdrant_container,
            api_key="",
            collection=collection_name,
            model_name="BAAI/bge-base-en-v1.5",
            recreate=True,
        )
    finally:
        if prev_collection is None:
            os.environ.pop("COLLECTION_NAME", None)
        else:
            os.environ["COLLECTION_NAME"] = prev_collection

    # Use the real model; allow time to download on first run
    env["EMBEDDING_MODEL"] = "BAAI/bge-base-en-v1.5"
    cmd = [
        sys.executable,
        "scripts/hybrid_search.py",
        "--query",
        "test",
        "--limit",
        "1",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=300)
    # Assert successful run with real model and output
    assert proc.returncode == 0
    assert (proc.stdout or "").strip() != ""
