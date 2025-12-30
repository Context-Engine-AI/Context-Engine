import os
import uuid
import importlib
import pytest

pytestmark = pytest.mark.integration

ing = importlib.import_module("scripts.ingest_code")
srv = importlib.import_module("scripts.mcp_indexer_server")
embedder = importlib.import_module("scripts.embedder")
hy = importlib.import_module("scripts.hybrid_search")


class FakeEmbedder:
    def __init__(self, model_name: str = "fake"):
        self.model_name = model_name

    class _Vec:
        def __init__(self, arr):
            self._arr = arr
        def tolist(self):
            return self._arr
        def __len__(self):
            return len(self._arr)

    def embed(self, texts):
        # Deterministic small vector by hashing; yields objects with .tolist()
        for t in texts:
            h = sum(ord(c) for c in t) % 997
            vec = [(float((h + i) % 13) / 13.0) for i in range(32)]
            yield self._Vec(vec)


# qdrant_container fixture is now provided by conftest.py
# It uses QDRANT_URL env var (CI) or testcontainers (local dev)


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("refrag_mode", ["0", "1"], ids=["refrag_off", "refrag_on"])
async def test_tier2_fallback_unconditional_with_language_filter(tmp_path, monkeypatch, qdrant_container, refrag_mode):
    # Use monkeypatch to ensure proper test isolation for env vars
    monkeypatch.setenv("QDRANT_URL", qdrant_container)
    monkeypatch.setenv("COLLECTION_NAME", f"test-{uuid.uuid4().hex[:8]}")
    monkeypatch.setenv("USE_TREE_SITTER", "0")
    monkeypatch.setenv("HYBRID_IN_PROCESS", "1")
    monkeypatch.setenv("EMBEDDING_MODEL", "fake")
    # Test Tier-2 fallback with both REFRAG_MODE on and off.
    # Tier-2 triggers when Tier-1 returns zero results (due to path_glob).
    monkeypatch.setenv("REFRAG_MODE", refrag_mode)
    monkeypatch.setenv("REFRAG_GATE_FIRST", "0")
    monkeypatch.setenv("PATTERN_VECTORS", "0")
    # Disable multi-collection fallback to avoid creating extra collections
    monkeypatch.setenv("CTX_MULTI_COLLECTION", "0")

    # Stub embeddings everywhere (FakeEmbedder produces 32-dim vectors)
    monkeypatch.setattr(ing, "TextEmbedding", lambda *a, **k: FakeEmbedder("fake"))
    monkeypatch.setattr(embedder, "get_embedding_model", lambda *a, **k: FakeEmbedder("fake"))
    monkeypatch.setattr(embedder, "get_model_dimension", lambda *a, **k: 32)  # Match FakeEmbedder dim
    monkeypatch.setattr(srv, "_get_embedding_model", lambda *a, **k: FakeEmbedder("fake"))
    monkeypatch.setattr(hy, "TextEmbedding", lambda *a, **k: FakeEmbedder("fake"))
    monkeypatch.setattr(hy, "_get_embedding_model", lambda *a, **k: FakeEmbedder("fake"))

    # Clear collection cache to ensure fresh schema checks
    from scripts.ingest.qdrant import ENSURED_COLLECTIONS
    ENSURED_COLLECTIONS.clear()

    # Create tiny repo
    (tmp_path / "pkg").mkdir()
    f1 = tmp_path / "pkg" / "a.py"
    f1.write_text("def f():\n    return 1\n")

    # Index via function call (no shell)
    ing.index_repo(
        root=tmp_path,
        qdrant_url=qdrant_container,
        api_key="",
        collection=os.environ["COLLECTION_NAME"],
        model_name="fake",
        recreate=True,
    )

    # Construct a path_glob that excludes the actual file so Tier-1 yields zero
    nonmatch_glob = str(tmp_path / "does_not_match" / "*.py")

    # Call context_answer with a language filter present; Tier‑2 should trigger and return results
    out = await srv.context_answer(
        query="def f",
        limit=3,
        per_path=1,
        include_snippet=False,
        language="python",
        path_glob=nonmatch_glob,
    )

    assert isinstance(out, dict)
    cits = out.get("citations") or []
    # Despite Tier-1 returning zero due to path_glob, Tier-2 relaxed search should find a.py
    assert any(str(f1) in (c.get("path") or "") for c in cits)
