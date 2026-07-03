import importlib
import pytest

srv = importlib.import_module("scripts.mcp_indexer_server")


@pytest.mark.service
def test_context_answer_path_mention_fallback(monkeypatch):
    # Mock embedding model to avoid loading real model
    monkeypatch.setattr(srv, "_get_embedding_model", lambda *a, **k: None)

    # Force retrieval to return nothing so path-mention fallback engages
    import scripts.hybrid_search as hs
    monkeypatch.setattr(hs, "run_hybrid_search", lambda **k: [])

    import scripts.refrag_llamacpp as ref

    class FakeLlama:
        def __init__(self, *a, **k):
            pass

        def generate_with_soft_embeddings(self, prompt: str, max_tokens: int = 64, **kw):
            # Should still include Sources and [1] with the mentioned file
            assert "Sources:" in prompt
            assert "[1]" in prompt
            return "ok [1]"

    monkeypatch.setattr(ref, "LlamaCppRefragClient", FakeLlama)
    monkeypatch.setattr(ref, "is_decoder_enabled", lambda: True)

    # Mention an actual file in this repo so fallback can find it
    q = "explain something in scripts/hybrid_search.py"
    out = srv.asyncio.get_event_loop().run_until_complete(
        srv.context_answer(query=q, limit=3, per_path=2)
    )
    assert isinstance(out, dict)
    cits = out.get("citations") or []
    assert len(cits) >= 1
    # Either path or rel_path should indicate the file
    p = cits[0].get("path") or ""
    rp = cits[0].get("rel_path") or ""
    assert p.endswith("scripts/hybrid_search.py") or rp.endswith("scripts/hybrid_search.py")

