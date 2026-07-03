from scripts.refrag_llamacpp import (
    is_decoder_enabled,
    get_runtime_kind,
    LlamaCppRefragClient,
)


def test_decoder_disabled_by_default(monkeypatch):
    # Default: off
    monkeypatch.delenv("REFRAG_DECODER", raising=False)
    assert is_decoder_enabled() is False


def test_client_runtime_guard(monkeypatch):
    # Ensure creating the client checks runtime but does not enable anything
    monkeypatch.setenv("REFRAG_DECODER", "0")
    monkeypatch.setenv("REFRAG_RUNTIME", "llamacpp")
    c = LlamaCppRefragClient()
    assert c.base_url.startswith("http://")


def test_generate_guard_raises_when_disabled(monkeypatch):
    monkeypatch.setenv("REFRAG_DECODER", "0")
    monkeypatch.setenv("REFRAG_RUNTIME", "llamacpp")
    c = LlamaCppRefragClient()
    try:
        c.generate_with_soft_embeddings("hello", soft_embeddings=[[0.0] * 10])
        assert False, "should have raised"
    except RuntimeError as e:
        assert "disabled" in str(e).lower()
