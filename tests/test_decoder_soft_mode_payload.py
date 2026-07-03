from scripts.refrag_llamacpp import LlamaCppRefragClient


def test_soft_mode_posts_soft_embeddings(monkeypatch):
    monkeypatch.setenv("REFRAG_DECODER", "1")
    monkeypatch.setenv("REFRAG_RUNTIME", "llamacpp")
    monkeypatch.setenv("REFRAG_DECODER_MODE", "soft")
    monkeypatch.setenv("REFRAG_SOFT_SCALE", "1.25")

    called = {}

    def fake_post(path, payload):
        called["path"] = path
        called["payload"] = payload
        return {"content": "ok"}

    c = LlamaCppRefragClient(base_url="http://fake")
    c._post = fake_post  # type: ignore

    soft = [[0.1, 0.2, 0.3], [0.0, -0.1, 0.1]]
    out = c.generate_with_soft_embeddings("Q:", soft_embeddings=soft, max_tokens=64)
    assert out == "ok"
    assert called["path"] == "/soft_completion"
    assert called["payload"]["soft_embeddings"] == soft
    assert called["payload"]["scale"] == 1.25
