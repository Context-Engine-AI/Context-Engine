from scripts.refrag_llamacpp import LlamaCppRefragClient, is_decoder_enabled


def test_prompt_mode_routes_and_builds_payload(monkeypatch):
    # Enable decoder and prompt mode
    monkeypatch.setenv("REFRAG_DECODER", "1")
    monkeypatch.setenv("REFRAG_RUNTIME", "llamacpp")
    monkeypatch.setenv("REFRAG_DECODER_MODE", "prompt")

    called = {}

    def fake_post(path, payload):
        called["path"] = path
        called["payload"] = payload
        return {"content": "ok"}

    c = LlamaCppRefragClient(base_url="http://fake")
    # monkeypatch internal transport
    c._post = fake_post  # type: ignore

    out = c.generate_with_soft_embeddings("Hello")
    assert out == "ok"
    assert called["path"] == "/completion"
    assert "prompt" in called["payload"] and called["payload"]["prompt"].startswith(
        "Hello"
    )
