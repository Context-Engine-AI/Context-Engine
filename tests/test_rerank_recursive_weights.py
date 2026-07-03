import numpy as np


def test_learned_projection_forward_hot_reload(tmp_path, monkeypatch):
    from scripts.rerank_recursive import projection as proj

    monkeypatch.setattr(proj.LearnedProjection, "WEIGHTS_DIR", str(tmp_path))
    lp = proj.LearnedProjection(input_dim=2, output_dim=2)
    lp.WEIGHTS_RELOAD_INTERVAL = 0.0001
    lp.set_collection("test")

    np.savez(lp._weights_path, W=np.eye(2, dtype=np.float32), version=1)
    lp._last_reload_check = 0.0
    lp._weights_mtime = 0.0
    out1 = lp.forward(np.array([1.0, 0.0], dtype=np.float32))
    assert np.allclose(out1, np.array([1.0, 0.0], dtype=np.float32))

    np.savez(lp._weights_path, W=np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32), version=2)
    lp._last_reload_check = 0.0
    lp._weights_mtime = 0.0
    out2 = lp.forward(np.array([1.0, 0.0], dtype=np.float32))
    assert np.allclose(out2, np.array([0.0, 1.0], dtype=np.float32))


def test_tinyscorer_persists_recent_losses(tmp_path, monkeypatch):
    from scripts.rerank_recursive import scorer as sc

    monkeypatch.setattr(sc.TinyScorer, "WEIGHTS_DIR", str(tmp_path))
    s1 = sc.TinyScorer(dim=2, hidden_dim=2)
    losses = [float(i) for i in range(250)]
    s1._recent_losses = list(losses)
    s1._save_weights()

    s2 = sc.TinyScorer(dim=2, hidden_dim=2)
    expected = np.array(losses[-200:], dtype=np.float32)
    assert np.allclose(np.array(s2._recent_losses, dtype=np.float32), expected)
