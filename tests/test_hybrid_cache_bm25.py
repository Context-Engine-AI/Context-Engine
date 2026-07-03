import importlib
from collections import OrderedDict
import os
import pytest

hyb = importlib.import_module("scripts.hybrid_search")


class _Pt:
    def __init__(self, pid, path, code=""):
        self.id = pid
        self.payload = {
            "metadata": {
                "path": path,
                "path_prefix": "/" + "/".join(path.split("/")[:-1]).strip("/") or "/",
                "start_line": 1,
                "end_line": 2,
                "code": code,
            }
        }


class _QP:
    def __init__(self, points):
        self.points = points


class _CountingQdrant:
    def __init__(self, points):
        self._points = points
        self.calls = 0

    def query_points(self, **kwargs):
        self.calls += 1
        return _QP(self._points)

    def search(self, **kwargs):
        # Some paths may still call legacy .search(), count it as well
        self.calls += 1
        return self._points


class _FakeEmbed:
    class _Vec:
        def __init__(self):
            self._v = [0.01] * 8

        def tolist(self):
            return self._v

    def embed(self, texts):
        for _ in texts:
            yield _FakeEmbed._Vec()


@pytest.mark.unit
def test_results_cache_hit_is_deterministic_and_avoids_second_backend_call(monkeypatch):
    # Ensure cache is enabled and small
    monkeypatch.setenv("HYBRID_RESULTS_CACHE_ENABLED", "1")
    monkeypatch.setenv("HYBRID_RESULTS_CACHE", "8")

    # Reset cache
    if hasattr(hyb, "_RESULTS_CACHE") and isinstance(hyb._RESULTS_CACHE, OrderedDict):
        hyb._RESULTS_CACHE.clear()

    pts = [
        _Pt("1", "src/foo.py", code="def foo():\n    return 1\n"),
        _Pt("2", "src/bar.py", code="def bar():\n    return 2\n"),
    ]
    backend = _CountingQdrant(pts)

    # Monkeypatch backends and model (patch client factory used in hybrid_search)
    monkeypatch.setattr(hyb, "get_qdrant_client", lambda *a, **k: backend)
    monkeypatch.setattr(hyb, "return_qdrant_client", lambda *a, **k: None)
    monkeypatch.setattr(hyb, "TextEmbedding", lambda *a, **k: _FakeEmbed())
    monkeypatch.setattr(hyb, "_get_embedding_model", lambda *a, **k: _FakeEmbed())
    monkeypatch.setenv("EMBEDDING_MODEL", "unit-test")
    monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")

    args = dict(
        queries=["foo"],
        limit=5,
        per_path=2,
        path_glob=["src/*.py"],
        expand=False,
        model=_FakeEmbed(),
    )

    items1 = hyb.run_hybrid_search(**args)
    assert items1, "expected items on first run"
    calls_after_first = backend.calls

    # Second identical call should hit cache and not increase backend calls
    items2 = hyb.run_hybrid_search(**args)
    assert items2 == items1, "cache miss altered results; expected deterministic hit"
    assert backend.calls == calls_after_first, "second run should use cache and avoid backend"


@pytest.mark.unit
def test_lexical_bm25_boost_is_gentle_and_matches_multiplier():
    # Build minimal metadata where tokens appear only in code (no path/symbol boosts)
    md = {
        "path": "src/sample.py",
        "symbol": "",
        "symbol_path": "",
        "code": "foo bar baz\n# foo and bar appear once each\n",
    }

    base = hyb.lexical_score(["foo bar"], md)
    # Sanity: both tokens should match in code -> base > 0
    assert base > 0.0

    # Token weights: make 'foo' heavier (>1), 'bar' lighter (<1)
    token_weights = {"foo": 3.0, "bar": 0.5}
    bm25_w = 0.2  # gentle factor

    weighted = hyb.lexical_score(["foo bar"], md, token_weights=token_weights, bm25_weight=bm25_w)

    # Expected per-token multiplier: 1 + bm25_w * (w - 1)
    m_foo = 1.0 + bm25_w * (token_weights["foo"] - 1.0)  # 1.4
    m_bar = 1.0 + bm25_w * (token_weights["bar"] - 1.0)  # 0.9

    # Because both tokens appear once in code and no other boosts are active,
    # base ~= 1 + 1 = 2, weighted ~= 1*m_foo + 1*m_bar
    # Check the weighted score follows the multiplier logic within a small tolerance.
    expected_weighted = (1.0 * m_foo) + (1.0 * m_bar)

    # Allow a small tolerance in case lexical_score adds tiny extras in some environments
    assert pytest.approx(weighted, rel=1e-6, abs=1e-6) == expected_weighted

    # Gentle behavior: overall change should be modest (within 50%)
    ratio = weighted / base
    assert 0.5 <= ratio <= 1.5, f"BM25 weighting should be gentle, got ratio={ratio:.3f}"

