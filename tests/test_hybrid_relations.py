import importlib
import pytest

hyb = importlib.import_module("scripts.hybrid_search")


class _Pt:
    def __init__(self, pid, path):
        self.id = pid
        # Minimal payload; include path_prefix so related_paths can be computed
        self.payload = {
            "metadata": {
                "path": path,
                "path_prefix": "/" + "/".join(path.split("/")[:-1]).strip("/") or "/",
                "start_line": 1,
                "end_line": 2,
                # Provide empty imports/calls lists
                "imports": [],
                "calls": [],
            }
        }


class _QP:
    def __init__(self, points):
        self.points = points


class FakeQdrant:
    def __init__(self, points):
        self._points = points

    def query_points(self, **kwargs):
        return _QP(self._points)

    def search(self, **kwargs):
        return self._points


class FakeEmbed:
    class _Vec:
        def __init__(self):
            self._v = [0.01] * 8

        def tolist(self):
            return self._v

    def embed(self, texts):
        for _ in texts:
            yield FakeEmbed._Vec()


@pytest.mark.unit
def test_run_hybrid_search_relations_and_related_paths(monkeypatch):
    # Two files in same directory -> should yield related_paths for each other
    pts = [
        _Pt("1", "src/a.py"),
        _Pt("2", "src/b.py"),
        _Pt("3", "tests/c_test.py"),
    ]
    # Add an import-like path so a.py imports b -> should appear via import-based resolution
    pts[0].payload["metadata"]["imports"] = ["./b"]

    # Patch client factory used inside hybrid_search so we don't hit real Qdrant
    monkeypatch.setattr(hyb, "get_qdrant_client", lambda *a, **k: FakeQdrant(pts))
    monkeypatch.setattr(hyb, "return_qdrant_client", lambda *a, **k: None)
    monkeypatch.setattr(hyb, "TextEmbedding", lambda *a, **k: FakeEmbed())
    monkeypatch.setattr(hyb, "_get_embedding_model", lambda *a, **k: FakeEmbed())
    monkeypatch.setenv("EMBEDDING_MODEL", "unit-test")
    monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")

    items = hyb.run_hybrid_search(
        queries=["foo"],
        limit=5,
        per_path=2,
        path_glob=["src/*.py", "tests/*"],
        expand=False,
        model=FakeEmbed(),
    )
    assert items, "expected some items"
    for it in items:
        # Keys should exist even if lists might be empty
        assert "relations" in it
        assert "related_paths" in it
    # Ensure related_paths captured sibling in same dir
    src_items = [it for it in items if str(it.get("path")).startswith("src/")]
    if len(src_items) >= 2:
        # Each src item should mention the other in related_paths
        paths = {it["path"] for it in src_items}
        for it in src_items:
            rels = set(it.get("related_paths") or [])
            assert rels & (paths - {it["path"]}), "expected sibling path in related_paths"

