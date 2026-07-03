import json

from scripts.benchmarks.cosqa.dataset import (
    CACHE_VERSION,
    CoSQACorpusEntry,
    CoSQADataset,
    CoSQAQuery,
    load_dataset_cache,
    save_dataset_cache,
)


def _make_dataset(fallback_used: bool = False) -> CoSQADataset:
    corpus_entry = CoSQACorpusEntry(
        code_id="c1",
        code="print('hi')",
        docstring="",
        func_name="",
    )
    query = CoSQAQuery(query_id="q1", query="q", relevant_code_ids=["c1"])
    return CoSQADataset(
        corpus={"c1": corpus_entry},
        queries={"q1": query},
        qrels={"q1": {"c1": 1}},
        split="test",
        fallback_used=fallback_used,
    )


def test_cosqa_cache_roundtrip_includes_version_and_fallback(tmp_path):
    dataset = _make_dataset(fallback_used=True)
    cache_path = tmp_path / "cosqa_cache.json"

    save_dataset_cache(dataset, cache_path=cache_path)
    loaded = load_dataset_cache(cache_path=cache_path, split="test")

    assert loaded is not None
    assert loaded.cache_version == CACHE_VERSION
    assert loaded.fallback_used is True
    assert loaded.corpus["c1"].code == "print('hi')"
    assert loaded.queries["q1"].query == "q"


def test_cosqa_cache_version_mismatch_returns_none(tmp_path):
    cache_path = tmp_path / "cosqa_cache.json"
    cache_path.write_text(json.dumps({"cache_version": CACHE_VERSION - 1}))

    loaded = load_dataset_cache(cache_path=cache_path, split="test")

    assert loaded is None
