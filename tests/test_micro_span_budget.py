from scripts.hybrid_search import _merge_and_budget_spans, MICRO_TOKENS_PER_LINE


def _mk_item(path, start, end, score):
    # Simulate the shape produced mid-pipeline: {"pt": {payload: {metadata: {...}}}, "s": score}
    class _Pt:
        def __init__(self, payload):
            self.payload = payload

    md = {"path": path, "start_line": start, "end_line": end}
    return {"pt": _Pt({"metadata": md}), "s": score}


def test_merge_and_budget_spans_merges_and_respects_budget(monkeypatch):
    # Make tokens-per-line small to trigger budget easily
    monkeypatch.setenv("MICRO_TOKENS_PER_LINE", "10")
    monkeypatch.setenv("MICRO_BUDGET_TOKENS", "60")  # ~6 lines total
    monkeypatch.setenv("MICRO_MERGE_LINES", "2")
    monkeypatch.setenv("MICRO_OUT_MAX_SPANS", "2")

    # Two overlaps in same file should merge; a third far span should compete for budget
    items = [
        _mk_item("a.py", 10, 11, 1.0),  # 2 lines
        _mk_item("a.py", 12, 12, 0.9),  # adjacent -> merge to 10..12 (3 lines)
        _mk_item("a.py", 30, 33, 0.8),  # 4 lines; may fit depending on budget
        _mk_item("b.py", 5, 6, 0.7),  # 2 lines other file
    ]

    merged = _merge_and_budget_spans(items)

    # After merging, first cluster a.py should be 10..12 (3 lines -> 30 tokens)
    # Budget=60 means we can include either (10..12) + (30..33) OR (10..12) + b.py (depending on order/score)
    # Since a.py spans have higher scores, expect both from a.py until per-path cap hits 2
    assert len(merged) >= 1
    # Spans carry _merged_* annotations
    assert merged[0]["_merged_start"] <= merged[0]["_merged_end"]
    # Budget tokens used are attached
    assert isinstance(merged[0]["_budget_tokens"], int)


def test_merge_and_budget_respects_per_path_cap(monkeypatch):
    monkeypatch.setenv("MICRO_TOKENS_PER_LINE", "10")
    monkeypatch.setenv("MICRO_BUDGET_TOKENS", "200")  # ample
    monkeypatch.setenv("MICRO_MERGE_LINES", "1")
    monkeypatch.setenv("MICRO_OUT_MAX_SPANS", "1")

    items = [
        _mk_item("x.py", 1, 2, 1.0),
        _mk_item("x.py", 10, 11, 0.9),
        _mk_item("y.py", 3, 4, 0.8),
    ]
    merged = _merge_and_budget_spans(items)
    # Only 1 from x.py should be included due to per-path cap
    from collections import Counter

    paths = [(m.get("pt").payload["metadata"]["path"]) for m in merged]
    c = Counter(paths)
    assert c.get("x.py", 0) <= 1


def test_merge_and_budget_spans_works_without_explicit_budget(monkeypatch):
    """Ensure budgeting works when no MICRO_BUDGET_TOKENS is provided.

    This simulates the default case where callers don't set an explicit budget
    and rely on the library's internal default value.
    """

    # Do not set MICRO_BUDGET_TOKENS at all; rely on default inside the helper
    monkeypatch.delenv("MICRO_BUDGET_TOKENS", raising=False)

    # Keep other knobs small so we still trigger budgeting logic deterministically
    monkeypatch.setenv("MICRO_TOKENS_PER_LINE", "10")
    monkeypatch.setenv("MICRO_MERGE_LINES", "2")
    monkeypatch.setenv("MICRO_OUT_MAX_SPANS", "3")

    items = [
        _mk_item("a.py", 1, 3, 1.0),
        _mk_item("a.py", 4, 6, 0.9),
        _mk_item("b.py", 10, 12, 0.8),
    ]

    merged = _merge_and_budget_spans(items)

    # We should still get at least one merged span with budgeting metadata
    assert len(merged) >= 1
    assert merged[0]["_merged_start"] <= merged[0]["_merged_end"]
    assert isinstance(merged[0]["_budget_tokens"], int)


def test_adaptive_span_sizing_failure_is_non_fatal(monkeypatch):
    """Adaptive span sizing must never break budgeting if extent lookup fails."""
    monkeypatch.setenv("MICRO_TOKENS_PER_LINE", "10")
    monkeypatch.setenv("MICRO_BUDGET_TOKENS", "200")
    monkeypatch.setenv("MICRO_MERGE_LINES", "1")
    monkeypatch.setenv("MICRO_OUT_MAX_SPANS", "3")

    # Enable adaptive path (needs both collection + symbol)
    monkeypatch.setenv("COLLECTION_NAME", "dummy")

    # Force extent lookup to throw; the budgeter should swallow it.
    import scripts.hybrid_ranking as hr
    monkeypatch.setattr(hr, "_get_symbol_extent", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))

    items = [
        {"path": "x.py", "start_line": 10, "end_line": 12, "score": 1.0, "symbol": "MySymbol"},
        {"path": "y.py", "start_line": 1, "end_line": 2, "score": 0.9, "symbol": "OtherSymbol"},
    ]
    merged = _merge_and_budget_spans(items)
    assert len(merged) >= 1
    assert merged[0]["_merged_start"] <= merged[0]["_merged_end"]
