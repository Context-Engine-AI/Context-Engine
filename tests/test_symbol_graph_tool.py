import pytest


@pytest.mark.asyncio
async def test_symbol_graph_under_uses_path_prefix_matchvalue():
    # Import internal helper to validate filter construction without needing a real Qdrant instance.
    from qdrant_client import models as qmodels
    from scripts.mcp_impl import symbol_graph as sg

    captured = {}

    class FakeClient:
        def scroll(self, *, collection_name, scroll_filter, limit, with_payload, with_vectors):
            captured["collection_name"] = collection_name
            captured["scroll_filter"] = scroll_filter
            return ([], None)

    await sg._query_array_field(  # type: ignore[attr-defined]
        client=FakeClient(),
        collection="codebase",
        field_key="metadata.calls",
        value="foo",
        limit=10,
        language="python",
        under=sg._norm_under("scripts"),  # type: ignore[attr-defined]
    )

    flt = captured.get("scroll_filter")
    assert isinstance(flt, qmodels.Filter)
    must = list(flt.must or [])
    keys = [getattr(c, "key", None) for c in must]
    assert "metadata.path_prefix" in keys

    # Ensure it's an exact match (MatchValue), not substring (MatchText)
    cond = next(c for c in must if getattr(c, "key", None) == "metadata.path_prefix")
    assert isinstance(cond.match, qmodels.MatchValue)
    assert cond.match.value == "/work/scripts"


