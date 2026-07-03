import asyncio
import pytest

# These tests exercise argument plumbing independent of live retrieval.

@pytest.mark.asyncio
async def test_per_path_zero_is_echoed_and_respected_in_args():
    from scripts.mcp_indexer_server import repo_search

    res = await repo_search(query="anything", limit=3, per_path=0)
    assert isinstance(res, dict)
    args = res.get("args") or {}
    assert args.get("per_path") == 0, f"expected per_path echoed as 0, got {args.get('per_path')}"


@pytest.mark.asyncio
async def test_compact_string_false_is_normalized_in_args():
    from scripts.mcp_indexer_server import repo_search

    # Passing compact as a string "false" should normalize to False in echoed args
    res = await repo_search(query="anything", limit=1, compact="false")
    assert isinstance(res, dict)
    args = res.get("args") or {}
    assert args.get("compact") is False, f"expected compact False, got {args.get('compact')}"
