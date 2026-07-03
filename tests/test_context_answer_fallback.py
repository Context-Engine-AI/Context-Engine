import asyncio
import pytest
from unittest.mock import patch

@pytest.mark.asyncio
async def test_context_answer_has_no_filesystem_fallback_when_no_hits():
    """When retrieval yields no spans, we do NOT glob or read the host filesystem.
    Citations may be empty, and that's expected.
    """
    from scripts.mcp_indexer_server import context_answer
    import scripts.mcp_indexer_server as srv

    # Mock embedding model to avoid loading real model
    with patch.object(srv, "_get_embedding_model", return_value=None):
        out = await context_answer(
            query="Describe module roles",
            limit=3,
            per_path=1,
            include_snippet=True,
            path_glob=["scripts/hybrid_search.py"],
            # Force a very unlikely match to simulate empty retrieval
            language="nonexistentlang",
        )
        assert isinstance(out, dict)
        # No fallback: citations can be empty
        cits = out.get("citations") or []
        assert len(cits) == 0

