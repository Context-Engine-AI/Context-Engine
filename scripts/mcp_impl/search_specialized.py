#!/usr/bin/env python3
"""
mcp/search_specialized.py - Specialized search implementations for MCP indexer server.

Extracted from mcp_indexer_server.py for better modularity.
Contains:
- _search_tests_for_impl: Search for test files
- _search_config_for_impl: Search for config files
- _search_callers_for_impl: Search for callers/usages
- _search_importers_for_impl: Search for importers

Note: The @mcp.tool() decorated functions remain in mcp_indexer_server.py
as thin wrappers that call these implementations.
"""

from __future__ import annotations

__all__ = [
    "_search_tests_for_impl",
    "_search_config_for_impl", 
    "_search_callers_for_impl",
    "_search_importers_for_impl",
]

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Imports from sibling modules
# ---------------------------------------------------------------------------
from scripts.mcp_impl.utils import _extract_kwargs_payload


# Test file globs
TEST_GLOBS = [
    "tests/**",
    "test/**",
    "**/*test*.*",
    "**/*_test.*",
    "**/Test*/**",
]

# Config file globs  
CONFIG_GLOBS = [
    "**/*.yml",
    "**/*.yaml",
    "**/*.json",
    "**/*.toml",
    "**/*.ini",
    "**/*.env",
    "**/*.config",
    "**/*.conf",
    "**/*.properties",
    "**/*.csproj",
    "**/*.props",
    "**/*.targets",
    "**/*.xml",
    "**/appsettings*.json",
]

# Code file globs for importers
CODE_GLOBS = [
    "**/*.py",
    "**/*.js",
    "**/*.ts",
    "**/*.tsx",
    "**/*.jsx",
    "**/*.mjs",
    "**/*.cjs",
    "**/*.go",
    "**/*.java",
    "**/*.cs",
    "**/*.rb",
    "**/*.php",
    "**/*.rs",
    "**/*.c",
    "**/*.h",
    "**/*.cpp",
    "**/*.hpp",
]


async def _search_tests_for_impl(
    query: Any = None,
    limit: Any = None,
    include_snippet: Any = None,
    context_lines: Any = None,
    under: Any = None,
    language: Any = None,
    session: Any = None,
    compact: Any = None,
    kwargs: Any = None,
    ctx: Any = None,
    repo_search_fn=None,
) -> Dict[str, Any]:
    """Find test files related to a query.

    What it does:
    - Presets common test file globs and forwards to repo_search
    - Accepts extra filters via kwargs (e.g., language, under, case)

    Parameters:
    - query: str or list[str]; limit; include_snippet/context_lines; under; language; compact

    Returns: repo_search result shape.
    """
    globs = list(TEST_GLOBS)
    # Allow caller to add more with path_glob kwarg
    # Handle kwargs being passed as a string by some MCP clients
    _kwargs = _extract_kwargs_payload(kwargs) if kwargs else {}
    extra_glob = _kwargs.get("path_glob")
    if extra_glob:
        if isinstance(extra_glob, (list, tuple)):
            globs.extend([str(x) for x in extra_glob])
        else:
            globs.append(str(extra_glob))
    
    if repo_search_fn is None:
        from scripts.mcp_impl.search import _repo_search_impl
        repo_search_fn = _repo_search_impl
    
    return await repo_search_fn(
        query=query,
        limit=limit,
        include_snippet=include_snippet,
        context_lines=context_lines,
        under=under,
        language=language,
        path_glob=globs,
        session=session,
        compact=compact,
        ctx=ctx,
        kwargs={k: v for k, v in _kwargs.items() if k not in {"path_glob"}},
    )


async def _search_config_for_impl(
    query: Any = None,
    limit: Any = None,
    include_snippet: Any = None,
    context_lines: Any = None,
    under: Any = None,
    session: Any = None,
    compact: Any = None,
    kwargs: Any = None,
    ctx: Any = None,
    repo_search_fn=None,
) -> Dict[str, Any]:
    """Find likely configuration files for a service/query.

    What it does:
    - Presets config file globs (yaml/json/toml/etc.) and forwards to repo_search
    - Accepts extra filters via kwargs

    Returns: repo_search result shape.
    """
    globs = list(CONFIG_GLOBS)
    # Handle kwargs being passed as a string by some MCP clients
    _kwargs = _extract_kwargs_payload(kwargs) if kwargs else {}
    extra_glob = _kwargs.get("path_glob")
    if extra_glob:
        if isinstance(extra_glob, (list, tuple)):
            globs.extend([str(x) for x in extra_glob])
        else:
            globs.append(str(extra_glob))
    
    if repo_search_fn is None:
        from scripts.mcp_impl.search import _repo_search_impl
        repo_search_fn = _repo_search_impl
    
    return await repo_search_fn(
        query=query,
        limit=limit,
        include_snippet=include_snippet,
        context_lines=context_lines,
        under=under,
        session=session,
        path_glob=globs,
        compact=compact,
        ctx=ctx,
        kwargs={k: v for k, v in _kwargs.items() if k not in {"path_glob"}},
    )


async def _search_callers_for_impl(
    query: Any = None,
    limit: Any = None,
    language: Any = None,
    session: Any = None,
    kwargs: Any = None,
    ctx: Any = None,
    repo_search_fn=None,
) -> Dict[str, Any]:
    """Heuristic search for callers/usages of a symbol.

    When to use:
    - You want files that reference/invoke a function/class

    Notes:
    - Thin wrapper over repo_search today; pass language or path_glob to narrow
    - Returns repo_search result shape
    """
    if repo_search_fn is None:
        from scripts.mcp_impl.search import _repo_search_impl
        repo_search_fn = _repo_search_impl

    return await repo_search_fn(
        query=query,
        limit=limit,
        language=language,
        session=session,
        ctx=ctx,
        kwargs=kwargs,
    )


async def _search_importers_for_impl(
    query: Any = None,
    limit: Any = None,
    language: Any = None,
    session: Any = None,
    kwargs: Any = None,
    ctx: Any = None,
    repo_search_fn=None,
) -> Dict[str, Any]:
    """Find files likely importing or referencing a module/symbol.

    What it does:
    - Presets code globs across common languages; forwards to repo_search
    - Accepts additional filters via kwargs (e.g., under, case)

    Returns: repo_search result shape.
    """
    globs = list(CODE_GLOBS)
    # Handle kwargs being passed as a string by some MCP clients
    _kwargs = _extract_kwargs_payload(kwargs) if kwargs else {}
    extra_glob = _kwargs.get("path_glob")
    if extra_glob:
        if isinstance(extra_glob, (list, tuple)):
            globs.extend([str(x) for x in extra_glob])
        else:
            globs.append(str(extra_glob))

    if repo_search_fn is None:
        from scripts.mcp_impl.search import _repo_search_impl
        repo_search_fn = _repo_search_impl

    # Forward to repo_search with preset path_glob; caller can still pass other filters
    return await repo_search_fn(
        query=query,
        limit=limit,
        language=language,
        path_glob=globs,
        session=session,
        ctx=ctx,
        kwargs={k: v for k, v in _kwargs.items() if k not in {"path_glob"}},
    )
