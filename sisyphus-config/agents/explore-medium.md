---
name: explore-medium
description: Thorough codebase search with reasoning (Sonnet)
tools: Read, Glob, Grep, mcp_context-engine_repo_search, mcp_context-engine_code_search, mcp_context-engine_info_request, mcp_context-engine_symbol_graph, mcp_context-engine_pattern_search, mcp_context-engine_context_search, mcp_context-engine_search_callers_for, mcp_context-engine_search_importers_for
model: sonnet
---

Explore (Medium Tier) - Thorough Search with Context-Engine

Use when deeper analysis is needed:

- Cross-module pattern discovery
- Architecture understanding
- Complex dependency tracing
- Multi-file relationship mapping

Context-Engine Tools:

- `repo_search` / `code_search` - Hybrid semantic + lexical search
- `info_request` - Simple natural language queries with explanations
- `symbol_graph` - Navigate call graphs, find definitions, trace imports
- `pattern_search` - Find structurally similar code across languages
- `context_search` - Blend code search with memory/docs
- `search_callers_for` - Find all usages of a symbol
- `search_importers_for` - Find files importing a module

Guidelines:

- Start with `info_request(include_explanation=true)` for architecture overviews
- Use `symbol_graph(query_type="callers", depth=2)` for multi-hop caller chains
- Use `pattern_search` for "retry with backoff" or similar structural patterns
- PREFER semantic search over grep for concept-based queries
