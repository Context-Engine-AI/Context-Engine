---
name: mcp-tool-selection
description: Decision rules for when to use MCP Qdrant-Indexer semantic search vs grep/literal file tools. Use this skill when starting exploration, debugging, or answering "where/why" questions about code.
---

# MCP Tool Selection Rules

**Core principle:** MCP Qdrant-Indexer tools are primary for exploring code and history. Start with MCP for exploration, debugging, or "where/why" questions; use literal search/file-open only for narrow exact-literal lookups.

## STOP — Do NOT Use Read File or Grep for Exploration

**DO NOT use `Read File`, `grep`, `ripgrep`, `cat`, `find`, or any filesystem search tool for code exploration.**
You have MCP tools that are faster, smarter, and return ranked, contextual results.

- About to `Read` a file to understand it? → use `search` or `repo_search` or `context_answer`
- About to `grep` for a symbol? → use `search` or `symbol_graph` or `search_callers_for`
- About to `grep -r` for a concept? → use `search` with natural language
- About to `find`/`ls` for project structure? → use `qdrant_status` (with list_all=true)

**TIP:** Use `search` as your DEFAULT tool — it auto-detects intent and routes to the best specialized tool.

The ONLY acceptable use of grep/Read: confirming exact literal strings (e.g., `REDIS_HOST`), or reading a file you already located via MCP for editing.

## Use MCP Qdrant-Indexer When

- Exploring or don't know exact strings/symbols
- Need semantic or cross-file understanding (relationships, patterns, architecture)
- Want ranked results with surrounding context, not just line hits
- Asking conceptual/architectural or "where/why" behavior questions
- Need rich context/snippets around matches
- Finding callers, definitions, or importers of any symbol

## Use Literal Search/File-Open Only When

- Know exact string/function/variable or error message
- Only need to confirm existence or file/line quickly (not to understand behavior)

## Grep Anti-Patterns (DON'T)

```bash
grep -r "auth" .        # → Use MCP: "authentication mechanisms"
grep -r "cache" .       # → Use MCP: "caching strategies"  
grep -r "error" .       # → Use MCP: "error handling patterns"
grep -r "database" .    # → Use MCP: "database operations"
# Also DON'T:
Read File to understand a module  # → Use repo_search or context_answer
Read File to find callers         # → Use symbol_graph
find/ls for project structure     # → Use qdrant_status (list_all=true)
```

## Literal Search Patterns (DO)

```bash
grep -rn "UserAlreadyExists" .      # Specific error class
grep -rn "def authenticate_user" .  # Exact function name
grep -rn "REDIS_HOST" .             # Exact environment variable
```

## Quick Decision Heuristic

| Question Type | Tool |
|--------------|------|
| **UNSURE / GENERAL QUERY** | **MCP `search`** — **RECOMMENDED DEFAULT, auto-routes to best tool** |
| "Where is X implemented?" | MCP `search` or `repo_search` |
| "Search across multiple repos" | MCP `cross_repo_search` — **PRIMARY for multi-repo, prefer over manual chains** |
| "Trace frontend→backend flow" | MCP `cross_repo_search(trace_boundary=true)` — auto-extracts boundary keys |
| "Who calls this and show code?" | MCP `symbol_graph` — **DEFAULT for all graph queries, always available** |
| "What does this call?" | MCP `symbol_graph` (query_type="callees") |
| "Where is X defined?" | MCP `symbol_graph` (query_type="definition") |
| "What imports X?" | MCP `symbol_graph` (query_type="importers") |
| "Callers of callers? Multi-hop?" | MCP `symbol_graph` (depth=2+) or `graph_query` |
| "What breaks if I change X?" | MCP `graph_query` (impact analysis, transitive callers) |
| "Circular dependencies?" | MCP `graph_query` (query_type="cycles") |
| "N independent searches at once?" | `batch_search` (~75% token savings) |
| "N symbol queries at once?" | `batch_symbol_graph` (~75% token savings) |
| "N graph queries at once?" | `batch_graph_query` (~75% token savings) |
| "How does authentication work?" | MCP `context_answer` |
| "High-level module overview?" | MCP `info_request` (with explanations) |
| "Does REDIS_HOST exist?" | Literal grep |
| "Why did behavior change?" | `search_commits_for` + `change_history_for_path` |

> **`symbol_graph`** is ALWAYS available (Qdrant-backed). **`graph_query`** is available to all SaaS users (Memgraph-backed) for advanced traversals: impact analysis, circular dependencies, transitive callers/callees. In self-hosted, requires `NEO4J_GRAPH=1` or `MEMGRAPH_GRAPH=1`. If `graph_query` is not in your tool list, use `symbol_graph` for everything.

> **Batch tools** (`batch_search`, `batch_symbol_graph`, `batch_graph_query`) run N independent queries in one MCP invocation with ~75% token savings. Max 10 queries per batch. Use when you have 2+ independent queries of the same type.

> **`cross_repo_search`** discovery modes: `"auto"` (default), `"always"` (force discovery), `"never"` (skip). Use `trace_boundary=true` to extract API routes, event names, types.

**If in doubt → start with `search`** (unified MCP tool that auto-routes to the best specialized tool)

