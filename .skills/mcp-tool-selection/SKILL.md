---
name: mcp-tool-selection
description: Decision rules for when to use MCP Qdrant-Indexer semantic search vs grep/literal file tools. Use this skill when starting exploration, debugging, or answering "where/why" questions about code.
---

# MCP Tool Selection Rules

**Core principle:** MCP Qdrant-Indexer tools are primary for exploring code and history. Start with MCP for exploration, debugging, or "where/why" questions; use literal search/file-open only for narrow exact-literal lookups.

## Use MCP Qdrant-Indexer When

- Exploring or don't know exact strings/symbols
- Need semantic or cross-file understanding (relationships, patterns, architecture)
- Want ranked results with surrounding context, not just line hits
- Asking conceptual/architectural or "where/why" behavior questions
- Need rich context/snippets around matches

## Use Literal Search/File-Open Only When

- Know exact string/function/variable or error message
- Only need to confirm existence or file/line quickly (not to understand behavior)

## Grep Anti-Patterns (DON'T)

```bash
grep -r "auth" .        # → Use MCP: "authentication mechanisms"
grep -r "cache" .       # → Use MCP: "caching strategies"  
grep -r "error" .       # → Use MCP: "error handling patterns"
grep -r "database" .    # → Use MCP: "database operations"
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
| "Where is X implemented?" | MCP `repo_search` |
| "Who calls this and show code?" | MCP `symbol_graph` (hydrated w/ snippets) |
| "Callers of callers? Multi-hop?" | MCP `neo4j_graph_query` (transitive_callers, depth=2) |
| "What breaks if I change X?" | MCP `neo4j_graph_query` (impact, depth=2) |
| "Circular dependencies?" | MCP `neo4j_graph_query` (cycles) |
| "How does authentication work?" | MCP `context_answer` |
| "High-level module overview?" | MCP `info_request` (with explanations) |
| "Does REDIS_HOST exist?" | Literal grep |
| "Why did behavior change?" | `search_commits_for` + `change_history_for_path` |

**If in doubt → start with MCP**

