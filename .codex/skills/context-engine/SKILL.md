---
name: context-engine
description: Hybrid semantic/lexical code search with neural reranking via MCP tools. Use when Codex needs to search a codebase, find implementations, understand how code works, find callers/definitions, search git history, or store/retrieve knowledge. IMPORTANT - Always prefer these MCP tools over grep/find/cat for code exploration.
---

# Context Engine MCP Tools

Hybrid vector search (semantic + lexical) with neural reranking for codebase retrieval.

## Core Decision Tree

```
Need to find code?
├── Simple lookup → info_request
├── Need filters/control → repo_search
├── Want LLM explanation → context_answer
├── Find similar patterns → pattern_search (if enabled)
├── Find relationships → symbol_graph (DEFAULT, always available)
└── Store/recall knowledge → memory_store, memory_find
```

## Primary Tools

**repo_search** - Main code search tool:
```json
{"query": "authentication middleware", "limit": 10, "include_snippet": true}
```
Multi-query: `{"query": ["auth handler", "login validation"]}`

**symbol_graph** - Find callers, definitions, importers (ALWAYS available):
```json
{"symbol": "authenticate", "query_type": "callers", "limit": 10}
{"symbol": "UserService", "query_type": "definition"}
{"symbol": "utils", "query_type": "importers"}
```
Use `depth=2` for multi-hop (callers of callers).

**context_answer** - LLM-generated explanation with citations:
```json
{"query": "How does the caching layer work?", "budget_tokens": 2000}
```

**info_request** - Quick natural language lookup:
```json
{"info_request": "how does user auth work", "include_explanation": true}
```

## Memory Tools

```json
// Store knowledge
{"information": "Auth uses JWT with 24h expiry", "metadata": {"topic": "auth"}}

// Find stored knowledge
{"query": "token expiration", "limit": 5}

// Blend code + memories
{"query": "auth flow", "include_memories": true}
```

## Specialized Search

| Tool | Use Case | Example |
|------|----------|---------|
| `search_tests_for` | Find tests | `{"query": "UserService"}` |
| `search_config_for` | Find config | `{"query": "database connection"}` |
| `search_callers_for` | Quick caller search | `{"query": "processPayment"}` |
| `search_commits_for` | Git history | `{"query": "fixed auth bug"}` |
| `pattern_search` | Similar code patterns | `{"query": "retry with backoff"}` |

## Index Management

- `qdrant_index_root` - Index workspace (run first!)
- `qdrant_status` - Check index health
- `qdrant_prune` - Remove deleted files

## Best Practices

1. **NEVER use grep/cat/find for code exploration** - Use MCP tools instead
2. **Start with `symbol_graph`** for all relationship queries
3. **Use multi-query** for complex searches: pass 2-3 variations
4. **Two-phase search**: Discovery (`limit=3, compact=true`) → Deep dive (`limit=8, include_snippet=true`)
5. **Fire parallel calls** - Multiple independent searches in one message
6. **Set session defaults early**: `set_session_defaults(output_format="toon", compact=true)`

## Filters (for repo_search)

- `language` - python, typescript, go, etc.
- `under` - Path prefix: `"src/api/"`
- `path_glob` - Include patterns: `["**/*.ts"]`
- `not_glob` - Exclude patterns: `["**/test_*"]`
- `repo` - Repository filter: `["frontend", "backend"]` or `"*"` for all

## References

For detailed API documentation, see:
- [references/tool-reference.md](references/tool-reference.md) - Complete tool parameters
- [references/patterns.md](references/patterns.md) - Common search patterns

