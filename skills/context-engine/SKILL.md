---
name: context-engine
description: Codebase search and context retrieval for any programming language. Hybrid semantic/lexical search with neural reranking. Use for code lookup, finding implementations, understanding codebases, Q&A grounded in source code, and persistent memory across sessions.
---

# Context-Engine

Search and retrieve code context from any codebase using hybrid vector search (semantic + lexical) with neural reranking.

## Decision Tree: Choosing the Right Tool

```
What do you need?
    |
    +-- Find code locations/implementations
    |       |
    |       +-- Simple query --> info_request
    |       +-- Need filters/control --> repo_search
    |
    +-- Understand how something works
    |       |
    |       +-- Want LLM explanation --> context_answer
    |       +-- Just code snippets --> repo_search with include_snippet=true
    |
    +-- Find specific file types
    |       |
    |       +-- Test files --> search_tests_for
    |       +-- Config files --> search_config_for
    |
    +-- Find relationships
    |       |
    |       +-- Who calls this function --> search_callers_for
    |       +-- Who imports this module --> search_importers_for
    |
    +-- Git history --> search_commits_for
    |
    +-- Store/recall knowledge --> store, find
    |
    +-- Blend code + notes --> context_search with include_memories=true
```

## Primary Search: repo_search

Use `repo_search` (or its alias `code_search`) for most code lookups. Reranking is ON by default.

```json
{
  "query": "database connection handling",
  "limit": 10,
  "include_snippet": true,
  "context_lines": 3
}
```

Returns:
```json
{
  "results": [
    {"score": 3.2, "path": "src/db/pool.py", "symbol": "ConnectionPool", "start_line": 45, "end_line": 78, "snippet": "..."}
  ],
  "total": 8,
  "used_rerank": true
}
```

**Multi-query for better recall** - pass a list to fuse results:
```json
{
  "query": ["auth middleware", "authentication handler", "login validation"]
}
```

**Apply filters** to narrow results:
```json
{
  "query": "error handling",
  "language": "python",
  "under": "src/api/",
  "not_glob": ["**/test_*", "**/*_test.*"]
}
```

**Search across repos**:
```json
{
  "query": "shared types",
  "repo": ["frontend", "backend"]
}
```
Use `repo: "*"` to search all indexed repos.

### Available Filters

- `language` - Filter by programming language
- `under` - Path prefix (e.g., "src/api/")
- `path_glob` - Include patterns (e.g., ["**/*.ts", "lib/**"])
- `not_glob` - Exclude patterns (e.g., ["**/test_*"])
- `symbol` - Symbol name match
- `kind` - AST node type (function, class, etc.)
- `ext` - File extension
- `repo` - Repository filter for multi-repo setups
- `case` - Case-sensitive matching

## Simple Lookup: info_request

Use `info_request` for natural language queries with minimal parameters:

```json
{
  "info_request": "how does user authentication work"
}
```

Add explanations:
```json
{
  "info_request": "database connection pooling",
  "include_explanation": true
}
```

## Q&A with Citations: context_answer

Use `context_answer` when you need an LLM-generated explanation grounded in code:

```json
{
  "query": "How does the caching layer invalidate entries?",
  "budget_tokens": 2000
}
```

Returns an answer with file/line citations. Use `expand: true` to generate query variations for better retrieval.

## Specialized Search Tools

**Find tests**:
```json
search_tests_for: {"query": "UserService", "limit": 10}
```

**Find config files**:
```json
search_config_for: {"query": "database connection", "limit": 5}
```

**Find callers of a symbol**:
```json
search_callers_for: {"query": "processPayment", "language": "typescript"}
```

**Find importers**:
```json
search_importers_for: {"query": "utils/helpers", "limit": 10}
```

**Search git commits**:
```json
search_commits_for: {"query": "fixed authentication bug", "limit": 10}
```

**File change history**:
```json
change_history_for_path: {"path": "src/api/auth.py", "include_commits": true}
```

## Memory: Store and Recall Knowledge

Use `store` (or `memory_store`) to persist information for later retrieval:
```json
{
  "information": "Auth service uses JWT tokens with 24h expiry. Refresh tokens last 7 days.",
  "metadata": {"topic": "auth", "date": "2024-01"}
}
```

Use `find` to retrieve stored knowledge by similarity:
```json
{"query": "token expiration", "limit": 5}
```

Use `context_search` to blend code results with stored memories:
```json
{
  "query": "authentication flow",
  "include_memories": true,
  "per_source_limits": {"code": 6, "memory": 3}
}
```

## Index Management

**First-time setup or full reindex**:
```json
qdrant_index_root: {}
```

**Rebuild from scratch** (drops existing data):
```json
qdrant_index_root: {"recreate": true}
```

**Index only a subdirectory**:
```json
qdrant_index: {"subdir": "src/"}
```

**Remove deleted files from index**:
```json
qdrant_prune: {}
```

**Check index status**:
```json
qdrant_status: {}
```

**List all collections**:
```json
qdrant_list: {}
```

## Workspace Tools

**Get current workspace info**:
```json
workspace_info: {}
```

**List all indexed workspaces**:
```json
list_workspaces: {}
```

**View collection-to-repo mappings**:
```json
collection_map: {"include_samples": true}
```

**Set session defaults** (collection, filters):
```json
set_session_defaults: {"collection": "my-project", "language": "python"}
```

## Query Expansion

Generate query variations for better recall:
```json
expand_query: {"query": "auth flow", "max_new": 2}
```

## Output Formats

- `json` (default) - Structured output
- `toon` - Token-efficient compressed format

Set via `output_format` parameter.

## Aliases and Compat Wrappers

Some tools have aliases for convenience:
- `code_search` = `repo_search`
- `memory_store` = `store`

Compat wrappers accept alternate parameter names:
- `repo_search_compat` - Accepts `q`, `text`, `top_k` as aliases
- `context_answer_compat` - Accepts `q`, `text` as aliases

Use the primary tools when possible. Compat wrappers exist for legacy clients.

## Error Handling

All tools return structured responses. On failure:
```json
{"ok": false, "error": "Collection not found. Run qdrant_index_root first."}
```

Common issues:
- **Collection not found** - Run `qdrant_index_root` to create the index
- **Empty results** - Broaden query, check filters, verify index exists
- **Timeout on rerank** - Set `rerank_enabled: false` or reduce `limit`

## Best Practices

1. **Start broad, then filter** - Begin with a semantic query, add filters if too many results
2. **Use multi-query** - Pass 2-3 query variations for better recall on complex searches
3. **Include snippets** - Set `include_snippet: true` to see code context in results
4. **Store decisions** - Use `store` to save architectural decisions and context for later
5. **Check index health** - Run `qdrant_status` if searches return unexpected results
6. **Prune after refactors** - Run `qdrant_prune` after moving/deleting files
7. **Index before search** - Always run `qdrant_index_root` on first use or after cloning a repo

