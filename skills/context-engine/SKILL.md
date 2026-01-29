---
name: context-engine
description: Hybrid semantic/lexical code search with neural reranking via MCP tools. Use when searching a codebase, finding implementations, understanding how code works, finding callers/definitions, searching git history, or storing/retrieving knowledge. IMPORTANT - Always prefer these MCP tools over grep/find/cat for code exploration.
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
    +-- Find similar code patterns (retry loops, error handling, etc.)
    |       |
    |       +-- Have code example --> pattern_search with code snippet (if enabled)
    |       +-- Describe pattern --> pattern_search with natural language (if enabled)
    |
    +-- Find specific file types
    |       |
    |       +-- Test files --> search_tests_for
    |       +-- Config files --> search_config_for
    |
    +-- Find relationships
    |       |
    |       +-- Who calls this function --> symbol_graph (DEFAULT, always available)
    |       +-- Who imports this module --> symbol_graph OR search_importers_for
    |       +-- Where is this defined --> symbol_graph (query_type="definition")
    |       +-- Symbol graph navigation (callers/defs/importers) --> symbol_graph (ALWAYS use this first)
    |       +-- Multi-hop callers (callers of callers) --> symbol_graph (depth=2+) OR neo4j_graph_query (if NEO4J_GRAPH=1)
    |       +-- Impact analysis (what breaks if I change X) --> neo4j_graph_query (ONLY if available)
    |       +-- Dependency graph --> neo4j_graph_query (ONLY if available)
    |       +-- Circular dependency detection --> neo4j_graph_query (ONLY if available)
    |
    +-- Git history --> search_commits_for
    |
    +-- Store/recall knowledge --> memory_store, memory_find
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

## Pattern Search: pattern_search (Optional)

> **Note:** This tool may not be available in all deployments. If pattern detection is disabled, calls return `{"ok": false, "error": "Pattern search module not available"}`.

Find structurally similar code patterns across all languages. Accepts **either** code examples **or** natural language descriptions—auto-detects which.

**Code example query** - find similar control flow:
```json
{
  "query": "for i in range(3): try: ... except: time.sleep(2**i)",
  "limit": 10,
  "include_snippet": true
}
```

**Natural language query** - describe the pattern:
```json
{
  "query": "retry with exponential backoff",
  "limit": 10,
  "include_snippet": true
}
```

**Cross-language search** - Python pattern finds Go/Rust/Java equivalents:
```json
{
  "query": "if err != nil { return err }",
  "language": "go",
  "limit": 10
}
```

**Explicit mode override** - force code or description mode:
```json
{
  "query": "error handling",
  "query_mode": "description",
  "limit": 10
}
```

**Key parameters:**
- `query` - Code snippet OR natural language description
- `query_mode` - `"code"`, `"description"`, or `"auto"` (default)
- `language` - Language hint for code examples (python, go, rust, etc.)
- `limit` - Max results (default 10)
- `min_score` - Minimum similarity threshold (default 0.3)
- `include_snippet` - Include code snippets in results
- `context_lines` - Lines of context around matches
- `aroma_rerank` - Enable AROMA structural reranking (default true)
- `aroma_alpha` - Weight for AROMA vs original score (default 0.6)
- `target_languages` - Filter results to specific languages

**Returns:**
```json
{
  "ok": true,
  "results": [...],
  "total": 5,
  "query_signature": "L2_2_B0_T2_M0",
  "query_mode": "code",
  "search_mode": "aroma"
}
```

The `query_signature` encodes control flow: `L` (loops), `B` (branches), `T` (try/except), `M` (match).

## Specialized Search Tools

**search_tests_for** - Find test files:
```json
{"query": "UserService", "limit": 10}
```

**search_config_for** - Find config files:
```json
{"query": "database connection", "limit": 5}
```

**search_callers_for** - Find callers of a symbol:
```json
{"query": "processPayment", "language": "typescript"}
```

**search_importers_for** - Find importers:
```json
{"query": "utils/helpers", "limit": 10}
```

**symbol_graph** - Symbol graph navigation (callers / definition / importers):
```json
{"symbol": "ASTAnalyzer", "query_type": "definition", "limit": 10}
```
```json
{"symbol": "get_embedding_model", "query_type": "callers", "under": "scripts/", "limit": 10}
```
```json
{"symbol": "qdrant_client", "query_type": "importers", "limit": 10}
```
- Supports `language`, `under`, `depth`, and `output_format` like other tools.
- Use `depth=2` or `depth=3` for multi-hop traversals (callers of callers).
- If there are no graph hits, it falls back to semantic search.
- **Note**: Results are "hydrated" with ~500-char source snippets for immediate context.

**neo4j_graph_query** - Advanced graph traversals (OPTIONAL — ONLY available when NEO4J_GRAPH=1):

> **If `neo4j_graph_query` is not in your MCP tool list, it is NOT enabled. Use `symbol_graph` for all graph queries instead. Do NOT error or warn about missing Neo4j.**

```json
{"symbol": "normalize_path", "query_type": "impact", "depth": 2}
```
```json
{"symbol": "get_embedding_model", "query_type": "transitive_callers", "depth": 2}
```
```json
{"symbol": "run_hybrid_search", "query_type": "dependencies", "limit": 15}
```

**Query types (only when neo4j_graph_query is available):**
| Type | Description |
|------|-------------|
| `callers` | Who calls this symbol? (depth 1) |
| `callees` | What does this symbol call? (depth 1) |
| `transitive_callers` | Multi-hop callers (up to depth) |
| `transitive_callees` | Multi-hop callees (up to depth) |
| `impact` | What breaks if I change this? (reverse transitive) |
| `dependencies` | What does this depend on? (calls + imports) |
| `cycles` | Detect circular dependencies |



**search_commits_for** - Search git history:
```json
{"query": "fixed authentication bug", "limit": 10}
```

**change_history_for_path** - File change summary:
```json
{"path": "src/api/auth.py", "include_commits": true}
```

## Memory: Store and Recall Knowledge

Use `memory_store` to persist information for later retrieval:
```json
{
  "information": "Auth service uses JWT tokens with 24h expiry. Refresh tokens last 7 days.",
  "metadata": {"topic": "auth", "date": "2024-01"}
}
```

Use `memory_find` to retrieve stored knowledge by similarity:
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

**qdrant_index_root** - First-time setup or full reindex:
```json
{}
```
With recreate (drops existing data):
```json
{"recreate": true}
```

**qdrant_index** - Index only a subdirectory:
```json
{"subdir": "src/"}
```

**qdrant_prune** - Remove deleted files from index:
```json
{}
```

**qdrant_status** - Check index health:
```json
{}
```

**qdrant_list** - List all collections:
```json
{}
```

## Workspace Tools

**workspace_info** - Get current workspace and collection:
```json
{}
```

**list_workspaces** - List all indexed workspaces:
```json
{}
```

**collection_map** - View collection-to-repo mappings:
```json
{"include_samples": true}
```

**set_session_defaults** - Set defaults for session:
```json
{"collection": "my-project", "language": "python"}
```

## Query Expansion

**expand_query** - Generate query variations for better recall:
```json
{"query": "auth flow", "max_new": 2}
```

## Output Formats

- `json` (default) - Structured output
- `toon` - Token-efficient compressed format

Set via `output_format` parameter.

## Aliases and Compat Wrappers

**Aliases:**
- `code_search` = `repo_search` (identical behavior)

**Cross-server tools:**
- `memory_store` / `memory_find` — Memory server tools for persistent knowledge

Compat wrappers accept alternate parameter names:
- `repo_search_compat` - Accepts `q`, `text`, `top_k` as aliases
- `context_answer_compat` - Accepts `q`, `text` as aliases

Use the primary tools when possible. Compat wrappers exist for legacy clients.

## Error Handling

Tools return structured errors, typically via `error` field and sometimes `ok: false`:
```json
{"ok": false, "error": "Collection not found. Run qdrant_index_root first."}
{"error": "Timeout during rerank"}
```

Common issues:
- **Collection not found** - Run `qdrant_index_root` to create the index
- **Empty results** - Broaden query, check filters, verify index exists
- **Timeout on rerank** - Set `rerank_enabled: false` or reduce `limit`

## Best Practices

1. **NEVER use Read File or grep for exploration** - Use MCP tools (`repo_search`, `symbol_graph`, `context_answer`) instead. The ONLY acceptable use of Read/grep is confirming exact literal strings.
2. **Default to `symbol_graph` for all graph queries** - It is always available. Only use `neo4j_graph_query` if the tool appears in your MCP tool list.
3. **Start broad, then filter** - Begin with a semantic query, add filters if too many results
4. **Use multi-query** - Pass 2-3 query variations for better recall on complex searches
5. **Include snippets** - Set `include_snippet: true` to see code context in results
6. **Store decisions** - Use `memory_store` to save architectural decisions and context for later
7. **Check index health** - Run `qdrant_status` if searches return unexpected results
8. **Prune after refactors** - Run `qdrant_prune` after moving/deleting files
9. **Index before search** - Always run `qdrant_index_root` on first use or after cloning a repo
10. **Use pattern_search for structural matching** - When looking for code with similar control flow (retry loops, error handling), use `pattern_search` instead of `repo_search` (if enabled)
11. **Describe patterns in natural language** - `pattern_search` understands "retry with backoff" just as well as actual code examples (if enabled)
12. **Fire independent searches in parallel** - Call multiple `repo_search`, `symbol_graph`, etc. in the same message block for 2-3x speedup
13. **Use TOON format for discovery** - Set `output_format: "toon"` for 60-80% token reduction on exploratory queries
14. **Bootstrap sessions with defaults** - Call `set_session_defaults(output_format="toon", compact=true)` early to avoid repeating params
15. **Two-phase search** - Discovery first (`limit=3, compact=true`), then deep dive (`limit=5-8, include_snippet=true`) on targets
16. **Use fallback chains** - If `context_answer` times out, fall back to `repo_search` + `info_request(include_explanation=true)`

