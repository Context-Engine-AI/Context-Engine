name: context-engine
description: Performs hybrid semantic/lexical search with neural reranking for codebase retrieval. Use for finding implementations, Q&A grounded in source code, and cross-session persistent memory.
---

# Context-Engine

Search and retrieve code context from any codebase using hybrid vector search (semantic + lexical) with neural reranking.

## Decision Tree: Choosing the Right Tool

```
What do you need?
    |
    +-- UNSURE or GENERAL QUERY --> search (RECOMMENDED DEFAULT)
    |       |
    |       +-- Auto-detects intent and routes to the best tool
    |       +-- Handles: code search, Q&A, tests, config, symbols, imports
    |       +-- Use this when you don't know which specialized tool to pick
    |
    +-- Find code locations/implementations
    |       |
    |       +-- Simple query --> search OR info_request
    |       +-- Need filters/control --> search OR repo_search
    |
    +-- Understand how something works
    |       |
    |       +-- Want LLM explanation --> search OR context_answer
    |       +-- Just code snippets --> search OR repo_search with include_snippet=true
    |
    +-- Find similar code patterns (retry loops, error handling, etc.)
    |       |
    |       +-- Have code example --> pattern_search with code snippet (if enabled)
    |       +-- Describe pattern --> pattern_search with natural language (if enabled)
    |
    +-- Find specific file types
    |       |
    |       +-- Test files --> search OR search_tests_for
    |       +-- Config files --> search OR search_config_for
    |
    +-- Find relationships
    |       |
    |       +-- Who calls this function --> search OR symbol_graph (DEFAULT, always available)
    |       +-- Who imports this module --> search OR symbol_graph OR search_importers_for
    |       +-- Where is this defined --> symbol_graph (query_type="definition")
    |       +-- Find subclasses --> symbol_graph (query_type="subclasses")
    |       +-- Find base classes --> symbol_graph (query_type="base_classes")
    |       +-- Symbol graph navigation (callers/defs/importers/subclasses) --> symbol_graph (ALWAYS use this first)
    |       +-- Multi-hop callers (callers of callers) --> symbol_graph (depth=2+)
    |
    +-- Git history
    |       |
    |       +-- Find commits --> search_commits_for
    |       +-- Predict co-changing files --> search_commits_for with predict_related=true
    |
    +-- Store/recall knowledge --> memory_store, memory_find
    |
    +-- Blend code + notes --> context_search with include_memories=true
    |
    +-- Multiple independent queries at once
            |
            +-- batch_search (runs N repo_search calls in one invocation, ~75% token savings)
            +-- batch_symbol_graph (runs N symbol_graph queries in one invocation)
            +-- batch_graph_query (runs N graph_query queries in one invocation)
```

## Unified Search: search (RECOMMENDED DEFAULT)

**Use `search` as your PRIMARY tool.** It auto-detects query intent and routes to the best specialized tool. No need to choose between 15+ tools.

```json
{
  "query": "authentication middleware"
}
```

Returns:
```json
{
  "ok": true,
  "intent": "search",
  "confidence": 0.92,
  "tool": "repo_search",
  "result": {
    "results": [...],
    "total": 8
  },
  "plan": ["detect_intent", "dispatch_repo_search"],
  "execution_time_ms": 245
}
```

**What it handles automatically:**
- Code search ("find auth middleware") -> routes to `repo_search`
- Q&A ("how does caching work?") -> routes to `context_answer`
- Test discovery ("tests for payment") -> routes to `search_tests_for`
- Config lookup ("database settings") -> routes to `search_config_for`
- Symbol queries ("who calls authenticate") -> routes to `symbol_graph`
- Import tracing ("what imports CacheManager") -> routes to `search_importers_for`

**Override parameters** (all optional):
```json
{
  "query": "error handling patterns",
  "limit": 5,
  "language": "python",
  "under": "src/api/",
  "include_snippet": true
}
```

**When to use specialized tools instead:**
- Cross-repo search -> `cross_repo_search`
- Multiple independent searches -> `batch_search` (N searches in one call, ~75% token savings)
- Multiple independent symbol queries -> `batch_symbol_graph` (N symbol_graph queries in one call)
- Multiple independent graph queries -> `batch_graph_query` (N graph_query queries in one call)
- Advanced graph traversal / impact analysis -> `graph_query`
- Memory storage/retrieval -> `memory_store`, `memory_find`
- Admin/diagnostics -> `qdrant_status`, `qdrant_list`
- Pattern matching (structural) -> `pattern_search`

## Primary Search: repo_search

Use `repo_search` (or its alias `code_search`) for direct code lookups when you need full control. Reranking is ON by default.

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

**Search across repos** (same collection):
```json
{
  "query": "shared types",
  "repo": ["frontend", "backend"]
}
```
Use `repo: "*"` to search all indexed repos.

**Search across repos** (separate collections — use `cross_repo_search`):
```json
// cross_repo_search
{"query": "shared types", "target_repos": ["frontend", "backend"]}
// With boundary tracing for cross-repo flow discovery
{"query": "login submit", "trace_boundary": true}
```

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

## Batch Search: batch_search

Run N independent `repo_search` calls in a single MCP tool invocation. Reduces token overhead by ~75-85% compared to sequential calls.

```json
{
  "searches": [
    {"query": "authentication middleware", "limit": 5},
    {"query": "rate limiting implementation", "limit": 5},
    {"query": "error handling patterns"}
  ],
  "compact": true,
  "output_format": "toon"
}
```

Returns:
```json
{
  "ok": true,
  "batch_results": [result_set_0, result_set_1, result_set_2],
  "count": 3,
  "elapsed_ms": 245
}
```

Each `result_set` has the same schema as `repo_search` output.

**Shared parameters** (applied to all searches unless overridden per-search):
- `collection`, `output_format`, `compact`, `limit`, `language`, `under`, `repo`, `include_snippet`, `rerank_enabled`

**Per-search overrides**: Each entry in `searches` can include any `repo_search` parameter to override the shared defaults.

**Limits**: Maximum 10 searches per batch.

**When to use `batch_search` vs multiple `search` calls:**
- Use `batch_search` when you have 2+ independent code searches and want to minimize token usage and round-trips
- Use individual `search` calls when you need intent routing (Q&A, symbol graph, etc.) or when searches depend on each other's results

## Batch Symbol Graph: batch_symbol_graph

Run N independent `symbol_graph` queries in a single MCP tool invocation. Same ~75-85% token savings as `batch_search`.

```json
{
  "queries": [
    {"symbol": "authenticate", "query_type": "callers"},
    {"symbol": "CacheManager", "query_type": "definition"},
    {"symbol": "BaseModel", "query_type": "subclasses"}
  ],
  "limit": 10
}
```

Returns:
```json
{
  "ok": true,
  "batch_results": [result_set_0, result_set_1, result_set_2],
  "count": 3,
  "elapsed_ms": 180
}
```

Each `result_set` has the same schema as `symbol_graph` output.

**Shared parameters** (applied to all queries unless overridden per-query):
- `collection`, `language`, `under`, `repo`, `limit`, `depth`, `output_format`

**Per-query overrides**: Each entry in `queries` must have a `symbol` key and can include any `symbol_graph` parameter (`query_type`, `depth`, `limit`, etc.) to override shared defaults.

**Limits**: Maximum 10 queries per batch.

**When to use `batch_symbol_graph` vs multiple `symbol_graph` calls:**
- Use `batch_symbol_graph` when you need callers/definitions/importers for multiple symbols at once
- Use individual `symbol_graph` calls when queries depend on each other's results

## Batch Graph Query: batch_graph_query

Run N independent `graph_query` calls in a single MCP tool invocation. Same ~75-85% token savings.

```json
{
  "queries": [
    {"symbol": "User", "query_type": "impact", "depth": 3},
    {"symbol": "auth", "query_type": "cycles"},
    {"symbol": "PaymentService", "query_type": "transitive_callers"}
  ],
  "limit": 15
}
```

Returns:
```json
{
  "ok": true,
  "batch_results": [result_set_0, result_set_1, result_set_2],
  "count": 3,
  "elapsed_ms": 250
}
```

Each `result_set` has the same schema as `graph_query` output.

**Shared parameters** (applied to all queries unless overridden per-query):
- `collection`, `repo`, `language`, `depth`, `limit`, `include_paths`, `output_format`

**Per-query overrides**: Each entry in `queries` must have a `symbol` key and can include any `graph_query` parameter (`query_type`, `depth`, `include_paths`, etc.) to override shared defaults.

**Limits**: Maximum 10 queries per batch.

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

**symbol_graph** - Symbol graph navigation (callers / callees / definition / importers / subclasses / base classes):

**Query types:**
| Type | Description |
|------|-------------|
| `callers` | Who calls this symbol? |
| `callees` | What does this symbol call? |
| `definition` | Where is this symbol defined? |
| `importers` | Who imports this module/symbol? |
| `subclasses` | What classes inherit from this symbol? |
| `base_classes` | What classes does this symbol inherit from? |

**Examples:**
```json
{"symbol": "ASTAnalyzer", "query_type": "definition", "limit": 10}
```
```json
{"symbol": "get_embedding_model", "query_type": "callers", "under": "scripts/", "limit": 10}
```
```json
{"symbol": "qdrant_client", "query_type": "importers", "limit": 10}
```
```json
{"symbol": "authenticate", "query_type": "callees", "limit": 10}
```
```json
{"symbol": "BaseModel", "query_type": "subclasses", "limit": 20}
```
```json
{"symbol": "MyService", "query_type": "base_classes"}
```
- Supports `language`, `under`, `depth`, and `output_format` like other tools.
- Use `depth=2` or `depth=3` for multi-hop traversals (callers of callers).
- If there are no graph hits, it falls back to semantic search.
- **Note**: Results are "hydrated" with ~500-char source snippets for immediate context.

**graph_query** - Advanced graph traversals and impact analysis (available to all SaaS users):

**Query types:**
| Type | Description |
|------|-------------|
| `callers` | Direct callers of this symbol |
| `callees` | Direct callees of this symbol |
| `transitive_callers` | Multi-hop callers (up to depth) |
| `transitive_callees` | Multi-hop callees (up to depth) |
| `impact` | What would break if I change this symbol? |
| `dependencies` | Combined calls + imports |
| `definition` | Where is this symbol defined? |
| `cycles` | Detect circular dependencies involving this symbol |

**Examples:**
```json
{"symbol": "UserService", "query_type": "impact", "depth": 3}
```
```json
{"symbol": "auth_module", "query_type": "cycles"}
```
```json
{"symbol": "processPayment", "query_type": "transitive_callers", "depth": 2, "limit": 20}
```
- Supports `language`, `under`, `depth`, `limit`, `include_paths`, and `output_format`.
- Use `include_paths: true` to get full traversal paths in results.
- Use `depth` to control how many hops to traverse (default varies by query type).
- **Note**: `symbol_graph` is always available (Qdrant-backed). `graph_query` provides advanced Memgraph-backed traversals and is available to all SaaS users.


**search_commits_for** - Search git history:
```json
{"query": "fixed authentication bug", "limit": 10}
```

**Predict co-changing files** (predict_related mode):
```json
{"path": "src/api/auth.py", "predict_related": true, "limit": 10}
```
Returns ranked files that historically co-change with the given path, along with the most relevant commit message explaining why.

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

## Admin and Diagnostics

> **SaaS mode:** In SaaS deployments, indexing is handled automatically by the VS Code extension upload service. The tools `qdrant_index_root`, `qdrant_index`, and `qdrant_prune` are **not available** in SaaS mode. All search, symbol graph, memory, and session tools work normally.

**Available in all modes:**

**qdrant_status** - Check index health:
```json
{}
```

**qdrant_list** - List all collections:
```json
{}
```

**embedding_pipeline_stats** - Get cache efficiency, bloom filter stats, pipeline performance:
```json
{}
```

**set_session_defaults** - Set defaults for session:
```json
{"collection": "my-project", "language": "python"}
```

**Self-hosted only (not available in SaaS):**

**qdrant_index_root** - Index entire workspace:
```json
{"recreate": true}
```

**qdrant_index** - Index subdirectory:
```json
{"subdir": "src/"}
```

**qdrant_prune** - Remove stale entries from deleted files:
```json
{}
```

## Multi-Repo Navigation (CRITICAL)

When multiple repositories are indexed, you MUST discover and explicitly target collections.

### Discovery (Lazy — only when needed)

Don't discover at every session start. Trigger when: search returns no/irrelevant results, user asks a cross-repo question, or you're unsure which collection to target.

```json
// qdrant_list — discover available collections
{}
```

### Context Switching (Session Defaults = `cd`)

Treat `set_session_defaults` like `cd` — it scopes ALL subsequent searches:
```json
// "cd" into backend repo — all searches now target this collection
// set_session_defaults
{"collection": "backend-api-abc123"}

// One-off peek at another repo (does NOT change session default)
// search (or repo_search)
{"query": "login form", "collection": "frontend-app-def456"}
```

For unified collections: use `"repo": "*"` or `"repo": ["frontend", "backend"]`

### Cross-Repo Flow Tracing (Boundary-Driven)

NEVER search both repos with the same vague query. Find the **interface boundary** in Repo A, extract the **hard key**, then search Repo B with that specific key.

**Pattern 1 — Interface Handshake (API/RPC):**
```json
// 1. Find client call in frontend
// search
{"query": "login API call", "collection": "frontend-col"}
// → Found: axios.post('/auth/v1/login', ...)

// 2. Search backend for that exact route
// search
{"query": "'/auth/v1/login'", "collection": "backend-col"}
```

**Pattern 2 — Shared Contract (Types/Schemas):**
```json
// 1. Find type usage in consumer
// symbol_graph
{"symbol": "UserProfile", "query_type": "importers", "collection": "frontend-col"}

// 2. Find definition in source
// search
{"query": "interface UserProfile", "collection": "shared-lib-col"}
```

**Pattern 3 — Event Relay (Pub/Sub):**
```json
// 1. Find producer → extract event name
// search
{"query": "publish event", "collection": "service-a-col"}
// → Found: bus.publish("USER_CREATED", payload)

// 2. Find consumer with exact event name
// search
{"query": "'USER_CREATED'", "collection": "service-b-col"}
```

### Automated Cross-Repo Search (PRIMARY for Multi-Repo)

`cross_repo_search` is the PRIMARY tool for multi-repo scenarios. Use it BEFORE manual `qdrant_list` + `repo_search` chains.

**Discovery Modes:**
| Mode | Behavior | When to Use |
|------|----------|-------------|
| `"auto"` (default) | Discovers only if results empty or no targeting | Normal usage |
| `"always"` | Always runs discovery before search | First search in session, exploring new codebase |
| `"never"` | Skips discovery, uses explicit collection | When you know exact collection, speed-critical |

```json
// Search across all repos at once (auto-discovers collections)
// cross_repo_search
{"query": "authentication flow", "discover": "auto"}

// Target specific repos by name
// cross_repo_search
{"query": "login handler", "target_repos": ["frontend", "backend"]}

// Boundary tracing — auto-extracts routes/events/types from results
// cross_repo_search
{"query": "login submit", "trace_boundary": true}
// → Returns boundary_keys: ["/api/auth/login"] + trace_hint for next search

// Follow boundary key to another repo
// cross_repo_search
{"boundary_key": "/api/auth/login", "collection": "backend-col"}
```
Use `cross_repo_search` when you need breadth across repos. Use `search` (or `repo_search`) with explicit `collection` when you need depth in one repo.

### Multi-Repo Anti-Patterns
- **DON'T** search both repos with the same vague query (noisy, confusing)
- **DON'T** assume the default collection is correct — verify with `qdrant_list`
- **DON'T** forget to "cd back" after cross-referencing another repo
- **DO** extract exact strings (route paths, event names, type names) as search anchors

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
{"ok": false, "error": "Collection not found."}
{"error": "Timeout during rerank"}
```

Common issues:
- **Collection not found** - Verify collection with `qdrant_list` or check that the codebase has been indexed
- **Empty results** - Broaden query, check filters, verify index exists
- **Timeout on rerank** - Set `rerank_enabled: false` or reduce `limit`

## Best Practices

1. **Use `search` as your default tool** - It auto-routes to the best specialized tool. Only use specific tools when you need precise control or features `search` doesn't handle (cross-repo, memory, admin).
2. **NEVER use Read File or grep for exploration** - Use MCP tools (`search`, `repo_search`, `symbol_graph`, `context_answer`) instead. The ONLY acceptable use of Read/grep is confirming exact literal strings.
3. **Use `symbol_graph` for graph queries** - It handles callers, callees, definitions, importers, subclasses, and base classes. Use `graph_query` for advanced traversals: impact analysis, circular dependency detection, and transitive callers/callees.
4. **Start broad, then filter** - Begin with `search` or a semantic query, add filters if too many results
5. **Use multi-query** - Pass 2-3 query variations for better recall on complex searches
6. **Include snippets** - Set `include_snippet: true` to see code context in results
7. **Store decisions** - Use `memory_store` to save architectural decisions and context for later
8. **Check index health** - Run `qdrant_status` if searches return unexpected results
11. **Use pattern_search for structural matching** - When looking for code with similar control flow (retry loops, error handling), use `pattern_search` instead of `repo_search` (if enabled)
12. **Describe patterns in natural language** - `pattern_search` understands "retry with backoff" just as well as actual code examples (if enabled)
13. **Fire independent searches in parallel** - Call multiple `search`, `repo_search`, `symbol_graph`, etc. in the same message block for 2-3x speedup. Or use batch tools (`batch_search`, `batch_symbol_graph`, `batch_graph_query`) to run N queries in a single invocation with ~75% token savings
14. **Use TOON format for discovery** - Set `output_format: "toon"` for 60-80% token reduction on exploratory queries
15. **Bootstrap sessions with defaults** - Call `set_session_defaults(output_format="toon", compact=true)` early to avoid repeating params
16. **Two-phase search** - Discovery first (`limit=3, compact=true`), then deep dive (`limit=5-8, include_snippet=true`) on targets
17. **Use fallback chains** - If `context_answer` times out, fall back to `search` or `repo_search` + `info_request(include_explanation=true)`

