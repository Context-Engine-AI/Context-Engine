---
name: context-engine
description: Performs hybrid semantic/lexical search with neural reranking for codebase retrieval. Use for finding implementations, Q&A grounded in source code, and cross-session persistent memory.
---

# Context-Engine

Search and retrieve code context from any codebase using hybrid vector search (semantic + lexical) with neural reranking.

Client- or provider-specific wrapper files should stay thin and defer to this document for shared MCP tool-selection and search guidance.

## Quickstart

1. Start with `search` for most codebase questions.
2. Use `symbol_graph` first for direct symbol relationships such as callers, definitions, importers, subclasses, and base classes.
3. Use `graph_query` only if that tool is available and you need transitive impact, dependency, or cycle analysis; otherwise combine `symbol_graph` with targeted search.
4. Prefer MCP tools for exploration. Narrow grep/file-open use is still fine for exact literal confirmation, exact file/path confirmation, or opening a file you already identified for editing.
5. Use `cross_repo_search` for multi-repo questions. For public V1 `context_search`, treat `include_memories=true` as compatibility-only: it preserves response shape but keeps results code-only and may add `memory_note`.

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
    |       +-- Unsure what tool to use → search (DEFAULT - routes to repo_search if needed)
    |       +-- Speed-critical or complex filters → repo_search (skip routing overhead)
    |       +-- Want LLM explanation → context_answer
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
    |       +-- Direct callers/defs/importers/inheritance --> search OR symbol_graph
    |       +-- Multi-hop callers --> symbol_graph (depth=2+)
    |       +-- Deep impact/dependencies/cycles --> graph_query (if available) OR symbol_graph + targeted search
    |
    +-- Git history
    |       |
    |       +-- Find commits --> search_commits_for
    |       +-- Predict co-changing files --> search_commits_for with predict_related=true
    |
    +-- Store/recall knowledge --> memory_store, memory_find
    |
    +-- Preserve public search shape while accepting memory flags --> context_search with include_memories=true (compatibility-only in V1)
    |
    +-- Multiple independent searches at once
            |
            +-- batch_search (runs N repo_search calls in one invocation, ~75% token savings)
```

## Standard Parameters Reference

All SaaS-exposed tools organize parameters into families with consistent naming and behavior.

### Family 1: Code Search Tools
**Applies to:** `search`, `repo_search`, `code_search`, `batch_search`, `info_request`, `context_search`

**Standard Parameters:**

| Parameter | Type | Required? | Default | Purpose |
|-----------|------|-----------|---------|---------|
| `query` | string or string[] | YES | — | Single query OR array of queries for fusion |
| `language` | string | optional | (auto-detect) | Filter by language: "python", "typescript", "go", etc. |
| `under` | string | optional | (root) | Path prefix filter, e.g., "src/api/" or "tests/" |
| `path_glob` | string[] | optional | (all) | Include patterns: `["**/*.ts", "lib/**"]` |
| `not_glob` | string[] | optional | (none) | Exclude patterns: `["**/test_*", "**/*_test.*"]` |
| `symbol` | string | optional | (all) | Filter by symbol name (function, class, variable) |
| `kind` | string | optional | (all) | AST node type: "function", "class", "method", "variable" |
| `ext` | string | optional | (all) | File extension: "py", "ts", "go" (alias for language) |
| `repo` | string or string[] | optional | (default) | Repository filter: single repo OR list OR "*" for all |
| `limit` | int | optional | 10 | Max results to return (1-100) |
| `include_snippet` | bool | optional | true | Include code snippets in results |
| `compact` | bool | optional | false | Strip verbose fields from response |
| `output_format` | string | optional | "json" | "json" (structured) OR "toon" (token-efficient) |
| `rerank_enabled` | bool | optional | true | Enable neural reranking (default ON) |
| `case` | string | optional | (insensitive) | "sensitive" for case-sensitive matching |
| `context_lines` | int | optional | 2 | Lines of context around matches |
| `per_path` | int | optional | 2 | Max results per file |

**Standard Constraints:**
- `limit` max 100 (higher values slow queries)
- `query` max 400 characters / 50 words
- `language` must be valid code language or auto-detection will fail silently
- `path_glob` / `not_glob` support glob patterns (*, **, ?)
- Multiple `query` terms are fused via Reciprocal Rank Fusion (RRF) for better recall

### Family 2: Symbol Graph Tools
**Applies to:** `symbol_graph`, `batch_symbol_graph`, `graph_query`, `batch_graph_query`

**Standard Parameters:**

| Parameter | Type | Required? | Default | Purpose |
|-----------|------|-----------|---------|---------|
| `symbol` | string | YES | — | Symbol name to analyze (e.g., "authenticate", "UserService.get_user") |
| `query_type` | string | optional | "callers" | "callers", "callees", "definition", "importers", "subclasses", "base_classes", "impact", "cycles", "transitive_callers", "transitive_callees", "dependencies" |
| `depth` | int | optional | 1 | Traversal depth: 1=direct, 2=callers of callers, etc. (symbol_graph: max 3, graph_query: max 5+) |
| `language` | string | optional | (auto) | Filter by language for multi-language codebases |
| `under` | string | optional | (all) | Path prefix filter |
| `limit` | int | optional | 20 | Max results to return |
| `include_paths` | bool | optional | false | Include full traversal paths (graph_query only) |
| `output_format` | string | optional | "json" | "json" or "toon" |
| `repo` | string | optional | (default) | Repository filter |
| `collection` | string | optional | (session) | Target collection (use session defaults) |

**Standard Constraints:**
- `symbol` must be exact match or use fuzzy fallback
- `query_type` is case-sensitive
- `depth > 3` may be slow on large graphs
- Results are auto-hydrated with code snippets

### Family 3: Specialized Search Tools
**Applies to:** `search_tests_for`, `search_config_for`, `search_callers_for`, `search_importers_for`, `search_commits_for`

**Standard Parameters:**

| Parameter | Type | Required? | Default | Purpose |
|-----------|------|-----------|---------|---------|
| `query` | string | YES | — | Natural language or symbol name |
| `limit` | int | optional | 10 | Max results to return |
| `language` | string | optional | (auto) | Filter by language |
| `under` | string | optional | (all) | Path prefix filter |

**Additional Parameters:**

| Tool | Extra Parameters |
|------|------------------|
| `search_commits_for` | `path` (optional), `predict_related` (bool, default false) |
| All others | (inherit code search family) |

### Family 4: Memory Tools
**Applies to:** `memory_store`, `memory_find`

**Standard Parameters:**

| Parameter | Type | Required? | Default | Purpose |
|-----------|------|-----------|---------|---------|
| `information` | string | YES (store) | — | Knowledge to persist (clear, self-contained) |
| `query` | string | YES (find) | — | Search for stored knowledge by similarity |
| `metadata` | dict | optional (store) | {} | Structured metadata: kind, topic, priority (1-5), tags, author |
| `kind` | string | optional (find) | (all) | Filter by kind: "memory", "note", "decision", "convention", "gotcha", "policy" |
| `topic` | string | optional (find) | (all) | Filter by topic: "auth", "database", "api", "caching", etc. |
| `tags` | string or string[] | optional (find) | (all) | Filter by tags: ["security", "sql", ...] |
| `priority_min` | int | optional (find) | 1 | Minimum priority threshold (1-5) |
| `limit` | int | optional | 10 | Max results to return |

### Family 5: Batch Tools
**Applies to:** `batch_search`, `batch_symbol_graph`, `batch_graph_query`

**Standard Parameters (Shared across all queries):**

| Parameter | Type | Purpose |
|-----------|------|---------|
| `searches` / `queries` | array | Array of individual search/query specs (max 10 items) |
| `collection` | string | Shared default collection for all queries |
| `language` | string | Shared default language filter |
| `under` | string | Shared default path prefix |
| `limit` | int | Shared default result limit |
| `output_format` | string | "json" or "toon" for all results |

**Per-Search Overrides:**
Each item in `searches` / `queries` can override ANY shared parameter.

Example: `searches[0]` has different `limit` than `searches[1]`

### Family 6: Cross-Repo & Admin Tools
**Applies to:** `cross_repo_search`, `qdrant_status`, `qdrant_list`, `set_session_defaults`

**Standard Parameters:**

| Tool | Parameters |
|------|------------|
| `cross_repo_search` | `query`, `collection`, `target_repos`, `discover`, `trace_boundary`, `boundary_key` |
| `qdrant_status` / `qdrant_list` | (no parameters) |
| `set_session_defaults` | `collection`, `language`, `under`, `output_format`, `limit` |

---

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

**When to use `search`:**
- You're unsure which specialized tool to use
- You want intent auto-detection (routing to repo_search, context_answer, symbol_graph, tests, etc.)
- Acceptable latency overhead: ~50-100ms for routing + tool execution
- You're doing exploratory queries where routing overhead is negligible

**When NOT to use `search`:**
- You know you need raw code results (use `repo_search` directly)
- Time is critical (<100ms target) and routing overhead matters
- You're in a tight loop doing 10+ sequential searches (use `batch_search` instead)

**Routing Performance:**
- Intent detection: ~10-20ms
- Tool dispatch: ~5-10ms
- Total routing overhead: ~20-40ms typical, up to ~100ms worst-case
- For time-critical loops: skip routing with `repo_search` directly

**When to use specialized tools instead:**
- Cross-repo search -> `cross_repo_search`
- Multiple independent searches -> `batch_search` (N searches in one call, ~75% token savings)
- Memory storage/retrieval -> `memory_store`, `memory_find`
- Admin/diagnostics -> `qdrant_status`, `qdrant_list`
- Pattern matching (structural) -> `pattern_search`

**When to use `repo_search` instead of `search`:**
- **Full control over filters**: You know exactly what you're searching for and want to apply specific language, path, or symbol filters without auto-detection overhead
  - Example: "In a polyglot repo, I need Python code only" → use `repo_search` with `language="python"` to avoid search's auto-detected `language=javascript`
  - Example: "Find only test files matching a pattern" → use `repo_search` with `path_glob="**/test_*.py"` directly
- **Speed-critical queries** (<100ms target): You can't afford the ~20-40ms routing overhead
  - Example: Time-sensitive tool loops where each query must complete in <50ms
- **Complex filter combinations**: You need `language` + `under` + `not_glob` together, not guessed by auto-detection
- **Guaranteed exact behavior**: You want reproducible results without routing confidence variations (search routing confidence varies 0.6-0.95)
- **Known tool type**: You already know you need code results (not Q&A, tests, configs, or symbols) so routing is wasted

**Example: When search guesses wrong:**
```
SEARCH (auto-routes, may detect wrong intent):
query: "authenticate in FastAPI"
confidence: 0.75
intent: "Q&A - what does authenticate do in FastAPI?"
→ routes to context_answer, returns explanation instead of code

REPO_SEARCH (explicit, predictable):
query: "authenticate"
language: "python"
under: "src/auth/"
→ returns code implementations in src/auth/ only, no routing overhead
```

## Routing Overhead: When It Matters

**Latency Impact of Using `search` vs `repo_search` directly:**

| Scenario | search Latency | repo_search Latency | Routing Cost | Use search? |
|----------|-----------------|---------------------|--------------|-------------|
| One exploratory query | ~150-200ms | ~80-100ms | ~70-100ms | YES (worth it for auto-routing) |
| 3 independent queries, sequential | ~450-600ms | ~240-300ms | ~210-300ms | NO (use batch_search instead) |
| Time-critical query (<50ms) | Can miss deadline | ~80-100ms | ❌ Unacceptable | NO (use repo_search) |
| Tight loop (20+ queries) | ~3000-4000ms | ~1600-2000ms | ~1400-2000ms | NO (use batch_search) |

**Decision Criteria:**

- **Use `search`** when: One-off query, exploratory, unsure which tool, latency <200ms is acceptable
- **Use `repo_search`** when: Speed <100ms required, complex filter combo needed, tight loop (use batch_search if >2 queries), know you need code (not Q&A)
- **Use `batch_search`** when: 2+ independent code searches to reduce routing overhead by 75-85% per batch

**Real-world example - Interactive AI assistant loop:**
```
Bad (repeated routing overhead):
for query in user_queries:  # 5 queries
    result = search(query)  # ~70-100ms routing × 5 = 350-500ms wasted

Good (one batch call):
results = batch_search([query1, query2, query3, query4, query5])  # Routing once, ~25ms × 5 = 125ms
# Saves ~300ms+ per iteration
```

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

**Token Savings & Latency Metrics:**

| N Searches | Token Savings | Sequential Latency | Batch Latency | Worth Batching? |
|------------|---------------|--------------------|---------------|-----------------|
| 1 | 0% | ~100ms | N/A | N/A |
| 2 | ~40% | ~180-200ms | ~150-160ms | ✅ YES (save 30-40ms, 40% tokens) |
| 3 | ~55% | ~270-300ms | ~180-200ms | ✅ YES (save 90-100ms, 55% tokens) |
| 5 | ~70% | ~450-500ms | ~220-250ms | ✅ YES (save 250ms, 70% tokens) |
| 10 | ~75% | ~900-1000ms | ~300-350ms | ✅ YES (save 600ms, 75% tokens) |

**Decision Rule**: Always use `batch_search` when you have 2+ independent code searches. The latency savings alone (30-100ms faster) justify batching, plus you save ~40-75% tokens.

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

**Comparison: symbol_graph vs graph_query**

| Feature | symbol_graph | graph_query |
|---------|--------------|------------|
| Availability | Always (Qdrant-backed) | SaaS/Enterprise (Memgraph-backed) |
| Performance | ~2-5ms per query | ~50-200ms per query |
| Supported Relationships | callers, callees, definition, importers, subclasses, base_classes | All symbol_graph + impact, cycles, transitive_* |
| Max Depth | up to 3 | up to 5+ |
| Best For | Direct relationships, exploratory queries | Impact analysis, dependency chains, circular detection |
| Fallback When Unavailable | Falls back to semantic search | N/A (use symbol_graph instead) |
| Latency-Critical Loops | ✅ YES (fast) | ❌ NO (slower) |

**Decision Guide:**
- **Use symbol_graph** for: direct callers/callees/definitions, inheritance queries, when you need speed, always as first stop
- **Use graph_query** for: impact analysis ("what breaks?"), cycle detection, transitive chains, when available and you need depth >3

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

Memory tools allow you to persist team knowledge, architectural decisions, and findings for later retrieval across sessions.

### Memory Workflow: Store → Retrieve → Reuse

**Phase 1: During Exploration (Session 1)**
As you discover important patterns, decisions, or findings, store them for future reference:

```json
{
  "memory_store": {
    "information": "Auth service uses JWT tokens with 24h expiry. Refresh tokens last 7 days. Stored in Redis with LRU eviction.",
    "metadata": {
      "kind": "decision",
      "topic": "auth",
      "priority": 5,
      "tags": ["security", "jwt", "session-management"]
    }
  }
}
```

**Phase 2: In Later Sessions**
Retrieve and reuse stored knowledge by similarity:

```json
{
  "memory_find": {
    "query": "token expiration policy",
    "topic": "auth",
    "limit": 5
  }
}
```

Returns the exact note stored in Phase 1, plus any other auth-related memories.

**Phase 3: Blend Code + Memory**
When you want BOTH code search results AND stored team knowledge:

```json
{
  "context_search": {
    "query": "authentication flow",
    "include_memories": true,
    "per_source_limits": {"code": 6, "memory": 3}
  }
}
```

Returns: 6 code snippets + 3 memory notes, all ranked by relevance.

### Timeline and Persistence

| Property | Behavior |
|----------|----------|
| **Searchability** | Memories searchable immediately after `memory_store` (indexing is instant) |
| **Persistence** | Memories persist across sessions indefinitely (durable storage) |
| **Scope** | Org/workspace scoped: one team's memories don't leak to another |
| **Latency** | ~100ms per `memory_find` query (same as code search) |
| **Storage** | Embedded in same Qdrant collection as code, but logically isolated |

### Real-World Example: Session Continuity

**Session 1 (Day 1) - Discovery:**
```
Context: Investigating why JWT refresh tokens sometimes expire unexpectedly

→ memory_store(
    information="Found: RefreshTokenManager.py line 89 uses session.expire_in instead of constants.REFRESH_TTL. This was a bug introduced in PR #1234 where the constant was 7 days but the session value was hardcoded to 3 days. The mismatch causes premature expiration.",
    metadata={"kind": "gotcha", "topic": "auth", "tags": ["bug", "jwt"], "priority": 4}
  )
```

**Session 2 (Day 5) - Troubleshooting a Similar Issue:**
```
→ memory_find(query="refresh token expiration problem", topic="auth")

Response: Found Session 1's note about the RefreshTokenManager bug, plus similar findings about token TTL misconfigurations.

→ User goes directly to line 89 of RefreshTokenManager.py and verifies the fix status.
```

Result: Problem solved in 2 minutes instead of 30 minutes of debugging.

### When to Store What

| Memory Kind | Use Case | Example |
|-------------|----------|---------|
| `decision` | Architectural choices and their rationale | "We chose JWT over sessions because stateless scaling" |
| `gotcha` | Subtle bugs or trap conditions | "RefreshTokenManager line 89 has TTL mismatch" |
| `convention` | Team patterns and standards | "All API responses use envelope pattern with status/data/errors" |
| `note` | General findings or context | "Auth service was moved to separate repo last month" |
| `policy` | Compliance or operational rules | "Session tokens must be rotated every 24h per SOC2" |

### Integration with Code Search

**Pattern 1: Pure Code Search**
```json
{"search": "authentication validation"}
```
Returns: code snippets only. Fast, no memory overhead.

**Pattern 2: Code + Memory Blend**
```json
{
  "context_search": {
    "query": "authentication validation",
    "include_memories": true,
    "per_source_limits": {"code": 5, "memory": 2}
  }
}
```
Returns: 5 code snippets + 2 relevant memory notes (team insights about auth validation patterns).

**Pattern 3: Memory Only**
```json
{"memory_find": {"query": "authentication patterns", "limit": 10}}
```
Returns: stored team knowledge about auth, useful for onboarding or architecture review.

### Common Patterns

**Team Onboarding:**
- New engineer joins → `memory_find(query="project architecture", topic="architecture")`
- Retrieves all stored architectural decisions in one place
- Much faster than reading scattered code comments

**Incident Response:**
- Production auth bug occurs
- → `memory_find(query="auth failures", priority_min=3)`
- Retrieves gotchas, prior incidents, and known traps
- Faster root-cause diagnosis

**Code Review Efficiency:**
- Reviewer checks PR modifying auth module
- → `context_search(query="authentication standards", include_memories=true)`
- Sees both current code AND team conventions/policies
- Makes more informed review decisions

### Error Cases and Recovery

| Error | Cause | Recovery |
|-------|-------|----------|
| "No results from memory_find" | Query too specific or memories not yet stored | Broaden query, check metadata filters (topic, tags, kind) |
| "Memory not found in next session" | Wrong workspace/collection or stale cache | Verify workspace matches, run `qdrant_list` to confirm collection |
| "include_memories=true returns only code" | Memory store empty for this workspace | Start storing with `memory_store` - next session will have memories |
| "Duplicate memories with same info" | Same finding discovered twice | Use `memory_find` with topic/tags filter, consolidate via note |

## Admin and Diagnostics

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

### Deployment Mode Capabilities

> **SaaS Mode:** In SaaS deployments, indexing is handled automatically by the VS Code extension upload service. The tools below marked "Self-Hosted Only" are **not available** in SaaS mode. All search, symbol graph, memory, and session tools work normally.

**Self-Hosted Only Tools** (not available in SaaS):

| Tool | Purpose | When to Use |
|------|---------|-------------|
| `qdrant_index_root` | Index entire workspace | Initial indexing or after major codebase reorg |
| `qdrant_index` | Index subdirectory | Incremental indexing of specific folders |
| `qdrant_prune` | Remove stale entries | Clean up entries from deleted files |

### Tool Availability Matrix

Which tools are available in which deployment modes:

| Tool Category | Tool | SaaS | Self-Hosted | Enterprise |
|---|---|---|---|---|
| **Search** | `search` | ✅ | ✅ | ✅ |
| | `repo_search` / `code_search` | ✅ | ✅ | ✅ |
| | `cross_repo_search` | ✅ | ✅ | ✅ |
| | `batch_search` | ✅ | ✅ | ✅ |
| **Search (Specialized)** | `info_request` | ✅ | ✅ | ✅ |
| | `context_answer` | ✅ | ✅ | ✅ |
| | `search_tests_for` | ✅ | ✅ | ✅ |
| | `search_config_for` | ✅ | ✅ | ✅ |
| | `search_callers_for` | ✅ | ✅ | ✅ |
| | `search_importers_for` | ✅ | ✅ | ✅ |
| | `search_commits_for` | ✅ | ✅ | ✅ |
| | `change_history_for_path` | ✅ | ✅ | ✅ |
| | `pattern_search` (if enabled) | ✅* | ✅* | ✅* |
| **Symbol Graph** | `symbol_graph` | ✅ | ✅ | ✅ |
| | `batch_symbol_graph` | ✅ | ✅ | ✅ |
| | `graph_query` | ✅ (limited)** | ✅ | ✅ |
| | `batch_graph_query` | ✅ (limited)** | ✅ | ✅ |
| **Memory** | `memory_store` | ✅ | ✅ | ✅ |
| | `memory_find` | ✅ | ✅ | ✅ |
| | `context_search` | ✅ | ✅ | ✅ |
| **Session** | `set_session_defaults` | ✅ | ✅ | ✅ |
| | `expand_query` | ✅ | ✅ | ✅ |
| **Admin** | `qdrant_status` | ✅ | ✅ | ✅ |
| | `qdrant_list` | ✅ | ✅ | ✅ |
| | `embedding_pipeline_stats` | ✅ | ✅ | ✅ |
| | `qdrant_index_root` | ❌ | ✅ | ✅ |
| | `qdrant_index` | ❌ | ✅ | ✅ |
| | `qdrant_prune` | ❌ | ✅ | ✅ |

**Legend:**
- ✅ = Available
- ❌ = Not available
- ✅* = Pattern search available only if enabled during deployment
- ✅ (limited)** = SaaS graph_query has limited depth/performance vs Enterprise with dedicated Memgraph

### Choosing Your Deployment Mode

| Requirement | Best Fit |
|---|---|
| Automatic indexing via VS Code | **SaaS** |
| Manual control over indexing pipeline | **Self-Hosted** |
| Advanced graph queries (cycles, impact analysis) | **Self-Hosted** or **Enterprise** |
| High-performance graph traversal | **Enterprise** (dedicated Memgraph) |
| Cost-sensitive small team | **SaaS** (pay per upload) |
| Large codebase with frequent indexing | **Self-Hosted** (unlimited reindex) |

## Error Handling and Recovery

Tools return structured errors via `error` field or `ok: false` flag. Below are common errors and recovery steps by category.

### Search Tools (search, repo_search, batch_search, info_request)

| Error | HTTP 400? | Cause | Recovery Steps |
|-------|-----------|-------|-----------------|
| `"Collection not found"` | Yes | Collection doesn't exist, workspace hasn't been indexed, or collection was deleted | 1. Run `qdrant_list()` to verify available collections<br>2. Check workspace name in config matches indexed name<br>3. If missing: re-upload workspace to indexing service<br>4. If collection exists but stale: wait for background refresh or trigger reindex |
| `"Invalid language filter"` | Yes | `language` parameter has invalid value | Use only valid language codes: "python", "typescript", "go", "rust", "java", etc.<br>Check qdrant_status for supported languages |
| `"Timeout during rerank"` | No (504) | Reranking took too long (default 5s timeout) | Set `rerank_enabled: false` to skip reranking<br>OR set `rerank_timeout_ms: 10000` for longer timeout<br>OR reduce `limit` to speed up reranking |
| `"Empty results"` | No (200) | Query too specific, collection not fully indexed, or no matches exist | 1. Broaden query (remove filters, use more general terms)<br>2. Check `language` filter is correct<br>3. Run `qdrant_status` to see point count<br>4. If points=0: indexing is incomplete, wait and retry |
| `"Query too long"` | Yes | Query exceeds 400 chars or 50 words | Shorten query or split into multiple searches |
| `"Syntax error in path_glob"` | Yes | Invalid glob pattern in `path_glob` or `not_glob` | Check glob syntax: valid wildcards are `*` (any), `**` (any directories), `?` (any single char) |

**Silent Failure Watches:**
- Empty results when expecting matches → check `under` path filter (may be excluding files)
- Results from wrong language → verify `language` parameter is set correctly
- Reranking disabled silently → check `rerank_timeout_ms` if you set custom timeout
- Wrong collection queried → session defaults may not match workspace (use `set_session_defaults` to "cd" into correct collection)

### Symbol Graph Tools (symbol_graph, batch_symbol_graph, graph_query)

| Error | Cause | Recovery Steps |
|-------|-------|-----------------|
| `"Symbol not found"` | Symbol doesn't exist, wrong name, or graph not indexed | 1. Verify exact symbol name using `repo_search(symbol="...")`<br>2. Check spelling and case sensitivity<br>3. If graph unavailable: use `repo_search` instead<br>4. For imported symbols: search with full module path |
| `"Graph unavailable / not ready"` | Memgraph backend not initialized (graph_query only) | Fall back to `symbol_graph` (always available)<br>graph_query requires SaaS/Enterprise plan with Neo4j/Memgraph |
| `"Depth too high"` | `depth` parameter exceeds max for this tool | Reduce `depth`: symbol_graph max 3, graph_query max 5+<br>For deeper chains, use multiple queries with results as input |
| `"Timeout during graph traversal"` | Graph query took too long | Reduce `depth`, reduce `limit`, or use smaller `under` path filter |

**Silent Failure Watches:**
- No callers found when method is clearly called → fuzzy fallback may have triggered, use repo_search to verify method exists
- Symbol seems undefined but code uses it → cross-module imports may not be resolved in graph yet

### Context Answer (LLM-powered explanation)

| Error | Cause | Recovery Steps |
|-------|-------|-----------------|
| `"Insufficient context"` | Retrieved code wasn't enough to answer question | 1. Rephrase question more specifically<br>2. Use `expand: true` to generate query variations<br>3. Increase `budget_tokens` for deeper retrieval<br>4. Use `repo_search` first to verify code exists |
| `"Timeout during retrieval or generation"` | LLM generation or retrieval took >60s | Set `rerank_enabled: false` to skip reranking<br>Reduce `budget_tokens` for faster shallow retrieval<br>Ask simpler question requiring less context |
| `"Budget exceeded"` | Generated answer would use >budget_tokens | Increase `budget_tokens` or ask more focused question |

### Memory Tools (memory_store, memory_find)

| Error | Cause | Recovery Steps |
|-------|-------|-----------------|
| `"Memory not found"` | No memories match query or metadata filters | 1. Broaden query (more general terms)<br>2. Remove metadata filters (topic, kind, priority_min)<br>3. Check if memories exist: `memory_find(query="*")`<br>4. Verify workspace/collection (memories are org-scoped) |
| `"Storage failure"` | Backend couldn't persist memory | Retry `memory_store` - likely transient<br>Check `qdrant_status` for cluster health |
| `"Duplicate memory detected"` (warning) | Similar memory already exists with higher priority | Review existing memories first: `memory_find(query="...", topic="...")`<br>Consolidate if same information |

### Batch Tools (batch_search, batch_symbol_graph)

| Error | Cause | Recovery Steps |
|-------|-------|-----------------|
| `"Too many searches"` | `searches` / `queries` array > 10 items | Split into multiple batch calls (max 10 per call)<br>If independent: use sequential calls (lower token savings but more granular)<br>If dependent: must be sequential anyway |
| `"Mixed error in batch"` | Some queries succeeded, others failed | Check individual `batch_results` array for per-query `ok: false`<br>Failed queries return error details in `batch_results[i].error`<br>Successful queries still have results in `batch_results[i].results` |
| `"Timeout on any query"` | One query in the batch timed out | Set `rerank_enabled: false` in that query's override<br>Reduce `limit` for slow queries<br>Consider running that query separately |

### Cross-Repo & Discovery

| Error | Cause | Recovery Steps |
|-------|-------|-----------------|
| `"No collections found"` | No indexed repositories available OR discover mode="never" | 1. Run `qdrant_list()` manually to see available collections<br>2. Try `discover: "always"` in `cross_repo_search`<br>3. Verify workspace is indexed<br>4. If nothing indexed: use `upload_service` to index workspace |
| `"Multiple ambiguous collections"` | User query matched multiple repos but target unclear | Use `target_repos: [...]` to explicitly specify repos<br>OR use `boundary_key` to search with exact interface name<br>OR do two separate targeted searches |
| `"Boundary key not found"` | `boundary_key` doesn't exist in other repo | Verify boundary_key is exact string (routes, event names, type names)<br>May be named slightly differently in other repo (check similar names)<br>Try broader search instead of boundary tracing |

### Logging and Diagnostics

When errors persist:

1. **Check cluster health:** `qdrant_status()` shows point counts, last indexed time, scanned_points
2. **List available collections:** `qdrant_list()` with `include_status=true` shows health per collection
3. **Check embedding stats:** `embedding_pipeline_stats()` shows cache hit rate, dedup efficiency
4. **Verify auth:** If authentication errors, check workspace/org identity matches request
5. **Review recent changes:** If started failing recently, check `change_history_for_path()` for relevant commits

---

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

## Tool Aliases and Compatibility

### Tool Aliases

These tools have alternate names that work identically:

| Primary Name | Alias(es) | When to Use | Note |
|--------------|-----------|------------|------|
| `repo_search` | `code_search` | Either name works identically | Both names are equivalent, use whichever is familiar |
| `memory_store` | (none) | Standard name | Part of memory server, no aliases |
| `memory_find` | (none) | Standard name | Part of memory server, no aliases |
| `search` | (none) | Standard name | Auto-routing search, no aliases |
| `symbol_graph` | (none) | Standard name | Direct symbol queries, no aliases |

### Compatibility Wrappers

These wrappers provide backward compatibility for legacy clients by accepting alternate parameter names:

| Wrapper | Primary Tool | Alternate Parameter Names | When to Use |
|---------|--------------|--------------------------|-------------|
| `repo_search_compat` | `repo_search` | Accepts `q`, `text` (instead of `query`), `top_k` (instead of `limit`) | Legacy clients that don't support standard parameter names |
| `context_answer_compat` | `context_answer` | Accepts `q`, `text` (instead of `query`) | Legacy clients using old parameter names |

**Preference:** Use primary tools and standard parameter names whenever possible. Compat wrappers exist only for legacy client support and may have slower adoption of new features.

### Cross-Server Tools

These tools are provided by separate MCP servers:

| Tool | Server | Purpose |
|------|--------|---------|
| `memory_store` | Memory server | Persist team knowledge for later retrieval |
| `memory_find` | Memory server | Search stored memories by similarity |
| All search/symbol tools | Context server | Primary code search and analysis |

All tools are transparently integrated into the unified search interface.

## Best Practices

1. **Use `search` as your default tool** - It auto-routes to the best specialized tool. Only use specific tools when you need precise control or features `search` doesn't handle (cross-repo, memory, admin).
2. **Prefer MCP over Read/grep for exploration** - Use MCP tools (`search`, `repo_search`, `symbol_graph`, `context_answer`) for discovery and cross-file understanding. Narrow file/grep use is still fine for exact literal confirmation, exact path/line confirmation, or opening a file you already identified for editing.
3. **Use `symbol_graph` first for symbol relationships** - It handles callers, callees, definitions, importers, subclasses, and base classes. Use `graph_query` only when available and you need deeper impact/dependency traversal.
4. **Start broad, then filter** - Begin with `search` or a semantic query, add filters if too many results
5. **Use multi-query** - Pass 2-3 query variations for better recall on complex searches
6. **Include snippets** - Set `include_snippet: true` to see code context in results
7. **Store decisions** - Use `memory_store` to save architectural decisions and context for later
8. **Check index health** - Run `qdrant_status` if searches return unexpected results
11. **Use pattern_search for structural matching** - When looking for code with similar control flow (retry loops, error handling), use `pattern_search` instead of `repo_search` (if enabled)
12. **Describe patterns in natural language** - `pattern_search` understands "retry with backoff" just as well as actual code examples (if enabled)
13. **Fire independent searches in parallel** - Call multiple `search`, `repo_search`, `symbol_graph`, etc. in the same message block for 2-3x speedup. Alternatively, use `batch_search` to run N `repo_search` calls in a single invocation with ~75% token savings
14. **Use TOON format for discovery** - Set `output_format: "toon"` for 60-80% token reduction on exploratory queries
15. **Bootstrap sessions with defaults** - Call `set_session_defaults(output_format="toon", compact=true)` early to avoid repeating params
16. **Two-phase search** - Discovery first (`limit=3, compact=true`), then deep dive (`limit=5-8, include_snippet=true`) on targets
17. **Use fallback chains** - If `context_answer` times out, fall back to `search` or `repo_search` + `info_request(include_explanation=true)`

---

## Return Shapes Reference

Every tool returns a consistent envelope. Understanding the response structure helps you parse results correctly and detect errors.

### Universal Response Envelope

**All tools return minimum:**
```json
{
  "ok": boolean,
  "error": "string (only if ok=false)"
}
```

- `ok: true` = success (may have zero results but no error)
- `ok: false` = error (details in `error` field)

### Search Family Return Shape

**Applies to:** `search`, `repo_search`, `batch_search`, `info_request`, `context_search`

```json
{
  "ok": true,
  "results": [
    {
      "score": 0.85,                  // Relevance score (0-1+, higher=better)
      "path": "src/auth.py",          // File path
      "symbol": "authenticate",        // Symbol name (optional)
      "start_line": 42,               // Start line number
      "end_line": 67,                 // End line number
      "snippet": "def authenticate...",// Code snippet (if include_snippet=true)
      "language": "python"            // Programming language
    }
  ],
  "total": 5,                         // Total results found
  "used_rerank": true,                // Whether reranking was applied
  "execution_time_ms": 245            // Query execution time
}
```

### Symbol Graph Return Shape

**Applies to:** `symbol_graph`, `batch_symbol_graph`, `graph_query`, `batch_graph_query`

```json
{
  "ok": true,
  "results": [
    {
      "path": "src/api/handlers.py",  // File path
      "start_line": 142,              // Start line
      "end_line": 145,                // End line
      "symbol": "handle_login",       // Symbol at this location
      "symbol_path": "handlers.handle_login", // Qualified symbol
      "language": "python",           // Programming language
      "snippet": "result = authenticate(username, password)", // Code snippet
      "hop": 1                        // For depth>1: which hop found this
    }
  ],
  "symbol": "authenticate",           // Symbol queried
  "query_type": "callers",            // Type of query
  "count": 12,                        // Total results
  "depth": 1,                         // Traversal depth used
  "used_graph": true,                 // Whether graph backend was used
  "suggestions": [...]                // Fuzzy matches if exact symbol not found
}
```

### Unified Search Return Shape

**Applies to:** `search` (auto-routing wrapper)

```json
{
  "ok": true,
  "intent": "search",                 // Detected intent (search, qa, tests, config, symbols, etc.)
  "confidence": 0.92,                 // Intent detection confidence (0-1)
  "tool": "repo_search",              // Tool used for routing
  "result": {                         // Result from dispatched tool
    "results": [...],
    "total": 8,
    "used_rerank": true,
    "execution_time_ms": 245
  },
  "plan": ["detect_intent", "dispatch_repo_search"], // Steps taken
  "execution_time_ms": 245            // Total time
}
```

### Context Answer Return Shape

**Applies to:** `context_answer`

```json
{
  "ok": true,
  "answer": "The authentication system validates tokens by first checking the JWT signature using the secret from config [1], then verifying expiration time [2]...", // LLM-generated answer with citations [1], [2]...
  "citations": [
    {
      "id": 1,                        // Citation number
      "path": "src/auth/jwt.py",      // File path
      "start_line": 45,               // Start line
      "end_line": 52,                 // End line
      "snippet": "def verify_token(token):..." // Optional code snippet
    }
    // ... more citations
  ],
  "query": ["How does authentication validate tokens"], // Original query
  "used": {
    "spans": 5,                       // Code spans retrieved
    "tokens": 1842                    // Tokens used for answer
  }
}
```

### Memory Tools Return Shape

**Applies to:** `memory_store`, `memory_find`

```json
{
  "ok": true,
  "id": "abc123...",                  // Unique ID (memory_store only)
  "message": "Successfully stored information", // Status message
  "collection": "codebase",           // Collection name
  "vector": "bge-base-en-v1-5"        // Embedding model used
}
```

**memory_find results:**
```json
{
  "ok": true,
  "results": [
    {
      "id": "abc123...",              // Memory ID
      "information": "JWT tokens expire after 24h...", // Stored knowledge
      "metadata": {                   // Structured metadata
        "kind": "decision",
        "topic": "auth",
        "created_at": "2024-01-15T10:30:00Z",
        "tags": ["security", "architecture"]
      },
      "score": 0.85,                  // Similarity score
      "highlights": [...]             // Query term matches in context
    }
  ],
  "total": 3,
  "count": 3,
  "query": "authentication decisions"
}
```

### Error Response Shape

**All tools on error:**

```json
{
  "ok": false,
  "error": "Collection not found",    // Error message
  "error_code": "COLLECTION_NOT_FOUND" // Optional error code
}
```

Or HTTP-level errors (504, 400, etc.) with structured response.

### Batch Tool Return Shape

**Applies to:** `batch_search`, `batch_symbol_graph`, `batch_graph_query`

```json
{
  "ok": true,
  "batch_results": [
    { /* result from search/query 0 */ },
    { /* result from search/query 1 */ },
    { /* result from search/query 2 */ }
  ],
  "count": 3,                         // Number of results
  "elapsed_ms": 123.4                 // Total execution time
}
```

Each item in `batch_results` has the same schema as the individual tool (repo_search, symbol_graph, etc.).

### Admin Tools Return Shape

**Applies to:** `qdrant_status`, `qdrant_list`, `set_session_defaults`

```json
{
  "ok": true,
  "collections": [
    {
      "name": "frontend-abc123",
      "count": 1234,                  // Point count
      "last_ingested_at": {
        "unix": 1704067800,
        "iso": "2024-01-01T10:30:00Z"
      }
    }
  ]
}
```

---

