---
name: context-engine
description: Hybrid semantic/lexical code search with neural reranking via MCP tools. Use when Codex needs to search a codebase, find implementations, understand how code works, find callers/definitions, search git history, or store/retrieve knowledge. IMPORTANT - Always prefer these MCP tools over grep/find/cat for code exploration.
---

# Context Engine MCP Tools

Hybrid vector search (semantic + lexical) with neural reranking for codebase retrieval.

> **IMPORTANT: Always use `search` as your FIRST tool for ANY code exploration, lookup, or question.** It auto-detects intent and routes to the best specialized tool. Only use `repo_search`, `symbol_graph`, or other tools directly when you need specific parameters or features that `search` does not expose (cross-repo, memory, admin). When in doubt, use `search`.

## Core Decision Tree

```
Need to find code?
├── UNSURE / GENERAL QUERY → search (RECOMMENDED DEFAULT)
│   └── Auto-routes to best tool based on query intent
│   └── Handles: code search, Q&A, tests, config, symbols, imports
├── Simple lookup → search OR info_request
├── Need filters/control → search OR repo_search
├── Search across multiple repos → cross_repo_search
├── Want LLM explanation → search OR context_answer
├── Find similar patterns → pattern_search (if enabled)
├── Find relationships
│   ├── Who calls / who imports / where defined → symbol_graph (DEFAULT, always available)
│   ├── What does this call → symbol_graph (query_type="callees")
│   ├── Find subclasses / base classes → symbol_graph (query_type="subclasses" or "base_classes")
│   ├── Multi-hop (callers of callers) → symbol_graph (depth=2+)
│   ├── Impact analysis (what breaks if I change X) → graph_query (ONLY if available)
│   ├── Dependency graph → graph_query (ONLY if available)
│   └── Circular dependency detection → graph_query (ONLY if available)
├── Git history
│   ├── Find commits → search_commits_for
│   └── Predict co-changing files → search_commits_for (predict_related=true)
├── Blend code + notes → context_search (include_memories=true)
├── Store/recall knowledge → memory_store, memory_find
└── Multiple independent queries at once
    ├── batch_search (runs N repo_search calls in one invocation, ~75% token savings)
    ├── batch_symbol_graph (runs N symbol_graph queries in one invocation)
    └── batch_graph_query (runs N graph_query queries in one invocation)
```

## Primary Tools

**search** - ALWAYS USE FIRST (unified entry point, auto-routes):
```json
{"query": "authentication middleware"}
{"query": "how does caching work?"}          // → routes to context_answer
{"query": "who calls authenticate()"}        // → routes to symbol_graph
{"query": "tests for payment processing"}    // → routes to search_tests_for
```
Auto-detects intent and routes to the best tool. Returns `{ok, intent, confidence, tool, result, execution_time_ms}`.

Optional params: `query`, `collection`, `limit`, `language`, `under`, `include_snippet`, `compact`, `context_lines`, `ext`, `not_glob`, `path_glob`, `output_format`, `rerank_enabled`.

Use specialized tools directly only for: cross-repo search, batch search, memory, admin, or when you need params `search` doesn't expose.

**batch_search** - Run N independent searches in one call (~75% token savings):
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
Returns `{ok, batch_results: [result_set_0, ...], count, elapsed_ms}`. Max 10 searches per batch. Shared params (`collection`, `limit`, `language`, etc.) apply to all searches unless overridden per-search. Use when you have 2+ independent code searches; use individual `search` calls when you need intent routing or searches depend on each other.

**repo_search** - Direct code search (full control):
```json
{"query": "authentication middleware", "limit": 10, "include_snippet": true}
```
Multi-query: `{"query": ["auth handler", "login validation"]}`

**symbol_graph** - Find callers, callees, definitions, importers, subclasses, base classes (ALWAYS available):
```json
{"symbol": "authenticate", "query_type": "callers", "limit": 10}
{"symbol": "authenticate", "query_type": "callees", "limit": 10}
{"symbol": "UserService", "query_type": "definition"}
{"symbol": "utils", "query_type": "importers"}
{"symbol": "BaseModel", "query_type": "subclasses"}
{"symbol": "MyClass", "query_type": "base_classes"}
```
Query types: `callers`, `callees`, `definition`, `importers`, `subclasses`, `base_classes`. Use `depth=2` for multi-hop. Falls back to semantic search if no graph hits. Results include ~500-char source snippets.

**context_answer** - LLM-generated explanation with citations:
```json
{"query": "How does the caching layer work?", "budget_tokens": 2000}
```

**graph_query** - Advanced graph traversals and impact analysis (available to all SaaS users):
```json
{"symbol": "UserService", "query_type": "impact", "depth": 3}
{"symbol": "auth_module", "query_type": "cycles"}
{"symbol": "processPayment", "query_type": "transitive_callers", "depth": 2}
```
Query types: `callers`, `callees`, `transitive_callers`, `transitive_callees`, `impact`, `dependencies`, `definition`, `cycles`. Use `include_paths=true` for full traversal paths. Memgraph-backed; `symbol_graph` (Qdrant-backed) is always available as fallback.

**batch_symbol_graph** - Run N independent symbol_graph queries in one call (~75% token savings):
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
Returns `{ok, batch_results: [result_set_0, ...], count, elapsed_ms}`. Max 10 queries per batch. Each query must have a `symbol` key. Shared params (`collection`, `language`, `under`, `repo`, `limit`, `depth`) apply to all unless overridden per-query.

**batch_graph_query** - Run N independent graph_query queries in one call (~75% token savings):
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
Returns `{ok, batch_results: [result_set_0, ...], count, elapsed_ms}`. Max 10 queries per batch. Shared params (`collection`, `repo`, `language`, `depth`, `limit`, `include_paths`) apply to all unless overridden per-query.

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
| `search_commits_for` | Predict co-changing files | `{"path": "src/auth.py", "predict_related": true}` |
| `change_history_for_path` | File change summary | `{"path": "src/auth.py", "include_commits": true}` |
| `pattern_search` | Similar code patterns (if enabled) | `{"query": "retry with backoff"}` |
| `search_importers_for` | Find importers | `{"query": "utils/helpers"}` |
| `graph_query` | Advanced graph traversals / impact analysis | `{"symbol": "User", "query_type": "impact", "depth": 3}` |
| `batch_search` | N searches in one call | `{"searches": [{"query": "auth"}, {"query": "cache"}]}` |
| `batch_symbol_graph` | N symbol_graph queries in one call | `{"queries": [{"symbol": "auth", "query_type": "callers"}, {"symbol": "Cache", "query_type": "definition"}]}` |
| `batch_graph_query` | N graph_query queries in one call | `{"queries": [{"symbol": "User", "query_type": "impact"}, {"symbol": "auth", "query_type": "cycles"}]}` |

## Index Management

> **SaaS mode:** Indexing is handled automatically by the VS Code extension upload service. `qdrant_index_root`, `qdrant_index`, and `qdrant_prune` are **not available** in SaaS. All search, symbol graph, memory, and session tools work normally.

**Available in all modes:**
- `qdrant_status` - Check index health
- `qdrant_list` - List all collections
- `set_session_defaults` - Set collection, output_format, compact, limit
- `embedding_pipeline_stats` - Cache efficiency, bloom filter stats

**Self-hosted only (not available in SaaS):**
- `qdrant_index_root` - Index workspace
- `qdrant_index` - Index subdirectory
- `qdrant_prune` - Remove stale entries from deleted files

## Best Practices

1. **ALWAYS start with `search`** - It is your PRIMARY tool. Auto-routes to the best specialized tool. Only fall back to specific tools when you need params `search` doesn't expose.
2. **NEVER use grep/cat/find for code exploration** - Use MCP tools instead. Only acceptable use: confirming exact literal strings.
3. **Start with `symbol_graph`** for relationship queries - always available. Use `graph_query` for advanced traversals: impact analysis, circular dependencies, transitive callers/callees (available to all SaaS users)
4. **Use multi-query** for complex searches: pass 2-3 variations as a list
5. **Two-phase search**: Discovery (`limit=3, compact=true`) → Deep dive (`limit=8, include_snippet=true`)
6. **Fire parallel calls** - Multiple independent `search`, `repo_search`, `symbol_graph` in one message. Or use batch tools (`batch_search`, `batch_symbol_graph`, `batch_graph_query`) to run N queries in a single invocation with ~75% token savings
7. **Set session defaults early**: `set_session_defaults(output_format="toon", compact=true)`
8. **Use TOON format** - `output_format: "toon"` for 60-80% token reduction on exploratory queries
9. **Use `cross_repo_search`** for multi-repo scenarios instead of manual collection switching
10. **Predict co-changing files** - `search_commits_for(path=..., predict_related=true)` finds historically coupled files

## Error Fallbacks

- `context_answer` timeout → `search` + `info_request(include_explanation=true)`
- `pattern_search` unavailable → `search` with structural query terms
- `graph_query` unavailable → `symbol_graph` (always available, Qdrant-backed)
  - grep/Read File → use `search`, `symbol_graph`, `info_request` instead

## Filters (for repo_search)

- `language` - python, typescript, go, etc.
- `under` - Path prefix: `"src/api/"`
- `path_glob` - Include patterns: `["**/*.ts"]`
- `not_glob` - Exclude patterns: `["**/test_*"]`
- `repo` - Repository filter: `["frontend", "backend"]` or `"*"` for all

## Multi-Repo Navigation (CRITICAL)

When multiple repositories are indexed, you MUST discover and explicitly target collections.

### Discovery (Lazy — only when needed)
Don't discover at every session start. Trigger when: search returns no/irrelevant results, user asks a cross-repo question, or you're unsure which collection to target.
```json
// qdrant_list — discover available collections
{}
// cross_repo_search — auto-discover and search across repos
{"query": "your search", "discover": "always"}
```

### Context Switching (Session Defaults = `cd`)
Treat `set_session_defaults` like `cd` — scopes ALL subsequent searches:
```json
// "cd" into backend repo
{"collection": "backend-api-abc123"}
// One-off peek at another repo (does NOT change session default)
{"query": "login form", "collection": "frontend-app-def456"}
```
For unified collections: `"repo": "*"` or `"repo": ["frontend", "backend"]`

### Cross-Repo Flow Tracing (Boundary-Driven)
NEVER search both repos with the same vague query. Find the **interface boundary** in Repo A, extract the **hard key**, search Repo B with that key.

**Pattern 1 — Interface Handshake (API/RPC):**
1. Find client call: `repo_search(query="login API call", collection="frontend-col")`
2. Extract route: `/auth/v1/login`
3. Find handler: `repo_search(query="'/auth/v1/login'", collection="backend-col")`

**Pattern 2 — Shared Contract (Types):**
1. Find usage: `symbol_graph(symbol="UserProfile", query_type="importers", collection="frontend-col")`
2. Find definition: `repo_search(query="interface UserProfile", collection="shared-lib-col")`

**Pattern 3 — Event Relay (Pub/Sub):**
1. Find producer: `repo_search(query="publish event", collection="service-a-col")`
2. Extract event: `"USER_CREATED"`
3. Find consumer: `repo_search(query="'USER_CREATED'", collection="service-b-col")`

### Automated Cross-Repo Search (PRIMARY for Multi-Repo)
`cross_repo_search` is the PRIMARY tool for multi-repo scenarios. Use BEFORE manual `qdrant_list` + `repo_search` chains.

**Discovery Modes:**
| Mode | Behavior | When to Use |
|------|----------|-------------|
| `"auto"` | Discovers only if results empty | Normal usage |
| `"always"` | Always runs discovery | First search, new codebase |
| `"never"` | Skip discovery | Speed-critical |

**Examples:**
1. Search all repos: `cross_repo_search(query="authentication flow", discover="auto")`
2. Target repos: `cross_repo_search(query="login handler", target_repos=["frontend", "backend"])`
3. Boundary tracing: `cross_repo_search(query="login submit", trace_boundary=true)` → returns `boundary_keys`
4. Follow key: `cross_repo_search(boundary_key="/api/auth/login", collection="backend-col")`

Use `cross_repo_search` when you need breadth across repos. Use `repo_search` with explicit `collection=` when you need depth in one repo.

### Anti-Patterns
- DON'T search both repos with the same vague query
- DON'T assume the default collection is correct — verify with `qdrant_list`
- DO extract exact strings (routes, event names, types) as search anchors

## References

For detailed API documentation, see:
- [references/tool-reference.md](references/tool-reference.md) - Complete tool parameters
- [references/patterns.md](references/patterns.md) - Common search patterns

