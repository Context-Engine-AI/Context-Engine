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
├── Search across multiple repos → cross_repo_search
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

## Multi-Repo Navigation (CRITICAL)

When multiple repositories are indexed, you MUST discover and explicitly target collections.

### Discovery (Lazy — only when needed)
Don't discover at every session start. Trigger when: search returns no/irrelevant results, user asks a cross-repo question, or you're unsure which collection to target.
```json
// qdrant_list — discover available collections
{}
// collection_map — map repos to collections with sample files
{"include_samples": true}
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
- DON'T assume the default collection is correct — verify with `collection_map`
- DO extract exact strings (routes, event names, types) as search anchors

## References

For detailed API documentation, see:
- [references/tool-reference.md](references/tool-reference.md) - Complete tool parameters
- [references/patterns.md](references/patterns.md) - Common search patterns

