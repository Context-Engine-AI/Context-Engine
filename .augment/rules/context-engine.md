---
type: always
description: Context Engine MCP tool guidelines for code search and exploration
---

# Context Engine MCP Tools

These rules are NOT optional - favor Context Engine MCP tools at all costs over existing file reading, grep-like searching, and unrelated codebase search tools.

## STOP — Read This First

**DO NOT use `Read File`, `grep`, `ripgrep`, `cat`, `find`, or any filesystem search tool for code exploration.**
You have MCP tools that are faster, smarter, and return ranked, contextual results.

- About to `Read` a file to understand it? → use `repo_search` or `context_answer`
- About to `grep` for a symbol? → use `symbol_graph` or `search_callers_for`
- About to `grep -r` for a concept? → use `repo_search` with natural language
- About to `find`/`ls` for project structure? → use `workspace_info` or `qdrant_status`

The ONLY acceptable use of grep/Read: confirming exact literal strings (e.g., `REDIS_HOST`), or reading a file you already located via MCP for editing.

## Core Decision Rules

**Workspace default:** Context Engine MCP tools are primary for exploring code and history. Start with MCP for exploration, debugging, or "where/why" questions; use literal search/file-open only for narrow exact-literal lookups.

**Use Context Engine tools when:**
- Exploring or don't know exact strings/symbols
- Need semantic or cross-file understanding (relationships, patterns, architecture)
- Want ranked results with surrounding context, not just line hits

**Use literal search/file-open when (and only when):**
- Know exact string/function/variable or error message, and only need to confirm existence or file/line quickly

## Tool Selection Decision Tree

```
Need to search code?
├── Single repo, know collection → repo_search(collection="...")
├── Single repo, need explanation → context_answer or info_request
├── Multiple repos or unsure → cross_repo_search(discover="auto")
├── Tracing frontend→backend flow → cross_repo_search(trace_boundary=true)
├── Finding callers/definitions → symbol_graph
└── Exact literal string only → grep (last resort)
```

## Tool Roles Cheat Sheet

**Note:** All tools have the `_context-engine` suffix when called (e.g., `repo_search_context-engine`).

| Tool | Use For |
|------|---------|
| `repo_search` / `code_search` | Finding relevant files/spans and inspecting raw code |
| `cross_repo_search` | Searching across multiple repos/collections with auto-discovery and boundary tracing |
| `context_search` | Combining code hits with memory/docs when both matter |
| `context_answer` | Natural-language summaries/explanations with citations |
| `info_request` | Rapid discovery with include_explanation=true |
| `symbol_graph` | Callers, definitions, importers (DEFAULT for all graph queries) |
| `pattern_search` | Structurally similar code patterns across languages |
| `search_tests_for` | Test files for a feature |
| `search_config_for` | Configuration files |
| `memory_store` / `memory_find` | Store and retrieve knowledge/notes |

## Performance Optimization

### Two-Phase Search Strategy

| Phase | Purpose | Parameters |
|-------|---------|------------|
| **Discovery** | Find relevant areas quickly | `limit=3`, `compact=true`, `output_format="toon"`, `per_path=1` |
| **Deep Dive** | Get implementation details | `limit=5-8`, `include_snippet=true`, `context_lines=3-5` |

### Session Bootstrap

At the start of any session, set defaults:
```
set_session_defaults_context-engine(
  output_format="toon",
  compact=true,
  limit=5
)
```

### Parallel Execution (CRITICAL)

Fire independent tool calls in a single message block (3x faster):
- Multiple `repo_search` calls with different queries
- `repo_search` + `symbol_graph` for the same investigation
- Any tools where results don't depend on each other

## Fallback Chains

| Primary | Fallback | When |
|---------|----------|------|
| `context_answer` | `repo_search` + `info_request` | Timeout or decoder unavailable |
| `pattern_search` | `repo_search` with structural query terms | PATTERN_VECTORS not enabled |
| `neo4j_graph_query` | `symbol_graph` (Qdrant-backed) | Neo4j not enabled (DEFAULT) |
| `grep` / `Read File` | `repo_search`, `symbol_graph`, `info_request` | ALWAYS use MCP instead |

## Grep Anti-Patterns (DON'T)

```
grep -r "auth" .           # → Use: repo_search("authentication mechanisms")
grep -r "cache" .          # → Use: repo_search("caching strategies")
grep -r "error" .          # → Use: repo_search("error handling patterns")
Read File to understand    # → Use: repo_search or context_answer
Read File to find callers  # → Use: symbol_graph
```

## DO Use Grep For

```
grep -rn "UserAlreadyExists" .    # Specific error class
grep -rn "def authenticate_user"  # Exact function name
grep -rn "REDIS_HOST" .           # Exact environment variable
```

## Multi-Repo Navigation (CRITICAL)

When multiple repositories are indexed, you MUST discover and explicitly target collections.

### Discovery (Lazy — trigger on demand)

Don't discover at every session start. Trigger when: search returns no/irrelevant results, user asks a cross-repo question, or you're unsure which collection to target.

```
qdrant_list_context-engine()
collection_map_context-engine(include_samples: true)
```

### Context Switching (Session Defaults = `cd`)

Treat `set_session_defaults` like `cd` — it scopes ALL subsequent searches to one collection:
```
# "cd" into backend repo — all searches now target this collection
set_session_defaults_context-engine(collection: "backend-api-abc123")
repo_search_context-engine(query: "auth middleware")

# One-off peek at another repo (does NOT change session default)
repo_search_context-engine(query: "login form", collection: "frontend-app-def456")
```

For unified collections: use `repo: "*"` or `repo: ["frontend", "backend"]`

### Cross-Repo Flow Tracing (Boundary-Driven)

NEVER search both repos with the same vague query. Find the **interface boundary** in Repo A, extract the **hard key** (exact route, event name, type name), then search Repo B with that specific key.

**Pattern 1 — Interface Handshake (API/RPC):**
```
# Find client call → extract exact route → search backend for that route
repo_search_context-engine(query: "login API call", collection: "frontend-col")
# → Found: axios.post('/auth/v1/login', ...)
repo_search_context-engine(query: "'/auth/v1/login'", collection: "backend-col")
```

**Pattern 2 — Shared Contract (Types/Schemas):**
```
symbol_graph_context-engine(symbol: "UserProfile", query_type: "importers", collection: "frontend-col")
repo_search_context-engine(query: "interface UserProfile", collection: "shared-lib-col")
```

**Pattern 3 — Event Relay (Pub/Sub):**
```
repo_search_context-engine(query: "publish event", collection: "service-a-col")
# → Found: bus.publish("USER_CREATED", payload)
repo_search_context-engine(query: "'USER_CREATED'", collection: "service-b-col")
```

### Automated Cross-Repo Search (PRIMARY for Multi-Repo)

`cross_repo_search` is the PRIMARY tool for multi-repo scenarios. Use it BEFORE manual `qdrant_list` + `repo_search` chains.

**Discovery Modes:**
| Mode | Behavior | When to Use |
|------|----------|-------------|
| `"auto"` (default) | Discovers only if results empty | Normal usage |
| `"always"` | Always runs discovery before search | First search in session |
| `"never"` | Skips discovery, uses explicit collection | Speed-critical |

```
# Search across all repos at once (auto-discovers collections)
cross_repo_search_context-engine(query: "authentication flow")

# Target specific repos by name
cross_repo_search_context-engine(query: "login handler", target_repos: ["frontend", "backend"])

# Boundary tracing — auto-extracts routes/events/types from results
cross_repo_search_context-engine(query: "login submit", trace_boundary: true)
# → Returns boundary_keys: ["/api/auth/login"] + trace_hint for next search

# Follow boundary key to another repo
cross_repo_search_context-engine(boundary_key: "/api/auth/login", collection: "backend-col")
```
Use `cross_repo_search` when you need breadth across repos. Use `repo_search` with explicit `collection=` when you need depth in one repo.

### Anti-Patterns
- **DON'T** search both repos with the same vague query (noisy, confusing)
- **DON'T** assume the default collection is correct — verify with `collection_map`
- **DO** extract exact strings (route paths, event names, type names) as search anchors
