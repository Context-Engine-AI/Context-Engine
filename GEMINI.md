# AI Agent Rules for Context-Engine MCP Tools

> **These rules are NOT optional.** Favor Context Engine MCP tools at all costs over grep, file reading, and unrelated codebase search tools.

## STOP — Read This First

**DO NOT use `Read File`, `grep`, `ripgrep`, `cat`, `find`, or any filesystem search tool for code exploration.**
These tools exist in your IDE but they are WRONG for this codebase. You have MCP tools that are faster, smarter, and return ranked, contextual results.

**If you catch yourself about to `Read` a file to understand it** → use `repo_search` or `context_answer` instead.
**If you catch yourself about to `grep` for a symbol** → use `symbol_graph` or `search_callers_for` instead.
**If you catch yourself about to `grep -r` for a concept** → use `repo_search` with a natural language query instead.

The ONLY acceptable use of `grep`/`Read` is confirming an exact literal string you already know exists (e.g., an env var name like `REDIS_HOST`).

## Introduction

This document defines requirements for AI agents using Context-Engine's MCP tools. The system provides two MCP servers (Memory Server on port 8000/8002, Indexer Server on port 8001/8003) with 30+ specialized tools for semantic code search, memory storage, and codebase exploration.

**Core Principle:** Context Engine MCP tools are PRIMARY for exploring code and history. Start with MCP for exploration, debugging, or "where/why" questions; use literal search/file-open only for narrow exact-literal lookups.

## Glossary

- **MCP**: Model Context Protocol - standardized interface for exposing tools to AI agents
- **Indexer Server**: MCP server for code search, indexing, symbol graphs (port 8001 SSE, 8003 HTTP)
- **Memory Server**: MCP server for knowledge storage and retrieval (port 8000 SSE, 8002 HTTP)
- **Hybrid Search**: Dense semantic vectors + lexical BM25 + neural reranking (ONNX)
- **ReFRAG**: Micro-chunking with 16-24 token windows for precise code retrieval
- **TOON**: Token-Oriented Object Notation - compact output format (60-80% token reduction)
- **Symbol Graph**: Indexed metadata for calls, imports, and definitions navigation
- **Collection**: Qdrant vector database collection storing indexed code chunks

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

## Requirements

### Requirement 1: MCP-First Tool Selection

**User Story:** As an AI agent, I want to prioritize MCP tools over grep/file-reading, so that I get semantic understanding efficiently.

#### Acceptance Criteria

1. WHEN exploring code or answering "where/why" questions, THE Agent SHALL use MCP Indexer tools as the primary method
2. WHEN the agent needs semantic understanding, cross-file relationships, or ranked results with context, THE Agent SHALL use MCP tools
3. WHEN the agent knows an exact literal string AND only needs to confirm existence/location, THE Agent SHALL use grep or file-open
4. IF the agent is uncertain which approach to use, THEN THE Agent SHALL default to MCP tools
5. THE Agent SHALL NOT use `grep -r "auth"` for concepts (use MCP: "authentication mechanisms")

### Requirement 2: Query Formulation

**User Story:** As an AI agent, I want to write effective semantic queries, so that I retrieve relevant code spans.

#### Acceptance Criteria

1. WHEN writing queries for `repo_search`, THE Agent SHALL use short natural-language fragments (e.g., "database connection handling")
2. THE Agent SHALL NOT use boolean operators (OR, AND), regex syntax, or code patterns in semantic queries
3. WHEN searching broad concepts, THE Agent SHALL use descriptive phrases like "error reporting patterns" not `grep -r "error"`
4. THE Agent SHALL write queries as descriptions of what to find, not as literal code strings
5. WHEN searching for specific symbols, THE Agent SHALL use the `symbol` parameter alongside the query

### Requirement 3: Performance Optimization

**User Story:** As an AI agent, I want to minimize token usage and latency, so that I work efficiently within context limits.

#### Acceptance Criteria

1. WHEN starting discovery, THE Agent SHALL use `limit=3`, `compact=true`, `per_path=1`
2. WHEN needing implementation details, THE Agent SHALL increase to `limit=5`, `include_snippet=true`
3. WHEN token efficiency is critical, THE Agent SHALL use `output_format="toon"` for 60-80% reduction
4. THE Agent SHALL NOT use `limit=20` with `include_snippet=true` (excessive token waste)
5. THE Agent SHALL NOT use high `context_lines` for pure discovery (unnecessary tokens)
6. THE Agent SHALL fire independent tool calls in parallel (same message block) for 2-3x speedup
7. THE Agent SHALL prefer `output_format="toon"` as default for all discovery queries

### Requirement 4: Core Search Tools

**User Story:** As an AI agent, I want to use the right search tool for each task, so that I get optimal results.

#### Acceptance Criteria

1. WHEN finding relevant files/spans and inspecting code, THE Agent SHALL use `repo_search` or `code_search`
2. WHEN combining code hits with memory/docs, THE Agent SHALL use `context_search` with `include_memories=true`
3. WHEN needing natural-language explanations with citations, THE Agent SHALL use `context_answer`
4. WHEN needing quick discovery with summaries, THE Agent SHALL use `info_request` with `include_explanation=true`
5. WHEN finding structurally similar patterns across languages, THE Agent SHALL use `pattern_search`

### Requirement 5: Symbol Graph Navigation (DEFAULT for all graph queries)

**User Story:** As an AI agent, I want to navigate code relationships efficiently, so that I understand call graphs and dependencies.

> **IMPORTANT:** `symbol_graph` is the DEFAULT and ALWAYS-AVAILABLE tool for graph queries. It works with the Qdrant-backed symbol index — no Neo4j required. Use `symbol_graph` FIRST for any caller/definition/importer query. Do NOT attempt `neo4j_graph_query` unless you know Neo4j is enabled.

#### Acceptance Criteria

1. WHEN finding who calls a function, THE Agent SHALL use `symbol_graph(symbol="name", query_type="callers")`
2. WHEN finding where a symbol is defined, THE Agent SHALL use `symbol_graph(symbol="name", query_type="definition")`
3. WHEN finding what imports a module, THE Agent SHALL use `symbol_graph(symbol="name", query_type="importers")`
4. THE Agent SHALL prefer `symbol_graph` over `search_callers_for` for structured navigation with metadata
5. THE Agent SHALL use `language` and `under` filters to narrow symbol graph results
6. WHEN needing multi-hop traversals, THE Agent SHALL use `symbol_graph` with `depth=2` or `depth=3`
7. **THE Agent SHALL default to `symbol_graph` for ALL graph/relationship queries.** It is always available regardless of Neo4j status.
8. THE Agent SHALL NOT attempt `neo4j_graph_query` unless the tool is visible in the MCP tool list (it only registers when `NEO4J_GRAPH=1`)

### Requirement 5b: Neo4j Advanced Graph Queries (OPTIONAL — only when NEO4J_GRAPH=1)

**User Story:** As an AI agent, I want to perform advanced graph traversals when Neo4j is available, so that I understand impact, dependencies, and circular references.

> **NOTE:** The `neo4j_graph_query` tool is ONLY available when `NEO4J_GRAPH=1` is set. If this tool is not in your MCP tool list, it is NOT enabled — use `symbol_graph` instead for all graph queries. Do NOT error or warn about missing Neo4j; just use `symbol_graph`.

#### Acceptance Criteria

1. WHEN `neo4j_graph_query` IS available AND needing multi-hop callers, THE Agent SHALL use `neo4j_graph_query(query_type="transitive_callers", symbol="name", depth=2)`
2. WHEN `neo4j_graph_query` IS available AND analyzing "what would break if I change X?", THE Agent SHALL use `neo4j_graph_query(query_type="impact", symbol="name", depth=2)`
3. WHEN `neo4j_graph_query` IS available AND finding all dependencies, THE Agent SHALL use `neo4j_graph_query(query_type="dependencies", symbol="name")`
4. WHEN `neo4j_graph_query` IS available AND detecting circular dependencies, THE Agent SHALL use `neo4j_graph_query(query_type="cycles", symbol="name")`
5. WHEN `neo4j_graph_query` IS NOT available, THE Agent SHALL fall back to `symbol_graph` for callers/definitions/importers queries
6. THE Agent SHALL NEVER error or complain about Neo4j being unavailable — just use `symbol_graph`

### Requirement 6: Specialized Search Tools

**User Story:** As an AI agent, I want to use specialized tools for common search patterns, so that I find specific code types quickly.

#### Acceptance Criteria

1. WHEN finding test files, THE Agent SHALL use `search_tests_for` (preset test globs)
2. WHEN finding configuration files, THE Agent SHALL use `search_config_for` (preset config globs)
3. WHEN finding symbol usages heuristically, THE Agent SHALL use `search_callers_for`
4. WHEN finding import references, THE Agent SHALL use `search_importers_for`
5. THE Agent SHALL pass `language`, `under`, and `limit` filters to narrow specialized searches

### Requirement 7: Context Answer Best Practices

**User Story:** As an AI agent, I want high-quality explanations from context_answer, so that I understand code behavior accurately.

#### Acceptance Criteria

1. WHEN asking about specific modules, THE Agent SHALL mention filenames explicitly in queries
2. WHEN asking cross-file questions, THE Agent SHALL use behavior-describing queries without filenames
3. THE Agent SHALL use `budget_tokens` to control context size (default: MICRO_BUDGET_TOKENS env)
4. THE Agent SHALL set `temperature=0.2` or `temperature=0.3` for deterministic answers
5. THE Agent SHALL NOT use `context_answer` as a debugger for low-level helpers; prefer `repo_search` + direct reading

### Requirement 8: Info Request Simplification

**User Story:** As an AI agent, I want a simple interface for quick code discovery, so that I can find relevant code with minimal parameters.

#### Acceptance Criteria

1. WHEN needing simple search, THE Agent SHALL use `info_request(info_request="description")`
2. WHEN needing summaries and concepts, THE Agent SHALL set `include_explanation=true`
3. WHEN needing relationship data (imports/calls), THE Agent SHALL set `include_relationships=true`
4. THE Agent SHALL understand smart limits: 15 for short queries, 8 for questions, 10 default
5. THE Agent SHALL use `info_request` for quick discovery before deeper `repo_search` dives

### Requirement 9: Pattern Search Usage

**User Story:** As an AI agent, I want to find structurally similar code across languages, so that I detect patterns and duplication.

#### Acceptance Criteria

1. WHEN finding similar control flow, THE Agent SHALL use `pattern_search` with code examples OR descriptions
2. THE Agent SHALL use `query_mode="auto"` (default) to let the system detect code vs description
3. WHEN searching specific target languages, THE Agent SHALL use `target_languages` filter
4. THE Agent SHALL set `min_score=0.3` or higher to filter low-quality matches
5. IF pattern_search is unavailable (PATTERN_VECTORS!=1), THEN THE Agent SHALL fall back to `repo_search`

### Requirement 10: Git History Integration

**User Story:** As an AI agent, I want to understand code evolution, so that I answer "when/why did X change" questions.

#### Acceptance Criteria

1. WHEN finding current implementation, THE Agent SHALL first use `repo_search` to locate relevant files
2. WHEN summarizing recent changes, THE Agent SHALL use `change_history_for_path(path="...", include_commits=true)`
3. WHEN finding commits for specific behavior, THE Agent SHALL use `search_commits_for(query="behavior phrase", path="...")`
4. THE Agent SHALL read `lineage_goal`, `lineage_symbols`, `lineage_tags` from commit results
5. WHEN explaining current behavior after finding files, THE Agent SHALL use `context_answer`

### Requirement 11: Memory System Usage

**User Story:** As an AI agent, I want to store and retrieve knowledge effectively, so that I build on previous work.

#### Acceptance Criteria

1. WHEN storing code snippets, THE Agent SHALL use `memory_store` with metadata: `{code, language, path, kind="snippet"}`
2. WHEN storing explanations, THE Agent SHALL use `kind="explanation"` with `tags` and `topic`
3. THE Agent SHALL set `priority` (1-10) to indicate importance (higher = more important)
4. WHEN searching memories, THE Agent SHALL use `memory_find` with filters: `kind`, `language`, `tags`, `priority_min`
5. THE Agent SHALL use `context_search(include_memories=true)` to blend code + memory results

### Requirement 12: Cross-Repo Search

**User Story:** As an AI agent, I want to search across repositories effectively, so that I find code regardless of location.

#### Acceptance Criteria

1. WHEN searching a single repository, THE Agent SHALL use `repo="repo-name"`
2. WHEN searching multiple repositories, THE Agent SHALL use `repo=["frontend", "backend"]`
3. WHEN searching all indexed repositories, THE Agent SHALL use `repo="*"`
4. WHEN `repo` is omitted, THE Agent SHALL rely on auto-detection via CURRENT_REPO env (REPO_AUTO_FILTER=1)
5. THE Agent SHALL specify `collection` parameter when multiple collections exist

### Requirement 12b: Cross-Repo Search Tool (PRIMARY for Multi-Repo)

**User Story:** As an AI agent, I want to search across repositories with automatic discovery and boundary tracing, so that I efficiently trace flows between frontend and backend.

> **IMPORTANT:** `cross_repo_search` is the PRIMARY tool for multi-repo scenarios. It combines discovery, search, and boundary extraction in one call. Use it BEFORE manual `qdrant_list` + `collection_map` + `repo_search` chains.

#### Acceptance Criteria

1. WHEN searching across multiple repos or unsure which repo contains code, THE Agent SHALL use `cross_repo_search` with `discover="auto"` or `discover="always"`
2. WHEN tracing frontend→backend flows, THE Agent SHALL use `trace_boundary=true` to extract API routes, event names, and type names
3. WHEN following a boundary key to another repo, THE Agent SHALL use `cross_repo_search(boundary_key="extracted_key", collection="target")`
4. WHEN targeting specific repos by name, THE Agent SHALL use `target_repos=["frontend", "backend"]`
5. WHEN you know the exact collection, THE Agent SHALL use `cross_repo_search(collection="exact-id")` with `discover="never"` for speed
6. THE Agent SHALL prefer `cross_repo_search` over manual `repo_search` + `collection_map` chains for multi-repo work
7. THE Agent SHALL check `discovery.hint` and `trace_hint` fields for actionable next-step suggestions

#### Discovery Modes

| Mode | Behavior | When to Use |
|------|----------|-------------|
| `"auto"` (default) | Discovers only if results empty or no explicit targeting | Normal usage |
| `"always"` | Always runs discovery before search | First search in session, exploring new codebase |
| `"never"` | Skips discovery, uses explicit collection | When you know exact collection, speed-critical |

#### Cross-Repo Tracing Workflow (Simplified)

**Before (manual):**
```
1. repo_search(collection="frontend", query="...")
2. Manually extract API route from results
3. repo_search(collection="backend", query="that route")
```

**After (with cross_repo_search):**
```
# Step 1: Search frontend with boundary tracing
cross_repo_search(
  query="login form submit",
  collection="frontend-col",
  trace_boundary=true
)
# → Returns: boundary_keys: ["/api/auth/login"], trace_hint: "Found API route..."

# Step 2: Follow boundary key to backend
cross_repo_search(
  boundary_key="/api/auth/login",
  collection="backend-col"
)
# → Returns: exact backend handler for that route
```

### Requirement 13: Session Management

**User Story:** As an AI agent, I want to maintain session context, so that I don't repeat parameters unnecessarily.

#### Acceptance Criteria

1. WHEN starting a session, THE Agent SHALL call `set_session_defaults` for both indexer and memory servers
2. THE Agent SHALL set default `collection` in session to avoid repeating it in every request
3. THE Agent SHALL understand precedence: explicit args > per-connection defaults > token defaults > env default
4. THE Agent SHALL use session tokens for cross-connection reuse when needed
5. THE Agent SHALL call `set_session_defaults(collection="...", output_format="toon", compact=true)` early

### Requirement 14: Query Expansion Strategy

**User Story:** As an AI agent, I want to use query expansion judiciously, so that I improve results without excessive overhead.

#### Acceptance Criteria

1. THE Agent SHALL attempt normal search BEFORE using `expand_query` or `expand=true`
2. WHEN initial search returns poor results, THE Agent SHALL use `expand=true` on `context_answer`
3. THE Agent SHALL use `max_new=2` or `max_new=3` for expansion (default 3)
4. THE Agent SHALL understand expansion uses local LLM (llama.cpp, GLM, or MiniMax) and is expensive
5. THE Agent SHALL treat `expand_query` as a last resort, not a default

### Requirement 15: Error Handling

**User Story:** As an AI agent, I want to handle errors gracefully, so that I recover and continue working.

#### Acceptance Criteria

1. WHEN MCP tools return responses, THE Agent SHALL check the `ok` field for success/failure
2. WHEN reranking times out, THE Agent SHALL accept fallback to hybrid-only results (still valid)
3. WHEN decoder is disabled, THE Agent SHALL accept `context_answer` returning citations without generated text
4. WHEN pattern_search is unavailable, THE Agent SHALL fall back to `repo_search`
5. THE Agent SHALL parse error messages and adjust parameters accordingly

### Requirement 16: Indexing and Maintenance

**User Story:** As an AI agent, I want to manage indexing operations, so that the codebase stays current.

#### Acceptance Criteria

1. WHEN indexing entire workspace, THE Agent SHALL use `qdrant_index_root`
2. WHEN indexing subdirectories, THE Agent SHALL use `qdrant_index(subdir="path")`
3. WHEN recreating collections from scratch, THE Agent SHALL use `recreate=true`
4. WHEN removing stale points (deleted files), THE Agent SHALL use `qdrant_prune`
5. WHEN checking index health, THE Agent SHALL use `qdrant_status` for count and timestamps

### Requirement 17: Workspace Discovery

**User Story:** As an AI agent, I want to discover available workspaces and collections, so that I target the right codebase.

#### Acceptance Criteria

1. WHEN listing all collections, THE Agent SHALL use `qdrant_list`
2. WHEN checking current workspace state, THE Agent SHALL use `workspace_info`
3. WHEN discovering multiple workspaces, THE Agent SHALL use `list_workspaces`
4. WHEN mapping collections to repos, THE Agent SHALL use `collection_map`
5. THE Agent SHALL understand workspace state includes: indexing_status, indexing_config, active_repo_slug

### Requirement 17b: Multi-Repo Navigation Workflow (CRITICAL)

**User Story:** As an AI agent working with multiple repositories, I want a clear workflow to discover, switch between, and trace flows across codebases, so that I navigate multi-repo environments effectively without cross-contamination.

> **IMPORTANT:** In multi-repo setups, the default collection may not be what you want. You MUST discover available collections and explicitly target them.

#### Multi-Repo Discovery (Lazy — trigger on demand)

Don't discover at every session start. Trigger when: search returns no/irrelevant results, user asks a cross-repo question, or you're unsure which collection to target.

```
qdrant_list_context-engine()
collection_map_context-engine(include_samples: true)
```

#### Context Switching (Session Defaults = `cd`)

Treat `set_session_defaults` like `cd` — it scopes ALL subsequent searches to one collection:
```
# "cd" into backend repo
set_session_defaults_context-engine(collection: "backend-api-abc123")
# All searches now target backend
repo_search_context-engine(query: "auth middleware")

# One-off peek at another repo (does NOT change session default)
repo_search_context-engine(query: "login form", collection: "frontend-app-def456")
```

#### Cross-Repo Flow Tracing (Boundary-Driven)

NEVER search both repos with the same vague query. Find the **interface boundary** in Repo A, extract the **hard key** (exact route, event name, type name), then search Repo B with that key.

**Pattern 1 — Interface Handshake (API/RPC):**
```
# Find client call in frontend → extract exact route
repo_search_context-engine(query: "login API call", collection: "frontend-col")
# → Found: axios.post('/auth/v1/login', ...)
# Search backend for that exact route
repo_search_context-engine(query: "'/auth/v1/login'", collection: "backend-col")
```

**Pattern 2 — Shared Contract (Types/Schemas):**
```
# Find type usage in consumer → find definition in source
symbol_graph_context-engine(symbol: "UserProfile", query_type: "importers", collection: "frontend-col")
repo_search_context-engine(query: "interface UserProfile", collection: "shared-lib-col")
```

**Pattern 3 — Event Relay (Pub/Sub):**
```
# Find producer → extract event name → find consumer
repo_search_context-engine(query: "publish event", collection: "service-a-col")
# → Found: bus.publish("USER_CREATED", payload)
repo_search_context-engine(query: "'USER_CREATED'", collection: "service-b-col")
```

#### Automated Cross-Repo Search

For quick multi-collection searches, use `cross_repo_search` instead of manually switching collections:
```
# Search across all repos at once (auto-discovers collections)
cross_repo_search_context-engine(query: "authentication flow")

# Target specific repos by name
cross_repo_search_context-engine(query: "login handler", target_repos: ["frontend", "backend"])

# Boundary tracing — auto-extracts routes/events/types from results
cross_repo_search_context-engine(query: "login submit", trace_boundary: true)
# → Returns boundary_keys: ["'/auth/v1/login'"] + trace_hint for next search
```
Use `cross_repo_search` when you need breadth across repos. Use `repo_search` with explicit `collection=` when you need depth in one repo.

#### Acceptance Criteria

1. WHEN search returns no results or user asks a cross-repo question, THE Agent SHALL run `qdrant_list` + `collection_map` to discover available collections, OR use `cross_repo_search` with `discover: "auto"` for automatic discovery
2. WHEN unsure which collection contains a repo, THE Agent SHALL use `collection_map(include_samples: true)`
3. WHEN focusing on a specific repo, THE Agent SHALL call `set_session_defaults(collection: "target")` to scope subsequent searches
4. WHEN cross-referencing another repo, THE Agent SHALL pass `collection` explicitly on that single call (not change session defaults)
5. WHEN tracing cross-repo flows, THE Agent SHALL use boundary-driven tracing: find exact interface key in source repo, then search for that key in target repo — OR use `cross_repo_search` with `trace_boundary: true` for automated key extraction
6. THE Agent SHALL NOT search multiple repos with the same vague semantic query (noisy, confusing results)
7. THE Agent SHALL NOT assume the default collection is correct in multi-repo setups
8. WHEN searching across repos in a unified collection, THE Agent SHALL use `repo: "*"` or `repo: ["name1", "name2"]`
9. WHEN needing to search across repos in SEPARATE collections simultaneously, THE Agent SHALL use `cross_repo_search` with `target_repos`

### Requirement 18: Grep and File Read Anti-Patterns

**User Story:** As an AI agent, I want to recognize when grep and file reading are inappropriate, so that I use semantic search instead.

#### Acceptance Criteria

1. THE Agent SHALL NOT use `grep -r "auth"` (use MCP: "authentication mechanisms")
2. THE Agent SHALL NOT use `grep -r "cache"` (use MCP: "caching strategies")
3. THE Agent SHALL NOT use `grep -r "error"` (use MCP: "error handling patterns")
4. THE Agent SHALL NOT use `grep -r "database"` (use MCP: "database operations")
5. THE Agent SHALL use grep ONLY for exact literals: `grep -rn "UserAlreadyExists"`, `grep -rn "REDIS_HOST"`
6. THE Agent SHALL NOT use `Read File` to understand what a file does — use `repo_search` or `context_answer` with the filename in the query
7. THE Agent SHALL NOT use `Read File` to find callers/imports — use `symbol_graph` instead
8. THE Agent SHALL NOT open files to "browse" the codebase — use `info_request` or `repo_search` for discovery
9. THE Agent SHALL NOT use `find` or `ls` to discover project structure — use `workspace_info` or `qdrant_status`
10. THE ONLY acceptable uses of `Read File` are: (a) editing a file you already located via MCP, (b) reading config files you know the exact path of

### Requirement 19: Advanced Reranking Features

**User Story:** As an AI agent, I want to leverage reranking effectively, so that I get the most relevant results.

#### Acceptance Criteria

1. THE Agent SHALL use `rerank_enabled=true` (default) for complex queries needing best relevance
2. WHEN faster results are acceptable, THE Agent SHALL set `rerank_enabled=false`
3. THE Agent SHALL understand results include `learning_score` and `refinement_iterations` from ONNX teacher
4. THE Agent SHALL use `rerank_top_n` to control candidate pool size (default 20)
5. THE Agent SHALL use `rerank_return_m` to control final result count after reranking

### Requirement 20: Output Format and Token Efficiency

**User Story:** As an AI agent, I want to choose appropriate output formats, so that I optimize token usage.

#### Acceptance Criteria

1. WHEN token efficiency is critical, THE Agent SHALL use `output_format="toon"` for 60-80% reduction
2. WHEN needing full structured data, THE Agent SHALL use `output_format="json"` (default)
3. WHEN using `compact=true`, THE Agent SHALL expect reduced result fields (path, symbol, lines, score)
4. THE Agent SHALL use `include_snippet=false` when only file/line references are needed
5. THE Agent SHALL combine `compact=true` + `limit=3` + `per_path=1` for minimal discovery queries

---

## Tool Quick Reference

### Search Tools
| Tool | Use Case | Key Parameters |
|------|----------|----------------|
| `repo_search` | General code search | `query`, `limit`, `compact`, `language`, `under`, `symbol` |
| `code_search` | Alias for repo_search | Same as repo_search |
| `cross_repo_search` | Search across multiple repos/collections | `query`, `target_repos`, `discover`, `trace_boundary`, `boundary_key` |
| `context_search` | Code + memory blend | `include_memories`, `per_source_limits` |
| `context_answer` | NL explanations with citations | `query`, `budget_tokens`, `temperature` |
| `info_request` | Quick discovery (multi-granular) | `info_request`, `include_explanation` (use for broad architecture/overviews) |
| `pattern_search` | Structural similarity | `query`, `query_mode`, `target_languages` |

### Navigation Tools
| Tool | Use Case | Key Parameters | Availability |
|------|----------|----------------|--------------|
| `symbol_graph` | Call/import/definition graphs (hydrated w/ snippets) — **DEFAULT for all graph queries** | `symbol`, `query_type`, `limit`, `depth` | **Always available** |
| `neo4j_graph_query` | Advanced traversals (impact, transitive, cycles) | `symbol`, `query_type`, `depth`, `limit` | Only when `NEO4J_GRAPH=1` |
| `search_callers_for` | Symbol usages (heuristic) | `query`, `language` | Always available |
| `search_importers_for` | Import references | `query`, `language` | Always available |
| `search_tests_for` | Test files | `query`, `limit` | Always available |
| `search_config_for` | Config files | `query`, `limit` | Always available |

### History Tools
| Tool | Use Case | Key Parameters |
|------|----------|----------------|
| `change_history_for_path` | File change summary | `path`, `include_commits` |
| `search_commits_for` | Commit search | `query`, `path`, `limit` |

### Memory Tools
| Tool | Use Case | Key Parameters |
|------|----------|----------------|
| `memory_store` | Store knowledge | `information`, `metadata` |
| `memory_find` | Search memories | `query`, `kind`, `language`, `tags` |

### Admin Tools
| Tool | Use Case | Key Parameters |
|------|----------|----------------|
| `qdrant_index_root` | Index workspace | `recreate` |
| `qdrant_index` | Index subdirectory | `subdir`, `recreate` |
| `qdrant_prune` | Remove stale points | - |
| `qdrant_status` | Collection health | `collection` |
| `qdrant_list` | List collections | - |
| `workspace_info` | Workspace state | - |
| `set_session_defaults` | Session config | `collection`, `language`, `under` |
| `warmup_status` | Check model warmup | - |

---

## Agentic Optimization Patterns

### Parallel Execution (CRITICAL for ROI)

**Fire independent tool calls in a single message block** - this is the highest-ROI optimization.

```
# WRONG (sequential - 3x slower)
result1 = repo_search(query="authentication")
result2 = repo_search(query="error handling")
result3 = symbol_graph(symbol="authenticate")

# CORRECT (parallel - all at once)
# In a single message, call all three:
repo_search(query="authentication", limit=3, compact=true)
repo_search(query="error handling", limit=3, compact=true)
symbol_graph(symbol="authenticate", query_type="callers")
# Results arrive together
```

**When to parallelize:**
- Multiple `repo_search` calls with different queries
- `repo_search` + `symbol_graph` for the same investigation
- `search_tests_for` + `search_config_for` when exploring a feature
- Any tools where results don't depend on each other

### Two-Phase Search Strategy

| Phase | Purpose | Parameters | When to Use |
|-------|---------|------------|-------------|
| **Discovery** | Find relevant areas quickly | `limit=3`, `compact=true`, `output_format="toon"`, `per_path=1` | Always start here |
| **Deep Dive** | Get implementation details | `limit=5-8`, `include_snippet=true`, `context_lines=3-5` | After identifying targets |

```
# Phase 1: Discovery
info_request(info_request="authentication flow", limit=3)
# Check confidence.level - if "high", proceed; if "low", refine query

# Phase 2: Deep dive on high-value targets only
repo_search(
  query="jwt token validation",
  limit=5,
  include_snippet=true,
  context_lines=5,
  under="src/auth"
)
```

### Session Bootstrap

**At the start of any session, set defaults to optimize all subsequent operations:**

```
# Set defaults (inherited by all subsequent calls)
set_session_defaults(
  output_format="toon",    # 60-80% token reduction
  compact=true,            # Minimal result fields
  limit=5                  # Reasonable default
)
```

### Token Efficiency Defaults

| Parameter | Discovery | Deep Dive | Notes |
|-----------|-----------|-----------|-------|
| `limit` | 3 | 5-8 | Start small, expand if needed |
| `per_path` | 1 | 2 | Prevents duplicate file results |
| `compact` | true | false | Strips verbose metadata |
| `output_format` | "toon" | "json" | TOON saves 60-80% tokens |
| `include_snippet` | false | true | Headers-only for discovery |
| `context_lines` | 0 | 3-5 | Only when reading code |
| `rerank_enabled` | true | true | Disable only for speed |

### Fallback Chains

When primary tools fail or timeout, use these fallback patterns:

| Primary | Fallback | When |
|---------|----------|------|
| `context_answer` | `repo_search` + `info_request(include_explanation=true)` | Timeout or decoder unavailable |
| `pattern_search` | `repo_search` with structural query terms | PATTERN_VECTORS not enabled |
| `neo4j_graph_query` | `symbol_graph` (Qdrant-backed, ALWAYS available) | Neo4j not enabled (`NEO4J_GRAPH!=1`) or unavailable — **this is the DEFAULT** |
| `memory_find` | `context_search(include_memories=true)` | Memory server issues |
| `grep` / `Read File` | `repo_search`, `symbol_graph`, `info_request` | **ALWAYS** — do not use grep/read for exploration |

```
# Example: context_answer fallback
result = context_answer(query="how does auth work?")
if result.get("error") or result.get("answer") == "insufficient context":
    # Fallback to search + explanation
    search_result = repo_search(query="authentication implementation", limit=5)
    explanation = info_request(info_request="authentication flow", include_explanation=true)
```

---

## Advanced Features & Examples

### Using Confidence Metrics

The `info_request` tool returns confidence metrics including score variance analysis to help you understand search quality:

```json
// Example: Low confidence search triggers suggestion
{
  "query": "auth",
  "confidence": {
    "level": "low",
    "score": 0.42,
    "variance_score": 0.18,
    "score_spread": 0.6,
    "consistency_level": "low",
    "coefficient_of_variation": 0.43,
    "min_score": 0.2,
    "max_score": 0.8,
    "low_confidence_hint": "Try more specific terms or include function/class names for better results"
  }
}
```

**Interpreting Confidence Metrics:**
- `level`: Overall confidence ("high", "medium", "low", "none")
- `score`: Average result score
- `consistency_level`: How similar scores are ("high" = CV<0.2, "medium" = CV 0.2-0.4, "low" = CV>0.4)
- `coefficient_of_variation`: Relative variability (higher = more diverse results)
- `low_confidence_hint`: Actionable suggestion when confidence is low

**Agent Behavior:**
- **Low confidence + high CV**: Results are diverse but uncertain → refine query with more specific terms
- **High confidence + low CV**: Strong, consistent results → query is effective
- **Low confidence + low CV**: Consistently poor results → try different search approach

### Symbol Suggestions on Typo

The `symbol_graph` tool provides fuzzy matching suggestions when exact symbol match fails:

```json
// Example: Typo in symbol name gets helpful suggestions
{
  "symbol": "getUserProf",
  "query_type": "callers",
  "results": [],
  "count": 0,
  "suggestions": ["getUserProfile", "getUser", "UserProfile"],
  "hint": "Symbol 'getUserProf' not found. Did you mean: getUserProfile?"
}
```

**How Suggestions Work:**
- **Edit distance ≤2**: Catches typos like "getUSerProfile" → "getUserProfile"
- **Prefix matching**: Partial names like "getUser" → "getUserProfile"
- **CamelCase/snake_case tokens**: Matches "get_user_profile" to "getUserProfile"
- **Scored ranking**: Top 3 suggestions ordered by similarity (1.0 = exact, 0.9 = prefix, 0.8 = token match, 0.6-0.8 = edit distance)

**Agent Behavior:**
- When `suggestions` field present, try the top suggestion first
- Suggestions are cached for 60s to reduce load
- Controlled via `SYMBOL_SUGGESTIONS_LIMIT` env (default: 3)

### Score Variance Detection

The `repo_search` and `context_search` tools detect high score variance and automatically expand spans for better context:

```json
// Example: High variance triggers span expansion
{
  "results": [
    {"path": "auth.py", "start_line": 10, "end_line": 45, "_adaptive_expanded": true},
    {"path": "user.py", "start_line": 20, "end_line": 35}
  ],
  "score_analysis": {
    "cv": 0.45,
    "high_variance": true,
    "variance": 0.08,
    "std": 0.28,
    "mean": 0.62,
    "adaptive_spans_used": 8
  }
}
```

**Variance Metrics:**
- `cv` (Coefficient of Variation): Relative score variability (std/mean)
- `high_variance`: Flag when CV > threshold (default 0.3)
- `adaptive_spans_used`: Count of spans expanded to full symbol boundaries

**Adaptive Behavior:**
- **High variance (CV > 0.3)** + `ADAPTIVE_SPAN_SIZING=1`: Expands micro-chunks to full function/class boundaries
- Max expansion: 80 lines, up to 3 spans, 40% of token budget
- Only expands when symbol metadata available
- Logs with `DEBUG_ADAPTIVE_SPAN=1`

**Environment Variables:**
```bash
SCORE_VARIANCE_THRESHOLD=0.3    # CV threshold for high_variance flag
VARIANCE_SPAN_EXPANSION=1.5     # Span multiplier when high variance
ADAPTIVE_SPAN_SIZING=1          # Enable/disable adaptive expansion
```

### Intent Confidence Analysis

Intent classification logs all queries to JSONL for offline analysis:

```bash
# Analyze intent classification quality over last 7 days
python scripts/analyze_intent_confidence.py --days 7

# Output example:
# ================================================================================
# INTENT CONFIDENCE ANALYSIS
# ================================================================================
#
# Total Events: 1,247
#
# Strategy Distribution:
#   rules     :   823 (66.0%)
#   ml        :   424 (34.0%)
#
# Intent Distribution:
#   search            :   512 (41.1%)
#   answer            :   302 (24.2%)
#   search_tests      :   156 (12.5%)
#   symbol_graph      :   134 (10.7%)
#   memory_find       :    89 ( 7.1%)
#
# Average Confidence: 0.78
# Fallback Rate (ML → search): 12%
#
# Top 10 Low-Confidence Queries:
#   1. [0.23] "show implementation..." → search (top: answer)
#   2. [0.24] "where is cache..." → search (top: answer)
#   3. [0.26] "find config for..." → search (top: search_config)
#   ...
```

**Event Log Format (JSONL):**
```json
{
  "timestamp": 1704974400.0,
  "query": "find tests for authentication",
  "intent": "search_tests",
  "confidence": 1.0,
  "strategy": "rules",
  "threshold": null,
  "candidates": []
}
```

**Environment Variables:**
```bash
INTENT_TRACKING_ENABLED=1        # Enable event logging (default: 1)
INTENT_EVENTS_DIR=./events       # Log directory (default: ./events)
INTENT_LOG_ROTATE_MB=100         # Max file size before rotation
```

**Agent Behavior:**
- **Low confidence (<0.4)**: Query may be ambiguous → check `candidates` for alternatives
- **High fallback rate**: Rules may need tuning → review top fallback queries
- **Strategy="rules"**: Fast, deterministic classification (confidence=1.0)
- **Strategy="ml"**: Semantic embedding-based fallback (confidence=0.0-1.0)

**Use Cases:**
- Monitor classification accuracy over time
- Identify queries that need better rule coverage
- Tune confidence thresholds based on real usage
- Debug misclassifications with full candidate scores

### Neo4j Graph Query Types (ONLY when NEO4J_GRAPH=1)

> **If `neo4j_graph_query` is not in your MCP tool list, skip this section entirely. Use `symbol_graph` for all graph queries instead.**

The `neo4j_graph_query` tool provides advanced graph traversals that are **impossible with grep**:

| Query Type | Description | Example |
|------------|-------------|---------|
| `callers` | Who calls this symbol? (depth 1) | `neo4j_graph_query(symbol="authenticate", query_type="callers")` |
| `callees` | What does this symbol call? (depth 1) | `neo4j_graph_query(symbol="main", query_type="callees")` |
| `transitive_callers` | Multi-hop callers (up to depth) | `neo4j_graph_query(symbol="get_embedding_model", query_type="transitive_callers", depth=2)` |
| `transitive_callees` | Multi-hop callees (up to depth) | `neo4j_graph_query(symbol="init", query_type="transitive_callees", depth=3)` |
| `impact` | What breaks if I change this? | `neo4j_graph_query(symbol="normalize_path", query_type="impact", depth=2)` |
| `dependencies` | What does this depend on? | `neo4j_graph_query(symbol="run_hybrid_search", query_type="dependencies")` |
| `cycles` | Detect circular dependencies | `neo4j_graph_query(symbol="ServiceA", query_type="cycles")` |

**Key Parameters:**
- `symbol`: The function, class, or module to analyze
- `query_type`: One of the above query types
- `depth`: Traversal depth for transitive queries (default 1, max ~5)
- `limit`: Maximum results (default 50)
- `include_paths`: Include full traversal paths in results
- `repo`: Filter by repository name

**ROI vs Grep:**
| Metric | Grep | Neo4j Graph | ROI |
|--------|------|-------------|-----|
| Multi-hop callers | Impossible | 80ms | ∞ |
| Impact analysis | 10+ iterations | 1 call, 3ms | 10x |
| Noise filtering | Manual | Automatic | 90%+ reduction |
| Time for simple lookup | 3-5s | 80ms | 40x faster |

**Example Response:**
```json
{
  "ok": true,
  "results": [
    {"symbol": "main", "hop": 1, "path_nodes": ["main", "normalize_path"], "repo": "work"},
    {"symbol": "evaluate_query", "hop": 2, "path_nodes": ["evaluate_query", "matches_relevant", "normalize_path"]}
  ],
  "total": 4,
  "query": {"query_type": "impact", "symbol": "normalize_path", "depth": 2},
  "backend": "neo4j",
  "query_time_ms": 3.43
}
```
