# Context Engine Tool Reference

Complete parameter reference for all MCP tools.

## search (UNIFIED - DEFAULT)

**Use this by DEFAULT for any code search, exploration, or question.** Automatically detects query intent and routes to the optimal specialized tool.

| Parameter | Type | Description |
|-----------|------|-------------|
| `query` | string | Natural language query describing what you need |
| `collection` | string | Target a specific collection |
| `limit` | int | Max results (default varies by detected intent) |
| `language` | string | Filter by programming language |
| `under` | string | Filter by directory path prefix |
| `include_snippet` | bool | Include code snippets in results |
| `compact` | bool | Compact output format |

**Returns:** `{ok, intent, confidence, tool, result, plan, execution_time_ms}`

**Dispatchable tools:** `repo_search`, `context_answer`, `search_tests_for`, `search_config_for`, `symbol_graph`, `search_callers_for`, `search_importers_for`, `info_request`, `context_search`

## repo_search / code_search

Primary hybrid search tool. Reranking enabled by default.

| Parameter | Type | Description |
|-----------|------|-------------|
| `query` | string/list | Search query (multi-query fuses results) |
| `limit` | int | Max results (default 10) |
| `per_path` | int | Max results per file (default 2) |
| `include_snippet` | bool | Include code snippets |
| `context_lines` | int | Lines of context around matches |
| `language` | string | Filter by language |
| `under` | string | Path prefix filter |
| `path_glob` | list | Include patterns |
| `not_glob` | list | Exclude patterns |
| `symbol` | string | Symbol name filter |
| `repo` | string/list | Repository filter ("*" for all) |
| `rerank_enabled` | bool | Enable neural reranking (default true) |
| `output_format` | string | "json" or "toon" (compact) |
| `compact` | bool | Minimal response fields |

## symbol_graph

AST-backed symbol relationship queries. Always available.

| Parameter | Type | Description |
|-----------|------|-------------|
| `symbol` | string | Symbol to analyze |
| `query_type` | string | "callers", "definition", "importers", "callees", "subclasses", "base_classes" |
| `depth` | int | Traversal depth (1=direct, 2+=multi-hop) |
| `limit` | int | Max results (default 20) |
| `language` | string | Filter by language |
| `under` | string | Path prefix filter |
| `repo` | string | Repository filter |

## context_answer

LLM-generated answers with code citations.

| Parameter | Type | Description |
|-----------|------|-------------|
| `query` | string/list | Question requiring explanation |
| `budget_tokens` | int | Token budget for context |
| `max_tokens` | int | Max tokens for answer |
| `expand` | bool | Generate query expansions |
| `include_snippet` | bool | Include code in response |
| `language` | string | Filter retrieval |
| `under` | string | Path prefix filter |

## info_request

Simplified discovery with optional explanations.

| Parameter | Type | Description |
|-----------|------|-------------|
| `info_request` | string | Natural language query |
| `include_explanation` | bool | Add NL summary |
| `include_relationships` | bool | Add imports/calls info |
| `limit` | int | Max results |

## pattern_search (Optional)

Structural code pattern matching. May not be enabled in all deployments.

| Parameter | Type | Description |
|-----------|------|-------------|
| `query` | string | Code snippet OR pattern description |
| `query_mode` | string | "code", "description", or "auto" |
| `language` | string | Language hint for code examples |
| `target_languages` | list | Filter results to languages |
| `min_score` | float | Minimum similarity (default 0.3) |
| `aroma_rerank` | bool | AROMA structural reranking |

## graph_query

Advanced Memgraph-backed graph traversals and impact analysis. Available to all SaaS users.

| Parameter | Type | Description |
|-----------|------|-------------|
| `symbol` | string | Symbol to analyze |
| `query_type` | string | "callers", "callees", "transitive_callers", "transitive_callees", "impact", "dependencies", "definition", "cycles" |
| `depth` | int | Max traversal depth (default varies by query type) |
| `limit` | int | Max results (default 20) |
| `language` | string | Filter by language |
| `under` | string | Path prefix filter |
| `repo` | string | Repository filter |
| `include_paths` | bool | Include full traversal paths in results |
| `output_format` | string | "json" or "toon" |

## batch_search

Run N independent `repo_search` calls in one MCP invocation. ~75% token savings.

| Parameter | Type | Description |
|-----------|------|-------------|
| `searches` | list[dict] | List of search specs (each with at least a `query` key) |
| `collection` | string | Shared collection (overridable per-search) |
| `limit` | int | Shared max results (overridable per-search) |
| `language` | string | Shared language filter |
| `under` | string | Shared path prefix filter |
| `repo` | string/list | Shared repository filter |
| `include_snippet` | bool | Shared snippet toggle |
| `rerank_enabled` | bool | Shared reranking toggle |
| `output_format` | string | "json" or "toon" |
| `compact` | bool | Minimal response fields |

**Returns:** `{ok, batch_results: [result_set_0, ...], count, elapsed_ms}`. Max 10 searches per batch.

## batch_symbol_graph

Run N independent `symbol_graph` queries in one MCP invocation. ~75% token savings.

| Parameter | Type | Description |
|-----------|------|-------------|
| `queries` | list[dict] | List of query specs (each must have a `symbol` key) |
| `collection` | string | Shared collection (overridable per-query) |
| `language` | string | Shared language filter |
| `under` | string | Shared path prefix filter |
| `repo` | string | Shared repository filter |
| `limit` | int | Shared max results |
| `depth` | int | Shared traversal depth |
| `output_format` | string | "json" or "toon" |

**Returns:** `{ok, batch_results: [result_set_0, ...], count, elapsed_ms}`. Max 10 queries per batch.

## batch_graph_query

Run N independent `graph_query` calls in one MCP invocation. ~75% token savings.

| Parameter | Type | Description |
|-----------|------|-------------|
| `queries` | list[dict] | List of query specs (each must have a `symbol` key) |
| `collection` | string | Shared collection (overridable per-query) |
| `repo` | string | Shared repository filter |
| `language` | string | Shared language filter |
| `depth` | int | Shared traversal depth |
| `limit` | int | Shared max results |
| `include_paths` | bool | Shared include traversal paths |
| `output_format` | string | "json" or "toon" |

**Returns:** `{ok, batch_results: [result_set_0, ...], count, elapsed_ms}`. Max 10 queries per batch.

## Memory Tools

**memory_store**
| Parameter | Type | Description |
|-----------|------|-------------|
| `information` | string | Knowledge to store |
| `metadata` | object | Tags, topic, priority, etc. |

**memory_find**
| Parameter | Type | Description |
|-----------|------|-------------|
| `query` | string | Search query |
| `limit` | int | Max results |
| `kind` | string | Filter by type |
| `topic` | string | Filter by topic |

**context_search**
| Parameter | Type | Description |
|-----------|------|-------------|
| `query` | string | Search query |
| `include_memories` | bool | Blend with stored memories |
| `per_source_limits` | object | `{"code": 6, "memory": 3}` |

## Index Management

> **SaaS mode:** Indexing is handled automatically by the VS Code extension upload service. `qdrant_index_root`, `qdrant_index`, and `qdrant_prune` are **not available** in SaaS. All search, symbol graph, memory, and session tools work normally.

**Available in all modes:**
- **qdrant_status** - Check health
- **qdrant_list** - List all collections (alias for `qdrant_status(list_all=True)`)
- **set_session_defaults** - Set collection, output_format, compact, limit
- **embedding_pipeline_stats** - Cache efficiency, bloom filter stats, pipeline performance

**Self-hosted only (not available in SaaS):**
- **qdrant_index_root** - `{"recreate": true}` to drop existing data
- **qdrant_index** - `{"subdir": "src/"}` for partial index
- **qdrant_prune** - Remove stale entries

