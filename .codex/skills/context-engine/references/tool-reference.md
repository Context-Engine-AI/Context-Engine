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
| `query_type` | string | "callers", "definition", "importers", "callees" |
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

**qdrant_index_root** - `{"recreate": true}` to drop existing data
**qdrant_index** - `{"subdir": "src/"}` for partial index
**qdrant_prune** - Remove stale entries
**qdrant_status** - Check health
**set_session_defaults** - Set collection, output_format, compact, limit

