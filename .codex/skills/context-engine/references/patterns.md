# Common Search Patterns

This file keeps only stable usage patterns for Codex. For exhaustive tool behavior, defer to `skills/context-engine/SKILL.md`.

## Optional Session Bootstrap

If you expect repeated searches, consider setting lightweight defaults early:

```json
// set_session_defaults
{"output_format": "toon", "compact": true, "limit": 5}
```

## Two-Phase Search Strategy

### Phase 1: Discovery

```json
{"query": "authentication", "limit": 3, "compact": true, "per_path": 1}
```

### Phase 2: Deep Dive

```json
{"query": "JWT token validation", "limit": 8, "include_snippet": true, "context_lines": 5}
```

## Multi-Query or Batched Search

For a single search with multiple phrasings:

```json
{
  "query": ["authentication handler", "login middleware", "auth validation"],
  "limit": 10
}
```

For multiple independent code searches, prefer `batch_search` over repeated single-tool calls.

## Direct Symbol Relationships

Direct callers:

```json
{"symbol": "processPayment", "query_type": "callers", "limit": 15}
```

Definitions:

```json
{"symbol": "UserService", "query_type": "definition"}
```

Importers:

```json
{"symbol": "utils/helpers", "query_type": "importers", "limit": 10}
```

Multi-hop callers:

```json
{"symbol": "processPayment", "query_type": "callers", "depth": 2, "limit": 20}
```

## Impact / Dependency Analysis

If `graph_query` is available, use it for deeper impact/dependency/cycle work.

If it is not available, stay with `symbol_graph` plus targeted `search` rather than assuming a deeper graph tool exists.

## Cross-Repo Search

Use `cross_repo_search` first for multi-repo questions:

```json
{"query": "authentication flow", "discover": "auto"}
```

Boundary tracing pattern:

```json
{"query": "login submit", "trace_boundary": true}
```

If you already know you are inside one unified collection, repo filters are still fine:

```json
{"query": "shared types", "repo": ["frontend", "backend"]}
```

## Filtering by Language / Path

```json
{
  "query": "error handling",
  "language": "python",
  "under": "src/api/",
  "not_glob": ["**/test_*", "**/*_test.*"]
}
```

## Pattern Search (if enabled)

Natural-language pattern:

```json
{"query": "retry with exponential backoff", "limit": 10}
```

Code-shaped pattern:

```json
{"query": "try: ... except: logger.error()", "query_mode": "code"}
```

## Git History Search

Find commits about a topic:

```json
{"query": "fixed authentication bug", "limit": 10}
```

File change history:

```json
{"path": "src/api/auth.py", "include_commits": true}
```

## Memory Patterns

Store an architectural decision:

```json
{
  "information": "Auth service uses JWT with 24h expiry. Refresh tokens last 7 days.",
  "metadata": {"topic": "auth", "kind": "decision"}
}
```

Blend code search with stored notes:

```json
{
  "query": "authentication flow",
  "include_memories": true,
  "per_source_limits": {"code": 6, "memory": 3}
}
```

## Anti-Patterns

- Avoid defaulting to grep/find/file-open for cross-file exploration; use MCP search tools first.
- Narrow grep/file-open usage is still okay for exact literal confirmation, exact path/file confirmation, or opening a file you already identified for editing.
- Avoid repeated wide searches when `batch_search`, multi-query search, or a two-phase search will do.
- Avoid large `limit` plus large snippets during discovery; start compact and deepen only after you find the right area.

