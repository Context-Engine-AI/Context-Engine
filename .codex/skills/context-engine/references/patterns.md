# Common Search Patterns

## Session Bootstrap

Always start a session with defaults to avoid repeating parameters:
```json
// set_session_defaults
{"output_format": "toon", "compact": true, "limit": 5}
```

## Two-Phase Search Strategy

### Phase 1: Discovery (find relevant areas)
```json
{"query": "authentication", "limit": 3, "compact": true, "per_path": 1}
```

### Phase 2: Deep Dive (get details)
```json
{"query": "JWT token validation", "limit": 8, "include_snippet": true, "context_lines": 5}
```

## Multi-Query Fusion

For complex concepts, pass multiple query variations:
```json
{
  "query": ["authentication handler", "login middleware", "auth validation"],
  "limit": 10
}
```

## Finding Callers (symbol_graph)

Direct callers:
```json
{"symbol": "processPayment", "query_type": "callers", "limit": 15}
```

Callers of callers (multi-hop):
```json
{"symbol": "processPayment", "query_type": "callers", "depth": 2, "limit": 20}
```

## Finding Definitions

```json
{"symbol": "UserService", "query_type": "definition"}
```

## Finding Importers

```json
{"symbol": "utils/helpers", "query_type": "importers", "limit": 10}
```

## Cross-Repo Search

Search specific repos:
```json
{"query": "shared types", "repo": ["frontend", "backend"]}
```

Search all indexed repos:
```json
{"query": "shared types", "repo": "*"}
```

## Filtering by Language/Path

```json
{
  "query": "error handling",
  "language": "python",
  "under": "src/api/",
  "not_glob": ["**/test_*", "**/*_test.*"]
}
```

## Pattern Search (if enabled)

Find retry patterns:
```json
{"query": "retry with exponential backoff", "limit": 10}
```

Find error handling:
```json
{"query": "try: ... except: logger.error()", "query_mode": "code"}
```

Cross-language (Go pattern → Python/Rust):
```json
{"query": "if err != nil { return err }", "language": "go", "target_languages": ["python", "rust"]}
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

Store architectural decision:
```json
{
  "information": "Auth service uses JWT with 24h expiry. Refresh tokens last 7 days.",
  "metadata": {"topic": "auth", "kind": "decision"}
}
```

Find stored knowledge:
```json
{"query": "token expiration", "limit": 5}
```

Blend code search with memories:
```json
{
  "query": "authentication flow",
  "include_memories": true,
  "per_source_limits": {"code": 6, "memory": 3}
}
```

## Anti-Patterns (AVOID)

❌ `grep -r "auth" .` → Use `repo_search("authentication mechanisms")`
❌ `cat file.py` to understand → Use `context_answer("how does file.py work")`
❌ `find . -name "*.py"` → Use `repo_search` with `path_glob`
❌ Sequential searches → Fire parallel calls in one message
❌ `limit=20, include_snippet=true` → Token waste, use two-phase instead

