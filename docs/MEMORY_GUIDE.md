# Memory Usage Guide

Best practices for using Context Engine's memory system effectively.

**Documentation:** [README](../README.md) · [Getting Started](GETTING_STARTED.md) · [Configuration](CONFIGURATION.md) · [IDE Clients](IDE_CLIENTS.md) · [MCP API](MCP_API.md) · [ctx CLI](CTX_CLI.md) · [Memory Guide](MEMORY_GUIDE.md) · [Architecture](ARCHITECTURE.md) · [Multi-Repo](MULTI_REPO_COLLECTIONS.md) · [Observability](OBSERVABILITY.md) · [Kubernetes](../deploy/kubernetes/README.md) · [VS Code Extension](vscode-extension.md) · [Troubleshooting](TROUBLESHOOTING.md) · [Development](DEVELOPMENT.md)

---

**On this page:**
- [When to Use Memories vs Code Search](#when-to-use-memories-vs-code-search)
- [Recommended Metadata Schema](#recommended-metadata-schema)
- [Example Operations](#example-operations)
- [Memory Blending](#memory-blending)
- [Collection Naming](#collection-naming)

---

## When to Use Memories vs Code Search

| Use Memories For | Use Code Search For |
|------------------|---------------------|
| Conventions, runbooks, decisions | APIs, functions, classes |
| Links, known issues, FAQs | Configuration files |
| "How we do X here" notes | Cross-file relationships |
| Team wiki-style content | Anything you'd grep for |

**Blend both** for tasks like "how to run E2E tests" where instructions (memory) reference scripts in the repo (code).

---

## Recommended Metadata Schema

Memory entries are stored as points in Qdrant with a consistent payload:

| Key | Type | Description |
|-----|------|-------------|
| `kind` | string | **Required.** Always "memory" to enable filtering/blending |
| `topic` | string | Short category (e.g., "dev-env", "release-process") |
| `tags` | list[str] | Searchable tags (e.g., ["qdrant", "indexing", "prod"]) |
| `source` | string | Origin (e.g., "chat", "manual", "tool", "issue-123") |
| `author` | string | Who added it (username or email) |
| `created_at` | string | ISO8601 timestamp (UTC) |
| `expires_at` | string | ISO8601 timestamp if memory should be pruned later |
| `repo` | string | Optional repo identifier for shared instances |
| `link` | string | Optional URL to docs, tickets, or dashboards |
| `priority` | float | 0.0-1.0 weight for ranking when blending |

**Tips:**
- Keep values small (short strings, small lists)
- Put details in the `information` text, not payload
- Use lowercase snake_case keys
- For secrets/PII: store references or vault paths, never plaintext

---

## Example Operations

### Store Memory

Via MCP Memory server tool `memory_store`:

```json
{
  "information": "Run full reset: INDEX_MICRO_CHUNKS=1 MAX_MICRO_CHUNKS_PER_FILE=200 make reset-dev",
  "metadata": {
    "kind": "memory",
    "topic": "dev-env",
    "tags": ["make", "reset"],
    "source": "chat"
  }
}
```

### Find Memories

Via MCP Memory server tool `memory_find`:

```json
{
  "query": "reset-dev",
  "limit": 5
}
```

### Blend Memories into Code Search

Via Indexer MCP `context_search`:

```json
{
  "query": "async file watcher",
  "include_memories": true,
  "limit": 5,
  "include_snippet": true
}
```

---

## Query Tips

- Use precise queries (2-5 tokens)
- Add synonyms if needed; server supports multiple phrasings
- Combine `topic`/`tags` in memory text for easier discovery

---

## Enable Memory Blending

1. Ensure Memory MCP is running on :8000 (default in compose)

2. Enable SSE memory blending on Indexer MCP:

```yaml
services:
  mcp_indexer:
    environment:
      - MEMORY_SSE_ENABLED=true
      - MEMORY_MCP_URL=http://mcp:8000/sse
      - MEMORY_MCP_TIMEOUT=6
```

3. Restart indexer:

```bash
docker compose up -d mcp_indexer
```

4. Validate with `context_search`:

```json
{
  "query": "your test memory text",
  "include_memories": true,
  "limit": 5
}
```

Expected: non-zero results with blended items; memory hits will have `metadata.kind = "memory"`.

---

## Collection Naming Strategies

Different hash lengths for different workspace types:

**Local Workspaces:** `repo-name-8charhash`
- Example: `Anesidara-e8d0f5fc`
- Used by local indexer/watcher
- Assumes unique repo names within workspace

**Remote Uploads:** `folder-name-16charhash-8charhash`
- Example: `testupload2-04e680d5939dd035-b8b8d4cc`
- Collision avoidance for duplicate folder names
- 16-char hash identifies workspace, 8-char hash identifies collection

---

## Operational Notes

- Collection name comes from `COLLECTION_NAME` (see .env)
- This stack defaults to a single collection for both code and memories
- Filtering uses `metadata.kind` to distinguish memory from code
- Consider pruning expired memories by filtering `expires_at < now`

---

## Backup and Migration

### Memory Backup/Restore Scripts

```bash
# Export memories to JSON
python scripts/memory_backup.py --collection codebase --output memories.json

# Restore memories from backup
python scripts/memory_restore.py --input memories.json --collection codebase
```

For production-grade backup/migration strategies, see the official Qdrant documentation for snapshots and export/import. For local development, rely on Docker volumes and reindexing when needed.
