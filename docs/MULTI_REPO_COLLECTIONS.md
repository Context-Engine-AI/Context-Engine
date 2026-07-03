# Multi-Repository Collection Architecture

**Documentation:** [README](../README.md) · [Getting Started](GETTING_STARTED.md) · [Configuration](CONFIGURATION.md) · [IDE Clients](IDE_CLIENTS.md) · [MCP API](MCP_API.md) · [ctx CLI](CTX_CLI.md) · [Memory Guide](MEMORY_GUIDE.md) · [Architecture](ARCHITECTURE.md) · [Multi-Repo](MULTI_REPO_COLLECTIONS.md) · [Observability](OBSERVABILITY.md) · [Kubernetes](../deploy/kubernetes/README.md) · [VS Code Extension](vscode-extension.md) · [Troubleshooting](TROUBLESHOOTING.md) · [Development](DEVELOPMENT.md)

---

**On this page:**
- [Overview](#overview)
- [Architecture Principles](#architecture-principles)
- [Indexing Multiple Repositories](#indexing-multiple-repositories)
- [Filtering by Repository](#filtering-by-repository)
- [Remote Deployment](#remote-deployment)

---

## Overview

Context Engine supports multi-repository operation through unified collection architecture:

- **Single unified collection** (default: `codebase`) for seamless cross-repo search
- **Per-repo metadata** for filtering and isolation when needed
- **Remote deployment** on Kubernetes clusters with stronger hardware
- **Minimal code changes** - existing single-repo workflows unchanged

## Architecture Principles

### 1. Unified Collection Model

All repositories index into **single shared collection** by default (`codebase`):

- **Seamless cross-repo search**: Query across all code at once
- **Simplified management**: One collection to monitor and maintain
- **Efficient resource usage**: Shared HNSW index and vector storage

### 2. Per-Repository Metadata

Each indexed chunk includes repository identification in its payload:

```json
{
  "metadata": {
    "repo": "my-backend-service",
    "path": "/work/src/api/handler.py",
    "host_path": "/Users/john/projects/backend/src/api/handler.py",
    "container_path": "/work/src/api/handler.py",
    "language": "python",
    "kind": "function",
    "symbol": "handle_request",
    ...
  }
}
```

**Key metadata fields for multi-repo:**
- `metadata.repo`: Logical repository name (auto-detected from git or folder name)
- `metadata.path`: Container path (always starts with `/work`)
- `metadata.host_path`: Original host filesystem path
- `metadata.container_path`: Normalized container path for remote deployments

### 3. Workspace State Management

Each repository maintains its own `.codebase/state.json` file:

```json
{
  "workspace_path": "/work",
  "created_at": "2025-01-15T10:30:00",
  "updated_at": "2025-01-15T14:22:00",
  "qdrant_collection": "codebase",
  "indexing_status": {
    "state": "watching",
    "started_at": "2025-01-15T14:20:00",
    "progress": {
      "files_processed": 1250,
      "total_files": 1250
    }
  },
  "last_activity": {
    "timestamp": "2025-01-15T14:22:00",
    "action": "indexed",
    "file_path": "/work/src/main.py"
  }
}
```

## Collection Naming Strategy

### Default: Unified Collection

**Recommended for most users:**
- Collection name: `codebase` (default)
- All repositories share this collection
- Filter by `metadata.repo` when you need repo-specific results

**Benefits:**
- Cross-repo search works out of the box
- Simpler configuration
- Better for monorepos and related microservices

### Optional: Per-Repository Collections

**Use when you need strict isolation:**
- Set `COLLECTION_NAME=my-service-name` per repository
- Each repo gets its own collection
- Requires explicit collection parameter in MCP calls

**Trade-offs:**
- More collections to manage
- Cross-repo search requires multiple queries
- Higher memory overhead (separate HNSW indexes)

## Remote Deployment Architecture

### Kubernetes Deployment Model

```
┌─────────────────────────────────────────────────────────────┐
│                     Kubernetes Cluster                       │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Qdrant     │  │  Memory MCP  │  │ Indexer MCP  │     │
│  │  (StatefulSet)│  │ (Deployment) │  │ (Deployment) │     │
│  │  Port: 6333  │  │  Port: 8000  │  │  Port: 8001  │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│         │                  │                  │             │
│         └──────────────────┴──────────────────┘             │
│                            │                                │
│  ┌─────────────────────────┴────────────────────────────┐  │
│  │           Persistent Volume (repos)                   │  │
│  │  /repos/backend/    /repos/frontend/   /repos/ml/    │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Watcher    │  │   Watcher    │  │   Watcher    │     │
│  │  (backend)   │  │  (frontend)  │  │    (ml)      │     │
│  │ (Deployment) │  │ (Deployment) │  │ (Deployment) │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Uploader Pod (optional)                  │  │
│  │  Accepts file uploads and writes to /repos volume    │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
         │                    │                    │
         │                    │                    │
    ┌────▼────┐          ┌────▼────┐         ┌────▼────┐
    │  Dev    │          │  Dev    │         │  Dev    │
    │ Client  │          │ Client  │         │ Client  │
    │   #1    │          │   #2    │         │   #3    │
    └─────────┘          └─────────┘         └─────────┘
```

### Volume Structure

```
/repos/
├── backend/
│   ├── .codebase/
│   │   └── state.json          # Collection: codebase, repo: backend
│   ├── src/
│   └── ...
├── frontend/
│   ├── .codebase/
│   │   └── state.json          # Collection: codebase, repo: frontend
│   ├── src/
│   └── ...
└── ml-service/
    ├── .codebase/
    │   └── state.json          # Collection: codebase, repo: ml-service
    ├── models/
    └── ...
```

## MCP Tool Collection Support

All MCP tools accept an optional `collection` parameter:

### Search Tools

```python
# Search across all repos in the unified collection
await repo_search(
    query="authentication handler",
    limit=10
)

# Filter to specific repo
await repo_search(
    query="authentication handler",
    limit=10,
    # Use metadata filter (not collection param) for repo filtering
    # Collection param is for switching between different Qdrant collections
)

# Search in a different collection (if using per-repo collections)
await repo_search(
    query="authentication handler",
    collection="backend-service",
    limit=10
)
```

### Memory Tools

```python
# Store memory in default collection
await memory_store(
    information="Use JWT tokens for API authentication",
    metadata={"kind": "memory", "topic": "auth", "repo": "backend"}
)

# Store in specific collection
await memory_store(
    information="Frontend uses OAuth2 flow",
    metadata={"kind": "memory", "topic": "auth", "repo": "frontend"},
    collection="codebase"
)
```

### Indexing Tools

```python
# Index a specific workspace into the unified collection
await qdrant_index_root(
    collection="codebase"  # Optional, defaults to workspace state
)

# Index with explicit collection override
await qdrant_index(
    subdir="",
    recreate=False,
    collection="my-custom-collection"
)
```

## Filtering by Repository

Use Qdrant's payload filters to scope searches to specific repositories:

```python
# In hybrid_search.py or via MCP tools
results = hybrid_search(
    queries=["authentication"],
    collection="codebase",
    # Add repo filter via metadata
    # (Implementation detail: tools should support repo= parameter)
)
```

**Recommended enhancement:** Add `repo` parameter to search tools that translates to a payload filter on `metadata.repo`.

## Workspace Discovery

The `list_workspaces` function scans for all `.codebase/state.json` files:

```python
from scripts.workspace_state import list_workspaces

workspaces = list_workspaces(search_root="/repos")
# Returns:
# [
#   {
#     "workspace_path": "/repos/backend",
#     "collection_name": "codebase",
#     "last_updated": "2025-01-15T14:22:00",
#     "indexing_state": "watching"
#   },
#   {
#     "workspace_path": "/repos/frontend",
#     "collection_name": "codebase",
#     "last_updated": "2025-01-15T14:20:00",
#     "indexing_state": "idle"
#   }
# ]
```

## Migration Guide

### From Single-Repo to Multi-Repo

**No migration needed!** The default unified collection model works automatically:

1. **Keep using `codebase` collection** (default)
2. **Index additional repos** - they'll share the same collection
3. **Filter by repo name** when you need repo-specific results

### From Per-Repo Collections to Unified

If you previously used separate collections per repo:

1. **Create new unified collection:**
   ```bash
   COLLECTION_NAME=codebase make reindex
   ```

2. **Reindex all repositories** into the unified collection:
   ```bash
   for repo in backend frontend ml-service; do
     HOST_INDEX_PATH=/path/to/$repo COLLECTION_NAME=codebase make index
   done
   ```

3. **Update MCP client configs** to use `codebase` collection

4. **Optional:** Delete old per-repo collections via Qdrant API

## Best Practices

### 1. Use Unified Collection by Default

- Simplifies cross-repo search
- Reduces operational overhead
- Better for related codebases

### 2. Set Meaningful Repo Names

- Use `REPO_NAME` env var or rely on git repo name
- Keep names consistent across environments
- Use kebab-case: `backend-api`, `frontend-web`, `ml-training`

### 3. Leverage Payload Indexes

The indexer creates payload indexes on `metadata.repo` for efficient filtering:

```python
# Fast repo-scoped search (uses payload index)
results = client.search(
    collection_name="codebase",
    query_vector=embedding,
    query_filter=models.Filter(
        must=[
            models.FieldCondition(
                key="metadata.repo",
                match=models.MatchValue(value="backend-api")
            )
        ]
    )
)
```

### 4. Monitor Collection Health

```bash
# Check collection status
make qdrant-status

# List all collections
make qdrant-list

# Prune stale points
make prune
```

### 5. Use Watchers Per Repository

Deploy one watcher per repository in Kubernetes:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: watcher-backend
spec:
  replicas: 1
  template:
    spec:
      containers:
      - name: watcher
        image: context-engine-indexer:latest
        command: ["python", "/app/scripts/watch_index.py"]
        env:
        - name: WATCH_ROOT
          value: "/repos/backend"
        - name: COLLECTION_NAME
          value: "codebase"
        - name: REPO_NAME
          value: "backend"
        volumeMounts:
        - name: repos
          mountPath: /repos
          subPath: backend
```

## Compatibility

### Backward Compatibility

All existing single-repo workflows continue to work:

- Default collection name: `codebase`
- Workspace state auto-created if missing
- Collection parameter optional in all MCP tools
- Existing Docker Compose setup unchanged

### Forward Compatibility

The architecture supports future enhancements:

- Multi-collection queries (search across multiple collections)
- Collection-level access control
- Collection-specific embedding models
- Cross-collection deduplication

## See Also

- [Kubernetes Deployment Guide](../deploy/kubernetes/README.md)
- [MCP API Reference](MCP_API.md)
- [Architecture Overview](ARCHITECTURE.md)
- [Development Guide](DEVELOPMENT.md)

