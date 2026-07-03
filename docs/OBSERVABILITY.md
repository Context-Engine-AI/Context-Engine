# Observability with OpenLit

Self-hosted LLM and vector database tracing via OpenTelemetry.

**Documentation:** [README](../README.md) · [Getting Started](GETTING_STARTED.md) · [Configuration](CONFIGURATION.md) · [IDE Clients](IDE_CLIENTS.md) · [MCP API](MCP_API.md) · [ctx CLI](CTX_CLI.md) · [Memory Guide](MEMORY_GUIDE.md) · [Architecture](ARCHITECTURE.md) · [Multi-Repo](MULTI_REPO_COLLECTIONS.md) · [Observability](OBSERVABILITY.md) · [Kubernetes](../deploy/kubernetes/README.md) · [VS Code Extension](vscode-extension.md) · [Troubleshooting](TROUBLESHOOTING.md) · [Development](DEVELOPMENT.md)

---

Context-Engine integrates with [OpenLit](https://github.com/openlit/openlit) for self-hosted LLM and vector database observability via OpenTelemetry.

## Quick Start

### 1. Start the Stack

```bash
docker compose -f docker-compose.yml -f docker-compose.openlit.yml up -d
```

### 2. Access Dashboard

- **URL**: http://localhost:3000
- **Email**: `user@openlit.io`
- **Password**: `openlituser`

### 3. Enable Tracing

Set in your `.env`:

```bash
OPENLIT_ENABLED=1
```

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  MCP Indexer    │────▶│    OpenLit      │────▶│   ClickHouse    │
│  MCP Memory     │     │  (OTLP 4318)    │     │   (Port 8123)   │
│  Context-Engine │     │  Dashboard:3000 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

## What Gets Traced

| Component | Operations |
|-----------|------------|
| **Qdrant** | `.query_points()`, `.scroll()`, `.upsert()`, `.delete()` |
| **LLMs** | GLM-4.x, Ollama, llama.cpp (via OpenAI-compatible API) |
| **Search** | `repo_search`, `context_search`, `context_answer` |

### Screenshots

<p align="center">
  <img src="../llm.png" alt="LLM Traces" width="50%"/>
</p>
<p align="center"><em>LLM call traces with token usage and latency</em></p>

<p align="center">
  <img src="../traces.png" alt="Request Traces" width="50%"/>
</p>
<p align="center"><em>End-to-end request traces showing search pipeline</em></p>

<p align="center">
  <img src="../vectors.png" alt="Vector DB Operations" width="50%"/>
</p>
<p align="center"><em>Qdrant vector database operations</em></p>

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENLIT_ENABLED` | `0` | Master toggle |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `http://openlit:4318` | OTLP HTTP endpoint |
| `OPENLIT_APP_NAME` | `context-engine` | Application name in traces |
| `OPENLIT_ENVIRONMENT` | `development` | Environment tag |

## Ports

| Port | Service | Purpose |
|------|---------|---------|
| 3000 | OpenLit | Dashboard UI |
| 4317 | OpenLit | OTLP gRPC receiver |
| 4318 | OpenLit | OTLP HTTP receiver |
| 8123 | ClickHouse | HTTP interface |
| 9000 | ClickHouse | Native protocol |

## Troubleshooting

### No Qdrant traces appearing

The `openlit` SDK must be initialized **before** importing `QdrantClient`. Context-Engine handles this via `scripts/openlit_init.py`, which is imported at the top of MCP server entrypoints.

**Required import order:**
```python
# 1. OpenLit init FIRST
from scripts import openlit_init

# 2. Then everything else
from qdrant_client import QdrantClient
```

### Qdrant client version

Use `qdrant-client>=1.15.0,<1.16.0`. Version 1.16+ changed to `.query_points()` which breaks OpenLit's instrumentation hooks.

## Disabling

Set `OPENLIT_ENABLED=0` or remove from environment. The SDK gracefully no-ops when disabled.
