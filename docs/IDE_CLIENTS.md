# IDE & Client Configuration

Connect IDE to running Context-Engine stack. No need to clone this repo into your project.

**Documentation:** [README](../README.md) · [Getting Started](GETTING_STARTED.md) · [Configuration](CONFIGURATION.md) · [IDE Clients](IDE_CLIENTS.md) · [MCP API](MCP_API.md) · [ctx CLI](CTX_CLI.md) · [Memory Guide](MEMORY_GUIDE.md) · [Architecture](ARCHITECTURE.md) · [Multi-Repo](MULTI_REPO_COLLECTIONS.md) · [Observability](OBSERVABILITY.md) · [Kubernetes](../deploy/kubernetes/README.md) · [VS Code Extension](vscode-extension.md) · [Troubleshooting](TROUBLESHOOTING.md) · [Development](DEVELOPMENT.md)

---

**On this page:**
- [Quick Start](#quick-start)
- [Supported Clients](#supported-clients)
- [SSE Clients](#sse-clients-port-80008001)
- [RMCP Clients](#rmcp-clients-port-80028003)
- [Mixed Transport](#mixed-transport-examples)
- [Remote Server](#remote-server)
- [Verification](#verification)

---

## Quick Start

**Prerequisites:** Context-Engine running (localhost, remote server, or Kubernetes).

**Minimal config (SSE)** — for clients that only understand SSE or use `mcp-remote`:
```json
{
  "mcpServers": {
    "context-engine": { "url": "http://localhost:8001/sse" }
  }
}
```

**HTTP (recommended for RMCP-capable IDEs)** — prefer when IDE supports HTTP MCP / RMCP (Claude Code, Windsurf, Qodo, etc.):

```json
{
  "mcpServers": {
    "memory": { "url": "http://localhost:8002/mcp" },
    "qdrant-indexer": { "url": "http://localhost:8003/mcp" }
  }
}
```

HTTP `/mcp` avoids FastMCP initialization race that SSE clients hit when sending `listTools` in parallel with `initialize`, causing:

```text
Failed to validate request: Received request before initialization was complete
```

If tools/resources appear only after second reconnect using SSE, switch to HTTP endpoints.

Replace `localhost` with server IP/hostname for remote setups.

---

## Supported Clients

| Client | Transport | Notes |
|--------|-----------|-------|
| Roo | SSE/RMCP | Both SSE and RMCP connections |
| Cline | SSE/RMCP | Both SSE and RMCP connections |
| Windsurf | SSE/RMCP | Both SSE and RMCP connections |
| Zed | SSE | Uses mcp-remote bridge |
| Kiro | SSE | Uses mcp-remote bridge |
| Qodo | RMCP | Direct HTTP endpoints |
| OpenAI Codex | RMCP | TOML config |
| Augment | SSE | Simple JSON configs |
| AmpCode | SSE | Simple URL for SSE endpoints |
| Claude Code CLI | SSE / HTTP (RMCP) | Simple JSON configs via .mcp.json |

**Claude Desktop (Connectors):** Claude Desktop also supports remote MCP servers over SSE and streamable HTTP, but configuration happens via the Claude Connectors UI (Settings → Connectors on claude.ai), not local `.mcp.json`. Treat Context-Engine as a normal remote MCP server there; this guide focuses on IDEs where you control MCP URLs/config files directly (Claude Code, Windsurf, etc.).

---

## SSE Clients (port 8000/8001)

### Roo / Cline / Windsurf

```json
{
  "mcpServers": {
    "memory": { "type": "sse", "url": "http://localhost:8000/sse", "disabled": false },
    "qdrant-indexer": { "type": "sse", "url": "http://localhost:8001/sse", "disabled": false }
  }
}
```

### Augment

**Option 1: Direct SSE connection** (requires Context-Engine running locally):
```json
{
  "mcpServers": {
    "memory": { "type": "sse", "url": "http://localhost:8000/sse", "disabled": false },
    "qdrant-indexer": { "type": "sse", "url": "http://localhost:8001/sse", "disabled": false }
  }
}
```

**Option 2: MCP Bridge** (recommended - unified server with workspace awareness):
```json
{
  "mcpServers": {
    "context-engine": {
      "command": "npx",
      "args": [
        "@context-engine-bridge/context-engine-mcp-bridge",
        "mcp-serve",
        "--workspace",
        "/path/to/your/project",
        "--indexer-url",
        "http://localhost:8003/mcp",
        "--memory-url",
        "http://localhost:8002/mcp"
      ],
      "env": {
        "COLLECTION_NAME": "codebase"
      }
    }
  }
}
```

**Notes:**
- Replace `/path/to/your/project` with your actual workspace path
- The `COLLECTION_NAME` env var ensures searches use the correct Qdrant collection
- The bridge provides a unified MCP server combining both indexer and memory tools
- Use `http://localhost:8003/mcp` and `http://localhost:8002/mcp` for HTTP transport (recommended)

### Kiro

Create `.kiro/settings/mcp.json` in your workspace:

```json
{
  "mcpServers": {
    "qdrant-indexer": { "command": "npx", "args": ["mcp-remote", "http://localhost:8001/sse", "--transport", "sse-only"] },
    "memory": { "command": "npx", "args": ["mcp-remote", "http://localhost:8000/sse", "--transport", "sse-only"] }
  }
}
```

**Notes:**
- Kiro expects command/args (stdio). `mcp-remote` bridges to remote SSE endpoints.
- If `npx` prompts in your environment, add `-y` right after `npx`.
- Workspace config overrides user-level config (`~/.kiro/settings/mcp.json`).

**Troubleshooting:**
- Error: "Enabled MCP Server must specify a command, ignoring." → Use the command/args form; do not use type:url in Kiro.

### Zed

Add to your Zed `settings.json` (Command Palette → "Settings: Open Settings (JSON)"):

```json
{
  "qdrant-indexer": {
    "command": "npx",
    "args": ["mcp-remote", "http://localhost:8001/sse", "--transport", "sse-only"],
    "env": {}
  }
}
```

**Notes:**
- Zed expects MCP servers at the root level of settings.json
- Uses command/args (stdio). mcp-remote bridges to remote SSE endpoints
- If npx prompts, add `-y` right after npx: `"args": ["-y", "mcp-remote", ...]`

**Alternative (direct HTTP):**
```json
{
  "qdrant-indexer": {
    "type": "http",
    "url": "http://localhost:8003/mcp"
  }
}
```

---

## RMCP Clients (port 8002/8003)

### Qodo

Add each MCP tool separately through the UI:

**Tool 1 - memory:**
```json
{
  "memory": { "url": "http://localhost:8002/mcp" }
}
```

**Tool 2 - qdrant-indexer:**
```json
{
  "qdrant-indexer": { "url": "http://localhost:8003/mcp" }
}
```

**Note:** Qodo can talk to RMCP endpoints directly, no `mcp-remote` wrapper needed.

### OpenAI Codex

TOML configuration:

```toml
experimental_use_rmcp_client = true

[mcp_servers.memory_http]
url = "http://127.0.0.1:8002/mcp"

[mcp_servers.qdrant_indexer_http]
url = "http://127.0.0.1:8003/mcp"
```

---

## Mixed Transport (stdio + SSE)

### Windsurf/Cursor

```json
{
  "mcpServers": {
    "qdrant": {
      "command": "uvx",
      "args": ["mcp-server-qdrant"],
      "env": {
        "QDRANT_URL": "http://localhost:6333",
        "COLLECTION_NAME": "my-collection",
        "EMBEDDING_MODEL": "BAAI/bge-base-en-v1.5"
      },
      "disabled": false
    }
  }
}
```

---

## Remote Server

When Context-Engine runs on a remote server (e.g., `context.yourcompany.com`):

```json
{
  "mcpServers": {
    "context-engine": { "url": "http://context.yourcompany.com:8001/sse" }
  }
}
```

If your IDE supports HTTP MCP / RMCP, prefer the HTTP endpoints instead:

```json
{
  "mcpServers": {
    "memory": { "url": "http://context.yourcompany.com:8002/mcp" },
    "qdrant-indexer": { "url": "http://context.yourcompany.com:8003/mcp" }
  }
}
```

This uses the HTTP `/mcp` transport and avoids the initialization race described above.

**Indexing your local project to the remote server:**
```bash
# Using VS Code extension (recommended)
# Install vscode-context-engine, configure server URL, click "Upload Workspace"

# Using CLI
scripts/remote_upload_client.py --server http://context.yourcompany.com:9090 --path /your/project
```

> See [docs/MULTI_REPO_COLLECTIONS.md](MULTI_REPO_COLLECTIONS.md) for multi-repo and Kubernetes deployment.

---

## Important Notes for IDE Agents

- **Do not send null values** to MCP tools. Omit the field or pass an empty string "" instead.
- **qdrant-index examples:**
  - `{"subdir":"","recreate":false,"collection":"my-collection","repo_name":"workspace"}`
  - `{"subdir":"scripts","recreate":true}`
- For indexing repo root with no params, use `qdrant_index_root` (zero-arg) or call `qdrant-index` with `subdir:""`.

---

## Verification

After configuring, you should see tools from both servers:
- `store`, `find` (Memory)
- `repo_search`, `code_search`, `context_search`, `context_answer` (Indexer)
- `qdrant_list`, `qdrant_index`, `qdrant_prune`, `qdrant_status` (Indexer)

Test connectivity:
- Call `qdrant_list` to confirm Qdrant connectivity
- Call `qdrant_status` to check collection point count and last indexed time
- Call `qdrant_index` with `{ "subdir": "scripts", "recreate": true }` to test indexing
- Call `context_search` with `{ "include_memories": true }` to test memory blending

---

## Troubleshooting

### Search returns empty results

If `repo_search` or `context_search` returns no results but `qdrant_status` shows points exist:

1. **Check the collection name** - The search may be using a different collection than expected
2. **Set the collection explicitly** via `COLLECTION_NAME` env var or `--collection` flag
3. **Use `set_session_defaults`** - Call `set_session_defaults(collection="codebase")` to set the default for your session
4. **Verify with `qdrant_list`** - Lists all available collections to confirm the correct name

### MCP Bridge not finding collection

When using `@context-engine-bridge/context-engine-mcp-bridge`, ensure you set `COLLECTION_NAME`:

```json
{
  "env": {
    "COLLECTION_NAME": "codebase"
  }
}
```

The default collection name is `codebase` unless you've configured a different one during indexing.

