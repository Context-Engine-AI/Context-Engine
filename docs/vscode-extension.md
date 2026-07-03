# VS Code Extension

Context Engine Uploader extension for automatic workspace sync and Prompt+ integration.

**Documentation:** [README](../README.md) · [Getting Started](GETTING_STARTED.md) · [Configuration](CONFIGURATION.md) · [IDE Clients](IDE_CLIENTS.md) · [MCP API](MCP_API.md) · [ctx CLI](CTX_CLI.md) · [Memory Guide](MEMORY_GUIDE.md) · [Architecture](ARCHITECTURE.md) · [Multi-Repo](MULTI_REPO_COLLECTIONS.md) · [Observability](OBSERVABILITY.md) · [Kubernetes](../deploy/kubernetes/README.md) · [VS Code Extension](vscode-extension.md) · [Troubleshooting](TROUBLESHOOTING.md) · [Development](DEVELOPMENT.md)

---

**On this page:**
- [Quick Start](#quick-start)
- [Features](#features)
- [Workflow Examples](#workflow-examples)
- [Installation](#installation)
- [Configuration](#configuration)
- [Commands](#commands-and-lifecycle)

---

## Quick Start

1. **Install**: Build the `.vsix` and install in VS Code (see [Installation](#installation))
2. **Configure server**: Settings → `contextEngineUploader.endpoint` → `http://localhost:8004` for the dev-remote Docker stack (or your upload_service URL)
3. **Index workspace**: Click status bar button or run `Context Engine Uploader: Start`
4. **Use Prompt+**: Select code, click `Prompt+` in status bar to enhance with AI

## Features

- **Auto-sync**: Force sync on startup + watch mode keeps workspace indexed
- **Prompt+ button**: Status bar button to enhance selected text with unicorn mode
- **Output channel**: Real-time logs for force-sync and watch operations
- **GPU decoder support**: Configure llama.cpp, Ollama, or GLM as decoder backend
- **Remote server support**: Index to any Context-Engine server (local, remote, Kubernetes)
- **MCP + ctx scaffolding**: Auto-writes Claude Code/Windsurf MCP configs, optional Claude prompt hook, and `ctx_config.json` wired to collection and decoder settings

## Workflow Examples

### Local Development (dev-remote stack)
Context-Engine running via `docker-compose.yml` on same machine:
```
Endpoint: http://localhost:8004
Target Path: (leave empty - uses current workspace or auto-detect)
```
Open any project → extension auto-syncs → MCP tools have code context.

### Remote Server
Context-Engine on team server:
```
Endpoint: http://context.yourcompany.com:9090
Target Path: /Users/you/projects/my-app
```
Local code indexed to shared server. Team members search across indexed repos.

### Multi-Project Workflow
Index multiple projects to same server:
1. Open Project A → auto-syncs to `codebase` collection
2. Open Project B → auto-syncs to same collection
3. MCP tools search across both projects

### Prompt+ Enhancement
1. Select code or write a prompt in your editor
2. Click `Prompt+` in status bar (or run command)
3. Extension runs `ctx.py --unicorn` with your selection
4. Enhanced prompt replaces selection with code-grounded context

**Example input:**
```
Add error handling to the upload function
```

**Example output:**
```
Looking at upload_service.py lines 120-180, the upload_file() function currently lacks error handling. Add try/except blocks to handle:
1. Network timeouts (requests.exceptions.Timeout)
2. Invalid file paths (FileNotFoundError)
3. Server errors (HTTP 5xx responses)

Reference the existing error patterns in remote_upload_client.py lines 45-67 which use structured logging via logger.error().
```

### Claude Code hook (optional)

For Claude Code, you can also enable a `/ctx` hook so that each prompt is expanded via `ctx.py` before it reaches Claude:

- The extension can auto-write MCP config and, on Linux/dev-remote, a Claude hook when `claudeHookEnabled` is turned on.
- See `docs/ctx/claude-hook-example.json` for a minimal `UserPromptSubmit` hook that shells out to `ctx-hook-simple.sh`.

## Installation

### Build Prerequisites
- Node.js 18+ and npm
- Python 3 available on PATH for runtime testing
- VS Code Extension Manager `vsce` (`npm install -g @vscode/vsce`) or run via `npx`

Install Dependencies
--------------------
```bash
cd vscode-extension/context-engine-uploader
npm install
```

Package the Extension
---------------------
```bash
cd vscode-extension/context-engine-uploader
npx vsce package
```
This emits a `.vsix` file such as `context-engine-uploader-0.1.0.vsix`.

Test Locally
------------
1. In VS Code, open the command palette and select `Developer: Install Extension from Location...`.
2. Pick the generated `.vsix`.
3. Reload the window when prompted.

Key Settings After Install
--------------------------
- `Context Engine Upload` output channel shows force-sync and watch logs.
- `Context Engine Uploader: Index Codebase` command or status bar button runs a force sync followed by watch.
- Configure `contextEngineUploader.targetPath`, `endpoint`, and (optionally) MCP settings (`mcpIndexerUrl`, `mcpMemoryUrl`, `mcpTransportMode`, `mcpClaudeEnabled`, `mcpWindsurfEnabled`, `autoWriteMcpConfigOnStartup`) under Settings → Extensions → Context Engine Uploader.

## Prerequisites
Python 3.8+ must be available on the host so the bundled client can run.

## Configuration

All settings live under `Context Engine Uploader` in the VS Code settings UI or `settings.json`.

| Setting | Description |
| --- | --- |
| `contextEngineUploader.runOnStartup` | Runs the force sync automatically after VS Code starts, then starts watch mode. Leave enabled to mirror the old manual workflow. |
| `contextEngineUploader.pythonPath` | Python executable to use (`python3` by default). |
| `contextEngineUploader.scriptWorkingDirectory` | Optional override for the folder that contains `standalone_upload_client.py`. Leave blank to use the extension’s own copy. |
| `contextEngineUploader.decoderUrl` | Override `DECODER_URL` passed into `scripts/ctx.py` when running Prompt+. Defaults to local llama.cpp (`http://localhost:8081`, auto-appends `/completion`). Use `http://localhost:11434/api/chat` for Ollama. |
| `contextEngineUploader.useGlmDecoder` | Set `REFRAG_RUNTIME=glm` for Prompt+ to hit GLM instead of Ollama/llama.cpp. |
| `contextEngineUploader.useGpuDecoder` | Set `USE_GPU_DECODER=1` so ctx.py prefers the GPU llama.cpp sidecar. |
| `contextEngineUploader.targetPath` | Absolute path that should be passed to `--path` (for example `/users/mycode`). |
| `contextEngineUploader.endpoint` | Remote endpoint passed to `--endpoint`, defaulting to `http://localhost:8004`. |
| `contextEngineUploader.intervalSeconds` | Poll interval for watch mode. Set to `5` to match the previous command file. |
| `contextEngineUploader.extraForceArgs` | Optional string array appended to the force invocation. Leave empty for the standard workflow. |
| `contextEngineUploader.extraWatchArgs` | Optional string array appended to the watch invocation. |
| `contextEngineUploader.mcpClaudeEnabled` | Enable writing the project-local `.mcp.json` used by Claude Code MCP clients. |
| `contextEngineUploader.mcpWindsurfEnabled` | Enable writing Windsurf’s global MCP config. |
| `contextEngineUploader.autoWriteMcpConfigOnStartup` | Automatically run “Write MCP Config” on activation to keep `.mcp.json`, Windsurf config, and Claude hook in sync with these settings. |
| `contextEngineUploader.mcpTransportMode` | Transport for MCP configs: `sse-remote` (SSE via mcp-remote) or `http` (direct `/mcp` endpoints). |
| `contextEngineUploader.mcpIndexerUrl` | MCP indexer URL used when writing configs. For dev-remote, typical values are `http://localhost:8001/sse` (SSE) or `http://localhost:8003/mcp` (HTTP). |
| `contextEngineUploader.mcpMemoryUrl` | MCP memory URL used when writing configs. For dev-remote, typical values are `http://localhost:8000/sse` (SSE) or `http://localhost:8002/mcp` (HTTP). |
| `contextEngineUploader.ctxIndexerUrl` | HTTP MCP indexer endpoint used by `ctx.py` in the Claude Code `/ctx` hook, typically `http://localhost:8003/mcp` for dev-remote. |
| `contextEngineUploader.claudeHookEnabled` | Enable writing a Claude Code `/ctx` hook in `.claude/settings.local.json`. |

## Commands and lifecycle

- `Context Engine Uploader: Start` — executes the initial `--force` followed by `--watch` using the configured settings.
- `Context Engine Uploader: Stop` — terminates any running upload client processes.
- `Context Engine Uploader: Restart` — stops current processes and re-runs the startup sequence.
- `Context Engine Uploader: Show Upload Service Logs` — opens a terminal and tails `docker compose logs -f upload_service`.
- `Context Engine Uploader: Prompt+ (Unicorn Mode)` — runs `scripts/ctx.py --unicorn` on your current selection and replaces it with the enhanced prompt (status bar button).

The extension logs all subprocess output to the **Context Engine Upload** output channel so you can confirm uploads without leaving VS Code. The watch process shuts down automatically when VS Code exits or when you run the Stop command.

## Troubleshooting

### Extension not syncing
1. Check **Context Engine Upload** output channel for errors
2. Verify `endpoint` setting points to running upload service
3. Ensure Python 3.8+ is available at configured `pythonPath`

### Prompt+ not working
1. Verify decoder is running: `curl http://localhost:8081/health`
2. Check `decoderUrl` setting matches your decoder (llama.cpp, Ollama, or GLM)
3. For GPU decoder: enable `useGpuDecoder` setting

### Connection refused
```bash
# Verify upload service is running
curl http://localhost:8004/health

# Check Docker logs
docker compose logs upload_service
```

### Remote server issues
1. Ensure port 9090 is accessible from your machine
2. Check firewall rules allow inbound connections
3. Verify server's `upload_service` container is running
