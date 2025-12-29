Context Engine Uploader
=======================

Install
-------
- Install from the VS Code Marketplace (search for "Context Engine Uploader" or publisher `context-engine`). You can also install it directly from the Extensions view in VS Code.

Features
--------
- Runs a force sync (`Index Codebase`) followed by watch mode to keep a remote Context Engine instance in sync with your workspace.
- Auto-detects the first workspace folder as the default target path, storing it in workspace settings so the extension is portable.
- Provides commands and a status-bar button:
  - `Context Engine Uploader: Index Codebase` – force sync + watch with spinner feedback.
  - `Context Engine Uploader: Start/Stop/Restart` for manual lifecycle control.
- Streams detailed logs into the `Context Engine Upload` output channel for visibility into both force sync and watch phases.
- Status bar shows current state (indexing spinner, purple watching state) so you always know if uploads are active.

Configuration
-------------
- `Run On Startup` auto-triggers force sync + watch after VS Code finishes loading.
- `Python Path`, `Endpoint`, `Extra Force Args`, `Extra Watch Args`, and `Interval Seconds` can be tuned via standard VS Code settings.
- `Target Path` is auto-filled from the workspace but can be overridden if you need to upload a different folder.
- **Python dependencies:** the extension runs the standalone upload client via your configured `pythonPath`. Ensure the interpreter has `requests`, `urllib3`, `charset_normalizer`, and `watchdog` installed. Run `python3 -m pip install requests urllib3 charset_normalizer watchdog` (or replace `python3` with your configured path) before starting the uploader.
- **Path mapping:** `Host Root` + `Container Root` control how local paths are rewritten before reaching the remote service. By default the host root mirrors your `Target Path` and the container root is `/work`, which keeps Windows paths working without extra config.
- **Prompt+ decoder:** set `Context Engine Uploader: Decoder Url` (default `http://localhost:8081`, auto-appends `/completion`) to point at your local llama.cpp decoder. For Ollama, set it to `http://localhost:11434/api/chat`. Turn on `Use Gpu Decoder` to set `USE_GPU_DECODER=1` so ctx.py prefers the GPU llama.cpp sidecar. Prompt+ automatically runs the bundled `scripts/ctx.py` when an embedded copy is available, falling back to the workspace version if not.
- **Claude/Windsurf MCP config:**
  - `MCP Indexer Url` and `MCP Memory Url` control the URLs written into the project-local `.mcp.json` (Claude) and Windsurf `mcp_config.json` when you run the `Write MCP Config` command. These URLs are used **literally** (e.g. `http://localhost:8001/sse` or `http://localhost:8003/mcp`).
  - `MCP Server Mode` (`contextEngineUploader.mcpServerMode`) controls *what* servers are written:
    - `bridge`: write a single `context-engine` server that talks to the `ctxce` MCP bridge.
    - `direct`: write two servers, `qdrant-indexer` and `memory`, that talk directly to the configured URLs.
  - `MCP Transport Mode` (`contextEngineUploader.mcpTransportMode`) controls *how* those servers talk:
    - `sse-remote` (default): use stdio MCP processes behind an SSE tunnel (`bridge-stdio` / `direct-sse`).
    - `http`: use HTTP MCP endpoints directly (`bridge-http` / `direct-http`).
  - Common combinations:
    - `bridge` + `sse-remote` → **bridge-stdio**: write a single `context-engine` stdio server that runs `ctxce mcp-serve` behind an SSE tunnel.
    - `bridge` + `http` → **bridge-http**: write a single `context-engine` HTTP server that points at the local `ctxce mcp-http-serve` URL (e.g. `http://127.0.0.1:30810/mcp`).
    - `direct` + `http` → **direct-http**: write separate HTTP servers for the indexer and memory MCP backends.
- **MCP config on startup:**
  - `contextEngineUploader.autoWriteMcpConfigOnStartup` (default `false`) controls whether the extension automatically runs the same logic as `Write MCP Config` on activation. When enabled, it refreshes `.mcp.json`, Windsurf `mcp_config.json`, and the Claude hook (`.claude/settings.local.json`) to match your current settings and the installed extension version. If `scaffoldCtxConfig` is also `true`, this startup path will additionally scaffold/update `ctx_config.json` and `.env` as described below.
- **CTX + GLM settings:**
  - `contextEngineUploader.ctxIndexerUrl` is copied into `.env` (as `MCP_INDEXER_URL`) so the embedded `ctx.py` knows which MCP indexer to call when enhancing prompts.
  - `contextEngineUploader.glmApiKey`, `glmApiBase`, and `glmModel` are used when scaffolding `ctx_config.json`/`.env` to pre-fill GLM decoder options. Existing non-placeholder values are preserved, so you can override them in the files at any time.
- **Git history upload settings:**
  - `contextEngineUploader.gitMaxCommits` controls `REMOTE_UPLOAD_GIT_MAX_COMMITS`, bounding how many commits the upload client includes per bundle (set to 0 to disable git history).
  - `contextEngineUploader.gitSince` controls `REMOTE_UPLOAD_GIT_SINCE`, letting you constrain the git log window (e.g. `2 years ago` or `2023-01-01`).
- **Context scaffolding:**
  - `contextEngineUploader.scaffoldCtxConfig` (default `true`) controls whether the extension keeps a minimal `ctx_config.json` + `.env` in sync with your workspace. When enabled, running `Write MCP Config` or `Write CTX Config` will reuse the workspace’s existing files (if present) and only backfill placeholder or missing values from the bundled `env.example` plus the inferred collection name. Existing custom values are preserved.
  - The scaffolder also enforces CTX defaults (e.g., `MULTI_REPO_MODE=1`, `REFRAG_RUNTIME=glm`, `REFRAG_DECODER=1`) so the embedded `ctx.py` is ready for remote uploads, regardless of the “Use GLM Decoder” toggle.
  - `contextEngineUploader.surfaceQdrantCollectionHint` gates whether the Claude hook adds a hint line with the Qdrant collection ID when ctx is enhancing prompts. This setting is also respected when the extension writes `.claude/settings.local.json`.

MCP bridge (ctx-mcp-bridge) & MCP config lifecycle
---------------------------------------------------
- The MCP bridge (`@context-engine-bridge/context-engine-mcp-bridge`, CLI `ctxce`) is a small local MCP server that fans out to two upstream MCP services: the Qdrant indexer and the memory/search backend. The VS Code extension can drive it in two ways:
  - **Bridge stdio (`bridge-stdio`)** – a stdio MCP server (`ctxce mcp-serve`) wrapped behind an SSE tunnel.
  - **Bridge HTTP (`bridge-http`)** – an HTTP MCP server (`ctxce mcp-http-serve`) listening on `http://127.0.0.1:<port>/mcp`.
- Why use the bridge instead of two direct MCP entries?
  - **Single server entry:** IDEs only need to register one MCP server (`context-engine`) instead of juggling separate `qdrant-indexer` and `memory` entries, avoiding coordination mistakes.
  - **Shared session defaults:** the bridge loads `ctx_config.json` and injects collection name, repo metadata, and any other ctx defaults so every IDE window talks to the right collection without hand-editing `.mcp.json`.
  - **Per-user credential isolation:** each IDE maintains its own MCP session while the bridge multiplexes upstream calls. When you enable backend auth (via `CTXCE_AUTH_ENABLED` and `ctxce auth ...` sessions), uploads and MCP calls are gated by per-user sessions, so multiple IDEs can share the same stack while still having isolated access control and preferences. See **Optional auth with the MCP bridge (PoC)** below for details.
  - **Flexible transport:** stdio mode works everywhere (even when HTTP ports aren’t reachable), while HTTP mode keeps Claude/Windsurf happy when they want direct URLs; the extension automatically writes the right flavor.
  - **Centralized logging & health:** when the bridge process runs once per workspace you get a single stream of logs (`Context Engine Upload` output) and a single port to probe for health checks instead of multiple MCP child processes per IDE.
- When you run **`Write MCP Config`**, the extension:
  - Writes `.mcp.json` in the workspace for Claude Code.
  - Optionally writes Windsurf’s `mcp_config.json` (when `mcpWindsurfEnabled=true`).
  - Optionally scaffolds `ctx_config.json` + `.env` (when `scaffoldCtxConfig=true`).
- The effective wiring mode is determined by the two MCP settings:
  - `mcpServerMode = bridge`, `mcpTransportMode = sse-remote` → **bridge-stdio**.
  - `mcpServerMode = bridge`, `mcpTransportMode = http` → **bridge-http**.
  - `mcpServerMode = direct`, `mcpTransportMode = sse-remote` → **direct-sse** (two stdio `mcp-remote` servers).
  - `mcpServerMode = direct`, `mcpTransportMode = http` → **direct-http** (two HTTP servers, no bridge).
- In **bridge-stdio**, the configs run the `ctxce mcp-serve` CLI via `npx` (for example,
  `npx @context-engine-bridge/context-engine-mcp-bridge ctxce mcp-serve`), passing the
  workspace path (auto-detected from the uploader target path) plus `--indexer-url`
  and `--memory-url` derived from the MCP settings.
- In **bridge-http**, the extension can also **manage the bridge process**:
  - `autoStartMcpBridge=true` and `mcpServerMode='bridge'` with `mcpTransportMode='http'` → the extension starts `ctxce mcp-http-serve` in the background for the active workspace using `mcpBridgePort`.
  - The resulting HTTP URL (`http://127.0.0.1:<mcpBridgePort>/mcp`) is written into `.mcp.json` and Windsurf’s `mcp_config.json` as the `context-engine` server URL.
  - In **stdio or direct modes**, the HTTP bridge is **not** auto-started; only the explicit `Start MCP HTTP Bridge` command will launch it.
- Bridge settings are **workspace-scoped**, so different workspaces can choose different modes and ports (e.g., one workspace using stdio bridge, another using HTTP bridge on a different port).

Optional auth with the MCP bridge (PoC)
--------------------------------------

Auth is **off by default** and fully opt-in. When enabled, the MCP indexer and
memory servers expect a valid `session` id (issued by the backend) on protected
tools. The bridge CLI (`ctxce auth ...`) is the primary way to obtain and cache
that session.

High-level steps:

- Enable auth on the remote stack (e.g. dev-remote compose):
  - Set `CTXCE_AUTH_ENABLED=1` in the upload/indexer environment.
  - Optionally set `CTXCE_AUTH_SHARED_TOKEN` for token-based login.
  - Optional: set `CTXCE_AUTH_ADMIN_TOKEN` for creating additional users via `/auth/users`.
  - Optional (dev-only): set `CTXCE_AUTH_ALLOW_OPEN_TOKEN_LOGIN=1` **only** if you want `/auth/login`
    to issue sessions even when `CTXCE_AUTH_SHARED_TOKEN` is unset. By default (`0`/unset),
    token-based login is disabled when no shared token is configured.
- Point the bridge at the auth backend:
  - In your local shell (where you run the `ctxce` CLI via `npx`), you can either:
    - Explicitly set `CTXCE_AUTH_BACKEND_URL` to the upload service URL (e.g. `http://localhost:8004`), or
    - Let the CLI discover the backend URL in this order when you run `ctxce auth login`:
      `--backend-url` / `--auth-url` → `CTXCE_AUTH_BACKEND_URL` → any stored auth entry →
      the uploader's `upload_endpoint` (`CTXCE_UPLOAD_ENDPOINT` / `UPLOAD_ENDPOINT`) →
      `http://localhost:8004`.

Token-based login:

```bash
export CTXCE_AUTH_BACKEND_URL=http://localhost:8004   # optional when using this extension
export CTXCE_AUTH_TOKEN=change-me-dev-token   # must match CTXCE_AUTH_SHARED_TOKEN in the stack

# Obtain a session and cache it under ~/.ctxce/auth.json
npx @context-engine-bridge/context-engine-mcp-bridge ctxce auth login

# Check status (optional)
npx @context-engine-bridge/context-engine-mcp-bridge ctxce auth status
```

Username/password login (when you have real users):

- First, create the initial user (once) via `/auth/users` while the auth DB is
  empty (no admin token required). This is typically done with a small script
  or curl call against the upload service, for example:

  ```bash
  curl -X POST http://localhost:8004/auth/users \
    -H "Content-Type: application/json" \
    -d '{"username":"you@example.com","password":"your-password"}'
  ```

- Then login via the bridge:

```bash
npx @context-engine-bridge/context-engine-mcp-bridge ctxce auth login \
  --username you@example.com \
  --password 'your-password'
```

In both modes, the bridge stores the returned `session_id` keyed by
`CTXCE_AUTH_BACKEND_URL` and automatically injects it into all MCP tool calls
as the `session` field. Once you have at least one entry in `~/.ctxce/auth.json`,
the MCP bridge used by the extension will discover and reuse that session
automatically; MCP configs do not need to set `CTXCE_AUTH_BACKEND_URL` in their
`env` blocks. If `CTXCE_AUTH_ENABLED` is off on the backend, these auth settings
are ignored and the bridge behaves exactly as before.

Session lifetime:

- By default, issued sessions do **not** expire (`CTXCE_AUTH_SESSION_TTL_SECONDS`
  defaults to `0`).
- Operators who want expiry can set `CTXCE_AUTH_SESSION_TTL_SECONDS` (in seconds)
  on the backend services. Values `> 0` enable a sliding window: active sessions
  are refreshed when validated; values `<= 0` disable expiry.

Workspace-level ctx integration
-------------------------------
- The VSIX bundles an `env.example` template plus the ctx hook/CLI so you can dogfood the workflow without copying files manually.
- When scaffolding is enabled (see above), running the `Context Engine Uploader: Write CTX Config (ctx_config.json/.env)` command will:
  - Infer the collection name from the standalone upload client (`--show-mapping`).
  - Create or update `ctx_config.json` with that collection and sensible defaults (GLM runtime, `default_mode`, `require_context`, etc.).
  - Create or update `.env` from the bundled template, ensuring CTX-critical values such as `MULTI_REPO_MODE=1`, `REFRAG_RUNTIME=glm`, and `REFRAG_DECODER=1` are set. Non-placeholder values (e.g., a real `GLM_API_KEY`) are left alone.
- You still own the files: if you need a custom value, edit `.env` or `ctx_config.json` directly. The scaffolder only touches keys that are missing, empty, or obviously placeholders.
- The Claude hook + ctx prompt enhancement is currently wired for Linux/dev-remote environments only. On other platforms, MCP config and uploading still work, but the automatic prompt rewrite hook is disabled.

Commands
--------
- Command Palette → “Context Engine Uploader” exposes Start/Stop/Restart/Index Codebase and Prompt+ (unicorn) rewrite commands.
- Status-bar button (`Index Codebase`) mirrors Start/Stop/Restart/Index status, while the `Prompt+` status button runs the ctx rewrite command on the current selection.
- `Context Engine Uploader: Write MCP Config (.mcp.json)` writes or updates a project-local `.mcp.json` (plus Windsurf `mcp_config.json` when enabled) using the currently selected bridge/direct + transport modes. If bridge-http is required and not yet running, the extension starts `ctxce mcp-http-serve` before writing configs.
- `Context Engine Uploader: Write CTX Config (ctx_config.json/.env)` scaffolds the ctx config + env files as described above. This command runs automatically after `Write MCP Config` if scaffolding is enabled, but it is also exposed in the Command Palette for manual use.
- `Context Engine Uploader: Upload Git History (force sync bundle)` triggers a one-off force sync using the configured git history settings, producing a bundle that includes a `metadata/git_history.json` manifest for remote lineage ingestion.
- `Context Engine Uploader: Start MCP HTTP Bridge` launches `ctxce mcp-http-serve` using the workspace’s resolved target path, MCP URLs, and configured `mcpBridgePort`. Use this when you want to run the HTTP bridge manually (e.g., testing unpublished builds or sharing a port across IDEs).
- `Context Engine Uploader: Stop MCP HTTP Bridge` gracefully terminates a running HTTP bridge process.

Logs
----
Open `View → Output → Context Engine Upload` to see the remote uploader’s stdout/stderr, including any errors from the Python client.
