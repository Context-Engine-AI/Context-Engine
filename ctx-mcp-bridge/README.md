# Context Engine MCP Bridge

`@context-engine-bridge/context-engine-mcp-bridge` provides the `ctxce` CLI, a
Model Context Protocol (MCP) bridge that speaks to the Context Engine indexer
and memory servers and exposes them as a single MCP server.

It is primarily used by the VS Code **Context Engine Uploader** extension,
available on the Marketplace:

- <https://marketplace.visualstudio.com/items?itemName=context-engine.context-engine-uploader>

The bridge can also be run standalone (e.g. from a terminal, or wired into
other MCP clients) as long as the Context Engine stack is running.

## Prerequisites

- Node.js **>= 18** (see `engines` in `package.json`).
- A running Context Engine stack (e.g. via `docker-compose.yml`) with:
  - MCP indexer HTTP endpoint (default: `http://localhost:8003/mcp`).
  - MCP memory HTTP endpoint (optional, default: `http://localhost:8002/mcp`).
- For optional auth:
  - The upload/auth services must be configured with `CTXCE_AUTH_ENABLED=1` and
    a reachable auth backend URL (e.g. `http://localhost:8004`).

## Installation

You can install the package globally, or run it via `npx`.

### Global install

```bash
npm install -g @context-engine-bridge/context-engine-mcp-bridge
```

This installs the `ctxce` (and `ctxce-bridge`) CLI in your PATH.

### Using npx (no global install)

```bash
npx @context-engine-bridge/context-engine-mcp-bridge ctxce --help
```

The examples below assume `ctxce` is available on your PATH; if you use `npx`,
just prefix commands with `npx @context-engine-bridge/context-engine-mcp-bridge`.

## CLI overview

The main entrypoint is:

```bash
ctxce <command> [...args]
```

Supported commands (from `src/cli.js`):

- `ctxce mcp-serve`        – stdio MCP bridge (for stdio-based MCP clients).
- `ctxce mcp-http-serve`   – HTTP MCP bridge (for HTTP-based MCP clients).
- `ctxce auth <subcmd>`    – auth helper commands (`login`, `status`, `logout`).

### Environment variables

These environment variables are respected by the bridge:

- `CTXCE_INDEXER_URL` – MCP indexer URL (default: `http://localhost:8003/mcp`).
- `CTXCE_MEMORY_URL`  – MCP memory URL, or empty/omitted to disable memory
  (default: `http://localhost:8002/mcp`).
- `CTXCE_HTTP_PORT`   – port for `mcp-http-serve` (default: `30810`).

For auth (optional, shared with the upload/auth backend):

- `CTXCE_AUTH_ENABLED`       – whether auth is enabled in the backend.
- `CTXCE_AUTH_BACKEND_URL`   – auth backend URL (e.g. `http://localhost:8004`).
- `CTXCE_AUTH_TOKEN`         – dev/shared token for `ctxce auth login`.
- `CTXCE_AUTH_SESSION_TTL_SECONDS` – session TTL / sliding expiry (seconds).

The CLI also stores auth sessions in `~/.ctxce/auth.json`, keyed by backend URL.

## Running the MCP bridge (stdio)

The stdio bridge is suitable for MCP clients that speak stdio directly (for
example, certain editors or tools that expect an MCP server on stdin/stdout).

```bash
ctxce mcp-serve \
  --workspace /path/to/your/workspace \
  --indexer-url http://localhost:8003/mcp \
  --memory-url http://localhost:8002/mcp
```

Flags:

- `--workspace` / `--path` – workspace root (default: current working directory).
- `--indexer-url`          – override indexer URL (default: `CTXCE_INDEXER_URL` or
  `http://localhost:8003/mcp`).
- `--memory-url`           – override memory URL (default: `CTXCE_MEMORY_URL` or
  disabled when empty).

## Running the MCP bridge (HTTP)

The HTTP bridge exposes the MCP server via an HTTP endpoint (default
`http://127.0.0.1:30810/mcp`) and is what the VS Code extension uses in its
`http` transport mode.

```bash
ctxce mcp-http-serve \
  --workspace /path/to/your/workspace \
  --indexer-url http://localhost:8003/mcp \
  --memory-url http://localhost:8002/mcp \
  --port 30810
```

Flags:

- `--workspace` / `--path` – workspace root (default: current working directory).
- `--indexer-url`          – MCP indexer URL.
- `--memory-url`           – MCP memory URL (or omit/empty to disable memory).
- `--port`                 – HTTP port for the bridge (default: `CTXCE_HTTP_PORT`
  or `30810`).

Once running, you can point an MCP client at:

```text
http://127.0.0.1:<port>/mcp
```

## Auth helper commands (`ctxce auth ...`)

These commands are used both by the VS Code extension and standalone flows to
log in and manage auth sessions for the backend.

### Login (token)

```bash
ctxce auth login \
  --backend-url http://localhost:8004 \
  --token $CTXCE_AUTH_SHARED_TOKEN
```

This hits the backend `/auth/login` endpoint and stores a session entry in
`~/.ctxce/auth.json` under the given backend URL.

### Login (username/password)

```bash
ctxce auth login \
  --backend-url http://localhost:8004 \
  --username your-user \
  --password your-password
```

This calls `/auth/login/password` and persists the returned session the same
way as the token flow.

### Status

Human-readable status:

```bash
ctxce auth status --backend-url http://localhost:8004
```

Machine-readable status (used by the VS Code extension):

```bash
ctxce auth status --backend-url http://localhost:8004 --json
```

The `--json` variant prints a single JSON object to stdout, for example:

```json
{
  "backendUrl": "http://localhost:8004",
  "state": "ok",           // "ok" | "missing" | "expired" | "missing_backend"
  "sessionId": "...",
  "userId": "user-123",
  "expiresAt": 0           // 0 or a Unix timestamp
}
```

Exit codes:

- `0` – `state: "ok"` (valid session present).
- `1` – `state: "missing"` or `"missing_backend"`.
- `2` – `state: "expired"`.

### Logout

```bash
ctxce auth logout --backend-url http://localhost:8004
```

Removes the stored auth entry for the given backend URL from
`~/.ctxce/auth.json`.

## Relationship to the VS Code extension

The VS Code **Context Engine Uploader** extension is the recommended way to use
this bridge for day-to-day development. It:

- Launches the standalone upload client to push code into the remote stack.
- Starts/stops the MCP HTTP bridge (`ctxce mcp-http-serve`) for the active
  workspace when `autoStartMcpBridge` is enabled.
- Uses `ctxce auth status --json` and `ctxce auth login` under the hood to
  manage user sessions via UI prompts.

This package README is aimed at advanced users who want to:

- Run the MCP bridge outside of VS Code.
- Integrate the Context Engine MCP servers with other MCP-compatible clients.

You can safely mix both approaches: the extension and the standalone bridge
share the same auth/session storage in `~/.ctxce/auth.json`.
