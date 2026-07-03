#!/usr/bin/env bash
# One-shot, idempotent launcher for a stable dev environment
# - Rebuilds images without cache
# - Brings up Qdrant and waits for readiness
# - Initializes payload indexes (non-fatal if already exist)
# - Reindexes the repo from scratch
# - Starts MCP servers (8000, 8001) and watcher
# - Verifies SSE endpoints and prints status

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

log() { echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

# Detect docker compose command
if docker compose version >/dev/null 2>&1; then
  DC=(docker compose)
elif command -v docker-compose >/dev/null 2>&1; then
  DC=(docker-compose)
else
  die "docker compose or docker-compose is required"
fi

need() { command -v "$1" >/dev/null 2>&1 || die "Missing dependency: $1"; }
need curl

# Ensure .env exists if example is present
if [[ ! -f .env && -f .env.example ]]; then
  log "Creating .env from .env.example"
  cp .env.example .env
fi

wait_for_url() {
  local url="$1"; shift
  local name="$1"; shift || true
  local tries=${1:-60}
  for i in $(seq 1 "$tries"); do
    if curl -sf --max-time 2 "$url" >/dev/null; then
      log "$name is ready ($url)"
      return 0
    fi
    sleep 1
  done
  return 1
}

log "Rebuilding images (no cache) ..."
"${DC[@]}" build --no-cache

log "Bringing up Qdrant ..."
"${DC[@]}" up -d qdrant

log "Waiting for Qdrant readiness ..."
wait_for_url "${QDRANT_URL:-http://localhost:6333}/readyz" "Qdrant" || die "Qdrant did not become ready in time"

log "Initializing payload indexes (non-fatal if exists) ..."
"${DC[@]}" run --rm init_payload || true

log "Reindexing repository (full recreate) ..."
"${DC[@]}" run --rm indexer --root /work --recreate

log "Starting MCP servers and watcher ..."
"${DC[@]}" up -d mcp mcp_indexer watcher

log "Waiting for MCP SSE endpoints ..."
wait_for_url "http://localhost:8000/sse" "MCP (search server 8000)" || die "MCP 8000 not ready"
wait_for_url "http://localhost:8001/sse" "MCP Indexer (8001)" || die "MCP Indexer 8001 not ready"

log "Current container status:"
"${DC[@]}" ps

log "All services are up. Endpoints:"
echo "  - Qdrant:      ${QDRANT_URL:-http://localhost:6333}"
echo "  - MCP (SSE):   http://localhost:8000/sse"
echo "  - Indexer SSE: http://localhost:8001/sse"

echo
log "Copy-paste MCP config for Cursor/Windsurf (settings.json):"
cat <<'JSON'
{
  "mcpServers": {
    "qdrant": { "type": "sse", "url": "http://localhost:8000/sse", "disabled": false },
    "qdrant-indexer": { "type": "sse", "url": "http://localhost:8001/sse", "disabled": false }
  }
}
JSON

echo
log "Copy-paste MCP config for Augment:"
cat <<'JSON'
{
  "mcpServers": {
    "qdrant": { "type": "sse", "url": "http://localhost:8000/sse", "disabled": false },
    "qdrant-indexer": { "type": "sse", "url": "http://localhost:8001/sse", "disabled": false }
  }
}
JSON

log "Tip: tail logs via: ${DC[*]} logs -f --tail=100"

