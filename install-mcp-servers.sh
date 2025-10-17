#!/usr/bin/env bash
set -euo pipefail

# install-mcp-servers.sh
# Adds a set of essential MCP servers to Claude Code via the `claude` CLI.
# Usage:
#   chmod +x install-mcp-servers.sh
#   ./install-mcp-servers.sh
#
# Optional environment variables (set before running):
#   DIRS                          Space-separated directories for filesystem server
#                                 Default: "$HOME/Documents $HOME/Desktop $HOME/Downloads $HOME/Projects"
#   BRAVE_API_KEY                 Brave Search API key (enables brave-search server)
#   FIRECRAWL_API_KEY             Firecrawl API key (enables firecrawl server)
#   FIRECRAWL_RETRY_MAX_ATTEMPTS  Optional
#   FIRECRAWL_RETRY_INITIAL_DELAY Optional (ms)
#   FIRECRAWL_RETRY_MAX_DELAY     Optional (ms)
#   FIRECRAWL_CREDIT_WARNING_THRESHOLD Optional
#   BROWSER_TOOLS_URL             If set, will auto-add browser-tools over SSE

require() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Error: required command '$1' not found in PATH" >&2
    exit 1
  fi
}

require claude
require npx

echo "Using claude CLI: $(command -v claude)"
echo "Using npx: $(command -v npx)"

DIRS_DEFAULT="$HOME/Documents $HOME/Desktop $HOME/Downloads $HOME/Projects"
DIRS=${DIRS:-$DIRS_DEFAULT}

echo "\n==> Adding MCP server: sequential-thinking"
claude mcp add sequential-thinking -s user -- npx -y @modelcontextprotocol/server-sequential-thinking

echo "\n==> Adding MCP server: filesystem ($DIRS)"
claude mcp add filesystem -s user -- npx -y @modelcontextprotocol/server-filesystem $DIRS

echo "\n==> Adding MCP server: puppeteer"
claude mcp add puppeteer -s user -- npx -y @modelcontextprotocol/server-puppeteer

echo "\n==> Adding MCP server: fetch"
claude mcp add fetch -s user -- npx -y @kazuph/mcp-fetch

if [[ -n "${BRAVE_API_KEY:-}" ]]; then
  echo "\n==> Adding MCP server: brave-search"
  claude mcp add brave-search -s user -- env BRAVE_API_KEY="${BRAVE_API_KEY}" npx -y @modelcontextprotocol/server-brave-search
else
  echo "\n==> Skipping brave-search (set BRAVE_API_KEY to enable)"
fi

if [[ -n "${FIRECRAWL_API_KEY:-}" ]]; then
  echo "\n==> Adding MCP server: firecrawl"
  cmd=(claude mcp add firecrawl -s user -- env FIRECRAWL_API_KEY="${FIRECRAWL_API_KEY}")
  # Append optional Firecrawl env vars if present
  for v in FIRECRAWL_RETRY_MAX_ATTEMPTS FIRECRAWL_RETRY_INITIAL_DELAY FIRECRAWL_RETRY_MAX_DELAY FIRECRAWL_CREDIT_WARNING_THRESHOLD; do
    if [[ -n "${!v:-}" ]]; then cmd+=( "$v=${!v}" ); fi
  done
  cmd+=( npx -y firecrawl-mcp )
  "${cmd[@]}"
else
  echo "\n==> Skipping firecrawl (set FIRECRAWL_API_KEY to enable)"
fi

# Browser Tools (Chrome DevTools Integration)
# Step 1 (separate terminal): npx @agentdeskai/browser-tools-server@1.2.1
# Step 2 (this terminal): add via SSE using the URL the server prints (e.g., http://localhost:<PORT>/sse)
if [[ -n "${BROWSER_TOOLS_URL:-}" ]]; then
  echo "\n==> Adding MCP server: browser-tools (SSE)"
  claude mcp add browser-tools -s user --transport sse --url "${BROWSER_TOOLS_URL}"
else
  cat <<'EOF'

==> Browser Tools (manual step)
1) In a separate terminal, start the middleware server and keep it running:
   npx @agentdeskai/browser-tools-server@1.2.1
2) Note the SSE URL the server prints (e.g., http://localhost:<PORT>/sse), then run:
   export BROWSER_TOOLS_URL=http://localhost:<PORT>/sse
   claude mcp add browser-tools -s user --transport sse --url "$BROWSER_TOOLS_URL"
EOF
fi

echo "\nAll requested MCP servers processed."
