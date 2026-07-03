#!/usr/bin/env bash
set -euo pipefail

# Simple helper to login (if needed) and publish the package.
# Usage:
#   ./publish.sh            # publishes current version
#   ./publish.sh 0.0.2      # bumps version to 0.0.2 then publishes

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PACKAGE_NAME="@context-engine-bridge/context-engine-mcp-bridge"

echo "[publish] Verifying npm authentication..."
if ! npm whoami >/dev/null 2>&1; then
  echo "[publish] Not logged in; running npm login"
  npm login
else
  echo "[publish] Already authenticated as $(npm whoami)"
fi

if [[ $# -gt 0 ]]; then
  VERSION="$1"
  echo "[publish] Bumping version to $VERSION"
  npm version "$VERSION" --no-git-tag-version
fi

echo "[publish] Packing $PACKAGE_NAME for verification..."
npm pack >/dev/null

echo "[publish] Publishing $PACKAGE_NAME..."
npm publish --access public

echo "[publish] Done!"
