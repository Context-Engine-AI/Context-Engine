#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXT_DIR="$SCRIPT_DIR/../context-engine-uploader"
OUT_DIR="$SCRIPT_DIR/../out"
SRC_SCRIPT="$SCRIPT_DIR/../../scripts/standalone_upload_client.py"
CLIENT="standalone_upload_client.py"
STAGE_DIR="$OUT_DIR/extension-stage"
BUNDLE_DEPS="${1:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
HOOK_SRC="$SCRIPT_DIR/../../ctx-hook-simple.sh"
CTX_SRC="$SCRIPT_DIR/../../scripts/ctx.py"
ROUTER_SRC="$SCRIPT_DIR/../../scripts/mcp_router.py"
REFRAG_SRC="$SCRIPT_DIR/../../scripts/refrag_glm.py"
ENV_EXAMPLE_SRC="$SCRIPT_DIR/../../.env.example"
AUTH_SRC="$SCRIPT_DIR/../../scripts/upload_auth_utils.py"

cleanup() {
    rm -rf "$STAGE_DIR"
}
trap cleanup EXIT

echo "Building clean Context Engine Uploader extension..."

mkdir -p "$OUT_DIR"

# Ensure extension directory is clean
rm -f "$EXT_DIR/$CLIENT"

# Copy upload client to the distributable out directory
cp "$SRC_SCRIPT" "$OUT_DIR/$CLIENT"

# Prepare staging directory
rm -rf "$STAGE_DIR"
mkdir -p "$STAGE_DIR"
cp -a "$EXT_DIR/." "$STAGE_DIR/"

# Inject the upload client into the staged extension for packaging
cp "$OUT_DIR/$CLIENT" "$STAGE_DIR/$CLIENT"
chmod +x "$STAGE_DIR/$CLIENT"

# Bundle ctx hook script and ctx CLI into the staged extension for reference
if [[ -f "$HOOK_SRC" ]]; then
    cp "$HOOK_SRC" "$STAGE_DIR/ctx-hook-simple.sh"
    chmod +x "$STAGE_DIR/ctx-hook-simple.sh"
fi
if [[ -f "$CTX_SRC" ]]; then
    cp "$CTX_SRC" "$STAGE_DIR/ctx.py"
fi
if [[ -f "$ROUTER_SRC" ]]; then
    cp "$ROUTER_SRC" "$STAGE_DIR/mcp_router.py"
fi
if [[ -f "$REFRAG_SRC" ]]; then
    cp "$REFRAG_SRC" "$STAGE_DIR/refrag_glm.py"
fi

# Bundle auth helper used by standalone_upload_client.py
if [[ -f "$AUTH_SRC" ]]; then
    cp "$AUTH_SRC" "$STAGE_DIR/upload_auth_utils.py"
fi

if [[ -f "$ENV_EXAMPLE_SRC" ]]; then
    cp "$ENV_EXAMPLE_SRC" "$STAGE_DIR/env.example"
fi

# Optional: bundle Python deps into the staged extension when requested
if [[ "$BUNDLE_DEPS" == "--bundle-deps" ]]; then
    echo "Bundling Python dependencies into staged extension using $PYTHON_BIN..."
    # On macOS, urllib3 v2 + system LibreSSL emits NotOpenSSLWarning; pin <2 there.
    if [[ "$(uname -s)" == "Darwin" ]]; then
        echo "Detected macOS; pinning urllib3<2 to avoid LibreSSL/OpenSSL warning."
        "$PYTHON_BIN" -m pip install -t "$STAGE_DIR/python_libs" "urllib3<2" requests charset_normalizer "openai>=1.0" watchdog
    else
        "$PYTHON_BIN" -m pip install -t "$STAGE_DIR/python_libs" requests urllib3 charset_normalizer "openai>=1.0" watchdog
    fi
fi

pushd "$STAGE_DIR" >/dev/null
echo "Packaging extension..."
npx @vscode/vsce package --no-dependencies --out "$OUT_DIR"
popd >/dev/null

echo "Build complete! Check the /out directory for .vsix and .py files."
ls -la "$OUT_DIR"