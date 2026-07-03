#!/bin/bash

# Simplified Claude Code UserPromptSubmit hook for ctx.py
# Takes JSON input from Claude Code and outputs enhanced prompt

# Read JSON input from stdin
INPUT=$(cat)

# Extract the prompt text from Claude's JSON payload
if command -v jq >/dev/null 2>&1; then
	USER_MESSAGE=$(echo "$INPUT" | jq -r '.prompt')
	USER_CWD=$(echo "$INPUT" | jq -r '.cwd // empty')
else
	# Fallback: treat entire input as the prompt text
	USER_MESSAGE="$INPUT"
fi

# Skip if empty message
if [ -z "$USER_MESSAGE" ] || [ "$USER_MESSAGE" = "null" ]; then
	echo "$INPUT"
	exit 0
fi

# Set working directory to where the hook script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Determine workspace directory:
# - If CTX_WORKSPACE_DIR is already set, honor it.
# - If running from an embedded extension under ~/.windsurf-server/extensions,
#   default to the caller's CWD (Claude/VS Code workspace root).
# - Otherwise (repo-local hook), default to the script directory so it works
#   even when Claude runs from a parent folder.
if [ -n "${CTX_WORKSPACE_DIR:-}" ]; then
	WORKSPACE_DIR="$CTX_WORKSPACE_DIR"
elif [[ "$SCRIPT_DIR" == */.windsurf-server/extensions/* ]]; then
	WORKSPACE_DIR="$PWD"
else
	WORKSPACE_DIR="$SCRIPT_DIR"
fi
export CTX_WORKSPACE_DIR="$WORKSPACE_DIR"

# If the workspace root does not contain ctx_config.json, but exactly one
# direct child directory does, treat that child directory as the effective
# workspace. This supports multi-repo workspaces where the ctx-enabled repo
# (with ctx_config.json and .env) lives one level below the VS Code root.
if [ ! -f "$WORKSPACE_DIR/ctx_config.json" ]; then
	FOUND_SUBDIR=""
	for candidate in "$WORKSPACE_DIR"/*; do
		if [ -d "$candidate" ] && [ -f "$candidate/ctx_config.json" ]; then
			if [ -z "$FOUND_SUBDIR" ]; then
				FOUND_SUBDIR="$candidate"
			else
				# More than one candidate; ambiguous, keep original WORKSPACE_DIR
				FOUND_SUBDIR=""
				break
			fi
		fi
	done
	if [ -n "$FOUND_SUBDIR" ]; then
		WORKSPACE_DIR="$FOUND_SUBDIR"
		export CTX_WORKSPACE_DIR="$WORKSPACE_DIR"
	fi
fi

# Prefer workspace-level ctx_config.json, fall back to one next to the script
if [ -f "$WORKSPACE_DIR/ctx_config.json" ]; then
	CONFIG_FILE="$WORKSPACE_DIR/ctx_config.json"
elif [ -f "$SCRIPT_DIR/ctx_config.json" ]; then
	CONFIG_FILE="$SCRIPT_DIR/ctx_config.json"
else
	CONFIG_FILE=""
fi

# Optional: enable file logging when CTX_HOOK_LOG=1 or a .ctx_hook_log marker
# file exists in the workspace. When disabled, no log file is written.
if [ "${CTX_HOOK_LOG:-0}" = "1" ] || [ -f "$WORKSPACE_DIR/.ctx_hook_log" ]; then
	LOG_FILE="$WORKSPACE_DIR/ctx-hook.log"
	LOG_ENABLED=1
else
	LOG_ENABLED=0
fi

cd "$SCRIPT_DIR"

# Optional: enable extra debug information in the JSON payload
# when CTX_HOOK_DEBUG=1 is set in the environment, or when a
# .ctx_hook_debug marker file exists in the workspace.
CTX_HOOK_DEBUG="${CTX_HOOK_DEBUG:-}"
if [ -z "$CTX_HOOK_DEBUG" ] && [ -f "$WORKSPACE_DIR/.ctx_hook_debug" ]; then
	CTX_HOOK_DEBUG="1"
fi

# Log the incoming payload when logging is enabled
if [ "$LOG_ENABLED" = "1" ]; then
	{
		echo "[$(date -Iseconds)] HOOK INVOKED"
		echo "PWD   = $PWD"
		echo "WORKSPACE_DIR = $WORKSPACE_DIR"
		echo "INPUT = <<EOF"
		echo "$INPUT"
		echo "EOF"
		echo
	} >> "$LOG_FILE"
fi

# Read all settings from ctx_config.json
if [ -n "$CONFIG_FILE" ] && [ -f "$CONFIG_FILE" ]; then
	CTX_COLLECTION=$(grep -o '"default_collection"[[:space:]]*:[[:space:]]*"[^"]*"' "$CONFIG_FILE" | sed 's/.*"default_collection"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/' )
	REFRAG_RUNTIME=$(grep -o '"refrag_runtime"[[:space:]]*:[[:space:]]*"[^"]*"' "$CONFIG_FILE" | sed 's/.*"refrag_runtime"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/' || echo "glm")
	GLM_API_KEY=$(grep -o '"glm_api_key"[[:space:]]*:[[:space:]]*"[^"]*"' "$CONFIG_FILE" | sed 's/.*"glm_api_key"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/' )
	GLM_API_BASE=$(grep -o '"glm_api_base"[[:space:]]*:[[:space:]]*"[^"]*"' "$CONFIG_FILE" | sed 's/.*"glm_api_base"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/')
	GLM_MODEL=$(grep -o '"glm_model"[[:space:]]*:[[:space:]]*"[^\"]*"' "$CONFIG_FILE" | sed 's/.*"glm_model"[[:space:]]*:[[:space:]]*"\([^\"]*\)".*/\1/' || echo "glm-4.6")
	CTX_DEFAULT_MODE=$(grep -o '"default_mode"[[:space:]]*:[[:space:]]*"[^\"]*"' "$CONFIG_FILE" | sed 's/.*"default_mode"[[:space:]]*:[[:space:]]*"\([^\"]*\)".*/\1/')
	CTX_REQUIRE_CONTEXT=$(grep -o '"require_context"[[:space:]]*:[[:space:]]*\(true\|false\)' "$CONFIG_FILE" | sed 's/.*"require_context"[[:space:]]*:[[:space:]]*\(true\|false\).*/\1/')
	CTX_RELEVANCE_GATE=$(grep -o '"relevance_gate_enabled"[[:space:]]*:[[:space:]]*\(true\|false\)' "$CONFIG_FILE" | sed 's/.*"relevance_gate_enabled"[[:space:]]*:[[:space:]]*\(true\|false\).*/\1/')
	CTX_MIN_RELEVANCE=$(grep -o '"min_relevance"[[:space:]]*:[[:space:]]*[0-9.][0-9.]*' "$CONFIG_FILE" | sed 's/.*"min_relevance"[[:space:]]*:[[:space:]]*\([0-9.][0-9.]*\).*/\1/')
	CTX_REWRITE_MAX_TOKENS=$(grep -o '"rewrite_max_tokens"[[:space:]]*:[[:space:]]*[0-9][0-9]*' "$CONFIG_FILE" | sed 's/.*"rewrite_max_tokens"[[:space:]]*:[[:space:]]*\([0-9][0-9]*\).*/\1/')
	CTX_SURFACE_COLLECTION_CFG=$(grep -o '"surface_qdrant_collection_hint"[[:space:]]*:[[:space:]]*\(true\|false\)' "$CONFIG_FILE" | sed 's/.*"surface_qdrant_collection_hint"[[:space:]]*:[[:space:]]*\(true\|false\).*/\1/')
fi

# Set defaults if not found in config
CTX_COLLECTION=${CTX_COLLECTION:-"codebase"}
REFRAG_RUNTIME=${REFRAG_RUNTIME:-"glm"}
GLM_API_KEY=${GLM_API_KEY:-}
GLM_API_BASE=${GLM_API_BASE:-}
GLM_MODEL=${GLM_MODEL:-"glm-4.6"}
CTX_DEFAULT_MODE=${CTX_DEFAULT_MODE:-"default"}
CTX_REQUIRE_CONTEXT=${CTX_REQUIRE_CONTEXT:-true}
CTX_RELEVANCE_GATE=${CTX_RELEVANCE_GATE:-false}
CTX_MIN_RELEVANCE=${CTX_MIN_RELEVANCE:-0.1}
CTX_REWRITE_MAX_TOKENS=${CTX_REWRITE_MAX_TOKENS:-320}

# Normalize surface_qdrant_collection_hint from config (true/false) into 1/0
CFG_HINT=""
if [ -n "$CTX_SURFACE_COLLECTION_CFG" ]; then
	if [ "$CTX_SURFACE_COLLECTION_CFG" = "true" ]; then
		CFG_HINT="1"
	elif [ "$CTX_SURFACE_COLLECTION_CFG" = "false" ]; then
		CFG_HINT="0"
	fi
fi

# Precedence: explicit env override > ctx_config flag > auto-on when collection known
if [ -n "${CTX_SURFACE_COLLECTION_HINT:-}" ]; then
	:
elif [ -n "$CFG_HINT" ]; then
	CTX_SURFACE_COLLECTION_HINT="$CFG_HINT"
elif [ -n "$CTX_COLLECTION" ]; then
	CTX_SURFACE_COLLECTION_HINT="1"
fi

# Export GLM/context environment variables from config
export REFRAG_RUNTIME GLM_API_KEY GLM_API_BASE GLM_MODEL CTX_REQUIRE_CONTEXT CTX_RELEVANCE_GATE CTX_MIN_RELEVANCE CTX_REWRITE_MAX_TOKENS

# Easy bypass patterns - any of these will skip ctx enhancement
BYPASS_REASON=""
if [[ "$USER_MESSAGE" =~ ^(noctx|raw|bypass|skip|no-enhance): ]]; then
	BYPASS_REASON="prefix_tag"
elif [[ "$USER_MESSAGE" =~ ^\\ ]]; then
	BYPASS_REASON="leading_backslash"
elif [[ "$USER_MESSAGE" =~ ^\< ]]; then
	BYPASS_REASON="leading_angle_bracket"
elif [[ "$USER_MESSAGE" =~ ^(/help|/clear|/exit|/quit) ]]; then
	BYPASS_REASON="slash_command"
elif [[ "$USER_MESSAGE" =~ ^\?\s*$ ]]; then
	BYPASS_REASON="short_question_mark"
elif [ ${#USER_MESSAGE} -lt 12 ]; then
	BYPASS_REASON="too_short"
fi

if [ -n "$BYPASS_REASON" ]; then
	if [ "$CTX_HOOK_DEBUG" = "1" ]; then
		echo "[ctx_debug status=bypassed reason=$BYPASS_REASON script_dir=$SCRIPT_DIR workspace_dir=$WORKSPACE_DIR config_file=$CONFIG_FILE] $USER_MESSAGE"
	else
		echo "$USER_MESSAGE"
	fi
	exit 0
fi

# Build ctx command with optional unicorn flag
if [ -f "$SCRIPT_DIR/ctx.py" ]; then
	# Use embedded ctx.py when running from the packaged extension
	CTX_CMD=(python3 "$SCRIPT_DIR/ctx.py")
else
	# Fallback for repo-local usage
	CTX_CMD=(python3 scripts/ctx.py)
fi
case "${CTX_DEFAULT_MODE,,}" in
	unicorn)
		CTX_CMD+=("--unicorn")
		;;
	detail)
		CTX_CMD+=("--detail")
		;;
esac
CTX_CMD+=("$USER_MESSAGE" --collection "$CTX_COLLECTION")

# Run ctx with collection
# When CTX_DEBUG_PATHS is enabled, preserve stderr so path-level debug from ctx.py is visible
if [ -n "${CTX_DEBUG_PATHS:-}" ]; then
	ENHANCED=$(timeout 120s "${CTX_CMD[@]}" 2>&1 || echo "$USER_MESSAGE")
else
	ENHANCED=$(timeout 120s "${CTX_CMD[@]}" 2>/dev/null || echo "$USER_MESSAGE")
fi

if [ -n "$WORKSPACE_DIR" ] && [ "${CTX_ROOT_HINT:-1}" != "0" ]; then
	HINT="The user's project root directory is \"$WORKSPACE_DIR\" (WORKSPACE_DIR)."
	if [ "${CTX_SURFACE_COLLECTION_HINT:-0}" = "1" ] && [ -n "$CTX_COLLECTION" ]; then
		HINT="$HINT The Qdrant collection name for this workspace is \"$CTX_COLLECTION\". Specify this collection when using memory or qdrant-indexer MCP tool (if available)."
	fi
	if [ -n "${USER_CWD:-}" ]; then
		HINT="$HINT Claude's current working directory is \"$USER_CWD\" (user_cwd). \
When using tools like Read, Search, or Bash, treat WORKSPACE_DIR as the root \
for repository files. If WORKSPACE_DIR and user_cwd differ, do not assume \
files live under user_cwd; use the full paths under WORKSPACE_DIR or the \
project-relative paths shown above."
	fi
	ENHANCED="$HINT

$ENHANCED"
fi

# Log ctx output when logging is enabled
if [ "$LOG_ENABLED" = "1" ]; then
	{
		echo "[$(date -Iseconds)] CTX_OUTPUT"
		echo "PROMPT   = $USER_MESSAGE"
		echo "ENHANCED = <<EOF"
		echo "$ENHANCED"
		echo "EOF"
		echo
	} >> "$LOG_FILE"
fi

if [ "$CTX_HOOK_DEBUG" = "1" ]; then
	HOOK_STATUS="unchanged"
	if [ "$ENHANCED" != "$USER_MESSAGE" ]; then
		HOOK_STATUS="enhanced"
	fi
	echo "[ctx_debug status=$HOOK_STATUS script_dir=$SCRIPT_DIR workspace_dir=$WORKSPACE_DIR config_file=$CONFIG_FILE] $ENHANCED"
else
	echo "$ENHANCED"
fi