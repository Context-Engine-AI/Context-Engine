# ctx.py - Prompt Enhancer CLI

CLI that retrieves code context and rewrites input into context-aware prompts using the local LLM decoder. Works with questions and commands/instructions.

**Documentation:** [README](../README.md) · [Getting Started](GETTING_STARTED.md) · [Configuration](CONFIGURATION.md) · [IDE Clients](IDE_CLIENTS.md) · [MCP API](MCP_API.md) · [ctx CLI](CTX_CLI.md) · [Memory Guide](MEMORY_GUIDE.md) · [Architecture](ARCHITECTURE.md) · [Multi-Repo](MULTI_REPO_COLLECTIONS.md) · [Observability](OBSERVABILITY.md) · [Kubernetes](../deploy/kubernetes/README.md) · [VS Code Extension](vscode-extension.md) · [Troubleshooting](TROUBLESHOOTING.md) · [Development](DEVELOPMENT.md)

---

**On this page:**
- [Basic Usage](#basic-usage)
- [Detail Mode](#detail-mode)
- [Unicorn Mode](#unicorn-mode)
- [Advanced Features](#advanced-features)
- [GPU Acceleration](#gpu-acceleration)
- [Configuration](#configuration)

---

## Basic Usage

```bash
# Questions: Enhanced with specific details and aspects
scripts/ctx.py "What is ReFRAG?"

# Commands: Enhanced with concrete targets and implementation details
scripts/ctx.py "Refactor ctx.py"

# Via Make target
make ctx Q="Explain the caching logic to me in detail"

# Filter by language/path or adjust tokens
make ctx Q="Hybrid search details" ARGS="--language python --under scripts/ --limit 2 --rewrite-max-tokens 200"
```

## Detail Mode

Include compact code snippets in retrieved context for richer rewrites (trades speed for quality):

```bash
# Enable detail mode (adds short snippets)
scripts/ctx.py "Explain the caching logic" --detail

# Detail mode with commands
scripts/ctx.py "Add error handling to ctx.py" --detail

# Adjust snippet size (default is 1 line when --detail is used)
make ctx Q="Explain hybrid search" ARGS="--detail --context-lines 2"
```

**Notes:**
- Default: header-only (fastest). `--detail` adds short snippets
- Detail mode optimized for speed: clamps to max 4 results, 1 result per file

## Unicorn Mode

Use `--unicorn` for the highest quality prompt enhancement with a staged 2-3 pass approach:

```bash
# Unicorn mode with commands
scripts/ctx.py "refactor ctx.py" --unicorn

# Unicorn mode with questions
scripts/ctx.py "what is ReFRAG and how does it work?" --unicorn

# Works with all filters
scripts/ctx.py "add error handling" --unicorn --language python
```

**How it works:**

1. **Pass 1 (Draft)**: Retrieves rich code snippets (8 lines of context) to understand the codebase
2. **Pass 2 (Refine)**: Retrieves even richer snippets (12 lines) to ground the prompt with concrete code
3. **Pass 3 (Polish)**: Optional cleanup pass if output appears generic or incomplete

**Key features:**
- **Code-grounded**: References actual code behaviors and patterns
- **No hallucinations**: Only uses real code from your indexed repository
- **Multi-paragraph output**: Produces detailed, comprehensive prompts
- **Works with both questions and commands**

**When to use:**
- **Normal mode**: Quick, everyday prompts (fastest)
- **--detail**: Richer context without multi-pass overhead (balanced)
- **--unicorn**: When you need the absolute best prompt quality

## Advanced Features

### Streaming Output (Default)

All modes stream tokens as they arrive for instant feedback:

```bash
scripts/ctx.py "refactor ctx.py" --unicorn
```

To disable streaming, set `"streaming": false` in `~/.ctx_config.json`

### Memory Blending

Automatically falls back to `context_search` with memories when repo search returns no hits:

```bash
# If no code matches, ctx.py will search design docs and ADRs
scripts/ctx.py "What is our authentication strategy?"
```

### Adaptive Context Sizing

Automatically adjusts `limit` and `context_lines` based on query characteristics:
- **Short/vague queries** → More context for richer grounding
- **Queries with file/function names** → Lighter settings for speed

### Automatic Quality Assurance

Enhanced `_needs_polish()` heuristic triggers a third polish pass when:
- Output is too short (< 180 chars)
- Contains generic/vague language
- Missing concrete code references
- Lacks proper paragraph structure

### Personalized Templates

Create `~/.ctx_config.json` to customize behavior:

```json
{
  "always_include_tests": true,
  "prefer_bullet_commands": false,
  "extra_instructions": "Always consider error handling and edge cases",
  "streaming": true
}
```

**Available preferences:**
- `always_include_tests`: Add testing considerations to all prompts
- `prefer_bullet_commands`: Format commands as bullet points
- `extra_instructions`: Custom instructions added to every rewrite
- `streaming`: Enable/disable streaming output (default: true)

See `ctx_config.example.json` for a template.

## GPU Acceleration

For faster prompt rewriting, use the native Metal-accelerated decoder:

```bash
# Start the native llama.cpp server with Metal GPU
scripts/gpu_toggle.sh start

# Now ctx.py will automatically use the GPU decoder on port 8081
make ctx Q="Explain the caching logic"

# Stop the native GPU server
scripts/gpu_toggle.sh stop
```

## Configuration

| Setting | Description | Default |
|---------|-------------|---------|
| MCP_INDEXER_URL | Indexer HTTP RMCP endpoint | http://localhost:8003/mcp |
| USE_GPU_DECODER | Auto-detect GPU mode | 0 |
| LLAMACPP_URL | Docker decoder endpoint | http://localhost:8080 |

GPU decoder (after `gpu_toggle.sh gpu`): http://localhost:8081/completion

