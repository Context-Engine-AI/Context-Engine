# Getting Started (VS Code Extension)

**Documentation:** [README](../README.md) · [Getting Started](GETTING_STARTED.md) · [Configuration](CONFIGURATION.md) · [IDE Clients](IDE_CLIENTS.md) · [MCP API](MCP_API.md) · [ctx CLI](CTX_CLI.md) · [Memory Guide](MEMORY_GUIDE.md) · [Architecture](ARCHITECTURE.md) · [Multi-Repo](MULTI_REPO_COLLECTIONS.md) · [Observability](OBSERVABILITY.md) · [Kubernetes](../deploy/kubernetes/README.md) · [VS Code Extension](vscode-extension.md) · [Troubleshooting](TROUBLESHOOTING.md) · [Development](DEVELOPMENT.md)

This guide covers the detailed setup flow using the VS Code extension for the lowest-friction Context-Engine experience.

---

## 1. Prerequisites

- **Docker** (Docker Desktop or equivalent)
- **VS Code** (for the Context Engine Uploader extension)
- **An MCP-enabled IDE or client** such as:
  - Claude Code, Windsurf, Cursor, Roo, Cline, Zed (via `mcp-remote`), etc.

---

## 2. Extension Setup

### 2.1 Install Extension
1. Install [Context Engine Uploader](https://marketplace.visualstudio.com/items?itemName=context-engine.context-engine-uploader) from VS Code Marketplace

### 2.2 First-Time Setup
2. Open any VS Code project
3. Extension prompts to set up Context-Engine stack:
   - Choose location to clone Context-Engine (keeps it separate from your project)
   - Extension automatically starts the Docker stack
   - Configures MCP bridge for unified endpoint and automatic path handling
   - Writes MCP configs for Claude Code, Windsurf, and Augment

**Stack Services Started:**
- Qdrant: `http://localhost:6333`
- Memory MCP: `http://localhost:8000/sse` (SSE) and `http://localhost:8002/mcp` (HTTP)
- Indexer MCP: `http://localhost:8001/sse` (SSE) and `http://localhost:8003/mcp` (HTTP)
- Upload Service: `http://localhost:8004`

### 2.3 Index Your Workspace
4. Click "Upload Workspace" in VS Code status bar
5. Extension indexes your code automatically:
   - Mirrors project to container workspace
   - Chunks files into 5-50 line spans using ReFRAG-inspired micro-chunking
   - Stores vectors + metadata in Qdrant for hybrid search
   - Tracks file hashes for efficient reindexing
   - **Smart path handling** automatically maps container paths to your local files

---

## 3. IDE Configuration

The extension automatically configures MCP settings for Claude Code, Windsurf, and Augment.

### For Other IDEs or Manual Setup

**Manual CLI indexing** (if you prefer not to use the extension):
```bash
git clone https://github.com/m1rl0k/Context-Engine.git && cd Context-Engine
make bootstrap  # One-shot setup, or step-by-step:
HOST_INDEX_PATH=/path/to/your/project docker compose run --rm indexer
```

**MCP configuration for other IDEs**:
See [docs/IDE_CLIENTS.md](IDE_CLIENTS.md) for complete copy-paste configuration examples for:
- Claude Code, Windsurf, Cursor, Cline, Codex, Augment, Zed (via mcp-remote)
- Both HTTP and SSE transport options
- Workspace-aware bridge configuration with npx

---

## 4. Usage Examples

Once your IDE is connected and indexing complete, ask your AI assistant questions like:

### Code Search Examples

- "Find places where remote uploads are retried"
- "Show functions that call ingest_code.index_repo"
- "Search for performance bottlenecks in upload_service script"
- "How does the memory MCP store and retrieve entries?"

### Git History Examples (if enabled)

- "When did we add git history ingestion to the upload client?"
- "Show commits related to the chunking algorithm changes"

Your AI assistant will call the appropriate MCP tools (`repo_search`, `context_answer`, `search_commits_for`) automatically.

---

## 5. Next Steps

- **Memory Usage**: Store team knowledge alongside your code - see [Memory Guide](MEMORY_GUIDE.md)
- **Advanced Configuration**: Fine-tune search and indexing - see [Configuration](CONFIGURATION.md)
- **Troubleshooting**: Common issues and solutions - see [Troubleshooting](TROUBLESHOOTING.md)
- **Development**: Contributing and local development - see [Development](DEVELOPMENT.md)

---

## 6. Advanced Features

### Prompt+ Enhancement
The extension includes a **Prompt+** button that enhances selected code or prompts using Context-Engine's built-in `ctx.py` system.

**How Prompt+ works:**
- Select code or write a prompt in your editor
- Click `Prompt+` in the status bar (or use the command palette)
- Context-Engine retrieves relevant code context and rewrites your input into a more precise, context-aware prompt
- Enhanced prompt is ready to send to your AI assistant with rich code context

**Example enhancement:**
- **Original prompt**: "How do I handle errors?"
- **Enhanced by Prompt+**: "How do I handle exceptions in the authentication flow in `src/auth/login.py`, considering the `ValidationError` class and the retry mechanism in `src/utils/retry.py`?"

### Memory System
Store team knowledge alongside your code:
- Installation notes, runbooks, decisions
- Links, known issues, FAQs
- "How we do X here" team knowledge

See [Memory Guide](MEMORY_GUIDE.md) for detailed usage.

### Enterprise Features

**Unified MCP Bridge**
- Combines indexer and memory services into a single endpoint
- Automatic session management and workspace-aware collection injection
- No need to configure multiple MCP servers manually

**Authentication (Optional)**
- Built-in authentication system for team deployments
- Session management with automatic refresh
- Secure credential storage in `~/.ctxce/auth.json`

**Smart Path Handling**
- Automatic conversion between container paths (`/work/`) and your local filesystem
- Cross-platform compatibility (Windows/macOS/Linux)
- No manual path configuration required

---

## 7. Troubleshooting

If you encounter issues:

1. **Stack not starting**: Check Docker Desktop is running
2. **No search results**: Verify indexing completed (check extension logs)
3. **MCP connection issues**: Confirm IDE configuration matches extension output
4. **Performance**: Try the [Troubleshooting](TROUBLESHOOTING.md) guide for common optimizations

Common fixes are usually in [Troubleshooting](TROUBLESHOOTING.md).

For complete extension capabilities and settings, see [VS Code Extension](vscode-extension.md).
