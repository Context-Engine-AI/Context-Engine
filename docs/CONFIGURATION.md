# Configuration Reference

Complete environment variable reference for Context Engine.

**Documentation:** [README](../README.md) · [Getting Started](GETTING_STARTED.md) · [Configuration](CONFIGURATION.md) · [IDE Clients](IDE_CLIENTS.md) · [MCP API](MCP_API.md) · [ctx CLI](CTX_CLI.md) · [Memory Guide](MEMORY_GUIDE.md) · [Architecture](ARCHITECTURE.md) · [Multi-Repo](MULTI_REPO_COLLECTIONS.md) · [Observability](OBSERVABILITY.md) · [Kubernetes](../deploy/kubernetes/README.md) · [VS Code Extension](vscode-extension.md) · [Troubleshooting](TROUBLESHOOTING.md) · [Development](DEVELOPMENT.md)

---

**On this page:**
- [Core Settings](#core-settings)
- [Embedding Models](#embedding-models)
- [Indexing & Micro-Chunks](#indexing--micro-chunks)
- [Query Optimization](#query-optimization)
- [Watcher Settings](#watcher-settings)
- [Reranker](#reranker)
- [Learning Reranker](#learning-reranker)
- [Decoder (llama.cpp / OpenAI / GLM / MiniMax)](#decoder-llamacpp--openai--glm--minimax)
- [Git History & Commit Indexing](#git-history--commit-indexing)
- [ReFRAG](#refrag)
- [Pattern Search](#pattern-search)
- [Lexical Vector Settings](#lexical-vector-settings)
- [Ports](#ports)
- [Search & Expansion](#search--expansion)
- [info_request Tool](#info_request-tool)
- [Memory Blending](#memory-blending)

---

## Core Settings

| Name | Description | Default |
|------|-------------|---------|
| COLLECTION_NAME | Qdrant collection name (unified across all repos) | codebase |
| REPO_NAME | Logical repo tag stored in payload for filtering | auto-detect from git/folder |
| HOST_INDEX_PATH | Host path mounted at /work in containers | current repo (.) |
| QDRANT_URL | Qdrant base URL | container: http://qdrant:6333; local: http://localhost:6333 |
| MULTI_REPO_MODE | Enable multi-repo collections (each subdir gets own collection) | 0 (disabled) |
| LOG_LEVEL | Logging verbosity: DEBUG, INFO, WARNING, ERROR, CRITICAL | INFO |
| CTXCE_AUTH_ENABLED | Enable API authentication (requires token header) | 0 (disabled) |
| CTXCE_AUTH_ADMIN_TOKEN | Admin token for authenticated requests | unset |

### Tool Description Customization

Override default MCP tool descriptions (useful for agent tuning).

| Name | Description | Default |
|------|-------------|---------|
| TOOL_STORE_DESCRIPTION | Custom description for memory_store tool | built-in |
| TOOL_FIND_DESCRIPTION | Custom description for memory_find tool | built-in |

## Embedding Models

Context Engine supports multiple embedding models via the `EMBEDDING_MODEL` and `EMBEDDING_PROVIDER` settings.

### Default (BGE-base)

The default configuration uses `BAAI/bge-base-en-v1.5` via fastembed:

| Name | Description | Default |
|------|-------------|---------|
| EMBEDDING_MODEL | Model name for dense embeddings | BAAI/bge-base-en-v1.5 |
| EMBEDDING_PROVIDER | Backend provider | fastembed |
| EMBEDDING_SEED | Seed for deterministic embeddings (used in benchmarks) | unset |

### Qwen3-Embedding (Experimental)

Qwen3-Embedding-0.6B offers improved semantic understanding with instruction-aware encoding. Enable via feature flag:

| Name | Description | Default |
|------|-------------|---------|
| QWEN3_EMBEDDING_ENABLED | Enable Qwen3 embedding support | 0 (disabled) |
| QWEN3_QUERY_INSTRUCTION | Add instruction prefix to search queries | 1 (enabled when Qwen3 active) |
| QWEN3_INSTRUCTION_TEXT | Custom instruction prefix | `Instruct: Given a code search query, retrieve relevant code snippets\nQuery:` |

**Setup:**
```bash
# In .env
QWEN3_EMBEDDING_ENABLED=1
EMBEDDING_MODEL=electroglyph/Qwen3-Embedding-0.6B-onnx-uint8
QWEN3_QUERY_INSTRUCTION=1
# Optional: customize instruction
# QWEN3_INSTRUCTION_TEXT=Instruct: Find code implementing this feature\nQuery:
```

**Important:** Switching embedding models requires a full reindex:
```bash
make reset-dev-dual  # Recreates collection and reindexes
```

**Dimension comparison:**
| Model | Dimensions | Notes |
|-------|-----------|-------|
| BGE-base-en-v1.5 | 768 | Default, well-tested |
| Qwen3-Embedding-0.6B | 1024 | Instruction-aware, experimental |

## Indexing & Micro-Chunks

| Name | Description | Default |
|------|-------------|---------|
| INDEX_MICRO_CHUNKS | Enable token-based micro-chunking | 0 (off) |
| MAX_MICRO_CHUNKS_PER_FILE | Cap micro-chunks per file | 200 |
| TOKENIZER_URL | HF tokenizer.json URL (for Make download) | n/a |
| TOKENIZER_PATH | Local path where tokenizer is saved (Make) | models/tokenizer.json |
| TOKENIZER_JSON | Runtime path for tokenizer (indexer) | models/tokenizer.json |
| USE_TREE_SITTER | Enable tree-sitter parsing (py/js/ts) | 1 (on) |
| INDEX_USE_ENHANCED_AST | Enable advanced AST-based semantic chunking | 1 (on) |
| INDEX_SEMANTIC_CHUNKS | Enable semantic chunking (preserve function/class boundaries) | 1 (on) |
| INDEX_CHUNK_LINES | Lines per chunk (non-micro mode) | 120 |
| INDEX_CHUNK_OVERLAP | Overlap lines between chunks | 20 |
| INDEX_BATCH_SIZE | Upsert batch size | 64 |
| INDEX_PROGRESS_EVERY | Log progress every N files | 200 |
| SMART_SYMBOL_REINDEXING | Reuse embeddings when only symbols change | 1 (enabled) |
| MAX_CHANGED_SYMBOLS_RATIO | Threshold for full reindex vs smart update | 0.6 |

## Query Optimization

Dynamic HNSW_EF tuning and intelligent query routing for 2x faster simple queries.

| Name | Description | Default |
|------|-------------|---------|
| QUERY_OPTIMIZER_ADAPTIVE | Enable adaptive EF optimization | 1 (on) |
| QUERY_OPTIMIZER_MIN_EF | Minimum EF value | 64 |
| QUERY_OPTIMIZER_MAX_EF | Maximum EF value | 512 |
| QUERY_OPTIMIZER_SIMPLE_THRESHOLD | Complexity threshold for simple queries | 0.3 |
| QUERY_OPTIMIZER_COMPLEX_THRESHOLD | Complexity threshold for complex queries | 0.7 |
| QUERY_OPTIMIZER_SIMPLE_FACTOR | EF multiplier for simple queries | 0.5 |
| QUERY_OPTIMIZER_SEMANTIC_FACTOR | EF multiplier for semantic queries | 1.0 |
| QUERY_OPTIMIZER_COMPLEX_FACTOR | EF multiplier for complex queries | 2.0 |
| QUERY_OPTIMIZER_DENSE_THRESHOLD | Complexity threshold for dense-only routing | 0.2 |
| QUERY_OPTIMIZER_COLLECTION_SIZE | Approximate collection size for scaling | 10000 |
| QDRANT_EF_SEARCH | Base HNSW_EF value (overridden by optimizer) | 128 |

## Watcher Settings

| Name | Description | Default |
|------|-------------|---------|
| WATCH_DEBOUNCE_SECS | Debounce between FS events | 1.5 |
| INDEX_UPSERT_BATCH | Upsert batch size (watcher) | 128 |
| INDEX_UPSERT_RETRIES | Retry count | 5 |
| INDEX_UPSERT_BACKOFF | Seconds between retries | 0.5 |
| QDRANT_TIMEOUT | HTTP timeout seconds | watcher: 60; search: 20 |
| MCP_TOOL_TIMEOUT_SECS | Max duration for long-running MCP tools | 3600 |

## Reranker

Cross-encoder reranking improves search quality by scoring query-document pairs directly. Context Engine supports two configuration methods:

### FastEmbed Model (Recommended)

Set `RERANKER_MODEL` to use FastEmbed's auto-downloading cross-encoder models:

| Name | Description | Default |
|------|-------------|---------|
| RERANKER_MODEL | FastEmbed reranker model name | unset |
| RERANKER_ENABLED | Enable reranker by default | 1 (enabled) |

**Popular models:**
- `jinaai/jina-reranker-v2-base-multilingual` - Multilingual, good quality
- `BAAI/bge-reranker-base` - English-focused, fast
- `Xenova/ms-marco-MiniLM-L-6-v2` - Lightweight, fast inference

Example:
```bash
RERANKER_MODEL=jinaai/jina-reranker-v2-base-multilingual
RERANKER_ENABLED=1
```

### Manual ONNX Paths (Legacy)

For custom models or explicit control, set both ONNX path and tokenizer:

| Name | Description | Default |
|------|-------------|---------|
| RERANKER_ONNX_PATH | Local ONNX cross-encoder model path | unset |
| RERANKER_TOKENIZER_PATH | Tokenizer path for reranker | unset |
| RERANKER_ENABLED | Enable reranker by default | 1 (enabled) |

**Note:** If both `RERANKER_MODEL` and `RERANKER_ONNX_PATH` are set, `RERANKER_MODEL` takes priority.

### Reranker Tuning

| Name | Description | Default |
|------|-------------|---------|
| RERANKER_TOPN | Candidates to retrieve before reranking | 50 |
| RERANKER_RETURN_M | Final results after reranking | 12 |
| RERANKER_TIMEOUT_MS | Rerank timeout in milliseconds | 2000 |
| RERANK_BLEND_WEIGHT | Ratio of rerank vs fusion score (0.0-1.0) | 0.6 |
| RERANK_TIMEOUT_FLOOR_MS | Min timeout to avoid cold-start failures | 1000 |
| POST_RERANK_SYMBOL_BOOST | Score boost for exact symbol matches after rerank | 1.0 |
| EMBEDDING_WARMUP | Warm up embedding model on startup | 0 (disabled) |
| RERANK_WARMUP | Warm up reranker model on startup | 0 (disabled) |

## Learning Reranker

The learning reranker trains a lightweight neural network (TinyScorer) to improve search rankings over time. See [Architecture](ARCHITECTURE.md#5-learning-reranker-system) for details.

**This feature is optional and enabled by default.** To disable:

```bash
# Disable learning scorer in search results
RERANK_LEARNING=0

# Disable event logging (no training data collected)
RERANK_EVENTS_ENABLED=0

# Or simply don't run the learning_worker container
```

### Enable/Disable

| Name | Description | Default |
|------|-------------|---------|
| RERANK_LEARNING | Enable learning scorer in search results | 1 (enabled) |
| RERANK_EVENTS_ENABLED | Enable event logging for training | 1 (enabled) |
| RERANK_EVENTS_SAMPLE_RATE | Fraction of events to log (0.0-1.0) | 0.33 |

### Weight Management

| Name | Description | Default |
|------|-------------|---------|
| RERANKER_WEIGHTS_DIR | Directory for learned weight files | /tmp/rerank_weights |
| RERANKER_WEIGHTS_RELOAD_INTERVAL | How often to check for new weights (seconds) | 60 |
| RERANKER_MAX_CHECKPOINTS | Number of weight versions to retain | 5 |

### Learning Rate

| Name | Description | Default |
|------|-------------|---------|
| RERANKER_LR_DECAY_STEPS | Updates between learning rate decay | 1000 |
| RERANKER_LR_DECAY_RATE | Decay multiplier (e.g., 0.95 = 5% reduction) | 0.95 |
| RERANKER_MIN_LR | Minimum learning rate floor | 0.0001 |

### Event Logging

| Name | Description | Default |
|------|-------------|---------|
| RERANK_EVENTS_DIR | Directory for search event logs | /tmp/rerank_events |
| RERANK_EVENTS_RETENTION_DAYS | Days to keep event files before cleanup | 7 |

### Learning Worker

| Name | Description | Default |
|------|-------------|---------|
| RERANK_LEARNING_BATCH_SIZE | Number of events per training batch | 32 |
| RERANK_LEARNING_POLL_INTERVAL | Seconds between checking for new events | 30 |
| RERANK_LEARNING_RATE | Initial learning rate for TinyScorer | 0.001 |
| RERANK_LLM_TEACHER | Enable LLM-teacher guided learning | 1 (enabled) |
| RERANK_LLM_SAMPLE_RATE | Fraction of queries to evaluate with LLM teacher | 1.0 |
| RERANK_VICREG_WEIGHT | Weight for VICReg consistency loss | 0.1 |

## Decoder (llama.cpp / OpenAI / GLM / MiniMax)

| Name | Description | Default |
|------|-------------|---------|
| REFRAG_DECODER | Enable decoder for context_answer (required for llamacpp) | 1 (enabled) |
| REFRAG_RUNTIME | Decoder backend: llamacpp, openai, glm, or minimax | llamacpp |
| LLAMACPP_URL | llama.cpp server endpoint | http://llamacpp:8080 or http://host.docker.internal:8081 |
| LLAMACPP_TIMEOUT_SEC | Decoder request timeout | 300 |
| DECODER_MAX_TOKENS | Max tokens for decoder responses | 4000 |
| REFRAG_DECODER_MODE | prompt or soft (soft requires patched llama.cpp) | prompt |
| OPENAI_API_KEY | API key for OpenAI provider | unset |
| OPENAI_MODEL | OpenAI model name | gpt-4.1-mini |
| OPENAI_API_BASE | OpenAI API base URL (supports Azure/compatible endpoints) | https://api.openai.com/v1 |
| GLM_API_KEY | API key for GLM provider | unset |
| GLM_MODEL | GLM model name (used for context_answer) | glm-4.6 |
| GLM_MODEL_FAST | GLM model for expand_query/simple tasks (higher concurrency) | glm-4.5 |
| GLM_TIMEOUT_SEC | GLM request timeout in seconds | unset |
| PSEUDO_BATCH_CONCURRENCY | Parallel API calls for pseudo-tag indexing (1=sequential, 4=4x speedup) | 1 |
| MINIMAX_API_KEY | API key for MiniMax M2 provider | unset |
| MINIMAX_MODEL | MiniMax model name | MiniMax-M2 |
| MINIMAX_API_BASE | MiniMax API base URL | https://api.minimax.io/v1 |
| MINIMAX_TIMEOUT_SEC | MiniMax request timeout in seconds | unset |
| USE_GPU_DECODER | Native Metal decoder (1) vs Docker (0) | 0 (docker) |
| LLAMACPP_GPU_LAYERS | Number of layers to offload to GPU, -1 for all | 32 |

### Runtime Selection

Set `REFRAG_RUNTIME` explicitly to choose a decoder backend:
- **llamacpp**: Local llama.cpp server (requires `REFRAG_DECODER=1`)
- **openai**: OpenAI API (GPT-4.1, GPT-4.1-mini, o1, etc.)
- **glm**: ZhipuAI GLM models (GLM-4.5, GLM-4.6, GLM-4.7)
- **minimax**: MiniMax M2 API

No auto-detection is performed to avoid surprise API calls. If `REFRAG_RUNTIME` is unset, it defaults to `llamacpp`.

## Git History & Commit Indexing

Settings for indexing git commit history and enabling commit-aware search.

| Name | Description | Default |
|------|-------------|---------|
| REFRAG_COMMIT_DESCRIBE | Enable commit lineage goals for indexing | 1 (enabled) |
| COMMIT_VECTOR_SEARCH | Enable vector search over commit messages | 0 (disabled) |
| REMOTE_UPLOAD_GIT_MAX_COMMITS | Max commits per upload bundle (0 = no git history) | 500 |
| GIT_HISTORY_PRUNE | Prune old git_message points on manifest ingest | 1 (enabled) |
| GIT_HISTORY_DELETE_MANIFEST | Delete manifest files after successful ingest | 1 (enabled) |
| GIT_HISTORY_MANIFEST_MAX_FILES | Cap manifest files per .remote-git dir (0 = unlimited) | 50 |

**Note:** Git history indexing stores commit messages and metadata as searchable points. Use `search_commits_for` MCP tool to query.

## ReFRAG (Micro-Chunking & Retrieval)

| Name | Description | Default |
|------|-------------|---------|
| REFRAG_MODE | Enable micro-chunking and span budgeting | 1 (enabled) |
| REFRAG_GATE_FIRST | Enable mini-vector gating | 1 (enabled) |
| REFRAG_CANDIDATES | Candidates for gate-first filtering | 200 |
| REFRAG_PSEUDO_DESCRIBE | Enable LLM-based pseudo/tags generation during indexing | 0 (disabled) |
| MICRO_BUDGET_TOKENS | Token budget for context_answer | 5000 (GLM: 6000-8192) |
| MICRO_OUT_MAX_SPANS | Max spans returned per query | 8 (GLM: 24) |
| MICRO_CHUNK_TOKENS | Tokens per micro-chunk window | 16 |
| MICRO_CHUNK_STRIDE | Stride between windows | 8 |
| MICRO_MERGE_LINES | Lines to merge adjacent spans | 4 |
| MICRO_TOKENS_PER_LINE | Estimated tokens per line | 32 |

**LLM-Based Pseudo/Tags (`REFRAG_PSEUDO_DESCRIBE`):**

When enabled, the indexer uses the configured decoder (via `REFRAG_RUNTIME`) to generate semantic descriptions and tags for each code chunk. This enriches the lexical vectors with natural language terms, improving NL→code retrieval.

```bash
# Enable LLM pseudo/tags generation (requires decoder configured)
REFRAG_PSEUDO_DESCRIBE=1
REFRAG_RUNTIME=glm  # or openai, minimax, llamacpp
```

**Note:** This significantly increases indexing time and API costs. Best used with batch concurrency (`PSEUDO_BATCH_CONCURRENCY=4`).

### Pseudo Backfill Worker

Deferred pseudo/tag generation runs asynchronously after initial indexing.

| Name | Description | Default |
|------|-------------|---------|
| PSEUDO_BACKFILL_ENABLED | Enable async pseudo/tag backfill worker | 0 (disabled) |
| PSEUDO_DEFER_TO_WORKER | Skip inline pseudo, defer to backfill worker | 0 (disabled) |

### Adaptive Span Sizing

Expand search hits to full symbol boundaries for better context.

| Name | Description | Default |
|------|-------------|---------|
| ADAPTIVE_SPAN_SIZING | Expand hits to encompassing symbol boundaries | 1 (enabled) |
| DEBUG_ADAPTIVE_SPAN | Enable debug logging for span expansion | 0 (disabled) |

### Context Answer Shaping

Controls output formatting for `context_answer` responses.

| Name | Description | Default |
|------|-------------|---------|
| CTX_SUMMARY_CHARS | Max chars for summary (0 = disabled) | 0 |
| CTX_SNIPPET_CHARS | Max chars per code snippet | 400 |
| DEBUG_CONTEXT_ANSWER | Enable debug logging for context_answer | 0 (disabled) |

### Mini Vector Gating

Compact 64-dim vectors for fast candidate filtering before full dense search.

| Name | Description | Default |
|------|-------------|---------|
| MINI_VECTOR_NAME | Name of mini vector index | mini |
| MINI_VEC_DIM | Dimension of mini vectors | 64 |
| MINI_VEC_SEED | Random projection seed (for reproducibility) | 1337 |
| HYBRID_MINI_WEIGHT | Weight of mini vectors in hybrid scoring | 0.5 |

## Pattern Search

Structural code pattern matching across languages. Disabled by default.

| Name | Description | Default |
|------|-------------|---------|
| PATTERN_VECTORS | Enable pattern_search tool and pattern vector indexing | 0 (disabled) |

**Enable:**
```bash
# In .env or docker-compose
PATTERN_VECTORS=1
```

When enabled, the indexer extracts control-flow signatures (loops, branches, try/except, etc.) and stores them as pattern vectors. The `pattern_search` MCP tool allows finding structurally similar code across languages—e.g., a Python retry loop can match Go/Rust equivalents.

**Note:** Enabling requires reindexing to generate pattern vectors for existing files.

## Lexical Vector Settings

Controls the sparse lexical (keyword) vectors used for hybrid search.

| Name | Description | Default |
|------|-------------|---------|
| LEX_VECTOR_NAME | Name of lexical vector in Qdrant | lex |
| LEX_VECTOR_DIM | Dimension of lexical hash vector | 2048 |
| LEX_MULTI_HASH | Hash functions per token (more = better collision resistance) | 3 |
| LEX_BIGRAMS | Enable bigram hashing for phrase matching | 1 (enabled) |
| LEX_BIGRAM_WEIGHT | Weight for bigram entries relative to unigrams | 0.7 |

### Sparse Vector Settings (Experimental)

True sparse vectors for lossless lexical matching (no hash collisions).

| Name | Description | Default |
|------|-------------|---------|
| LEX_SPARSE_MODE | Enable sparse lexical vectors instead of dense hash vectors | 0 (off) |
| LEX_SPARSE_NAME | Name of sparse vector index in Qdrant | lex_sparse |

**Note:** Enabling `LEX_SPARSE_MODE` requires the collection to have a sparse vector index configured. Use `--recreate` flag when switching modes. If sparse query fails or returns empty, the system automatically falls back to dense lexical vectors.

**Note:** Changing `LEX_VECTOR_DIM` requires recreating collections (`--recreate` flag).
To use legacy settings (pre-v2): `LEX_VECTOR_DIM=4096 LEX_MULTI_HASH=1 LEX_BIGRAMS=0`

## Ports

| Name | Description | Default |
|------|-------------|---------|
| FASTMCP_PORT | Memory MCP server port (SSE) | 8000 |
| FASTMCP_INDEXER_PORT | Indexer MCP server port (SSE) | 8001 |
| FASTMCP_HTTP_PORT | Memory RMCP host port mapping | 8002 |
| FASTMCP_INDEXER_HTTP_PORT | Indexer RMCP host port mapping | 8003 |
| FASTMCP_HEALTH_PORT | Health port (memory/indexer) | memory: 18000; indexer: 18001 |

## Search & Expansion

| Name | Description | Default |
|------|-------------|---------|
| HYBRID_EXPAND | Enable heuristic multi-query expansion | 0 (off) |
| LLM_EXPAND_MAX | Max number of alternate queries to generate via LLM (0 = disabled) | 0 |
| EXPAND_MAX_TOKENS | Max tokens for LLM query expansion response | 512 |
| REPO_AUTO_FILTER | Auto-detect and filter to current repo in searches | 1 (enabled) |
| HYBRID_IN_PROCESS | Run hybrid search in-process (faster, falls back to subprocess) | 1 (enabled) |
| RERANK_IN_PROCESS | Run reranker in-process (faster, falls back to subprocess) | 1 (enabled) |
| PARALLEL_DENSE_QUERIES | Enable parallel dense query execution | 1 (enabled) |
| PARALLEL_DENSE_THRESHOLD | Min queries to trigger parallelization | 4 |
| HYBRID_SYMBOL_BOOST | Score boost for exact symbol matches | 0.35 |
| HYBRID_RECENCY_WEIGHT | Weight for recently modified files | 0.1 |
| HYBRID_PER_PATH | Max results per file path | 2 |
| HYBRID_SNIPPET_DISK_READ | Allow snippet scoring to read file contents | 1 (enabled) |
| PRF_ENABLED | Enable Pseudo-Relevance Feedback (refined second pass) | 1 (enabled) |
| RERANK_EXPAND | Expand candidates before reranking | 1 (enabled) |
| REPO_SEARCH_DEFAULT_LIMIT | Default result limit for repo_search | 10 |

**Note:** `REPO_AUTO_FILTER=0` disables automatic repo scoping, useful for benchmarks or cross-repo searches.

### Caching

| Name | Description | Default |
|------|-------------|---------|
| MAX_EMBED_CACHE | Max cached embeddings | 16384 |
| HYBRID_RESULTS_CACHE | Max cached search results | 128 |
| HYBRID_RESULTS_CACHE_ENABLED | Enable search result caching | 1 (enabled) |

### Semantic Expansion

Synonym/related term expansion for improved recall on natural language queries.

| Name | Description | Default |
|------|-------------|---------|
| SEMANTIC_EXPANSION_ENABLED | Enable semantic term expansion | 1 (enabled) |
| SEMANTIC_EXPANSION_TOP_K | Number of similar terms to consider | 5 |
| SEMANTIC_EXPANSION_SIMILARITY_THRESHOLD | Min similarity for expansion terms | 0.7 |
| SEMANTIC_EXPANSION_MAX_TERMS | Max expansion terms added per query | 3 |
| SEMANTIC_EXPANSION_CACHE_SIZE | Cache size for expansion lookups | 1000 |
| SEMANTIC_EXPANSION_CACHE_TTL | Cache TTL in seconds | 3600 |

### LLM Query Expansion

Query expansion uses the decoder infrastructure (set via `REFRAG_RUNTIME`):
- **openai**: Uses OpenAI API when `REFRAG_RUNTIME=openai` and `OPENAI_API_KEY` is set
- **glm**: Uses GLM API when `REFRAG_RUNTIME=glm` and `GLM_API_KEY` is set
- **minimax**: Uses MiniMax API when `REFRAG_RUNTIME=minimax` and `MINIMAX_API_KEY` is set
- **llamacpp**: Uses local llama.cpp when `REFRAG_RUNTIME=llamacpp` and `REFRAG_DECODER=1`

Set `LLM_EXPAND_MAX=4` to enable LLM-assisted query expansion (generates up to 4 alternate phrasings).
`EXPAND_MAX_TOKENS` controls the response length budget for the LLM call.

### Filename Boost

The search engine can boost files whose paths match query terms—production-grade algorithm for real-world codebases.

| Name | Description | Default |
|------|-------------|---------|
| FNAME_BOOST | Base score boost factor for path/query token matches | 0.15 |

**Naming convention support:**
- snake_case, camelCase, PascalCase, kebab-case, SCREAMING_CASE
- Acronyms: `XMLParser` → xml, parser; `HTTPClient` → http, client
- Prefixes stripped: `IUserService` → user, service; `_private` → private
- Dot notation: `com.company.auth` → com, company, auth

**Abbreviation normalization:**
- auth ↔ authenticate/authentication
- config ↔ configuration/cfg/conf
- repo ↔ repository, util ↔ utility, impl ↔ implementation, etc.

**Scoring tiers:**
- Exact token match: 1.0 × factor
- Normalized match (abbreviation/plural): 0.8 × factor
- Substring containment: 0.4 × factor
- Filename bonus: 1.5× multiplier for filename vs directory matches
- Common token penalty: 0.5× for tokens like "utils", "index", "main"

**Example:** Query "authenticate user handler" matching `auth/UserAuthHandler.ts`:
- "user" exact match in filename (1.0 × 1.5 = 1.5)
- "authenticate" → "auth" normalized (0.8 × 1.5 = 1.2)
- "handler" exact match in filename (1.0 × 1.5 = 1.5)
- Total: 4.2 × 0.15 = 0.63 boost

Set `FNAME_BOOST=0` to disable, or increase (e.g., `0.25`) for stronger path weighting.

## info_request Tool

Simplified codebase retrieval with optional explanation mode.

| Name | Description | Default |
|------|-------------|---------|
| INFO_REQUEST_LIMIT | Default result limit for info_request queries | 10 |
| INFO_REQUEST_CONTEXT_LINES | Context lines in snippets (richer than repo_search) | 5 |

## Output Formatting

### TOON (Token-Oriented Object Notation)

Compact output format that reduces token usage by 40-60%.

| Name | Description | Default |
|------|-------------|---------|
| TOON_ENABLED | Enable TOON format by default for all search output | 0 (disabled) |

Set `output_format="toon"` per-call, or enable globally via `TOON_ENABLED=1`.

## Memory Blending

| Name | Description | Default |
|------|-------------|---------|
| MEMORY_SSE_ENABLED | Enable SSE memory blending | false |
| MEMORY_MCP_URL | Memory MCP endpoint for blending | http://mcp:8000/sse |
| MEMORY_MCP_TIMEOUT | Timeout for memory queries | 6 |
| MEMORY_AUTODETECT | Auto-detect memory collection | 1 |
| MEMORY_COLLECTION_TTL_SECS | Cache TTL for collection detection | 300 |

---

## Exclusions (.qdrantignore)

The indexer supports a `.qdrantignore` file at the repo root (similar to `.gitignore`).

**Default exclusions** (overridable):
- `/models`, `/node_modules`, `/dist`, `/build`
- `/.venv`, `/venv`, `/__pycache__`, `/.git`
- `*.onnx`, `*.bin`, `*.safetensors`, `tokenizer.json`, `*.whl`, `*.tar.gz`

**Override via env or flags:**
```bash
# Disable defaults
QDRANT_DEFAULT_EXCLUDES=0

# Custom ignore file
QDRANT_IGNORE_FILE=.myignore

# Additional excludes
QDRANT_EXCLUDES='tokenizer.json,*.onnx,/third_party'
```

**CLI examples:**
```bash
docker compose run --rm indexer --root /work --ignore-file .qdrantignore
docker compose run --rm indexer --root /work --no-default-excludes --exclude '/vendor' --exclude '*.bin'
```

---

## Scaling Recommendations

| Repo Size | Chunk Lines | Overlap | Batch Size |
|-----------|------------|---------|------------|
| Small (<100 files) | 80-120 | 16-24 | 32-64 |
| Medium (100s-1k files) | 120-160 | ~20 | 64-128 |
| Large (1k+ files) | 120 (default) | 20 | 128+ |

For large monorepos, set `INDEX_PROGRESS_EVERY=200` for visibility.

