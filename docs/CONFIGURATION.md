# Configuration Reference

Complete environment variable reference for Context Engine.

**Documentation:** [README](../README.md) · [Getting Started](GETTING_STARTED.md) · [Configuration](CONFIGURATION.md) · [IDE Clients](IDE_CLIENTS.md) · [MCP API](MCP_API.md) · [ctx CLI](CTX_CLI.md) · [Memory Guide](MEMORY_GUIDE.md) · [Architecture](ARCHITECTURE.md) · [Multi-Repo](MULTI_REPO_COLLECTIONS.md) · [Kubernetes](../deploy/kubernetes/README.md) · [VS Code Extension](vscode-extension.md) · [Troubleshooting](TROUBLESHOOTING.md) · [Development](DEVELOPMENT.md)

---

**On this page:**
- [Core Settings](#core-settings)
- [Embedding Models](#embedding-models)
- [Indexing & Micro-Chunks](#indexing--micro-chunks)
- [Algorithm Optimizations](#algorithm-optimizations)
- [Query Optimization](#query-optimization)
- [Watcher Settings](#watcher-settings)
- [Reranker](#reranker)
- [Learning Reranker](#learning-reranker)
- [Decoder (llama.cpp / GLM / MiniMax)](#decoder-llamacpp--glm--minimax)
- [ReFRAG](#refrag)
- [Ports](#ports)
- [Search & Expansion](#search--expansion)
- [Memory Blending](#memory-blending)

---

## Core Settings

| Name | Description | Default |
|------|-------------|---------|
| COLLECTION_NAME | Qdrant collection name (unified across all repos) | codebase |
| REPO_NAME | Logical repo tag stored in payload for filtering | auto-detect from git/folder |
| HOST_INDEX_PATH | Host path mounted at /work in containers | current repo (.) |
| QDRANT_URL | Qdrant base URL | container: http://qdrant:6333; local: http://localhost:6333 |

## Embedding Models

Context Engine supports multiple embedding models via the `EMBEDDING_MODEL` and `EMBEDDING_PROVIDER` settings.

### Default (BGE-base)

The default configuration uses `BAAI/bge-base-en-v1.5` via fastembed:

| Name | Description | Default |
|------|-------------|---------|
| EMBEDDING_MODEL | Model name for dense embeddings | BAAI/bge-base-en-v1.5 |
| EMBEDDING_PROVIDER | Backend provider | fastembed |

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

## Algorithm Optimizations

Context Engine includes several optional algorithm-based optimizations for improved search relevance and indexing performance. All are opt-in via environment variables.

### Fuzzy Symbol Matching

Jaro-Winkler similarity matching for symbol names. Handles camelCase/snake_case variants and minor typos (e.g., `getUserInfo` matches `get_user_info`).

| Name | Description | Default |
|------|-------------|---------|
| FUZZY_SYMBOL_ENABLED | Enable Jaro-Winkler fuzzy symbol boost | 0 (off) |
| SYMBOL_FUZZY_THRESHOLD | Minimum JW similarity to apply boost (0.0-1.0) | 0.85 |
| SYMBOL_FUZZY_BOOST | Score boost multiplied by JW similarity | 0.1 |

### Aho-Corasick Multi-Pattern Matching

Replaces O(n×m) regex loops with O(n) automaton-based matching for lexical scoring. Beneficial for queries with many terms.

| Name | Description | Default |
|------|-------------|---------|
| AC_LEXICAL_ENABLED | Enable Aho-Corasick for lexical scoring | 0 (off) |

### Bloom Filter Skip

Fast probabilistic check during indexing to skip files that haven't changed. Uses content hashing with near-zero false positives.

| Name | Description | Default |
|------|-------------|---------|
| BLOOM_SKIP_ENABLED | Enable Bloom filter for unchanged file detection | 0 (off) |

### Symbol Diff Detection

Detects moved/renamed code blocks between file versions using difflib. Enables embedding reuse for relocated code.

| Name | Description | Default |
|------|-------------|---------|
| SYMBOL_DIFF_ENABLED | Enable symbol diff for moved code detection | 0 (off) |

### MinHash LSH Deduplication

Locality-sensitive hashing for near-duplicate detection. Identifies similar code blocks across the codebase.

| Name | Description | Default |
|------|-------------|---------|
| DEDUP_MINHASH_ENABLED | Enable MinHash LSH deduplication | 0 (off) |

### HyperLogLog Cardinality

Probabilistic cardinality estimation for collection statistics. Memory-efficient unique path counting.

| Name | Description | Default |
|------|-------------|---------|
| HLL_STATS_ENABLED | Enable HyperLogLog cardinality tracking | 0 (off) |

### Path Trie Filtering

Trie-based data structure for efficient path prefix filtering. Speeds up `under:` filters on large collections.

| Name | Description | Default |
|------|-------------|---------|
| PATH_TRIE_ENABLED | Enable path trie for prefix filtering | 0 (off) |

**Recommended production settings:**
```bash
# .env - Enable all optimizations
FUZZY_SYMBOL_ENABLED=1
AC_LEXICAL_ENABLED=1
BLOOM_SKIP_ENABLED=1
SYMBOL_DIFF_ENABLED=1
DEDUP_MINHASH_ENABLED=1
HLL_STATS_ENABLED=1
PATH_TRIE_ENABLED=1
```

> **See [Architecture](ARCHITECTURE.md#algorithm-optimizations) for implementation details.**

---

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

| Name | Description | Default |
|------|-------------|---------|
| RERANKER_ONNX_PATH | Local ONNX cross-encoder model path | unset |
| RERANKER_TOKENIZER_PATH | Tokenizer path for reranker | unset |
| RERANKER_ENABLED | Enable reranker by default | 1 (enabled) |

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

## Decoder (llama.cpp / GLM / MiniMax)

| Name | Description | Default |
|------|-------------|---------|
| REFRAG_DECODER | Enable decoder for context_answer | 1 (enabled) |
| REFRAG_RUNTIME | Decoder backend: llamacpp, glm, or minimax | llamacpp |
| LLAMACPP_URL | llama.cpp server endpoint | http://llamacpp:8080 or http://host.docker.internal:8081 |
| LLAMACPP_TIMEOUT_SEC | Decoder request timeout | 300 |
| DECODER_MAX_TOKENS | Max tokens for decoder responses | 4000 |
| REFRAG_DECODER_MODE | prompt or soft (soft requires patched llama.cpp) | prompt |
| GLM_API_KEY | API key for GLM provider | unset |
| GLM_MODEL | GLM model name | glm-4.6 |
| GLM_TIMEOUT_SEC | GLM request timeout in seconds | unset |
| MINIMAX_API_KEY | API key for MiniMax M2 provider | unset |
| MINIMAX_MODEL | MiniMax model name | MiniMax-M2 |
| MINIMAX_API_BASE | MiniMax API base URL | https://api.minimax.io/v1 |
| MINIMAX_TIMEOUT_SEC | MiniMax request timeout in seconds | unset |
| USE_GPU_DECODER | Native Metal decoder (1) vs Docker (0) | 0 (docker) |
| LLAMACPP_GPU_LAYERS | Number of layers to offload to GPU, -1 for all | 32 |

## ReFRAG (Micro-Chunking & Retrieval)

| Name | Description | Default |
|------|-------------|---------|
| REFRAG_MODE | Enable micro-chunking and span budgeting | 1 (enabled) |
| REFRAG_GATE_FIRST | Enable mini-vector gating | 1 (enabled) |
| REFRAG_CANDIDATES | Candidates for gate-first filtering | 200 |
| MICRO_BUDGET_TOKENS | Token budget for context_answer | 512 |
| MICRO_OUT_MAX_SPANS | Max spans returned per query | 3 |
| MICRO_CHUNK_TOKENS | Tokens per micro-chunk window | 16 |
| MICRO_CHUNK_STRIDE | Stride between windows | 8 |
| MICRO_MERGE_LINES | Lines to merge adjacent spans | 4 |
| MICRO_TOKENS_PER_LINE | Estimated tokens per line | 32 |

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
| LLM_EXPAND_MAX | Max alternate queries via LLM | 0 |

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

