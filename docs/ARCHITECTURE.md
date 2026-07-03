# Context Engine Architecture

**Documentation:** [README](../README.md) · [Getting Started](GETTING_STARTED.md) · [Configuration](CONFIGURATION.md) · [IDE Clients](IDE_CLIENTS.md) · [MCP API](MCP_API.md) · [ctx CLI](CTX_CLI.md) · [Memory Guide](MEMORY_GUIDE.md) · [Architecture](ARCHITECTURE.md) · [Multi-Repo](MULTI_REPO_COLLECTIONS.md) · [Observability](OBSERVABILITY.md) · [Kubernetes](../deploy/kubernetes/README.md) · [VS Code Extension](vscode-extension.md) · [Troubleshooting](TROUBLESHOOTING.md) · [Development](DEVELOPMENT.md)

---

**On this page:**
- [Overview](#overview)
- [Core Principles](#core-principles)
- [System Architecture](#system-architecture)
- [Learning Reranker System](#5-learning-reranker-system)
- [Data Flow](#data-flow)
- [ReFRAG Pipeline](#refrag-pipeline)

---

## Overview

Production-ready MCP (Model Context Protocol) retrieval stack unifying code indexing, hybrid search, and optional LLM decoding. Enables teams to ship context-aware AI agents with semantic and lexical search capabilities and dual-transport compatibility.

## Core Principles

- **Research-Grade Retrieval**: ReFRAG-inspired micro-chunking and span budgeting
- **Dual-Transport Support**: SSE (legacy) and HTTP RMCP (modern) protocols
- **Performance-First**: Intelligent caching, connection pooling, and async I/O
- **Production-Ready**: Comprehensive health checks, monitoring, and operational tooling

## System Architecture

### Component Diagram

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Client Apps   │◄──►│  MCP Servers    │◄──►│   Qdrant DB     │
│ (IDE, CLI, Web) │    │  (SSE + HTTP)   │    │  (Vector Store) │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │  LLM Decoder    │
                       │  (llama.cpp)    │
                       │   (Optional)    │
                       └─────────────────┘
```

## Core Components

### 1. MCP Servers

#### Memory Server (`scripts/mcp_memory_server.py`)
- **Purpose**: Knowledge base storage and retrieval
- **Transport**: SSE (port 8000) + HTTP RMCP (port 8002)
- **Key Features**:
  - Structured memory storage with rich metadata
  - Hybrid search (dense + lexical)
  - Dual vector support for embedding and lexical hashes
  - Automatic collection management

#### Indexer Server (`scripts/mcp_indexer_server.py`)
- **Purpose**: Code search, indexing, and management
- **Transport**: SSE (port 8001) + HTTP RMCP (port 8003)
- **Key Features**:
  - Hybrid code search with multiple filtering options
  - ReFRAG-inspired micro-chunking (16-token windows)
  - Context-aware Q&A with local LLM integration
  - Workspace and collection management
  - Live indexing and pruning capabilities

### 2. Search Pipeline

#### Hybrid Search Engine (`scripts/hybrid_search.py`)
- **Multi-Vector Architecture**:
  - **Dense Vectors**: Semantic embeddings (BAAI/bge-base-en-v1.5)
  - **Lexical Vectors**: BM25-style hashing (4096 dimensions)
  - **Mini Vectors**: ReFRAG gating (64 dimensions, optional)

- **Retrieval Process**:
  1. **Query Expansion**: Generate multiple query variations
  2. **Parallel Search**: Dense + lexical search with RRF fusion
  3. **Optional Reranking**: Cross-encoder neural reranking
  4. **Result Assembly**: Format with citations and metadata

- **Advanced Features**:
  - Request deduplication
  - Intelligent caching (multi-policy: LRU, LFU, TTL, FIFO)
  - Connection pooling to Qdrant
  - Batch processing support

#### ReFRAG Implementation
- **Micro-chunking**: Token-level windows (16 tokens, 8 stride)
- **Span Budgeting**: Global token budget management
- **Gate-First Filtering**: Mini-vector pre-filtering for efficiency

### 3. Storage Layer

#### Qdrant Vector Database
- **Primary Storage**: Embeddings and metadata
- **Collection Management**: Automatic creation and configuration
- **Named Vectors**: Separate storage for different embedding types
- **Performance**: HNSW indexing for fast approximate nearest neighbor search

#### Unified Cache System (`scripts/cache_manager.py`)
- **Eviction Policies**: LRU, LFU, TTL, FIFO
- **Memory Management**: Configurable size limits and monitoring
- **Thread Safety**: Proper locking for concurrent access
- **Statistics Tracking**: Hit rates, memory usage, eviction counts

### 4. Supporting Infrastructure

#### Async Subprocess Manager (`scripts/async_subprocess_manager.py`)
- **Process Management**: Async subprocess execution with resource cleanup
- **Connection Pooling**: Reused HTTP connections
- **Timeout Handling**: Configurable timeouts with graceful degradation
- **Resource Tracking**: Active process monitoring and statistics

#### Deduplication System (`scripts/deduplication.py`)
- **Request Deduplication**: Prevent redundant processing
- **Cache Integration**: Works with unified cache system
- **Performance Impact**: Significant reduction in duplicate work

#### Semantic Expansion (`scripts/semantic_expansion.py`)
- **Query Enhancement**: LLM-assisted query variation generation
- **Local LLM Integration**: llama.cpp for offline expansion
- **Caching**: Expanded query results cached for reuse

#### Pattern Detection (`scripts/pattern_detection/`)
- **Structural Search**: Find similar code patterns across languages via AST analysis
- **64-dim Pattern Vector**: WL graph kernel, CFG fingerprint, SimHash, spectral features
- **Auto-Detection**: Identifies retry patterns, resource cleanup, filter loops
- **Requires**: `PATTERN_VECTORS=1` to enable

### 5. Learning Reranker System (Optional)

The Learning Reranker is an **optional** self-improving ranking system that learns from search patterns to provide increasingly relevant results over time. It is enabled by default but can be disabled via `RERANK_LEARNING=0` and `RERANK_EVENTS_ENABLED=0` environment variables. See [Configuration](CONFIGURATION.md#learning-reranker) for all options.

#### Architecture Overview

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Search Query   │────►│  Hybrid Search   │────►│  TinyScorer     │
│                 │     │  (initial rank)  │     │  (learned rank) │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                         │
                        ┌──────────────────┐             │
                        │  Event Logger    │◄────────────┘
                        │  (NDJSON files)  │
                        └────────┬─────────┘
                                 │
                        ┌────────▼─────────┐
                        │ Learning Worker  │
                        │  (background)    │
                        └────────┬─────────┘
                                 │
                        ┌────────▼─────────┐
                        │  ONNX Teacher    │
                        │ (cross-encoder)  │
                        └────────┬─────────┘
                                 │
                        ┌────────▼─────────┐
                        │  Weight Updates  │
                        │  (.npz files)    │
                        └──────────────────┘
```

#### Components

**TinyScorer** (`scripts/rerank_recursive.py`)
- 2-layer MLP neural network (~3MB per collection)
- Scores query-document pairs based on learned patterns
- Hot-reloads weights every 60 seconds from disk
- Per-collection weights (each repo learns independently)

**Event Logger** (`scripts/rerank_events.py`)
- Logs every search to NDJSON files at `/tmp/rerank_events/`
- Records: query, candidates, initial scores, timestamps
- Hourly file rotation with configurable retention

**Learning Worker** (`scripts/learning_reranker_worker.py`)
- Background daemon that processes logged events
- Uses ONNX cross-encoder as "teacher" model
- Trains TinyScorer via knowledge distillation
- Saves versioned weight checkpoints atomically

#### Learning Flow

1. **Event Capture**: Every search logs query + candidates to NDJSON
2. **Teacher Scoring**: ONNX cross-encoder scores the candidates
3. **Student Training**: TinyScorer learns to match teacher rankings
4. **Weight Update**: New weights saved atomically with versioning
5. **Hot Reload**: Serving path picks up new weights within 60s
6. **Score Integration**: `learning_score` blends with other signals

#### Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `RERANKER_WEIGHTS_DIR` | Directory for weight files | `/tmp/rerank_weights` |
| `RERANKER_WEIGHTS_RELOAD_INTERVAL` | Hot-reload check interval (seconds) | 60 |
| `RERANKER_MAX_CHECKPOINTS` | Number of weight versions to keep | 5 |
| `RERANKER_LR_DECAY_STEPS` | Steps between learning rate decay | 1000 |
| `RERANKER_LR_DECAY_RATE` | Learning rate decay multiplier | 0.95 |
| `RERANKER_MIN_LR` | Minimum learning rate | 0.0001 |
| `RERANK_EVENTS_DIR` | Directory for event logs | `/tmp/rerank_events` |
| `RERANK_EVENTS_RETENTION_DAYS` | Days to keep event files | 7 |
| `RERANK_LEARNING_BATCH_SIZE` | Events per training batch | 32 |
| `RERANK_LEARNING_POLL_INTERVAL` | Worker poll interval (seconds) | 30 |
| `RERANK_LEARNING_RATE` | Initial learning rate | 0.001 |

#### Observability

Search results include learning metrics in the `why` field:
```json
{
  "score": 3.2,
  "why": ["lexical:1.0", "dense_rrf:0.05", "learning:3", "score:3.2"],
  "components": {
    "learning_score": 3.2,
    "learning_iterations": 3
  }
}
```

Worker logs show training progress:
```
[codebase] Processed 5 events | v12 | lr=0.001 | avg_loss=1.8 | converged=False
```

#### Benefits

- **Zero Manual Training**: Learns automatically from usage
- **Per-Collection Specialization**: Each codebase gets tuned rankings
- **Fast Inference**: TinyScorer adds <1ms to search latency
- **Continuous Improvement**: Rankings improve over time
- **Offline Capable**: Teacher runs locally, no external API calls

#### MCP Router (`scripts/mcp_router.py`)
- **Intent Classification**: Determines which MCP tool to call based on query
- **Tool Orchestration**: Routes to search, answer, memory, or index tools
- **HTTP Execution**: Executes tools via RMCP/HTTP without extra dependencies
- **Plan Mode**: Preview tool selection without execution

## Data Flow Architecture

### Search Request Flow
```
1. Client Query → MCP Server
2. Query Expansion (optional) → Multiple Query Variations
3. Parallel Execution → Dense Search + Lexical Search
4. RRF Fusion → Combined Results
5. Reranking (optional) → Enhanced Relevance
6. Result Formatting → Structured Response with Citations
7. Return to Client → MCP Protocol Response
```

### Indexing Flow
```
1. File Change Detection → File System Watcher
2. Content Processing → Tokenization + Chunking
3. Embedding Generation → Model Inference
4. Vector Creation → Dense + Lexical + Mini
5. Metadata Assembly → Path, symbols, language, etc.
6. Batch Upsert → Qdrant Storage
7. Cache Updates → Local Cache Refresh
```

## Configuration Architecture

### Environment-Based Configuration
- **Docker-Native**: All configuration via environment variables
- **Development Support**: Local .env file configuration
- **Production Ready**: External secret management integration

### Key Configuration Areas
- **Service Configuration**: Ports, hosts, transport protocols
- **Model Configuration**: Embedding models, reranker settings
- **Performance Tuning**: Cache sizes, batch sizes, timeouts
- **Feature Flags**: Experimental features, debug modes

## Transport Layer Architecture

### Dual-Transport Design
- **SSE (Server-Sent Events)**: Legacy client compatibility
- **HTTP RMCP**: Modern JSON-RPC over HTTP
- **Simultaneous Operation**: Both protocols can run together
- **Automatic Fallback**: Graceful degradation when transport fails

### MCP Protocol Implementation
- **FastMCP Framework**: Modern MCP server implementation
- **Tool Registry**: Automatic tool discovery and registration
- **Health Endpoints**: `/readyz` and `/tools` endpoints
- **Error Handling**: Structured error responses and logging

## Performance Architecture

### Caching Strategy
- **Multi-Level Caching**: Embedding cache, search cache, expansion cache
- **Intelligent Invalidation**: TTL-based and LRU eviction
- **Memory Management**: Configurable limits and monitoring
- **Performance Monitoring**: Hit rates, response times, memory usage

### Concurrency Model
- **Async I/O**: Non-blocking operations throughout
- **Connection Pooling**: Reused connections to external services
- **Batch Processing**: Efficient bulk operations
- **Resource Management**: Proper cleanup and resource limits

## Security Architecture

### Isolation and Safety
- **Container-Based**: Docker isolation for all services
- **Network Segmentation**: Internal service communication
- **Input Validation**: Comprehensive parameter validation
- **Resource Limits**: Configurable timeouts and memory limits

### Data Protection
- **No Hardcoded Secrets**: Environment-based configuration
- **API Key Management**: External secret manager integration
- **Audit Logging**: Structured logging for security events

## Operational Architecture

### Health Monitoring
- **Service Health**: `/readyz` endpoints for all services
- **Tool Availability**: Dynamic tool listing and status
- **Performance Metrics**: Response times, cache statistics
- **Error Tracking**: Structured error logging and alerting

### Deployment Patterns
- **Docker Compose**: Multi-service orchestration
- **Environment Parity**: Development ↔ Production consistency
- **Graceful Shutdown**: Proper resource cleanup on termination
- **Rolling Updates**: Zero-downtime deployment support

## Extensibility Architecture

### Plugin System
- **MCP Tool Extension**: Easy addition of new tools
- **Transport Flexibility**: Support for future MCP transports
- **Model Pluggability**: Support for different embedding models
- **Storage Abstraction**: Potential for alternative vector stores

### Configuration Extension
- **Environment-Driven**: Easy configuration via environment variables
- **Feature Flags**: Experimental feature toggling
- **A/B Testing**: Multiple configuration variants support

This architecture enables Context Engine to serve as a production-ready, scalable context layer for AI applications while maintaining the flexibility to evolve with changing requirements and technologies.