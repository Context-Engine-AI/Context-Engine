# Development Guide

Setting up development environment, understanding codebase structure, and contributing to Context Engine.

**Documentation:** [README](../README.md) · [Getting Started](GETTING_STARTED.md) · [Configuration](CONFIGURATION.md) · [IDE Clients](IDE_CLIENTS.md) · [MCP API](MCP_API.md) · [ctx CLI](CTX_CLI.md) · [Memory Guide](MEMORY_GUIDE.md) · [Architecture](ARCHITECTURE.md) · [Multi-Repo](MULTI_REPO_COLLECTIONS.md) · [Observability](OBSERVABILITY.md) · [Kubernetes](../deploy/kubernetes/README.md) · [VS Code Extension](vscode-extension.md) · [Troubleshooting](TROUBLESHOOTING.md) · [Development](DEVELOPMENT.md)

---

**On this page:**
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [Docker Development](#docker-development)

---

## Prerequisites

### Required Software
- **Python 3.11+**: Primary development language
- **Docker & Docker Compose**: Containerized development environment
- **Make**: Build automation (recommended)
- **Git**: Version control
- **Node.js & npm**: For MCP development dependencies

### Optional Tools
- **pytest**: Testing framework (included in requirements.txt)
- **pre-commit**: Git hooks for code quality
- **jq**: JSON processing for CLI tools

## Quick Start

### 1. Repository Setup
```bash
# Clone repository
git clone <repository-url>
cd Context-Engine

# Copy environment configuration
cp .env.example .env

# Install Python dependencies
pip install -r requirements.txt
```

### 2. Development Environment
```bash
# Start all services in development mode
make reset-dev-dual

# This starts:
# - Qdrant vector database (ports 6333/6334)
# - Memory MCP server (ports 8000 SSE, 8002 HTTP)
# - Indexer MCP server (ports 8001 SSE, 8003 HTTP)
# - Llama.cpp decoder (port 8080, optional)
```

### 3. Verify Setup
```bash
# Check service health
make health

# Test MCP connectivity
curl http://localhost:8000/sse  # Memory server SSE
curl http://localhost:8001/sse  # Indexer server SSE
```

### 4. IDE MCP Configuration

For MCP-aware IDEs (Claude Code, Windsurf, etc.), prefer the HTTP MCP endpoints:

```bash
# Memory MCP (HTTP)
http://localhost:8002/mcp

# Indexer MCP (HTTP)
http://localhost:8003/mcp

# Health checks
curl http://localhost:18002/readyz  # Memory health
curl http://localhost:18003/readyz  # Indexer health
```

SSE endpoints (`/sse`) remain available and are typically used via `mcp-remote`, but some clients that send MCP messages in parallel on a fresh session can hit a FastMCP initialization guard and intermittently log:

```text
Failed to validate request: Received request before initialization was complete
```

If you see tools/resources only appearing after a second reconnect, switch your IDE configuration to use the HTTP `/mcp` endpoints instead of SSE.

## Project Structure

```
Context-Engine/
├── scripts/                    # Core application code
│   ├── mcp_memory_server.py   # Memory MCP server implementation
│   ├── mcp_indexer_server.py  # Indexer MCP server implementation
│   ├── mcp_router.py          # Intent-based tool routing
│   ├── hybrid_search.py       # Search algorithm implementation
│   ├── ctx.py                 # CLI prompt enhancer
│   ├── cache_manager.py       # Unified caching system
│   ├── async_subprocess_manager.py  # Process management
│   ├── deduplication.py       # Request deduplication
│   ├── semantic_expansion.py  # Query expansion
│   ├── collection_health.py   # Cache/collection sync checks
│   ├── utils.py               # Shared utilities
│   ├── ingest_code.py         # Code indexing logic
│   ├── watch_index.py         # File system watcher
│   ├── upload_service.py      # Remote upload HTTP service
│   ├── remote_upload_client.py # Remote sync client
│   ├── memory_backup.py       # Memory export
│   ├── memory_restore.py      # Memory import
│   └── logger.py              # Structured logging
├── tests/                     # Test suite
│   ├── conftest.py           # Test configuration
│   ├── test_*.py            # Unit and integration tests
│   └── integration/          # Integration test helpers
├── docker/                    # Docker configurations
│   ├── Dockerfile.mcp        # Memory server image
│   ├── Dockerfile.mcp-indexer  # Indexer server image
│   └── scripts/              # Docker build scripts
├── docs/                      # Documentation
├── .env.example              # Environment template
├── docker-compose.yml        # Development environment
├── docker-compose.override.yml  # Development overrides
├── Makefile                  # Development commands
├── requirements.txt          # Python dependencies
└── README.md                 # Project overview
```

## Development Workflow

### Making Changes

1. **Create a feature branch**:
```bash
git checkout -b feature/your-feature-name
```

2. **Make your changes** following the coding standards outlined below.

3. **Run tests**:
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_hybrid_search.py

# Run with coverage
pytest --cov=scripts --cov-report=html
```

4. **Test changes in development environment**:
```bash
# Restart services with changes
docker-compose restart mcp mcp_indexer

# Or rebuild if changes affect Docker images
make rebuild-dev
```

5. **Commit changes**:
```bash
git add .
git commit -m "feat: add your feature description"
```

### Code Quality Standards

#### Python Style Guide
- **PEP 8 compliant**: Use standard Python formatting
- **Type hints**: Include type annotations for all public functions
- **Docstrings**: Google-style docstrings for all modules and public functions
- **Error handling**: Use structured error types from `scripts.logger`

#### Example Function:
```python
from typing import List, Dict, Any, Optional
from scripts.logger import get_logger, RetrievalError

logger = get_logger(__name__)

def search_code(
    query: str,
    limit: int = 10,
    filters: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """Search code using hybrid retrieval.

    Args:
        query: Search query string
        limit: Maximum number of results to return
        filters: Optional search filters

    Returns:
        List of search results with scores and metadata

    Raises:
        RetrievalError: If search operation fails
    """
    try:
        # Implementation
        pass
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise RetrievalError(f"Search operation failed: {e}") from e
```

## Adding New Features

### 1. Adding a New MCP Tool

Create a new tool in the appropriate MCP server file:

```python
# In scripts/mcp_indexer_server.py or scripts/mcp_memory_server.py

@mcp.tool()
async def my_new_tool(
    param1: str,
    param2: Optional[int] = None,
    param3: str = ""
) -> Dict[str, Any]:
    """Brief description of what this tool does.

    Args:
        param1: Description of required parameter
        param2: Description of optional parameter
        param3: Description of parameter with default

    Returns:
        Dictionary containing operation result

    Example:
        result = await my_new_tool("test", 42)
        print(result["ok"])  # True
    """
    try:
        # Validate inputs
        if not param1.strip():
            raise ValidationError("param1 cannot be empty")

        # Implementation logic
        result = do_something(param1, param2 or 0)

        return {
            "ok": True,
            "result": result,
            "message": "Operation completed successfully"
        }

    except Exception as e:
        logger.error(f"my_new_tool failed: {e}")
        return {
            "ok": False,
            "error": str(e),
            "message": "Operation failed"
        }
```

### 2. Adding New Search Filters

Extend the hybrid search system with new filtering capabilities:

```python
# In scripts/hybrid_search.py

def apply_my_filter(
    results: List[Dict[str, Any]],
    filter_value: str
) -> List[Dict[str, Any]]:
    """Apply custom filter to search results.

    Args:
        results: List of search results to filter
        filter_value: Filter criteria

    Returns:
        Filtered list of results
    """
    filtered_results = []
    for result in results:
        if matches_my_criteria(result, filter_value):
            filtered_results.append(result)
    return filtered_results

# Update the main search function
def hybrid_search(
    queries: Union[str, List[str]],
    # ... existing parameters ...
    my_filter: Optional[str] = None
) -> List[Dict[str, Any]]:
    # ... existing search logic ...

    # Apply new filter
    if my_filter:
        results = apply_my_filter(results, my_filter)

    return results
```

### 3. Adding New Embedding Models

1. **Update model mapping** in `scripts/utils.py`:
```python
def sanitize_vector_name(model_name: str) -> str:
    name = (model_name or "").strip().lower()

    # Add your model mapping
    if "your-new-model" in name:
        return "your-new-model-alias"

    # ... existing mappings ...
```

2. **Test with new model**:
```python
# In tests/test_embedding.py
def test_new_embedding_model():
    from scripts.utils import sanitize_vector_name

    assert sanitize_vector_name("your-new-model-v1") == "your-new-model-alias"
```

3. **Update Docker images** if the model requires additional dependencies.

## Testing

### Test Organization

#### Unit Tests (`tests/test_*.py`)
- Test individual functions and classes
- Mock external dependencies (Qdrant, embedding models)
- Fast execution, no external services required

#### Integration Tests (`tests/test_integration_*.py`)
- Test component interactions
- Use real Qdrant via testcontainers
- Slower but more realistic testing

### Writing Tests

#### Test Structure
```python
import pytest
from unittest.mock import Mock, patch
from scripts.hybrid_search import hybrid_search

class TestHybridSearch:
    @pytest.fixture
    def fake_embedder(self):
        """Mock embedding model for deterministic tests."""
        embedder = Mock()
        embedder.embed.return_value = [[0.1, 0.2, 0.3]]
        return embedder

    @pytest.fixture
    def mock_qdrant(self):
        """Mock Qdrant client."""
        client = Mock()
        client.search.return_value = [
            {"id": "1", "score": 0.9, "payload": {"text": "test"}}
        ]
        return client

    def test_basic_search(self, fake_embedder, mock_qdrant):
        """Test basic hybrid search functionality."""
        results = hybrid_search(
            queries=["test query"],
            embedder=fake_embedder,
            qdrant_client=mock_qdrant
        )

        assert len(results) > 0
        assert all("score" in r for r in results)
        assert all(0 <= r["score"] <= 1 for r in results)

    def test_search_with_filters(self, fake_embedder, mock_qdrant):
        """Test search with language and path filters."""
        results = hybrid_search(
            queries=["test query"],
            filters={"language": "python", "path": "src/"},
            embedder=fake_embedder,
            qdrant_client=mock_qdrant
        )

        # Verify filter application
        mock_qdrant.search.assert_called_once()
        call_args = mock_qdrant.search.call_args
        assert "filter" in call_args.kwargs
```

#### Integration Tests
```python
import pytest
from testcontainers.core.container import DockerContainer

@pytest.mark.integration
class TestSearchIntegration:
    @pytest.fixture(scope="module")
    def qdrant_container(self):
        """Set up real Qdrant container for integration tests."""
        container = DockerContainer("qdrant/qdrant:latest").with_exposed_ports(6333)
        container.start()
        yield f"http://{container.get_container_host_ip()}:{container.get_exposed_port(6333)}"
        container.stop()

    def test_end_to_end_search(self, qdrant_container):
        """Test complete search pipeline with real services."""
        # Set up test data
        # Perform search
        # Verify results
        pass
```

### Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run unit tests only (exclude integration)
pytest -m "not integration"

# Run integration tests only
pytest -m integration

# Run specific test file
pytest tests/test_hybrid_search.py

# Run with coverage report
pytest --cov=scripts --cov-report=html

# Run with specific markers
pytest -m "not slow"  # Skip slow tests
```

## Debugging

### Development Debugging

#### Enable Debug Logging
```bash
# Set debug environment variables
export DEBUG_CONTEXT_ANSWER=1
export HYBRID_DEBUG=1
export CACHE_DEBUG=1

# Restart services
docker-compose restart
```

#### Local Development
```bash
# Run MCP servers directly for easier debugging
python scripts/mcp_indexer_server.py

# Run with debugger
python -m pdb scripts/hybrid_search.py
```

### Common Debugging Scenarios

#### Search Issues
```python
# Enable detailed search logging
import logging
logging.getLogger("hybrid_search").setLevel(logging.DEBUG)

# Check search intermediate results
def debug_search(query):
    # Check embedding generation
    embeddings = embed_model.embed([query])
    print(f"Embedding shape: {len(list(embeddings)[0])}")

    # Check Qdrant query
    results = qdrant_client.search(...)
    print(f"Qdrant results: {len(results)}")

    return results
```

#### Cache Issues
```python
# Check cache statistics
from scripts.cache_manager import get_search_cache

cache = get_search_cache()
print(f"Cache stats: {cache.get_stats()}")
print(f"Cache size: {len(cache._cache)}")
```

## Performance Profiling

### Profiling Tools
```bash
# Profile with cProfile
python -m cProfile -o profile.stats scripts/hybrid_search.py

# Analyze profile results
python -c "
import pstats
p = pstats.Stats('profile.stats')
p.sort_stats('cumulative')
p.print_stats(20)
"
```

### Memory Profiling
```bash
# Install memory_profiler
pip install memory-profiler

# Profile memory usage
python -m memory_profiler scripts/hybrid_search.py
```

## Common Development Issues

### Environment Setup Issues
```bash
# Python path issues
export PYTHONPATH="${PYTHONPATH}:/path/to/Context-Engine"

# Docker issues
docker system prune -f  # Clean up Docker
docker-compose down -v  # Remove volumes
docker-compose up --build  # Rebuild images
```

### Import Issues
```bash
# Ensure code roots are on sys.path
export WORK_ROOTS="/path/to/Context-Engine,/app"

# Check Python path
python -c "import sys; print(sys.path)"
```

### MCP Server Issues
```bash
# Check MCP server connectivity
curl -H "Accept: text/event-stream" http://localhost:8000/sse

# Check MCP tools available
curl http://localhost:18001/tools
```

## Contributing Guidelines

### Before Submitting Changes
1. **Run full test suite**: `pytest`
2. **Check code style**: Use `black` and `flake8`
3. **Update documentation**: Add docstrings for new functions
4. **Test locally**: Verify changes work in development environment

### Pull Request Process
1. **Create descriptive PR title**: "feat: add new search filter"
2. **Provide detailed description**: Explain what changes and why
3. **Include test coverage**: Add tests for new functionality
4. **Update documentation**: Include API changes in docs

### Code Review Checklist
- [ ] Code follows style guidelines
- [ ] Tests pass for all changes
- [ ] Documentation is updated
- [ ] No hardcoded secrets or values
- [ ] Error handling is appropriate
- [ ] Performance impact is considered

This development guide should help you get started with contributing to Context Engine. For more specific questions, refer to the code documentation or create an issue in the repository.