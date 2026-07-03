"""Tests for pattern_search functionality.

Tests that:
1. Pattern extraction produces consistent signatures
2. Cross-language patterns have high similarity
3. Code vs NL detection works correctly
4. Integration with Qdrant (when available)
"""
import os
import uuid
import time
import pytest
import numpy as np

# Test fixtures
RETRY_PATTERN_PYTHON = '''
for attempt in range(3):
    try:
        result = make_request()
        break
    except Exception:
        time.sleep(2 ** attempt)
'''

RETRY_PATTERN_GO = '''
for i := 0; i < 3; i++ {
    result, err := makeRequest()
    if err == nil {
        break
    }
    time.Sleep(time.Duration(1<<i) * time.Second)
}
'''

FILE_READ_PATTERN = '''
with open(filename) as f:
    data = json.load(f)
'''


# ============================================================================
# Unit tests (no Qdrant required)
# ============================================================================

def test_cross_language_pattern_similarity():
    """Prove that Python retry pattern is structurally similar to Go retry."""
    from scripts.pattern_detection import PatternExtractor, PatternEncoder

    extractor = PatternExtractor()
    encoder = PatternEncoder()

    py_sig = extractor.extract(RETRY_PATTERN_PYTHON, "python")
    go_sig = extractor.extract(RETRY_PATTERN_GO, "go")
    file_sig = extractor.extract(FILE_READ_PATTERN, "python")

    py_vec = encoder.encode(py_sig)
    go_vec = encoder.encode(go_sig)
    file_vec = encoder.encode(file_sig)

    def cosine_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)

    py_go_sim = cosine_sim(py_vec, go_vec)
    py_file_sim = cosine_sim(py_vec, file_vec)

    assert py_go_sim > py_file_sim, (
        f"Retry patterns should be more similar ({py_go_sim:.3f}) "
        f"than retry vs file ({py_file_sim:.3f})"
    )
    assert py_go_sim > 0.3, f"Cross-language retry similarity too low: {py_go_sim:.3f}"


def test_pattern_vector_dimensions():
    """Test that pattern vectors have correct dimensions."""
    from scripts.pattern_detection import PatternExtractor, PatternEncoder

    extractor = PatternExtractor()
    encoder = PatternEncoder()

    sig = extractor.extract(RETRY_PATTERN_PYTHON, "python")
    vec = encoder.encode(sig)

    assert len(vec) == 64, f"Expected 64-dim vector, got {len(vec)}"
    assert all(isinstance(v, float) for v in vec), "All values should be floats"


def test_code_vs_nl_detection():
    """Test auto-detection of code vs natural language queries."""
    from scripts.mcp_impl.pattern_search import _detect_query_mode

    # Code examples (no language hint, should detect from syntax)
    assert _detect_query_mode("for i in range(3): try: pass except: pass", None) == "code"
    assert _detect_query_mode("if err != nil { return err }", None) == "code"
    assert _detect_query_mode("def foo(): return 42", None) == "code"
    assert _detect_query_mode("func main() {}", None) == "code"
    assert _detect_query_mode("fn main() -> Result<()>", None) == "code"

    # Natural language descriptions
    assert _detect_query_mode("retry with exponential backoff", None) == "description"
    assert _detect_query_mode("find error handling patterns", None) == "description"
    assert _detect_query_mode("resource cleanup code", None) == "description"
    assert _detect_query_mode("decorator pattern wrapping function", None) == "description"

    # Ambiguous two-word input is treated as code when a language hint is present
    assert _detect_query_mode("some text", "python") == "code"
    assert _detect_query_mode("some text", "go") == "code"


# ============================================================================
# Integration tests (require Qdrant service)
# ============================================================================

@pytest.fixture
def pattern_collection():
    """Create collection with pattern vectors and test data."""
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    from scripts.pattern_detection import PatternExtractor, PatternEncoder

    # Default to localhost for CI; allow override for local/dev containers.
    collection_name = f"test_pattern_{uuid.uuid4().hex[:8]}"
    qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    client = QdrantClient(url=qdrant_url, timeout=30)

    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "code": VectorParams(size=384, distance=Distance.COSINE),
                "pattern_vector": VectorParams(size=64, distance=Distance.COSINE),
            }
        )
    except Exception as e:
        pytest.skip(f"Qdrant not reachable at {qdrant_url}: {e}")

    extractor = PatternExtractor()
    encoder = PatternEncoder()

    snippets = [
        {"path": "retry_python.py", "code": RETRY_PATTERN_PYTHON, "lang": "python"},
        {"path": "retry_go.go", "code": RETRY_PATTERN_GO, "lang": "go"},
        {"path": "file_read.py", "code": FILE_READ_PATTERN, "lang": "python"},
    ]

    points = []
    for i, s in enumerate(snippets):
        sig = extractor.extract(s["code"], s["lang"])
        points.append(PointStruct(
            id=i + 1,
            vector={"code": [0.1] * 384, "pattern_vector": encoder.encode(sig)},
            payload={"path": s["path"], "language": s["lang"], "text": s["code"]},
        ))

    client.upsert(collection_name=collection_name, points=points)

    yield collection_name, client

    try:
        client.delete_collection(collection_name)
    except Exception:
        pass


@pytest.mark.service
def test_pattern_search_qdrant(pattern_collection):
    """Test pattern search against real Qdrant."""
    from scripts.pattern_detection.search import pattern_search

    collection_name, client = pattern_collection

    results = pattern_search(
        example=RETRY_PATTERN_PYTHON,
        language="python",
        collection=collection_name,
        limit=5,
        client=client,  # Pass client to avoid global cache pollution
    )

    assert results.total >= 1, "Should find at least one result"
    paths = [r.path for r in results.results]
    assert "retry_python.py" in paths, "Should find Python retry"
