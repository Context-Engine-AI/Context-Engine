import os
import sys
import time
import urllib.request
from pathlib import Path

import pytest

# Ensure repository root is on sys.path so `import scripts...` works locally
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Enable pattern vectors for pattern search tests
os.environ.setdefault("PATTERN_VECTORS", "1")


@pytest.fixture(scope="session", autouse=True)
def _ensure_mcp_imported():
    """Ensure mcp package is properly imported before any tests run.

    This prevents import conflicts when scripts.mcp_indexer_server imports
    from mcp.server.fastmcp and later tests try to import fastmcp.
    """
    try:
        import mcp.types  # noqa: F401
    except ImportError:
        pass  # mcp package not available, tests will skip if needed
    yield


@pytest.fixture(scope="session", autouse=True)
def _disable_tokenizers_parallelism():
    """Force tokenizers to stay single-threaded to avoid fork warnings during tests."""
    prev = os.environ.get("TOKENIZERS_PARALLELISM")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop("TOKENIZERS_PARALLELISM", None)
        else:
            os.environ["TOKENIZERS_PARALLELISM"] = prev


def _wait_for_qdrant(url: str, timeout: int = 60) -> bool:
    """Wait for Qdrant to be ready at the given URL."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url + "/readyz", timeout=2) as r:
                if 200 <= r.status < 300:
                    return True
        except Exception:
            pass
        time.sleep(1)
    return False


@pytest.fixture(scope="module")
def qdrant_url():
    """Provide Qdrant URL - uses CI service container or testcontainers (local).

    In CI (GitHub Actions), uses the pre-configured Qdrant service at localhost:6333.
    Locally, this fixture ALWAYS spins up a testcontainers Qdrant instance to avoid
    accidentally polluting a developer's local Qdrant with test collections.
    """
    # Only use pre-configured Qdrant in CI environment (GitHub Actions sets CI=true)
    is_ci = os.environ.get("CI", "").lower() in ("true", "1", "yes")
    if is_ci:
        # Hardcoded CI Qdrant URL - the GitHub Actions service container
        ci_url = "http://localhost:6333"
        if _wait_for_qdrant(ci_url, timeout=30):
            yield ci_url
            return
        # If not reachable in CI, fail explicitly
        pytest.fail(f"CI Qdrant service not reachable at {ci_url}")

    # Local development: ALWAYS use testcontainers (safe isolation)
    os.environ.setdefault("TESTCONTAINERS_RYUK_DISABLED", "true")
    os.environ.setdefault("TESTCONTAINERS_RYUK_TIMEOUT", "0")

    try:
        from testcontainers.core.container import DockerContainer
    except ImportError:
        pytest.skip("testcontainers not available and QDRANT_URL not set")

    container = (
        DockerContainer("qdrant/qdrant:latest")
        .with_env("TESTCONTAINERS_RYUK_DISABLED", "true")
        .with_env("TESTCONTAINERS_RYUK_TIMEOUT", "0")
        .with_exposed_ports(6333)
    )

    try:
        container.start()
        host = container.get_container_host_ip()

        # Wait for port mapping
        deadline = time.time() + 30
        port = None
        last_exc = None
        while time.time() < deadline:
            try:
                port = int(container.get_exposed_port(6333))
                break
            except Exception as exc:
                last_exc = exc
                time.sleep(0.25)

        if port is None:
            raise RuntimeError(f"qdrant port mapping unavailable: {last_exc}")

        url = f"http://{host}:{port}"

        if not _wait_for_qdrant(url, timeout=60):
            pytest.skip("Qdrant not ready in time")

        yield url
    finally:
        try:
            container.stop()
        except Exception:
            pass


# Alias for backward compatibility with existing tests
@pytest.fixture(scope="module")
def qdrant_container(qdrant_url):
    """Backward-compatible alias for qdrant_url fixture."""
    return qdrant_url


# Track collections created during tests for cleanup
_test_collections: list[tuple[str, str]] = []  # (qdrant_url, collection_name)


@pytest.fixture
def test_collection(qdrant_url):
    """Create a unique test collection name and clean it up after the test.

    Usage:
        def test_something(qdrant_url, test_collection):
            # test_collection is a unique name like "test-a1b2c3d4"
            # It will be automatically deleted after the test
    """
    import uuid
    collection_name = f"test-{uuid.uuid4().hex[:8]}"
    _test_collections.append((qdrant_url, collection_name))
    yield collection_name

    # Clean up THIS test's collection immediately after the test
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(url=qdrant_url, timeout=10)
        client.delete_collection(collection_name)
    except Exception:
        pass  # Collection might not exist or already deleted

    # Remove from tracking list
    _test_collections[:] = [(u, n) for u, n in _test_collections
                             if not (u == qdrant_url and n == collection_name)]


@pytest.fixture(scope="session", autouse=True)
def _final_cleanup_all_test_collections():
    """Final cleanup at end of test session - delete any remaining test-* collections."""
    yield

    # Only run cleanup in CI to avoid touching local Qdrant
    is_ci = os.environ.get("CI", "").lower() in ("true", "1", "yes")
    if not is_ci:
        return

    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(url="http://localhost:6333", timeout=10)

        # List all collections and delete ones starting with "test-"
        collections = client.get_collections().collections
        for col in collections:
            if col.name.startswith("test-"):
                try:
                    client.delete_collection(col.name)
                except Exception:
                    pass
    except Exception:
        pass  # Best effort cleanup
