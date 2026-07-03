"""
Common types and utilities for Context-Engine benchmarks.

Provides unified reporting format and proper statistical functions.
"""

import json
import statistics
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional


# ---------------------------------------------------------------------------
# Proper Percentile Calculation
# ---------------------------------------------------------------------------
def percentile(values: List[float], p: float) -> float:
    """
    Compute the p-th percentile of a list of values.
    
    Args:
        values: List of numeric values
        p: Percentile to compute (0.0 to 1.0)
    
    Returns:
        The p-th percentile value, or 0.0 if list is empty
    """
    if not values:
        return 0.0
    sorted_values = sorted(values)
    n = len(sorted_values)
    
    # Linear interpolation method (same as numpy default)
    idx = p * (n - 1)
    lower = int(idx)
    upper = min(lower + 1, n - 1)
    weight = idx - lower
    
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def compute_percentiles(values: List[float]) -> Dict[str, float]:
    """Compute standard percentiles (p50, p90, p95, p99)."""
    stddev = round(statistics.stdev(values), 2) if len(values) > 1 else 0.0
    return {
        "p50": round(percentile(values, 0.50), 2),
        "p90": round(percentile(values, 0.90), 2),
        "p95": round(percentile(values, 0.95), 2),
        "p99": round(percentile(values, 0.99), 2),
        "min": round(min(values), 2) if values else 0.0,
        "max": round(max(values), 2) if values else 0.0,
        "mean": round(statistics.mean(values), 2) if values else 0.0,
        "stddev": stddev,
    }


# ---------------------------------------------------------------------------
# Result Parsing Helpers
# ---------------------------------------------------------------------------
def extract_result_paths(result: Any, limit: Optional[int] = None) -> List[str]:
    """
    Extract file paths from MCP tool results.

    Supports:
    - JSON: {"results": [{"path": "..."} , ...]}
    - TOON-ish text: {"results": "<path>, ...\\n<path>, ..."} or {"text": "..."}
    - Mixed shapes; returns [] on unknown/empty.
    """
    paths: List[str] = []

    def _add(p: Any) -> None:
        if not isinstance(p, str):
            return
        s = p.strip()
        if not s:
            return
        paths.append(s)

    def _parse_text(blob: str) -> None:
        if not blob:
            return
        for raw in blob.splitlines():
            line = raw.strip()
            if not line:
                continue
            # Most TOON lines start with "<path>,..." – take the first CSV column.
            head = line.split(",", 1)[0].strip()
            # Heuristic: require something path-like.
            if "/" in head or "." in head:
                _add(head)

    if isinstance(result, dict):
        results_data = result.get("results")
        if isinstance(results_data, list):
            for r in results_data:
                if isinstance(r, dict) and "path" in r:
                    _add(r.get("path"))
        elif isinstance(results_data, str):
            _parse_text(results_data)

        # Some tools return a separate "text" field (e.g., TOON format).
        if not paths:
            text_data = result.get("text")
            if isinstance(text_data, str):
                _parse_text(text_data)

    if limit is not None and limit >= 0:
        return paths[:limit]
    return paths


def resolve_nonempty_collection(
    preferred: Optional[str] = None,
    qdrant_url: Optional[str] = None,
    min_points: int = 1,
) -> str:
    """
    Resolve a Qdrant collection name that is likely to contain indexed data.

    Benchmarks often run in environments where COLLECTION_NAME is set to a placeholder
    or an empty collection. If the preferred collection has <min_points> points, we
    pick the collection with the largest point count instead (best-effort).
    """
    pref = (preferred or "").strip()
    url = (qdrant_url or "").strip() or __import__("os").environ.get("QDRANT_URL", "http://localhost:6333")

    try:
        from qdrant_client import QdrantClient  # type: ignore
    except Exception:
        return pref or "codebase"

    try:
        client = QdrantClient(url=url)
        cols = client.get_collections().collections
    except Exception:
        return pref or "codebase"

    counts: Dict[str, int] = {}
    for c in cols:
        name = getattr(c, "name", "") or ""
        if not name:
            continue
        try:
            cnt = int(client.count(collection_name=name, exact=True).count or 0)
        except Exception:
            cnt = 0
        counts[name] = cnt

    # Prefer preferred if it has enough points
    if pref and counts.get(pref, 0) >= int(min_points):
        return pref

    # Else pick the largest non-empty collection
    if counts:
        best = max(counts.items(), key=lambda kv: kv[1])
        if best[1] >= int(min_points):
            return best[0]

    return pref or "codebase"


def require_collection(
    collection: str,
    qdrant_url: Optional[str] = None,
    min_points: int = 1,
) -> int:
    """
    FAIL-FAST: Require a specific collection to exist with at least min_points.
    
    Use this for deterministic benchmarks where we don't want silent fallback.
    Raises ValueError if collection is empty/missing.
    
    Returns: point count of the collection
    """
    import os
    coll = (collection or "").strip()
    if not coll:
        raise ValueError("require_collection: collection name is required")
    
    url = (qdrant_url or "").strip() or os.environ.get("QDRANT_URL", "http://localhost:6333")
    
    try:
        from qdrant_client import QdrantClient  # type: ignore
        client = QdrantClient(url=url)
        info = client.get_collection(coll)
        count = int(info.points_count or 0)
    except Exception as e:
        raise ValueError(f"require_collection: collection '{coll}' not accessible: {e}")
    
    if count < min_points:
        raise ValueError(
            f"require_collection: collection '{coll}' has {count} points, "
            f"but {min_points} required. Run indexing first or specify a valid collection."
        )
    
    return count


def resolve_collection_auto(
    preferred: Optional[str] = None,
    qdrant_url: Optional[str] = None,
    min_points: int = 1,
) -> str:
    """
    Unified collection resolver that respects BENCH_STRICT mode.
    
    - BENCH_STRICT=1: Fail fast if preferred collection is empty/missing
    - BENCH_STRICT=0 (default): Auto-select largest non-empty collection
    
    Use this in benchmark scripts for consistent behavior.
    """
    import os
    strict = os.environ.get("BENCH_STRICT", "").strip().lower() in ("1", "true", "yes")
    pref = (preferred or "").strip() or os.environ.get("COLLECTION_NAME", "")
    
    if strict:
        # Fail-fast mode - require the exact collection
        require_collection(pref, qdrant_url, min_points)
        return pref
    else:
        # Auto-select mode (legacy behavior)
        return resolve_nonempty_collection(pref, qdrant_url, min_points)


# ---------------------------------------------------------------------------
# Unified Reporting Format
# ---------------------------------------------------------------------------
@dataclass
class QueryResult:
    """Result from evaluating a single query."""
    query: str
    latency_ms: float
    metrics: Dict[str, float] = field(default_factory=dict)
    retrieved_paths: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkMetadata:
    """Metadata for a benchmark run."""
    benchmark_name: str
    version: str = "1.0.0"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    config: Dict[str, Any] = field(default_factory=dict)
    environment: Dict[str, str] = field(default_factory=dict)


@dataclass
class BenchmarkReport:
    """
    Unified benchmark report format.
    
    All benchmarks should emit this format for consistent analysis.
    """
    metadata: BenchmarkMetadata
    per_query: List[QueryResult] = field(default_factory=list)
    aggregates: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[Dict[str, str]] = field(default_factory=list)
    
    def compute_aggregates(self) -> None:
        """Compute aggregate statistics from per-query results."""
        if not self.per_query:
            return
        
        latencies = [q.latency_ms for q in self.per_query]
        self.aggregates["latency"] = compute_percentiles(latencies)
        self.aggregates["total_queries"] = len(self.per_query)
        
        # Aggregate all numeric metrics
        metric_keys = set()
        for q in self.per_query:
            metric_keys.update(q.metrics.keys())
        
        for key in metric_keys:
            values = [q.metrics.get(key, 0) for q in self.per_query if key in q.metrics]
            if values:
                stddev = round(statistics.stdev(values), 4) if len(values) > 1 else 0.0
                self.aggregates[key] = {
                    "mean": round(statistics.mean(values), 4),
                    "min": round(min(values), 4),
                    "max": round(max(values), 4),
                    "stddev": stddev,
                }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "metadata": asdict(self.metadata),
            "per_query": [asdict(q) for q in self.per_query],
            "aggregates": self.aggregates,
            "recommendations": self.recommendations,
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)
    
    def save(self, path: str) -> None:
        """Save report to JSON file."""
        with open(path, "w") as f:
            f.write(self.to_json())
    
    def print_summary(self) -> None:
        """Print human-readable summary."""
        print("\n" + "=" * 70)
        print(f"BENCHMARK: {self.metadata.benchmark_name}")
        print("=" * 70)
        print(f"Timestamp: {self.metadata.timestamp}")
        print(f"Queries: {self.aggregates.get('total_queries', len(self.per_query))}")
        
        # Print latency stats
        if "latency" in self.aggregates:
            lat = self.aggregates["latency"]
            print("-" * 70)
            print("LATENCY (ms):")
            print(f"  P50: {lat['p50']:<8} P90: {lat['p90']:<8} P99: {lat['p99']}")
            print(f"  Min: {lat['min']:<8} Max: {lat['max']:<8} Mean: {lat['mean']}")
        
        # Print other aggregates
        print("-" * 70)
        print("METRICS:")
        for key, val in self.aggregates.items():
            if key in ("latency", "total_queries"):
                continue
            if isinstance(val, dict) and "mean" in val:
                print(f"  {key}: {val['mean']:.4f} (min={val['min']:.4f}, max={val['max']:.4f})")
            else:
                print(f"  {key}: {val}")
        
        # Print recommendations
        if self.recommendations:
            print("-" * 70)
            print("RECOMMENDATIONS:")
            for rec in self.recommendations:
                priority = rec.get("priority", "").upper()
                action = rec.get("action", "")
                print(f"  [{priority}] {action}")
        
        print("=" * 70)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Sensitive patterns to EXCLUDE from env snapshot
# Derived from .env.example: API keys, tokens, passwords, auth
_ENV_SENSITIVE_SUFFIXES = frozenset({
    "API_KEY", "SECRET", "PASSWORD", "CREDENTIAL", "_TOKEN",
})
_ENV_SENSITIVE_PREFIXES = frozenset({
    "AUTH_",  # catches CTXCE_AUTH_USERNAME, AUTH_ADMIN, etc.
})


def _is_sensitive_key(key: str) -> bool:
    """Check if key contains sensitive patterns (case-insensitive)."""
    key_upper = key.upper()
    # Suffix match for API_KEY, SECRET, PASSWORD, CREDENTIAL, _TOKEN
    if any(key_upper.endswith(p) for p in _ENV_SENSITIVE_SUFFIXES):
        return True
    # Prefix match for AUTH_
    return any(key_upper.startswith(p) for p in _ENV_SENSITIVE_PREFIXES)


def get_env_snapshot() -> Dict[str, str]:
    """Capture current environment config for reproducibility.
    
    Dumps ALL env vars EXCEPT those matching sensitive patterns
    (API_KEY, TOKEN, PASSWORD, SECRET, CREDENTIAL).
    """
    import os
    return {
        k: v for k, v in os.environ.items()
        if v and not _is_sensitive_key(k)
    }


def get_git_sha() -> str:
    """Get current git commit SHA, or empty string if not in a git repo."""
    import subprocess
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5
        )
        return result.stdout.strip()[:12] if result.returncode == 0 else ""
    except Exception:
        return ""


def get_runtime_info() -> Dict[str, Any]:
    """Get full runtime info for reproducibility.

    Returns dict with:
        - timestamp: ISO format
        - git_sha: short commit hash
        - env_snapshot: relevant env vars
        - platform: architecture and OS info
    """
    return {
        "timestamp": datetime.now().isoformat(),
        "git_sha": get_git_sha(),
        "env_snapshot": get_env_snapshot(),
        "platform": get_platform_info(),
    }


def get_platform_info() -> Dict[str, str]:
    """Get platform info for reproducibility and ARM detection."""
    import platform
    return {
        "machine": platform.machine(),      # arm64, x86_64, etc.
        "system": platform.system(),        # Darwin, Linux, Windows
        "python": platform.python_version(),
        "processor": platform.processor(),  # More detailed CPU info
    }


def save_run_meta(
    output_dir: str,
    run_id: str,
    config: Dict[str, Any],
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    """Save run metadata to separate JSON file for audit.
    
    Args:
        output_dir: Directory to save the metadata file
        run_id: Unique identifier for this run
        config: Benchmark configuration dict
        extra: Optional additional metadata to include
    
    Returns:
        Path to the saved metadata file
    """
    import os
    meta = {
        "run_id": run_id,
        **get_runtime_info(),
        "config": config,
    }
    if extra:
        meta.update(extra)
    
    path = os.path.join(output_dir, f"{run_id}_meta.json")
    with open(path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    return path


def create_report(name: str, config: Optional[Dict] = None) -> BenchmarkReport:
    """Create a new benchmark report with metadata."""
    import os

    metadata = BenchmarkMetadata(
        benchmark_name=name,
        config=config or {},
        environment={
            "collection": os.environ.get("COLLECTION_NAME", ""),
            "refrag_runtime": os.environ.get("REFRAG_RUNTIME", ""),
        },
    )
    return BenchmarkReport(metadata=metadata)
