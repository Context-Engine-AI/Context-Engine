#!/usr/bin/env python3
"""
Context-Engine Efficiency Benchmarking Framework

Measures tool call frequency, token consumption, latency, and quality
to validate optimization strategies for agentic workflows.
"""

import argparse
import asyncio
import json
import os
import sys
import time
import hashlib
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Shared stats helpers (after sys.path setup)
from scripts.benchmarks.common import percentile, extract_result_paths, resolve_collection_auto

# Ensure correct collection is used (read from workspace state or env)
if not os.environ.get("COLLECTION_NAME"):
    try:
        from scripts.workspace_state import get_collection_name
        os.environ["COLLECTION_NAME"] = get_collection_name() or "codebase"
    except Exception:
        os.environ["COLLECTION_NAME"] = "codebase"
else:
    # If COLLECTION_NAME is set but empty/unindexed, pick a non-empty collection for benchmarks.
    try:
        os.environ["COLLECTION_NAME"] = resolve_collection_auto(os.environ.get("COLLECTION_NAME"))
    except Exception:
        pass

print(
    f"[bench] Using QDRANT_URL={os.environ.get('QDRANT_URL', '')} "
    f"COLLECTION_NAME={os.environ.get('COLLECTION_NAME', '')}"
)

# ---------------------------------------------------------------------------
# Token Estimation (using tiktoken if available, else heuristic)
# ---------------------------------------------------------------------------
try:
    import tiktoken
    _ENCODER = tiktoken.get_encoding("cl100k_base")
    def count_tokens(text: str) -> int:
        """Count tokens using tiktoken."""
        return len(_ENCODER.encode(text))
except ImportError:
    def count_tokens(text: str) -> int:
        """Heuristic: ~4 chars per token."""
        return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Dataclasses for Metrics
# ---------------------------------------------------------------------------
@dataclass
class ToolCallMetrics:
    """Metrics for a single tool call."""
    tool_name: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    success: bool
    timestamp: float = field(default_factory=time.time)


@dataclass
class ScenarioResult:
    """Results from running a complete scenario."""
    scenario_name: str
    total_tool_calls: int
    total_input_tokens: int
    total_output_tokens: int
    total_latency_ms: float
    quality_score: float  # MRR or relevance
    success: bool
    tool_calls: List[ToolCallMetrics] = field(default_factory=list)
    
    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens
    
    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / max(1, self.total_tool_calls)


@dataclass
class BenchmarkReport:
    """Aggregate report across all scenarios."""
    scenarios: List[ScenarioResult] = field(default_factory=list)
    baseline_mode: bool = False
    optimization_flags: List[str] = field(default_factory=list)
    
    @property
    def total_tool_calls(self) -> int:
        return sum(s.total_tool_calls for s in self.scenarios)
    
    @property
    def total_tokens(self) -> int:
        return sum(s.total_tokens for s in self.scenarios)
    
    @property
    def avg_quality(self) -> float:
        scores = [s.quality_score for s in self.scenarios if s.quality_score > 0]
        return sum(scores) / max(1, len(scores))
    
    @property
    def p90_latency_ms(self) -> float:
        all_latencies = []
        for s in self.scenarios:
            all_latencies.extend(tc.latency_ms for tc in s.tool_calls)
        return percentile(all_latencies, 0.90)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_scenarios": len(self.scenarios),
            "total_tool_calls": self.total_tool_calls,
            "total_tokens": self.total_tokens,
            "avg_quality_score": round(self.avg_quality, 4),
            "p90_latency_ms": round(self.p90_latency_ms, 2),
            "baseline_mode": self.baseline_mode,
            "optimization_flags": self.optimization_flags,
            "scenarios": [
                {
                    "name": s.scenario_name,
                    "tool_calls": s.total_tool_calls,
                    "tokens": s.total_tokens,
                    "quality": round(s.quality_score, 4),
                    "latency_ms": round(s.total_latency_ms, 2),
                }
                for s in self.scenarios
            ],
        }

    def to_full_dict(self) -> Dict[str, Any]:
        """Full, JSON-serializable report including per-tool-call details."""
        return {
            "summary": self.to_dict(),
            "details": asdict(self),
        }


# ---------------------------------------------------------------------------
# Tool Call Wrapper
# ---------------------------------------------------------------------------
class ToolCallTracker:
    """Tracks tool calls and their metrics."""
    
    def __init__(self):
        self.calls: List[ToolCallMetrics] = []
    
    def reset(self):
        self.calls = []
    
    async def call(
        self,
        tool_fn: Callable,
        tool_name: str,
        **kwargs,
    ) -> Tuple[Any, ToolCallMetrics]:
        """Call a tool and record metrics."""
        input_text = json.dumps(kwargs, default=str)
        input_tokens = count_tokens(input_text)
        
        start = time.perf_counter()
        try:
            result = await tool_fn(**kwargs)
            success = True
        except Exception as e:
            result = {"error": str(e)}
            success = False
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        output_text = json.dumps(result, default=str) if result else ""
        output_tokens = count_tokens(output_text)
        
        metrics = ToolCallMetrics(
            tool_name=tool_name,
            latency_ms=elapsed_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            success=success,
        )
        self.calls.append(metrics)
        return result, metrics
    
    def get_totals(self) -> Dict[str, Any]:
        return {
            "total_calls": len(self.calls),
            "total_input_tokens": sum(c.input_tokens for c in self.calls),
            "total_output_tokens": sum(c.output_tokens for c in self.calls),
            "total_latency_ms": sum(c.latency_ms for c in self.calls),
        }


# ---------------------------------------------------------------------------
# Quality Scorer
# ---------------------------------------------------------------------------
def compute_mrr(expected_paths: List[str], result_paths: List[str]) -> float:
    """Compute Mean Reciprocal Rank."""
    for i, path in enumerate(result_paths):
        if any(exp in path for exp in expected_paths):
            return 1.0 / (i + 1)
    return 0.0


def compute_recall_at_k(expected_paths: List[str], result_paths: List[str], k: int = 5) -> float:
    """Compute recall@k."""
    top_k = result_paths[:k]
    hits = sum(1 for exp in expected_paths if any(exp in r for r in top_k))
    return hits / max(1, len(expected_paths))


# ---------------------------------------------------------------------------
# Test Scenarios
# ---------------------------------------------------------------------------
SCENARIOS: Dict[str, Dict[str, Any]] = {
    "debugging": {
        "description": "Debug a failing function by tracing callers and finding tests",
        "queries": [
            {"tool": "repo_search", "query": "init_openlit error handling"},
            {"tool": "symbol_graph", "symbol": "init_openlit", "query_type": "callers"},
            {"tool": "search_tests_for", "query": "openlit initialization"},
        ],
        "expected_paths": ["openlit_init.py", "test_openlit"],
    },
    "feature_implementation": {
        "description": "Understand existing patterns to add a new feature",
        "queries": [
            {"tool": "repo_search", "query": "memory store implementation pattern"},
            {"tool": "context_answer", "query": "How does memory_store work?"},
            {"tool": "search_config_for", "query": "memory collection settings"},
        ],
        "expected_paths": ["mcp_impl/memory.py", "memory_store"],
    },
    "cross_file_analysis": {
        "description": "Trace dependencies across multiple files",
        "queries": [
            {"tool": "symbol_graph", "symbol": "get_embedding_model", "query_type": "callers"},
            {"tool": "search_importers_for", "query": "embedder"},
            {"tool": "repo_search", "query": "embedding dimension vector size"},
        ],
        "expected_paths": ["embedder.py", "rerank_recursive"],
    },
    "documentation": {
        "description": "Generate explanation of a module",
        "queries": [
            {"tool": "context_answer", "query": "What is the purpose of the hybrid search module?"},
            {"tool": "repo_search", "query": "hybrid search RRF ranking algorithm"},
        ],
        "expected_paths": ["hybrid/ranking.py", "hybrid_search"],
    },
}


# ---------------------------------------------------------------------------
# Scenario Runner
# ---------------------------------------------------------------------------

# Global fusion mode flag (set by run_benchmark)
_USE_FUSION = False


async def run_scenario(
    scenario_name: str,
    tracker: ToolCallTracker,
    tool_registry: Dict[str, Callable],
) -> ScenarioResult:
    """Run a single scenario and collect metrics."""
    scenario = SCENARIOS.get(scenario_name)
    if not scenario:
        return ScenarioResult(
            scenario_name=scenario_name,
            total_tool_calls=0,
            total_input_tokens=0,
            total_output_tokens=0,
            total_latency_ms=0,
            quality_score=0,
            success=False,
        )
    
    tracker.reset()
    all_result_paths: List[str] = []
    queries_list = list(scenario["queries"])  # Make a copy to avoid modifying original
    
    # Query fusion: batch all repo_search queries into one call
    if _USE_FUSION:
        repo_search_queries = []
        other_queries = []
        for q in queries_list:
            q_copy = dict(q)  # Make copy
            if q_copy.get("tool") == "repo_search":
                repo_search_queries.append(q_copy.get("query", ""))
            else:
                other_queries.append(q_copy)
        
        # Execute fused repo_search first
        if repo_search_queries and "repo_search" in tool_registry:
            fused_result, _ = await tracker.call(
                tool_registry["repo_search"],
                "repo_search",
                queries=repo_search_queries,
                limit=10,
            )
            all_result_paths.extend(extract_result_paths(fused_result))
        
        # Execute remaining queries
        queries_list = other_queries
    
    # Execute queries (or remaining queries if fusion was used)
    for query_spec in queries_list:
        q_copy = dict(query_spec)
        tool_name = q_copy.pop("tool")
        tool_fn = tool_registry.get(tool_name)
        if not tool_fn:
            continue
        
        result, _ = await tracker.call(tool_fn, tool_name, **q_copy)
        
        all_result_paths.extend(extract_result_paths(result))
    
    totals = tracker.get_totals()
    quality = compute_mrr(scenario["expected_paths"], all_result_paths)
    
    return ScenarioResult(
        scenario_name=scenario_name,
        total_tool_calls=totals["total_calls"],
        total_input_tokens=totals["total_input_tokens"],
        total_output_tokens=totals["total_output_tokens"],
        total_latency_ms=totals["total_latency_ms"],
        quality_score=quality,
        success=True,
        tool_calls=list(tracker.calls),
    )


# ---------------------------------------------------------------------------
# Main Benchmark Runner
# ---------------------------------------------------------------------------

# Session-level result cache for optimization testing
_SESSION_CACHE: Dict[str, Any] = {}
_CACHE_HITS = 0
_CACHE_MISSES = 0


def _cache_key(tool_name: str, kwargs: Dict[str, Any]) -> str:
    """Generate a cache key from tool name and arguments."""
    key_data = json.dumps({"tool": tool_name, **kwargs}, sort_keys=True, default=str)
    return hashlib.sha256(key_data.encode()).hexdigest()[:16]


def _normalize_for_dedup(tool_name: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize tool call kwargs into a stable dict for lightweight deduplication."""
    norm: Dict[str, Any] = {"tool": tool_name}

    # Treat query and queries consistently
    queries = kwargs.get("queries", None)
    if queries is None and "query" in kwargs:
        queries = kwargs.get("query")

    if isinstance(queries, str):
        norm["queries"] = [queries.strip().lower()]
    elif isinstance(queries, (list, tuple)):
        norm["queries"] = [str(q).strip().lower() for q in queries]
    elif queries is not None:
        norm["queries"] = [str(queries).strip().lower()]

    # Common filters / knobs we want dedup to consider
    for k in (
        "limit",
        "per_path",
        "language",
        "under",
        "kind",
        "symbol",
        "ext",
        "not_",
        "case",
        "path_regex",
        "path_glob",
        "not_glob",
        "expand",
        "collection",
        "vector_name",
        "mode",
        "repo",
        "rerank_enabled",
        "query_type",
    ):
        if k in kwargs and kwargs[k] is not None:
            v = kwargs[k]
            if isinstance(v, str):
                norm[k] = v.strip().lower()
            elif isinstance(v, (list, tuple)):
                norm[k] = [str(x).strip().lower() for x in v]
            elif isinstance(v, dict):
                norm[k] = {str(kk).strip().lower(): str(vv).strip().lower() for kk, vv in sorted(v.items())}
            else:
                norm[k] = v

    # Alias: some callers use "not" instead of "not_"
    if "not_" not in norm and kwargs.get("not") is not None:
        v = kwargs.get("not")
        if isinstance(v, str):
            norm["not_"] = v.strip().lower()
        elif isinstance(v, (list, tuple)):
            norm["not_"] = [str(x).strip().lower() for x in v]
        else:
            norm["not_"] = v

    return norm


def _dedup_key(tool_name: str, kwargs: Dict[str, Any]) -> str:
    """Generate a short stable key for deduplicating semantically-identical calls."""
    key_data = json.dumps(_normalize_for_dedup(tool_name, kwargs), sort_keys=True, default=str)
    return hashlib.sha256(key_data.encode()).hexdigest()[:16]


async def run_benchmark(
    scenarios: List[str],
    baseline: bool = False,
    optimize: Optional[str] = None,
) -> BenchmarkReport:
    """Run the full benchmark suite."""
    global _SESSION_CACHE, _CACHE_HITS, _CACHE_MISSES, _USE_FUSION
    
    # Parse optimization flags
    opt_flags = set((optimize or "").split("+")) if optimize else set()
    use_cache = "cache" in opt_flags
    use_dedup = "dedup" in opt_flags
    use_fusion = "fusion" in opt_flags
    
    # Set global fusion flag for run_scenario
    _USE_FUSION = use_fusion
    
    # Reset cache for this run
    _SESSION_CACHE = {}
    _CACHE_HITS = 0
    _CACHE_MISSES = 0

    # Session-level dedup map: key -> (timestamp, result)
    dedup_window = float(os.environ.get("DEDUP_WINDOW_SECONDS", "60"))
    _DEDUP_CACHE: Dict[str, Tuple[float, Any]] = {}
    _DEDUP_HITS = 0
    _DEDUP_MISSES = 0
    
    # Import MCP tools
    try:
        from scripts.mcp_indexer_server import (
            repo_search,
            context_answer,
            symbol_graph,
            search_tests_for,
            search_config_for,
            search_importers_for,
            memory_find,
        )
        tool_registry = {
            "repo_search": repo_search,
            "context_answer": context_answer,
            "symbol_graph": symbol_graph,
            "search_tests_for": search_tests_for,
            "search_config_for": search_config_for,
            "search_importers_for": search_importers_for,
            "memory_find": memory_find,
        }
    except ImportError as e:
        print(f"Failed to import MCP tools: {e}")
        return BenchmarkReport()
    
    # Wrap tools with deduplication if enabled (lightweight, per-session)
    if use_dedup:
        original_registry = tool_registry.copy()
        for tool_name, tool_fn in original_registry.items():
            async def dedup_tool(
                _fn=tool_fn, _name=tool_name, **kwargs
            ):
                nonlocal _DEDUP_HITS, _DEDUP_MISSES, _DEDUP_CACHE
                key = _dedup_key(_name, kwargs)
                now = time.time()

                # Opportunistic cleanup (keep it cheap)
                if len(_DEDUP_CACHE) > 2048:
                    cutoff = now - dedup_window
                    _DEDUP_CACHE = {k: v for k, v in _DEDUP_CACHE.items() if v[0] >= cutoff}

                if key in _DEDUP_CACHE and (now - _DEDUP_CACHE[key][0]) <= dedup_window:
                    _DEDUP_HITS += 1
                    return _DEDUP_CACHE[key][1]

                _DEDUP_MISSES += 1
                result = await _fn(**kwargs)
                _DEDUP_CACHE[key] = (now, result)
                return result

            tool_registry[tool_name] = dedup_tool

    # Wrap tools with caching if enabled
    if use_cache:
        original_registry = tool_registry.copy()
        for tool_name, tool_fn in original_registry.items():
            async def cached_tool(
                _fn=tool_fn, _name=tool_name, **kwargs
            ):
                global _CACHE_HITS, _CACHE_MISSES
                key = _cache_key(_name, kwargs)
                if key in _SESSION_CACHE:
                    _CACHE_HITS += 1
                    return _SESSION_CACHE[key]
                _CACHE_MISSES += 1
                result = await _fn(**kwargs)
                _SESSION_CACHE[key] = result
                return result
            tool_registry[tool_name] = cached_tool
    
    tracker = ToolCallTracker()
    report = BenchmarkReport(
        baseline_mode=baseline,
        optimization_flags=list(opt_flags) if opt_flags else [],
    )
    
    for scenario_name in scenarios:
        print(f"Running scenario: {scenario_name}...")
        result = await run_scenario(scenario_name, tracker, tool_registry)
        report.scenarios.append(result)
        print(f"  → {result.total_tool_calls} calls, {result.total_tokens} tokens, MRR={result.quality_score:.3f}")
    
    # Report cache stats if applicable
    if use_cache and (_CACHE_HITS + _CACHE_MISSES) > 0:
        print(f"\n[Cache Stats] Hits: {_CACHE_HITS}, Misses: {_CACHE_MISSES}, Hit Rate: {_CACHE_HITS / (_CACHE_HITS + _CACHE_MISSES):.1%}")

    # Report dedup stats if applicable
    if use_dedup and (_DEDUP_HITS + _DEDUP_MISSES) > 0:
        print(f"[Dedup Stats] Hits: {_DEDUP_HITS}, Misses: {_DEDUP_MISSES}, Hit Rate: {_DEDUP_HITS / (_DEDUP_HITS + _DEDUP_MISSES):.1%}, Window: {dedup_window:.0f}s")
    
    return report


def print_report(report: BenchmarkReport):
    """Print benchmark report."""
    print("\n" + "=" * 60)
    print("BENCHMARK REPORT")
    print("=" * 60)
    print(f"Mode: {'BASELINE' if report.baseline_mode else 'OPTIMIZED'}")
    if report.optimization_flags:
        print(f"Optimizations: {', '.join(report.optimization_flags)}")
    print("-" * 60)
    print(f"Total Scenarios: {len(report.scenarios)}")
    print(f"Total Tool Calls: {report.total_tool_calls}")
    print(f"Total Tokens: {report.total_tokens}")
    print(f"Avg Quality (MRR): {report.avg_quality:.4f}")
    print(f"P90 Latency: {report.p90_latency_ms:.2f}ms")
    print("-" * 60)
    for s in report.scenarios:
        print(f"  {s.scenario_name}: {s.total_tool_calls} calls, {s.total_tokens} tok, MRR={s.quality_score:.3f}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Context-Engine Efficiency Benchmark")
    parser.add_argument(
        "--scenario",
        choices=list(SCENARIOS.keys()) + ["all"],
        default="all",
        help="Scenario to run (or 'all')",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Run in baseline mode (no optimizations)",
    )
    parser.add_argument(
        "--optimize",
        type=str,
        default=None,
        help="Optimization flags (e.g., 'fusion+cache')",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for results",
    )
    args = parser.parse_args()
    
    scenarios = list(SCENARIOS.keys()) if args.scenario == "all" else [args.scenario]
    
    report = asyncio.run(run_benchmark(
        scenarios=scenarios,
        baseline=args.baseline,
        optimize=args.optimize,
    ))
    
    print_report(report)
    
    if args.output:
        with open(args.output, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"\nResults saved to: {args.output}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
