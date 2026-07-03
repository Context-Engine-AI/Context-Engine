#!/usr/bin/env python3
"""
Router Benchmark for Context-Engine

Measures tool selection accuracy, routing latency, and decision quality.
"""

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional
import statistics

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Load environment (optional) and fix Docker hostname
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass
if "qdrant:" in os.environ.get("QDRANT_URL", ""):
    os.environ["QDRANT_URL"] = "http://localhost:6333"

from scripts.benchmarks.common import (
    percentile,
    create_report,
    QueryResult as CommonQueryResult,
    resolve_collection_auto,
)

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


@dataclass
class RouterResult:
    """Result from router evaluation."""
    query: str
    expected_tool: str
    selected_tool: str
    confidence: float
    latency_ms: float
    correct: bool


@dataclass
class RouterReport:
    """Router benchmark report."""
    name: str
    total_queries: int
    accuracy: float
    avg_confidence: float
    avg_latency_ms: float
    p90_latency_ms: float
    results: List[RouterResult] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        base = {
            "name": self.name,
            "total_queries": self.total_queries,
            "metrics": {
                "accuracy": round(self.accuracy, 4),
                "avg_confidence": round(self.avg_confidence, 4),
                "avg_latency_ms": round(self.avg_latency_ms, 2),
                "p90_latency_ms": round(self.p90_latency_ms, 2),
            },
            "results": [asdict(r) for r in self.results],
        }
        # Also emit the unified BenchmarkReport shape for downstream tooling.
        rep = create_report("router_bench", config={"name": self.name})
        for r in self.results:
            rep.per_query.append(
                CommonQueryResult(
                    query=r.query,
                    latency_ms=r.latency_ms,
                    metrics={
                        "correct": 1.0 if r.correct else 0.0,
                        "confidence": float(r.confidence or 0.0),
                    },
                    retrieved_paths=[],
                    metadata={
                        "expected_tool": r.expected_tool,
                        "selected_tool": r.selected_tool,
                    },
                )
            )
        rep.compute_aggregates()
        base["unified"] = rep.to_dict()
        return base


ROUTER_TEST_CASES = [
    {"query": "find files that import embedder module", "expected": "search_importers_for"},
    {"query": "who calls the init_openlit function", "expected": "symbol_graph"},
    {"query": "explain how the hybrid search works", "expected": "context_answer"},
    {"query": "search for memory store implementation", "expected": "repo_search"},
    {"query": "find tests for the reranker", "expected": "search_tests_for"},
    {"query": "what configs exist for qdrant", "expected": "search_config_for"},
    {"query": "store a note about this finding", "expected": "memory_store"},
    {"query": "recall notes about authentication", "expected": "memory_find"},
]

TOOL_ALIASES: Dict[str, str] = {
    # Search
    "code_search": "repo_search",
    "repo_search_compat": "repo_search",
    # Answer
    "context_answer_compat": "context_answer",
    # Memory
    "find": "memory_find",
    "store": "memory_store",
}


def _canonical_tool_name(name: Any) -> str:
    if not name:
        return "unknown"
    s = str(name).strip()
    if not s:
        return "unknown"
    return TOOL_ALIASES.get(s, s)


async def run_router_benchmark(name: str = "default") -> RouterReport:
    """Run router benchmark."""
    try:
        from scripts.mcp_router import route_query
    except ImportError:
        try:
            from scripts.mcp_router.router import route_query
        except ImportError as e:
            print(f"Import error: {e}")
            return RouterReport(name=name, total_queries=0, accuracy=0,
                              avg_confidence=0, avg_latency_ms=0, p90_latency_ms=0)
    
    results: List[RouterResult] = []
    latencies: List[float] = []
    
    for case in ROUTER_TEST_CASES:
        query = case["query"]
        expected = case["expected"]
        
        start = time.perf_counter()
        try:
            # Try to route the query
            route_result = await route_query(query)
            if isinstance(route_result, dict):
                selected = route_result.get("tool", "unknown")
                confidence = route_result.get("confidence", 0.0)
            else:
                selected = str(route_result)
                confidence = 0.5
        except Exception as e:
            selected = "error"
            confidence = 0.0
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)
        
        selected_c = _canonical_tool_name(selected)
        expected_c = _canonical_tool_name(expected)
        correct = selected_c == expected_c
        results.append(RouterResult(
            query=query,
            expected_tool=expected,
            selected_tool=selected_c,
            confidence=confidence,
            latency_ms=elapsed_ms,
            correct=correct,
        ))
        status = "✓" if correct else "✗"
        print(f"  {status} {query[:35]:35} → {selected_c} (exp: {expected_c})")
    
    correct_count = sum(1 for r in results if r.correct)
    
    return RouterReport(
        name=name,
        total_queries=len(results),
        accuracy=correct_count / len(results) if results else 0,
        avg_confidence=statistics.mean(r.confidence for r in results) if results else 0,
        avg_latency_ms=statistics.mean(latencies) if latencies else 0,
        p90_latency_ms=percentile(latencies, 0.90),
        results=results,
    )


def print_report(report: RouterReport):
    print("\n" + "=" * 60)
    print(f"ROUTER BENCHMARK: {report.name}")
    print("=" * 60)
    print(f"Queries: {report.total_queries}")
    print(f"Accuracy: {report.accuracy:.1%}")
    print(f"Avg Confidence: {report.avg_confidence:.4f}")
    print(f"Avg Latency: {report.avg_latency_ms:.2f}ms")
    print(f"P90 Latency: {report.p90_latency_ms:.2f}ms")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Router Benchmark")
    parser.add_argument("--name", default="default", help="Benchmark name")
    parser.add_argument("--output", type=str, help="Output JSON file")
    args = parser.parse_args()
    
    print(f"Running router benchmark: {args.name}")
    report = asyncio.run(run_router_benchmark(name=args.name))
    print_report(report)
    
    if args.output:
        with open(args.output, "w") as f:
            json.dump(report.to_dict(), f, indent=2)


if __name__ == "__main__":
    main()
