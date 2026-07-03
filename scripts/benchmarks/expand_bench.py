#!/usr/bin/env python3
"""
Query Expansion Benchmark for Context-Engine

Measures expansion quality, semantic similarity, and retrieval impact.
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
        from scripts.benchmarks.common import resolve_collection_auto
        os.environ["COLLECTION_NAME"] = resolve_collection_auto(os.environ.get("COLLECTION_NAME"))
    except Exception:
        pass

print(
    f"[bench] Using QDRANT_URL={os.environ.get('QDRANT_URL', '')} "
    f"COLLECTION_NAME={os.environ.get('COLLECTION_NAME', '')}"
)

from scripts.benchmarks.common import (
    create_report,
    QueryResult as CommonQueryResult,
    extract_result_paths,
)
from scripts.benchmarks.eval_harness import EVAL_QUERIES, compute_mrr


@dataclass
class ExpansionResult:
    """Result from expansion evaluation."""
    original_query: str
    expanded_queries: List[str]
    expansion_count: int
    latency_ms: float
    retrieval_improvement: float  # MRR delta


@dataclass
class ExpansionReport:
    """Query expansion benchmark report."""
    name: str
    total_queries: int
    avg_expansions: float
    avg_improvement: float
    avg_latency_ms: float
    results: List[ExpansionResult] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        base = {
            "name": self.name,
            "total_queries": self.total_queries,
            "metrics": {
                "avg_expansions": round(self.avg_expansions, 2),
                "avg_improvement": round(self.avg_improvement, 4),
                "avg_latency_ms": round(self.avg_latency_ms, 2),
            },
            "results": [asdict(r) for r in self.results],
        }
        rep = create_report("expand_bench", config={"name": self.name, "source": "eval_harness.EVAL_QUERIES"})
        for r in self.results:
            rep.per_query.append(
                CommonQueryResult(
                    query=r.original_query,
                    latency_ms=r.latency_ms,
                    metrics={
                        "expansion_count": float(r.expansion_count),
                        "mrr_delta": float(r.retrieval_improvement),
                    },
                    retrieved_paths=[],
                    metadata={"expanded_queries": list(r.expanded_queries)},
                )
            )
        rep.compute_aggregates()
        base["unified"] = rep.to_dict()
        return base


def _get_eval_cases(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    cases = list(EVAL_QUERIES)
    return cases[:limit] if limit else cases


async def run_expansion_benchmark(name: str = "default") -> ExpansionReport:
    """Run query expansion benchmark."""
    try:
        from scripts.mcp_indexer_server import expand_query, repo_search
    except ImportError as e:
        print(f"Import error: {e}")
        return ExpansionReport(name=name, total_queries=0, avg_expansions=0,
                              avg_improvement=0, avg_latency_ms=0)
    
    results: List[ExpansionResult] = []
    
    for case in _get_eval_cases():
        query = case["query"]
        expected = case["expected"]
        print(f"  Expanding: {query}...")
        
        # Expand query
        start = time.perf_counter()
        try:
            expansion_result = await expand_query(query=query, max_new=3)
            alternates = expansion_result.get("alternates", []) if isinstance(expansion_result, dict) else []
        except Exception as e:
            print(f"    Expansion error: {e}")
            alternates = []
        expand_ms = (time.perf_counter() - start) * 1000
        
        # Measure retrieval improvement (MRR delta vs expected paths from eval harness)
        improvement = 0.0
        try:
            orig_result = await repo_search(query=query, limit=10)
            orig_paths = extract_result_paths(orig_result)
            orig_mrr = compute_mrr(expected, orig_paths)
        except Exception:
            orig_mrr = 0.0
        if alternates:
            try:
                exp_result = await repo_search(queries=[query] + alternates[:2], limit=10)
                exp_paths = extract_result_paths(exp_result)
                exp_mrr = compute_mrr(expected, exp_paths)
            except Exception:
                exp_mrr = orig_mrr
            improvement = exp_mrr - orig_mrr
        
        results.append(ExpansionResult(
            original_query=query,
            expanded_queries=alternates,
            expansion_count=len(alternates),
            latency_ms=expand_ms,
            retrieval_improvement=improvement,
        ))
        print(f"    expansions={len(alternates)}, ΔMRR={improvement:+.3f}")
    
    return ExpansionReport(
        name=name,
        total_queries=len(results),
        avg_expansions=statistics.mean(r.expansion_count for r in results) if results else 0,
        avg_improvement=statistics.mean(r.retrieval_improvement for r in results) if results else 0,
        avg_latency_ms=statistics.mean(r.latency_ms for r in results) if results else 0,
        results=results,
    )


def print_report(report: ExpansionReport):
    print("\n" + "=" * 60)
    print(f"QUERY EXPANSION BENCHMARK: {report.name}")
    print("=" * 60)
    print(f"Queries: {report.total_queries}")
    print(f"Avg Expansions: {report.avg_expansions:.2f}")
    print(f"Avg Improvement: {report.avg_improvement:.1%}")
    print(f"Avg Latency: {report.avg_latency_ms:.2f}ms")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Query Expansion Benchmark")
    parser.add_argument("--name", default="default", help="Benchmark name")
    parser.add_argument("--output", type=str, help="Output JSON file")
    args = parser.parse_args()
    
    print(f"Running expansion benchmark: {args.name}")
    report = asyncio.run(run_expansion_benchmark(name=args.name))
    print_report(report)
    
    if args.output:
        with open(args.output, "w") as f:
            json.dump(report.to_dict(), f, indent=2)


if __name__ == "__main__":
    main()
