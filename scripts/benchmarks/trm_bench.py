#!/usr/bin/env python3
"""
TRM/Reranker Benchmark for Context-Engine

Measures embedding quality, vector search performance, and reranking accuracy.
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


@dataclass
class RerankerResult:
    """Result from reranker evaluation."""
    query: str
    baseline_order: List[str]
    reranked_order: List[str]
    onnx_order: Optional[List[str]]
    kendall_tau: float
    latency_ms: float


@dataclass  
class TRMReport:
    """TRM/Reranker benchmark report."""
    name: str
    total_queries: int
    avg_kendall_tau: float
    avg_latency_ms: float
    p90_latency_ms: float
    embedding_dim: int
    results: List[RerankerResult] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "total_queries": self.total_queries,
            "metrics": {
                "kendall_tau": round(self.avg_kendall_tau, 4),
                "avg_latency_ms": round(self.avg_latency_ms, 2),
                "p90_latency_ms": round(self.p90_latency_ms, 2),
            },
            "embedding_dim": self.embedding_dim,
            "results": [asdict(r) for r in self.results],
        }


def compute_kendall_tau(list1: List[str], list2: List[str]) -> float:
    """Simplified Kendall's tau for ranking comparison."""
    if not list1 or not list2:
        return 0.0
    
    common = set(list1) & set(list2)
    if len(common) < 2:
        return 0.0
    
    # Count concordant/discordant pairs
    concordant = 0
    discordant = 0
    
    items = list(common)
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, b = items[i], items[j]
            try:
                pos1_a, pos1_b = list1.index(a), list1.index(b)
                pos2_a, pos2_b = list2.index(a), list2.index(b)
                
                if (pos1_a < pos1_b) == (pos2_a < pos2_b):
                    concordant += 1
                else:
                    discordant += 1
            except ValueError:
                continue
    
    total = concordant + discordant
    if total == 0:
        return 0.0
    return (concordant - discordant) / total


BENCHMARK_QUERIES = [
    "hybrid search RRF ranking algorithm",
    "recursive reranker learning feedback",
    "memory store vector embedding",
    "openlit instrumentation tracing",
    "workspace state persistence JSON",
]


async def run_trm_benchmark(name: str = "default") -> TRMReport:
    """Run TRM/reranker benchmark."""
    try:
        from scripts.embedder import get_embedding_model
        from scripts.rerank_recursive import rerank_with_learning
        from scripts.mcp_indexer_server import repo_search
    except ImportError as e:
        print(f"Import error: {e}")
        return TRMReport(name=name, total_queries=0, avg_kendall_tau=0,
                        avg_latency_ms=0, p90_latency_ms=0, embedding_dim=0)
    
    # Get embedding dimension
    try:
        model = get_embedding_model()
        embedding_dim = model.get_sentence_embedding_dimension()
    except Exception:
        embedding_dim = 384  # fallback
    
    results: List[RerankerResult] = []
    latencies: List[float] = []
    
    for query in BENCHMARK_QUERIES:
        print(f"  Benchmarking: {query[:40]}...")
        
        # Get baseline results
        try:
            baseline_result = await repo_search(query=query, limit=10, rerank_enabled=False)
            baseline_paths = extract_result_paths(baseline_result)
        except Exception:
            baseline_paths = []
        
        # Get reranked results
        start = time.perf_counter()
        try:
            reranked_result = await repo_search(query=query, limit=10, rerank_enabled=True)
            reranked_paths = extract_result_paths(reranked_result)
        except Exception:
            reranked_paths = []
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)
        
        tau = compute_kendall_tau(baseline_paths, reranked_paths)
        
        results.append(RerankerResult(
            query=query,
            baseline_order=baseline_paths[:5],
            reranked_order=reranked_paths[:5],
            onnx_order=None,
            kendall_tau=tau,
            latency_ms=elapsed_ms,
        ))
        print(f"    τ={tau:.3f}, latency={elapsed_ms:.0f}ms")
    
    return TRMReport(
        name=name,
        total_queries=len(results),
        avg_kendall_tau=statistics.mean(r.kendall_tau for r in results) if results else 0,
        avg_latency_ms=statistics.mean(latencies) if latencies else 0,
        p90_latency_ms=percentile(latencies, 0.90),
        embedding_dim=embedding_dim,
        results=results,
    )

def print_report(report: TRMReport):
    print("\n" + "=" * 60)
    print(f"TRM/RERANKER BENCHMARK: {report.name}")
    print("=" * 60)
    print(f"Queries: {report.total_queries}")
    print(f"Embedding Dim: {report.embedding_dim}")
    print(f"Avg Kendall-τ: {report.avg_kendall_tau:.4f}")
    print(f"Avg Latency: {report.avg_latency_ms:.2f}ms")
    print(f"P90 Latency: {report.p90_latency_ms:.2f}ms")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="TRM/Reranker Benchmark")
    parser.add_argument("--name", default="default", help="Benchmark name")
    parser.add_argument("--output", type=str, help="Output JSON file")
    args = parser.parse_args()
    
    print(f"Running TRM benchmark: {args.name}")
    report = asyncio.run(run_trm_benchmark(name=args.name))
    print_report(report)
    
    if args.output:
        with open(args.output, "w") as f:
            json.dump(report.to_dict(), f, indent=2)


if __name__ == "__main__":
    main()
