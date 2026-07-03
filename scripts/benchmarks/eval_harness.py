#!/usr/bin/env python3
"""
Comprehensive Evaluation Harness for Context-Engine

Measures MRR, recall@k, precision, latency across query types and datasets.
Supports A/B testing and generates optimization recommendations.
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

# Shared stats helpers
from scripts.benchmarks.common import (
    percentile,
    extract_result_paths,
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
    # Benchmarks frequently run with COLLECTION_NAME pointing at an empty placeholder.
    try:
        os.environ["COLLECTION_NAME"] = resolve_collection_auto(os.environ.get("COLLECTION_NAME"))
    except Exception:
        pass

print(
    f"[bench] Using QDRANT_URL={os.environ.get('QDRANT_URL', '')} "
    f"COLLECTION_NAME={os.environ.get('COLLECTION_NAME', '')}"
)

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
@dataclass
class QueryResult:
    """Result from evaluating a single query."""
    query: str
    expected_paths: List[str]
    retrieved_paths: List[str]
    latency_ms: float
    mrr: float
    recall_at_5: float
    recall_at_10: float
    precision_at_5: float


@dataclass
class EvalReport:
    """Aggregate evaluation report."""
    name: str
    total_queries: int
    avg_mrr: float
    avg_recall_5: float
    avg_recall_10: float
    avg_precision_5: float
    p50_latency_ms: float
    p90_latency_ms: float
    p99_latency_ms: float
    results: List[QueryResult] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        base = {
            "name": self.name,
            "total_queries": self.total_queries,
            "metrics": {
                "mrr": round(self.avg_mrr, 4),
                "recall@5": round(self.avg_recall_5, 4),
                "recall@10": round(self.avg_recall_10, 4),
                "precision@5": round(self.avg_precision_5, 4),
            },
            "latency": {
                "p50_ms": round(self.p50_latency_ms, 2),
                "p90_ms": round(self.p90_latency_ms, 2),
                "p99_ms": round(self.p99_latency_ms, 2),
            },
            "results": [asdict(r) for r in self.results],
        }
        # Also emit a unified BenchmarkReport payload for downstream tools.
        rep = create_report("eval_harness", config={"name": self.name})
        for r in self.results:
            rep.per_query.append(
                CommonQueryResult(
                    query=r.query,
                    latency_ms=r.latency_ms,
                    metrics={
                        "mrr": float(r.mrr),
                        "recall@5": float(r.recall_at_5),
                        "recall@10": float(r.recall_at_10),
                        "precision@5": float(r.precision_at_5),
                    },
                    retrieved_paths=list(r.retrieved_paths),
                    metadata={"expected_paths": list(r.expected_paths)},
                )
            )
        rep.compute_aggregates()
        base["unified"] = rep.to_dict()
        return base


def _matches_expected(path: str, exp: str) -> bool:
    """Check if a retrieved path matches an expected pattern.
    
    Supports:
    - File matching: path.endswith(exp) or path.endswith("/exp") for files
    - Directory matching: "/exp/" in path for directory expectations
    """
    p = str(path or "").replace("\\", "/")
    e = str(exp or "").strip().replace("\\", "/").strip("/")
    if not p or not e:
        return False

    # Decide expectation type:
    # - If it has a suffix (".py", ".js", etc.) treat it as a file expectation.
    # - Otherwise treat it as a directory/module token expectation.
    try:
        from pathlib import Path as _P
        is_file = bool(_P(e).suffix)
    except Exception:
        is_file = "." in e

    if is_file:
        return p.endswith("/" + e) or p.endswith(e)
    return (f"/{e}/" in p) or p.endswith("/" + e) or p.endswith(e)


def compute_mrr_at_k(expected: List[str], retrieved: List[str], k: int) -> float:
    """MRR@k (MRR computed over top-k)."""
    return compute_mrr(expected, retrieved[:k])


def compute_mrr(expected: List[str], retrieved: List[str]) -> float:
    """Mean Reciprocal Rank.
    
    Supports both file and directory expectations.
    """
    for i, path in enumerate(retrieved):
        if any(_matches_expected(path, exp) for exp in expected):
            return 1.0 / (i + 1)
    return 0.0


def compute_recall(expected: List[str], retrieved: List[str], k: int) -> float:
    """Recall at k.
    
    Supports both file and directory expectations.
    """
    top_k = retrieved[:k]
    hits = sum(1 for exp in expected if any(_matches_expected(r, exp) for r in top_k))
    return hits / max(1, len(expected))


def compute_precision(expected: List[str], retrieved: List[str], k: int) -> float:
    """Precision at k.
    
    Uses same matching logic as MRR/recall for consistency.
    """
    top_k = retrieved[:k]
    hits = sum(1 for r in top_k if any(_matches_expected(r, exp) for exp in expected))
    # If fewer than k results returned, precision should be over the retrieved count.
    return hits / max(1, len(top_k))


# ---------------------------------------------------------------------------
# Test Dataset
# ---------------------------------------------------------------------------
EVAL_QUERIES = [
    {"query": "hybrid search RRF ranking", "expected": ["hybrid/ranking.py"]},
    {"query": "memory store implementation", "expected": ["mcp_impl/memory.py"]},
    {"query": "openlit initialization tracing", "expected": ["openlit_init.py"]},
    {"query": "recursive reranker learning", "expected": ["rerank_recursive"]},
    {"query": "embedder model loading", "expected": ["embedder.py"]},
    {"query": "workspace state persistence", "expected": ["workspace_state.py"]},
    {"query": "symbol graph callers", "expected": ["symbol_graph.py", "mcp_impl"]},
    {"query": "context answer grounded citations", "expected": ["context_answer.py"]},
    {"query": "deduplication request fingerprint", "expected": ["deduplication.py"]},
    {"query": "upload service delta bundle", "expected": ["upload_service.py", "upload_delta"]},
]


# ---------------------------------------------------------------------------
# Evaluation Runner  
# ---------------------------------------------------------------------------
async def run_evaluation(
    name: str = "default",
    queries: Optional[List[Dict]] = None,
    config: Optional[Dict] = None,
) -> EvalReport:
    """Run evaluation harness."""
    queries = queries or EVAL_QUERIES
    config = config or {}
    
    # Import search function
    try:
        from scripts.mcp_indexer_server import repo_search
    except ImportError as e:
        print(f"Failed to import: {e}")
        return EvalReport(name=name, total_queries=0, avg_mrr=0, avg_recall_5=0,
                         avg_recall_10=0, avg_precision_5=0, p50_latency_ms=0,
                         p90_latency_ms=0, p99_latency_ms=0)
    
    results: List[QueryResult] = []
    latencies: List[float] = []
    
    for q in queries:
        query_text = q["query"]
        expected = q["expected"]
        
        start = time.perf_counter()
        try:
            result = await repo_search(
                query=query_text,
                limit=config.get("limit", 10),
                per_path=config.get("per_path", 2),
            )
            success = True
        except Exception as e:
            print(f"Query failed: {query_text}: {e}")
            result = {}
            success = False
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)
        
        retrieved = extract_result_paths(result)
        
        qr = QueryResult(
            query=query_text,
            expected_paths=expected,
            retrieved_paths=retrieved[:10],
            latency_ms=elapsed_ms,
            mrr=compute_mrr(expected, retrieved),
            recall_at_5=compute_recall(expected, retrieved, 5),
            recall_at_10=compute_recall(expected, retrieved, 10),
            precision_at_5=compute_precision(expected, retrieved, 5),
        )

        # --- Consistency checks (single expected file only) ---
        try:
            from pathlib import Path as _P
            is_single_file = (
                isinstance(expected, list)
                and len(expected) == 1
                and bool(_P(str(expected[0])).suffix)  # file-like expectation
            )
        except Exception:
            is_single_file = isinstance(expected, list) and len(expected) == 1 and "." in str(expected[0])

        if is_single_file:
            mrr5 = compute_mrr_at_k(expected, retrieved, 5)
            # If there's a hit in top-5, precision@5 must be > 0.
            if mrr5 > 0 and qr.precision_at_5 <= 0:
                raise AssertionError(
                    f"Invariant violated for query='{query_text}': mrr@5={mrr5} but precision@5={qr.precision_at_5}"
                )
            # If any hit exists in top-5, recall@5 must be > 0.
            if qr.precision_at_5 > 0 and qr.recall_at_5 <= 0:
                raise AssertionError(
                    f"Invariant violated for query='{query_text}': precision@5={qr.precision_at_5} but recall@5={qr.recall_at_5}"
                )

        results.append(qr)
        print(f"  {query_text[:40]:40} MRR={qr.mrr:.3f} R@5={qr.recall_at_5:.2f}")
    
    # Compute aggregates
    return EvalReport(
        name=name,
        total_queries=len(results),
        avg_mrr=statistics.mean(r.mrr for r in results) if results else 0,
        avg_recall_5=statistics.mean(r.recall_at_5 for r in results) if results else 0,
        avg_recall_10=statistics.mean(r.recall_at_10 for r in results) if results else 0,
        avg_precision_5=statistics.mean(r.precision_at_5 for r in results) if results else 0,
        p50_latency_ms=percentile(latencies, 0.50),
        p90_latency_ms=percentile(latencies, 0.90),
        p99_latency_ms=percentile(latencies, 0.99),
        results=results,
    )


# ---------------------------------------------------------------------------
# A/B Comparison
# ---------------------------------------------------------------------------
def compare_reports(baseline: EvalReport, optimized: EvalReport) -> Dict[str, Any]:
    """Compare two evaluation reports."""
    def delta(a: float, b: float) -> str:
        if a == 0:
            return "N/A"
        pct = ((b - a) / a) * 100
        return f"{pct:+.1f}%"
    
    return {
        "baseline": baseline.to_dict(),
        "optimized": optimized.to_dict(),
        "delta": {
            "mrr": delta(baseline.avg_mrr, optimized.avg_mrr),
            "recall@5": delta(baseline.avg_recall_5, optimized.avg_recall_5),
            "p90_latency": delta(baseline.p90_latency_ms, optimized.p90_latency_ms),
        },
        "recommendations": generate_recommendations(baseline, optimized),
    }


def generate_recommendations(baseline: EvalReport, optimized: EvalReport) -> List[Dict]:
    """Generate optimization recommendations."""
    recs = []
    
    # MRR improvement
    mrr_delta = optimized.avg_mrr - baseline.avg_mrr
    if mrr_delta > 0.05:
        recs.append({
            "priority": "high",
            "action": "Deploy optimized configuration",
            "impact": f"+{mrr_delta:.1%} MRR improvement",
        })
    elif mrr_delta < -0.05:
        recs.append({
            "priority": "high", 
            "action": "Revert - quality regression detected",
            "impact": f"{mrr_delta:.1%} MRR decrease",
        })
    
    # Latency improvement
    lat_delta = optimized.p90_latency_ms - baseline.p90_latency_ms
    if lat_delta < -50:
        recs.append({
            "priority": "medium",
            "action": "Latency optimization effective",
            "impact": f"{-lat_delta:.0f}ms P90 reduction",
        })
    
    return recs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def print_report(report: EvalReport):
    print("\n" + "=" * 60)
    print(f"EVALUATION: {report.name}")
    print("=" * 60)
    print(f"Queries: {report.total_queries}")
    print(f"MRR:         {report.avg_mrr:.4f}")
    print(f"Recall@5:    {report.avg_recall_5:.4f}")
    print(f"Recall@10:   {report.avg_recall_10:.4f}")
    print(f"Precision@5: {report.avg_precision_5:.4f}")
    print("-" * 60)
    print(f"P50 Latency: {report.p50_latency_ms:.2f}ms")
    print(f"P90 Latency: {report.p90_latency_ms:.2f}ms")
    print(f"P99 Latency: {report.p99_latency_ms:.2f}ms")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Context-Engine Evaluation Harness")
    parser.add_argument("--name", default="default", help="Evaluation run name")
    parser.add_argument("--output", type=str, help="Output JSON file")
    parser.add_argument("--queries", type=str, help="Custom queries JSON file")
    args = parser.parse_args()
    
    queries = EVAL_QUERIES
    if args.queries and Path(args.queries).exists():
        with open(args.queries) as f:
            queries = json.load(f)
    
    print(f"Running evaluation: {args.name}")
    report = asyncio.run(run_evaluation(name=args.name, queries=queries))
    print_report(report)
    
    if args.output:
        with open(args.output, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
