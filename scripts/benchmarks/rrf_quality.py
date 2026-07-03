#!/usr/bin/env python3
"""
Hybrid RRF Quality Benchmark.

Measures the effectiveness of hybrid search (vector + lexical RRF) by comparing:
- Dense-only search
- Lexical-only search  
- Hybrid RRF search

Validates that RRF provides lift over the best single retrieval method.
"""

import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variables from .env
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv(PROJECT_ROOT / ".env")
except Exception:
    pass

# Fix Qdrant URL for running outside Docker
qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
if "qdrant:" in qdrant_url:
    os.environ["QDRANT_URL"] = "http://localhost:6333"

# Ensure correct collection is used (read from workspace state or env)
try:
    from scripts.benchmarks.common import resolve_collection_auto
    if not os.environ.get("COLLECTION_NAME"):
        from scripts.workspace_state import get_collection_name
        os.environ["COLLECTION_NAME"] = get_collection_name() or "codebase"
    else:
        os.environ["COLLECTION_NAME"] = resolve_collection_auto(os.environ.get("COLLECTION_NAME"))
except Exception:
    pass

print(
    f"[bench] Using QDRANT_URL={os.environ.get('QDRANT_URL', '')} "
    f"COLLECTION_NAME={os.environ.get('COLLECTION_NAME', '')}"
)

@dataclass
class SearchResult:
    """Result from a single search."""
    query: str
    method: str  # "dense", "lexical", "hybrid"
    paths: List[str]
    latency_ms: float
    scores: List[float] = field(default_factory=list)


@dataclass
class QualityMetrics:
    """Search quality metrics."""
    method: str
    mrr: float  # Mean Reciprocal Rank
    recall_at_5: float
    recall_at_10: float
    precision_at_5: float
    avg_latency_ms: float
    query_count: int

    def to_dict(self) -> Dict[str, float]:
        return {
            "mrr": float(self.mrr),
            "recall_at_5": float(self.recall_at_5),
            "recall_at_10": float(self.recall_at_10),
            "precision_at_5": float(self.precision_at_5),
            "avg_latency_ms": float(self.avg_latency_ms),
            "query_count": int(self.query_count),
        }


@dataclass
class RRFReport:
    """Full RRF quality report."""
    timestamp: str
    dense_metrics: QualityMetrics
    lexical_metrics: QualityMetrics
    hybrid_metrics: QualityMetrics
    rrf_lift_over_dense: float  # % improvement
    rrf_lift_over_lexical: float
    rrf_lift_over_best: float  # % improvement over max(dense, lexical)
    best_single_method: str
    per_query: List[Dict[str, object]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "metrics": {
                "dense": self.dense_metrics.to_dict(),
                "lexical": self.lexical_metrics.to_dict(),
                "hybrid": self.hybrid_metrics.to_dict(),
                "rrf_lift_over_dense_pct": float(self.rrf_lift_over_dense),
                "rrf_lift_over_lexical_pct": float(self.rrf_lift_over_lexical),
                "rrf_lift_over_best_pct": float(self.rrf_lift_over_best),
                "best_single_method": self.best_single_method,
            },
            "per_query": self.per_query,
        }


def _match_expected_file(path: str, expected_files: Iterable[str]) -> Optional[str]:
    """
    Return the expected_file that matches this path, or None if no match.

    Matching is done on normalized POSIX paths using suffix matching to handle
    different roots (e.g. 'scripts/hybrid/ranking.py' vs 'hybrid/ranking.py').
    """
    norm_path = Path(path).as_posix().lower()
    for expected in expected_files:
        expected_norm = expected.replace("\\", "/").lower()
        if norm_path.endswith(expected_norm):
            return expected
    return None


# Ground truth queries with expected files
# Based on eval_harness.py patterns
GROUND_TRUTH_QUERIES = [
    {
        "query": "hybrid search RRF ranking",
        "expected_files": ["hybrid/ranking.py", "hybrid_search.py", "hybrid/__init__.py"],
    },
    {
        "query": "recursive reranker TRM learning",
        "expected_files": ["rerank_recursive/core.py", "rerank_recursive/learning.py", "rerank_tools/recursive_reranker.py"],
    },
    {
        "query": "context_answer grounding citations",
        "expected_files": ["mcp_impl/context_answer.py", "refrag.py"],
    },
    {
        "query": "embedding model initialization",
        "expected_files": ["embedder.py", "onnx_embedder.py"],
    },
    {
        "query": "qdrant client manager connection",
        "expected_files": ["qdrant_client_manager.py", "hybrid/qdrant.py"],
    },
    {
        "query": "AST analyzer symbol extraction",
        "expected_files": ["ast_analyzer.py", "chunker.py"],
    },
    {
        "query": "query expansion optimization",
        "expected_files": ["query_optimizer.py", "decoder.py"],
    },
    {
        "query": "MCP tool registration fastmcp",
        "expected_files": ["mcp_indexer_server.py", "mcp_router/__init__.py"],
    },
    {
        "query": "memory store find operations",
        "expected_files": ["memory_store.py", "mcp_impl/memory.py"],
    },
    {
        "query": "benchmark evaluation harness metrics",
        "expected_files": ["benchmarks/eval_harness.py", "benchmarks/common.py"],
    },
]


async def search_dense_only(query: str, limit: int = 10) -> SearchResult:
    """Execute dense-only vector search using low-level dense_query."""
    start = time.time()

    try:
        from scripts.hybrid.qdrant import dense_query, get_qdrant_client
        from scripts.hybrid.embed import get_embedding_model
        from scripts.utils import sanitize_vector_name
        
        # Get client and model
        client = get_qdrant_client()
        model = get_embedding_model()
        model_name = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
        vec_name = sanitize_vector_name(model_name)
        
        loop = asyncio.get_running_loop()
        
        # Embed query in executor (CPU-bound)
        embeddings = await loop.run_in_executor(None, lambda: list(model.embed([query])))
        if not embeddings:
            raise ValueError("Failed to compute embedding")
        vec = list(embeddings[0])
        
        # Execute direct dense query
        raw_results = await loop.run_in_executor(
            None,
            lambda: dense_query(
                client=client,
                vec_name=vec_name,
                v=vec,
                flt=None,
                per_query=limit,
                collection_name=None,
                query_text=query,
            )
        )
        
        # Extract paths and scores
        paths = []
        scores = []
        for pt in raw_results[:limit]:
            payload = getattr(pt, "payload", {}) or {}
            path = payload.get("metadata", {}).get("path", "") or payload.get("path", "")
            score = getattr(pt, "score", 0) or 0
            if path:
                paths.append(path)
                scores.append(score)

    except Exception as e:
        print(f"Dense search error: {type(e).__name__}: {e}", file=sys.stderr)
        paths = []
        scores = []

    latency = (time.time() - start) * 1000
    return SearchResult(query=query, method="dense", paths=paths, latency_ms=latency, scores=scores)


async def search_lexical_only(query: str, limit: int = 10) -> SearchResult:
    """Execute lexical-only search using low-level lex_query."""
    start = time.time()

    try:
        from scripts.hybrid.qdrant import lex_query, get_qdrant_client, lex_hash_vector
        
        # Get client
        client = get_qdrant_client()
        
        # Build lexical hash vector from query
        lex_vec = lex_hash_vector([query])
        
        loop = asyncio.get_running_loop()
        
        # Execute direct lexical query
        raw_results = await loop.run_in_executor(
            None,
            lambda: lex_query(
                client=client,
                v=lex_vec,
                flt=None,
                per_query=limit,
                collection_name=None,
            )
        )
        
        # Extract paths and scores
        paths = []
        scores = []
        for pt in raw_results[:limit]:
            payload = getattr(pt, "payload", {}) or {}
            path = payload.get("metadata", {}).get("path", "") or payload.get("path", "")
            score = getattr(pt, "score", 0) or 0
            if path:
                paths.append(path)
                scores.append(score)

    except Exception as e:
        print(f"Lexical search error: {type(e).__name__}: {e}", file=sys.stderr)
        paths = []
        scores = []

    latency = (time.time() - start) * 1000
    return SearchResult(query=query, method="lexical", paths=paths, latency_ms=latency, scores=scores)


async def search_hybrid_rrf(query: str, limit: int = 10) -> SearchResult:
    """Execute hybrid RRF search (uses run_hybrid_search with RRF fusion)."""
    start = time.time()

    try:
        from scripts.hybrid_search import run_hybrid_search

        loop = asyncio.get_running_loop()
        
        # Use standard hybrid search with RRF fusion
        results = await loop.run_in_executor(
            None,
            lambda: run_hybrid_search(
                queries=[query],
                limit=limit,
                expand=False,
            )
        )

        paths = [r.get("path", "") for r in results]
        scores = [r.get("score", 0) for r in results]

    except Exception as e:
        print(f"Hybrid search error: {type(e).__name__}: {e}", file=sys.stderr)
        paths = []
        scores = []

    latency = (time.time() - start) * 1000
    return SearchResult(query=query, method="hybrid", paths=paths, latency_ms=latency, scores=scores)


def calculate_mrr(results: List[SearchResult], ground_truth: List[Dict]) -> float:
    """Calculate Mean Reciprocal Rank."""
    if len(results) != len(ground_truth):
        raise ValueError(
            f"Results and ground_truth length mismatch: {len(results)} vs {len(ground_truth)}"
        )
    
    reciprocal_ranks: List[float] = []
    
    for result, gt in zip(results, ground_truth):
        expected_files = gt["expected_files"]
        rank = 0
        
        # Find first relevant result
        for i, path in enumerate(result.paths, start=1):
            if _match_expected_file(path, expected_files):
                rank = i
                break
        
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)
    
    return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0


def calculate_recall(results: List[SearchResult], ground_truth: List[Dict], k: int) -> float:
    """Calculate Recall@k."""
    if len(results) != len(ground_truth):
        raise ValueError(
            f"Results and ground_truth length mismatch: {len(results)} vs {len(ground_truth)}"
        )
    
    recalls: List[float] = []
    
    for result, gt in zip(results, ground_truth):
        expected = set(gt["expected_files"])
        retrieved = set()
        
        for path in result.paths[:k]:
            matched = _match_expected_file(path, expected)
            if matched:
                retrieved.add(matched)
        
        recall = len(retrieved) / len(expected) if expected else 0.0
        recalls.append(recall)
    
    return sum(recalls) / len(recalls) if recalls else 0.0


def calculate_precision(results: List[SearchResult], ground_truth: List[Dict], k: int) -> float:
    """Calculate Precision@k."""
    if len(results) != len(ground_truth):
        raise ValueError(
            f"Results and ground_truth length mismatch: {len(results)} vs {len(ground_truth)}"
        )
    
    precisions: List[float] = []
    
    for result, gt in zip(results, ground_truth):
        expected = set(gt["expected_files"])
        relevant = 0
        
        for path in result.paths[:k]:
            if _match_expected_file(path, expected):
                relevant += 1
        
        precision = relevant / k if k > 0 else 0.0
        precisions.append(precision)
    
    return sum(precisions) / len(precisions) if precisions else 0.0


async def run_rrf_benchmark() -> RRFReport:
    """Run the complete RRF quality benchmark."""
    print("=" * 70)
    print("HYBRID RRF QUALITY BENCHMARK")
    print("=" * 70)
    
    dense_results = []
    lexical_results = []
    hybrid_results = []
    per_query: List[Dict[str, Any]] = []
    
    print(f"\nRunning {len(GROUND_TRUTH_QUERIES)} queries x 3 methods...\n")
    
    for i, gt in enumerate(GROUND_TRUTH_QUERIES, 1):
        query = gt["query"]
        print(f"  [{i}/{len(GROUND_TRUTH_QUERIES)}] {query[:40]}...", end=" ")
        
        # Run all three methods
        dense = await search_dense_only(query)
        lexical = await search_lexical_only(query)
        hybrid = await search_hybrid_rrf(query)
        
        dense_results.append(dense)
        lexical_results.append(lexical)
        hybrid_results.append(hybrid)

        per_query.append(
            {
                "query": query,
                "expected_files": list(gt.get("expected_files", [])),
                "dense": {
                    "latency_ms": dense.latency_ms,
                    "paths": list(dense.paths),
                    "scores": list(dense.scores),
                },
                "lexical": {
                    "latency_ms": lexical.latency_ms,
                    "paths": list(lexical.paths),
                    "scores": list(lexical.scores),
                },
                "hybrid": {
                    "latency_ms": hybrid.latency_ms,
                    "paths": list(hybrid.paths),
                    "scores": list(hybrid.scores),
                },
            }
        )
        
        print(f"✓ (D:{dense.latency_ms:.0f}ms L:{lexical.latency_ms:.0f}ms H:{hybrid.latency_ms:.0f}ms)")
    
    # Calculate metrics for each method
    print("\n" + "-" * 70)
    print("CALCULATING METRICS...")
    
    dense_metrics = QualityMetrics(
        method="dense",
        mrr=calculate_mrr(dense_results, GROUND_TRUTH_QUERIES),
        recall_at_5=calculate_recall(dense_results, GROUND_TRUTH_QUERIES, 5),
        recall_at_10=calculate_recall(dense_results, GROUND_TRUTH_QUERIES, 10),
        precision_at_5=calculate_precision(dense_results, GROUND_TRUTH_QUERIES, 5),
        avg_latency_ms=sum(r.latency_ms for r in dense_results) / len(dense_results),
        query_count=len(dense_results),
    )
    
    lexical_metrics = QualityMetrics(
        method="lexical",
        mrr=calculate_mrr(lexical_results, GROUND_TRUTH_QUERIES),
        recall_at_5=calculate_recall(lexical_results, GROUND_TRUTH_QUERIES, 5),
        recall_at_10=calculate_recall(lexical_results, GROUND_TRUTH_QUERIES, 10),
        precision_at_5=calculate_precision(lexical_results, GROUND_TRUTH_QUERIES, 5),
        avg_latency_ms=sum(r.latency_ms for r in lexical_results) / len(lexical_results),
        query_count=len(lexical_results),
    )
    
    hybrid_metrics = QualityMetrics(
        method="hybrid",
        mrr=calculate_mrr(hybrid_results, GROUND_TRUTH_QUERIES),
        recall_at_5=calculate_recall(hybrid_results, GROUND_TRUTH_QUERIES, 5),
        recall_at_10=calculate_recall(hybrid_results, GROUND_TRUTH_QUERIES, 10),
        precision_at_5=calculate_precision(hybrid_results, GROUND_TRUTH_QUERIES, 5),
        avg_latency_ms=sum(r.latency_ms for r in hybrid_results) / len(hybrid_results),
        query_count=len(hybrid_results),
    )
    
    # Calculate RRF lift
    best_single = max(dense_metrics.mrr, lexical_metrics.mrr)
    # Use >= to prefer dense in tie cases (consistent labeling)
    best_method = "dense" if dense_metrics.mrr >= lexical_metrics.mrr else "lexical"
    
    rrf_lift_dense = (hybrid_metrics.mrr - dense_metrics.mrr) / dense_metrics.mrr * 100 if dense_metrics.mrr > 0 else 0
    rrf_lift_lexical = (hybrid_metrics.mrr - lexical_metrics.mrr) / lexical_metrics.mrr * 100 if lexical_metrics.mrr > 0 else 0
    rrf_lift_best = (hybrid_metrics.mrr - best_single) / best_single * 100 if best_single > 0 else 0
    
    report = RRFReport(
        timestamp=datetime.now().isoformat(),
        dense_metrics=dense_metrics,
        lexical_metrics=lexical_metrics,
        hybrid_metrics=hybrid_metrics,
        rrf_lift_over_dense=rrf_lift_dense,
        rrf_lift_over_lexical=rrf_lift_lexical,
        rrf_lift_over_best=rrf_lift_best,
        best_single_method=best_method,
        per_query=per_query,
    )
    
    return report


def print_rrf_report(report: RRFReport):
    """Print the RRF quality report."""
    print("\n" + "=" * 70)
    print("RRF QUALITY REPORT")
    print("=" * 70)
    
    print(f"\nTimestamp: {report.timestamp}")
    print(f"Queries evaluated: {report.hybrid_metrics.query_count}")
    
    print("\n" + "-" * 70)
    print("SEARCH METHOD COMPARISON:")
    print(f"{'Method':<12} {'MRR':>8} {'R@5':>8} {'R@10':>8} {'P@5':>8} {'Latency':>10}")
    print("-" * 60)
    
    for m in [report.dense_metrics, report.lexical_metrics, report.hybrid_metrics]:
        winner = " 🏆" if m == report.hybrid_metrics and report.rrf_lift_over_best > 0 else ""
        print(f"{m.method:<12} {m.mrr:>8.3f} {m.recall_at_5:>8.3f} {m.recall_at_10:>8.3f} {m.precision_at_5:>8.3f} {m.avg_latency_ms:>8.0f}ms{winner}")
    
    print("\n" + "-" * 70)
    print("RRF EFFECTIVENESS:")
    
    if report.rrf_lift_over_best > 0:
        print(f"  ✅ Hybrid RRF improves over best single method ({report.best_single_method})")
        print(f"  📈 Lift over dense:   {report.rrf_lift_over_dense:+.1f}%")
        print(f"  📈 Lift over lexical: {report.rrf_lift_over_lexical:+.1f}%")
        print(f"  📈 Lift over best:    {report.rrf_lift_over_best:+.1f}%")
    else:
        print(f"  ⚠️  Hybrid RRF does NOT improve over {report.best_single_method}")
        print(f"  📉 Consider tuning RRF weights or using single method")
    
    print("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Hybrid RRF Quality Benchmark")
    parser.add_argument("--output", "-o", help="Output JSON file")
    args = parser.parse_args()
    
    report = await run_rrf_benchmark()
    print_rrf_report(report)
    
    if args.output:
        output = {
            "timestamp": report.timestamp,
            "dense": {"mrr": report.dense_metrics.mrr, "recall_at_10": report.dense_metrics.recall_at_10},
            "lexical": {"mrr": report.lexical_metrics.mrr, "recall_at_10": report.lexical_metrics.recall_at_10},
            "hybrid": {"mrr": report.hybrid_metrics.mrr, "recall_at_10": report.hybrid_metrics.recall_at_10},
            "rrf_lift_over_best": report.rrf_lift_over_best,
        }
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
