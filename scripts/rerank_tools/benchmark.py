#!/usr/bin/env python3
"""
Production-Grade Benchmark: Recursive Reranker on Real Codebase Data

Tests against the actual indexed Context Engine codebase:
1. Queries the real Qdrant index
2. Compares recursive reranker vs baseline vs ONNX
3. Measures actual latency and ranking changes
4. Uses ground truth from ONNX reranker as reference

Usage:
    python scripts/rerank_real_benchmark.py
"""

import os
import sys
import time
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Real queries based on actual Context Engine functionality
REAL_QUERIES = [
    "hybrid search RRF fusion implementation",
    "MCP server tool registration decorator",
    "embedding model initialization cache",
    "recursive reranker training deep supervision",
    "file watcher debounce indexing",
    "Qdrant vector search collection",
    "context answer LLM generation citations",
    "session aware latent state carryover",
    "cache eviction policy LRU TTL",
    "upload service delta bundle sync",
    "query expansion semantic similarity",
    "rerank ONNX cross encoder scoring",
    "memory store find retrieval",
    "subprocess manager process tracking",
    "Docker compose service orchestration",
]


@dataclass
class RealBenchmarkResult:
    """Result from benchmarking a single query."""
    query: str
    reranker: str
    latency_ms: float
    num_results: int
    top_5_paths: List[str]
    top_5_scores: List[float]
    # Ranking comparison metrics
    kendall_tau: float = 0.0  # Correlation with reference ranking
    top_3_overlap: float = 0.0  # Overlap of top-3 with reference


def get_real_candidates(query: str, limit: int = 30) -> List[Dict[str, Any]]:
    """Get real candidates from hybrid search (production pipeline)."""
    try:
        from scripts.hybrid_search import run_hybrid_search
        from scripts.embedder import get_embedding_model

        model_name = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
        model = get_embedding_model(model_name)

        # Run real hybrid search (dense + lexical fusion)
        results = run_hybrid_search(
            queries=[query],
            limit=limit,
            per_path=3,
            model=model,
        )

        # Convert to candidate format
        candidates = []
        for r in results:
            candidates.append({
                "path": r.get("path", ""),
                "symbol": r.get("symbol", ""),
                "code": r.get("code", r.get("snippet", "")),
                "score": float(r.get("score", 0)),
                "start_line": r.get("start_line", 0),
                "end_line": r.get("end_line", 0),
                "components": r.get("components", {}),
            })

        return candidates

    except Exception as e:
        print(f"Error getting candidates: {e}")
        import traceback
        traceback.print_exc()
        return []


def benchmark_baseline(query: str, candidates: List[Dict[str, Any]]) -> RealBenchmarkResult:
    """Benchmark baseline (no reranking, just use hybrid scores)."""
    start = time.perf_counter()

    # Just sort by existing score
    sorted_cands = sorted(candidates, key=lambda x: x.get("score", 0), reverse=True)

    latency = (time.perf_counter() - start) * 1000

    return RealBenchmarkResult(
        query=query,
        reranker="baseline",
        latency_ms=latency,
        num_results=len(sorted_cands),
        top_5_paths=[c.get("path", "") for c in sorted_cands[:5]],
        top_5_scores=[float(c.get("score", 0)) for c in sorted_cands[:5]],
    )


def benchmark_recursive(query: str, candidates: List[Dict[str, Any]], n_iters: int = 3) -> RealBenchmarkResult:
    """Benchmark recursive reranker."""
    try:
        from scripts.rerank_recursive import RecursiveReranker
    except ImportError:
        from rerank_recursive import RecursiveReranker

    reranker = RecursiveReranker(n_iterations=n_iters, dim=256)
    initial_scores = [c.get("score", 0) for c in candidates]

    start = time.perf_counter()
    reranked = reranker.rerank(query, candidates, initial_scores)
    latency = (time.perf_counter() - start) * 1000

    return RealBenchmarkResult(
        query=query,
        reranker=f"recursive_{n_iters}",
        latency_ms=latency,
        num_results=len(reranked),
        top_5_paths=[c.get("path", "") for c in reranked[:5]],
        top_5_scores=[float(c.get("score", 0)) for c in reranked[:5]],
    )


def benchmark_onnx(query: str, candidates: List[Dict[str, Any]]) -> Optional[RealBenchmarkResult]:
    """Benchmark ONNX cross-encoder reranker on pre-fetched candidates."""
    try:
        try:
            from scripts.rerank_local import rerank_local
        except ImportError:
            from rerank_local import rerank_local

        # Prepare pairs for ONNX reranker
        pairs = []
        for c in candidates:
            doc = c.get("code", "") or c.get("snippet", "")
            pairs.append((query, doc))

        start = time.perf_counter()
        scores = rerank_local(pairs)
        latency = (time.perf_counter() - start) * 1000

        # Combine scores with candidates and sort
        scored = list(zip(scores, candidates))
        scored.sort(key=lambda x: x[0], reverse=True)
        reranked = [{"score": s, **c} for s, c in scored]

        return RealBenchmarkResult(
            query=query,
            reranker="onnx",
            latency_ms=latency,
            num_results=len(reranked),
            top_5_paths=[c.get("path", "") for c in reranked[:5]],
            top_5_scores=[float(c.get("score", 0)) for c in reranked[:5]],
        )
    except Exception as e:
        print(f"ONNX reranker error: {e}")
        return None


def benchmark_session_aware(query: str, candidates: List[Dict[str, Any]], session_id: str) -> RealBenchmarkResult:
    """Benchmark session-aware recursive reranker."""
    try:
        from scripts.rerank_recursive import SessionAwareReranker
    except ImportError:
        from rerank_recursive import SessionAwareReranker

    reranker = SessionAwareReranker(n_iterations=3, dim=256)
    initial_scores = [c.get("score", 0) for c in candidates]

    start = time.perf_counter()
    reranked = reranker.rerank(query, candidates, session_id=session_id, initial_scores=initial_scores)
    latency = (time.perf_counter() - start) * 1000

    return RealBenchmarkResult(
        query=query,
        reranker="session_aware",
        latency_ms=latency,
        num_results=len(reranked),
        top_5_paths=[c.get("path", "") for c in reranked[:5]],
        top_5_scores=[float(c.get("score", 0)) for c in reranked[:5]],
    )


# Global reranker for online learning (persists across queries)
_LEARNING_RERANKER = None


def get_learning_reranker():
    """Get or create the learning-enabled reranker."""
    global _LEARNING_RERANKER
    if _LEARNING_RERANKER is None:
        try:
            from scripts.rerank_recursive import RecursiveReranker
        except ImportError:
            from rerank_recursive import RecursiveReranker
        _LEARNING_RERANKER = RecursiveReranker(n_iterations=3, dim=256)
    return _LEARNING_RERANKER


def benchmark_with_learning(
    query: str,
    candidates: List[Dict[str, Any]],
    teacher_scores: Optional[List[float]] = None,
) -> RealBenchmarkResult:
    """
    Benchmark recursive reranker WITH online learning from ONNX teacher.

    This learns from the ONNX scores to improve over time.
    """
    reranker = get_learning_reranker()
    initial_scores = [c.get("score", 0) for c in candidates]

    # Encode query and docs for learning
    doc_texts = []
    for c in candidates:
        parts = []
        if c.get("symbol"):
            parts.append(str(c["symbol"]))
        if c.get("path"):
            parts.append(str(c["path"]))
        code = c.get("code") or c.get("snippet") or ""
        if code:
            parts.append(str(code)[:500])
        doc_texts.append(" ".join(parts) if parts else "empty")

    # Get embeddings (cached after first call)
    query_emb = reranker._encode([query])[0]
    doc_embs = reranker._encode(doc_texts)
    query_emb = reranker._project_to_dim(query_emb.reshape(1, -1))[0]
    doc_embs = reranker._project_to_dim(doc_embs)

    # Learn from ONNX teacher if available
    if teacher_scores is not None and len(teacher_scores) == len(candidates):
        teacher_arr = np.array(teacher_scores, dtype=np.float32)
        z = query_emb.copy()  # Initial latent
        reranker.scorer.learn_from_teacher(query_emb, doc_embs, z, teacher_arr)

    # Now do inference
    start = time.perf_counter()
    reranked = reranker.rerank(query, candidates, initial_scores)
    latency = (time.perf_counter() - start) * 1000

    return RealBenchmarkResult(
        query=query,
        reranker="learning",
        latency_ms=latency,
        num_results=len(reranked),
        top_5_paths=[c.get("path", "") for c in reranked[:5]],
        top_5_scores=[float(c.get("score", 0)) for c in reranked[:5]],
    )


def compute_ranking_correlation(ranking1: List[str], ranking2: List[str]) -> float:
    """Compute ranking correlation (simplified Kendall's tau)."""
    if not ranking1 or not ranking2:
        return 0.0

    # Create position maps
    pos1 = {p: i for i, p in enumerate(ranking1)}
    pos2 = {p: i for i, p in enumerate(ranking2)}

    common = set(ranking1) & set(ranking2)
    if len(common) < 2:
        return 0.0

    concordant = 0
    discordant = 0

    common_list = list(common)
    for i in range(len(common_list)):
        for j in range(i + 1, len(common_list)):
            a, b = common_list[i], common_list[j]
            # Compare relative ordering
            order1 = pos1[a] < pos1[b]
            order2 = pos2[a] < pos2[b]
            if order1 == order2:
                concordant += 1
            else:
                discordant += 1

    total = concordant + discordant
    if total == 0:
        return 0.0

    return (concordant - discordant) / total


def run_real_benchmark():
    """Run the full benchmark on real data."""
    print("=" * 80)
    print("PRODUCTION BENCHMARK: Recursive Reranker on Real Codebase")
    print("=" * 80)

    # Check Qdrant connection
    try:
        from qdrant_client import QdrantClient
        url = os.environ.get("QDRANT_URL", "http://localhost:6333")
        client = QdrantClient(url=url)
        coll = os.environ.get("COLLECTION_NAME", "codebase")
        info = client.get_collection(coll)
        print(f"\nQdrant collection: {coll}")
        print(f"Points indexed: {info.points_count}")
    except Exception as e:
        print(f"Qdrant connection error: {e}")
        print("Make sure Qdrant is running and indexed")
        return

    results_by_reranker = {
        "baseline": [],
        "recursive_3": [],
        "session_aware": [],
        "learning": [],
        "onnx": [],
    }

    print(f"\nRunning {len(REAL_QUERIES)} queries...")
    print("-" * 80)

    session_id = "benchmark_session"

    for i, query in enumerate(REAL_QUERIES):
        print(f"\n[{i+1}/{len(REAL_QUERIES)}] Query: {query[:50]}...")

        # Get real candidates
        candidates = get_real_candidates(query, limit=25)
        if not candidates:
            print("  No candidates found, skipping")
            continue

        print(f"  Candidates: {len(candidates)}")

        # Benchmark each reranker
        baseline = benchmark_baseline(query, candidates)
        results_by_reranker["baseline"].append(baseline)
        print(f"  Baseline: {baseline.latency_ms:.2f}ms")

        # Get ONNX scores first (teacher signal)
        onnx_result = benchmark_onnx(query, candidates)
        teacher_scores = None
        if onnx_result:
            results_by_reranker["onnx"].append(onnx_result)
            print(f"  ONNX: {onnx_result.latency_ms:.2f}ms")
            teacher_scores = onnx_result.top_5_scores  # Use full scores
            # Get full ONNX scores for learning
            try:
                from scripts.rerank_local import rerank_local
            except ImportError:
                from rerank_local import rerank_local
            pairs = [(query, c.get("code", "") or c.get("snippet", "")) for c in candidates]
            teacher_scores = rerank_local(pairs)

        # Learning-enabled reranker (learns from ONNX before inference)
        learning_result = benchmark_with_learning(query, candidates, teacher_scores)
        results_by_reranker["learning"].append(learning_result)
        print(f"  Learning: {learning_result.latency_ms:.2f}ms")

        recursive = benchmark_recursive(query, candidates, n_iters=3)
        results_by_reranker["recursive_3"].append(recursive)
        print(f"  Recursive(3): {recursive.latency_ms:.2f}ms")

        session = benchmark_session_aware(query, candidates, session_id)
        results_by_reranker["session_aware"].append(session)
        print(f"  Session-aware: {session.latency_ms:.2f}ms")

        # Compare rankings with ONNX (ground truth)
        if onnx_result:
            learning_result.kendall_tau = compute_ranking_correlation(
                onnx_result.top_5_paths, learning_result.top_5_paths
            )
            recursive.kendall_tau = compute_ranking_correlation(
                onnx_result.top_5_paths, recursive.top_5_paths
            )
            session.kendall_tau = compute_ranking_correlation(
                onnx_result.top_5_paths, session.top_5_paths
            )
        else:
            # Fall back to baseline comparison
            learning_result.kendall_tau = compute_ranking_correlation(
                baseline.top_5_paths, learning_result.top_5_paths
            )
            recursive.kendall_tau = compute_ranking_correlation(
                baseline.top_5_paths, recursive.top_5_paths
            )
            session.kendall_tau = compute_ranking_correlation(
                baseline.top_5_paths, session.top_5_paths
            )

    # Print summary
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 80)

    print(f"\n{'Reranker':<15} {'Queries':<8} {'Latency (ms)':<20} {'Rank Change':<15}")
    print("-" * 60)

    for name, results in results_by_reranker.items():
        if not results:
            continue

        n = len(results)
        latencies = [r.latency_ms for r in results]
        mean_lat = np.mean(latencies)
        std_lat = np.std(latencies)

        taus = [r.kendall_tau for r in results if r.kendall_tau != 0]
        mean_tau = np.mean(taus) if taus else 0.0

        lat_str = f"{mean_lat:.2f} ± {std_lat:.2f}"
        tau_str = f"{mean_tau:.3f}" if mean_tau else "N/A"

        print(f"{name:<15} {n:<8} {lat_str:<20} {tau_str:<15}")

    print("-" * 60)

    # Show example ranking differences
    print("\n" + "=" * 80)
    print("RANKING COMPARISON (First Query)")
    print("=" * 80)

    if results_by_reranker["baseline"]:
        print(f"\nQuery: {results_by_reranker['baseline'][0].query}")

        print("\nBaseline Top-5:")
        for i, (path, score) in enumerate(zip(
            results_by_reranker["baseline"][0].top_5_paths,
            results_by_reranker["baseline"][0].top_5_scores
        )):
            print(f"  {i+1}. {path[-50:]} (score: {score:.3f})")

        if results_by_reranker["recursive_3"]:
            print("\nRecursive(3) Top-5:")
            for i, (path, score) in enumerate(zip(
                results_by_reranker["recursive_3"][0].top_5_paths,
                results_by_reranker["recursive_3"][0].top_5_scores
            )):
                print(f"  {i+1}. {path[-50:]} (score: {score:.3f})")

        if results_by_reranker["learning"]:
            print("\nLearning Top-5 (after training on ONNX):")
            for i, (path, score) in enumerate(zip(
                results_by_reranker["learning"][0].top_5_paths,
                results_by_reranker["learning"][0].top_5_scores
            )):
                print(f"  {i+1}. {path[-50:]} (score: {score:.3f})")

        if results_by_reranker["onnx"]:
            print("\nONNX Top-5 (teacher/ground truth):")
            for i, (path, score) in enumerate(zip(
                results_by_reranker["onnx"][0].top_5_paths,
                results_by_reranker["onnx"][0].top_5_scores
            )):
                print(f"  {i+1}. {path[-50:]} (score: {score:.3f})")

    # Show learning progress
    if results_by_reranker["learning"]:
        print("\n" + "=" * 80)
        print("LEARNING PROGRESS (correlation with ONNX over time)")
        print("=" * 80)

        taus = [r.kendall_tau for r in results_by_reranker["learning"]]
        for i, tau in enumerate(taus[:10]):  # First 10
            bar = "█" * int(tau * 20) if tau > 0 else ""
            print(f"  Query {i+1}: {tau:.3f} {bar}")

        if len(taus) > 10:
            print(f"  ... ({len(taus) - 10} more queries)")

        # Check if learning improved over time
        if len(taus) >= 5:
            early_avg = np.mean(taus[:len(taus)//2])
            late_avg = np.mean(taus[len(taus)//2:])
            improvement = late_avg - early_avg
            print(f"\n  Early avg: {early_avg:.3f}, Late avg: {late_avg:.3f}")
            if improvement > 0:
                print(f"  Improvement: +{improvement:.3f} (learning is working!)")
            else:
                print(f"  Change: {improvement:.3f}")

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    run_real_benchmark()

