#!/usr/bin/env python3
"""
Offline Reranker Evaluation Pipeline

Runs fixed-query evaluation for reranker quality with MRR/Recall/latency metrics.
Designed for CI/regression testing - deterministic, no sampling.

Usage:
    python scripts/rerank_eval.py [--queries QUERIES_FILE] [--output OUTPUT_FILE]
    python scripts/rerank_eval.py --ablations  # Run all ablation modes

Metrics reported:
    - MRR@k (Mean Reciprocal Rank)
    - Recall@k (fraction of relevant docs in top-k)
    - Latency p50/p95/p99
"""

import argparse
import copy
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Fixed evaluation queries (deterministic, no sampling)
DEFAULT_EVAL_QUERIES = [
    "hybrid search RRF fusion implementation",
    "MCP server tool registration decorator",
    "embedding model initialization cache",
    "recursive reranker training deep supervision",
    "Qdrant vector search collection create",
    "context answer LLM generation citations",
    "cache eviction policy LRU TTL",
    "query expansion semantic similarity",
    "rerank ONNX cross encoder scoring",
    "memory store find retrieval",
]


@dataclass
class EvalResult:
    """Evaluation result for a single query."""
    query: str
    mode: str
    latency_ms: float
    top_k_paths: List[str]
    top_k_scores: List[float]
    mrr: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0


@dataclass
class EvalSummary:
    """Aggregated evaluation summary."""
    mode: str
    num_queries: int
    mrr_mean: float
    recall_at_5_mean: float
    recall_at_10_mean: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    results: List[EvalResult] = field(default_factory=list)


def get_candidates(query: str, limit: int = 30) -> List[Dict[str, Any]]:
    """Get candidates from hybrid search."""
    try:
        from scripts.hybrid_search import run_hybrid_search
        from scripts.embedder import get_embedding_model

        # Use BAAI/bge-base-en-v1.5 which is supported by fastembed
        model_name = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
        model = get_embedding_model(model_name)

        results = run_hybrid_search(
            queries=[query],
            limit=limit,
            per_path=3,
            model=model,
        )

        candidates = []
        for r in results:
            candidates.append({
                "path": r.get("path", ""),
                "symbol": r.get("symbol", ""),
                "start_line": r.get("start_line", 0),
                "end_line": r.get("end_line", 0),
                "score": float(r.get("score", 0)),
                "snippet": r.get("snippet", "")[:500] if r.get("snippet") else "",
            })
        return candidates
    except Exception as e:
        print(f"Warning: Could not get candidates: {e}", file=sys.stderr)
        return []


def get_onnx_scores(query: str, candidates: List[Dict[str, Any]]) -> Optional[List[float]]:
    """Get ONNX reranker scores (ground truth)."""
    try:
        from scripts.rerank_local import rerank_local
        pairs = []
        for c in candidates:
            doc_parts = []
            if c.get("symbol"):
                doc_parts.append(str(c["symbol"]))
            if c.get("path"):
                doc_parts.append(str(c["path"]))
            code = c.get("code") or c.get("snippet") or ""
            if code:
                doc_parts.append(code[:500])
            doc = " ".join(doc_parts) if doc_parts else "empty"
            pairs.append((query, doc))
        return rerank_local(pairs)
    except Exception:
        return None


def rerank_baseline(query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Baseline: no reranking, just return as-is."""
    return candidates


def rerank_recursive(
    query: str,
    candidates: List[Dict[str, Any]],
    n_iterations: int = 3,
) -> List[Dict[str, Any]]:
    """Recursive reranker (no learning)."""
    try:
        from scripts.rerank_recursive import RecursiveReranker
        reranker = RecursiveReranker(n_iterations=n_iterations, dim=256)
        initial_scores = [c.get("score", 0) for c in candidates]
        return reranker.rerank(query, candidates, initial_scores)
    except Exception as e:
        print(f"Warning: Recursive rerank failed: {e}", file=sys.stderr)
        return candidates


def rerank_learning(
    query: str,
    candidates: List[Dict[str, Any]],
    collection: str = "eval",
) -> List[Dict[str, Any]]:
    """Learning reranker (uses trained weights)."""
    try:
        from scripts.rerank_recursive import rerank_with_learning
        return rerank_with_learning(
            query=query,
            candidates=candidates,
            limit=len(candidates),
            n_iterations=3,
            learn_from_onnx=False,  # Eval mode: no training
            collection=collection,
        )
    except Exception as e:
        print(f"Warning: Learning rerank failed: {e}", file=sys.stderr)
        return candidates


def rerank_onnx(query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """ONNX reranker (teacher/ground truth)."""
    scores = get_onnx_scores(query, candidates)
    if scores is None:
        return candidates
    for c, s in zip(candidates, scores):
        c["score"] = s
    return sorted(candidates, key=lambda x: x.get("score", 0), reverse=True)


def compute_mrr(ranked_paths: List[str], relevant_paths: List[str], k: int = 10) -> float:
    """Compute Mean Reciprocal Rank."""
    # Deduplicate while preserving order (paths can repeat due to multi-span retrieval)
    rel_set = {p for p in relevant_paths if p}
    if not rel_set:
        return 0.0

    seen = set()
    uniq_ranked: List[str] = []
    for p in ranked_paths:
        if not p or p in seen:
            continue
        seen.add(p)
        uniq_ranked.append(p)
        if len(uniq_ranked) >= k:
            break

    for i, path in enumerate(uniq_ranked):
        if path in rel_set:
            return 1.0 / (i + 1)
    return 0.0


def compute_recall_at_k(ranked_paths: List[str], relevant_paths: List[str], k: int) -> float:
    """Compute Recall@k."""
    rel_set = {p for p in relevant_paths if p}
    if not rel_set:
        return 0.0

    seen = set()
    uniq_ranked: List[str] = []
    for p in ranked_paths:
        if not p or p in seen:
            continue
        seen.add(p)
        uniq_ranked.append(p)
        if len(uniq_ranked) >= k:
            break

    found = len(set(uniq_ranked) & rel_set)
    return found / len(rel_set)


def eval_single_query(
    query: str,
    mode: str,
    candidates: List[Dict[str, Any]],
    reference_paths: List[str],
) -> EvalResult:
    """Evaluate a single query with a specific reranking mode."""
    start = time.perf_counter()

    if mode == "baseline":
        reranked = rerank_baseline(query, candidates)
    elif mode == "recursive":
        reranked = rerank_recursive(query, candidates)
    elif mode == "learning":
        reranked = rerank_learning(query, candidates)
    elif mode == "onnx":
        reranked = rerank_onnx(query, candidates)
    else:
        reranked = candidates

    latency_ms = (time.perf_counter() - start) * 1000

    # Export unique paths (avoid duplicates from multi-span retrieval)
    ranked_paths: List[str] = []
    ranked_scores: List[float] = []
    seen_paths = set()
    for c in reranked:
        path = str(c.get("path", "") or "")
        if not path or path in seen_paths:
            continue
        seen_paths.add(path)
        ranked_paths.append(path)
        ranked_scores.append(float(c.get("score", 0)))
        if len(ranked_paths) >= 10:
            break

    mrr = compute_mrr(ranked_paths, reference_paths)
    recall_5 = compute_recall_at_k(ranked_paths, reference_paths, 5)
    recall_10 = compute_recall_at_k(ranked_paths, reference_paths, 10)

    return EvalResult(
        query=query,
        mode=mode,
        latency_ms=latency_ms,
        top_k_paths=ranked_paths,
        top_k_scores=ranked_scores,
        mrr=mrr,
        recall_at_5=recall_5,
        recall_at_10=recall_10,
    )


def run_eval(
    queries: List[str],
    modes: List[str],
    use_onnx_reference: bool = True,
) -> Dict[str, EvalSummary]:
    """Run evaluation across all queries and modes."""
    summaries: Dict[str, EvalSummary] = {}

    # Warmup: pre-cache embeddings for all queries (cold start is not representative)
    print("Warming up embedding cache...")
    for query in queries:
        candidates = get_candidates(query)
        if candidates and "learning" in modes:
            # Run once to cache embeddings
            rerank_learning(query, copy.deepcopy(candidates))
    print("Warmup complete.")

    for mode in modes:
        results = []
        latencies = []

        for query in queries:
            candidates = get_candidates(query)
            if not candidates:
                continue

            # Use ONNX top-5 as "relevant" ground truth
            # Deep copy to prevent score mutation from leaking between modes
            reference_paths = []
            if use_onnx_reference:
                onnx_ranked = rerank_onnx(query, copy.deepcopy(candidates))
                reference_paths = [c.get("path", "") for c in onnx_ranked[:5]]

            result = eval_single_query(query, mode, copy.deepcopy(candidates), reference_paths)
            results.append(result)
            latencies.append(result.latency_ms)

        if not results:
            continue

        latencies_arr = np.array(latencies)
        summaries[mode] = EvalSummary(
            mode=mode,
            num_queries=len(results),
            mrr_mean=np.mean([r.mrr for r in results]),
            recall_at_5_mean=np.mean([r.recall_at_5 for r in results]),
            recall_at_10_mean=np.mean([r.recall_at_10 for r in results]),
            latency_p50_ms=float(np.percentile(latencies_arr, 50)),
            latency_p95_ms=float(np.percentile(latencies_arr, 95)),
            latency_p99_ms=float(np.percentile(latencies_arr, 99)),
            results=results,
        )

    return summaries


def print_summary(summaries: Dict[str, EvalSummary]):
    """Print evaluation summary table."""
    print("\n" + "=" * 80)
    print("RERANKER EVALUATION SUMMARY")
    print("=" * 80)
    print(f"{'Mode':<12} {'MRR':<8} {'R@5':<8} {'R@10':<8} {'p50ms':<8} {'p95ms':<8} {'p99ms':<8}")
    print("-" * 80)

    # Track for comparison
    baseline_mrr = summaries.get("baseline", EvalSummary("baseline", 0, 0, 0, 0, 0, 0, 0)).mrr_mean
    onnx_mrr = summaries.get("onnx", EvalSummary("onnx", 0, 0, 0, 0, 0, 0, 0)).mrr_mean
    onnx_p50 = summaries.get("onnx", EvalSummary("onnx", 0, 0, 0, 0, 0, 0, 0)).latency_p50_ms
    learning_mrr = summaries.get("learning", EvalSummary("learning", 0, 0, 0, 0, 0, 0, 0)).mrr_mean
    learning_p50 = summaries.get("learning", EvalSummary("learning", 0, 0, 0, 0, 0, 0, 0)).latency_p50_ms

    for mode, summary in summaries.items():
        print(
            f"{mode:<12} "
            f"{summary.mrr_mean:<8.3f} "
            f"{summary.recall_at_5_mean:<8.3f} "
            f"{summary.recall_at_10_mean:<8.3f} "
            f"{summary.latency_p50_ms:<8.1f} "
            f"{summary.latency_p95_ms:<8.1f} "
            f"{summary.latency_p99_ms:<8.1f}"
        )
    print("=" * 80)

    # Self-improving search analysis
    print("\nSELF-IMPROVING SEARCH ANALYSIS:")
    print("-" * 50)

    if baseline_mrr > 0:
        learning_vs_baseline = ((learning_mrr - baseline_mrr) / baseline_mrr) * 100
        print(f"Learning vs Baseline: {learning_vs_baseline:+.1f}% MRR improvement")

    if onnx_p50 > 0 and learning_p50 > 0:
        speedup = onnx_p50 / learning_p50
        print(f"Learning vs ONNX:     {speedup:.1f}x faster ({learning_p50:.1f}ms vs {onnx_p50:.1f}ms)")

    if onnx_mrr > 0:
        distill_quality = (learning_mrr / onnx_mrr) * 100
        print(f"Distillation quality: {distill_quality:.1f}% of ONNX MRR")

    print("-" * 50)


def main():
    parser = argparse.ArgumentParser(description="Offline reranker evaluation")
    parser.add_argument("--queries", type=str, help="JSON file with query list")
    parser.add_argument("--output", type=str, help="Output JSON file for results")
    parser.add_argument("--ablations", action="store_true", help="Run all ablation modes")
    parser.add_argument("--modes", type=str, default="baseline,recursive,learning",
                        help="Comma-separated modes to evaluate")
    args = parser.parse_args()

    queries = DEFAULT_EVAL_QUERIES
    if args.queries:
        with open(args.queries) as f:
            queries = json.load(f)

    modes = args.modes.split(",")
    if args.ablations:
        modes = ["baseline", "recursive", "learning", "onnx"]

    print(f"Running evaluation: {len(queries)} queries, modes: {modes}")
    summaries = run_eval(queries, modes)
    print_summary(summaries)

    if args.output:
        output_data = {mode: asdict(s) for mode, s in summaries.items()}
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()

