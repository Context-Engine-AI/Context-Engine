#!/usr/bin/env python3
"""
Quality evaluation harness for retrieval benchmarks.

Computes Hit@k and MRR against a gold query set.

Usage:
    python bench/eval_quality.py \
        --collection ctx-public-v1-bge \
        --model "BAAI/bge-base-en-v1.5" \
        --gold-file bench/gold/public_v1.queries.jsonl \
        --limit 20

    # With config toggles
    python bench/eval_quality.py \
        --collection ctx-public-v1-bge \
        --model "BAAI/bge-base-en-v1.5" \
        --gold-file bench/gold/public_v1.queries.jsonl \
        --config-id no_rerank \
        -- --no-rerank
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

BENCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCH_DIR.parent


@dataclass
class GoldQuery:
    """A gold-standard query with expected relevant files."""
    id: str
    query: str
    repo: str
    relevant: List[str]  # List of relative paths
    task_type: str = "navigation"
    k: int = 20
    notes: str = ""


@dataclass
class QueryResult:
    """Result of evaluating a single query."""
    query_id: str
    query: str
    retrieved: List[str]  # Paths returned by search
    relevant: Set[str]  # Gold relevant paths
    hits: Dict[int, bool]  # hit@k for various k
    rr: float  # reciprocal rank
    latency_s: float


@dataclass
class EvalResult:
    """Aggregated evaluation results."""
    dataset_id: str
    collection: str
    model: str
    config_id: str
    n_queries: int
    metrics: Dict[str, float]
    per_query: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "collection": self.collection,
            "model": self.model,
            "config_id": self.config_id,
            "n_queries": self.n_queries,
            "metrics": self.metrics,
            "per_query": self.per_query,
            "timestamp": self.timestamp,
        }


def load_gold_queries(path: Path, repo_filter: Optional[str] = None) -> List[GoldQuery]:
    """Load gold queries from JSONL file."""
    queries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            
            if repo_filter and obj.get("repo") != repo_filter:
                continue
            
            relevant_paths = [r["path"] for r in obj.get("relevant", [])]
            
            queries.append(GoldQuery(
                id=obj["id"],
                query=obj["query"],
                repo=obj["repo"],
                relevant=relevant_paths,
                task_type=obj.get("task_type", "navigation"),
                k=obj.get("k", 20),
                notes=obj.get("notes", ""),
            ))
    return queries


def normalize_path(path: str) -> str:
    """Normalize path for comparison (remove leading ./ and trailing /)."""
    p = path.strip()
    while p.startswith("./"):
        p = p[2:]
    while p.startswith("/"):
        p = p[1:]
    return p.rstrip("/")


def run_search(
    query: str,
    collection: str,
    model: str,
    limit: int,
    extra_args: Sequence[str],
    env_base: Dict[str, str],
) -> tuple[List[str], float]:
    """Run hybrid search and return retrieved paths + latency."""
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "hybrid_search.py"),
        "--collection", collection,
        "--limit", str(limit),
        "--per-path", "1",
        "--json",
        "-q", query,
    ]
    cmd.extend(list(extra_args))
    
    env = dict(env_base)
    env["EMBEDDING_MODEL"] = model
    
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    latency = time.time() - t0
    
    if proc.returncode != 0:
        print(f"[eval] WARNING: search failed for query: {query[:50]}...")
        return [], latency
    
    # Parse results
    paths = []
    for line in proc.stdout.strip().splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
            if "results" in obj:
                for r in obj["results"]:
                    if "path" in r:
                        paths.append(normalize_path(r["path"]))
            elif "path" in obj:
                paths.append(normalize_path(obj["path"]))
        except json.JSONDecodeError:
            continue
    
    return paths, latency


def compute_hits(retrieved: List[str], relevant: Set[str], k_values: List[int]) -> Dict[int, bool]:
    """Compute hit@k for various k values."""
    hits = {}
    for k in k_values:
        top_k = set(retrieved[:k])
        hits[k] = bool(top_k & relevant)
    return hits


def compute_rr(retrieved: List[str], relevant: Set[str]) -> float:
    """Compute reciprocal rank (1/rank of first relevant result)."""
    for i, path in enumerate(retrieved):
        if path in relevant:
            return 1.0 / (i + 1)
    return 0.0


def evaluate_query(
    gold: GoldQuery,
    collection: str,
    model: str,
    limit: int,
    extra_args: Sequence[str],
    env_base: Dict[str, str],
    k_values: List[int],
) -> QueryResult:
    """Evaluate a single gold query."""
    retrieved, latency = run_search(
        query=gold.query,
        collection=collection,
        model=model,
        limit=limit,
        extra_args=extra_args,
        env_base=env_base,
    )
    
    # Normalize relevant paths
    relevant = {normalize_path(p) for p in gold.relevant}
    
    # For matching, we check if retrieved path ends with any relevant path
    # (handles cases where collection has full paths but gold has relative)
    def matches_relevant(retrieved_path: str, relevant_set: Set[str]) -> bool:
        rp = normalize_path(retrieved_path)
        for rel in relevant_set:
            if rp.endswith(rel) or rel.endswith(rp) or rp == rel:
                return True
        return False
    
    # Convert retrieved to matched/unmatched
    matched_retrieved = []
    for rp in retrieved:
        if matches_relevant(rp, relevant):
            matched_retrieved.append(rp)
    
    hits = compute_hits(retrieved, relevant, k_values)
    
    # Recompute hits using fuzzy matching
    hits_fuzzy = {}
    for k in k_values:
        top_k = retrieved[:k]
        hits_fuzzy[k] = any(matches_relevant(p, relevant) for p in top_k)
    
    # Compute RR with fuzzy matching
    rr = 0.0
    for i, path in enumerate(retrieved):
        if matches_relevant(path, relevant):
            rr = 1.0 / (i + 1)
            break
    
    return QueryResult(
        query_id=gold.id,
        query=gold.query,
        retrieved=retrieved[:limit],
        relevant=relevant,
        hits=hits_fuzzy,
        rr=rr,
        latency_s=latency,
    )


def aggregate_results(
    results: List[QueryResult],
    k_values: List[int],
) -> Dict[str, float]:
    """Aggregate per-query results into summary metrics."""
    if not results:
        return {}
    
    n = len(results)
    metrics = {}
    
    # Hit@k (mean over queries)
    for k in k_values:
        hits = [1.0 if r.hits.get(k, False) else 0.0 for r in results]
        metrics[f"hit@{k}"] = sum(hits) / n
    
    # MRR (mean reciprocal rank)
    mrr = sum(r.rr for r in results) / n
    metrics["mrr"] = round(mrr, 4)
    
    # Latency stats
    latencies = [r.latency_s for r in results]
    metrics["latency_mean"] = round(sum(latencies) / n, 4)
    metrics["latency_p50"] = round(sorted(latencies)[n // 2], 4)
    if n >= 20:
        metrics["latency_p95"] = round(sorted(latencies)[int(n * 0.95)], 4)
    
    # Round hit@k
    for k in k_values:
        metrics[f"hit@{k}"] = round(metrics[f"hit@{k}"], 4)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval quality against gold query set"
    )
    parser.add_argument(
        "--collection", required=True,
        help="Qdrant collection to evaluate"
    )
    parser.add_argument(
        "--model", required=True,
        help="Embedding model name (for EMBEDDING_MODEL env var)"
    )
    parser.add_argument(
        "--gold-file", required=True,
        help="Path to gold queries JSONL file"
    )
    parser.add_argument(
        "--repo", type=str, default=None,
        help="Filter queries to specific repo (e.g., kubernetes/kubernetes)"
    )
    parser.add_argument(
        "--limit", type=int, default=20,
        help="Number of results to retrieve per query"
    )
    parser.add_argument(
        "--k-values", type=str, default="1,3,5,10",
        help="Comma-separated k values for hit@k"
    )
    parser.add_argument(
        "--config-id", type=str, default="default",
        help="Config variant identifier for output"
    )
    parser.add_argument(
        "--dataset-id", type=str, default=None,
        help="Dataset ID for output (defaults to gold file stem)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON file path"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print per-query results"
    )
    
    args, extra = parser.parse_known_args()
    
    # Handle -- separator for extra args to hybrid_search
    if extra and extra[0] == "--":
        extra = extra[1:]
    
    # Parse k values
    k_values = [int(k.strip()) for k in args.k_values.split(",")]
    
    # Load gold queries
    gold_path = Path(args.gold_file)
    if not gold_path.exists():
        print(f"ERROR: Gold file not found: {gold_path}")
        sys.exit(1)
    
    queries = load_gold_queries(gold_path, repo_filter=args.repo)
    if not queries:
        print(f"ERROR: No queries loaded from {gold_path}")
        sys.exit(1)
    
    dataset_id = args.dataset_id or gold_path.stem
    
    print(f"[eval] Collection: {args.collection}")
    print(f"[eval] Model: {args.model}")
    print(f"[eval] Gold queries: {len(queries)}")
    print(f"[eval] Config: {args.config_id}")
    print(f"[eval] k values: {k_values}")
    if extra:
        print(f"[eval] Extra args: {extra}")
    
    # Run evaluation
    env_base = dict(os.environ)
    results: List[QueryResult] = []
    
    for i, gold in enumerate(queries):
        result = evaluate_query(
            gold=gold,
            collection=args.collection,
            model=args.model,
            limit=args.limit,
            extra_args=extra,
            env_base=env_base,
            k_values=k_values,
        )
        results.append(result)
        
        if args.verbose:
            hit_str = " ".join(f"@{k}:{'✓' if result.hits.get(k) else '✗'}" for k in k_values)
            print(f"  [{i+1}/{len(queries)}] {gold.id}: RR={result.rr:.2f} {hit_str}")
        else:
            print(f"  [{i+1}/{len(queries)}] {gold.id}: RR={result.rr:.2f}", end="\r")
    
    print()  # Clear line
    
    # Aggregate
    metrics = aggregate_results(results, k_values)
    
    # Build output
    eval_result = EvalResult(
        dataset_id=dataset_id,
        collection=args.collection,
        model=args.model,
        config_id=args.config_id,
        n_queries=len(queries),
        metrics=metrics,
        per_query=[
            {
                "id": r.query_id,
                "rr": r.rr,
                "hits": {f"@{k}": v for k, v in r.hits.items()},
                "latency_s": round(r.latency_s, 4),
            }
            for r in results
        ],
        timestamp=datetime.now().isoformat(),
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("QUALITY METRICS")
    print("=" * 60)
    print(f"  MRR: {metrics.get('mrr', 0):.4f}")
    for k in k_values:
        print(f"  Hit@{k}: {metrics.get(f'hit@{k}', 0):.4f}")
    print(f"  Latency (mean): {metrics.get('latency_mean', 0):.3f}s")
    print(f"  Latency (p50): {metrics.get('latency_p50', 0):.3f}s")
    
    # Output JSON
    output_dict = eval_result.to_dict()
    print(json.dumps(output_dict))
    
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output_dict, f, indent=2)
        print(f"\n[eval] Results saved to {args.output}")


if __name__ == "__main__":
    main()
