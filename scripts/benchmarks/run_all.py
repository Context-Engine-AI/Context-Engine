#!/usr/bin/env python3
"""
Unified Benchmark Runner for Context-Engine

Runs all component benchmarks and generates a comprehensive report.
Uses unified reporting format from common.py.
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import unified metadata utilities
from scripts.benchmarks.common import BenchmarkMetadata
from scripts.benchmarks.common import resolve_collection_auto

# Ensure collection is set
if not os.environ.get("COLLECTION_NAME"):
    try:
        from scripts.workspace_state import get_collection_name
        os.environ["COLLECTION_NAME"] = get_collection_name() or "codebase"
    except Exception:
        pass
else:
    # If a collection is set but empty (common in multi-collection setups), pick a non-empty one.
    try:
        os.environ["COLLECTION_NAME"] = resolve_collection_auto(os.environ.get("COLLECTION_NAME"))
    except Exception:
        pass

print(
    f"[bench] Using QDRANT_URL={os.environ.get('QDRANT_URL', '')} "
    f"COLLECTION_NAME={os.environ.get('COLLECTION_NAME', '')}"
)


async def run_all_benchmarks(components: List[str]) -> Dict[str, Any]:
    """Run selected component benchmarks."""
    # Create unified report
    metadata = BenchmarkMetadata(
        benchmark_name="comprehensive",
        config={"components": components},
        environment={
            "collection": os.environ.get("COLLECTION_NAME", ""),
            "refrag_runtime": os.environ.get("REFRAG_RUNTIME", ""),
        },
    )
    
    results = {
        "metadata": {
            "benchmark_name": metadata.benchmark_name,
            "version": metadata.version,
            "timestamp": metadata.timestamp,
            "config": metadata.config,
            "environment": metadata.environment,
        },
        # Back-compat convenience for callers that expect a top-level timestamp.
        "timestamp": metadata.timestamp,
        "components": {},
        "aggregates": {},
    }
    
    if "eval" in components or "all" in components:
        try:
            from scripts.benchmarks.eval_harness import run_evaluation
            print("\n▶ Running Evaluation Harness...")
            report = await run_evaluation(name="eval")
            results["components"]["eval_harness"] = report.to_dict()
        except Exception as e:
            print(f"  Eval harness failed: {e}")
    
    if "trm" in components or "all" in components:
        try:
            from scripts.benchmarks.trm_bench import run_trm_benchmark
            print("\n▶ Running TRM/Reranker Benchmark...")
            report = await run_trm_benchmark(name="trm")
            results["components"]["trm_reranker"] = report.to_dict()
        except Exception as e:
            print(f"  TRM benchmark failed: {e}")
    
    if "refrag" in components or "all" in components:
        try:
            from scripts.benchmarks.refrag_bench import run_refrag_benchmark
            print("\n▶ Running ReFRAG Benchmark...")
            report = await run_refrag_benchmark(name="refrag")
            results["components"]["refrag"] = report.to_dict()
        except Exception as e:
            print(f"  ReFRAG benchmark failed: {e}")
    
    if "expand" in components or "all" in components:
        try:
            from scripts.benchmarks.expand_bench import run_expansion_benchmark
            print("\n▶ Running Query Expansion Benchmark...")
            report = await run_expansion_benchmark(name="expand")
            results["components"]["query_expansion"] = report.to_dict()
        except Exception as e:
            print(f"  Expansion benchmark failed: {e}")

    if "router" in components or "all" in components:
        try:
            from scripts.benchmarks.router_bench import run_router_benchmark
            print("\n▶ Running Router Benchmark...")
            report = await run_router_benchmark(name="router")
            results["components"]["router"] = report.to_dict()
        except Exception as e:
            print(f"  Router benchmark failed: {e}")

    if "rrf" in components or "all" in components:
        try:
            from scripts.benchmarks.rrf_quality import run_rrf_benchmark
            print("\n▶ Running RRF Quality Benchmark...")
            report = await run_rrf_benchmark()
            results["components"]["rrf_quality"] = report.to_dict()
        except Exception as e:
            print(f"  RRF quality benchmark failed: {e}")

    if "grounding" in components or "all" in components:
        try:
            from scripts.benchmarks.grounding_scorer import run_grounding_benchmark
            print("\n▶ Running Grounding Scorer Benchmark...")
            report = await run_grounding_benchmark()
            results["components"]["grounding"] = report.to_dict()
        except Exception as e:
            print(f"  Grounding benchmark failed: {e}")

    if "efficiency" in components or "all" in components:
        try:
            from scripts.benchmarks.efficiency_benchmark import run_benchmark as run_efficiency_benchmark, SCENARIOS
            print("\n▶ Running Efficiency Benchmark...")
            optimize = os.environ.get("EFFICIENCY_OPTIMIZE", "fusion+cache+dedup")
            report = await run_efficiency_benchmark(
                scenarios=list(SCENARIOS.keys()),
                baseline=False,
                optimize=optimize,
            )
            results["components"]["efficiency"] = report.to_full_dict()
        except Exception as e:
            print(f"  Efficiency benchmark failed: {e}")
    
    # Embedding model comparison (requires pre-built bench-* collections)
    if "embedding" in components:
        try:
            import subprocess
            import json as _json
            print("\n▶ Running Embedding Model Comparison...")
            # Get targets from env or use defaults
            targets = os.environ.get("EMBED_BENCH_TARGETS", "bench-bge:BAAI/bge-base-en-v1.5,bench-minilm:sentence-transformers/all-MiniLM-L6-v2")
            target_pairs = [t.split(":") for t in targets.split(",") if ":" in t]
            if len(target_pairs) < 2:
                raise ValueError("EMBED_BENCH_TARGETS needs at least 2 targets (coll:model pairs)")
            
            cmd = [sys.executable, str(PROJECT_ROOT / "bench" / "eval.py")]
            for coll, model in target_pairs:
                cmd.extend(["--target", coll.strip(), model.strip()])
            cmd.extend(["--repeats", os.environ.get("EMBED_BENCH_REPEATS", "5"), "--warmup", "1"])
            
            proc = subprocess.run(cmd, capture_output=True, text=True, env=os.environ.copy())
            if proc.returncode == 0:
                # Parse JSON output from last line
                for line in reversed(proc.stdout.strip().splitlines()):
                    if line.startswith("{"):
                        results["components"]["embedding"] = _json.loads(line)
                        break
            else:
                print(f"  Embedding bench stderr: {proc.stderr[:200]}")
        except Exception as e:
            print(f"  Embedding benchmark failed: {e}")
    
    # Generate recommendations
    results["recommendations"] = generate_recommendations(results["components"])
    
    return results


def generate_recommendations(components: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate optimization recommendations from benchmark results."""
    recs = []
    
    # Check eval harness MRR
    if "eval_harness" in components:
        mrr = components["eval_harness"].get("metrics", {}).get("mrr", 0)
        if mrr < 0.7:
            recs.append({
                "priority": "high",
                "component": "eval_harness",
                "action": "Enable learning reranker or tune hybrid weights",
                "impact": f"Current MRR {mrr:.2f} is below 0.7 target",
            })
    
    # Check ReFRAG grounding
    if "refrag" in components:
        grounding = components["refrag"].get("metrics", {}).get("grounding_rate", 0)
        if grounding < 0.8:
            recs.append({
                "priority": "medium",
                "component": "refrag",
                "action": "Increase MICRO_BUDGET_TOKENS or improve retrieval",
                "impact": f"Grounding rate {grounding:.0%} indicates insufficient context",
            })
    
    # Check TRM latency
    if "trm_reranker" in components:
        p90 = components["trm_reranker"].get("metrics", {}).get("p90_latency_ms", 0)
        if p90 > 500:
            recs.append({
                "priority": "medium",
                "component": "trm_reranker",
                "action": "Consider ONNX fallback or reduce candidate pool",
                "impact": f"P90 latency {p90:.0f}ms may impact UX",
            })

    # Check Router accuracy
    if "router" in components:
        acc = components["router"].get("metrics", {}).get("accuracy", 0)
        if acc < 0.8:
            recs.append({
                "priority": "medium",
                "component": "router",
                "action": "Tune router prompts/heuristics or add tool examples for weak intents",
                "impact": f"Router accuracy {acc:.0%} is below 80% target",
            })
    
    return recs


def print_summary(results: Dict[str, Any]):
    """Print summary report."""
    print("\n" + "=" * 70)
    print("COMPREHENSIVE BENCHMARK REPORT")
    print("=" * 70)
    ts = (results.get("metadata") or {}).get("timestamp") or results.get("timestamp", "")
    print(f"Timestamp: {ts}")
    print("-" * 70)
    
    for name, data in results.get("components", {}).items():
        # Most components have {"metrics": {...}}.
        # Efficiency uses {"summary": {...}, "details": {...}}.
        metrics: Any = {}
        if isinstance(data, dict):
            if isinstance(data.get("metrics"), dict):
                metrics = data.get("metrics") or {}
            elif isinstance(data.get("summary"), dict):
                metrics = data.get("summary") or {}
        print(f"\n{name.upper()}:")
        if isinstance(metrics, dict):
            for k, v in metrics.items():
                print(f"  {k}: {v}")
        else:
            print(f"  {metrics}")
    
    recs = results.get("recommendations", [])
    if recs:
        print("\n" + "-" * 70)
        print("RECOMMENDATIONS:")
        for r in recs:
            print(f"  [{r['priority'].upper()}] {r['component']}: {r['action']}")
    
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Context-Engine Unified Benchmark")
    parser.add_argument(
        "--components",
        nargs="+",
        default=["all"],
        choices=["all", "eval", "trm", "refrag", "expand", "router", "rrf", "grounding", "efficiency", "embedding"],
        help="Components to benchmark",
    )
    parser.add_argument("--output", type=str, help="Output JSON file")
    parser.add_argument("--collection", type=str, help="Override COLLECTION_NAME for this run")
    args = parser.parse_args()
    
    # Apply collection override before running benchmarks
    if args.collection:
        os.environ["COLLECTION_NAME"] = args.collection
        print(f"[bench] Collection override: {args.collection}")
    
    print("Starting comprehensive benchmark suite...")
    results = asyncio.run(run_all_benchmarks(args.components))
    print_summary(results)
    
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nFull report saved to: {args.output}")


if __name__ == "__main__":
    main()
