#!/usr/bin/env python3
"""
Unified Benchmark Matrix Pipeline.

Orchestrates corpus-based comparison across embedding models and rerankers:
1. Requires source collection (fail-fast)
2. Builds variant collections (copy/re-embed per model)
3. Runs eval harness + benchmarks per variant
4. Outputs comparison matrix

Usage:
    python bench/run_matrix.py \
        --src Context-Engine-41e67959 \
        --models "BAAI/bge-base-en-v1.5,sentence-transformers/all-MiniLM-L6-v2" \
        --benchmarks "eval,router,rrf" \
        --output results/bench_matrix.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass
class ModelVariant:
    """A benchmark variant with a specific embedding model."""
    name: str  # e.g., "bench-bge"
    model: str  # e.g., "BAAI/bge-base-en-v1.5"
    collection: str  # Qdrant collection name


@dataclass
class ConfigVariant:
    """A config variant for A/B testing."""
    id: str  # e.g., "no_rerank"
    name: str  # e.g., "No Reranker"
    args: List[str]  # e.g., ["--no-rerank"]
    description: str = ""


def load_configs(config_file: Optional[str]) -> List[ConfigVariant]:
    """Load config variants from JSON file."""
    if not config_file:
        return [ConfigVariant(id="default", name="Default", args=[])]
    
    path = Path(config_file)
    if not path.exists():
        print(f"[matrix] WARNING: Config file not found: {config_file}")
        return [ConfigVariant(id="default", name="Default", args=[])]
    
    with open(path) as f:
        data = json.load(f)
    
    configs = []
    for cfg in data.get("configs", []):
        configs.append(ConfigVariant(
            id=cfg["id"],
            name=cfg.get("name", cfg["id"]),
            args=cfg.get("args", []),
            description=cfg.get("description", ""),
        ))
    
    return configs if configs else [ConfigVariant(id="default", name="Default", args=[])]


@dataclass
class MatrixResult:
    """Results from running the benchmark matrix."""
    source: str
    timestamp: str
    models: List[str]
    benchmarks: List[str]
    matrix: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    comparison: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "timestamp": self.timestamp,
            "models": self.models,
            "benchmarks": self.benchmarks,
            "matrix": self.matrix,
            "comparison": self.comparison,
            "errors": self.errors,
        }


def require_source_collection(collection: str, qdrant_url: str, min_points: int = 100) -> int:
    """Fail-fast: require source collection to exist with sufficient points."""
    from qdrant_client import QdrantClient
    
    client = QdrantClient(url=qdrant_url)
    try:
        info = client.get_collection(collection)
        count = int(info.points_count or 0)
    except Exception as e:
        raise ValueError(f"Source collection '{collection}' not accessible: {e}")
    
    if count < min_points:
        raise ValueError(
            f"Source collection '{collection}' has {count} points, "
            f"need at least {min_points} for meaningful benchmarks."
        )
    
    print(f"[matrix] Source collection '{collection}' verified: {count} points")
    return count


def build_variant_collections(
    src: str,
    models: List[str],
    qdrant_url: str,
    skip_existing: bool = True,
) -> List[ModelVariant]:
    """Build variant collections for each embedding model."""
    from scripts.utils import sanitize_vector_name
    from qdrant_client import QdrantClient
    
    client = QdrantClient(url=qdrant_url)
    variants = []
    
    for model in models:
        # Generate collection name from model
        safe_name = sanitize_vector_name(model)
        coll_name = f"bench-{safe_name[:30]}"  # Truncate for sanity
        
        # Check if already exists
        try:
            info = client.get_collection(coll_name)
            count = int(info.points_count or 0)
            if skip_existing and count > 0:
                print(f"[matrix] Variant '{coll_name}' exists with {count} points, skipping rebuild")
                variants.append(ModelVariant(name=safe_name, model=model, collection=coll_name))
                continue
        except Exception:
            pass  # Collection doesn't exist, will create
        
        # Build variant via copy-coll.py
        print(f"[matrix] Building variant '{coll_name}' with model '{model}'...")
        
        # Determine if this is the first model (clone) or needs re-embed
        is_first = (model == models[0])
        
        cmd = [
            sys.executable,
            str(REPO_ROOT / "bench" / "copy-coll.py"),
            "--qdrant-url", qdrant_url,
            "--src", src,
        ]
        
        if is_first:
            # First model: clone as-is (assume source uses this model)
            cmd.extend(["--bge-dest", coll_name, "--bge-model", model, "--skip-minilm"])
        else:
            # Other models: re-embed
            cmd.extend(["--minilm-dest", coll_name, "--minilm-model", model, "--skip-bge"])
        
        t0 = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True)
        dt = time.time() - t0
        
        if proc.returncode != 0:
            print(f"[matrix] ERROR building {coll_name}: {proc.stderr[:500]}")
            continue
        
        print(f"[matrix] Built '{coll_name}' in {dt:.1f}s")
        variants.append(ModelVariant(name=safe_name, model=model, collection=coll_name))
    
    return variants


def run_benchmarks_for_variant(
    variant: ModelVariant,
    benchmarks: List[str],
    qdrant_url: str,
    query_file: Optional[str] = None,
    gold_file: Optional[str] = None,
    repeats: int = 5,
    warmup: int = 1,
    config_id: str = "default",
    extra_args: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run selected benchmarks against a specific variant collection."""
    results: Dict[str, Any] = {
        "model": variant.model,
        "collection": variant.collection,
    }
    
    # Set up environment for this variant
    env = os.environ.copy()
    env["COLLECTION_NAME"] = variant.collection
    env["EMBEDDING_MODEL"] = variant.model
    env["QDRANT_URL"] = qdrant_url
    env["BENCH_STRICT"] = "1"  # Fail-fast mode
    
    # Run benchmarks via run_all.py or directly
    if benchmarks:
        bench_components = ",".join(benchmarks)
        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "benchmarks" / "run_all.py"),
            "--components",
        ] + benchmarks
        
        print(f"[matrix] Running benchmarks {benchmarks} for {variant.collection}...")
        t0 = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
        dt = time.time() - t0
        
        results["benchmark_time_s"] = round(dt, 2)
        
        if proc.returncode == 0:
            # Try to parse JSON from output
            for line in reversed(proc.stdout.strip().splitlines()):
                if line.startswith("{"):
                    try:
                        results["benchmarks"] = json.loads(line)
                        break
                    except json.JSONDecodeError:
                        pass
        else:
            results["error"] = proc.stderr[:500] if proc.stderr else "Unknown error"
    
    # Run quality eval if gold file provided
    if gold_file:
        print(f"[matrix] Running quality eval for {variant.collection}...")
        quality_cmd = [
            sys.executable,
            str(REPO_ROOT / "bench" / "eval_quality.py"),
            "--collection", variant.collection,
            "--model", variant.model,
            "--gold-file", gold_file,
            "--config-id", config_id,
            "--limit", "20",
        ]
        if extra_args:
            quality_cmd.append("--")
            quality_cmd.extend(extra_args)
        
        proc = subprocess.run(quality_cmd, capture_output=True, text=True, env=env)
        if proc.returncode == 0:
            for line in reversed(proc.stdout.strip().splitlines()):
                if line.startswith("{"):
                    try:
                        quality_data = json.loads(line)
                        results["quality"] = quality_data.get("metrics", {})
                        break
                    except json.JSONDecodeError:
                        pass
        else:
            results["quality_error"] = proc.stderr[:300] if proc.stderr else "Unknown error"
    
    # Run latency eval with repeats
    print(f"[matrix] Running latency eval for {variant.collection}...")
    eval_cmd = [
        sys.executable,
        str(REPO_ROOT / "bench" / "eval.py"),
        "--target", variant.collection, variant.model,
        "--target", variant.collection, variant.model,  # Dummy second target (required)
        "--repeats", str(repeats),
        "--warmup", str(warmup),
    ]
    if query_file:
        eval_cmd.extend(["--query-file", query_file])
    
    proc = subprocess.run(eval_cmd, capture_output=True, text=True, env=env)
    if proc.returncode == 0:
        for line in reversed(proc.stdout.strip().splitlines()):
            if line.startswith("{"):
                try:
                    latency_data = json.loads(line)
                    results["latency"] = latency_data.get("results", {}).get(variant.collection, {})
                    break
                except json.JSONDecodeError:
                    pass
    
    return results


def compute_comparison(matrix: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Compute comparison metrics across variants."""
    if len(matrix) < 2:
        return {}
    
    comparison: Dict[str, Any] = {}
    
    # Extract MRR values (prefer quality metrics from eval_quality.py)
    mrr_values = {}
    hit_at_5 = {}
    latency_p50 = {}
    for name, data in matrix.items():
        # First try quality metrics from eval_quality.py
        quality = data.get("quality", {})
        if quality.get("mrr"):
            mrr_values[name] = quality["mrr"]
            if quality.get("hit@5"):
                hit_at_5[name] = quality["hit@5"]
        else:
            # Fallback to legacy benchmarks
            benchmarks = data.get("benchmarks", {}).get("components", {})
            if "eval_harness" in benchmarks:
                mrr = benchmarks["eval_harness"].get("metrics", {}).get("mrr", 0)
                mrr_values[name] = mrr
        lat = data.get("latency", {}).get("p50", 0)
        if lat:
            latency_p50[name] = lat
    
    if mrr_values:
        best_mrr_name = max(mrr_values, key=lambda k: mrr_values[k])
        worst_mrr_name = min(mrr_values, key=lambda k: mrr_values[k])
        comparison["mrr"] = {
            "best": {"name": best_mrr_name, "value": mrr_values[best_mrr_name]},
            "worst": {"name": worst_mrr_name, "value": mrr_values[worst_mrr_name]},
            "delta": round(mrr_values[best_mrr_name] - mrr_values[worst_mrr_name], 4),
        }
    
    if latency_p50:
        fastest_name = min(latency_p50, key=lambda k: latency_p50[k])
        slowest_name = max(latency_p50, key=lambda k: latency_p50[k])
        if latency_p50[slowest_name] > 0:
            improvement_pct = (1 - latency_p50[fastest_name] / latency_p50[slowest_name]) * 100
        else:
            improvement_pct = 0
        comparison["latency"] = {
            "fastest": {"name": fastest_name, "p50": latency_p50[fastest_name]},
            "slowest": {"name": slowest_name, "p50": latency_p50[slowest_name]},
            "improvement_pct": round(improvement_pct, 1),
        }
    
    # Generate recommendation
    if comparison.get("mrr") and comparison.get("latency"):
        mrr_best = comparison["mrr"]["best"]["name"]
        lat_best = comparison["latency"]["fastest"]["name"]
        if mrr_best == lat_best:
            comparison["recommendation"] = f"{mrr_best} wins on both quality and latency"
        else:
            comparison["recommendation"] = (
                f"{mrr_best} wins on quality (MRR); "
                f"{lat_best} wins on latency"
            )
    
    return comparison


async def run_matrix(
    src: str,
    models: List[str],
    benchmarks: List[str],
    qdrant_url: str,
    query_file: Optional[str] = None,
    gold_file: Optional[str] = None,
    config_file: Optional[str] = None,
    repeats: int = 5,
    warmup: int = 1,
    skip_existing: bool = True,
    min_points: int = 100,
) -> MatrixResult:
    """Run the full benchmark matrix over (model × config) pairs."""
    # Load config variants
    configs = load_configs(config_file)
    
    result = MatrixResult(
        source=src,
        timestamp=datetime.now().isoformat(),
        models=models,
        benchmarks=benchmarks,
    )
    
    # Step 1: Verify source collection
    try:
        require_source_collection(src, qdrant_url, min_points)
    except ValueError as e:
        result.errors.append(str(e))
        return result
    
    # Step 2: Build variant collections (one per model)
    variants = build_variant_collections(src, models, qdrant_url, skip_existing)
    if not variants:
        result.errors.append("No variant collections could be built")
        return result
    
    # Step 3: Run benchmarks for each (model × config) pair
    for variant in variants:
        for config in configs:
            # Create unique key for this (model, config) combination
            if len(configs) > 1:
                matrix_key = f"{variant.name}__{config.id}"
            else:
                matrix_key = variant.name
            
            print(f"\n{'='*60}")
            print(f"[matrix] Evaluating: {variant.name} + {config.name}")
            print(f"[matrix] Config args: {config.args}")
            print(f"{'='*60}")
            
            variant_results = run_benchmarks_for_variant(
                variant, benchmarks, qdrant_url, query_file, gold_file,
                repeats, warmup, config.id, config.args if config.args else None
            )
            variant_results["config_id"] = config.id
            variant_results["config_name"] = config.name
            result.matrix[matrix_key] = variant_results
    
    # Step 4: Compute comparison
    result.comparison = compute_comparison(result.matrix)
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Run unified benchmark matrix across embedding models"
    )
    parser.add_argument(
        "--src", required=True,
        help="Source collection to use as benchmark corpus"
    )
    parser.add_argument(
        "--models", default="BAAI/bge-base-en-v1.5,sentence-transformers/all-MiniLM-L6-v2",
        help="Comma-separated list of embedding models to compare"
    )
    parser.add_argument(
        "--benchmarks", default="eval,router",
        help="Comma-separated list of benchmarks to run (eval,router,rrf,refrag,grounding)"
    )
    parser.add_argument(
        "--qdrant-url", default=os.environ.get("QDRANT_URL", "http://localhost:6333"),
        help="Qdrant server URL"
    )
    parser.add_argument(
        "--query-file", type=str, default=None,
        help="Path to query file for latency eval (one query per line)"
    )
    parser.add_argument(
        "--gold-file", type=str, default=None,
        help="Path to gold queries JSONL for quality eval (Hit@k, MRR)"
    )
    parser.add_argument(
        "--config-file", type=str, default=None,
        help="Path to A/B config JSON file (e.g., bench/configs/ctx_ab_configs.json)"
    )
    parser.add_argument(
        "--repeats", type=int, default=5,
        help="Number of benchmark repeats for latency stats"
    )
    parser.add_argument(
        "--warmup", type=int, default=1,
        help="Number of warmup runs before measurement"
    )
    parser.add_argument(
        "--rebuild", action="store_true",
        help="Rebuild variant collections even if they exist"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON file path"
    )
    parser.add_argument(
        "--min-points", type=int, default=100,
        help="Minimum points required in source collection (default: 100)"
    )
    
    args = parser.parse_args()
    
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    benchmarks = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
    
    print(f"[matrix] Source: {args.src}")
    print(f"[matrix] Models: {models}")
    print(f"[matrix] Benchmarks: {benchmarks}")
    print(f"[matrix] Qdrant: {args.qdrant_url}")
    if args.gold_file:
        print(f"[matrix] Gold file: {args.gold_file}")
    if args.config_file:
        print(f"[matrix] Config file: {args.config_file}")
    
    result = asyncio.run(run_matrix(
        src=args.src,
        models=models,
        benchmarks=benchmarks,
        qdrant_url=args.qdrant_url,
        query_file=args.query_file,
        gold_file=args.gold_file,
        config_file=args.config_file,
        repeats=args.repeats,
        warmup=args.warmup,
        skip_existing=not args.rebuild,
        min_points=args.min_points,
    ))
    
    # Print summary
    print(f"\n{'='*60}")
    print("MATRIX RESULTS")
    print(f"{'='*60}")
    
    for name, data in result.matrix.items():
        print(f"\n{name}:")
        if "latency" in data:
            lat = data["latency"]
            print(f"  Latency: p50={lat.get('p50', 0):.3f}s p95={lat.get('p95', 0):.3f}s")
        # Print quality metrics from eval_quality.py
        if "quality" in data:
            q = data["quality"]
            print(f"  MRR: {q.get('mrr', 0):.4f}  Hit@5: {q.get('hit@5', 0):.4f}")
        else:
            # Fallback to legacy benchmarks
            benchmarks_data = data.get("benchmarks", {}).get("components", {})
            if "eval_harness" in benchmarks_data:
                metrics = benchmarks_data["eval_harness"].get("metrics", {})
                print(f"  MRR: {metrics.get('mrr', 0):.3f}")
    
    if result.comparison:
        print(f"\nCOMPARISON:")
        if "recommendation" in result.comparison:
            print(f"  → {result.comparison['recommendation']}")
    
    if result.errors:
        print(f"\nERRORS:")
        for e in result.errors:
            print(f"  ❌ {e}")
    
    # Write output
    output = result.to_dict()
    print(json.dumps(output, indent=2))
    
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\n[matrix] Results saved to {args.output}")


if __name__ == "__main__":
    main()
