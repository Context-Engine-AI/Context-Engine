#!/usr/bin/env python3
"""
A/B Testing Infrastructure for Rerankers.

Provides:
1. Random assignment to reranker variants
2. Metric logging per variant
3. Aggregation and statistical comparison
4. Session-aware bucketing (same user gets same variant)

Usage:
    # In your search pipeline:
    from scripts.rerank_ab_test import ABTestManager, RerankerVariant

    ab = ABTestManager()
    variant = ab.get_variant(session_id="user_123")
    results = variant.rerank(query, candidates)
    ab.log_metrics(session_id, latency_ms=5.2, clicked_rank=1)

    # Analyze results:
    ab.print_summary()
"""

import os
import json
import time
import hashlib
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class VariantType(Enum):
    """Available reranker variants."""
    BASELINE = "baseline"  # No reranking, use initial scores
    ONNX = "onnx"  # ONNX cross-encoder
    RECURSIVE = "recursive"  # TRM-inspired recursive reranker
    RECURSIVE_ONNX = "recursive_onnx"  # ONNX + recursive refinement


@dataclass
class ABMetric:
    """Single metric observation."""
    session_id: str
    variant: str
    timestamp: float
    latency_ms: float = 0.0
    clicked_rank: Optional[int] = None  # 1-indexed rank of clicked result
    num_results: int = 0
    query_length: int = 0
    custom: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VariantStats:
    """Aggregated statistics for a variant."""
    variant: str
    n_observations: int = 0
    latency_sum: float = 0.0
    latency_sq_sum: float = 0.0  # For std calculation
    mrr_sum: float = 0.0  # Mean Reciprocal Rank
    clicks_at_1: int = 0
    clicks_at_3: int = 0
    clicks_at_5: int = 0
    total_clicks: int = 0

    def add_observation(self, metric: ABMetric):
        """Add a single observation."""
        self.n_observations += 1
        self.latency_sum += metric.latency_ms
        self.latency_sq_sum += metric.latency_ms ** 2

        if metric.clicked_rank is not None:
            self.total_clicks += 1
            self.mrr_sum += 1.0 / metric.clicked_rank
            if metric.clicked_rank == 1:
                self.clicks_at_1 += 1
            if metric.clicked_rank <= 3:
                self.clicks_at_3 += 1
            if metric.clicked_rank <= 5:
                self.clicks_at_5 += 1

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        if self.n_observations == 0:
            return {"variant": self.variant, "n_observations": 0}

        n = self.n_observations
        mean_latency = self.latency_sum / n
        var_latency = (self.latency_sq_sum / n) - (mean_latency ** 2)
        std_latency = var_latency ** 0.5 if var_latency > 0 else 0

        mrr = self.mrr_sum / self.total_clicks if self.total_clicks > 0 else 0

        return {
            "variant": self.variant,
            "n_observations": n,
            "latency_ms_mean": round(mean_latency, 2),
            "latency_ms_std": round(std_latency, 2),
            "mrr": round(mrr, 4),
            "click_rate_at_1": round(self.clicks_at_1 / n, 4) if n > 0 else 0,
            "click_rate_at_3": round(self.clicks_at_3 / n, 4) if n > 0 else 0,
            "click_rate_at_5": round(self.clicks_at_5 / n, 4) if n > 0 else 0,
            "total_clicks": self.total_clicks,
        }


class RerankerVariant:
    """Wrapper for a reranker variant with timing."""

    def __init__(self, variant_type: VariantType, rerank_fn: Callable):
        self.variant_type = variant_type
        self.rerank_fn = rerank_fn

    def rerank(self, query: str, candidates: List[Dict[str, Any]], **kwargs) -> tuple:
        """Rerank candidates and return (results, latency_ms)."""
        start = time.perf_counter()
        results = self.rerank_fn(query, candidates, **kwargs)
        latency_ms = (time.perf_counter() - start) * 1000
        return results, latency_ms


class ABTestManager:
    """
    Manages A/B testing for reranker variants.

    Features:
    - Consistent hashing for session assignment (same session = same variant)
    - Thread-safe metric logging
    - Persistent storage of metrics
    - Statistical comparison utilities
    """

    def __init__(
        self,
        variants: Optional[List[VariantType]] = None,
        weights: Optional[List[float]] = None,
        log_path: str = "data/ab_test_metrics.jsonl",
        experiment_id: Optional[str] = None,
    ):
        self.variants = variants or [VariantType.BASELINE, VariantType.RECURSIVE]
        self.weights = weights or [1.0] * len(self.variants)
        self.log_path = Path(log_path)
        self.experiment_id = experiment_id or datetime.now().strftime("%Y%m%d_%H%M%S")

        # Normalize weights (guard against zero/negative total)
        total_weight = sum(self.weights)
        if total_weight <= 0:
            total_weight = len(self.weights)  # Fallback to uniform
            self.weights = [1.0] * len(self.variants)
        self.cumulative_weights = []
        cumsum = 0.0
        for w in self.weights:
            cumsum += w / total_weight
            self.cumulative_weights.append(cumsum)

        # In-memory stats
        self.stats: Dict[str, VariantStats] = {
            v.value: VariantStats(variant=v.value) for v in self.variants
        }

        # Thread safety
        self._lock = threading.Lock()

        # Variant implementations
        self._variant_impls: Dict[VariantType, RerankerVariant] = {}
        self._init_variants()

    def _init_variants(self):
        """Initialize reranker implementations for each variant."""
        # Baseline: just sort by initial score
        def baseline_rerank(query, candidates, **kwargs):
            return sorted(candidates, key=lambda x: x.get("score", 0), reverse=True)

        self._variant_impls[VariantType.BASELINE] = RerankerVariant(
            VariantType.BASELINE, baseline_rerank
        )

        # Recursive reranker
        try:
            try:
                from scripts.rerank_recursive import rerank_recursive
            except ImportError:
                from rerank_recursive import rerank_recursive

            self._variant_impls[VariantType.RECURSIVE] = RerankerVariant(
                VariantType.RECURSIVE,
                lambda q, c, **kw: rerank_recursive(q, c, n_iterations=3)
            )
        except ImportError:
            pass

        # ONNX reranker
        try:
            try:
                from scripts.rerank_local import rerank_in_process
            except ImportError:
                from rerank_local import rerank_in_process

            self._variant_impls[VariantType.ONNX] = RerankerVariant(
                VariantType.ONNX,
                lambda q, c, **kw: rerank_in_process(q, c, limit=len(c))
            )
        except ImportError:
            pass

    def _hash_to_bucket(self, session_id: str) -> float:
        """Hash session ID to a value in [0, 1) for consistent bucketing."""
        h = hashlib.md5(f"{self.experiment_id}:{session_id}".encode()).hexdigest()
        return int(h[:8], 16) / (16 ** 8)

    def get_variant_type(self, session_id: str) -> VariantType:
        """Get the variant type for a session (consistent assignment)."""
        bucket = self._hash_to_bucket(session_id)
        for i, threshold in enumerate(self.cumulative_weights):
            if bucket < threshold:
                return self.variants[i]
        return self.variants[-1]

    def get_variant(self, session_id: str) -> RerankerVariant:
        """Get the reranker variant for a session."""
        variant_type = self.get_variant_type(session_id)

        if variant_type in self._variant_impls:
            return self._variant_impls[variant_type]

        # Fall back to baseline if variant not available
        return self._variant_impls.get(VariantType.BASELINE,
            RerankerVariant(VariantType.BASELINE, lambda q, c, **kw: c))

    def log_metrics(
        self,
        session_id: str,
        latency_ms: float = 0.0,
        clicked_rank: Optional[int] = None,
        num_results: int = 0,
        query_length: int = 0,
        **custom
    ):
        """Log metrics for an observation."""
        variant_type = self.get_variant_type(session_id)

        metric = ABMetric(
            session_id=session_id,
            variant=variant_type.value,
            timestamp=time.time(),
            latency_ms=latency_ms,
            clicked_rank=clicked_rank,
            num_results=num_results,
            query_length=query_length,
            custom=custom,
        )

        with self._lock:
            # Update in-memory stats
            if variant_type.value in self.stats:
                self.stats[variant_type.value].add_observation(metric)

            # Append to log file
            self._append_metric(metric)

    def _append_metric(self, metric: ABMetric):
        """Append metric to log file."""
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        record = {
            "experiment_id": self.experiment_id,
            "session_id": metric.session_id,
            "variant": metric.variant,
            "timestamp": metric.timestamp,
            "latency_ms": metric.latency_ms,
            "clicked_rank": metric.clicked_rank,
            "num_results": metric.num_results,
            "query_length": metric.query_length,
            **metric.custom,
        }

        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for all variants."""
        return {
            "experiment_id": self.experiment_id,
            "variants": {v: self.stats[v].get_summary() for v in self.stats},
        }

    def print_summary(self):
        """Print formatted summary."""
        summary = self.get_summary()

        print("\n" + "=" * 70)
        print(f"A/B TEST RESULTS: {summary['experiment_id']}")
        print("=" * 70)

        print(f"\n{'Variant':<15} {'N':<8} {'Latency (ms)':<15} {'MRR':<10} {'CTR@1':<10}")
        print("-" * 70)

        for variant_name, stats in summary["variants"].items():
            n = stats.get("n_observations", 0)
            lat = f"{stats.get('latency_ms_mean', 0):.1f} ± {stats.get('latency_ms_std', 0):.1f}"
            mrr = f"{stats.get('mrr', 0):.4f}"
            ctr = f"{stats.get('click_rate_at_1', 0):.2%}"

            print(f"{variant_name:<15} {n:<8} {lat:<15} {mrr:<10} {ctr:<10}")

        print("-" * 70)
        print()


def simulate_ab_test(n_sessions: int = 100, n_queries_per_session: int = 5):
    """Simulate an A/B test with synthetic data."""
    import random

    ab = ABTestManager(
        variants=[VariantType.BASELINE, VariantType.RECURSIVE],
        weights=[0.5, 0.5],
    )

    print(f"Simulating A/B test with {n_sessions} sessions...")

    for session_idx in range(n_sessions):
        session_id = f"session_{session_idx}"
        variant = ab.get_variant(session_id)

        for query_idx in range(n_queries_per_session):
            # Generate fake candidates
            candidates = [
                {"path": f"file_{i}.py", "score": random.random()}
                for i in range(10)
            ]

            # Rerank
            results, latency_ms = variant.rerank("test query", candidates)

            # Simulate click (higher probability for top results)
            clicked_rank = None
            for rank, result in enumerate(results[:5], 1):
                if random.random() < 0.3 / rank:  # Decreasing probability
                    clicked_rank = rank
                    break

            # Log metrics
            ab.log_metrics(
                session_id=session_id,
                latency_ms=latency_ms,
                clicked_rank=clicked_rank,
                num_results=len(results),
                query_length=len("test query"),
            )

    ab.print_summary()
    return ab


if __name__ == "__main__":
    simulate_ab_test(n_sessions=100, n_queries_per_session=5)
