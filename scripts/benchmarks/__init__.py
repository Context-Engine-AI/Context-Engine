"""Benchmarks package for Context-Engine performance measurement."""

from scripts.benchmarks.efficiency_benchmark import (
    ToolCallTracker,
    ToolCallMetrics,
    ScenarioResult,
    BenchmarkReport,
    run_benchmark,
    run_scenario,
    compute_mrr,
    compute_recall_at_k,
    count_tokens,
    SCENARIOS,
)

__all__ = [
    # Efficiency benchmark
    "ToolCallTracker",
    "ToolCallMetrics",
    "ScenarioResult",
    "BenchmarkReport",
    "run_benchmark",
    "run_scenario",
    "compute_mrr",
    "compute_recall_at_k",
    "count_tokens",
    "SCENARIOS",
    # Component benchmarks (import on demand)
    # - eval_harness
    # - trm_bench
    # - router_bench
    # - refrag_bench
    # - expand_bench
    # - run_all
]

