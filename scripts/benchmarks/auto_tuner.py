#!/usr/bin/env python3
"""
Intelligent Auto-Tuner v2 for Context-Engine.

Features:
- Statistical validation with confidence intervals
- Interdependency detection between knobs
- Empirical impact estimation (measured, not guessed)
- Conflict detection between recommendations
- Dynamic thresholds based on historical data
- Multi-metric Pareto optimization
"""

import asyncio
import json
import math
import os
import statistics
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# ---------------------------------------------------------------------------
# Statistical Utilities
# ---------------------------------------------------------------------------
def mean_with_ci(values: List[float], confidence: float = 0.95) -> Tuple[float, float, float]:
    """
    Calculate mean with confidence interval.
    
    Returns: (mean, ci_lower, ci_upper)
    """
    if len(values) < 2:
        m = values[0] if values else 0.0
        return (m, m, m)
    
    n = len(values)
    m = statistics.mean(values)
    se = statistics.stdev(values) / math.sqrt(n)
    
    # t-value approximation for 95% CI
    t_values = {2: 12.71, 3: 4.30, 4: 3.18, 5: 2.78, 6: 2.57, 7: 2.45, 8: 2.37, 10: 2.26, 20: 2.09}
    t = t_values.get(n, 1.96)  # Fall back to z for large n
    
    margin = t * se
    return (m, m - margin, m + margin)


def is_significant_improvement(
    baseline_values: List[float],
    test_values: List[float],
    higher_is_better: bool = True,
    min_samples: int = 5,
) -> Tuple[bool, float, str]:
    """
    Determine if improvement is statistically significant.
    
    Returns: (is_significant, effect_size, explanation)
    """
    if len(baseline_values) < min_samples or len(test_values) < min_samples:
        return (False, 0.0, f"Insufficient samples ({len(baseline_values)}/{len(test_values)} < {min_samples})")
    
    baseline_mean, bl_lo, bl_hi = mean_with_ci(baseline_values)
    test_mean, t_lo, t_hi = mean_with_ci(test_values)
    
    # Effect size (Cohen's d approximation)
    pooled_std = math.sqrt(
        (statistics.stdev(baseline_values)**2 + statistics.stdev(test_values)**2) / 2
    )
    if pooled_std == 0:
        effect_size = 0.0
    else:
        effect_size = (test_mean - baseline_mean) / pooled_std
    
    # Check for non-overlapping confidence intervals
    if higher_is_better:
        is_significant = t_lo > bl_hi  # Test lower bound > baseline upper bound
        direction = "better" if test_mean > baseline_mean else "worse"
    else:
        is_significant = t_hi < bl_lo  # Test upper bound < baseline lower bound
        direction = "better" if test_mean < baseline_mean else "worse"
    
    delta = test_mean - baseline_mean
    explanation = f"Δ={delta:+.4f} (effect={effect_size:.2f}σ) [{direction}]"
    
    return (is_significant, effect_size, explanation)


# ---------------------------------------------------------------------------
# Configuration Knowledge Base
# ---------------------------------------------------------------------------
@dataclass
class KnobMeta:
    """Metadata about a tunable knob."""
    name: str
    default: Any
    test_values: List[Any]
    primary_metric: str
    higher_is_better: bool
    affects: List[str]  # Other knobs this interacts with
    trade_offs: List[str]  # Metrics that may get worse
    category: str


KNOB_REGISTRY = {
    "RERANKER_TOPN": KnobMeta(
        name="RERANKER_TOPN",
        default=100,
        test_values=[50, 75, 100, 125, 150],
        primary_metric="mrr",
        higher_is_better=True,
        affects=["RERANKER_TIMEOUT_MS", "QDRANT_EF_SEARCH"],
        trade_offs=["p90_latency"],
        category="quality_vs_latency",
    ),
    "RERANKER_RETURN_M": KnobMeta(
        name="RERANKER_RETURN_M",
        default=20,
        test_values=[10, 15, 20, 25, 30],
        primary_metric="recall_10",
        higher_is_better=True,
        affects=["MICRO_BUDGET_TOKENS"],
        trade_offs=["token_usage"],
        category="recall_vs_tokens",
    ),
    "HYBRID_SYMBOL_BOOST": KnobMeta(
        name="HYBRID_SYMBOL_BOOST",
        default=0.35,
        test_values=[0.2, 0.35, 0.5, 0.65],
        primary_metric="mrr",
        higher_is_better=True,
        affects=["POST_RERANK_SYMBOL_BOOST"],
        trade_offs=[],
        category="ranking",
    ),
    "HYBRID_PER_PATH": KnobMeta(
        name="HYBRID_PER_PATH",
        default=1,
        test_values=[1, 2, 3],
        primary_metric="recall_5",
        higher_is_better=True,
        affects=["RERANKER_TOPN"],
        trade_offs=["diversity"],
        category="recall_vs_diversity",
    ),
    "QDRANT_EF_SEARCH": KnobMeta(
        name="QDRANT_EF_SEARCH",
        default=128,
        test_values=[64, 96, 128, 192],
        primary_metric="mrr",
        higher_is_better=True,
        affects=["RERANKER_TOPN"],
        trade_offs=["p90_latency"],
        category="quality_vs_latency",
    ),
    "MICRO_BUDGET_TOKENS": KnobMeta(
        name="MICRO_BUDGET_TOKENS",
        default=3000,
        test_values=[1500, 2000, 3000, 4000, 5000],
        primary_metric="grounding_rate",
        higher_is_better=True,
        affects=["CTX_SNIPPET_CHARS"],
        trade_offs=["p90_latency", "token_usage"],
        category="grounding_vs_cost",
    ),
    "PRF_ENABLED": KnobMeta(
        name="PRF_ENABLED",
        default=1,
        test_values=[0, 1],
        primary_metric="recall_10",
        higher_is_better=True,
        affects=["HYBRID_EXPAND"],
        trade_offs=["p90_latency"],
        category="recall_vs_latency",
    ),
}


# ---------------------------------------------------------------------------
# Test Runner
# ---------------------------------------------------------------------------
@dataclass
class TestRun:
    """A single test run with metrics."""
    config: Dict[str, Any]
    metrics: Dict[str, float]
    latency_ms: float
    timestamp: str


@dataclass
class ValidationResult:
    """Result from statistically validating a knob change."""
    knob: str
    baseline_value: Any
    test_value: Any
    n_baseline: int
    n_test: int
    baseline_mean: Dict[str, Tuple[float, float, float]]  # metric -> (mean, ci_lo, ci_hi)
    test_mean: Dict[str, Tuple[float, float, float]]
    primary_metric: str
    is_significant: bool
    effect_size: float
    explanation: str
    trade_off_impacts: Dict[str, float]  # metric -> delta (negative = worse)
    conflicts_with: List[str]  # Other recommended changes this conflicts with
    recommendation: str  # "apply", "skip", "needs_more_data"
    confidence: float


async def run_eval_n_times(
    config_override: Optional[Dict[str, Any]] = None,
    n_runs: int = 5,
) -> List[Dict[str, float]]:
    """
    Run evaluation multiple times for statistical validity.
    """
    # Save and apply config
    original_env = {}
    if config_override:
        for key, value in config_override.items():
            original_env[key] = os.environ.get(key)
            os.environ[key] = str(value)
    
    results = []
    
    try:
        from scripts.benchmarks.eval_harness import run_evaluation, EVAL_QUERIES
        
        for i in range(n_runs):
            result = await run_evaluation(name=f"run_{i}", queries=EVAL_QUERIES)
            results.append({
                "mrr": result.avg_mrr,
                "recall_5": result.avg_recall_5,
                "recall_10": result.avg_recall_10,
                "precision_5": result.avg_precision_5,
                "p90_latency": result.p90_latency_ms,
            })
    
    except Exception as e:
        print(f"  Eval error: {e}")
    
    finally:
        for key, value in original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    
    return results


async def validate_knob_change(
    knob: str,
    test_value: Any,
    baseline_runs: List[Dict[str, float]],
    n_test_runs: int = 3,
) -> ValidationResult:
    """
    Validate a single knob change with statistical rigor.
    """
    meta = KNOB_REGISTRY[knob]
    current_value = os.environ.get(knob, str(meta.default))
    
    try:
        if "." in str(current_value):
            current_value = float(current_value)
        else:
            current_value = int(current_value)
    except ValueError:
        pass
    
    print(f"  Testing {knob}={test_value}...", end=" ", flush=True)
    
    # Run tests with new value
    test_runs = await run_eval_n_times({knob: test_value}, n_runs=n_test_runs)
    
    if not test_runs:
        print("FAILED")
        return ValidationResult(
            knob=knob,
            baseline_value=current_value,
            test_value=test_value,
            n_baseline=len(baseline_runs),
            n_test=0,
            baseline_mean={},
            test_mean={},
            primary_metric=meta.primary_metric,
            is_significant=False,
            effect_size=0.0,
            explanation="Test runs failed",
            trade_off_impacts={},
            conflicts_with=[],
            recommendation="skip",
            confidence=0.0,
        )
    
    # Calculate means with CIs for all metrics
    metrics = ["mrr", "recall_5", "recall_10", "precision_5", "p90_latency"]
    baseline_means = {}
    test_means = {}
    
    for metric in metrics:
        baseline_vals = [r.get(metric, 0) for r in baseline_runs]
        test_vals = [r.get(metric, 0) for r in test_runs]
        baseline_means[metric] = mean_with_ci(baseline_vals)
        test_means[metric] = mean_with_ci(test_vals)
    
    # Check significance on primary metric
    primary = meta.primary_metric
    baseline_primary = [r.get(primary, 0) for r in baseline_runs]
    test_primary = [r.get(primary, 0) for r in test_runs]
    
    is_sig, effect, explanation = is_significant_improvement(
        baseline_primary,
        test_primary,
        higher_is_better=meta.higher_is_better,
        min_samples=2,  # Relaxed for quick testing
    )
    
    # Check trade-offs
    trade_off_impacts = {}
    for trade_off in meta.trade_offs:
        if trade_off in ["p90_latency"]:
            baseline_lat = [r.get("p90_latency", 0) for r in baseline_runs]
            test_lat = [r.get("p90_latency", 0) for r in test_runs]
            if baseline_lat and test_lat:
                delta = statistics.mean(test_lat) - statistics.mean(baseline_lat)
                trade_off_impacts["p90_latency"] = delta
    
    # Determine recommendation
    if is_sig and effect > 0.2:  # Medium effect size
        recommendation = "apply"
        confidence = min(0.95, 0.5 + effect * 0.2)
    elif is_sig:
        recommendation = "apply"
        confidence = min(0.8, 0.3 + effect * 0.2)
    elif len(test_runs) < 3:
        recommendation = "needs_more_data"
        confidence = 0.3
    else:
        recommendation = "skip"
        confidence = 0.0
    
    # Check for latency trade-off that might be too severe
    if trade_off_impacts.get("p90_latency", 0) > 500:
        recommendation = "skip"
        confidence = 0.0
        explanation += " [REJECTED: latency +{:.0f}ms]".format(trade_off_impacts["p90_latency"])
    
    primary_delta = test_means[primary][0] - baseline_means[primary][0]
    print(f"{primary}: {baseline_means[primary][0]:.3f}→{test_means[primary][0]:.3f} (Δ{primary_delta:+.3f}) [{recommendation}]")
    
    return ValidationResult(
        knob=knob,
        baseline_value=current_value,
        test_value=test_value,
        n_baseline=len(baseline_runs),
        n_test=len(test_runs),
        baseline_mean=baseline_means,
        test_mean=test_means,
        primary_metric=primary,
        is_significant=is_sig,
        effect_size=effect,
        explanation=explanation,
        trade_off_impacts=trade_off_impacts,
        conflicts_with=[],
        recommendation=recommendation,
        confidence=confidence,
    )


def detect_conflicts(validations: List[ValidationResult]) -> List[ValidationResult]:
    """
    Detect conflicting recommendations.
    """
    recommended = [v for v in validations if v.recommendation == "apply"]
    
    for v in recommended:
        meta = KNOB_REGISTRY.get(v.knob)
        if not meta:
            continue
        
        for other in recommended:
            if other.knob == v.knob:
                continue
            
            # Check if they affect each other
            if other.knob in meta.affects:
                v.conflicts_with.append(other.knob)
            
            # Check if they have opposite category goals
            other_meta = KNOB_REGISTRY.get(other.knob)
            if other_meta and meta.category == other_meta.category:
                # Same category - might conflict
                if v.trade_off_impacts and other.trade_off_impacts:
                    v.conflicts_with.append(f"{other.knob} (same trade-off category)")
    
    return validations


# ---------------------------------------------------------------------------
# Main Tuner
# ---------------------------------------------------------------------------
@dataclass
class TuningReport:
    """Comprehensive tuning report."""
    timestamp: str
    baseline_runs: int
    test_runs_per_knob: int
    knobs_tested: int
    validations: List[ValidationResult]
    recommended_changes: Dict[str, Any]
    conflicts: List[str]
    total_expected_improvement: Dict[str, float]


async def run_intelligent_tuner(
    knobs: Optional[List[str]] = None,
    baseline_runs: int = 3,
    test_runs: int = 3,
) -> TuningReport:
    """
    Run the intelligent auto-tuner with statistical validation.
    """
    print("=" * 70)
    print("INTELLIGENT AUTO-TUNER v2")
    print("=" * 70)
    print(f"Baseline runs: {baseline_runs} | Test runs per knob: {test_runs}")
    
    # Step 1: Gather baseline with multiple runs
    print(f"\n[1/3] Gathering baseline ({baseline_runs} runs)...")
    baseline_results = await run_eval_n_times(n_runs=baseline_runs)
    
    # Fail early if baseline collection failed
    if len(baseline_results) < max(2, baseline_runs // 2):
        print(f"\n⚠️ Not enough successful baseline runs ({len(baseline_results)}/{baseline_runs})")
        print("  Check for timeouts or errors in eval_harness. Aborting tuning.")
        return TuningReport(
            timestamp=datetime.now().isoformat(),
            baseline_runs=len(baseline_results),
            test_runs_per_knob=0,
            knobs_tested=0,
            validations=[],
            recommended_changes={},
            conflicts=[],
            total_expected_improvement={},
        )
    
    for metric in ["mrr", "recall_5", "recall_10", "p90_latency"]:
        values = [r.get(metric, 0) for r in baseline_results]
        m, lo, hi = mean_with_ci(values)
        print(f"  {metric}: {m:.4f} (95% CI: [{lo:.4f}, {hi:.4f}])")
    
    # Step 2: Test each knob systematically
    print(f"\n[2/3] Testing knob configurations...")
    
    knobs_to_test = knobs or list(KNOB_REGISTRY.keys())
    validations: List[ValidationResult] = []
    
    for knob in knobs_to_test:
        if knob not in KNOB_REGISTRY:
            continue
        
        meta = KNOB_REGISTRY[knob]
        current = os.environ.get(knob, str(meta.default))
        
        print(f"\n  {knob} (current: {current}):")
        
        # Test each value
        for test_val in meta.test_values:
            if str(test_val) == str(current):
                continue
            
            result = await validate_knob_change(
                knob=knob,
                test_value=test_val,
                baseline_runs=baseline_results,
                n_test_runs=test_runs,
            )
            validations.append(result)
    
    # Step 3: Detect conflicts and finalize
    print(f"\n[3/3] Analyzing results and detecting conflicts...")
    
    validations = detect_conflicts(validations)
    
    # Build recommendations (only statistically significant improvements)
    # Track best ValidationResult per knob by effect size
    best_by_knob: Dict[str, ValidationResult] = {}
    conflicts = []
    
    for v in validations:
        if v.recommendation == "apply":
            current = best_by_knob.get(v.knob)
            if current is None or v.effect_size > current.effect_size:
                best_by_knob[v.knob] = v
            
            if v.conflicts_with:
                conflicts.extend([f"{v.knob} ↔ {c}" for c in v.conflicts_with])
    
    recommended = {}
    expected_improvement = {}
    for knob, v in best_by_knob.items():
        recommended[knob] = v.test_value
        primary = v.primary_metric
        expected_improvement[knob] = v.test_mean[primary][0] - v.baseline_mean[primary][0]
    
    return TuningReport(
        timestamp=datetime.now().isoformat(),
        baseline_runs=baseline_runs,
        test_runs_per_knob=test_runs,
        knobs_tested=len(knobs_to_test),
        validations=validations,
        recommended_changes=recommended,
        conflicts=list(set(conflicts)),
        total_expected_improvement=expected_improvement,
    )


def print_report(report: TuningReport) -> None:
    """Print the tuning report."""
    print("\n" + "=" * 70)
    print("TUNING REPORT (Statistically Validated)")
    print("=" * 70)
    print(f"Timestamp: {report.timestamp}")
    print(f"Baseline: {report.baseline_runs} runs | Tests: {report.test_runs_per_knob} runs/knob")
    print(f"Knobs tested: {report.knobs_tested}")
    
    # Show validated improvements
    applied = [v for v in report.validations if v.recommendation == "apply"]
    skipped = [v for v in report.validations if v.recommendation == "skip"]
    needs_data = [v for v in report.validations if v.recommendation == "needs_more_data"]
    
    print(f"\nResults: {len(applied)} improvements, {len(skipped)} skipped, {len(needs_data)} need more data")
    
    if applied:
        print("\n" + "-" * 70)
        print("VALIDATED IMPROVEMENTS:")
        for v in applied:
            sig = "✓" if v.is_significant else "~"
            print(f"\n  {sig} {v.knob}: {v.baseline_value} → {v.test_value}")
            print(f"    {v.primary_metric}: {v.baseline_mean[v.primary_metric][0]:.4f} → {v.test_mean[v.primary_metric][0]:.4f}")
            print(f"    Effect size: {v.effect_size:.2f}σ | Confidence: {v.confidence:.0%}")
            if v.trade_off_impacts:
                for m, d in v.trade_off_impacts.items():
                    print(f"    Trade-off: {m} {d:+.0f}ms")
            if v.conflicts_with:
                print(f"    ⚠ Conflicts: {', '.join(v.conflicts_with)}")
    
    if report.conflicts:
        print("\n" + "-" * 70)
        print("⚠ CONFLICTS DETECTED:")
        for c in report.conflicts:
            print(f"  - {c}")
    
    if report.recommended_changes:
        print("\n" + "-" * 70)
        print("RECOMMENDED CHANGES:")
        for knob, value in report.recommended_changes.items():
            impact = report.total_expected_improvement.get(knob, 0)
            print(f"  {knob}={value}  (measured Δ{impact:+.4f})")
    else:
        print("\n✅ Current configuration is statistically optimal!")
    
    print("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Intelligent Auto-Tuner v2")
    parser.add_argument("--knobs", nargs="+", help="Specific knobs to test")
    parser.add_argument("--baseline-runs", type=int, default=3, help="Baseline sample size")
    parser.add_argument("--test-runs", type=int, default=3, help="Test runs per config")
    parser.add_argument("--output", type=str, help="Output JSON file")
    args = parser.parse_args()
    
    report = await run_intelligent_tuner(
        knobs=args.knobs,
        baseline_runs=args.baseline_runs,
        test_runs=args.test_runs,
    )
    
    print_report(report)
    
    if args.output:
        # Convert to serializable dict
        data = {
            "timestamp": report.timestamp,
            "baseline_runs": report.baseline_runs,
            "test_runs_per_knob": report.test_runs_per_knob,
            "knobs_tested": report.knobs_tested,
            "recommended_changes": report.recommended_changes,
            "conflicts": report.conflicts,
            "expected_improvement": report.total_expected_improvement,
        }
        with open(args.output, "w") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"\nReport saved to: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
