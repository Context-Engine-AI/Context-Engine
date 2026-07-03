#!/usr/bin/env python3
"""
Auto-Tuner Validation Loop.

Validates that recommended config changes actually improve metrics.
Closes the feedback loop: recommend → apply → measure → confirm.
"""

import asyncio
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

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

from scripts.benchmarks.trace_optimizer import query_clickhouse, analyze_traces, TraceInsight


@dataclass
class ValidationResult:
    """Result of validating a config change."""
    metric: str
    change: str  # e.g., "128 → 64"
    before_p90: float
    after_p90: float
    improvement_pct: float
    verdict: str  # "CONFIRMED", "DEGRADED", "INCONCLUSIVE", "NEUTRAL"
    confidence: float
    sample_before: int
    sample_after: int


async def get_metric_baseline(minutes: int = 30) -> Dict[str, float]:
    """
    Get baseline metrics from recent traces.

    Args:
        minutes: Time window for metric collection.

    Returns dict with p50, p90, p99, mean, count.
    """
    sql = f"""
    SELECT 
        avg(Duration / 1e6) as mean_ms,
        quantile(0.5)(Duration / 1e6) as p50_ms,
        quantile(0.9)(Duration / 1e6) as p90_ms,
        quantile(0.99)(Duration / 1e6) as p99_ms,
        count() as sample_count
    FROM openlit.otel_traces 
    WHERE SpanName = 'mcp tools/call'
      AND SpanAttributes['mcp.request.payload'] LIKE '%repo_search%'
      AND Timestamp > now() - INTERVAL {minutes} MINUTE
    """
    
    results = await query_clickhouse(sql)
    
    if not results or results[0].get("sample_count", 0) == 0:
        return {"p50": 0, "p90": 0, "p99": 0, "mean": 0, "count": 0}
    
    r = results[0]
    return {
        "p50": float(r.get("p50_ms", 0) or 0),
        "p90": float(r.get("p90_ms", 0) or 0),
        "p99": float(r.get("p99_ms", 0) or 0),
        "mean": float(r.get("mean_ms", 0) or 0),
        "count": int(r.get("sample_count", 0)),
    }


async def apply_config_change(metric: str, new_value: Any) -> str:
    """
    Apply a config change. Returns the old value.
    
    For now, this updates the TunableConfig. In production,
    you might want to update .env and signal a reload.
    """
    from scripts.benchmarks.tunable_config import CONFIG
    
    # Map metric names to config keys
    metric_to_key = {
        "QDRANT_EF_SEARCH": "ef_search",
        "HYBRID_SYMBOL_BOOST": "symbol_boost",
        "MICRO_BUDGET_TOKENS": "budget_tokens",
        "RERANKER_TOPN": "rerank_top_n",
    }
    
    key = metric_to_key.get(metric)
    if not key:
        return "unknown"
    
    old_value = CONFIG.get(key)
    CONFIG.update(**{key: new_value}, source="validation_loop")
    
    # Also set env var for processes that read directly
    os.environ[metric] = str(new_value)
    
    return str(old_value)


async def rollback_config_change(metric: str, old_value: Any):
    """Rollback a config change to the old value."""
    await apply_config_change(metric, old_value)


async def validate_recommendation(
    insight: TraceInsight,
    wait_minutes: int = 5,
    min_samples: int = 20,
) -> ValidationResult:
    """
    Validate a single recommendation by applying it and measuring results.
    
    Steps:
    1. Collect baseline metrics
    2. Apply the recommended change
    3. Wait for new traces to accumulate
    4. Collect new metrics
    5. Compare and determine verdict
    """
    metric = insight.metric
    old_value = insight.current_value
    new_value = insight.suggested_value
    
    print(f"\n{'='*60}")
    print(f"VALIDATING: {metric}")
    print(f"Change: {old_value} → {new_value}")
    print(f"{'='*60}")
    
    # Step 1: Collect baseline
    print(f"\n[1/4] Collecting baseline metrics (last 30 min)...")
    baseline = await get_metric_baseline(minutes=30)
    print(f"  Baseline P90: {baseline['p90']:.0f}ms ({baseline['count']} samples)")
    
    if baseline["count"] < min_samples:
        return ValidationResult(
            metric=metric,
            change=f"{old_value} → {new_value}",
            before_p90=baseline["p90"],
            after_p90=0,
            improvement_pct=0,
            verdict="INCONCLUSIVE",
            confidence=0,
            sample_before=baseline["count"],
            sample_after=0,
        )
    
    # Step 2: Apply change
    print(f"\n[2/4] Applying change: {metric}={new_value}")
    previous_value = await apply_config_change(metric, new_value)
    
    # Handle unsupported metrics
    if previous_value == "unknown":
        print(f"  ⚠️ Metric {metric} is not supported by validation loop; skipping.")
        return ValidationResult(
            metric=metric,
            change=f"{old_value} → {new_value}",
            before_p90=baseline["p90"],
            after_p90=baseline["p90"],
            improvement_pct=0.0,
            verdict="INCONCLUSIVE",
            confidence=0.0,
            sample_before=baseline["count"],
            sample_after=baseline["count"],
        )
    
    print(f"  Previous value was: {previous_value}")

    # Step 3: Wait for traces
    print(f"\n[3/4] Waiting {wait_minutes} minutes for new traces...")
    for i in range(wait_minutes):
        await asyncio.sleep(60)
        print(f"  {i+1}/{wait_minutes} minutes elapsed")
    
    # Step 4: Collect new metrics
    print(f"\n[4/4] Collecting post-change metrics...")
    after = await get_metric_baseline(minutes=wait_minutes)
    print(f"  After P90: {after['p90']:.0f}ms ({after['count']} samples)")
    
    # Calculate improvement
    if baseline["p90"] > 0:
        improvement_pct = (baseline["p90"] - after["p90"]) / baseline["p90"] * 100
    else:
        improvement_pct = 0
    
    # Determine verdict with statistical consideration
    if after["count"] < min_samples:
        verdict = "INCONCLUSIVE"
        confidence = 0.3
    elif improvement_pct > 10:
        verdict = "CONFIRMED"
        confidence = min(0.5 + after["count"] / 100, 0.95)
    elif improvement_pct < -10:
        verdict = "DEGRADED"
        confidence = min(0.5 + after["count"] / 100, 0.95)
        # Rollback!
        print(f"\n⚠️  DEGRADATION DETECTED - Rolling back to {old_value}")
        await rollback_config_change(metric, old_value)
    else:
        verdict = "NEUTRAL"
        confidence = 0.5
    
    result = ValidationResult(
        metric=metric,
        change=f"{old_value} → {new_value}",
        before_p90=baseline["p90"],
        after_p90=after["p90"],
        improvement_pct=improvement_pct,
        verdict=verdict,
        confidence=confidence,
        sample_before=baseline["count"],
        sample_after=after["count"],
    )
    
    # Print verdict
    emoji = {"CONFIRMED": "✅", "DEGRADED": "❌", "INCONCLUSIVE": "❓", "NEUTRAL": "➖"}
    print(f"\n{emoji.get(verdict, '?')} VERDICT: {verdict}")
    print(f"  Improvement: {improvement_pct:+.1f}%")
    print(f"  Confidence: {confidence:.0%}")
    
    return result


async def run_validation_loop(
    wait_minutes: int = 5,
    auto_apply: bool = False,
) -> List[ValidationResult]:
    """
    Run the full validation loop:
    1. Get recommendations from trace_optimizer
    2. Validate each high-confidence recommendation
    3. Report results
    """
    print("=" * 70)
    print("AUTO-TUNER VALIDATION LOOP")
    print("=" * 70)
    
    # Get current recommendations
    print("\n[Step 1] Getting recommendations from trace optimizer...")
    report = await analyze_traces()
    
    if not report.recommended_changes:
        print("\n✅ No high-confidence recommendations to validate")
        return []
    
    print(f"\nRecommendations to validate: {len(report.recommended_changes)}")
    for metric, value in report.recommended_changes.items():
        print(f"  {metric}: → {value}")
    
    if not auto_apply:
        print("\n⚠️  Auto-apply is OFF. Set --auto-apply to actually apply changes.")
        return []
    
    # Validate each recommendation
    results = []
    for insight in report.insights:
        if insight.confidence >= 0.6:
            result = await validate_recommendation(
                insight,
                wait_minutes=wait_minutes,
            )
            results.append(result)
    
    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    
    for r in results:
        emoji = {"CONFIRMED": "✅", "DEGRADED": "❌", "INCONCLUSIVE": "❓", "NEUTRAL": "➖"}
        print(f"  {emoji.get(r.verdict, '?')} {r.metric}: {r.change} → {r.verdict} ({r.improvement_pct:+.1f}%)")
    
    confirmed = sum(1 for r in results if r.verdict == "CONFIRMED")
    print(f"\n{confirmed}/{len(results)} recommendations CONFIRMED")
    
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Auto-Tuner Validation Loop")
    parser.add_argument("--wait", type=int, default=5, help="Minutes to wait for new traces")
    parser.add_argument("--auto-apply", action="store_true", help="Actually apply changes")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be validated")
    args = parser.parse_args()
    
    if args.dry_run:
        print("DRY RUN - Getting recommendations...")
        report = await analyze_traces()
        print(f"\nWould validate: {list(report.recommended_changes.keys())}")
        return
    
    results = await run_validation_loop(
        wait_minutes=args.wait,
        auto_apply=args.auto_apply,
    )
    
    # Save results
    if results:
        output = {
            "timestamp": datetime.now().isoformat(),
            "results": [
                {
                    "metric": r.metric,
                    "change": r.change,
                    "before_p90": r.before_p90,
                    "after_p90": r.after_p90,
                    "improvement_pct": r.improvement_pct,
                    "verdict": r.verdict,
                }
                for r in results
            ],
        }
        with open("/tmp/validation_results.json", "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to /tmp/validation_results.json")


if __name__ == "__main__":
    asyncio.run(main())
