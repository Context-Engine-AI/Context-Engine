#!/usr/bin/env python3
"""
Trace-Based Auto-Tuner for Context-Engine.

Uses OpenLit traces from ClickHouse to learn optimal parameters from production usage.
No synthetic benchmarks - learns from real query patterns and outcomes.

Supports multiple time windows: 24h (default), 7d, 30d
"""

import asyncio
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# ---------------------------------------------------------------------------
# Time Window Configuration
# ---------------------------------------------------------------------------
# Supported windows: "24h", "7d", "30d"
TRACE_WINDOW = os.environ.get("TRACE_WINDOW", "24h")

def get_interval_clause(window: str = None) -> str:
    """Convert window string to ClickHouse INTERVAL clause."""
    w = window or TRACE_WINDOW
    intervals = {
        "1h": "INTERVAL 1 HOUR",
        "6h": "INTERVAL 6 HOUR", 
        "12h": "INTERVAL 12 HOUR",
        "24h": "INTERVAL 24 HOUR",
        "7d": "INTERVAL 7 DAY",
        "30d": "INTERVAL 30 DAY",
    }
    return intervals.get(w, "INTERVAL 24 HOUR")

def get_window_label(window: str = None) -> str:
    """Get human-readable label for the window."""
    w = window or TRACE_WINDOW
    labels = {
        "1h": "1 hour",
        "6h": "6 hours",
        "12h": "12 hours", 
        "24h": "24 hours",
        "7d": "7 days",
        "30d": "30 days",
    }
    return labels.get(w, "24 hours")


# ---------------------------------------------------------------------------
# ClickHouse Connection
# ---------------------------------------------------------------------------
async def query_clickhouse(sql: str) -> List[Dict[str, Any]]:
    """Execute a query against ClickHouse and return results."""
    try:
        import httpx
        
        # Get connection details from env or use defaults from docker-compose.openlit.yml
        host = os.environ.get("CLICKHOUSE_HOST", "localhost")
        port = os.environ.get("CLICKHOUSE_PORT", "8123")
        user = os.environ.get("CLICKHOUSE_USER", "default")
        password = os.environ.get("CLICKHOUSE_PASSWORD", "OPENLIT")
        
        url = f"http://{host}:{port}"
        
        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=5.0, read=55.0)) as client:
            response = await client.post(
                url,
                content=sql + " FORMAT JSON",
                headers={"Content-Type": "text/plain"},
                params={"user": user, "password": password},
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get("data", [])
            else:
                print(f"ClickHouse error: {response.status_code} {response.text[:200]}")
                return []
    
    except Exception as e:
        print(f"ClickHouse connection error: {e}")
        return []


# ---------------------------------------------------------------------------
# Trace Analyzers
# ---------------------------------------------------------------------------
@dataclass
class TraceInsight:
    """Insight extracted from production traces."""
    metric: str
    current_value: Any
    suggested_value: Any
    confidence: float  # 0.0 - 1.0
    evidence: str  # Human-readable explanation
    sample_size: int
    trend: str = ""  # "improving", "degrading", "stable"
    p50: float = 0.0
    p90: float = 0.0
    p99: float = 0.0


def calculate_percentiles(values: List[float]) -> Dict[str, float]:
    """Calculate P50, P90, P99 percentiles."""
    if not values:
        return {"p50": 0, "p90": 0, "p99": 0, "min": 0, "max": 0, "mean": 0}
    
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    
    def percentile(p: float) -> float:
        idx = (n - 1) * p
        lower = int(idx)
        upper = min(lower + 1, n - 1)
        weight = idx - lower
        return sorted_vals[lower] * (1 - weight) + sorted_vals[upper] * weight
    
    return {
        "p50": percentile(0.5),
        "p90": percentile(0.9),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "min": sorted_vals[0],
        "max": sorted_vals[-1],
        "mean": sum(values) / n,
    }


async def get_tool_usage_stats() -> Dict[str, Any]:
    """Get comprehensive MCP tool usage statistics."""
    # Note: JSON-RPC payload structure uses params.name for tool name, not params.args[1]
    sql = f"""
    SELECT 
        JSONExtractString(SpanAttributes['mcp.request.payload'], 'params', 'name') as tool_name,
        count() as call_count,
        avg(Duration / 1e6) as avg_latency_ms,
        quantile(0.9)(Duration / 1e6) as p90_latency_ms
    FROM openlit.otel_traces 
    WHERE SpanName = 'mcp tools/call'
      AND Timestamp > now() - {get_interval_clause()}
    GROUP BY tool_name
    ORDER BY call_count DESC
    LIMIT 20
    """
    
    results = await query_clickhouse(sql)
    
    stats = {}
    for r in results:
        tool = r.get("tool_name", "unknown")
        if tool:
            stats[tool] = {
                "calls": int(r.get("call_count", 0)),
                "avg_latency_ms": float(r.get("avg_latency_ms", 0)),
                "p90_latency_ms": float(r.get("p90_latency_ms", 0)),
            }
    
    return stats


async def get_hourly_trends(metric_sql: str, hours: int = 24) -> List[Tuple[str, float]]:
    """Get hourly trend for a metric."""
    sql = f"""
    SELECT 
        toStartOfHour(Timestamp) as hour,
        {metric_sql} AS value
    FROM openlit.otel_traces 
    WHERE Timestamp > now() - INTERVAL {hours} HOUR
    GROUP BY hour
    ORDER BY hour
    """
    
    results = await query_clickhouse(sql)
    return [(r.get("hour", ""), float(r.get("value", 0))) for r in results]


def detect_trend(values: List[float], window: int = 3) -> str:
    """Detect if metric is improving, degrading, or stable."""
    if len(values) < window * 2:
        return "insufficient_data"
    
    early = sum(values[:window]) / window
    late = sum(values[-window:]) / window
    
    pct_change = (late - early) / early if early > 0 else 0
    
    if abs(pct_change) < 0.05:
        return "stable"
    elif pct_change > 0:
        return "degrading"  # Higher latency = worse
    else:
        return "improving"


async def analyze_llm_token_usage() -> Optional[TraceInsight]:
    """
    Analyze GLM token usage patterns to optimize DECODER_MAX_TOKENS.
    """
    sql = f"""
    SELECT 
        toFloat64OrZero(SpanAttributes['gen_ai.usage.output_tokens']) as output_tokens,
        toFloat64OrZero(SpanAttributes['gen_ai.usage.input_tokens']) as input_tokens,
        toFloat64OrZero(SpanAttributes['gen_ai.usage.total_tokens']) as total_tokens,
        toFloat64OrZero(SpanAttributes['gen_ai.request.max_tokens']) as max_tokens,
        Duration / 1e9 as duration_sec
    FROM openlit.otel_traces 
    WHERE SpanName LIKE '%glm%'
      AND StatusCode = 'Ok'
      AND Timestamp > now() - {get_interval_clause()}
    ORDER BY Timestamp DESC
    LIMIT 200
    """
    
    results = await query_clickhouse(sql)
    
    if len(results) < 10:
        return None
    
    # Analyze actual vs requested tokens
    output_tokens = [r["output_tokens"] for r in results if r["output_tokens"] > 0]
    input_tokens = [r["input_tokens"] for r in results if r["input_tokens"] > 0]
    total_tokens = [r["total_tokens"] for r in results if r["total_tokens"] > 0]
    max_tokens = [r["max_tokens"] for r in results if r["max_tokens"] > 0]
    
    if not output_tokens:
        return None
    
    avg_output = sum(output_tokens) / len(output_tokens)
    avg_input = sum(input_tokens) / len(input_tokens) if input_tokens else 0
    avg_total = sum(total_tokens) / len(total_tokens) if total_tokens else avg_output + avg_input
    
    stats = calculate_percentiles(output_tokens)
    p95_output = stats["p95"] if "p95" in stats else stats.get("p90", avg_output)
    
    current_max = int(os.environ.get("DECODER_MAX_TOKENS", "1200"))
    
    # Calculate utilization
    utilization = avg_output / current_max if current_max > 0 else 0
    
    if utilization < 0.3 and current_max > 500:
        suggested = max(int(p95_output * 1.5), 500)
        return TraceInsight(
            metric="DECODER_MAX_TOKENS",
            current_value=current_max,
            suggested_value=suggested,
            confidence=0.8 if len(output_tokens) > 50 else 0.5,
            evidence=f"Avg output: {avg_output:.0f} tokens (P95: {p95_output:.0f}), avg input: {avg_input:.0f} - only {utilization:.0%} of {current_max} budget used",
            sample_size=len(output_tokens),
        )
    elif utilization > 0.9:
        suggested = min(int(current_max * 1.5), 4096)
        return TraceInsight(
            metric="DECODER_MAX_TOKENS",
            current_value=current_max,
            suggested_value=suggested,
            confidence=0.7,
            evidence=f"Token budget nearly exhausted ({utilization:.0%}) - responses may be truncated. Avg total: {avg_total:.0f}",
            sample_size=len(output_tokens),
        )
    
    return None


async def analyze_context_token_usage() -> Dict[str, Any]:
    """
    Analyze context token usage across MCP tool calls and LLM completions.
    
    Returns comprehensive token usage statistics:
    - Per-tool token consumption
    - Context efficiency (useful vs total)
    - Cost estimation
    """
    # Get LLM token usage
    llm_sql = f"""
    SELECT 
        sum(toFloat64OrZero(SpanAttributes['gen_ai.usage.input_tokens'])) as total_input,
        sum(toFloat64OrZero(SpanAttributes['gen_ai.usage.output_tokens'])) as total_output,
        sum(toFloat64OrZero(SpanAttributes['gen_ai.usage.total_tokens'])) as total_tokens,
        count() as call_count,
        avg(toFloat64OrZero(SpanAttributes['gen_ai.usage.input_tokens'])) as avg_input,
        avg(toFloat64OrZero(SpanAttributes['gen_ai.usage.output_tokens'])) as avg_output
    FROM openlit.otel_traces 
    WHERE (SpanName LIKE '%glm%' OR SpanName LIKE '%chat%')
      AND Timestamp > now() - {get_interval_clause()}
    """
    
    llm_result = await query_clickhouse(llm_sql)
    
    # Get MCP tool call payload sizes (simpler query without extractJSONPathRaw)
    mcp_sql = f"""
    SELECT 
        count() as call_count,
        sum(length(SpanAttributes['mcp.request.payload'])) as total_payload_size,
        sum(length(SpanAttributes['mcp.response.payload'])) as total_response_size,
        avg(length(SpanAttributes['mcp.request.payload'])) as avg_payload_size,
        avg(length(SpanAttributes['mcp.response.payload'])) as avg_response_size
    FROM openlit.otel_traces 
    WHERE SpanName = 'mcp tools/call'
      AND Timestamp > now() - {get_interval_clause()}
    """
    
    mcp_results = await query_clickhouse(mcp_sql)
    
    # Calculate totals
    llm = llm_result[0] if llm_result else {}
    total_input = float(llm.get("total_input", 0) or 0)
    total_output = float(llm.get("total_output", 0) or 0)
    total_tokens = float(llm.get("total_tokens", 0) or total_input + total_output)
    llm_calls = int(llm.get("call_count", 0) or 0)
    
    # Estimate cost (using typical pricing: $0.01/1K input, $0.03/1K output)
    estimated_cost = (total_input * 0.00001) + (total_output * 0.00003)
    
    # Calculate MCP payload sizes  
    mcp = mcp_results[0] if mcp_results else {}
    mcp_calls = int(mcp.get("call_count", 0) or 0)
    total_mcp_payload = int(float(mcp.get("total_payload_size", 0) or 0))
    total_mcp_response = int(float(mcp.get("total_response_size", 0) or 0))
    avg_payload = int(float(mcp.get("avg_payload_size", 0) or 0))
    avg_response = int(float(mcp.get("avg_response_size", 0) or 0))
    
    return {
        "llm": {
            "total_input_tokens": int(total_input),
            "total_output_tokens": int(total_output),
            "total_tokens": int(total_tokens),
            "call_count": llm_calls,
            "avg_input_per_call": int(total_input / llm_calls) if llm_calls > 0 else 0,
            "avg_output_per_call": int(total_output / llm_calls) if llm_calls > 0 else 0,
            "estimated_cost_usd": round(estimated_cost, 4),
        },
        "mcp": {
            "call_count": mcp_calls,
            "avg_payload_bytes": avg_payload,
            "avg_response_bytes": avg_response,
            "total_payload_bytes": total_mcp_payload,
            "total_response_bytes": total_mcp_response,
        },
        "efficiency": {
            "input_output_ratio": round(total_input / total_output, 2) if total_output > 0 else 0,
            "tokens_per_mcp_call": int(total_tokens / mcp_calls) if mcp_calls > 0 else 0,
        },
    }


async def analyze_search_latency() -> Optional[TraceInsight]:
    """
    Analyze search latency patterns from MCP tool calls.
    
    Uses proper statistical analysis:
    - Coefficient of variation for confidence scoring
    - Percentile-based adaptive thresholds
    - Sample size weighting
    - Outlier handling
    """
    sql = f"""
    SELECT 
        SpanAttributes['mcp.request.payload'] as payload,
        Duration / 1e6 as duration_ms
    FROM openlit.otel_traces 
    WHERE SpanName = 'mcp tools/call'
      AND SpanAttributes['mcp.request.payload'] LIKE '%repo_search%'
      AND Timestamp > now() - {get_interval_clause()}
    ORDER BY Timestamp DESC
    LIMIT 500
    """
    
    results = await query_clickhouse(sql)
    
    # Require minimum samples for statistical validity
    MIN_SAMPLES = 30
    if len(results) < MIN_SAMPLES:
        return None
    
    latencies = [r["duration_ms"] for r in results if r["duration_ms"] > 0]
    
    if len(latencies) < MIN_SAMPLES:
        return None
    
    # Remove outliers (>3 std from mean) for cleaner analysis
    mean_lat = sum(latencies) / len(latencies)
    std_lat = (sum((x - mean_lat) ** 2 for x in latencies) / len(latencies)) ** 0.5
    
    if std_lat > 0:
        filtered = [x for x in latencies if abs(x - mean_lat) <= 3 * std_lat]
        if len(filtered) >= MIN_SAMPLES:
            latencies = filtered
    
    # Calculate statistics
    stats = calculate_percentiles(latencies)
    p50, p90, p99 = stats["p50"], stats["p90"], stats["p99"]
    mean_val = stats["mean"]
    
    # Calculate coefficient of variation (CV) - lower = more consistent = higher confidence
    cv = std_lat / mean_val if mean_val > 0 else float('inf')
    
    # Confidence based on sample size and consistency
    # Higher samples + lower CV = higher confidence
    sample_factor = min(len(latencies) / 100, 1.0)  # Max out at 100 samples
    consistency_factor = max(0, 1 - min(cv, 2) / 2)  # CV > 2 = low consistency
    base_confidence = 0.5 + 0.3 * sample_factor + 0.2 * consistency_factor
    
    current_ef = int(os.environ.get("QDRANT_EF_SEARCH", "128"))
    
    # Define SLO targets (these could be env vars too)
    TARGET_P90_MS = float(os.environ.get("SEARCH_TARGET_P90_MS", "500"))
    TARGET_P99_MS = float(os.environ.get("SEARCH_TARGET_P99_MS", "2000"))
    
    # Decision logic based on percentile analysis
    if p90 > TARGET_P90_MS * 2:  # P90 > 2x target = definitely reduce
        # Calculate suggested EF reduction proportional to how much we're over
        overage_ratio = p90 / TARGET_P90_MS
        reduction = min(int(current_ef * 0.25 * min(overage_ratio - 1, 2)), 64)
        suggested = max(current_ef - reduction, 64)
        
        return TraceInsight(
            metric="QDRANT_EF_SEARCH",
            current_value=current_ef,
            suggested_value=suggested,
            confidence=round(min(base_confidence + 0.1, 0.95), 2),
            evidence=f"P90={p90:.0f}ms exceeds target {TARGET_P90_MS:.0f}ms by {overage_ratio:.1f}x (P50={p50:.0f}ms, P99={p99:.0f}ms, CV={cv:.2f})",
            sample_size=len(latencies),
            p50=p50,
            p90=p90,
            p99=p99,
        )
    elif p90 < TARGET_P90_MS * 0.3 and p99 < TARGET_P99_MS * 0.5 and current_ef < 256:
        # Headroom to increase quality - but lower confidence since it's optional
        return TraceInsight(
            metric="QDRANT_EF_SEARCH",
            current_value=current_ef,
            suggested_value=min(current_ef + 32, 256),
            confidence=round(base_confidence * 0.7, 2),  # Lower confidence for optional improvements
            evidence=f"P90={p90:.0f}ms well under target {TARGET_P90_MS:.0f}ms - can increase EF for better recall (P99={p99:.0f}ms, CV={cv:.2f})",
            sample_size=len(latencies),
            p50=p50,
            p90=p90,
            p99=p99,
        )
    
    return None


async def analyze_query_patterns() -> Optional[TraceInsight]:
    """
    Analyze query patterns from MCP tool calls to optimize HYBRID_SYMBOL_BOOST.
    
    Uses comprehensive symbol detection:
    - snake_case, camelCase, PascalCase patterns
    - File path patterns (with /, \\, .)
    - Single-word queries
    - Semantic vs lookup query classification
    """
    sql = f"""
    SELECT 
        SpanAttributes['mcp.request.payload'] as payload
    FROM openlit.otel_traces 
    WHERE SpanName = 'mcp tools/call'
      AND SpanAttributes['mcp.request.payload'] LIKE '%repo_search%'
      AND Timestamp > now() - {get_interval_clause()}
    ORDER BY Timestamp DESC
    LIMIT 500
    """
    
    results = await query_clickhouse(sql)
    
    MIN_SAMPLES = 30
    if len(results) < MIN_SAMPLES:
        return None
    
    # Parse payload JSON to extract queries
    # Note: JSON-RPC uses params.arguments, not params.args
    queries = []
    for r in results:
        try:
            payload = json.loads(r.get("payload", "{}"))
            params = payload.get("params", {})
            # Try params.arguments first (standard JSON-RPC), fallback to params.args
            arguments = params.get("arguments") or {}
            if not arguments and isinstance(params.get("args"), list) and len(params["args"]) >= 2:
                arguments = params["args"][1] if isinstance(params["args"][1], dict) else {}
            
            # Try multiple query field patterns
            query = None
            # 1. Direct query field
            if arguments.get("query"):
                query = arguments.get("query")
            # 2. Multi-query (queries field for fusion/expansion)
            elif arguments.get("queries"):
                q = arguments.get("queries")
                query = q if isinstance(q, str) else " ".join(str(x) for x in q) if isinstance(q, list) else None
            # 3. Nested kwargs.query
            elif isinstance(arguments.get("kwargs"), dict) and arguments["kwargs"].get("query"):
                query = arguments["kwargs"].get("query")
            
            # Handle list of queries
            if isinstance(query, list):
                query = " ".join(str(q) for q in query)
            if query and len(str(query).strip()) > 0:
                queries.append(str(query).strip())
        except (json.JSONDecodeError, KeyError, IndexError, TypeError):
            continue
    
    if len(queries) < MIN_SAMPLES:
        return None
    
    def is_symbol_like(query: str) -> bool:
        """Comprehensive symbol detection."""
        words = query.split()
        
        # Path-like queries (contain / or \\ or start with .)
        if "/" in query or "\\" in query or query.startswith("."):
            return True
        
        # Single word with underscore = snake_case
        if len(words) == 1 and "_" in query:
            return True
        
        # Single word with internal caps = camelCase or PascalCase
        if len(words) == 1 and len(query) > 1:
            has_internal_upper = any(c.isupper() for c in query[1:])
            has_lower = any(c.islower() for c in query)
            if has_internal_upper and has_lower:
                return True
        
        # Two words with underscores or mixed case
        if len(words) == 2:
            for word in words:
                if "_" in word or (len(word) > 1 and any(c.isupper() for c in word[1:])):
                    return True
        
        # Very short (1-2 words, no question words) = likely lookup
        if len(words) <= 2 and not any(w in query.lower() for w in ["how", "what", "why", "when", "where"]):
            return True
        
        return False
    
    def is_semantic_like(query: str) -> bool:
        """Detect semantic/natural language queries."""
        words = query.split()
        
        # Contains question words
        if any(w in query.lower() for w in ["how", "what", "why", "explain", "describe", "tell me"]):
            return True
        
        # Long queries (5+ words) without code patterns
        if len(words) >= 5 and "_" not in query and not any(c.isupper() for c in query[1:] if query):
            return True
        
        return False
    
    # Classify all queries
    symbol_count = sum(1 for q in queries if is_symbol_like(q))
    semantic_count = sum(1 for q in queries if is_semantic_like(q))
    
    symbol_ratio = symbol_count / len(queries)
    semantic_ratio = semantic_count / len(queries)
    
    # Calculate confidence based on how clear the pattern is
    pattern_clarity = abs(symbol_ratio - semantic_ratio)
    sample_factor = min(len(queries) / 100, 1.0)
    base_confidence = 0.4 + 0.3 * pattern_clarity + 0.3 * sample_factor
    
    current_boost = float(os.environ.get("HYBRID_SYMBOL_BOOST", "0.35"))
    
    # Decision thresholds with hysteresis to avoid flip-flopping
    if symbol_ratio >= 0.5 and current_boost < 0.45:
        suggested = round(min(0.5 + (symbol_ratio - 0.5) * 0.3, 0.65), 2)
        return TraceInsight(
            metric="HYBRID_SYMBOL_BOOST",
            current_value=current_boost,
            suggested_value=suggested,
            confidence=round(base_confidence, 2),
            evidence=f"{symbol_ratio:.0%} symbol-like vs {semantic_ratio:.0%} semantic queries - increase boost for precision",
            sample_size=len(queries),
        )
    elif semantic_ratio >= 0.6 and current_boost > 0.30:
        suggested = round(max(0.35 - (semantic_ratio - 0.6) * 0.2, 0.20), 2)
        return TraceInsight(
            metric="HYBRID_SYMBOL_BOOST",
            current_value=current_boost,
            suggested_value=suggested,
            confidence=round(base_confidence, 2),
            evidence=f"{semantic_ratio:.0%} semantic vs {symbol_ratio:.0%} symbol-like queries - reduce boost for better semantic matching",
            sample_size=len(queries),
        )
    
    return None


async def analyze_grounding_rate() -> Optional[TraceInsight]:
    """
    Analyze context_answer grounding to optimize MICRO_BUDGET_TOKENS.
    """
    sql = f"""
    SELECT 
        SpanAttributes['gen_ai.response.grounded'] as grounded,
        SpanAttributes['gen_ai.response.citation_count'] as citations,
        SpanAttributes['gen_ai.request.budget_tokens'] as budget
    FROM openlit.otel_traces 
    WHERE SpanName LIKE '%context_answer%'
      AND StatusCode = 'Ok'
      AND Timestamp > now() - {get_interval_clause()}
    ORDER BY Timestamp DESC
    LIMIT 100
    """
    
    results = await query_clickhouse(sql)
    
    if len(results) < 10:
        return None
    
    grounded_count = sum(1 for r in results if r.get("grounded") in [True, "true", "1", 1])
    grounding_rate = grounded_count / len(results) if results else 0
    
    avg_citations = sum(float(r.get("citations", 0)) for r in results) / len(results)
    
    current_budget = int(os.environ.get("MICRO_BUDGET_TOKENS", "3000"))
    
    if grounding_rate < 0.9:
        suggested = min(current_budget + 1000, 6000)
        return TraceInsight(
            metric="MICRO_BUDGET_TOKENS",
            current_value=current_budget,
            suggested_value=suggested,
            confidence=0.8,
            evidence=f"Grounding rate: {grounding_rate:.0%} (avg {avg_citations:.1f} citations) - increase budget",
            sample_size=len(results),
        )
    elif grounding_rate > 0.98 and avg_citations > 10:
        # Very high grounding, maybe can reduce budget
        suggested = max(current_budget - 500, 2000)
        return TraceInsight(
            metric="MICRO_BUDGET_TOKENS",
            current_value=current_budget,
            suggested_value=suggested,
            confidence=0.4,
            evidence=f"Excellent grounding ({grounding_rate:.0%}) with {avg_citations:.1f} citations - could reduce budget",
            sample_size=len(results),
        )
    
    return None


async def analyze_rerank_effectiveness() -> Optional[TraceInsight]:
    """
    Analyze if reranking actually improves results.
    """
    sql = f"""
    SELECT 
        SpanAttributes['rerank.position_changes'] as changes,
        SpanAttributes['rerank.top1_changed'] as top1_changed,
        Duration / 1e6 as duration_ms
    FROM openlit.otel_traces 
    WHERE SpanName LIKE '%rerank%'
      AND StatusCode = 'Ok'
      AND Timestamp > now() - {get_interval_clause()}
    ORDER BY Timestamp DESC
    LIMIT 100
    """
    
    results = await query_clickhouse(sql)
    
    if len(results) < 20:
        return None
    
    # Count how often reranking changes top-1 result
    top1_changes = sum(1 for r in results if r.get("top1_changed") in [True, "true", "1", 1])
    change_rate = top1_changes / len(results) if results else 0
    
    avg_latency = sum(r.get("duration_ms", 0) for r in results) / len(results)
    
    current_enabled = int(os.environ.get("RERANKER_ENABLED", "1"))
    
    if change_rate < 0.1 and avg_latency > 200:
        return TraceInsight(
            metric="RERANKER_ENABLED",
            current_value=current_enabled,
            suggested_value=0,
            confidence=0.6,
            evidence=f"Reranking changes top-1 only {change_rate:.0%} of time but adds {avg_latency:.0f}ms latency",
            sample_size=len(results),
        )
    elif change_rate > 0.5:
        # Reranking is very effective
        current_topn = int(os.environ.get("RERANKER_TOPN", "100"))
        if current_topn < 150:
            return TraceInsight(
                metric="RERANKER_TOPN",
                current_value=current_topn,
                suggested_value=150,
                confidence=0.7,
                evidence=f"Reranking changes top-1 {change_rate:.0%} - more candidates may help further",
                sample_size=len(results),
            )
    
    return None


async def analyze_error_patterns() -> Optional[TraceInsight]:
    """
    Identify tools/queries that fail frequently and suggest fixes.
    
    Maps common error types to actionable tuning parameters.
    """
    sql = f"""
    SELECT 
        SpanName,
        SpanAttributes['error.message'] as error_msg,
        SpanAttributes['error.type'] as error_type,
        count() as cnt
    FROM openlit.otel_traces 
    WHERE StatusCode = 'Error'
      AND Timestamp > now() - {get_interval_clause()}
    GROUP BY SpanName, error_msg, error_type
    ORDER BY cnt DESC
    LIMIT 20
    """
    
    results = await query_clickhouse(sql)
    
    if not results:
        return None
    
    total_errors = sum(int(r.get("cnt", 0)) for r in results)
    
    if total_errors < 5:
        return None
    
    # Get total requests for error rate calculation
    total_sql = f"""
    SELECT count() as total FROM openlit.otel_traces 
    WHERE Timestamp > now() - {get_interval_clause()}
    """
    total_result = await query_clickhouse(total_sql)
    if not total_result or "total" not in total_result[0]:
        print("Failed to get total request count; skipping error pattern analysis.")
        return None
    total_requests = max(1, int(total_result[0]["total"]))
    
    error_rate = total_errors / total_requests
    
    top_error = results[0]
    error_msg = str(top_error.get("error_msg", "unknown"))[:100]
    span_name = top_error.get("SpanName", "unknown")
    
    # Map common errors to actionable tuning
    suggested_param = None
    suggested_action = None
    
    if "timeout" in error_msg.lower():
        suggested_param = "TOOL_TIMEOUT_SECS"
        suggested_action = "increase timeout threshold"
    elif "context" in error_msg.lower() or "length" in error_msg.lower():
        suggested_param = "MICRO_BUDGET_TOKENS"
        suggested_action = "reduce token budget"
    elif "rate" in error_msg.lower() or "limit" in error_msg.lower():
        suggested_param = "CONCURRENT_REQUESTS"
        suggested_action = "reduce concurrency"
    elif "memory" in error_msg.lower():
        suggested_param = "CACHE_TTL_SECONDS"
        suggested_action = "reduce cache size"
    
    evidence = f"Error rate: {error_rate:.2%} ({total_errors}/{total_requests}). Top: {span_name} - '{error_msg}'"
    
    if suggested_param:
        return TraceInsight(
            metric=suggested_param,
            current_value=os.environ.get(suggested_param, "default"),
            suggested_value=suggested_action,
            confidence=min(0.5 + error_rate * 5, 0.9),  # Higher error rate = higher confidence
            evidence=evidence,
            sample_size=total_errors,
        )
    elif error_rate > 0.05:  # >5% error rate is concerning
        return TraceInsight(
            metric="ERROR_RATE",
            current_value=f"{error_rate:.2%}",
            suggested_value="investigate errors",
            confidence=0.8,
            evidence=evidence,
            sample_size=total_errors,
        )
    
    return None


async def analyze_latency_breakdown() -> Dict[str, Dict[str, float]]:
    """
    Break down latency by component to identify bottlenecks.
    
    Returns dict mapping component -> {avg_ms, p90_ms, pct_of_total}
    """
    components = {
        "embed": "SpanName LIKE '%embed%'",
        "qdrant_search": "(SpanName LIKE '%qdrant%' OR SpanName LIKE '%GET Context%')",
        "rerank": "SpanName LIKE '%rerank%'",
        "llm": "(SpanName LIKE '%glm%' OR SpanName LIKE '%chat%')",
        "mcp_tool": "SpanName = 'mcp tools/call'",
    }
    
    breakdown = {}
    total_time = 0
    
    for name, condition in components.items():
        sql = f"""
        SELECT 
            avg(Duration / 1e6) as avg_ms,
            quantile(0.9)(Duration / 1e6) as p90_ms,
            count() as cnt
        FROM openlit.otel_traces 
        WHERE {condition}
          AND Timestamp > now() - {get_interval_clause()}
        """
        results = await query_clickhouse(sql)
        
        if results and results[0].get("cnt", 0) > 0:
            avg_ms = float(results[0].get("avg_ms", 0) or 0)
            p90_ms = float(results[0].get("p90_ms", 0) or 0)
            cnt = int(results[0].get("cnt", 0))
            
            breakdown[name] = {
                "avg_ms": avg_ms,
                "p90_ms": p90_ms,
                "count": cnt,
            }
            total_time += avg_ms * cnt
    
    # Calculate percentage contribution
    if total_time > 0:
        for name in breakdown:
            component_time = breakdown[name]["avg_ms"] * breakdown[name]["count"]
            breakdown[name]["pct_of_total"] = component_time / total_time * 100
    
    return breakdown


async def analyze_time_patterns() -> Optional[TraceInsight]:
    """
    Detect if performance varies significantly by hour (load patterns).
    
    Useful for detecting if dynamic scaling or time-based params are needed.
    """
    sql = f"""
    SELECT 
        toHour(Timestamp) as hour,
        avg(Duration / 1e6) as avg_latency,
        quantile(0.9)(Duration / 1e6) as p90_latency,
        count() as volume
    FROM openlit.otel_traces 
    WHERE SpanName = 'mcp tools/call'
      AND Timestamp > now() - INTERVAL 7 DAY
    GROUP BY hour
    HAVING volume > 10
    ORDER BY hour
    """
    
    results = await query_clickhouse(sql)
    
    if len(results) < 12:  # Need at least half the day covered
        return None
    
    latencies = {int(r["hour"]): float(r["avg_latency"]) for r in results}
    volumes = {int(r["hour"]): int(r["volume"]) for r in results}
    
    if not latencies:
        return None
    
    # Find peak and trough weighted by volume
    peak_hour = max(latencies, key=lambda h: latencies[h])
    trough_hour = min(latencies, key=lambda h: latencies[h])
    
    peak_lat = latencies[peak_hour]
    trough_lat = latencies[trough_hour]
    
    if trough_lat <= 0:
        return None
    
    ratio = peak_lat / trough_lat
    total_volume = sum(volumes.values())
    
    # Calculate coefficient of variation across hours
    mean_lat = sum(latencies.values()) / len(latencies)
    std_lat = (sum((v - mean_lat) ** 2 for v in latencies.values()) / len(latencies)) ** 0.5
    cv = std_lat / mean_lat if mean_lat > 0 else 0
    
    if ratio > 2.0 and cv > 0.3:
        return TraceInsight(
            metric="TIME_BASED_SCALING",
            current_value="static",
            suggested_value=f"dynamic (peak hour {peak_hour}:00)",
            confidence=round(min(0.5 + cv * 0.5, 0.85), 2),
            evidence=f"Latency varies {ratio:.1f}x between hours {trough_hour}:00 ({trough_lat:.0f}ms) and {peak_hour}:00 ({peak_lat:.0f}ms), CV={cv:.2f}",
            sample_size=total_volume,
        )
    
    return None


# ---------------------------------------------------------------------------
# Dynamic Query-Time Optimizer
# ---------------------------------------------------------------------------
def get_query_complexity(query: str) -> str:
    """Classify query complexity for dynamic parameter selection."""
    words = query.split()
    
    # Very short = likely symbol lookup
    if len(words) <= 2:
        return "symbol"
    
    # Has question words = needs more context
    if any(w in query.lower() for w in ["how", "why", "explain", "what does"]):
        return "complex"
    
    # Medium length = semantic search
    if len(words) <= 8:
        return "semantic"
    
    return "complex"


def get_dynamic_params(query: str, result_count: int = 0) -> Dict[str, Any]:
    """
    Get dynamically tuned parameters based on query characteristics.
    
    Call this at query time for adaptive behavior.
    """
    complexity = get_query_complexity(query)
    
    if complexity == "symbol":
        return {
            "dense_weight": 0.3,
            "lexical_weight": 0.7,
            "symbol_boost": 0.6,
            "ef_search": 64,  # Lower for speed
            "limit": 5,
        }
    elif complexity == "semantic":
        return {
            "dense_weight": 0.6,
            "lexical_weight": 0.4,
            "symbol_boost": 0.35,
            "ef_search": 128,
            "limit": 10,
        }
    else:  # complex
        return {
            "dense_weight": 0.7,
            "lexical_weight": 0.3,
            "symbol_boost": 0.2,
            "ef_search": 192,
            "limit": 15,
            "budget_tokens": 4000,  # More context for complex queries
        }


def get_adaptive_budget(query: str, file_count: int) -> int:
    """
    Calculate adaptive token budget based on query and results.
    """
    base = 2000
    
    # More files = need more budget to cover context
    file_factor = min(file_count * 150, 2000)
    
    # Complex queries need more reasoning space
    complexity = get_query_complexity(query)
    complexity_factor = 1.5 if complexity == "complex" else 1.0
    
    return int((base + file_factor) * complexity_factor)


# ---------------------------------------------------------------------------
# Main Analyzer
# ---------------------------------------------------------------------------
@dataclass
class TraceAnalysisReport:
    """Report from trace-based analysis."""
    timestamp: str
    trace_window: str
    insights: List[TraceInsight]
    recommended_changes: Dict[str, Any]
    tool_stats: Dict[str, Any] = field(default_factory=dict)
    token_usage: Dict[str, Any] = field(default_factory=dict)
    total_traces: int = 0
    dynamic_enabled: bool = True


async def analyze_traces(verbose: bool = False) -> TraceAnalysisReport:
    """
    Analyze OpenLit traces and generate optimization recommendations.
    """
    print("=" * 70)
    print("TRACE-BASED AUTO-TUNER v2")
    print("=" * 70)
    print("Analyzing production traces from ClickHouse...")
    
    # Get total trace count
    count_sql = f"""
    SELECT count() as cnt FROM openlit.otel_traces 
    WHERE Timestamp > now() - {get_interval_clause()}
    """
    count_result = await query_clickhouse(count_sql)
    total_traces = int(count_result[0]["cnt"]) if count_result else 0
    print(f"  Total traces ({get_window_label()}): {total_traces:,}")
    
    # Get tool usage stats
    print("\n[1/3] Analyzing tool usage patterns...")
    tool_stats = await get_tool_usage_stats()
    if tool_stats:
        top_tools = sorted(tool_stats.items(), key=lambda x: x[1]["calls"], reverse=True)[:5]
        for tool, stats in top_tools:
            print(f"  {tool}: {stats['calls']} calls, P90={stats['p90_latency_ms']:.0f}ms")
    
    insights: List[TraceInsight] = []
    
    # Run all analyzers
    print("\n[2/3] Running insight analyzers...")
    analyzers = [
        ("LLM Token Usage", analyze_llm_token_usage),
        ("Search Latency", analyze_search_latency),
        ("Query Patterns", analyze_query_patterns),
        ("Grounding Rate", analyze_grounding_rate),
        ("Rerank Effectiveness", analyze_rerank_effectiveness),
        ("Error Patterns", analyze_error_patterns),
        ("Time Patterns", analyze_time_patterns),
    ]
    
    for name, analyzer in analyzers:
        print(f"  {name}...", end=" ")
        try:
            insight = await analyzer()
            if insight:
                print(f"✓ ({insight.sample_size} samples, {insight.confidence:.0%} confidence)")
                insights.append(insight)
            else:
                print("○ No actionable insights")
        except Exception as e:
            print(f"✗ Error: {e}")
    
    # Build recommendations (high confidence only)
    print("\n[3/3] Generating recommendations...")
    recommended = {}
    for insight in insights:
        if insight.confidence >= 0.6:
            recommended[insight.metric] = insight.suggested_value
            print(f"  ✓ {insight.metric}: {insight.current_value} → {insight.suggested_value}")
    
    if not recommended:
        print("  No high-confidence changes")
    
    # Collect token usage stats
    token_usage = await analyze_context_token_usage()
    
    return TraceAnalysisReport(
        timestamp=datetime.now().isoformat(),
        trace_window=TRACE_WINDOW,
        insights=insights,
        recommended_changes=recommended,
        tool_stats=tool_stats,
        token_usage=token_usage,
        total_traces=total_traces,
    )


def print_trace_report(report: TraceAnalysisReport) -> None:
    """Print the trace analysis report."""
    print("\n" + "=" * 70)
    print("TRACE ANALYSIS REPORT")
    print("=" * 70)
    print(f"Timestamp: {report.timestamp}")
    print(f"Trace window: {report.trace_window}")
    print(f"Total traces: {report.total_traces:,}")
    print(f"Insights found: {len(report.insights)}")
    
    # Tool usage stats
    if report.tool_stats:
        print("\n" + "-" * 70)
        print("TOOL USAGE (24h):")
        print(f"  {'Tool':<25} {'Calls':>8} {'Avg(ms)':>10} {'P90(ms)':>10}")
        print("  " + "-" * 55)
        for tool, stats in sorted(report.tool_stats.items(), key=lambda x: x[1]["calls"], reverse=True)[:8]:
            print(f"  {tool:<25} {stats['calls']:>8} {stats['avg_latency_ms']:>10.0f} {stats['p90_latency_ms']:>10.0f}")
    
    # Token usage stats
    if report.token_usage:
        print("\n" + "-" * 70)
        print("TOKEN USAGE (24h):")
        llm = report.token_usage.get("llm", {})
        mcp = report.token_usage.get("mcp", {})
        eff = report.token_usage.get("efficiency", {})
        
        if llm.get("call_count", 0) > 0:
            print(f"  LLM Calls:         {llm.get('call_count', 0):,}")
            print(f"  Total Input:       {llm.get('total_input_tokens', 0):,} tokens")
            print(f"  Total Output:      {llm.get('total_output_tokens', 0):,} tokens")
            print(f"  Avg Input/Call:    {llm.get('avg_input_per_call', 0):,} tokens")
            print(f"  Avg Output/Call:   {llm.get('avg_output_per_call', 0):,} tokens")
            print(f"  Est. Cost (24h):   ${llm.get('estimated_cost_usd', 0):.4f}")
            
            if eff:
                print(f"\n  Input/Output Ratio: {eff.get('input_output_ratio', 0):.1f}x")
                if eff.get('tokens_per_mcp_call', 0) > 0:
                    print(f"  Tokens per MCP Call: {eff.get('tokens_per_mcp_call', 0):,}")
        
        if mcp.get("call_count", 0) > 0:
            print(f"\n  MCP Tool Calls:    {mcp.get('call_count', 0):,}")
            print(f"  Avg Payload:       {mcp.get('avg_payload_bytes', 0):,} bytes")
            print(f"  Avg Response:      {mcp.get('avg_response_bytes', 0):,} bytes")
            print(f"  Total Payload:     {mcp.get('total_payload_bytes', 0):,} bytes")
            print(f"  Total Response:    {mcp.get('total_response_bytes', 0):,} bytes")
    
    
    if report.insights:
        print("\n" + "-" * 70)
        print("INSIGHTS FROM PRODUCTION DATA:")
        
        for insight in report.insights:
            conf_bar = "█" * int(insight.confidence * 10) + "░" * (10 - int(insight.confidence * 10))
            status = "✓" if insight.confidence >= 0.6 else "○"
            trend = f" [{insight.trend}]" if insight.trend else ""
            
            print(f"\n  {status} {insight.metric}{trend}")
            print(f"    Current: {insight.current_value} → Suggested: {insight.suggested_value}")
            print(f"    Confidence: {insight.confidence:.0%} {conf_bar}")
            print(f"    Evidence: {insight.evidence}")
            print(f"    Sample size: {insight.sample_size}")
    
    if report.recommended_changes:
        print("\n" + "-" * 70)
        print("RECOMMENDED CHANGES (≥60% confidence):")
        for metric, value in report.recommended_changes.items():
            # Find the insight for this metric
            insight = next((i for i in report.insights if i.metric == metric), None)
            conf = f" ({insight.confidence:.0%})" if insight else ""
            print(f"  {metric}={value}{conf}")
        
        print("\n# Apply with:")
        print("  " + " ".join(f"{k}={v}" for k, v in report.recommended_changes.items()))
    else:
        print("\n✅ Current configuration is optimal based on production data!")
    
    print("\n" + "-" * 70)
    print("DYNAMIC PARAMETERS (per-query):")
    test_queries = {
        "symbol": "getUserById",
        "semantic": "database connection pooling implementation",
        "complex": "how does the hybrid search algorithm combine dense and lexical scoring",
    }
    for complexity, test_query in test_queries.items():
        params = get_dynamic_params(test_query)
        print(f"  {complexity:<10}: dense_w={params['dense_weight']}, lex_w={params['lexical_weight']}, boost={params.get('symbol_boost', 'N/A')}, ef={params['ef_search']}")
    
    print("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
async def main():
    global TRACE_WINDOW
    import argparse
    
    parser = argparse.ArgumentParser(description="Trace-Based Auto-Tuner")
    parser.add_argument("--output", type=str, help="Output JSON file")
    parser.add_argument(
        "--window", "-w", 
        type=str, 
        default="24h",
        choices=["1h", "6h", "12h", "24h", "7d", "30d"],
        help="Time window for trace analysis (default: 24h)"
    )
    args = parser.parse_args()
    
    # Set the global window
    TRACE_WINDOW = args.window
    
    report = await analyze_traces()
    print_trace_report(report)
    
    if args.output:
        data = {
            "timestamp": report.timestamp,
            "trace_window": report.trace_window,
            "insights": [
                {
                    "metric": i.metric,
                    "current": i.current_value,
                    "suggested": i.suggested_value,
                    "confidence": i.confidence,
                    "evidence": i.evidence,
                    "samples": i.sample_size,
                }
                for i in report.insights
            ],
            "recommended_changes": report.recommended_changes,
        }
        with open(args.output, "w") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"\nReport saved to: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
