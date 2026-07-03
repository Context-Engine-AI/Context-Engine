"""
Enhanced Recommendation Engine for Context-Engine Benchmarks.

Reads current env configuration and suggests specific tuning based on benchmark results.
Validates recommendations by running quick A/B tests before suggesting.
"""

import asyncio
import os
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Complete Tunable Knobs Registry (from .env)
# ---------------------------------------------------------------------------
TUNABLE_KNOBS = {
    # === Token/Budget Controls ===
    "MICRO_BUDGET_TOKENS": {
        "description": "Token budget for micro-span context retrieval",
        "default": 3000, "type": "int", "range": (500, 8000),
        "impacts": ["grounding_rate", "answer_quality", "latency"],
        "category": "tokens",
    },
    "DECODER_MAX_TOKENS": {
        "description": "Max tokens for LLM decoder output",
        "default": 1200, "type": "int", "range": (256, 4096),
        "impacts": ["answer_length", "cost"],
        "category": "tokens",
    },
    "MICRO_CHUNK_TOKENS": {
        "description": "Tokens per micro-chunk (smaller = more precise)",
        "default": 24, "type": "int", "range": (8, 64),
        "impacts": ["precision", "index_size"],
        "category": "tokens",
    },
    "CTX_SNIPPET_CHARS": {
        "description": "Max chars per snippet in context_answer",
        "default": 400, "type": "int", "range": (100, 1000),
        "impacts": ["context_density", "token_usage"],
        "category": "tokens",
    },
    
    # === Reranker Settings ===
    "RERANKER_ENABLED": {
        "description": "Enable cross-encoder reranking",
        "default": 1, "type": "bool", "range": (0, 1),
        "impacts": ["mrr", "latency"],
        "category": "reranker",
    },
    "RERANKER_TOPN": {
        "description": "Candidates to rerank (more = better quality, slower)",
        "default": 100, "type": "int", "range": (20, 300),
        "impacts": ["mrr", "recall", "latency"],
        "category": "reranker",
    },
    "RERANKER_RETURN_M": {
        "description": "Results to return after reranking",
        "default": 20, "type": "int", "range": (5, 50),
        "impacts": ["recall", "token_usage"],
        "category": "reranker",
    },
    "RERANKER_TIMEOUT_MS": {
        "description": "Reranker timeout (ms)",
        "default": 3000, "type": "int", "range": (1000, 10000),
        "impacts": ["timeout_rate", "latency"],
        "category": "reranker",
    },
    "RERANK_BLEND_WEIGHT": {
        "description": "Ratio of rerank vs fusion score (0=fusion, 1=rerank)",
        "default": 0.6, "type": "float", "range": (0.0, 1.0),
        "impacts": ["ranking_bias"],
        "category": "reranker",
    },
    "POST_RERANK_SYMBOL_BOOST": {
        "description": "Boost for exact symbol matches after rerank",
        "default": 1.0, "type": "float", "range": (0.0, 2.0),
        "impacts": ["symbol_precision"],
        "category": "reranker",
    },
    
    # === Query Optimizer (HNSW/EF tuning) ===
    "QUERY_OPTIMIZER_ADAPTIVE": {
        "description": "Enable adaptive ef optimization",
        "default": 1, "type": "bool", "range": (0, 1),
        "impacts": ["latency"],
        "category": "query_optimizer",
    },
    "QUERY_OPTIMIZER_MIN_EF": {
        "description": "Minimum ef for simple queries",
        "default": 64, "type": "int", "range": (16, 256),
        "impacts": ["simple_query_latency"],
        "category": "query_optimizer",
    },
    "QUERY_OPTIMIZER_MAX_EF": {
        "description": "Maximum ef for complex queries",
        "default": 512, "type": "int", "range": (128, 1024),
        "impacts": ["complex_query_recall"],
        "category": "query_optimizer",
    },
    "QDRANT_EF_SEARCH": {
        "description": "Default ef for Qdrant HNSW search",
        "default": 128, "type": "int", "range": (32, 512),
        "impacts": ["latency", "recall"],
        "category": "query_optimizer",
    },
    
    # === Hybrid Search ===
    "HYBRID_PER_PATH": {
        "description": "Max results per file path",
        "default": 1, "type": "int", "range": (1, 5),
        "impacts": ["diversity", "recall"],
        "category": "hybrid",
    },
    "HYBRID_SYMBOL_BOOST": {
        "description": "Score boost for symbol matches",
        "default": 0.35, "type": "float", "range": (0.0, 1.0),
        "impacts": ["symbol_precision", "mrr"],
        "category": "hybrid",
    },
    "HYBRID_RECENCY_WEIGHT": {
        "description": "Weight for recently modified files",
        "default": 0.1, "type": "float", "range": (0.0, 0.5),
        "impacts": ["recency_bias"],
        "category": "hybrid",
    },
    "HYBRID_EXPAND": {
        "description": "Enable query expansion in hybrid search",
        "default": 0, "type": "bool", "range": (0, 1),
        "impacts": ["recall", "latency"],
        "category": "hybrid",
    },
    "HYBRID_MINI_WEIGHT": {
        "description": "Weight for mini-vector gating",
        "default": 1.0, "type": "float", "range": (0.0, 2.0),
        "impacts": ["gate_precision"],
        "category": "hybrid",
    },
    
    # === ReFRAG / Micro-chunking ===
    "REFRAG_MODE": {
        "description": "Enable ReFRAG micro-chunking mode",
        "default": 1, "type": "bool", "range": (0, 1),
        "impacts": ["precision", "grounding"],
        "category": "refrag",
    },
    "REFRAG_GATE_FIRST": {
        "description": "Use mini-vectors to pre-filter before dense",
        "default": 1, "type": "bool", "range": (0, 1),
        "impacts": ["latency", "precision"],
        "category": "refrag",
    },
    "REFRAG_CANDIDATES": {
        "description": "Candidates for ReFRAG gating stage",
        "default": 200, "type": "int", "range": (50, 500),
        "impacts": ["recall", "latency"],
        "category": "refrag",
    },
    "MICRO_OUT_MAX_SPANS": {
        "description": "Max spans in micro-chunked output",
        "default": 10, "type": "int", "range": (3, 20),
        "impacts": ["context_density"],
        "category": "refrag",
    },
    "MICRO_MERGE_LINES": {
        "description": "Lines threshold to merge adjacent spans",
        "default": 6, "type": "int", "range": (2, 15),
        "impacts": ["span_coherence"],
        "category": "refrag",
    },
    
    # === PRF / Expansion ===
    "PRF_ENABLED": {
        "description": "Enable pseudo-relevance feedback",
        "default": 1, "type": "bool", "range": (0, 1),
        "impacts": ["recall", "latency"],
        "category": "expansion",
    },
    "RERANK_EXPAND": {
        "description": "Expand results after reranking",
        "default": 1, "type": "bool", "range": (0, 1),
        "impacts": ["diversity"],
        "category": "expansion",
    },
    
    # === Learning Reranker ===
    "RERANK_LLM_TEACHER": {
        "description": "Enable LLM teacher for learning reranker",
        "default": 1, "type": "bool", "range": (0, 1),
        "impacts": ["learning_quality"],
        "category": "learning",
    },
    "RERANK_LLM_SAMPLE_RATE": {
        "description": "Sample rate for LLM teacher feedback",
        "default": 1.0, "type": "float", "range": (0.0, 1.0),
        "impacts": ["learning_speed", "cost"],
        "category": "learning",
    },
    "RERANK_VICREG_WEIGHT": {
        "description": "Weight for VICReg regularization loss",
        "default": 0.1, "type": "float", "range": (0.0, 1.0),
        "impacts": ["embedding_diversity"],
        "category": "learning",
    },
    
    # === Limits / Defaults ===
    "REPO_SEARCH_DEFAULT_LIMIT": {
        "description": "Default result limit for repo_search",
        "default": 5, "type": "int", "range": (3, 20),
        "impacts": ["token_usage", "recall"],
        "category": "limits",
    },
    "INFO_REQUEST_LIMIT": {
        "description": "Default limit for info_request tool",
        "default": 10, "type": "int", "range": (3, 20),
        "impacts": ["token_usage"],
        "category": "limits",
    },
    "INFO_REQUEST_CONTEXT_LINES": {
        "description": "Context lines for info_request",
        "default": 5, "type": "int", "range": (0, 15),
        "impacts": ["context_density"],
        "category": "limits",
    },
    
    # === Lexical Search ===
    "LEX_MULTI_HASH": {
        "description": "Multi-hash buckets per token (reduces collisions)",
        "default": 3, "type": "int", "range": (1, 5),
        "impacts": ["lexical_precision"],
        "category": "lexical",
    },
    "LEX_BIGRAMS": {
        "description": "Enable bigram tokens for phrase matching",
        "default": 1, "type": "bool", "range": (0, 1),
        "impacts": ["phrase_matching"],
        "category": "lexical",
    },
    "LEX_BIGRAM_WEIGHT": {
        "description": "Weight for bigram matches",
        "default": 0.7, "type": "float", "range": (0.0, 1.0),
        "impacts": ["phrase_weight"],
        "category": "lexical",
    },
}


def get_current_config() -> Dict[str, Any]:
    """Read current configuration from environment."""
    config = {}
    for key, meta in TUNABLE_KNOBS.items():
        raw = os.environ.get(key)
        if raw is not None:
            try:
                if meta["type"] == "float":
                    config[key] = float(raw)
                elif meta["type"] == "bool":
                    config[key] = int(raw) if raw.isdigit() else (raw.lower() in ("1", "true", "yes"))
                else:
                    config[key] = int(raw)
            except ValueError:
                config[key] = meta["default"]
        else:
            config[key] = meta["default"]
    return config


def get_knob_info(key: str) -> Optional[Dict[str, Any]]:
    """Get metadata for a tunable knob."""
    return TUNABLE_KNOBS.get(key)


# ---------------------------------------------------------------------------
# Recommendation Logic
# ---------------------------------------------------------------------------
def generate_recommendations(
    benchmark_results: Dict[str, Any],
    current_config: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Generate actionable recommendations based on benchmark results.
    
    Returns list of recommendations with:
      - priority: high/medium/low
      - component: affected system component
      - metric: the metric driving this recommendation
      - current_value: current metric value
      - target_value: target to achieve
      - action: what to change
      - config_changes: specific env var changes
      - expected_impact: predicted improvement
      - confidence: high/medium/low based on evidence
    """
    if current_config is None:
        current_config = get_current_config()
    
    recommendations = []
    components = benchmark_results.get("components", benchmark_results)
    
    # Extract metrics
    eval_metrics = components.get("eval_harness", {}).get("metrics", {})
    eval_latency = components.get("eval_harness", {}).get("latency", {})
    trm_metrics = components.get("trm_reranker", {}).get("metrics", {})
    refrag_metrics = components.get("refrag", {}).get("metrics", {})
    expand_metrics = components.get("query_expansion", {}).get("metrics", {})
    
    mrr = eval_metrics.get("mrr", 0)
    recall_5 = eval_metrics.get("recall@5", 0)
    recall_10 = eval_metrics.get("recall@10", 0)
    precision_5 = eval_metrics.get("precision@5", 0)
    # Note: eval harness emits p90_ms, not p90
    p90_latency = eval_latency.get("p90_ms", 0) or trm_metrics.get("p90_latency_ms", 0)
    grounding = refrag_metrics.get("grounding_rate", 1.0)
    citations = refrag_metrics.get("avg_citations", 0)
    kendall_tau = trm_metrics.get("kendall_tau", 0)
    
    # =========================================================================
    # MRR Recommendations
    # =========================================================================
    if mrr < 0.7:
        topn = current_config.get("RERANKER_TOPN", 100)
        if topn < 150:
            recommendations.append({
                "priority": "high",
                "component": "reranker",
                "metric": "mrr",
                "current_value": mrr,
                "target_value": 0.7,
                "action": f"Increase RERANKER_TOPN: {topn} → {min(topn + 50, 200)}",
                "config_changes": {"RERANKER_TOPN": min(topn + 50, 200)},
                "expected_impact": "+5-10% MRR (more candidates = better top-1)",
                "confidence": "high",
                "rationale": "Reranking more candidates increases the chance of finding the best result.",
            })
        
        symbol_boost = current_config.get("HYBRID_SYMBOL_BOOST", 0.35)
        if symbol_boost < 0.45:
            recommendations.append({
                "priority": "medium",
                "component": "hybrid_search",
                "metric": "mrr",
                "current_value": mrr,
                "target_value": 0.7,
                "action": f"Increase HYBRID_SYMBOL_BOOST: {symbol_boost} → 0.5",
                "config_changes": {"HYBRID_SYMBOL_BOOST": 0.5},
                "expected_impact": "+3-5% MRR for symbol-based queries",
                "confidence": "medium",
                "rationale": "Many code queries target specific symbols; boosting them improves precision.",
            })
    
    # =========================================================================
    # Recall Recommendations
    # =========================================================================
    if recall_5 < 0.7:
        per_path = current_config.get("HYBRID_PER_PATH", 1)
        if per_path < 2:
            recommendations.append({
                "priority": "medium",
                "component": "hybrid_search",
                "metric": "recall@5",
                "current_value": recall_5,
                "target_value": 0.7,
                "action": f"Increase HYBRID_PER_PATH: {per_path} → 2",
                "config_changes": {"HYBRID_PER_PATH": 2},
                "expected_impact": "+10-15% Recall@5 (more results per file)",
                "confidence": "high",
                "rationale": "Relevant code often spans multiple spans in the same file.",
            })
        
        return_m = current_config.get("RERANKER_RETURN_M", 20)
        if return_m < 25:
            recommendations.append({
                "priority": "low",
                "component": "reranker",
                "metric": "recall@10",
                "current_value": recall_10,
                "target_value": 0.85,
                "action": f"Increase RERANKER_RETURN_M: {return_m} → 25",
                "config_changes": {"RERANKER_RETURN_M": 25},
                "expected_impact": "+5% Recall@10",
                "confidence": "medium",
                "rationale": "More returned results increases recall but costs more tokens.",
            })
    
    # =========================================================================
    # Grounding Recommendations
    # =========================================================================
    if grounding < 0.95:
        budget = current_config.get("MICRO_BUDGET_TOKENS", 3000)
        if budget < 4000:
            recommendations.append({
                "priority": "high",
                "component": "refrag",
                "metric": "grounding_rate",
                "current_value": grounding,
                "target_value": 0.95,
                "action": f"Increase MICRO_BUDGET_TOKENS: {budget} → {budget + 1000}",
                "config_changes": {"MICRO_BUDGET_TOKENS": budget + 1000},
                "expected_impact": "+10-20% grounding rate",
                "confidence": "high",
                "rationale": "More token budget = more context = better grounding.",
            })
    
    if citations < 5:
        max_spans = current_config.get("MICRO_OUT_MAX_SPANS", 10)
        if max_spans < 12:
            recommendations.append({
                "priority": "medium",
                "component": "refrag",
                "metric": "avg_citations",
                "current_value": citations,
                "target_value": 5,
                "action": f"Increase MICRO_OUT_MAX_SPANS: {max_spans} → 12",
                "config_changes": {"MICRO_OUT_MAX_SPANS": 12},
                "expected_impact": "+2-3 citations per answer",
                "confidence": "medium",
                "rationale": "More spans = more citable sources.",
            })
    
    # =========================================================================
    # Latency Recommendations
    # =========================================================================
    if p90_latency > 3000:
        topn = current_config.get("RERANKER_TOPN", 100)
        ef = current_config.get("QDRANT_EF_SEARCH", 128)
        
        recommendations.append({
            "priority": "high",
            "component": "latency",
            "metric": "p90_latency_ms",
            "current_value": p90_latency,
            "target_value": 3000,
            "action": "Reduce RERANKER_TOPN and/or QDRANT_EF_SEARCH",
            "config_changes": {
                "RERANKER_TOPN": max(topn - 30, 50),
                "QDRANT_EF_SEARCH": max(ef - 32, 64),
            },
            "expected_impact": f"-30-50% latency (current: {p90_latency:.0f}ms)",
            "confidence": "high",
            "rationale": "Fewer candidates to rerank + simpler HNSW graph traversal.",
        })
        
        if current_config.get("PRF_ENABLED", 1) == 1:
            recommendations.append({
                "priority": "medium",
                "component": "latency",
                "metric": "p90_latency_ms",
                "current_value": p90_latency,
                "target_value": 3000,
                "action": "Disable PRF_ENABLED for latency-critical paths",
                "config_changes": {"PRF_ENABLED": 0},
                "expected_impact": "-200-500ms per query",
                "confidence": "medium",
                "rationale": "PRF adds an extra retrieval round-trip.",
            })
    
    # =========================================================================
    # Learning Reranker Recommendations
    # =========================================================================
    if kendall_tau > 0.9:
        # High correlation = learning is working well
        if current_config.get("RERANK_LLM_SAMPLE_RATE", 1.0) > 0.5:
            recommendations.append({
                "priority": "low",
                "component": "learning",
                "metric": "kendall_tau",
                "current_value": kendall_tau,
                "target_value": 0.95,
                "action": "Reduce RERANK_LLM_SAMPLE_RATE to save cost",
                "config_changes": {"RERANK_LLM_SAMPLE_RATE": 0.3},
                "expected_impact": "-70% teacher API calls, minimal quality loss",
                "confidence": "medium",
                "rationale": "High tau indicates model has learned well; less feedback needed.",
            })
    
    # Sort by priority
    priority_order = {"high": 0, "medium": 1, "low": 2}
    recommendations.sort(key=lambda r: priority_order.get(r.get("priority", "low"), 3))
    
    return recommendations


# ---------------------------------------------------------------------------
# Validation (A/B Testing)
# ---------------------------------------------------------------------------
async def validate_recommendation(
    recommendation: Dict[str, Any],
    quick_test: bool = True,
) -> Dict[str, Any]:
    """
    Run a quick A/B test to validate a recommendation before suggesting.
    
    Returns the recommendation with added validation results.
    """
    if not quick_test:
        recommendation["validated"] = False
        recommendation["validation_note"] = "Validation skipped"
        return recommendation
    
    # Save current env
    original_env = {k: os.environ.get(k) for k in recommendation.get("config_changes", {})}
    
    try:
        from scripts.benchmarks.eval_harness import run_evaluation, EVAL_QUERIES
        
        # Run baseline (current config)
        baseline_result = await run_evaluation(name="baseline", queries=EVAL_QUERIES[:3])
        baseline_mrr = baseline_result.avg_mrr
        baseline_latency = baseline_result.p90_latency_ms
        
        # Apply recommended config
        for key, value in recommendation.get("config_changes", {}).items():
            os.environ[key] = str(value)
        
        # Run with new config
        test_result = await run_evaluation(name="test", queries=EVAL_QUERIES[:3])
        test_mrr = test_result.avg_mrr
        test_latency = test_result.p90_latency_ms
        
        # Calculate improvement
        mrr_delta = test_mrr - baseline_mrr
        latency_delta = test_latency - baseline_latency
        
        recommendation["validated"] = True
        recommendation["validation_results"] = {
            "baseline_mrr": round(baseline_mrr, 4),
            "test_mrr": round(test_mrr, 4),
            "mrr_delta": round(mrr_delta, 4),
            "baseline_latency": round(baseline_latency, 2),
            "test_latency": round(test_latency, 2),
            "latency_delta": round(latency_delta, 2),
        }
        
        # Determine if recommendation is valid
        if recommendation.get("metric") == "mrr" and mrr_delta > 0:
            recommendation["validation_status"] = "confirmed"
        elif recommendation.get("metric") == "p90_latency_ms" and latency_delta < 0:
            recommendation["validation_status"] = "confirmed"
        elif mrr_delta < -0.05:
            recommendation["validation_status"] = "rejected"
            recommendation["confidence"] = "low"
        else:
            recommendation["validation_status"] = "inconclusive"
        
    except Exception as e:
        recommendation["validated"] = False
        recommendation["validation_error"] = str(e)
    
    finally:
        # Restore original env
        for key, value in original_env.items():
            if value is not None:
                os.environ[key] = value
            elif key in os.environ:
                del os.environ[key]
    
    return recommendation


# ---------------------------------------------------------------------------
# Output Formatting
# ---------------------------------------------------------------------------
def print_recommendations(recommendations: List[Dict[str, Any]], verbose: bool = False) -> None:
    """Print recommendations in a readable format."""
    if not recommendations:
        print("\n✅ All metrics within targets - no changes recommended!")
        return
    
    print("\n" + "=" * 75)
    print("TUNING RECOMMENDATIONS")
    print("=" * 75)
    
    for rec in recommendations:
        priority = rec.get("priority", "").upper()
        icon = "🔴" if priority == "HIGH" else "🟡" if priority == "MEDIUM" else "🟢"
        validated = rec.get("validation_status", "")
        val_icon = " ✓" if validated == "confirmed" else " ?" if validated == "inconclusive" else ""
        
        print(f"\n{icon} [{priority}] {rec.get('component', '')} ({rec.get('metric', '')}){val_icon}")
        print(f"   Current: {rec.get('current_value', 'N/A')} → Target: {rec.get('target_value', 'N/A')}")
        print(f"   Action: {rec.get('action', '')}")
        print(f"   Impact: {rec.get('expected_impact', '')}")
        print(f"   Confidence: {rec.get('confidence', 'unknown')}")
        
        if rec.get("config_changes"):
            changes = ", ".join(f"{k}={v}" for k, v in rec["config_changes"].items())
            print(f"   Config: {changes}")
        
        if verbose and rec.get("rationale"):
            print(f"   Rationale: {rec.get('rationale')}")
        
        if rec.get("validation_results"):
            vr = rec["validation_results"]
            print(f"   Validated: MRR {vr['baseline_mrr']:.3f}→{vr['test_mrr']:.3f} ({vr['mrr_delta']:+.3f})")
    
    print("\n" + "=" * 75)


def generate_env_patch(recommendations: List[Dict[str, Any]]) -> str:
    """Generate .env patch from recommendations."""
    lines = ["# Recommended configuration changes", f"# Based on benchmark results", ""]
    
    for rec in recommendations:
        if rec.get("config_changes"):
            status = rec.get("validation_status", "")
            suffix = " (validated)" if status == "confirmed" else ""
            lines.append(f"# {rec.get('action', '')}{suffix}")
            for key, value in rec["config_changes"].items():
                lines.append(f"{key}={value}")
            lines.append("")
    
    return "\n".join(lines)


def list_all_knobs(category: Optional[str] = None) -> None:
    """List all tunable knobs, optionally filtered by category."""
    categories = set(k["category"] for k in TUNABLE_KNOBS.values())
    
    if category and category not in categories:
        print(f"Unknown category: {category}")
        print(f"Available: {', '.join(sorted(categories))}")
        return
    
    config = get_current_config()
    
    print("\n" + "=" * 75)
    print(f"TUNABLE KNOBS" + (f" ({category})" if category else ""))
    print("=" * 75)
    
    for key, meta in sorted(TUNABLE_KNOBS.items(), key=lambda x: (x[1]["category"], x[0])):
        if category and meta["category"] != category:
            continue
        
        current = config.get(key, meta["default"])
        default = meta["default"]
        range_str = f"[{meta['range'][0]}-{meta['range'][1]}]"
        
        diff = "=" if current == default else "≠"
        print(f"\n  {key} {diff}")
        print(f"    {meta['description']}")
        print(f"    Current: {current} | Default: {default} | Range: {range_str}")
        print(f"    Impacts: {', '.join(meta['impacts'])}")
