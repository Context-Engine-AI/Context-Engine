#!/usr/bin/env python3
"""
Runtime Tunable Configuration for Context-Engine.

Thread-safe parameter store with:
- Live reload without restart
- Trace-based auto-tuning feedback loop
- Query complexity classification
- Parameter history tracking
"""

import asyncio
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable


# ---------------------------------------------------------------------------
# TunableConfig - Thread-safe parameter store with live reload
# ---------------------------------------------------------------------------
@dataclass
class TunableConfig:
    """
    Auto-tunable parameters with thread-safe live reload.
    
    Usage:
        from scripts.benchmarks.tunable_config import CONFIG
        
        # Read params (thread-safe)
        limit = CONFIG.default_limit
        
        # Update params (thread-safe)
        CONFIG.update(default_limit=15, rerank_top_n=100)
        
        # Get history
        print(CONFIG.get_history())
    """
    
    # --- Search params ---
    default_limit: int = 10
    default_per_path: int = 2
    context_lines: int = 5
    
    # --- Reranker params ---
    rerank_enabled: bool = True
    rerank_top_n: int = 100
    rerank_return_m: int = 20
    rerank_timeout_ms: int = 3000
    rerank_blend_weight: float = 0.6
    
    # --- LLM/GLM params ---
    max_tokens: int = 1200
    temperature: float = 0.0
    budget_tokens: int = 3000
    
    # --- Hybrid search weights ---
    dense_weight: float = 0.5
    lexical_weight: float = 0.5
    symbol_boost: float = 0.35
    recency_weight: float = 0.1
    
    # --- Query optimizer ---
    ef_search: int = 128
    ef_min: int = 64
    ef_max: int = 512
    
    # --- Timeouts ---
    tool_timeout_secs: float = 3600.0
    qdrant_timeout: int = 20
    
    # --- Internal state ---
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _history: List[Dict[str, Any]] = field(default_factory=list, repr=False)
    _callbacks: List[Callable] = field(default_factory=list, repr=False)
    _last_updated: float = field(default_factory=time.time, repr=False)
    
    def __post_init__(self):
        """Initialize from environment variables."""
        self._load_from_env()
    
    def _load_from_env(self):
        """Load initial values from environment variables."""
        env_mappings = {
            "default_limit": ("REPO_SEARCH_DEFAULT_LIMIT", int),
            "rerank_top_n": ("RERANKER_TOPN", int),
            "rerank_return_m": ("RERANKER_RETURN_M", int),
            "rerank_timeout_ms": ("RERANKER_TIMEOUT_MS", int),
            "rerank_blend_weight": ("RERANK_BLEND_WEIGHT", float),
            "max_tokens": ("DECODER_MAX_TOKENS", int),
            "budget_tokens": ("MICRO_BUDGET_TOKENS", int),
            "symbol_boost": ("HYBRID_SYMBOL_BOOST", float),
            "recency_weight": ("HYBRID_RECENCY_WEIGHT", float),
            "ef_search": ("QDRANT_EF_SEARCH", int),
            "qdrant_timeout": ("QDRANT_TIMEOUT", int),
        }
        
        for attr, (env_var, type_fn) in env_mappings.items():
            val = os.environ.get(env_var)
            if val:
                try:
                    setattr(self, attr, type_fn(val))
                except ValueError:
                    pass
    
    def update(self, source: str = "manual", **kwargs) -> Dict[str, Any]:
        """
        Update parameters thread-safely.
        
        Args:
            source: Who triggered the update (e.g., "trace_analyzer", "manual")
            **kwargs: Parameter name=value pairs
        
        Returns:
            Dict of changes made
        """
        changes = {}
        
        with self._lock:
            for key, value in kwargs.items():
                if hasattr(self, key) and not key.startswith("_"):
                    old_value = getattr(self, key)
                    if old_value != value:
                        setattr(self, key, value)
                        changes[key] = {"old": old_value, "new": value}
                        self._history.append({
                            "timestamp": datetime.now().isoformat(),
                            "source": source,
                            "param": key,
                            "old": old_value,
                            "new": value,
                        })
            
            if changes:
                self._last_updated = time.time()
        
        # Notify callbacks outside lock
        for callback in self._callbacks:
            try:
                callback(changes)
            except Exception as e:
                print(f"Config callback error: {e}")
        
        return changes
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a parameter value."""
        return getattr(self, key, default)
    
    def get_all(self) -> Dict[str, Any]:
        """Get all tunable parameters as a dict."""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith("_")
        }
    
    def get_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent parameter change history."""
        with self._lock:
            return self._history[-limit:]
    
    def on_change(self, callback: Callable[[Dict[str, Any]], None]):
        """Register a callback for parameter changes."""
        self._callbacks.append(callback)
    
    def reset_to_env(self):
        """Reset all parameters to environment variable values."""
        self._load_from_env()
        self._history.append({
            "timestamp": datetime.now().isoformat(),
            "source": "reset",
            "param": "*",
            "note": "Reset to environment values",
        })
    
    def save(self, path: Optional[str] = None):
        """
        Save current config to disk for persistence across restarts.
        """
        import json
        
        path = path or os.environ.get("CONFIG_PERSIST_PATH", "/tmp/context-engine-config.json")
        data = {
            "saved_at": datetime.now().isoformat(),
            "config": self.get_all(),
            "history": self._history[-20:],  # Last 20 changes
        }
        
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        
        return path
    
    def load(self, path: Optional[str] = None) -> bool:
        """
        Load config from disk. Returns True if loaded successfully.
        """
        import json
        
        path = path or os.environ.get("CONFIG_PERSIST_PATH", "/tmp/context-engine-config.json")
        
        if not os.path.exists(path):
            return False
        
        try:
            with open(path) as f:
                data = json.load(f)
            
            with self._lock:
                for key, value in data.get("config", {}).items():
                    if hasattr(self, key) and not key.startswith("_"):
                        setattr(self, key, value)
                
                self._history.append({
                    "timestamp": datetime.now().isoformat(),
                    "source": "load",
                    "param": "*",
                    "note": f"Loaded from {path}",
                })
            
            return True
        except Exception as e:
            print(f"Error loading config: {e}")
            return False


# Global singleton
CONFIG = TunableConfig()


# ---------------------------------------------------------------------------
# Query Complexity Classifier
# ---------------------------------------------------------------------------
def classify_query(query: str) -> Dict[str, Any]:
    """
    Classify query to select optimal parameters.
    
    Returns a dict of parameter overrides for this query type.
    """
    if not query:
        return {}
    
    tokens = query.split()
    query_lower = query.lower()
    
    # Detect query type
    is_lookup = len(tokens) <= 3 or any(w in query_lower for w in ["where is", "find", "show me", "get"])
    is_explain = any(w in query_lower for w in ["how", "why", "explain", "what does", "describe"])
    is_multi_file = any(w in query_lower for w in ["all", "every", "across", "related to", "depends on"])
    is_symbol = "_" in query or any(c.isupper() for c in query[1:] if len(query) > 1)
    
    if is_lookup and is_symbol:
        # Symbol lookup - fast, precise, lexical-heavy
        return {
            "limit": 5,
            "budget_tokens": 1500,
            "lexical_weight": 0.7,
            "dense_weight": 0.3,
            "symbol_boost": 0.6,
            "ef_search": 64,
        }
    elif is_explain:
        # Complex explanation - more context, higher limits
        return {
            "limit": 15,
            "budget_tokens": 5000,
            "max_tokens": 2000,
            "ef_search": 192,
        }
    elif is_multi_file:
        # Cross-file search - broad coverage
        return {
            "limit": 20,
            "per_path": 1,
            "budget_tokens": 4000,
            "ef_search": 192,
        }
    elif is_lookup:
        # Simple lookup
        return {
            "limit": 8,
            "budget_tokens": 2000,
        }
    
    # Default - balanced
    return {}


def get_params_for_query(query: str) -> Dict[str, Any]:
    """
    Get effective parameters for a query, merging base config with query-specific overrides.
    """
    base = CONFIG.get_all()
    overrides = classify_query(query)
    return {**base, **overrides}


# ---------------------------------------------------------------------------
# Trace-Based Feedback Loop
# ---------------------------------------------------------------------------
async def apply_trace_feedback(insights: List[Dict[str, Any]], auto_apply: bool = False) -> Dict[str, Any]:
    """
    Apply insights from trace analysis to the config.
    
    Args:
        insights: List of insights from trace_optimizer.analyze_traces()
        auto_apply: If True, automatically apply high-confidence changes
    
    Returns:
        Dict of applied changes
    """
    applied = {}
    
    for insight in insights:
        if not isinstance(insight, dict):
            # Handle TraceInsight dataclass
            insight = {
                "metric": getattr(insight, "metric", ""),
                "suggested_value": getattr(insight, "suggested_value", None),
                "confidence": getattr(insight, "confidence", 0),
            }
        
        metric = insight.get("metric", "")
        value = insight.get("suggested_value")
        confidence = insight.get("confidence", 0)
        
        if confidence < 0.6:
            continue
        
        # Map trace metrics to config params
        metric_to_config = {
            "QDRANT_EF_SEARCH": "ef_search",
            "HYBRID_SYMBOL_BOOST": "symbol_boost",
            "MICRO_BUDGET_TOKENS": "budget_tokens",
            "DECODER_MAX_TOKENS": "max_tokens",
            "RERANKER_TOPN": "rerank_top_n",
            "RERANKER_ENABLED": "rerank_enabled",
        }
        
        config_key = metric_to_config.get(metric)
        if config_key and value is not None:
            if auto_apply:
                changes = CONFIG.update(**{config_key: value}, source="trace_optimizer")
                applied.update(changes)
            else:
                applied[config_key] = {"pending": value, "confidence": confidence}
    
    return applied


# ---------------------------------------------------------------------------
# Periodic Auto-Tune Loop (optional background task)
# ---------------------------------------------------------------------------
_auto_tune_task: Optional[asyncio.Task] = None


async def start_auto_tune_loop(interval_minutes: int = 30):
    """
    Start a background loop that periodically analyzes traces and tunes config.
    """
    global _auto_tune_task
    
    async def loop():
        while True:
            try:
                from scripts.benchmarks.trace_optimizer import analyze_traces
                
                print(f"[AutoTune] Running trace analysis...")
                report = await analyze_traces()
                
                if report.insights:
                    applied = await apply_trace_feedback(report.insights, auto_apply=True)
                    if applied:
                        print(f"[AutoTune] Applied changes: {applied}")
                
            except Exception as e:
                print(f"[AutoTune] Error: {e}")
            
            await asyncio.sleep(interval_minutes * 60)
    
    _auto_tune_task = asyncio.create_task(loop())
    return _auto_tune_task


def stop_auto_tune_loop():
    """Stop the background auto-tune loop."""
    global _auto_tune_task
    if _auto_tune_task:
        _auto_tune_task.cancel()
        _auto_tune_task = None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Tunable Config Manager")
    parser.add_argument("--show", action="store_true", help="Show current config")
    parser.add_argument("--history", action="store_true", help="Show change history")
    parser.add_argument("--classify", type=str, help="Classify a query")
    args = parser.parse_args()
    
    if args.show:
        print("Current Configuration:")
        for k, v in CONFIG.get_all().items():
            print(f"  {k}: {v}")
    
    if args.history:
        print("Change History:")
        for entry in CONFIG.get_history():
            print(f"  {entry}")
    
    if args.classify:
        print(f"Query: {args.classify}")
        print(f"Params: {get_params_for_query(args.classify)}")
