#!/usr/bin/env python3.11
"""
CoSQA Evaluation Runner for Context-Engine Benchmarks.

Runs evaluation against the CoSQA benchmark, computing:
- MRR@k (Mean Reciprocal Rank)
- NDCG@k (Normalized Discounted Cumulative Gain)
- Recall@k

## Context-Engine Features Used

**Search Pipeline (via repo_search):**
- [x] Hybrid search (dense + lexical RRF fusion)
- [x] ONNX reranker (BAAI/bge-reranker-base)
- [x] Query expansion (synonyms via HYBRID_EXPAND=1)
- [x] Semantic expansion (SEMANTIC_EXPANSION_ENABLED=1)
- [x] Multi-query fusion

**Indexing Pipeline:**
- [x] AST symbol extraction (_extract_symbols)
- [x] Import/call extraction
- [x] Enriched embeddings (symbols + imports + docstring + code)
- [x] Lexical hash vectors (for hybrid search)
- [ ] ReFRAG/micro-chunks (requires REFRAG_MODE=1, off by default)
- [ ] Pattern vectors (requires indexed pattern vectors)

**Not Applicable (CoSQA limitations):**
- N/A Semantic chunking (snippets are atomic units)
- N/A File-level metadata (no real file paths)
- N/A Git metadata (not a real repo)

## Environment Variables

Enable additional features:
    HYBRID_EXPAND=1          # Query expansion (synonyms)
    SEMANTIC_EXPANSION_ENABLED=1  # Semantic similarity expansion
    LLM_EXPAND_MAX=3         # LLM query expansion
    REFRAG_MODE=1            # ReFRAG micro-chunking (if indexed)
    HYBRID_IN_PROCESS=1      # Run hybrid search in-process

Reranking:
    RERANK_ENABLED=1         # Enable ONNX reranker (default: on)
    RERANK_IN_PROCESS=1      # Run reranker in-process (required for reliability)
    RERANK_TOP_N=50          # Number of candidates to rerank
    RERANK_RETURN_M=20       # Number of results to return after rerank

References:
- CoSQA+ paper: https://arxiv.org/abs/2406.11589
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dotenv import load_dotenv

# Load .env immediately to ensure all subsequent imports (like scripts.ingest.config)
# see the correct environment variables.
load_dotenv(override=True)

from scripts.benchmarks.qdrant_utils import (
    get_qdrant_client, 
    probe_pseudo_tags, 
    verify_config_compatibility
)

# Force-disable OpenLit/OTel for benchmarks so they never try to talk to openlit-dashboard
os.environ["OPENLIT_ENABLED"] = "0"
os.environ["OTEL_SDK_DISABLED"] = "true"

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# NOTE: .env loading moved to _load_benchmark_env() to avoid polluting
# environment when this module is imported (e.g., by tests or __init__.py).
# Call _load_benchmark_env() explicitly before running benchmarks.
_ENV_LOADED = False


def _load_benchmark_env() -> None:
    """Load .env file for benchmark config. Safe to call multiple times."""
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    _ENV_LOADED = True
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).parent.parent.parent.parent / ".env")
    except ImportError:
        pass  # dotenv optional

    # Disable repo auto-filter for benchmarks (benchmark docs don't have metadata.repo)
    if "REPO_AUTO_FILTER" not in os.environ:
        os.environ["REPO_AUTO_FILTER"] = "0"

    # Force localhost for Qdrant - benchmarks run outside Docker
    # Override any .env setting that might use Docker hostname (e.g., "qdrant:6333")
    qdrant_url = os.environ.get("QDRANT_URL", "")
    if not qdrant_url or "qdrant:" in qdrant_url:
        os.environ["QDRANT_URL"] = "http://localhost:6333"

    # Set defaults AFTER loading .env so .env takes priority
    os.environ.setdefault("RERANKER_MODEL", "jinaai/jina-reranker-v2-base-multilingual")
    os.environ.setdefault("RERANK_IN_PROCESS", "1")
    # Hard-disable learning/recursive reranker for deterministic benchmarks
    # (unless explicitly enabled via --learning-worker flag or COSQA_ENABLE_LEARNING env var)
    if not os.environ.get("COSQA_ENABLE_LEARNING") and os.environ.get("RERANK_LEARNING") != "1":
        os.environ["RERANK_LEARNING"] = "0"
        os.environ["RERANK_EVENTS_ENABLED"] = "0"
    # Disable sparse vectors for CoSQA benchmarks to avoid missing lex-sparse dims
    os.environ["LEX_SPARSE_MODE"] = "0"
    # Benchmarks should not be scoped or cached by workspace repo state
    os.environ["REPO_AUTO_FILTER"] = "0"          # search all repos (no auto-detected repo)
    os.environ["CURRENT_REPO"] = ""               # clear any inherited repo name
    os.environ.setdefault("HYBRID_RESULTS_CACHE_ENABLED", "0")
    os.environ.setdefault("HYBRID_RESULTS_CACHE", "0")

# Avoid tokenizers fork warning in benchmark runs.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from scripts.benchmarks.common import percentile, get_runtime_info, save_run_meta

# NOTE: scripts.benchmarks.core_indexer imports scripts.ingest.config, which reads
# env-derived constants (e.g. LEX_VECTOR_DIM) at import time. This runner
# intentionally loads .env lazily in _load_benchmark_env(), so we must avoid
# importing core_indexer at module import-time to prevent config drift when
# LEX_VECTOR_DIM is not the default.

# Collection name for CoSQA corpus
DEFAULT_COLLECTION = "cosqa-corpus"

# Baseline MRR from CoSQA+ paper (Table 5)
# https://arxiv.org/abs/2406.11589
#
# Notes on expected performance:
# - Context-Engine uses bge-base-en (general embedding), not code-specific
# - CodeBERT/UniXcoder are code-specific models trained on code search
# - With our hybrid search + rerank, we expect ~0.24-0.30 MRR
# - This beats BM25 (+45-70%) and general embeddings like CodeT5+
# - To match CodeBERT, we'd need a code-specific embedding model
PAPER_BASELINES = {
    "Lucene (BM25)": 0.167,            # CoSQA+ Table 5
    "BoW": 0.065,                      # CoSQA+ Table 5
    "CodeBERT": 0.392,                 # CoSQA+ Table 5 (code-specific)
    "UniXcoder": 0.319,                # CoSQA+ Table 5 (code-specific)
    "CodeT5+ embedding": 0.266,        # CoSQA+ Table 5
    "text-embedding-3-large": 0.393,   # CoSQA+ Table 5 (best, code-specific)
}


@dataclass
class CoSQAQueryResult:
    """Result for a single CoSQA query evaluation."""
    query_id: str
    query_text: str
    relevant_code_ids: List[str]
    retrieved_code_ids: List[str]
    mrr: float
    ndcg_5: float
    ndcg_10: float
    recall_1: float
    recall_5: float
    recall_10: float
    latency_ms: float
    hit_at_1: bool
    hit_at_5: bool
    hit_at_10: bool


@dataclass
class CoSQAReport:
    """CoSQA benchmark report."""
    name: str
    config: Dict[str, Any]
    total_queries: int
    corpus_size: int
    # Aggregate metrics
    mrr: float
    ndcg_5: float
    ndcg_10: float
    recall_1: float
    recall_5: float
    recall_10: float
    hit_rate_1: float
    hit_rate_5: float
    hit_rate_10: float
    # Latency
    avg_latency_ms: float
    p50_latency_ms: float
    p90_latency_ms: float
    p99_latency_ms: float
    # Per-query results
    results: List[CoSQAQueryResult] = field(default_factory=list)
    # Comparison to baselines
    baseline_comparison: Dict[str, float] = field(default_factory=dict)
    # Runtime info for reproducibility
    runtime_info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "config": self.config,
            "corpus_size": self.corpus_size,
            "total_queries": self.total_queries,
            "runtime_info": self.runtime_info,
            "metrics": {
                "mrr": round(self.mrr, 4),
                "ndcg@5": round(self.ndcg_5, 4),
                "ndcg@10": round(self.ndcg_10, 4),
                "recall@1": round(self.recall_1, 4),
                "recall@5": round(self.recall_5, 4),
                "recall@10": round(self.recall_10, 4),
                "hit_rate@1": round(self.hit_rate_1, 4),
                "hit_rate@5": round(self.hit_rate_5, 4),
                "hit_rate@10": round(self.hit_rate_10, 4),
            },
            "latency": {
                "avg_ms": round(self.avg_latency_ms, 2),
                "p50_ms": round(self.p50_latency_ms, 2),
                "p90_ms": round(self.p90_latency_ms, 2),
                "p99_ms": round(self.p99_latency_ms, 2),
            },
            "baseline_comparison": self.baseline_comparison,
            "per_query": [asdict(r) for r in self.results],
        }


# ---------------------------------------------------------------------------
# Metrics Computation
# ---------------------------------------------------------------------------

def compute_mrr(relevant_ids: List[str], retrieved_ids: List[str]) -> float:
    """Compute Mean Reciprocal Rank.

    MRR = 1/rank of first relevant item, or 0 if none found.
    """
    relevant_set = set(relevant_ids)
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_set:
            return 1.0 / (i + 1)
    return 0.0


def compute_dcg(relevances: List[float], k: int) -> float:
    """Compute Discounted Cumulative Gain at k."""
    dcg = 0.0
    for i, rel in enumerate(relevances[:k]):
        # DCG = sum(rel_i / log2(i+2))
        dcg += rel / math.log2(i + 2)
    return dcg


def compute_ndcg(relevant_ids: List[str], retrieved_ids: List[str], k: int) -> float:
    """Compute Normalized Discounted Cumulative Gain at k.

    NDCG@k = DCG@k / IDCG@k
    """
    if not relevant_ids:
        return 0.0

    relevant_set = set(relevant_ids)

    # Compute DCG: relevance is 1 if doc is relevant, 0 otherwise
    relevances = [1.0 if doc_id in relevant_set else 0.0 for doc_id in retrieved_ids[:k]]
    dcg = compute_dcg(relevances, k)

    # Compute IDCG: ideal ranking has all relevant docs first
    ideal_relevances = [1.0] * min(len(relevant_ids), k)
    idcg = compute_dcg(ideal_relevances, k)

    if idcg == 0:
        return 0.0
    return dcg / idcg


def compute_recall(relevant_ids: List[str], retrieved_ids: List[str], k: int) -> float:
    """Compute Recall at k.

    Recall@k = |relevant ∩ retrieved@k| / |relevant|
    """
    if not relevant_ids:
        return 0.0
    relevant_set = set(relevant_ids)
    hits = sum(1 for doc_id in retrieved_ids[:k] if doc_id in relevant_set)
    return hits / len(relevant_ids)


def has_hit(relevant_ids: List[str], retrieved_ids: List[str], k: int) -> bool:
    """Check if any relevant item is in top-k."""
    relevant_set = set(relevant_ids)
    return any(doc_id in relevant_set for doc_id in retrieved_ids[:k])


# ---------------------------------------------------------------------------
# Search Interface
# ---------------------------------------------------------------------------


def _scaled_rerank_top_n(corpus_size: int) -> int:
    """Return rerank_top_n scaled by corpus size for CoSQA benchmarks.

    Respects RERANKER_TOPN env var if set, otherwise uses heuristic:
        - <= 10k docs: 100 candidates
        - 10k–50k docs: 200 candidates
        - > 50k docs: 400 candidates (cap to keep reranking tractable)
    """
    # Respect env var if explicitly set
    env_topn = os.environ.get("RERANKER_TOPN", "").strip()
    if env_topn:
        try:
            return int(env_topn)
        except ValueError:
            pass

    # Fallback to corpus-scaled heuristic
    if corpus_size <= 0:
        return 100
    if corpus_size <= 500:
        return 100
    if corpus_size <= 2_000:
        # Scale linearly: 100 + (corpus_size / 20)
        # e.g. 1000 docs -> 150, 2000 docs -> 200
        return min(200, 100 + corpus_size // 20)
    if corpus_size <= 10_000:
        return 200
    if corpus_size <= 50_000:
        return 200
    return 400


async def search_cosqa_corpus(
    query: str,
    collection: str = DEFAULT_COLLECTION,
    limit: int = 10,
    rerank_enabled: bool = True,
    mode: str = "hybrid",
    rerank_top_n: Optional[int] = None,
    debug: bool = False,
) -> Tuple[List[str], float]:
    """Search CoSQA corpus using Context-Engine's repo_search.

    Uses the same search path as CoIR and production for consistency.

    Args:
        query: Natural language query
        collection: Qdrant collection name
        limit: Maximum results to return
        rerank_enabled: Whether to use reranker
        mode: Search mode (passed to repo_search)
        rerank_top_n: Optional candidate pool size for reranker
        debug: If True, print detailed debug info

    Returns:
        Tuple of (list of code_ids, latency_ms)
    """
    from scripts.mcp_indexer_server import repo_search

    start = time.perf_counter()

    # Use repo_search for consistency with CoIR and production.
    # If rerank_top_n is provided, use it; otherwise fall back to legacy default (100).
    eff_rerank_top_n = None
    if rerank_enabled:
        eff_rerank_top_n = rerank_top_n if rerank_top_n is not None else 100

    # When not reranking, request more results to allow for deduplication.
    # Chunked indexing may return multiple chunks per code_id, so we need
    # extra candidates to ensure we get `limit` unique code_ids after dedup.
    eff_limit = limit
    if not rerank_enabled:
        eff_limit = limit * 5  # Request 5x to account for chunk duplicates

    result = await repo_search(
        query=query,
        limit=eff_limit,
        collection=collection,
        rerank_enabled=rerank_enabled,
        rerank_top_n=eff_rerank_top_n,
        rerank_return_m=limit if rerank_enabled else None,  # Rerank down to limit
        mode=mode,
    )

    # Debug: dump full result
    if debug:
        print(f"\n{'='*70}")
        print(f"DEBUG SEARCH: {query[:80]}...")
        print(f"{'='*70}")
        for i, r in enumerate((result.get("results") or [])[:10]):
            payload = r.get("payload") or {}
            meta = payload.get("metadata") or {}
            code_id = r.get("code_id") or r.get("doc_id") or payload.get("code_id") or payload.get("_id")
            print(f"\n[{i+1}] code_id={code_id}")
            print(f"    path: {(r.get('path') or meta.get('path', ''))[:60]}")
            print(f"    score: {r.get('score', 0):.4f}")
            print(f"    why: {r.get('why', [])}")
            comps = r.get("components") or {}
            print(f"    components: rerank={comps.get('rerank_onnx', 'N/A')}, blend={comps.get('blended', 'N/A')}")
            # Show pseudo/tags if available
            pseudo = payload.get("pseudo") or meta.get("pseudo") or ""
            tags = payload.get("tags") or meta.get("tags") or ""
            if pseudo:
                print(f"    pseudo: {str(pseudo)[:100]}...")
            if tags:
                print(f"    tags: {str(tags)[:100]}...")
            # Code snippet
            code = payload.get("text") or payload.get("code") or ""
            if code:
                print(f"    code: {code[:120].replace(chr(10), ' ')}...")
        print(f"{'='*70}\n")

    def _coerce_id(val: Any) -> Optional[str]:
        if val is None:
            return None
        try:
            s = str(val).strip()
        except Exception:
            return None
        return s or None

    def _cosqa_id_from_path(p: str) -> Optional[str]:
        if not p:
            return None
        s = str(p).strip().replace("\\", "/")
        if not s:
            return None
        # Accept either "cosqa/<id>.py" or any path containing "/cosqa/<id>.py".
        # If no cosqa segment is present, fall back to basename (still useful for synthetic paths).
        if "/cosqa/" in s:
            s = s.split("/cosqa/", 1)[1]
        elif s.startswith("cosqa/"):
            s = s[len("cosqa/") :]
        # Use last path segment
        name = s.rsplit("/", 1)[-1]
        if name.endswith(".py"):
            name = name[: -3]
        return name.strip() or None

    # Extract stable code_ids for evaluation.
    # NOTE: rerank paths may not include payload; for CoSQA we can fall back to parsing
    # the synthetic path "cosqa/<code_id>.py".
    # IMPORTANT: Deduplicate by code_id to avoid wasting top-k slots on chunks from same doc.
    code_ids: List[str] = []
    seen_ids: set[str] = set()
    for idx, r in enumerate(result.get("results", []) or []):
        if not isinstance(r, dict):
            continue
        payload = r.get("payload")
        if not isinstance(payload, dict):
            payload = {}
        code_id = (
            _coerce_id(r.get("code_id"))
            or _coerce_id(r.get("doc_id"))
            or _coerce_id(payload.get("code_id"))
            or _coerce_id(payload.get("doc_id"))
            or _coerce_id(payload.get("_id"))
            or _coerce_id(payload.get("id"))
        )
        # DEBUG: Log extraction
        if debug and idx < 5:
            print(f"  [extract {idx}] r.code_id={r.get('code_id')}, payload.code_id={payload.get('code_id')}, final={code_id}")
        if not code_id:
            code_id = _cosqa_id_from_path(str(r.get("path") or ""))
            if debug and idx < 5:
                print(f"    -> fell back to path: {code_id}")
        if code_id and code_id not in seen_ids:
            code_ids.append(code_id)
            seen_ids.add(code_id)

    # Defensive: ensure we never return more than requested.
    if limit is not None and limit > 0:
        code_ids = code_ids[:limit]

    latency_ms = (time.perf_counter() - start) * 1000
    return code_ids, latency_ms


# ---------------------------------------------------------------------------
# Evaluation Runner
# ---------------------------------------------------------------------------

async def run_cosqa_benchmark(
    queries: List[Tuple[str, str, List[str]]],
    collection: str = DEFAULT_COLLECTION,
    corpus_size: int = 0,
    limit: int = 10,
    rerank_enabled: bool = True,
    mode: str = "hybrid",
    name: str = "cosqa",
    subset_note: Optional[str] = None,
    progress_callback: Optional[callable] = None,
) -> CoSQAReport:
    """Run CoSQA evaluation on provided queries.

    Args:
        queries: List of (query_id, query_text, relevant_code_ids)
        collection: Qdrant collection name
        corpus_size: Size of indexed corpus
        limit: Max results per query
        rerank_enabled: Whether to use reranker
        mode: Search mode passed to repo_search ("hybrid" or "dense")
        name: Benchmark run name
        progress_callback: Optional callback(completed, total)

    Returns:
        CoSQAReport with all metrics
    """

    # Load .env for benchmark config (only when actually running benchmark)
    _load_benchmark_env()

    # Get actual points_count from Qdrant for proper scaling (chunks != docs)
    from scripts.benchmarks.qdrant_utils import get_qdrant_client
    try:
        _client = get_qdrant_client()
        _coll_info = _client.get_collection(collection)
        points_count = _coll_info.points_count or corpus_size
    except Exception:
        points_count = corpus_size

    # Scale rerank_top_n with points_count (not corpus_size) so relevant docs reach reranker
    scaled_top_n = _scaled_rerank_top_n(points_count)

    # When rerank is disabled, widen candidate pool for hybrid search to avoid
    # recall collapse as corpus grows. The default per_query=24 is too small.
    # Use points_count for scaling since chunked indexing creates multiple points per doc.
    if not rerank_enabled and points_count > 0:
        # At least 100 candidates, up to 10x limit, capped by points_count
        hybrid_per_query = min(points_count, max(100, 10 * limit))
        os.environ["HYBRID_PER_QUERY"] = str(hybrid_per_query)

    results: List[CoSQAQueryResult] = []
    latencies: List[float] = []

    config = {
        "collection": collection,
        "limit": limit,
        "rerank_enabled": rerank_enabled,
        "rerank_top_n": scaled_top_n,
        "query_count": len(queries),
        "mode": mode,
        # Capture full config for reproducibility and auditability
        "env": {
            # Embedding & Model
            "EMBEDDING_MODEL": os.environ.get("EMBEDDING_MODEL", ""),
            "INDEX_DENSE_MODE": os.environ.get("INDEX_DENSE_MODE", ""),
            # Reranker
            "RERANKER_MODEL": os.environ.get("RERANKER_MODEL", ""),
            "RERANK_IN_PROCESS": os.environ.get("RERANK_IN_PROCESS", ""),
            "RERANK_LEARNING": os.environ.get("RERANK_LEARNING", ""),
            # Hybrid search
            "HYBRID_IN_PROCESS": os.environ.get("HYBRID_IN_PROCESS", ""),
            "HYBRID_EXPAND": os.environ.get("HYBRID_EXPAND", ""),
            "HYBRID_PER_QUERY": os.environ.get("HYBRID_PER_QUERY", ""),
            "SEMANTIC_EXPANSION_ENABLED": os.environ.get("SEMANTIC_EXPANSION_ENABLED", ""),
            "HYBRID_LEXICAL_TEXT_MODE": os.environ.get("HYBRID_LEXICAL_TEXT_MODE", ""),
            "QDRANT_TIMEOUT": os.environ.get("QDRANT_TIMEOUT", ""),
            # Boost/penalty settings (should all be "0" for fair benchmarks)
            "FNAME_BOOST": os.environ.get("FNAME_BOOST", ""),
            "HYBRID_SYMBOL_BOOST": os.environ.get("HYBRID_SYMBOL_BOOST", ""),
            "HYBRID_IMPLEMENTATION_BOOST": os.environ.get("HYBRID_IMPLEMENTATION_BOOST", ""),
            "HYBRID_TEST_FILE_PENALTY": os.environ.get("HYBRID_TEST_FILE_PENALTY", ""),
            "POST_RERANK_SYMBOL_BOOST": os.environ.get("POST_RERANK_SYMBOL_BOOST", ""),
            # LLM features (should be "0" for baseline benchmarks)
            "REFRAG_PSEUDO_DESCRIBE": os.environ.get("REFRAG_PSEUDO_DESCRIBE", ""),
            "LLM_EXPAND_MAX": os.environ.get("LLM_EXPAND_MAX", ""),
            "DENSE_LLM_EXPAND": os.environ.get("DENSE_LLM_EXPAND", ""),
        },
    }
    if subset_note:
        config["subset_note"] = subset_note

    print(
        f"Running CoSQA benchmark: {len(queries)} queries, limit={limit}, "
        f"mode={mode}, rerank={rerank_enabled}, rerank_top_n={scaled_top_n}"
    )
    
    # Warmup query to avoid cold-start latency in first real query
    # This ensures embedding model and reranker are loaded before timing
    if queries:
        warmup_query = queries[0][1]  # Use first query text
        try:
            await search_cosqa_corpus(
                query=warmup_query,
                collection=collection,
                limit=5,
                rerank_enabled=rerank_enabled,
                rerank_top_n=50 if rerank_enabled else None,
                mode=mode,
            )
            print("  [warmup] Completed warmup query")
        except Exception as e:
            print(f"  [warmup] Warning: warmup failed: {e}")
    
    error_count = 0  # Track silent failures

    # Parallel query evaluation with controlled concurrency
    concurrency = int(os.environ.get("COSQA_QUERY_CONCURRENCY", "16"))
    semaphore = asyncio.Semaphore(concurrency)

    async def eval_single_query(idx: int, query_id: str, query_text: str, relevant_ids: List[str]):
        """Evaluate a single query with semaphore-controlled concurrency."""
        async with semaphore:
            try:
                retrieved_ids, latency_ms = await search_cosqa_corpus(
                    query=query_text,
                    collection=collection,
                    limit=limit,
                    rerank_enabled=rerank_enabled,
                    rerank_top_n=scaled_top_n if rerank_enabled else None,
                    mode=mode,
                )
            except Exception as e:
                print(f"  [ERROR] Query {idx+1} failed: {e}")
                retrieved_ids = []
                latency_ms = 0.0
                return (idx, query_id, query_text, relevant_ids, retrieved_ids, latency_ms, True)
            return (idx, query_id, query_text, relevant_ids, retrieved_ids, latency_ms, False)

    # Launch all queries in parallel
    tasks = [
        eval_single_query(idx, query_id, query_text, relevant_ids)
        for idx, (query_id, query_text, relevant_ids) in enumerate(queries)
    ]
    query_results = await asyncio.gather(*tasks)

    # Process results in order
    for idx, query_id, query_text, relevant_ids, retrieved_ids, latency_ms, had_error in query_results:
        if had_error:
            error_count += 1
        latencies.append(latency_ms)

        # Compute metrics
        mrr = compute_mrr(relevant_ids, retrieved_ids)
        ndcg_5 = compute_ndcg(relevant_ids, retrieved_ids, 5)
        ndcg_10 = compute_ndcg(relevant_ids, retrieved_ids, 10)
        recall_1 = compute_recall(relevant_ids, retrieved_ids, 1)
        recall_5 = compute_recall(relevant_ids, retrieved_ids, 5)
        recall_10 = compute_recall(relevant_ids, retrieved_ids, 10)
        hit_1 = has_hit(relevant_ids, retrieved_ids, 1)
        hit_5 = has_hit(relevant_ids, retrieved_ids, 5)
        hit_10 = has_hit(relevant_ids, retrieved_ids, 10)

        # DEBUG: Print miss details inline
        if not hit_10:
            print(f"\n  [MISS] q={query_id} expected={relevant_ids}")
            print(f"         got={retrieved_ids[:5]}")
            print(f"         query: {query_text[:60]}...")

        results.append(CoSQAQueryResult(
            query_id=query_id,
            query_text=query_text,
            relevant_code_ids=relevant_ids,
            retrieved_code_ids=retrieved_ids[:limit],
            mrr=mrr,
            ndcg_5=ndcg_5,
            ndcg_10=ndcg_10,
            recall_1=recall_1,
            recall_5=recall_5,
            recall_10=recall_10,
            latency_ms=latency_ms,
            hit_at_1=hit_1,
            hit_at_5=hit_5,
            hit_at_10=hit_10,
        ))

    # Progress (after parallel completion)
    if progress_callback:
        progress_callback(len(queries), len(queries))
    print(f"  Evaluated {len(queries)}/{len(queries)} queries (parallel, concurrency={concurrency})")

    # Aggregate metrics
    n = len(results)
    avg_mrr = sum(r.mrr for r in results) / n if n else 0
    avg_ndcg_5 = sum(r.ndcg_5 for r in results) / n if n else 0
    avg_ndcg_10 = sum(r.ndcg_10 for r in results) / n if n else 0
    avg_recall_1 = sum(r.recall_1 for r in results) / n if n else 0
    avg_recall_5 = sum(r.recall_5 for r in results) / n if n else 0
    avg_recall_10 = sum(r.recall_10 for r in results) / n if n else 0
    hit_rate_1 = sum(1 for r in results if r.hit_at_1) / n if n else 0
    hit_rate_5 = sum(1 for r in results if r.hit_at_5) / n if n else 0
    hit_rate_10 = sum(1 for r in results if r.hit_at_10) / n if n else 0

    # Baseline comparison
    baseline_comparison = {}
    for name_b, mrr_b in PAPER_BASELINES.items():
        delta = avg_mrr - mrr_b
        baseline_comparison[name_b] = {
            "paper_mrr": mrr_b,
            "our_mrr": round(avg_mrr, 4),
            "delta": round(delta, 4),
            "improvement": f"{delta/mrr_b*100:+.1f}%" if mrr_b else "N/A",
        }

    # Add error count to config for visibility in report
    if error_count > 0:
        config["error_count"] = error_count
        config["error_rate"] = f"{error_count / n * 100:.1f}%" if n else "N/A"

    report = CoSQAReport(
        name=name,
        config=config,
        total_queries=n,
        corpus_size=corpus_size,
        mrr=avg_mrr,
        ndcg_5=avg_ndcg_5,
        ndcg_10=avg_ndcg_10,
        recall_1=avg_recall_1,
        recall_5=avg_recall_5,
        recall_10=avg_recall_10,
        hit_rate_1=hit_rate_1,
        hit_rate_5=hit_rate_5,
        hit_rate_10=hit_rate_10,
        avg_latency_ms=sum(latencies) / len(latencies) if latencies else 0,
        p50_latency_ms=percentile(latencies, 0.50),
        p90_latency_ms=percentile(latencies, 0.90),
        p99_latency_ms=percentile(latencies, 0.99),
        results=results,
        baseline_comparison=baseline_comparison,
        runtime_info=get_runtime_info(),
    )

    return report



def print_report(report: CoSQAReport) -> None:
    """Print human-readable benchmark report."""
    print("\n" + "=" * 70)
    print(f"CoSQA BENCHMARK REPORT: {report.name}")
    print("=" * 70)

    print(f"Queries: {report.total_queries} | Corpus: {report.corpus_size}")
    print(f"Config: {report.config}")
    if report.config.get("subset_note"):
        print(f"NOTE: {report.config['subset_note']}")

    print("\n" + "-" * 70)
    print("METRICS:")
    print(f"  MRR:        {report.mrr:.4f}")
    print(f"  NDCG@5:     {report.ndcg_5:.4f}  |  NDCG@10:    {report.ndcg_10:.4f}")
    print(
        f"  Recall@1:   {report.recall_1:.4f}  |  Recall@5:   {report.recall_5:.4f}  |  Recall@10:  {report.recall_10:.4f}"
    )
    print(
        f"  Hit@1:      {report.hit_rate_1:.4f}  |  Hit@5:      {report.hit_rate_5:.4f}  |  Hit@10:     {report.hit_rate_10:.4f}"
    )

    print("\n" + "-" * 70)
    print("LATENCY:")
    print(
        f"  Avg: {report.avg_latency_ms:.1f}ms | P50: {report.p50_latency_ms:.1f}ms | P90: {report.p90_latency_ms:.1f}ms | P99: {report.p99_latency_ms:.1f}ms"
    )

    print("\n" + "-" * 70)
    print("BASELINE COMPARISON (MRR):")
    if report.config.get("subset_note"):
        print("  NOTE: Subset results are not comparable to paper baselines.")
    for name, comp in report.baseline_comparison.items():
        print(f"  vs {name}: {comp['paper_mrr']:.3f} → {comp['our_mrr']:.3f} ({comp['improvement']})")

    print("=" * 70)


def _prepare_cosqa_subset(
    dataset: "CoSQADataset",
    *,
    query_limit: Optional[int],
    corpus_limit: Optional[int],
    corpus_seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Tuple[str, str, List[str]]]]:
    """Prepare (corpus, queries) for a CoSQA run.

    If corpus_limit is set, we build a qrels-covered subset:
    - take the first `query_limit` evaluation queries (or all queries)
    - include *all* their relevant docs in the corpus subset
    - fill remaining slots with random distractors (seeded)

    This is meant for tuning/smoke runs so the subset behaves like the full
    benchmark (no missing-qrels collapse).
    """
    from scripts.benchmarks.cosqa.dataset import get_queries_for_evaluation

    all_queries = get_queries_for_evaluation(dataset, limit=None)
    queries = all_queries[:query_limit] if query_limit else all_queries

    import random

    all_corpus_entries = list(dataset.iter_corpus())

    # Full corpus
    if not corpus_limit:
        corpus = [entry.to_index_payload() for entry in all_corpus_entries]
        return corpus, queries

    all_corpus_ids = {e.code_id for e in all_corpus_entries}

    # Collect all relevant docs for the selected queries
    required_ids: set[str] = set()
    for _qid, _qtext, rel_ids in queries:
        required_ids.update(rel_ids)

    # Debug: check if required docs exist in corpus
    missing_ids = required_ids - all_corpus_ids
    if missing_ids:
        print(f"  [WARN] {len(missing_ids)} required docs NOT in corpus: {list(missing_ids)[:5]}...")

    # Preserve corpus order for required docs (deterministic)
    required_entries = [e for e in all_corpus_entries if e.code_id in required_ids]
    print(f"  [subset] {len(required_ids)} required docs, {len(required_entries)} found in corpus")

    # If the limit is too small to include all required docs, expand it.
    eff_corpus_limit = int(corpus_limit)
    if len(required_entries) > eff_corpus_limit:
        print(
            f"  [WARN] corpus_limit={corpus_limit} is smaller than required relevant docs "
            f"({len(required_entries)}). Expanding corpus to include all required docs."
        )
        eff_corpus_limit = len(required_entries)

    # Fill remaining slots with random distractors
    remaining = [e for e in all_corpus_entries if e.code_id not in required_ids]
    rng = random.Random(int(corpus_seed))
    extra_n = max(0, min(eff_corpus_limit - len(required_entries), len(remaining)))
    extra_entries = rng.sample(remaining, extra_n) if extra_n else []

    corpus_entries = required_entries + extra_entries
    corpus = [entry.to_index_payload() for entry in corpus_entries]
    return corpus, queries


async def run_full_benchmark(
    split: str = "test",
    collection: str = DEFAULT_COLLECTION,
    limit: int = 10,
    query_limit: Optional[int] = None,
    corpus_limit: Optional[int] = None,
    rerank_enabled: bool = True,
    mode: str = "hybrid",
    recreate_index: bool = False,
    index_only: bool = False,
    skip_index: bool = False,
) -> CoSQAReport:
    """Run complete CoSQA benchmark: download, index, evaluate.

    Args:
        split: Dataset split to use
        collection: Qdrant collection name
        limit: Max results per query
        query_limit: Limit number of queries (for quick testing)
        corpus_limit: Limit corpus size (for quick testing)
        rerank_enabled: Whether to use reranker
        mode: Search mode passed to repo_search ("hybrid" or "dense")
        recreate_index: Whether to recreate the index

    Returns:
        CoSQAReport with all metrics
    """
    # Load .env for benchmark config (only when actually running benchmark)
    _load_benchmark_env()

    # Disable ALL artificial boosts/penalties for pure semantic/lexical benchmarking.
    # These heuristics inflate/deflate scores based on file type, path, symbol matches etc.,
    # obscuring the true retrieval quality of the embedding + lexical pipeline.
    # Use --pure-semantic=false if you want prod-like behavior with heuristics enabled.
    
    # Filename/symbol match boosts
    os.environ["FNAME_BOOST"] = "0"
    os.environ["HYBRID_SYMBOL_BOOST"] = "0"
    os.environ["HYBRID_SYMBOL_EQUALITY_BOOST"] = "0"
    os.environ["POST_RERANK_SYMBOL_BOOST"] = "1.0"  # Neutral (no boost)
    
    # File-type boosts/penalties
    os.environ["HYBRID_CORE_FILE_BOOST"] = "0"
    os.environ["HYBRID_IMPLEMENTATION_BOOST"] = "0"
    os.environ["HYBRID_TEST_FILE_PENALTY"] = "0"
    os.environ["HYBRID_CONFIG_FILE_PENALTY"] = "0"
    os.environ["HYBRID_DOCUMENTATION_PENALTY"] = "0"
    os.environ["HYBRID_VENDOR_PENALTY"] = "0"
    os.environ["HYBRID_INTENT_IMPL_BOOST"] = "0"
    os.environ["HYBRID_LANG_MATCH_BOOST"] = "0"
    os.environ["HYBRID_RECENCY_WEIGHT"] = "0"

    # Set EMBEDDING_SEED for deterministic embeddings (like CoIR)
    os.environ.setdefault("EMBEDDING_SEED", "42")

    # Keep search env aligned with the requested collection to avoid stale defaults
    os.environ["COLLECTION_NAME"] = collection

    from scripts.benchmarks.cosqa.dataset import load_cosqa
    from scripts.benchmarks.cosqa.indexer import index_corpus

    # Step 1: Load dataset
    print("\n▶ Step 1: Loading CoSQA dataset...")
    dataset = load_cosqa(split=split)

    corpus, queries = _prepare_cosqa_subset(
        dataset,
        query_limit=query_limit,
        corpus_limit=corpus_limit,
        corpus_seed=42,
    )

    # Step 2: Index corpus
    print("\n▶ Step 2: Indexing corpus...")
    if corpus_limit:
        print(f"  Limited corpus to {len(corpus)} entries")

    if skip_index:
        print("  [skip-index] Skipping indexing...")
        result = {"reused": True, "indexed": len(corpus), "skipped": 0, "errors": 0}
    else:
        # Check if already indexed (use fingerprint matching, not just points_count)
        # The indexer handles fingerprint checking internally and will recreate if needed
        result = index_corpus(corpus, collection=collection, recreate=recreate_index)
    if result.get("reused"):
        print(f"  Corpus already indexed (fingerprint match, {result.get('indexed', 0)} entries)")
        # Warn if current config doesn't match how collection was indexed
        from scripts.benchmarks.core_indexer import BenchmarkDoc, warn_config_mismatch
        docs = [BenchmarkDoc(doc_id=c.get("code_id", ""), text=c.get("code", ""), metadata=c) for c in corpus[:100]]
        warning = warn_config_mismatch(None, collection, docs)  # Pass None to use default client
        if warning:
            print(f"\n{warning}\n")
    else:
        print(f"  Indexed {result.get('indexed', 0)} entries ({result.get('skipped', 0)} skipped, {result.get('errors', 0)} errors)")

    # Probe collection for pseudo/tags presence (validation)
    probe_pseudo_tags(collection)

    if index_only:
        print("\n[index-only] Ingestion complete. Exiting.")
        return None  # Early exit, no report
    print("\n▶ Step 3: Preparing queries...")
    print(f"  {len(queries)} queries with relevance judgments")

    # Step 4: Run evaluation
    print("\n▶ Step 4: Running evaluation...")
    subset_note = None
    if corpus_limit or query_limit:
        parts = []
        if corpus_limit:
            parts.append(f"corpus_limit={corpus_limit}")
        if query_limit:
            parts.append(f"query_limit={query_limit}")
        subset_note = f"subset evaluation ({', '.join(parts)})"
    report = await run_cosqa_benchmark(
        queries=queries,
        collection=collection,
        corpus_size=len(corpus),
        limit=limit,
        rerank_enabled=rerank_enabled,
        mode=mode,
        name=f"cosqa-{split}",
        subset_note=subset_note,
    )

    return report


def _spawn_learning_worker(collection: str, project_root: Path) -> subprocess.Popen:
    cmd = [
        sys.executable,
        str(project_root / "scripts" / "learning_reranker_worker.py"),
        "--daemon",
        "--collection",
        collection,
    ]
    return subprocess.Popen(cmd, cwd=project_root, env=os.environ.copy())


def main():
    """CLI entrypoint for CoSQA benchmark."""

    parser = argparse.ArgumentParser(description="CoSQA Benchmark for Context-Engine")
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"],
                        help="Dataset split to use")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION,
                        help="Qdrant collection name")
    parser.add_argument("--limit", type=int, default=10,
                        help="Max results per query")
    parser.add_argument("--query-limit", type=int, default=None,
                        help="Limit number of queries (for testing)")
    parser.add_argument("--corpus-limit", type=int, default=None,
                        help="Limit corpus size (for testing)")
    parser.add_argument("--no-rerank", action="store_true",
                        help="Disable reranker")
    parser.add_argument("--no-expand", action="store_true",
                        help="Disable query expansion")
    parser.add_argument("--recreate", action="store_true",
                        help="Recreate index from scratch")
    parser.add_argument("--learning-worker", action="store_true",
                        help="Spawn learning reranker worker during the run (enables learning + event logging)")
    parser.add_argument("--pure-semantic", action="store_true",
                        help="Disable FNAME_BOOST and other heuristics (old hardened mode)")
    parser.add_argument("--enable-llm", action="store_true",
                        help="Enable search-time LLM query expansion (for semantic-rich eval)")
    # Generate default output filename with timestamp
    default_output = f"cosqa_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    parser.add_argument("--output", type=str, default=default_output,
                        help="Output JSON file (default: timestamped file in current directory)")
    parser.add_argument("--failures", action="store_true",
                        help="Show failed queries (recall@10 = 0)")
    parser.add_argument("--debug-failures", action="store_true",
                        help="Re-run failed queries with detailed debug output")
    parser.add_argument("--index-only", action="store_true",
                        help="Run ingestion only and exit")
    parser.add_argument("--skip-index", action="store_true",
                        help="Skip indexing if collection already exists")
    parser.add_argument("--mode", type=str, default="hybrid", choices=["hybrid", "dense", "lexical"],
                        help="Search mode: 'hybrid' (default), 'dense' (pure semantic), or 'lexical' (pure BM25-style)")
    args = parser.parse_args()

    # Enable Context-Engine features for accurate benchmarking.
    # Semantic expansion is always enabled (it may still be a no-op if query expansion is disabled).
    os.environ["SEMANTIC_EXPANSION_ENABLED"] = "1"
    if not args.no_expand:
        os.environ["HYBRID_EXPAND"] = "1"
    else:
        # Avoid env drift from shell/.env: explicitly disable query expansion when requested.
        os.environ["HYBRID_EXPAND"] = "0"
    os.environ["HYBRID_IN_PROCESS"] = "1"  # Use in-process hybrid search
    os.environ["RERANK_IN_PROCESS"] = "1"  # Use in-process reranker (required)

    # Handle --mode dense: pure semantic search without lexical components
    if args.mode == "dense":
        os.environ["HYBRID_LEXICAL_WEIGHT"] = "0"
        os.environ["HYBRID_LEX_VECTOR_WEIGHT"] = "0"
        os.environ["HYBRID_EXPAND"] = "0"
        os.environ["SEMANTIC_EXPANSION_ENABLED"] = "0"
        print("  [mode=dense] Pure semantic search enabled (no lexical, no expansion)")
        # NOTE: Previously auto-disabled rerank for dense mode, but we want to test dense+rerank
        # Use --no-rerank explicitly if you want dense without reranking

    # Handle --mode lexical: pure lexical search without dense components
    elif args.mode == "lexical":
        os.environ["HYBRID_DENSE_WEIGHT"] = "0"
        os.environ["HYBRID_LEXICAL_WEIGHT"] = "1.0"
        os.environ["HYBRID_LEX_VECTOR_WEIGHT"] = "1.0"
        os.environ["HYBRID_EXPAND"] = "0"
        os.environ["SEMANTIC_EXPANSION_ENABLED"] = "0"
        print("  [mode=lexical] Pure lexical search enabled (no dense embeddings)")

    # Disable all LLM calls by default for fast benchmarking
    # Use --enable-llm to enable search-time LLM query expansion
    # Note: GLM pseudo/tags at index time is controlled by .env (REFRAG_PSEUDO_DESCRIBE=1) + reindex
    if not args.enable_llm:
        os.environ["LLM_EXPAND_MAX"] = "0"  # Disable LLM query expansion
        os.environ["REFRAG_DECODER"] = "0"  # Disable decoder entirely
    else:
        # Enable search-time LLM query expansion
        os.environ.setdefault("LLM_EXPAND_MAX", "3")
        print("  [enable-llm] LLM query expansion ENABLED (search-time)")

    if args.pure_semantic:
        print("  [pure-semantic] Boosts disabled (now default behavior)")

    # Set reranker model paths (relative to project root)
    _project_root = Path(__file__).parent.parent.parent.parent
    os.environ["RERANKER_ONNX_PATH"] = str(_project_root / "models" / "model_qint8_avx512_vnni.onnx")
    os.environ["RERANKER_TOKENIZER_PATH"] = str(_project_root / "models" / "tokenizer.json")

    learning_proc = None
    if args.learning_worker:
        if args.no_rerank:
            print("  [WARN] --learning-worker ignored because --no-rerank is set")
        else:
            # Pre-compute PCA initialization for projection layer (cold-start fix)
            print("  [learning] Pre-computing PCA initialization...")
            from scripts.benchmarks.cosqa.pca_init import compute_pca_init_for_collection
            pca_success = compute_pca_init_for_collection(
                collection=args.collection,
                sample_limit=1000,
            )
            if not pca_success:
                print("  [WARN] PCA initialization failed, using random init")

            os.environ["RERANK_LEARNING"] = "1"
            os.environ["RERANK_EVENTS_ENABLED"] = "1"
            learning_proc = _spawn_learning_worker(args.collection, _project_root)
            print(f"  [learning-worker] Started (pid {learning_proc.pid}) for {args.collection}")

    # Verify config compatibility BEFORE running anything
    if not args.recreate:
        try:
            # We use get_qdrant_client() which is already imported
            verify_config_compatibility(get_qdrant_client(), args.collection)
        except Exception as e:
            print(f"\nCONFIGURATION ERROR: {e}")
            if learning_proc and learning_proc.poll() is None:
                learning_proc.kill()
            sys.exit(1)

    try:
        report = asyncio.run(run_full_benchmark(
            split=args.split,
            collection=args.collection,
            limit=args.limit,
            query_limit=args.query_limit,
            corpus_limit=args.corpus_limit,
            rerank_enabled=not args.no_rerank,
            mode=args.mode,
            recreate_index=args.recreate,
            index_only=args.index_only,
            skip_index=args.skip_index,
        ))
    finally:
        if learning_proc and learning_proc.poll() is None:
            learning_proc.terminate()
            try:
                learning_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                learning_proc.kill()

    if report:
        print_report(report)

        # Show failures if requested
        if args.failures:
            failures = [r for r in report.results if not r.hit_at_10]
            if failures:
                print("\n" + "=" * 70)
                print(f"FAILED QUERIES ({len(failures)} with recall@10 = 0):")
                print("=" * 70)
                for f in failures:
                    print(f"\n[{f.query_id}] {f.query_text[:100]}...")
                    print(f"  Expected: {f.relevant_code_ids}")
                    print(f"  Got top 5: {f.retrieved_code_ids[:5]}")

                # Re-run failed queries with debug output
                if args.debug_failures:
                    print("\n" + "=" * 70)
                    print("DEBUG OUTPUT FOR FAILED QUERIES:")
                    print("=" * 70)
                    debug_top_n = report.config.get("rerank_top_n")
                    for f in failures:
                        asyncio.run(search_cosqa_corpus(
                            query=f.query_text,
                            collection=args.collection,
                            limit=args.limit,
                            rerank_enabled=not args.no_rerank,
                            rerank_top_n=debug_top_n,
                            mode=args.mode,
                            debug=True,
                        ))
            else:
                print("\n✓ No failures - all queries have recall@10 > 0")

        if args.output:
            with open(args.output, "w") as f:
                json.dump(report.to_dict(), f, indent=2)
            print(f"\nSaved report to: {args.output}")
            
            # Save standalone metadata file for audit
            output_dir = str(Path(args.output).parent)
            run_id = f"cosqa_{report.config.get('collection', 'unknown')}_{Path(args.output).stem}"
            meta_path = save_run_meta(
                output_dir,
                run_id,
                report.config,
                extra={"benchmark": "cosqa", "corpus_size": report.corpus_size, "queries": report.total_queries},
            )
            print(f"Saved metadata to: {meta_path}")


if __name__ == "__main__":
    main()
