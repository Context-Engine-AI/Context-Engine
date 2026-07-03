"""
CoSQA Benchmark for Context-Engine Code Search Evaluation.

This module provides a comprehensive evaluation framework using the CoSQA
(Code Search and Question Answering) benchmark dataset for measuring
code search quality against academic baselines.

Key Components:
- dataset: Download and process CoSQA dataset from HuggingFace
- indexer: Index CoSQA code corpus into Qdrant
- runner: Evaluate search quality with MRR, NDCG, Recall metrics

Usage:
    # Full benchmark (download, index, evaluate)
    python -m scripts.benchmarks.cosqa.runner --limit 10

    # Quick test with limited queries
    python -m scripts.benchmarks.cosqa.runner --query-limit 100

    # Compare with/without reranker
    python -m scripts.benchmarks.cosqa.runner --output with_rerank.json
    python -m scripts.benchmarks.cosqa.runner --no-rerank --output without_rerank.json

Programmatic Usage:
    from scripts.benchmarks.cosqa import run_cosqa_benchmark_async
    
    report = await run_cosqa_benchmark_async()
    print(f"MRR: {report.mrr:.4f}")

Integration with run_all.py:
    The run_cosqa_benchmark() function returns a CoSQAReport that can be
    serialized via .to_dict() for unified benchmark reporting.

References:
- CoSQA Paper: https://arxiv.org/abs/2105.13239
- Dataset: https://huggingface.co/datasets/gonglinyuan/CoSQA

Paper Baselines (MRR):
- CodeBERT: 0.428
- GraphCodeBERT: 0.445
- CodeBERT-biencoder: 0.483
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional, TYPE_CHECKING

# Re-export main classes and functions
from scripts.benchmarks.cosqa.dataset import (
    CoSQADataset,
    CoSQACorpusEntry,
    CoSQAQuery,
    load_cosqa,
    get_corpus_for_indexing,
    get_queries_for_evaluation,
)

# Keep a stable default without importing indexer at module import time.
DEFAULT_COLLECTION = "cosqa-corpus"

if TYPE_CHECKING:
    from scripts.benchmarks.cosqa.runner import CoSQAReport, CoSQAQueryResult

__all__ = [
    # Dataset
    "CoSQADataset",
    "CoSQACorpusEntry", 
    "CoSQAQuery",
    "load_cosqa",
    "get_corpus_for_indexing",
    "get_queries_for_evaluation",
    # Indexer
    "index_corpus",
    "verify_collection",
    "DEFAULT_COLLECTION",
    # Runner
    "CoSQAReport",
    "CoSQAQueryResult",
    "run_cosqa_benchmark",
    "run_full_benchmark",
    "print_report",
    "PAPER_BASELINES",
    # Convenience
    "run_cosqa_benchmark_async",
    "run_cosqa_benchmark_sync",
]

def __getattr__(name: str) -> Any:  # PEP 562
    # Lazy-load indexer/runner so importing the package has no side effects.
    # This prevents env-based config (e.g., LEX_SPARSE_MODE) from getting frozen
    # before `scripts.benchmarks.cosqa.runner` loads .env when running via `-m`.
    if name in {"index_corpus", "verify_collection"}:
        from scripts.benchmarks.cosqa import indexer as _indexer
        return getattr(_indexer, name)
    if name in {
        "CoSQAReport",
        "CoSQAQueryResult",
        "run_cosqa_benchmark",
        "run_full_benchmark",
        "print_report",
        "PAPER_BASELINES",
    }:
        from scripts.benchmarks.cosqa import runner as _runner
        return getattr(_runner, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


async def run_cosqa_benchmark_async(
    split: str = "test",
    collection: str = DEFAULT_COLLECTION,
    limit: int = 10,
    query_limit: Optional[int] = None,
    rerank_enabled: bool = True,
    recreate_index: bool = False,
) -> "CoSQAReport":
    """Run CoSQA benchmark (async).
    
    This is the main entry point for programmatic use.
    
    Args:
        split: Dataset split ("train", "validation", "test")
        collection: Qdrant collection name
        limit: Max results per query
        query_limit: Limit queries for quick testing
        rerank_enabled: Whether to use reranker
        recreate_index: Recreate index from scratch
    
    Returns:
        CoSQAReport with all metrics and per-query results
    """
    from scripts.benchmarks.cosqa.runner import run_full_benchmark
    return await run_full_benchmark(
        split=split,
        collection=collection,
        limit=limit,
        query_limit=query_limit,
        rerank_enabled=rerank_enabled,
        recreate_index=recreate_index,
    )


def run_cosqa_benchmark_sync(
    split: str = "test",
    collection: str = DEFAULT_COLLECTION,
    limit: int = 10,
    query_limit: Optional[int] = None,
    rerank_enabled: bool = True,
    recreate_index: bool = False,
) -> "CoSQAReport":
    """Run CoSQA benchmark (sync wrapper).
    
    Convenience wrapper for non-async contexts.
    """
    return asyncio.run(run_cosqa_benchmark_async(
        split=split,
        collection=collection,
        limit=limit,
        query_limit=query_limit,
        rerank_enabled=rerank_enabled,
        recreate_index=recreate_index,
    ))
