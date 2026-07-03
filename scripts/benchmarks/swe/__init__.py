"""
SWE-bench Retrieval Benchmark for Context-Engine.

This module evaluates Context-Engine's ability to retrieve relevant files
given a GitHub issue description - the critical first step in automated
code repair pipelines.

## What is SWE-bench?

SWE-bench (Software Engineering Benchmark) is a collection of 2,294 real
GitHub issues from 12 popular Python repositories with their corresponding
pull request fixes. It's the gold standard for evaluating AI coding agents.

Full SWE-bench task: Issue → Generate working patch → Pass tests
**Retrieval subset**: Issue → Find files that need modification

## Why Retrieval Matters

Most SWE-bench solutions have a retrieval step:
1. Parse issue description
2. **Retrieve relevant code files** ← This is what we evaluate
3. Generate patch based on retrieved context
4. Validate against test suite

If retrieval fails, the entire pipeline fails. Context-Engine's strength
is precisely this retrieval step.

## Context-Engine Features Used

**Search Pipeline (via repo_search):**
- [x] Hybrid search (dense + lexical RRF fusion)
- [x] ONNX reranker (BAAI/bge-reranker-base)
- [x] Query expansion (synonyms + semantic)
- [x] Multi-query fusion (issue → multiple search queries)

**Indexing Pipeline:**
- [x] Real repository indexing (full AST, imports, symbols)
- [x] Cross-file relationships (import graph)
- [x] Semantic chunking
- [x] Git metadata (if available)

**Repo-level features (unique to SWE-bench):**
- [x] File-level recall (did we find the right files?)
- [x] Function-level precision (bonus: did we find the right functions?)
- [x] Cross-file reasoning (changes often span multiple files)

## Datasets

- **SWE-bench Lite**: 300 instances, curated subset (faster evaluation)
- **SWE-bench Full**: 2,294 instances (comprehensive)

## Usage

    # Quick evaluation on Lite subset
    python -m scripts.benchmarks.swe.runner --subset lite --limit 50

    # Full evaluation
    python -m scripts.benchmarks.swe.runner --subset full

    # Single repo evaluation
    python -m scripts.benchmarks.swe.runner --repo django/django --limit 10

## Metrics

- **File Recall@k**: % of ground-truth files found in top-k results
- **File Precision@k**: % of top-k results that are ground-truth files
- **MRR**: Mean Reciprocal Rank of first correct file
- **Pass@k**: % of instances where ALL ground-truth files found in top-k

## References

- SWE-bench paper: https://arxiv.org/abs/2310.06770
- Dataset: https://huggingface.co/datasets/princeton-nlp/SWE-bench
- Leaderboard: https://www.swebench.com/
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scripts.benchmarks.swe.runner import SWEReport

__all__ = [
    "SWEReport",
    "run_swe_benchmark_async",
    "run_swe_benchmark_sync",
]


async def run_swe_benchmark_async(
    subset: str = "lite",
    limit: int | None = None,
    repos: list[str] | None = None,
    top_k: int = 20,
    rerank_enabled: bool = True,
    cache_dir: str | None = None,
) -> "SWEReport":
    """Run SWE-bench retrieval evaluation (async).
    
    Args:
        subset: "lite" (300 instances) or "full" (2,294 instances)
        limit: Limit number of instances for quick testing
        repos: Filter to specific repos (e.g., ["django/django"])
        top_k: Number of files to retrieve per instance
        rerank_enabled: Whether to use reranker
        cache_dir: Directory for cached repos (default: ~/.cache/swe-bench)
    
    Returns:
        SWEReport with all metrics and per-instance results
    """
    from scripts.benchmarks.swe.runner import run_full_benchmark
    return await run_full_benchmark(
        subset=subset,
        limit=limit,
        repos=repos,
        top_k=top_k,
        rerank_enabled=rerank_enabled,
        cache_dir=cache_dir,
    )


def run_swe_benchmark_sync(
    subset: str = "lite",
    limit: int | None = None,
    repos: list[str] | None = None,
    top_k: int = 20,
    rerank_enabled: bool = True,
    cache_dir: str | None = None,
) -> "SWEReport":
    """Run SWE-bench retrieval evaluation (sync wrapper)."""
    import asyncio
    return asyncio.run(run_swe_benchmark_async(
        subset=subset,
        limit=limit,
        repos=repos,
        top_k=top_k,
        rerank_enabled=rerank_enabled,
        cache_dir=cache_dir,
    ))

