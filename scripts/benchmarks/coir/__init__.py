"""
CoIR (Code Information Retrieval) Benchmark for Context-Engine.

CoIR is a comprehensive benchmark suite for evaluating code retrieval systems
across multiple datasets and programming languages.

Included Tasks:
- cosqa: Python code search (web queries)
- codesearchnet-*: Multi-language code search (Python, Java, JS, Go, Ruby, PHP)
- stackoverflow-qa: StackOverflow code Q&A
- apps: Code generation retrieval
- codetrans-*: Code translation retrieval

Installation:
    pip install coir-eval

Usage:
    # Run all default tasks
    python -m scripts.benchmarks.coir.runner

    # Run specific tasks
    python -m scripts.benchmarks.coir.runner --tasks cosqa codesearchnet-python

    # Quick test with subset
    python -m scripts.benchmarks.coir.runner --tasks cosqa --query-limit 100 --corpus-limit 100

Programmatic Usage:
    from scripts.benchmarks.coir import run_coir_benchmark
    
    report = await run_coir_benchmark(tasks=["cosqa"])
    print(report.to_dict())

References:
- GitHub: https://github.com/CoIR-team/coir
- Paper: https://arxiv.org/abs/2407.02883
"""
from __future__ import annotations

# Available tasks in CoIR
COIR_TASKS = [
    # Code Search
    "cosqa",
    "codesearchnet-python",
    "codesearchnet-java", 
    "codesearchnet-javascript",
    "codesearchnet-go",
    "codesearchnet-ruby",
    "codesearchnet-php",
    # Q&A
    "stackoverflow-qa",
    # Apps
    "apps",
    # Code Translation
    "codetrans-dl",
    "codetrans-contest",
]

# Default tasks for quick evaluation
DEFAULT_TASKS = ["cosqa", "codesearchnet-python"]

__all__ = [
    "COIR_TASKS",
    "DEFAULT_TASKS",
    "ContextEngineRetriever",
    "run_coir_benchmark",
    "run_coir_benchmark_sync",
    "CoIRReport",
]

# Lazy imports to avoid loading heavy dependencies at module import
def __getattr__(name):
    if name == "ContextEngineRetriever":
        from scripts.benchmarks.coir.retriever import ContextEngineRetriever
        return ContextEngineRetriever
    elif name == "run_coir_benchmark":
        from scripts.benchmarks.coir.runner import run_coir_benchmark
        return run_coir_benchmark
    elif name == "run_coir_benchmark_sync":
        from scripts.benchmarks.coir.runner import run_coir_benchmark_sync
        return run_coir_benchmark_sync
    elif name == "CoIRReport":
        from scripts.benchmarks.coir.runner import CoIRReport
        return CoIRReport
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
