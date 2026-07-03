#!/usr/bin/env python3
"""
ReFRAG Benchmark for Context-Engine

Measures retrieval quality, token budgeting efficiency, and micro-chunking effectiveness.
"""

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List
import statistics

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Load environment (optional) and fix Docker hostname
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass
if "qdrant:" in os.environ.get("QDRANT_URL", ""):
    os.environ["QDRANT_URL"] = "http://localhost:6333"

from scripts.benchmarks.common import percentile, resolve_collection_auto

# Ensure correct collection is used (read from workspace state or env)
if not os.environ.get("COLLECTION_NAME"):
    try:
        from scripts.workspace_state import get_collection_name
        os.environ["COLLECTION_NAME"] = get_collection_name() or "codebase"
    except Exception:
        os.environ["COLLECTION_NAME"] = "codebase"
else:
    # If COLLECTION_NAME is set but empty/unindexed, pick a non-empty collection for benchmarks.
    try:
        os.environ["COLLECTION_NAME"] = resolve_collection_auto(os.environ.get("COLLECTION_NAME"))
    except Exception:
        pass

print(
    f"[bench] Using QDRANT_URL={os.environ.get('QDRANT_URL', '')} "
    f"COLLECTION_NAME={os.environ.get('COLLECTION_NAME', '')}"
)


@dataclass
class RefragResult:
    """Result from ReFRAG evaluation."""
    query: str
    answer_length: int
    citation_count: int
    tokens_used: int
    latency_ms: float
    grounded: bool


@dataclass
class RefragReport:
    """ReFRAG benchmark report."""
    name: str
    total_queries: int
    avg_citations: float
    avg_tokens: float
    grounding_rate: float
    avg_latency_ms: float
    p90_latency_ms: float
    results: List[RefragResult] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "total_queries": self.total_queries,
            "metrics": {
                "avg_citations": round(self.avg_citations, 2),
                "avg_tokens": round(self.avg_tokens, 0),
                "grounding_rate": round(self.grounding_rate, 4),
                "avg_latency_ms": round(self.avg_latency_ms, 2),
                "p90_latency_ms": round(self.p90_latency_ms, 2),
            },
            "results": [asdict(r) for r in self.results],
        }


REFRAG_QUERIES = [
    "How does the reranking learning system work?",
    "What is the purpose of the workspace state module?",
    "Explain the hybrid search RRF algorithm.",
    "How does memory storage use vectors?",
    "What does the openlit init module do?",
]


async def run_refrag_benchmark(name: str = "default") -> RefragReport:
    """Run ReFRAG benchmark."""
    try:
        from scripts.mcp_indexer_server import context_answer
    except ImportError as e:
        print(f"Import error: {e}")
        return RefragReport(name=name, total_queries=0, avg_citations=0,
                           avg_tokens=0, grounding_rate=0, avg_latency_ms=0, p90_latency_ms=0)
    
    results: List[RefragResult] = []
    latencies: List[float] = []
    
    for query in REFRAG_QUERIES:
        print(f"  Testing: {query[:50]}...")
        
        start = time.perf_counter()
        try:
            result = await context_answer(query=query)
            success = True
        except Exception as e:
            print(f"    Error: {e}")
            result = {}
            success = False
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)
        
        # Extract metrics
        answer = result.get("answer", "") if isinstance(result, dict) else ""
        citations = result.get("citations", []) if isinstance(result, dict) else []
        
        # Estimate tokens (rough)
        tokens = len(answer.split()) * 1.3
        
        grounded = len(citations) > 0 and "insufficient" not in answer.lower()
        
        results.append(RefragResult(
            query=query,
            answer_length=len(answer),
            citation_count=len(citations),
            tokens_used=int(tokens),
            latency_ms=elapsed_ms,
            grounded=grounded,
        ))
        print(f"    citations={len(citations)}, grounded={grounded}, {elapsed_ms:.0f}ms")
    
    return RefragReport(
        name=name,
        total_queries=len(results),
        avg_citations=statistics.mean(r.citation_count for r in results) if results else 0,
        avg_tokens=statistics.mean(r.tokens_used for r in results) if results else 0,
        grounding_rate=sum(1 for r in results if r.grounded) / len(results) if results else 0,
        avg_latency_ms=statistics.mean(latencies) if latencies else 0,
        p90_latency_ms=percentile(latencies, 0.90),
        results=results,
    )


def print_report(report: RefragReport):
    print("\n" + "=" * 60)
    print(f"REFRAG BENCHMARK: {report.name}")
    print("=" * 60)
    print(f"Queries: {report.total_queries}")
    print(f"Avg Citations: {report.avg_citations:.2f}")
    print(f"Avg Tokens: {report.avg_tokens:.0f}")
    print(f"Grounding Rate: {report.grounding_rate:.1%}")
    print(f"Avg Latency: {report.avg_latency_ms:.2f}ms")
    print(f"P90 Latency: {report.p90_latency_ms:.2f}ms")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="ReFRAG Benchmark")
    parser.add_argument("--name", default="default", help="Benchmark name")
    parser.add_argument("--output", type=str, help="Output JSON file")
    args = parser.parse_args()
    
    print(f"Running ReFRAG benchmark: {args.name}")
    report = asyncio.run(run_refrag_benchmark(name=args.name))
    print_report(report)
    
    if args.output:
        with open(args.output, "w") as f:
            json.dump(report.to_dict(), f, indent=2)


if __name__ == "__main__":
    main()
