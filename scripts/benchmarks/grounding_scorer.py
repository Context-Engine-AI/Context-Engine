#!/usr/bin/env python3
"""
Grounding Accuracy Scorer for context_answer.

Measures the quality and trustworthiness of LLM-generated answers:
- Citation accuracy: Do citations point to correct/relevant files?
- Grounding rate: % of answers with sufficient context
- Hallucination detection: Claims not supported by retrieved docs
- Token efficiency: Useful tokens / total tokens
"""

import asyncio
import json
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

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

# Ensure correct collection is used (read from workspace state or env)
try:
    from scripts.benchmarks.common import resolve_collection_auto
    if not os.environ.get("COLLECTION_NAME"):
        from scripts.workspace_state import get_collection_name
        os.environ["COLLECTION_NAME"] = get_collection_name() or "codebase"
    else:
        os.environ["COLLECTION_NAME"] = resolve_collection_auto(os.environ.get("COLLECTION_NAME"))
except Exception:
    # Best-effort; grounding scorer can still run with whatever collection is configured.
    pass

print(
    f"[bench] Using QDRANT_URL={os.environ.get('QDRANT_URL', '')} "
    f"COLLECTION_NAME={os.environ.get('COLLECTION_NAME', '')}"
)

# Timeout for each query (seconds)
QUERY_TIMEOUT = 60


@dataclass
class GroundingResult:
    """Result of grounding analysis for a single answer."""
    query: str
    answer: str
    citations: List[Dict[str, Any]]
    citation_accuracy: float  # % of citations that are relevant
    topic_coverage: float  # % of expected topics mentioned in answer
    is_grounded: bool  # Has sufficient context
    has_hedging: bool  # Contains uncertainty language
    code_references: int  # Number of code blocks/references
    answer_length: int  # Token count estimate
    useful_ratio: float  # Useful content ratio


@dataclass
class GroundingReport:
    """Aggregate grounding quality report."""
    timestamp: str
    total_queries: int
    grounding_rate: float  # % with sufficient context
    avg_citation_accuracy: float
    avg_topic_coverage: float  # % of expected topics covered on average
    avg_citations_per_answer: float
    hedging_rate: float  # % with uncertainty language
    avg_answer_length: int
    avg_useful_ratio: float
    results: List[GroundingResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "total_queries": self.total_queries,
            "metrics": {
                "grounding_rate": self.grounding_rate,
                "avg_citation_accuracy": self.avg_citation_accuracy,
                "avg_topic_coverage": self.avg_topic_coverage,
                "avg_citations_per_answer": self.avg_citations_per_answer,
                "hedging_rate": self.hedging_rate,
                "avg_answer_length": self.avg_answer_length,
                "avg_useful_ratio": self.avg_useful_ratio,
            },
            "results": [r.__dict__ for r in self.results],
        }


# Test queries for grounding evaluation
GROUNDING_TEST_QUERIES = [
    {
        "query": "How does the hybrid search combine dense and lexical results?",
        "expected_topics": ["RRF", "dense", "lexical", "ranking", "score"],
        "expected_files": ["hybrid_search.py", "ranking.py"],
    },
    {
        "query": "What is the purpose of the recursive reranker's latent state z?",
        "expected_topics": ["latent", "state", "refinement", "TRM", "iteration"],
        "expected_files": ["rerank_recursive", "core.py"],
    },
    {
        "query": "How does context_answer handle insufficient context?",
        "expected_topics": ["grounding", "citations", "context", "insufficient"],
        "expected_files": ["context_answer.py", "refrag.py"],
    },
    {
        "query": "What embedding model does the system use and why?",
        "expected_topics": ["embedding", "model", "BAAI", "bge", "dimension"],
        "expected_files": ["embedder.py", ".env"],
    },
    {
        "query": "Explain the MICRO_BUDGET_TOKENS parameter and its effect",
        "expected_topics": ["budget", "tokens", "span", "context", "limit"],
        "expected_files": ["refrag.py", ".env"],
    },
]


def estimate_tokens(text: str) -> int:
    """Rough token estimate (4 chars per token average)."""
    return len(text) // 4


def detect_hedging(text: str) -> bool:
    """Detect uncertainty/hedging language in answer."""
    hedging_phrases = [
        "i'm not sure",
        "i couldn't find",
        "insufficient context",
        "may be",
        "might be",
        "possibly",
        "it seems",
        "appears to",
        "i don't have enough",
        "unable to determine",
        "without more context",
    ]
    text_lower = text.lower()
    return any(phrase in text_lower for phrase in hedging_phrases)


def calculate_citation_accuracy(
    citations: List[Dict[str, Any]],
    expected_files: List[str],
) -> float:
    """Calculate what % of citations are relevant to expected files."""
    if not citations:
        return 0.0
    
    relevant = 0
    for citation in citations:
        path = citation.get("path", "")
        # Check if any expected file is in the citation path
        for expected in expected_files:
            if expected in path:
                relevant += 1
                break
    
    return relevant / len(citations)


def count_code_references(text: str) -> int:
    """Count code blocks and inline code references."""
    # Count markdown code blocks
    code_blocks = len(re.findall(r'```[\s\S]*?```', text))
    # Count inline code
    inline_code = len(re.findall(r'`[^`]+`', text))
    return code_blocks + inline_code


def calculate_topic_coverage(answer: str, expected_topics: List[str]) -> float:
    """Calculate what % of expected topics are mentioned in the answer."""
    if not expected_topics:
        return 1.0  # No expected topics = full coverage by default

    if not answer:
        return 0.0

    answer_lower = answer.lower()
    covered = 0

    for topic in expected_topics:
        # Check if topic (case-insensitive) appears in the answer
        if topic.lower() in answer_lower:
            covered += 1

    return covered / len(expected_topics)


def calculate_useful_ratio(answer: str, citations: List[Dict]) -> float:
    """
    Estimate the ratio of useful content.
    
    Useful = specific facts, code, file references
    Not useful = filler, hedging, generic statements
    """
    if not answer:
        return 0.0
    
    lines = answer.split('\n')
    useful_lines = 0
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Consider useful if contains:
        is_useful = (
            '`' in line  # Code reference
            or 'file' in line.lower() and ('/' in line or '.' in line)  # File reference
            or any(c.isupper() for c in line) and '_' in line  # CONSTANT or snake_case
            or re.search(r'\d+', line)  # Contains numbers (specific)
            or len(line.split()) > 10  # Detailed explanation
        )
        
        # Penalize if hedging
        is_hedging = detect_hedging(line)
        
        if is_useful and not is_hedging:
            useful_lines += 1
    
    return useful_lines / len(lines) if lines else 0


async def evaluate_single_query(
    query: str,
    expected_topics: List[str],
    expected_files: List[str],
) -> GroundingResult:
    """Evaluate grounding for a single query with timeout."""
    
    async def run_query():
        # Import the context_answer function from mcp_indexer_server
        from scripts.mcp_indexer_server import context_answer
        
        # Execute context_answer
        result = await context_answer(query=query)
        return result
    
    try:
        # Run with timeout to prevent hanging
        result = await asyncio.wait_for(run_query(), timeout=QUERY_TIMEOUT)
        
        answer = result.get("answer", "")
        citations = result.get("citations", [])
        # context_answer doesn't return an explicit "grounded" field,
        # so we detect it heuristically from the answer text
        is_grounded = "insufficient" not in answer.lower()
        
    except asyncio.TimeoutError:
        print(f"  ⏱️ Timeout after {QUERY_TIMEOUT}s")
        return GroundingResult(
            query=query,
            answer=f"Timeout after {QUERY_TIMEOUT}s",
            citations=[],
            citation_accuracy=0,
            is_grounded=False,
            has_hedging=True,
            code_references=0,
            answer_length=0,
            useful_ratio=0,
            topic_coverage=0,
        )
    except Exception as e:
        print(f"  Error: {e}")
        return GroundingResult(
            query=query,
            answer=f"Error: {e}",
            citations=[],
            citation_accuracy=0,
            topic_coverage=0,
            is_grounded=False,
            has_hedging=True,
            code_references=0,
            answer_length=0,
            useful_ratio=0,
        )

    # Calculate metrics
    citation_accuracy = calculate_citation_accuracy(citations, expected_files)
    topic_coverage = calculate_topic_coverage(answer, expected_topics)
    has_hedging = detect_hedging(answer)
    code_refs = count_code_references(answer)
    answer_length = estimate_tokens(answer)
    useful_ratio = calculate_useful_ratio(answer, citations)

    return GroundingResult(
        query=query,
        answer=answer[:500] + "..." if len(answer) > 500 else answer,
        citations=citations,
        citation_accuracy=citation_accuracy,
        topic_coverage=topic_coverage,
        is_grounded=is_grounded and not has_hedging,
        has_hedging=has_hedging,
        code_references=code_refs,
        answer_length=answer_length,
        useful_ratio=useful_ratio,
    )


async def run_grounding_benchmark() -> GroundingReport:
    """Run the full grounding quality benchmark."""
    print("=" * 70)
    print("GROUNDING ACCURACY BENCHMARK")
    print("=" * 70)
    
    results = []
    
    print(f"\nEvaluating {len(GROUNDING_TEST_QUERIES)} queries...\n")
    
    for i, test in enumerate(GROUNDING_TEST_QUERIES, 1):
        query = test["query"]
        print(f"  [{i}/{len(GROUNDING_TEST_QUERIES)}] {query[:50]}...", end=" ")
        
        result = await evaluate_single_query(
            query=query,
            expected_topics=test["expected_topics"],
            expected_files=test["expected_files"],
        )
        
        status = "✓" if result.is_grounded else "○"
        print(f"{status} ({len(result.citations)} citations, {result.citation_accuracy:.0%} accurate)")
        
        results.append(result)
    
    # Calculate aggregate metrics
    grounding_rate = sum(1 for r in results if r.is_grounded) / len(results)
    avg_citation_accuracy = sum(r.citation_accuracy for r in results) / len(results)
    avg_topic_coverage = sum(r.topic_coverage for r in results) / len(results)
    avg_citations = sum(len(r.citations) for r in results) / len(results)
    hedging_rate = sum(1 for r in results if r.has_hedging) / len(results)
    avg_length = sum(r.answer_length for r in results) // len(results)
    avg_useful = sum(r.useful_ratio for r in results) / len(results)

    report = GroundingReport(
        timestamp=datetime.now().isoformat(),
        total_queries=len(results),
        grounding_rate=grounding_rate,
        avg_citation_accuracy=avg_citation_accuracy,
        avg_topic_coverage=avg_topic_coverage,
        avg_citations_per_answer=avg_citations,
        hedging_rate=hedging_rate,
        avg_answer_length=avg_length,
        avg_useful_ratio=avg_useful,
        results=results,
    )
    
    return report


def print_grounding_report(report: GroundingReport):
    """Print the grounding quality report."""
    print("\n" + "=" * 70)
    print("GROUNDING QUALITY REPORT")
    print("=" * 70)
    
    print(f"\nTimestamp: {report.timestamp}")
    print(f"Queries evaluated: {report.total_queries}")
    
    print("\n" + "-" * 70)
    print("TRUST METRICS:")
    
    # Grounding rate
    emoji = "✅" if report.grounding_rate >= 0.8 else "⚠️" if report.grounding_rate >= 0.6 else "❌"
    print(f"  {emoji} Grounding Rate:      {report.grounding_rate:.0%}")
    
    # Citation accuracy
    emoji = "✅" if report.avg_citation_accuracy >= 0.8 else "⚠️" if report.avg_citation_accuracy >= 0.5 else "❌"
    print(f"  {emoji} Citation Accuracy:   {report.avg_citation_accuracy:.0%}")

    # Topic coverage
    emoji = "✅" if report.avg_topic_coverage >= 0.7 else "⚠️" if report.avg_topic_coverage >= 0.4 else "❌"
    print(f"  {emoji} Topic Coverage:      {report.avg_topic_coverage:.0%}")

    # Hedging rate (lower is better)
    emoji = "✅" if report.hedging_rate <= 0.2 else "⚠️" if report.hedging_rate <= 0.4 else "❌"
    print(f"  {emoji} Hedging Rate:        {report.hedging_rate:.0%} (lower is better)")
    
    # Useful ratio
    emoji = "✅" if report.avg_useful_ratio >= 0.6 else "⚠️" if report.avg_useful_ratio >= 0.4 else "❌"
    print(f"  {emoji} Useful Content:      {report.avg_useful_ratio:.0%}")
    
    print("\n" + "-" * 70)
    print("ANSWER STATISTICS:")
    print(f"  Avg citations/answer:  {report.avg_citations_per_answer:.1f}")
    print(f"  Avg answer length:     ~{report.avg_answer_length} tokens")
    
    print("\n" + "-" * 70)
    print("PER-QUERY RESULTS:")
    print(f"{'Query':<45} {'Grounded':>10} {'Citations':>10} {'Accuracy':>10}")
    print("-" * 75)
    
    for r in report.results:
        grounded = "✓" if r.is_grounded else "○"
        query_short = r.query[:42] + "..." if len(r.query) > 45 else r.query
        print(f"{query_short:<45} {grounded:>10} {len(r.citations):>10} {r.citation_accuracy:>9.0%}")
    
    # Overall verdict
    print("\n" + "-" * 70)
    if report.grounding_rate >= 0.8 and report.avg_citation_accuracy >= 0.7:
        print("🏆 VERDICT: HIGH QUALITY - Answers are well-grounded and trustworthy")
    elif report.grounding_rate >= 0.6:
        print("⚠️  VERDICT: MODERATE QUALITY - Some answers lack sufficient grounding")
    else:
        print("❌ VERDICT: NEEDS IMPROVEMENT - Many answers are poorly grounded")
    
    print("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Grounding Accuracy Scorer")
    parser.add_argument("--output", "-o", help="Output JSON file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show full answers")
    args = parser.parse_args()
    
    report = await run_grounding_benchmark()
    print_grounding_report(report)
    
    if args.verbose:
        print("\n" + "=" * 70)
        print("FULL ANSWERS:")
        for r in report.results:
            print(f"\n--- {r.query} ---")
            print(r.answer)
            print(f"Citations: {r.citations}")
    
    if args.output:
        output = {
            "timestamp": report.timestamp,
            "grounding_rate": report.grounding_rate,
            "citation_accuracy": report.avg_citation_accuracy,
            "hedging_rate": report.hedging_rate,
            "useful_ratio": report.avg_useful_ratio,
            "queries": len(report.results),
        }
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
