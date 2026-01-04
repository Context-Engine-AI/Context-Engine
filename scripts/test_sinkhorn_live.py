#!/usr/bin/env python3
"""Live A/B test for Sinkhorn fusion."""
import subprocess
import json
import sys
import os

COLLECTION = "Context-Engine-41e67959"
QUERIES = [
    "authentication middleware",
    "hybrid search implementation",
    "embedding model configuration",
    "qdrant client connection pooling",
    "MCP server tools",
]

def run_search(query: str, sinkhorn: bool) -> list:
    """Run search in subprocess with specific config."""
    env_val = "1" if sinkhorn else "0"
    code = f'''
import json
from scripts.hybrid_search import run_hybrid_search
results = run_hybrid_search(
    queries=[{repr(query)}],
    collection={repr(COLLECTION)},
    limit=10,
    per_path=2,
)
out = []
for r in results[:10]:
    out.append({{"path": r.get("path", "?"), "score": r.get("score", r.get("fusion_score", 0))}})
print(json.dumps(out))
'''
    env = os.environ.copy()
    env["HYBRID_SINKHORN"] = env_val
    env["RERANK_ENABLED"] = "0"
    env["COLLECTION_NAME"] = COLLECTION

    result = subprocess.run(
        ["python3.11", "-c", code],
        capture_output=True,
        text=True,
        env=env,
        cwd="/Users/mirlok/Desktop/Context-Engine",
    )
    # Parse JSON from last line (skip log output)
    if result.returncode != 0:
        print(f"ERROR: {result.stderr}", file=sys.stderr)
        return []
    try:
        lines = result.stdout.strip().split("\n")
        for line in reversed(lines):
            if line.startswith("["):
                return json.loads(line)
        return []
    except Exception as e:
        print(f"Parse error: {e}", file=sys.stderr)
        return []

def main():
    print("=" * 80)
    print("SINKHORN A/B COMPARISON (subprocess isolation)")
    print("=" * 80)

    for query in QUERIES:
        print(f"\nQuery: \"{query}\"")
        print("-" * 60)

        baseline = run_search(query, sinkhorn=False)
        sinkhorn = run_search(query, sinkhorn=True)

        print(f"  Baseline ({len(baseline)}) vs Sinkhorn ({len(sinkhorn)})")

        # Show top 5 side by side
        print("  Rank | Baseline                              | Sinkhorn")
        for i in range(min(5, max(len(baseline), len(sinkhorn)))):
            b_path = baseline[i]["path"][-35:] if i < len(baseline) else "-"
            b_score = baseline[i]["score"] if i < len(baseline) else 0
            s_path = sinkhorn[i]["path"][-35:] if i < len(sinkhorn) else "-"
            s_score = sinkhorn[i]["score"] if i < len(sinkhorn) else 0
            changed = "←" if b_path != s_path else " "
            print(f"  {i+1:4} | {b_score:.3f} ...{b_path:32} | {s_score:.3f} ...{s_path:32} {changed}")

if __name__ == "__main__":
    main()

