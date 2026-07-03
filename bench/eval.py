#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Target:
    collection: str
    model_name: str


@dataclass(frozen=True)
class RunResult:
    seconds: float
    results: List[Dict[str, Any]]


def _parse_json_lines(text: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, dict) and "results" in obj and isinstance(obj.get("results"), list):
            return list(obj.get("results") or [])
        if isinstance(obj, dict):
            out.append(obj)
    return out


def _run_hybrid_search(
    *,
    collection: str,
    model_name: str,
    queries: Sequence[str],
    limit: int,
    per_path: int,
    expand: bool,
    extra_args: Sequence[str],
    env_base: Dict[str, str],
) -> RunResult:
    cmd: List[str] = [
        sys.executable,
        "/app/scripts/hybrid_search.py",
        "--collection",
        collection,
        "--limit",
        str(limit),
        "--per-path",
        str(per_path),
        "--json",
    ]
    if expand:
        cmd.append("--expand")
    for q in queries:
        cmd.extend(["-q", q])
    cmd.extend(list(extra_args))

    env = dict(env_base)
    env["EMBEDDING_MODEL"] = model_name

    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    dt = time.time() - t0
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or f"hybrid_search failed: rc={proc.returncode}")
    return RunResult(seconds=dt, results=_parse_json_lines(proc.stdout))


def _topk_paths(results: Sequence[Dict[str, Any]], k: int) -> List[str]:
    out: List[str] = []
    for r in results:
        p = r.get("path")
        if isinstance(p, str) and p:
            out.append(p)
        if len(out) >= k:
            break
    return out


def _percentile(xs: Sequence[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    i = (len(s) - 1) * p
    lo = int(i)
    hi = min(lo + 1, len(s) - 1)
    frac = i - lo
    return s[lo] * (1.0 - frac) + s[hi] * frac


def _jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    sa = set(a)
    sb = set(b)
    if not sa and not sb:
        return 1.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return (inter / union) if union else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate hybrid_search latency + top-k overlap across collections/models")
    ap.add_argument(
        "--target",
        action="append",
        nargs=2,
        metavar=("COLLECTION", "MODEL"),
        required=True,
        help="Pair: collection name and EMBEDDING_MODEL to use for query embedding",
    )
    ap.add_argument("--query", action="append", default=[])
    ap.add_argument("--query-file", type=str, default=None)
    ap.add_argument("--repeats", type=int, default=int(os.environ.get("BENCH_REPEATS", "10") or 10))
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--per-path", type=int, default=1)
    ap.add_argument("--expand", action="store_true", default=False)
    ap.add_argument("--json-out", type=str, default=None)

    args, extra = ap.parse_known_args()
    if extra:
        if extra[:1] != ["--"]:
            raise SystemExit(
                "Unknown args: "
                + " ".join(extra)
                + "\nIf you intended to pass flags through to hybrid_search.py, put them after '--'."
            )
        extra = extra[1:]

    targets = [Target(collection=c, model_name=m) for c, m in (args.target or [])]
    if len(targets) < 2:
        raise SystemExit("need at least two --target pairs")

    queries = [q for q in (args.query or []) if str(q).strip()]
    if args.query_file:
        try:
            with open(args.query_file, "r", encoding="utf-8") as f:
                for raw in f.read().splitlines():
                    line = (raw or "").strip()
                    if not line:
                        continue
                    queries.append(line)
        except Exception as e:
            raise SystemExit(f"Failed to read --query-file {args.query_file}: {e}")
    if not queries:
        queries = [
            "how does repo_search combine dense and lexical",
            "where is sanitize_vector_name defined",
            "how is multi repo collection name derived",
            "where is auth and acl, where is it enforced?",
            "How does the CTX system work? How does extension integrate with it?"
        ]

    print(f"queries={len(queries)}", flush=True)
    for i, q in enumerate(queries):
        print(f"q{i}: {q}", flush=True)

    env_base = dict(os.environ)

    for t in targets:
        for _ in range(max(0, int(args.warmup))):
            _run_hybrid_search(
                collection=t.collection,
                model_name=t.model_name,
                queries=queries,
                limit=args.limit,
                per_path=args.per_path,
                expand=args.expand,
                extra_args=extra,
                env_base=env_base,
            )

    summary: Dict[str, Any] = {
        "queries": list(queries),
        "repeats": int(args.repeats),
        "warmup": int(args.warmup),
        "limit": int(args.limit),
        "topk": int(args.topk),
        "per_path": int(args.per_path),
        "expand": bool(args.expand),
        "hybrid_search_args": list(extra),
        "targets": [{"collection": t.collection, "model": t.model_name} for t in targets],
        "results": {},
        "overlap": {},
    }

    per_target_runs: Dict[str, List[RunResult]] = {}

    for t in targets:
        runs: List[RunResult] = []
        for i in range(int(args.repeats)):
            r = _run_hybrid_search(
                collection=t.collection,
                model_name=t.model_name,
                queries=queries,
                limit=args.limit,
                per_path=args.per_path,
                expand=args.expand,
                extra_args=extra,
                env_base=env_base,
            )
            runs.append(r)
            print(f"{t.collection} run{i}: {r.seconds:.3f}s", flush=True)
        per_target_runs[t.collection] = runs

        times = [x.seconds for x in runs]
        rec = {
            "n": len(times),
            "min": min(times) if times else 0.0,
            "max": max(times) if times else 0.0,
            "p50": statistics.median(times) if times else 0.0,
            "p95": _percentile(times, 0.95) if times else 0.0,
        }
        summary["results"][t.collection] = rec

    base = targets[0].collection
    for other in [t.collection for t in targets[1:]]:
        overlaps: List[float] = []
        for i in range(min(len(per_target_runs[base]), len(per_target_runs[other]))):
            a = _topk_paths(per_target_runs[base][i].results, args.topk)
            b = _topk_paths(per_target_runs[other][i].results, args.topk)
            overlaps.append(_jaccard(a, b))
        summary["overlap"][f"{base}__vs__{other}"] = {
            "n": len(overlaps),
            "p50": statistics.median(overlaps) if overlaps else 0.0,
            "p95": _percentile(overlaps, 0.95) if overlaps else 0.0,
            "min": min(overlaps) if overlaps else 0.0,
            "max": max(overlaps) if overlaps else 0.0,
        }

    print(json.dumps(summary, ensure_ascii=False))

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            f.write(json.dumps(summary, ensure_ascii=False, indent=2))
            f.write("\n")


if __name__ == "__main__":
    main()
