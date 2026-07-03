#!/usr/bin/env python3
import os
import argparse
import sys
from collections import defaultdict
from typing import List, Dict, Any
from pathlib import Path

from qdrant_client import QdrantClient, models
import re

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.utils import sanitize_vector_name

# Use embedder factory for Qwen3 support; fallback to direct fastembed
try:
    from scripts.embedder import get_embedding_model as _get_embedding_model
    _EMBEDDER_FACTORY = True
except ImportError:
    _EMBEDDER_FACTORY = False
    from fastembed import TextEmbedding


# Env configuration
QDRANT_URL = os.environ.get("QDRANT_URL", "http://qdrant:6333")
COLLECTION = os.environ.get("COLLECTION_NAME", "codebase")
MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")

# Quick-win boosts
SYMBOL_BOOST = float(os.environ.get("HYBRID_SYMBOL_BOOST", "0.15") or 0.15)
RECENCY_WEIGHT = float(os.environ.get("HYBRID_RECENCY_WEIGHT", "0.1") or 0.1)

# Minimal code-aware query expansion
CODE_SYNONYMS = {
    "function": ["method", "def", "fn"],
    "class": ["type", "object"],
    "create": ["init", "initialize", "construct"],
}


def expand_queries(
    queries: List[str], language: str | None = None, max_extra: int = 2
) -> List[str]:
    out: List[str] = list(queries)
    for q in list(queries):
        ql = q.lower()
        for word, syns in CODE_SYNONYMS.items():
            if word in ql:
                for s in syns[:max_extra]:
                    exp = re.sub(rf"\b{re.escape(word)}\b", s, q, flags=re.IGNORECASE)
                    if exp not in out:
                        out.append(exp)
    return out[: max(8, len(queries))]


def _env_truthy(val: str | None, default: bool) -> bool:
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def derive_vector_name(model_name: str) -> str:
    return sanitize_vector_name(model_name)


def score_candidate(
    base_score: float, count: int, md: Dict[str, Any], want_lang: str, want_prefix: str
) -> float:
    s = base_score
    # Encourage consensus across multiple phrasings
    s += 0.02 * max(0, count - 1)
    # Boosts for metadata matches
    lang = (md or {}).get("language") or ""
    if want_lang and lang.lower() == want_lang.lower():
        s += 0.03
    prefix = (md or {}).get("path_prefix") or ""
    if want_prefix and str(prefix).startswith(want_prefix):
        s += 0.03
    return s


def main():
    parser = argparse.ArgumentParser(
        description="Multi-query re-ranker for Qdrant code search (no new deps)"
    )
    parser.add_argument(
        "--query",
        "-q",
        action="append",
        required=True,
        help="Query text; repeat flag to add variants",
    )
    parser.add_argument(
        "--limit", type=int, default=8, help="Final top-N to print after re-ranking"
    )
    parser.add_argument(
        "--per-query",
        type=int,
        default=12,
        help="How many results to pull per query before fusing",
    )
    parser.add_argument(
        "--language", type=str, default="", help="Preferred language (boost)"
    )
    parser.add_argument(
        "--under",
        type=str,
        default="",
        help="Preferred path_prefix (boost), e.g. /work/scripts",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="",
        help="Optional server-side symbol filter (exact)",
    )
    # Expansion enabled by default; allow disabling via --no-expand or RERANK_EXPAND=0
    parser.add_argument(
        "--expand",
        dest="expand",
        action="store_true",
        default=_env_truthy(os.environ.get("RERANK_EXPAND"), True),
        help="Enable simple query expansion",
    )
    parser.add_argument(
        "--no-expand",
        dest="expand",
        action="store_false",
        help="Disable query expansion",
    )

    args = parser.parse_args()

    client = QdrantClient(url=QDRANT_URL)
    if _EMBEDDER_FACTORY:
        model = _get_embedding_model(MODEL)
    else:
        model = TextEmbedding(model_name=MODEL)
    vec_name = derive_vector_name(MODEL)

    # Build query list (optionally expanded)
    queries = list(args.query)
    if args.expand:
        queries = expand_queries(queries, args.language or None)

    cand: Dict[str, Dict[str, Any]] = {}
    counts: Dict[str, int] = defaultdict(int)

    for q in queries:
        v = next(model.embed([q]))
        # Optional server-side filter to reduce bandwidth
        flt = None
        if args.language or args.under or args.symbol:
            must = []
            if args.language:
                must.append(
                    models.FieldCondition(
                        key="metadata.language",
                        match=models.MatchValue(value=args.language),
                    )
                )
            if args.under:
                must.append(
                    models.FieldCondition(
                        key="metadata.path_prefix",
                        match=models.MatchValue(value=args.under),
                    )
                )
            if args.symbol:
                must.append(
                    models.FieldCondition(
                        key="metadata.symbol",
                        match=models.MatchValue(value=args.symbol),
                    )
                )
            flt = models.Filter(must=must)

        # Prefer modern query_points API with 'using' for named vector
        try:
            qp = client.query_points(
                collection_name=COLLECTION,
                query=v.tolist(),
                using=vec_name,
                query_filter=flt,
                search_params=models.SearchParams(hnsw_ef=128),
                limit=args.per_query,
                with_payload=True,
            )
            res_points = getattr(qp, "points", qp)
        except Exception:
            # Fallback to deprecated search API
            res_points = client.search(
                collection_name=COLLECTION,
                query_vector={"name": vec_name, "vector": v.tolist()},
                limit=args.per_query,
                with_payload=True,
                query_filter=flt,
            )
        for p in res_points:
            pid = str(p.id)
            counts[pid] += 1
            if pid not in cand or p.score > cand[pid]["base_score"]:
                cand[pid] = {
                    "base_score": float(p.score),
                    "payload": p.payload or {},
                }

    # Prepare recency normalization
    timestamps: List[int] = []
    for data in cand.values():
        md = (data.get("payload") or {}).get("metadata") or {}
        ts = md.get("ingested_at")
        if isinstance(ts, int):
            timestamps.append(ts)
    has_ts = len(timestamps) > 0
    if has_ts:
        tmin, tmax = min(timestamps), max(timestamps)
        span = max(1, tmax - tmin)

    # Auto-infer dominant language if none provided (no config needed)
    want_lang = args.language
    if not want_lang and cand:
        lang_counts: Dict[str, int] = defaultdict(int)
        for data in cand.values():
            md = (data.get("payload") or {}).get("metadata") or {}
            l = (md.get("language") or "").lower()
            if l:
                lang_counts[l] += 1
        if lang_counts:
            want_lang = max(lang_counts.items(), key=lambda x: x[1])[0]

    fused = []
    for pid, data in cand.items():
        md = (data["payload"] or {}).get("metadata") or {}
        final = score_candidate(
            data["base_score"], counts[pid], md, want_lang, args.under
        )
        # Symbol/path match boost using expanded queries
        sym_text = " ".join(
            [str(md.get("symbol") or ""), str(md.get("symbol_path") or "")]
        ).lower()
        if any((q.lower() in sym_text) for q in queries):
            final += SYMBOL_BOOST
        # Recency bump (normalized)
        if "span" in locals() and has_ts:
            ts = md.get("ingested_at")
            if isinstance(ts, int):
                norm = (ts - tmin) / span if span else 0.0
                final += RECENCY_WEIGHT * norm
        fused.append((final, pid, data))

    fused.sort(key=lambda x: x[0], reverse=True)

    print(f"Multi-query rerank: queries={len(args.query)} model={MODEL} vec={vec_name}")
    for i, (final, pid, data) in enumerate(fused[: args.limit], 1):
        md = (data["payload"] or {}).get("metadata") or {}
        info = (data["payload"] or {}).get("information") or (
            data["payload"] or {}
        ).get("document")
        path = md.get("path")
        lang = md.get("language")
        print(
            {
                "rank": i,
                "score": round(final, 4),
                "path": path,
                "language": lang,
                "information": info,
            }
        )


if __name__ == "__main__":
    main()
