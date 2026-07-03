#!/usr/bin/env python3
"""
mcp/search_history.py - Git history search implementations for MCP indexer server.

Extracted from mcp_indexer_server.py for better modularity.
Contains:
- _search_commits_for_impl: Search git commit history
- _change_history_for_path_impl: Get change history for a file path

Note: The @mcp.tool() decorated functions remain in mcp_indexer_server.py
as thin wrappers that call these implementations.
"""

from __future__ import annotations

__all__ = [
    "_search_commits_for_impl",
    "_change_history_for_path_impl",
]

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Environment
QDRANT_URL = os.environ.get("QDRANT_URL", "http://qdrant:6333")


async def _search_commits_for_impl(
    query: Any = None,
    path: Any = None,
    collection: Any = None,
    limit: Any = None,
    max_points: Any = None,
    default_collection_fn=None,
    get_embedding_model_fn=None,
) -> Dict[str, Any]:
    """Search git commit history indexed in Qdrant.

    What it does:
    - Queries commit documents ingested by scripts/ingest_history.py
    - Filters by optional file path (metadata.files contains path)

    Parameters:
    - query: str or list[str]; matched lexically against commit message/text
    - path: str (optional). Relative path under /work; filters commits that touched this file
    - collection: str (optional). Defaults to env/WS collection
    - limit: int (optional, default 10). Max commits to return
    - max_points: int (optional). Safety cap on scanned points (default 1000)

    Returns:
    - {"ok": true, "results": [{"commit_id", "author_name", "authored_date", "message", "files"}, ...], "scanned": int}
    - On error: {"ok": false, "error": "..."}
    """
    # Get default collection function
    if default_collection_fn is None:
        from scripts.mcp_impl.workspace import _default_collection
        default_collection_fn = _default_collection

    # Normalize inputs
    q_terms: list[str] = []
    if isinstance(query, (list, tuple)):
        for x in query:
            for tok in str(x).strip().split():
                if tok.strip():
                    q_terms.append(tok.strip().lower())
    elif query is not None:
        qs = str(query).strip()
        if qs:
            for tok in qs.split():
                if tok.strip():
                    q_terms.append(tok.strip().lower())
    p = str(path or "").strip()
    coll = str(collection or "").strip() or default_collection_fn()
    try:
        lim = int(limit) if limit not in (None, "") else 10
    except (ValueError, TypeError):
        lim = 10
    try:
        mcap = int(max_points) if max_points not in (None, "") else 1000
    except (ValueError, TypeError):
        mcap = 1000
    
    use_scoring = bool(q_terms)
    max_ids_for_scan = mcap if use_scoring else lim

    try:
        from qdrant_client import QdrantClient  # type: ignore
        from qdrant_client import models as qmodels  # type: ignore

        client = QdrantClient(
            url=QDRANT_URL,
            api_key=os.environ.get("QDRANT_API_KEY"),
            timeout=float(os.environ.get("QDRANT_TIMEOUT", "20") or 20),
        )

        # Restrict to commit documents ingested by ingest_history.py
        filt = qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="metadata.language", match=qmodels.MatchValue(value="git")
                ),
                qmodels.FieldCondition(
                    key="metadata.kind", match=qmodels.MatchValue(value="git_message")
                ),
            ]
        )

        # Optional vector-augmented scoring
        vector_scores: Dict[str, float] = {}
        use_vectors = use_scoring and str(
            os.environ.get("COMMIT_VECTOR_SEARCH", "0") or "1"
        ).strip().lower() in {"1", "true", "yes", "on"}
        
        if use_vectors:
            try:
                try:
                    from scripts.utils import sanitize_vector_name as _sanitize_vector_name
                except Exception:
                    _sanitize_vector_name = None

                model_name = os.environ.get("MODEL_NAME", "BAAI/bge-base-en-v1.5")
                vec_name: Optional[str]
                if _sanitize_vector_name is not None:
                    try:
                        vec_name = _sanitize_vector_name(model_name)
                    except Exception:
                        vec_name = None
                else:
                    vec_name = None

                if vec_name:
                    if get_embedding_model_fn is None:
                        from scripts.mcp_impl.admin_tools import _get_embedding_model
                        get_embedding_model_fn = _get_embedding_model
                    
                    embed_model = get_embedding_model_fn(model_name)
                    qtext = " ".join(q_terms) if q_terms else ""
                    if qtext.strip():
                        qvec = next(embed_model.embed([qtext])).tolist()

                        def _vec_search():
                            return client.search(
                                collection_name=coll,
                                query_vector={vec_name: qvec},
                                query_filter=filt,
                                limit=min(mcap, 128),
                                with_payload=True,
                                with_vectors=False,
                            )

                        v_hits = await asyncio.to_thread(_vec_search)
                        for sp in v_hits or []:
                            payload_v = getattr(sp, "payload", {}) or {}
                            md_v = payload_v.get("metadata") or {}
                            cid_v = md_v.get("commit_id") or md_v.get("symbol")
                            scid_v = str(cid_v) if cid_v is not None else ""
                            if not scid_v:
                                continue
                            try:
                                vs = float(getattr(sp, "score", 0.0) or 0.0)
                            except Exception:
                                vs = 0.0
                            if vs <= 0.0:
                                continue
                            if scid_v not in vector_scores or vs > vector_scores[scid_v]:
                                vector_scores[scid_v] = vs
            except Exception:
                vector_scores = {}

        page = None
        scanned = 0
        out: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        while scanned < mcap and len(seen_ids) < max_ids_for_scan:
            sc, page = await asyncio.to_thread(
                lambda: client.scroll(
                    collection_name=coll,
                    with_payload=True,
                    with_vectors=False,
                    limit=200,
                    offset=page,
                    scroll_filter=filt,
                )
            )
            if not sc:
                break
            for pt in sc:
                scanned += 1
                if scanned > mcap:
                    break
                payload = getattr(pt, "payload", {}) or {}
                md = payload.get("metadata") or {}
                msg = str(md.get("message") or "")
                info = str(payload.get("information") or "")
                files = md.get("files") or []
                try:
                    files_list = [str(f) for f in files]
                except Exception:
                    files_list = []
                # Optional lineage-style metadata
                lg = md.get("lineage_goal")
                if isinstance(lg, str):
                    lineage_goal = lg.strip()
                else:
                    lineage_goal = ""
                ls_raw = md.get("lineage_symbols") or []
                if isinstance(ls_raw, list):
                    lineage_symbols = [
                        str(x).strip() for x in ls_raw if str(x).strip()
                    ][:6]
                else:
                    lineage_symbols = []
                lt_raw = md.get("lineage_tags") or []
                if isinstance(lt_raw, list):
                    lineage_tags = [
                        str(x).strip() for x in lt_raw if str(x).strip()
                    ][:6]
                else:
                    lineage_tags = []

                # Field-aware lexical scoring
                score = 0.0
                if q_terms:
                    msg_l = msg.lower()
                    info_l = info.lower()
                    goal_l = lineage_goal.lower() if lineage_goal else ""
                    sym_l = " ".join(lineage_symbols).lower() if lineage_symbols else ""
                    tags_l = " ".join(lineage_tags).lower() if lineage_tags else ""
                    hits = 0
                    for t in q_terms:
                        term_hit = False
                        if goal_l and t in goal_l:
                            score += 3.0
                            term_hit = True
                        if tags_l and t in tags_l:
                            score += 2.0
                            term_hit = True
                        if sym_l and t in sym_l:
                            score += 1.5
                            term_hit = True
                        if msg_l and t in msg_l:
                            score += 1.0
                            term_hit = True
                        if info_l and t in info_l:
                            score += 0.5
                            term_hit = True
                        if term_hit:
                            hits += 1
                    if hits == 0:
                        continue
                if p:
                    if not any(p in f for f in files_list):
                        continue
                cid = md.get("commit_id") or md.get("symbol")
                scid = str(cid) if cid is not None else ""
                if not scid or scid in seen_ids:
                    continue
                # Blend in vector similarity score
                try:
                    if use_scoring and vector_scores and scid in vector_scores:
                        vec_score = float(vector_scores.get(scid, 0.0) or 0.0)
                        if vec_score > 0.0:
                            weight = float(
                                os.environ.get("COMMIT_VECTOR_WEIGHT", "2.0") or 2.0
                            )
                            score += weight * vec_score
                except Exception:
                    pass
                seen_ids.add(scid)
                out.append(
                    {
                        "commit_id": cid,
                        "author_name": md.get("author_name"),
                        "authored_date": md.get("authored_date"),
                        "message": msg.splitlines()[0] if msg else "",
                        "files": files_list,
                        "lineage_goal": lineage_goal,
                        "lineage_symbols": lineage_symbols,
                        "lineage_tags": lineage_tags,
                        "_score": score,
                    }
                )
                if len(seen_ids) >= max_ids_for_scan:
                    break
        results = out
        if use_scoring and results:
            try:
                results = sorted(
                    results,
                    key=lambda c: float(c.get("_score", 0.0)),
                    reverse=True,
                )
            except Exception:
                pass
            results = results[:lim]
            for c in results:
                c.pop("_score", None)
        return {"ok": True, "results": results, "scanned": scanned, "collection": coll}
    except Exception as e:
        return {"ok": False, "error": str(e), "collection": coll}


async def _change_history_for_path_impl(
    path: Any,
    collection: Any = None,
    max_points: Any = None,
    include_commits: Any = None,
    default_collection_fn=None,
    search_commits_fn=None,
) -> Dict[str, Any]:
    """Summarize recent change metadata for a file path from the index.

    Parameters:
    - path: str. Relative path under /work.
    - collection: str (optional). Defaults to env/WS default.
    - max_points: int (optional). Safety cap on scanned points.
    - include_commits: bool (optional). If true, attach a small list of recent commits
      touching this path based on the commit index.

    Returns:
    - {"ok": true, "summary": {...}} or {"ok": false, "error": "..."}.
    """
    # Get default collection function
    if default_collection_fn is None:
        from scripts.mcp_impl.workspace import _default_collection
        default_collection_fn = _default_collection

    p = str(path or "").strip()
    if not p:
        return {"error": "path required"}
    coll = str(collection or "").strip() or default_collection_fn()
    try:
        mcap = int(max_points) if max_points not in (None, "") else 200
    except (ValueError, TypeError):
        mcap = 200
    # Treat include_commits as a loose boolean flag
    inc_commits = False
    if include_commits not in (None, ""):
        try:
            inc_commits = str(include_commits).strip().lower() in {"1", "true", "yes", "on"}
        except Exception:
            inc_commits = False

    try:
        from qdrant_client import QdrantClient  # type: ignore
        from qdrant_client import models as qmodels  # type: ignore

        client = QdrantClient(
            url=QDRANT_URL,
            api_key=os.environ.get("QDRANT_API_KEY"),
            timeout=float(os.environ.get("QDRANT_TIMEOUT", "20") or 20),
        )
        # Strict exact match on metadata.path (Compose maps to /work)
        filt = qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="metadata.path", match=qmodels.MatchValue(value=p)
                )
            ]
        )
        page = None
        total = 0
        hashes = set()
        last_mods = []
        ingested = []
        churns = []
        while total < mcap:
            sc, page = await asyncio.to_thread(
                lambda: client.scroll(
                    collection_name=coll,
                    with_payload=True,
                    with_vectors=False,
                    limit=200,
                    offset=page,
                    scroll_filter=filt,
                )
            )
            if not sc:
                break
            for pt in sc:
                md = (getattr(pt, "payload", {}) or {}).get("metadata") or {}
                fh = md.get("file_hash")
                if fh:
                    hashes.add(str(fh))
                lm = md.get("last_modified_at")
                ia = md.get("ingested_at")
                ch = md.get("churn_count")
                if lm is not None:
                    last_mods.append(int(lm))
                if ia is not None:
                    ingested.append(int(ia))
                if ch is not None:
                    churns.append(int(ch))
                total += 1
                if total >= mcap:
                    break
        summary: Dict[str, Any] = {
            "path": p,
            "points_scanned": total,
            "distinct_hashes": len(hashes),
            "last_modified_min": min(last_mods) if last_mods else None,
            "last_modified_max": max(last_mods) if last_mods else None,
            "ingested_min": min(ingested) if ingested else None,
            "ingested_max": max(ingested) if ingested else None,
            "churn_count_max": max(churns) if churns else None,
        }
        if inc_commits:
            try:
                if search_commits_fn is None:
                    search_commits_fn = _search_commits_for_impl
                commits = await search_commits_fn(
                    query=None,
                    path=p,
                    collection=coll,
                    limit=10,
                    max_points=1000,
                )
                if isinstance(commits, dict) and commits.get("ok"):
                    raw = commits.get("results") or []
                    seen: set[str] = set()
                    uniq: list[dict[str, Any]] = []
                    for c in raw:
                        cid = c.get("commit_id") if isinstance(c, dict) else None
                        scid = str(cid) if cid is not None else ""
                        if not scid or scid in seen:
                            continue
                        seen.add(scid)
                        uniq.append(c)
                    summary["commits"] = uniq
            except Exception:
                pass
        return {"ok": True, "summary": summary}
    except Exception as e:
        return {"ok": False, "error": str(e), "path": p}

