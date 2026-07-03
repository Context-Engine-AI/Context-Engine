
#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from qdrant_client import QdrantClient
from qdrant_client import models

from fastembed import TextEmbedding

from scripts.utils import sanitize_vector_name


def _env_int(name: str, default: int) -> int:
    try:
        v = os.environ.get(name)
        if v is None:
            return default
        v = v.strip()
        if not v:
            return default
        return int(v)
    except Exception:
        return default


def _env_str(name: str, default: str) -> str:
    v = os.environ.get(name)
    if v is None:
        return default
    v = v.strip()
    return v or default


LEX_VECTOR_NAME = _env_str("LEX_VECTOR_NAME", "lex")
LEX_VECTOR_DIM = _env_int("LEX_VECTOR_DIM", 4096)
MINI_VECTOR_NAME = _env_str("MINI_VECTOR_NAME", "mini")
MINI_VEC_DIM = _env_int("MINI_VEC_DIM", 64)


try:
    from scripts.utils import lex_hash_vector_text
except Exception:
    lex_hash_vector_text = None

try:
    from scripts.ingest_code import project_mini
except Exception:
    project_mini = None


@dataclass(frozen=True)
class CopyStats:
    points_written: int
    missing_dense: int
    missing_lex: int
    missing_mini: int
    filled_lex: int
    filled_mini: int


def _payload_text_for_embedding(payload: Dict[str, Any]) -> str:
    info = payload.get("information") or payload.get("document")
    if isinstance(info, str) and info.strip():
        return info
    md = payload.get("metadata") or {}
    if isinstance(md, dict):
        code = md.get("code")
        if isinstance(code, str) and code.strip():
            return code
    return ""


def _payload_text_for_lex(payload: Dict[str, Any]) -> str:
    md = payload.get("metadata") or {}
    code = ""
    if isinstance(md, dict):
        v = md.get("code")
        if isinstance(v, str):
            code = v
    pseudo = payload.get("pseudo") if isinstance(payload.get("pseudo"), str) else ""
    tags = payload.get("tags") if isinstance(payload.get("tags"), list) else []
    tags_s = " ".join([str(t) for t in tags if t is not None]) if tags else ""
    out = code
    if pseudo:
        out = (out + " " + pseudo) if out else pseudo
    if tags_s:
        out = (out + " " + tags_s) if out else tags_s
    if out.strip():
        return out
    return _payload_text_for_embedding(payload)


def _as_floats(v: Any) -> List[float]:
    try:
        if hasattr(v, "tolist"):
            return list(v.tolist())
    except Exception:
        pass
    return list(v)


def _scroll_all(
    client: QdrantClient,
    collection: str,
    *,
    limit: int,
) -> Iterable[List[Any]]:
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            limit=limit,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        if not points:
            break
        yield points
        if offset is None:
            break


def _recreate_collection(
    client: QdrantClient,
    *,
    name: str,
    dense_vector_name: str,
    dense_dim: int,
    include_lex: bool,
    include_mini: bool,
    timeout_hint_s: int,
) -> None:
    try:
        client.delete_collection(name)
    except Exception:
        pass

    vectors_cfg: Dict[str, models.VectorParams] = {
        dense_vector_name: models.VectorParams(
            size=int(dense_dim),
            distance=models.Distance.COSINE,
        )
    }
    if include_lex:
        vectors_cfg[LEX_VECTOR_NAME] = models.VectorParams(
            size=int(LEX_VECTOR_DIM),
            distance=models.Distance.COSINE,
        )
    if include_mini:
        vectors_cfg[MINI_VECTOR_NAME] = models.VectorParams(
            size=int(MINI_VEC_DIM),
            distance=models.Distance.COSINE,
        )

    client.create_collection(
        collection_name=name,
        vectors_config=vectors_cfg,
        hnsw_config=models.HnswConfigDiff(m=16, ef_construct=256),
    )

    print(
        f"[created] {name} vectors={list(vectors_cfg.keys())} "
        f"(timeout_hint_s={timeout_hint_s})",
        flush=True,
    )


def clone_collection(
    client: QdrantClient,
    *,
    src: str,
    dest: str,
    src_dense_vector_name: str,
    dest_dense_vector_name: str,
    scroll_batch: int,
    upsert_batch: int,
    upsert_wait: bool,
    copy_lex: bool,
    copy_mini: bool,
    fill_missing_lex: bool,
    fill_missing_mini: bool,
) -> CopyStats:
    missing_dense = 0
    missing_lex = 0
    missing_mini = 0
    filled_lex = 0
    filled_mini = 0
    written = 0

    if fill_missing_lex and lex_hash_vector_text is None:
        raise RuntimeError("fill_missing_lex requested but scripts.utils.lex_hash_vector_text import failed")
    if fill_missing_mini and project_mini is None:
        raise RuntimeError("fill_missing_mini requested but scripts.ingest_code.project_mini import failed")

    t0 = time.time()
    buf: List[models.PointStruct] = []

    def flush_buf() -> None:
        nonlocal written
        if not buf:
            return
        client.upsert(collection_name=dest, points=buf, wait=upsert_wait)
        written += len(buf)
        buf.clear()

    for batch in _scroll_all(client, src, limit=scroll_batch):
        for p in batch:
            v = p.vector or {}
            if not isinstance(v, dict):
                missing_dense += 1
                continue

            dense = v.get(src_dense_vector_name)
            if dense is None:
                missing_dense += 1
                continue

            vecs: Dict[str, Any] = {dest_dense_vector_name: dense}

            lex = v.get(LEX_VECTOR_NAME)
            mini = v.get(MINI_VECTOR_NAME)

            if copy_lex:
                if lex is None:
                    missing_lex += 1
                    if fill_missing_lex:
                        payload = p.payload or {}
                        lex_text = _payload_text_for_lex(payload)
                        lex = lex_hash_vector_text(lex_text)
                        filled_lex += 1
                if lex is not None:
                    vecs[LEX_VECTOR_NAME] = lex

            if copy_mini:
                if mini is None:
                    missing_mini += 1
                    if fill_missing_mini:
                        mini = project_mini(_as_floats(dense), MINI_VEC_DIM)
                        filled_mini += 1
                if mini is not None:
                    vecs[MINI_VECTOR_NAME] = mini

            buf.append(models.PointStruct(id=p.id, vector=vecs, payload=p.payload))
            if len(buf) >= upsert_batch:
                flush_buf()

        flush_buf()
        if written and written % 2048 == 0:
            print(f"[clone] {dest}: written={written} elapsed={time.time()-t0:.1f}s", flush=True)

    flush_buf()
    return CopyStats(
        points_written=written,
        missing_dense=missing_dense,
        missing_lex=missing_lex,
        missing_mini=missing_mini,
        filled_lex=filled_lex,
        filled_mini=filled_mini,
    )


def reembed_dense(
    client: QdrantClient,
    *,
    src: str,
    dest: str,
    src_lex_vector_name: str,
    src_mini_vector_name: str,
    dest_dense_vector_name: str,
    model_name: str,
    scroll_batch: int,
    embed_batch: int,
    upsert_batch: int,
    upsert_wait: bool,
    copy_lex: bool,
    copy_mini: bool,
    fill_missing_lex: bool,
    reembed_mini: bool,
    fill_missing_mini: bool,
) -> CopyStats:
    missing_dense = 0
    missing_lex = 0
    missing_mini = 0
    filled_lex = 0
    filled_mini = 0
    written = 0

    if fill_missing_lex and lex_hash_vector_text is None:
        raise RuntimeError("fill_missing_lex requested but scripts.utils.lex_hash_vector_text import failed")
    if (reembed_mini or fill_missing_mini) and project_mini is None:
        raise RuntimeError("mini generation requested but scripts.ingest_code.project_mini import failed")

    model = TextEmbedding(model_name=model_name)
    t0 = time.time()

    pts_buf: List[models.PointStruct] = []

    def flush_pts() -> None:
        nonlocal written
        if not pts_buf:
            return
        client.upsert(collection_name=dest, points=pts_buf, wait=upsert_wait)
        written += len(pts_buf)
        pts_buf.clear()

    for batch in _scroll_all(client, src, limit=scroll_batch):
        items: List[Any] = []
        texts: List[str] = []
        for p in batch:
            payload = p.payload or {}
            texts.append(_payload_text_for_embedding(payload))
            items.append(p)

        for i in range(0, len(items), embed_batch):
            sub_items = items[i : i + embed_batch]
            sub_texts = texts[i : i + embed_batch]
            dense_vecs = list(model.embed(sub_texts))

            for p, dv in zip(sub_items, dense_vecs):
                v = p.vector or {}
                if not isinstance(v, dict):
                    missing_dense += 1
                    continue

                vecs: Dict[str, Any] = {dest_dense_vector_name: _as_floats(dv)}

                lex = v.get(src_lex_vector_name)
                mini = v.get(src_mini_vector_name)

                if copy_lex:
                    if lex is None:
                        missing_lex += 1
                        if fill_missing_lex:
                            payload = p.payload or {}
                            lex_text = _payload_text_for_lex(payload)
                            lex = lex_hash_vector_text(lex_text)
                            filled_lex += 1
                    if lex is not None:
                        vecs[LEX_VECTOR_NAME] = lex

                if copy_mini:
                    if reembed_mini:
                        mini = project_mini(vecs[dest_dense_vector_name], MINI_VEC_DIM)
                        filled_mini += 1
                    else:
                        if mini is None:
                            missing_mini += 1
                            if fill_missing_mini:
                                mini = project_mini(vecs[dest_dense_vector_name], MINI_VEC_DIM)
                                filled_mini += 1
                    if mini is not None:
                        vecs[MINI_VECTOR_NAME] = mini

                pts_buf.append(models.PointStruct(id=p.id, vector=vecs, payload=p.payload))
                if len(pts_buf) >= upsert_batch:
                    flush_pts()

            flush_pts()

        if written and written % 1024 == 0:
            print(f"[reembed] {dest}: written={written} elapsed={time.time()-t0:.1f}s", flush=True)

    flush_pts()
    return CopyStats(
        points_written=written,
        missing_dense=missing_dense,
        missing_lex=missing_lex,
        missing_mini=missing_mini,
        filled_lex=filled_lex,
        filled_mini=filled_mini,
    )


def _probe_dim(model_name: str) -> int:
    model = TextEmbedding(model_name=model_name)
    return len(next(model.embed(["dimension probe"])))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Clone a source Qdrant collection into benchmark collections (clone BGE; recreate + re-embed dense for MiniLM)."
    )
    ap.add_argument("--qdrant-url", default=os.environ.get("QDRANT_URL", "http://qdrant:6333"))
    ap.add_argument("--qdrant-timeout", type=int, default=int(os.environ.get("QDRANT_TIMEOUT", "180") or 180))
    ap.add_argument("--src", required=True, help="Source Qdrant collection name")

    ap.add_argument("--bge-dest", default="bench-bge")
    ap.add_argument("--bge-model", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--skip-bge", action="store_true")

    ap.add_argument("--minilm-dest", default="bench-minilm")
    ap.add_argument("--minilm-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--skip-minilm", action="store_true")

    ap.add_argument("--scroll-batch", type=int, default=256)
    ap.add_argument("--embed-batch", type=int, default=128)
    ap.add_argument("--upsert-batch", type=int, default=256)
    ap.add_argument("--upsert-wait", action="store_true", help="Wait for each upsert to be committed")

    ap.add_argument("--no-lex", action="store_true", help="Do not write lexical vectors")
    ap.add_argument("--no-mini", action="store_true", help="Do not write mini vectors")
    ap.add_argument("--fill-missing-lex", action="store_true", help="Compute lex vectors when missing in source")
    ap.add_argument(
        "--fill-missing-mini",
        action="store_true",
        help="Compute mini vectors from dense when missing in source",
    )
    ap.add_argument(
        "--reembed-mini",
        action="store_true",
        help="When re-embedding MiniLM dense, also recompute mini vector from the new dense vector",
    )

    args = ap.parse_args()

    include_lex = not args.no_lex
    include_mini = not args.no_mini

    client = QdrantClient(url=args.qdrant_url, timeout=args.qdrant_timeout)
    src_info = client.get_collection(args.src)
    src_vectors = src_info.config.params.vectors
    if not isinstance(src_vectors, dict):
        raise RuntimeError(f"Source collection {args.src} does not use named vectors")

    bge_dense_name = sanitize_vector_name(args.bge_model)
    minilm_dense_name = sanitize_vector_name(args.minilm_model)

    src_dense_candidates = [bge_dense_name] + [k for k in src_vectors.keys() if k not in {LEX_VECTOR_NAME, MINI_VECTOR_NAME}]
    src_dense_name = None
    for k in src_dense_candidates:
        if k in src_vectors:
            src_dense_name = k
            break
    if not src_dense_name:
        raise RuntimeError(f"Could not determine source dense vector name in {args.src}. vectors={list(src_vectors.keys())}")

    print(
        f"[src] {args.src} points={src_info.points_count} vectors={list(src_vectors.keys())} src_dense={src_dense_name}",
        flush=True,
    )

    if not args.skip_bge:
        bge_dim = getattr(src_vectors.get(src_dense_name), "size", None) or _probe_dim(args.bge_model)
        _recreate_collection(
            client,
            name=args.bge_dest,
            dense_vector_name=bge_dense_name,
            dense_dim=int(bge_dim),
            include_lex=include_lex,
            include_mini=include_mini,
            timeout_hint_s=args.qdrant_timeout,
        )
        t0 = time.time()
        stats = clone_collection(
            client,
            src=args.src,
            dest=args.bge_dest,
            src_dense_vector_name=src_dense_name,
            dest_dense_vector_name=bge_dense_name,
            scroll_batch=args.scroll_batch,
            upsert_batch=args.upsert_batch,
            upsert_wait=args.upsert_wait,
            copy_lex=include_lex,
            copy_mini=include_mini,
            fill_missing_lex=args.fill_missing_lex,
            fill_missing_mini=args.fill_missing_mini,
        )
        print(
            f"[done] clone -> {args.bge_dest}: points={stats.points_written} "
            f"elapsed={time.time()-t0:.1f}s missing=(dense={stats.missing_dense},lex={stats.missing_lex},mini={stats.missing_mini}) "
            f"filled=(lex={stats.filled_lex},mini={stats.filled_mini})",
            flush=True,
        )

    if not args.skip_minilm:
        minilm_dim = _probe_dim(args.minilm_model)
        _recreate_collection(
            client,
            name=args.minilm_dest,
            dense_vector_name=minilm_dense_name,
            dense_dim=int(minilm_dim),
            include_lex=include_lex,
            include_mini=include_mini,
            timeout_hint_s=args.qdrant_timeout,
        )
        t0 = time.time()
        stats = reembed_dense(
            client,
            src=args.src,
            dest=args.minilm_dest,
            src_lex_vector_name=LEX_VECTOR_NAME,
            src_mini_vector_name=MINI_VECTOR_NAME,
            dest_dense_vector_name=minilm_dense_name,
            model_name=args.minilm_model,
            scroll_batch=min(args.scroll_batch, 256),
            embed_batch=args.embed_batch,
            upsert_batch=args.upsert_batch,
            upsert_wait=args.upsert_wait,
            copy_lex=include_lex,
            copy_mini=include_mini,
            fill_missing_lex=args.fill_missing_lex,
            reembed_mini=args.reembed_mini,
            fill_missing_mini=args.fill_missing_mini,
        )
        print(
            f"[done] reembed -> {args.minilm_dest}: points={stats.points_written} "
            f"elapsed={time.time()-t0:.1f}s missing=(dense={stats.missing_dense},lex={stats.missing_lex},mini={stats.missing_mini}) "
            f"filled=(lex={stats.filled_lex},mini={stats.filled_mini})",
            flush=True,
        )

    print("[ok]", flush=True)


if __name__ == "__main__":
    main()

