#!/usr/bin/env python3
import os
import argparse
import sys
import threading
from pathlib import Path as _P
from typing import List, Dict, Any, TYPE_CHECKING

# Ensure project root is on sys.path when run as a script (so 'scripts' package imports work)
_ROOT = _P(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from qdrant_client import QdrantClient, models

# Import TextEmbedding for type hints (may not be available at runtime with embedder factory)
if TYPE_CHECKING:
    from fastembed import TextEmbedding

# Use embedder factory for Qwen3 support; fallback to direct fastembed
try:
    from scripts.embedder import get_embedding_model as _get_embedding_model
    _EMBEDDER_FACTORY = True
except ImportError:
    _EMBEDDER_FACTORY = False
    from fastembed import TextEmbedding

# Use centralized reranker factory (supports FastEmbed + ONNX backends)
try:
    from scripts.reranker import (
        get_reranker_model as _get_reranker_model,
        rerank_pairs as _rerank_pairs,
        is_reranker_available as _is_reranker_available,
    )
    _RERANKER_FACTORY = True
except ImportError:
    _RERANKER_FACTORY = False
    _get_reranker_model = None
    _rerank_pairs = None
    _is_reranker_available = None

# Legacy ONNX imports (fallback when factory unavailable)
try:
    import onnxruntime as ort  # type: ignore
    from tokenizers import Tokenizer  # type: ignore
except Exception:  # pragma: no cover
    ort = None
    Tokenizer = None

MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
API_KEY = os.environ.get("QDRANT_API_KEY")

# Legacy config (still supported for backwards compatibility)
RERANKER_ONNX_PATH = os.environ.get("RERANKER_ONNX_PATH", "")
RERANKER_TOKENIZER_PATH = os.environ.get("RERANKER_TOKENIZER_PATH", "")
RERANK_MAX_TOKENS = int(os.environ.get("RERANK_MAX_TOKENS", "512") or 512)
EF_SEARCH = int(os.environ.get("EF_SEARCH", "128") or 128)

# Module-level cache for legacy ONNX session (used when factory unavailable)
_RERANK_SESSION = None
_RERANK_TOKENIZER = None
_RERANK_LOCK = threading.Lock()

_WARMUP_DONE = False


def _get_rerank_session():
    """Get reranker session - uses factory if available, else legacy ONNX loading."""
    global _RERANK_SESSION, _RERANK_TOKENIZER

    # Prefer factory when available
    if _RERANKER_FACTORY and _get_reranker_model is not None:
        model = _get_reranker_model()
        if model is not None:
            # Return a marker that indicates factory mode
            return ("__factory__", model)
        # Factory available but no model configured - fall through to legacy

    # Legacy ONNX path
    if not (ort and Tokenizer and RERANKER_ONNX_PATH and RERANKER_TOKENIZER_PATH):
        return None, None
    if _RERANK_SESSION is not None and _RERANK_TOKENIZER is not None:
        return _RERANK_SESSION, _RERANK_TOKENIZER
    with _RERANK_LOCK:
        if _RERANK_SESSION is not None and _RERANK_TOKENIZER is not None:
            return _RERANK_SESSION, _RERANK_TOKENIZER
        tok = Tokenizer.from_file(RERANKER_TOKENIZER_PATH)
        try:
            tok.enable_truncation(max_length=RERANK_MAX_TOKENS)
        except Exception:
            pass
        try:
            # Provider selection: explicit RERANK_PROVIDERS overrides
            prov_env = os.environ.get("RERANK_PROVIDERS")
            providers = prov_env.split(",") if prov_env else None
            if not providers:
                try:
                    avail = set(ort.get_available_providers()) if ort else set()
                except Exception:
                    avail = set()
                use_trt = str(os.environ.get("RERANK_USE_TRT", "")).strip().lower() in {
                    "1",
                    "true",
                    "yes",
                    "on",
                }
                if use_trt and "TensorrtExecutionProvider" in avail:
                    providers = ["TensorrtExecutionProvider"]
                    if "CUDAExecutionProvider" in avail:
                        providers.append("CUDAExecutionProvider")
                    providers.append("CPUExecutionProvider")
                else:
                    providers = (
                        ["CUDAExecutionProvider"]
                        if "CUDAExecutionProvider" in avail
                        else []
                    ) + ["CPUExecutionProvider"]
            # Session options with full graph optimizations
            so = ort.SessionOptions()
            try:
                so.graph_optimization_level = getattr(
                    ort.GraphOptimizationLevel, "ORT_ENABLE_ALL", 99
                )
            except Exception:
                pass
            sess = ort.InferenceSession(
                RERANKER_ONNX_PATH, sess_options=so, providers=providers
            )
        except Exception:
            sess = None
        _RERANK_SESSION, _RERANK_TOKENIZER = sess, tok
        return _RERANK_SESSION, _RERANK_TOKENIZER


from scripts.utils import sanitize_vector_name as _sanitize_vector_name


def warmup_reranker():
    """Background warmup: load ONNX session and run a dummy inference."""
    global _WARMUP_DONE
    if _WARMUP_DONE:
        return
    sess, tok = _get_rerank_session()
    if sess and tok:
        try:
            # Dummy inference to warm up the session
            dummy_pairs = [("warmup query", "warmup document")]
            rerank_local(dummy_pairs)
        except Exception:
            pass
    _WARMUP_DONE = True


def _start_background_warmup():
    """Start background thread to warm up reranker."""
    if os.environ.get("RERANK_WARMUP", "1") == "1":
        t = threading.Thread(target=warmup_reranker, daemon=True)
        t.start()


# Auto-start warmup on module import
_start_background_warmup()


def _norm_under(u: str | None) -> str | None:
    if not u:
        return None
    u = str(u).strip().replace("\\", "/")
    u = "/".join([p for p in u.split("/") if p])
    if not u:
        return None
    if not u.startswith("/"):
        return "/work/" + u
    if not u.startswith("/work/"):
        return "/work/" + u.lstrip("/")
    return u


def _select_dense_vector_name(
    client: QdrantClient, collection: str, model: "TextEmbedding", dim: int
) -> str:
    try:
        info = client.get_collection(collection)
        cfg = info.config.params.vectors
        if isinstance(cfg, dict) and cfg:
            # Prefer name whose size matches embedding dim
            for name, params in cfg.items():
                psize = getattr(params, "size", None) or getattr(params, "dim", None)
                if psize and int(psize) == int(dim):
                    return name
            # If LEX exists, pick the other name as dense
            if hasattr(models, "VectorParams"):
                pass
            if "lex" in cfg:
                for name in cfg.keys():
                    if name != "lex":
                        return name
    except Exception:
        pass
    return _sanitize_vector_name(MODEL_NAME)


def dense_results(
    client: QdrantClient,
    model: "TextEmbedding",
    vec_name: str,
    query: str,
    flt,
    topk: int,
    collection_name: str,
) -> List[Any]:
    vec = next(model.embed([query])).tolist()
    try:
        qp = client.query_points(
            collection_name=collection_name,
            query=vec,
            using=vec_name,
            query_filter=flt,
            search_params=models.SearchParams(hnsw_ef=EF_SEARCH),
            limit=topk,
            with_payload=True,
        )
        return getattr(qp, "points", qp)
    except Exception:
        res = client.search(
            collection_name=collection_name,
            query_vector={"name": vec_name, "vector": vec},
            limit=topk,
            with_payload=True,
            query_filter=flt,
        )
        return res


def prepare_pairs(query: str, points: List[Any]) -> List[tuple[str, str]]:
    pairs: List[tuple[str, str]] = []
    for p in points:
        payload = p.payload or {}
        md = payload.get("metadata") or {}
        path = md.get("path") or ""
        lang = md.get("language") or ""
        kind = md.get("kind") or ""
        symp = md.get("symbol_path") or md.get("symbol") or ""
        code = (md.get("code") or "")[:600]
        # Include docstring/pseudo for better query matching
        docstring = (
            payload.get("docstring")
            or payload.get("pseudo")
            or md.get("docstring")
            or ""
        )
        header = f"[{lang}/{kind}] {symp} — {path}".strip()
        # Prepend docstring before code for reranker visibility
        doc_parts = [header]
        if docstring:
            doc_parts.append(docstring[:200])
        if code:
            doc_parts.append(code)
        doc = "\n".join(doc_parts).strip()
        if not doc:
            doc = payload.get("information") or ""
        pairs.append((query, doc))
    return pairs


def rerank_local(pairs: List[tuple[str, str]]) -> List[float]:
    """Score query-document pairs using available reranker backend.

    Supports both factory mode (FastEmbed/ONNX) and legacy direct ONNX mode.
    Backwards compatible - existing ONNX configs continue to work.
    """
    if not pairs:
        return []

    # Get reranker session (factory or legacy)
    result = _get_rerank_session()
    if result is None or result == (None, None):
        return [0.0 for _ in pairs]

    sess, tok = result

    # Factory mode: use centralized rerank_pairs
    if sess == "__factory__" and _RERANKER_FACTORY and _rerank_pairs is not None:
        return _rerank_pairs(pairs, model=tok)

    # Legacy ONNX mode: direct inference
    if sess is None or tok is None:
        return [0.0 for _ in pairs]

    # Proper pair encoding for token_type_ids
    enc = tok.encode_batch(pairs)
    input_ids = [e.ids for e in enc]
    attn = [e.attention_mask for e in enc]
    max_len = max((len(ids) for ids in input_ids), default=0)

    def pad(seq, pad_id=0):
        return seq + [pad_id] * (max_len - len(seq))

    input_ids = [pad(s) for s in input_ids]
    attn = [pad(s) for s in attn]
    input_names = [i.name for i in sess.get_inputs()]
    token_type_ids = (
        [[0] * max_len for _ in input_ids] if "token_type_ids" in input_names else None
    )
    feeds = {}
    if "input_ids" in input_names:
        feeds["input_ids"] = input_ids
    if "attention_mask" in input_names:
        feeds["attention_mask"] = attn
    if token_type_ids is not None:
        feeds["token_type_ids"] = token_type_ids
    if not feeds:
        feeds = {
            sess.get_inputs()[0].name: input_ids,
            sess.get_inputs()[1].name: attn,
        }
    out = sess.run(None, feeds)
    logits = out[0]
    scores: List[float] = []
    for row in logits:
        try:
            if isinstance(row, list) and len(row) == 2:
                scores.append(float(row[1]))
            elif hasattr(row, "__len__") and len(row) == 1:
                scores.append(float(row[0]))
            else:
                scores.append(float(sum(row) / max(1, len(row))))
        except Exception:
            scores.append(0.0)
    return scores


# In-process API: rerank using local ONNX; returns structured items
# Optional: pass an existing TextEmbedding instance via model to reuse cache


def rerank_in_process(
    query: str,
    topk: int = 50,
    limit: int = 12,
    language: str | None = None,
    under: str | None = None,
    model: "TextEmbedding | None" = None,
    collection: str | None = None,
) -> List[Dict[str, Any]]:
    eff_collection = (
        str(collection).strip()
        if isinstance(collection, str) and collection.strip()
        else (os.environ.get("COLLECTION_NAME") or "codebase")
    )
    client = QdrantClient(url=QDRANT_URL, api_key=API_KEY or None)
    if model:
        _model = model
    elif _EMBEDDER_FACTORY:
        _model = _get_embedding_model(MODEL_NAME)
    else:
        _model = TextEmbedding(model_name=MODEL_NAME)
    dim = len(next(_model.embed(["dimension probe"])))
    vec_name = _select_dense_vector_name(client, eff_collection, _model, dim)

    must = []
    if language:
        must.append(
            models.FieldCondition(
                key="metadata.language", match=models.MatchValue(value=language)
            )
        )
    eff_under = _norm_under(under)
    if eff_under:
        must.append(
            models.FieldCondition(
                key="metadata.path_prefix", match=models.MatchValue(value=eff_under)
            )
        )
    flt = models.Filter(must=must) if must else None

    pts = dense_results(client, _model, vec_name, query, flt, topk, eff_collection)
    if not pts and flt is not None:
        pts = dense_results(client, _model, vec_name, query, None, topk, eff_collection)
    if not pts:
        return []

    pairs = prepare_pairs(query, pts)
    scores = rerank_local(pairs)
    ranked = list(zip(scores, pts))
    ranked.sort(key=lambda x: x[0], reverse=True)
    items: List[Dict[str, Any]] = []
    for s, p in ranked[: max(0, int(limit))]:
        payload = p.payload or {}
        md = payload.get("metadata") or {}
        code_id = payload.get("code_id")
        doc_id = payload.get("doc_id") or payload.get("_id") or payload.get("id")
        payload_out: Dict[str, Any] = {}
        if code_id is not None:
            payload_out["code_id"] = code_id
        if doc_id is not None:
            payload_out["doc_id"] = doc_id
        if "_id" in payload:
            payload_out["_id"] = payload.get("_id")
        if "id" in payload:
            payload_out["id"] = payload.get("id")
        items.append(
            {
                "score": float(s),
                "path": md.get("path"),
                "symbol": md.get("symbol_path") or md.get("symbol") or "",
                "start_line": md.get("start_line"),
                "end_line": md.get("end_line"),
                "components": {"rerank_onnx": float(s)},
                "code_id": str(code_id) if code_id is not None else None,
                "doc_id": str(doc_id) if doc_id is not None else None,
                "payload": payload_out if payload_out else None,
            }
        )
    return items


def main():
    ap = argparse.ArgumentParser(description="Local cross-encoder rerank (ONNX)")
    ap.add_argument("--query", "-q", required=True)
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--language", type=str, default=None)
    ap.add_argument("--under", type=str, default=None)
    ap.add_argument("--collection", type=str, default=None)
    args = ap.parse_args()

    client = QdrantClient(url=QDRANT_URL, api_key=API_KEY or None)
    if _EMBEDDER_FACTORY:
        model = _get_embedding_model(MODEL_NAME)
    else:
        model = TextEmbedding(model_name=MODEL_NAME)
    dim = len(next(model.embed(["dimension probe"])))

    eff_collection = (
        str(args.collection).strip()
        if isinstance(args.collection, str) and args.collection.strip()
        else (os.environ.get("COLLECTION_NAME") or "codebase")
    )
    vec_name = _select_dense_vector_name(client, eff_collection, model, dim)

    must = []
    if args.language:
        must.append(
            models.FieldCondition(
                key="metadata.language", match=models.MatchValue(value=args.language)
            )
        )
    eff_under = _norm_under(args.under)
    if eff_under:
        must.append(
            models.FieldCondition(
                key="metadata.path_prefix", match=models.MatchValue(value=eff_under)
            )
        )
    flt = models.Filter(must=must) if must else None

    pts = dense_results(client, model, vec_name, args.query, flt, args.topk, eff_collection)
    # Fallback: if filtered search yields nothing, retry without filters to avoid empty rerank
    if not pts and flt is not None:
        pts = dense_results(client, model, vec_name, args.query, None, args.topk, eff_collection)
    if not pts:
        return
    pairs = prepare_pairs(args.query, pts)
    scores = rerank_local(pairs)
    ranked = list(zip(scores, pts))
    ranked.sort(key=lambda x: x[0], reverse=True)
    for s, p in ranked[: args.limit]:
        md = (p.payload or {}).get("metadata") or {}
        print(
            f"{s:.3f}\t{md.get('path')}\t{md.get('symbol_path') or md.get('symbol') or ''}\t{md.get('start_line')}-{md.get('end_line')}"
        )


if __name__ == "__main__":
    main()
