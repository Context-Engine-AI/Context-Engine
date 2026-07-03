"""Centralized reranker factory with FastEmbed cross-encoder support.

This module provides a unified interface for reranker model initialization,
supporting both FastEmbed's TextCrossEncoder (auto-downloads ONNX) and
manual ONNX paths for custom models.

Environment Variables:
    RERANKER_MODEL: FastEmbed reranker model name (e.g., jinaai/jina-reranker-v2-base-multilingual)
    RERANKER_ONNX_PATH: Manual ONNX model path (fallback if RERANKER_MODEL not set)
    RERANKER_TOKENIZER_PATH: Manual tokenizer path (required with RERANKER_ONNX_PATH)
    RERANK_MAX_TOKENS: Max tokens for reranker input (default: 512)
"""

from __future__ import annotations

import os
import threading
from typing import Any, List, Optional, Tuple

# Default FastEmbed reranker model (None = disabled, use ONNX paths)
DEFAULT_RERANKER_MODEL: Optional[str] = None


def _get_reranker_model_name() -> Optional[str]:
    """Get RERANKER_MODEL from environment, allowing dotenv to work."""
    return os.environ.get("RERANKER_MODEL", "").strip() or None


def _get_reranker_onnx_path() -> Optional[str]:
    """Get RERANKER_ONNX_PATH from environment."""
    return os.environ.get("RERANKER_ONNX_PATH", "").strip() or None


def _get_reranker_tokenizer_path() -> Optional[str]:
    """Get RERANKER_TOKENIZER_PATH from environment."""
    return os.environ.get("RERANKER_TOKENIZER_PATH", "").strip() or None


def _get_rerank_max_tokens() -> int:
    """Get RERANK_MAX_TOKENS from environment."""
    return int(os.environ.get("RERANK_MAX_TOKENS", "512") or 512)


# Module-level cache and locks
_RERANKER_CACHE: dict[str, Any] = {}
_RERANKER_LOCK = threading.Lock()

# Feature detection
_HAS_FASTEMBED_RERANK: Optional[bool] = None
_HAS_ONNX: Optional[bool] = None


def _check_fastembed_rerank() -> bool:
    """Check if FastEmbed reranking is available."""
    global _HAS_FASTEMBED_RERANK
    if _HAS_FASTEMBED_RERANK is None:
        try:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
            _HAS_FASTEMBED_RERANK = True
        except ImportError:
            _HAS_FASTEMBED_RERANK = False
    return _HAS_FASTEMBED_RERANK


def _check_onnx() -> bool:
    """Check if ONNX runtime is available."""
    global _HAS_ONNX
    if _HAS_ONNX is None:
        try:
            import onnxruntime
            from tokenizers import Tokenizer
            _HAS_ONNX = True
        except ImportError:
            _HAS_ONNX = False
    return _HAS_ONNX


def get_reranker_model(model_name: Optional[str] = None) -> Optional[Any]:
    """Get or create a cached reranker model instance.

    Priority:
    1. If model_name or RERANKER_MODEL is set -> use FastEmbed TextCrossEncoder
    2. Else if RERANKER_ONNX_PATH + RERANKER_TOKENIZER_PATH -> return ONNX session tuple
    3. Else -> return None (no reranker available)

    Args:
        model_name: Model name override. If None, uses RERANKER_MODEL env var.

    Returns:
        TextCrossEncoder instance, (session, tokenizer) tuple, or None.
    """
    effective_model = model_name or _get_reranker_model_name()

    # FastEmbed path
    if effective_model and _check_fastembed_rerank():
        cached = _RERANKER_CACHE.get(f"fastembed:{effective_model}")
        if cached is not None:
            return cached

        with _RERANKER_LOCK:
            cached = _RERANKER_CACHE.get(f"fastembed:{effective_model}")
            if cached is not None:
                return cached

            try:
                from fastembed.rerank.cross_encoder import TextCrossEncoder
                reranker = TextCrossEncoder(model_name=effective_model)
                _RERANKER_CACHE[f"fastembed:{effective_model}"] = reranker
                return reranker
            except Exception:
                return None

    # ONNX fallback path
    onnx_path = _get_reranker_onnx_path()
    tokenizer_path = _get_reranker_tokenizer_path()
    if onnx_path and tokenizer_path and _check_onnx():
        cached = _RERANKER_CACHE.get("onnx:manual")
        if cached is not None:
            return cached

        with _RERANKER_LOCK:
            cached = _RERANKER_CACHE.get("onnx:manual")
            if cached is not None:
                return cached

            try:
                import onnxruntime as ort
                from tokenizers import Tokenizer

                tok = Tokenizer.from_file(tokenizer_path)
                max_tokens = _get_rerank_max_tokens()
                try:
                    tok.enable_truncation(max_length=max_tokens)
                except Exception:
                    pass

                sess = ort.InferenceSession(
                    onnx_path,
                    providers=["CPUExecutionProvider"]
                )
                result = (sess, tok)
                _RERANKER_CACHE["onnx:manual"] = result
                return result
            except Exception:
                return None

    return None


def rerank_pairs(
    pairs: List[Tuple[str, str]],
    model: Optional[Any] = None,
) -> List[float]:
    """Score query-document pairs using the reranker.

    Args:
        pairs: List of (query, document) tuples.
        model: Reranker model (from get_reranker_model). If None, auto-loads.

    Returns:
        List of relevance scores (higher = more relevant).
    """
    if not pairs:
        return []

    if model is None:
        model = get_reranker_model()

    if model is None:
        return [0.0] * len(pairs)

    # FastEmbed TextCrossEncoder
    if hasattr(model, "rerank"):
        try:
            # TextCrossEncoder.rerank expects query and documents separately
            query = pairs[0][0]
            documents = [doc for _, doc in pairs]
            results = list(model.rerank(query, documents, top_k=len(documents)))
            # Results are floats (scores in document order)
            return [float(s) for s in results]
        except Exception:
            return [0.0] * len(pairs)

    # ONNX session tuple (session, tokenizer)
    if isinstance(model, tuple) and len(model) == 2:
        sess, tok = model
        return _onnx_rerank(sess, tok, pairs)

    return [0.0] * len(pairs)


def _onnx_rerank(sess: Any, tok: Any, pairs: List[Tuple[str, str]]) -> List[float]:
    """Score pairs using ONNX cross-encoder session."""
    try:
        enc = tok.encode_batch(pairs)
        input_ids = [e.ids for e in enc]
        attn = [e.attention_mask for e in enc]
        max_len = max((len(ids) for ids in input_ids), default=0)
        if max_len == 0:
            return [0.0] * len(pairs)

        def pad(seq, pad_id=0):
            return seq + [pad_id] * (max_len - len(seq))

        input_ids = [pad(s) for s in input_ids]
        attn = [pad(s) for s in attn]

        input_names = [i.name for i in sess.get_inputs()]
        feeds = {}
        if "input_ids" in input_names:
            feeds["input_ids"] = input_ids
        if "attention_mask" in input_names:
            feeds["attention_mask"] = attn
        if "token_type_ids" in input_names:
            feeds["token_type_ids"] = [[0] * max_len for _ in input_ids]

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
    except Exception:
        return [0.0] * len(pairs)


def is_reranker_available() -> bool:
    """Check if any reranker backend is available."""
    model = _get_reranker_model_name()
    if model and _check_fastembed_rerank():
        return True
    onnx_path = _get_reranker_onnx_path()
    tokenizer_path = _get_reranker_tokenizer_path()
    if onnx_path and tokenizer_path and _check_onnx():
        return True
    return False


def get_reranker_info() -> dict:
    """Get information about the configured reranker."""
    model = _get_reranker_model_name()
    if model and _check_fastembed_rerank():
        return {
            "backend": "fastembed",
            "model": model,
            "available": True,
        }
    onnx_path = _get_reranker_onnx_path()
    tokenizer_path = _get_reranker_tokenizer_path()
    if onnx_path and tokenizer_path and _check_onnx():
        return {
            "backend": "onnx",
            "onnx_path": onnx_path,
            "tokenizer_path": tokenizer_path,
            "available": True,
        }
    return {
        "backend": None,
        "available": False,
    }


