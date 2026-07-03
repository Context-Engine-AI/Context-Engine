#!/usr/bin/env python3
"""
ingest/vectors.py - Vector generation utilities for lexical and mini vectors.

This module provides functions for creating lexical hash vectors, sparse vectors,
and mini (projection) vectors used in the hybrid search system.
"""
from __future__ import annotations

import os
import re
import hashlib
from typing import List, Dict, Any

from scripts.ingest.config import (
    LEX_VECTOR_DIM,
    MINI_VEC_DIM,
    _STOP,
)

# ---------------------------------------------------------------------------
# Mini vector projection cache
# ---------------------------------------------------------------------------
_MINI_PROJ_CACHE: dict[tuple[int, int, int], list[list[float]]] = {}


def _get_mini_proj(
    in_dim: int, out_dim: int, seed: int | None = None
) -> list[list[float]]:
    """Get or create a random projection matrix for mini vectors."""
    import math
    import random

    s = int(os.environ.get("MINI_VEC_SEED", "1337")) if seed is None else int(seed)
    key = (in_dim, out_dim, s)
    M = _MINI_PROJ_CACHE.get(key)
    if M is None:
        rnd = random.Random(s)
        scale = 1.0 / math.sqrt(out_dim)
        # Dense Rademacher matrix (+/-1) scaled; good enough for fast gating
        M = [
            [scale * (1.0 if rnd.random() < 0.5 else -1.0) for _ in range(out_dim)]
            for _ in range(in_dim)
        ]
        _MINI_PROJ_CACHE[key] = M
    return M


def project_mini(vec: list[float], out_dim: int | None = None) -> list[float]:
    """Project a dense vector to a compact mini vector using random projection."""
    if not vec:
        return [0.0] * (int(out_dim or MINI_VEC_DIM))
    od = int(out_dim or MINI_VEC_DIM)
    M = _get_mini_proj(len(vec), od)
    out = [0.0] * od
    # y = x @ M
    for i, val in enumerate(vec):
        if val == 0.0:
            continue
        row = M[i]
        for j in range(od):
            out[j] += val * row[j]
    # L2 normalize to keep scale consistent
    norm = (sum(x * x for x in out) or 0.0) ** 0.5 or 1.0
    return [x / norm for x in out]


def _split_ident_lex(s: str) -> List[str]:
    """Split identifier into tokens (snake_case and camelCase aware)."""
    parts = re.split(r"[^A-Za-z0-9]+", s)
    out: List[str] = []
    for p in parts:
        if not p:
            continue
        segs = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+", p)
        out.extend([x for x in segs if x])
    return [x.lower() for x in out if x and x.lower() not in _STOP]


def _lex_hash_vector(text: str, dim: int = LEX_VECTOR_DIM) -> list[float]:
    """Create a lexical hash vector from text using hashing trick."""
    if not text:
        return [0.0] * dim
    vec = [0.0] * dim
    # Tokenize identifiers & words
    toks = _split_ident_lex(text)
    if not toks:
        return vec
    for t in toks:
        h = int(hashlib.md5(t.encode("utf-8", errors="ignore")).hexdigest()[:8], 16)
        idx = h % dim
        vec[idx] += 1.0
    # L2 normalize (avoid huge magnitudes)
    import math
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


# ---------------------------------------------------------------------------
# Pattern vector extraction (structural similarity)
# ---------------------------------------------------------------------------
_PATTERN_EXTRACTOR = None
_PATTERN_ENCODER = None


def _get_pattern_tools():
    """Lazy load pattern extraction tools."""
    global _PATTERN_EXTRACTOR, _PATTERN_ENCODER
    if _PATTERN_EXTRACTOR is None:
        try:
            from scripts.pattern_detection import PatternExtractor, PatternEncoder
            _PATTERN_EXTRACTOR = PatternExtractor()
            _PATTERN_ENCODER = PatternEncoder()
        except ImportError:
            pass
    return _PATTERN_EXTRACTOR, _PATTERN_ENCODER


def extract_pattern_vector(code: str, language: str) -> list[float] | None:
    """Extract structural pattern vector from code.

    Returns 64-dim pattern vector or None if pattern detection unavailable.
    """
    extractor, encoder = _get_pattern_tools()
    if extractor is None or encoder is None:
        return None
    try:
        signature = extractor.extract(code, language)
        return encoder.encode(signature)
    except Exception:
        return None
