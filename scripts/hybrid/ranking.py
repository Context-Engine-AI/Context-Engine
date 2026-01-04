#!/usr/bin/env python3
"""
Ranking and scoring logic for hybrid search.

This module extracts the core ranking, scoring, and diversification functions
from hybrid_search.py for reuse and testing.
"""

__all__ = [
    "rrf", "_scale_rrf_k", "_adaptive_per_query", "_normalize_scores",
    "_sinkhorn_knopp", "_birkhoff_diversify", "_sinkhorn_query_doc_balance",
    "_soft_rrf_fusion",
    "sparse_lex_score", "lexical_score",
    "_compute_query_stats", "_adaptive_weights", "_bm25_token_weights_from_results",
    "_mmr_diversify", "_merge_and_budget_spans",
    "_detect_implementation_intent", "_IMPL_INTENT_PATTERNS",
    "_get_collection_stats", "_COLL_STATS_CACHE", "_COLL_STATS_TTL",
    "_get_symbol_extent", "ADAPTIVE_SPAN_SIZING",
]

import os
import re
import math
import logging
from typing import List, Dict, Any, Tuple, Optional

logger = logging.getLogger("hybrid_ranking")

# ---------------------------------------------------------------------------
# Helper: safe type coercion
# ---------------------------------------------------------------------------

def _safe_int(val: Any, default: int) -> int:
    try:
        if val is None or (isinstance(val, str) and val.strip() == ""):
            return default
        return int(val)
    except (ValueError, TypeError):
        return default


def _safe_float(val: Any, default: float) -> float:
    try:
        if val is None or (isinstance(val, str) and val.strip() == ""):
            return default
        return float(val)
    except (ValueError, TypeError):
        return default


# ---------------------------------------------------------------------------
# Configuration constants (loaded from environment with defaults)
# ---------------------------------------------------------------------------

RRF_K = _safe_int(os.environ.get("HYBRID_RRF_K", "30"), 30)
DENSE_WEIGHT = _safe_float(os.environ.get("HYBRID_DENSE_WEIGHT", "1.5"), 1.5)
LEXICAL_WEIGHT = _safe_float(os.environ.get("HYBRID_LEXICAL_WEIGHT", "0.20"), 0.20)
LEX_VECTOR_WEIGHT = _safe_float(
    os.environ.get("HYBRID_LEX_VECTOR_WEIGHT", str(LEXICAL_WEIGHT)), LEXICAL_WEIGHT
)
PSEUDO_BOOST = _safe_float(os.environ.get("HYBRID_PSEUDO_BOOST", "0.0"), 0.0)

# Large codebase scaling
LARGE_COLLECTION_THRESHOLD = _safe_int(os.environ.get("HYBRID_LARGE_THRESHOLD", "10000"), 10000)
MAX_RRF_K_SCALE = _safe_float(os.environ.get("HYBRID_MAX_RRF_K_SCALE", "3.0"), 3.0)
SCORE_NORMALIZE_ENABLED = os.environ.get("HYBRID_SCORE_NORMALIZE", "1").lower() in {"1", "true", "yes", "on"}

# Sparse lexical vector scoring
SPARSE_LEX_MAX_SCORE = _safe_float(os.environ.get("HYBRID_SPARSE_LEX_MAX", "15.0"), 15.0)
SPARSE_RRF_MAX = 1.0 / (RRF_K + 1)
SPARSE_RRF_MIN = 1.0 / (RRF_K + 50)

# Micro-span budgeting defaults
def _get_micro_defaults() -> Tuple[int, int, int, int]:
    """Return (max_spans, merge_lines, budget_tokens, tokens_per_line) based on runtime and micro chunk mode.

    Budget tokens floor is 5000 to ensure context_answer has enough context for quality answers.
    """
    micro_enabled = os.environ.get("INDEX_MICRO_CHUNKS", "1").strip().lower() in {"1", "true", "yes", "on"}
    try:
        from scripts.refrag_glm import detect_glm_runtime
        is_glm = detect_glm_runtime()
    except ImportError:
        is_glm = False
    if is_glm:
        if micro_enabled:
            return (24, 6, 8192, 32)
        else:
            return (12, 4, 6000, 32)
    else:
        # Non-GLM: still need reasonable budget for quality context_answer
        return (8, 4, 5000, 32)

_MICRO_DEFAULTS = _get_micro_defaults()
MICRO_OUT_MAX_SPANS = _safe_int(os.environ.get("MICRO_OUT_MAX_SPANS", str(_MICRO_DEFAULTS[0])), _MICRO_DEFAULTS[0])
MICRO_MERGE_LINES = _safe_int(os.environ.get("MICRO_MERGE_LINES", str(_MICRO_DEFAULTS[1])), _MICRO_DEFAULTS[1])
MICRO_BUDGET_TOKENS = _safe_int(os.environ.get("MICRO_BUDGET_TOKENS", str(_MICRO_DEFAULTS[2])), _MICRO_DEFAULTS[2])
MICRO_TOKENS_PER_LINE = _safe_int(os.environ.get("MICRO_TOKENS_PER_LINE", str(_MICRO_DEFAULTS[3])), _MICRO_DEFAULTS[3])

# Adaptive Span Sizing: expand spans to full symbol boundaries when budget permits
ADAPTIVE_SPAN_SIZING = os.environ.get("ADAPTIVE_SPAN_SIZING", "1").strip().lower() in {"1", "true", "yes", "on"}
# Internal limits (not configurable - just sensible defaults)
_ADAPTIVE_MAX_EXPAND_LINES = 80      # Max extra lines per expansion
_ADAPTIVE_MAX_BUDGET_PCT = 0.4       # Max % of budget a single expansion can use
_ADAPTIVE_MAX_EXPANDED = 3           # Max spans to expand

# Intent detection for implementation preference
INTENT_IMPL_BOOST = _safe_float(os.environ.get("HYBRID_INTENT_IMPL_BOOST", "0.15"), 0.15)

_IMPL_INTENT_PATTERNS = frozenset({
    "implementation", "how does", "how is", "where is", "code for",
    "function that", "method that", "class that", "implements",
    "defined", "definition", "source", "logic", "algorithm",
    "where", "find", "locate", "show me", "actual code",
})

# ---------------------------------------------------------------------------
# Collection stats cache (for large collection scaling)
# ---------------------------------------------------------------------------

_COLL_STATS_CACHE: Dict[str, Tuple[float, Dict[str, Any]]] = {}
_COLL_STATS_TTL = 300  # 5 minutes


def _get_collection_stats(client: Any, coll_name: str) -> Dict[str, Any]:
    """Get cached collection statistics for scaling decisions."""
    import time
    now = time.time()
    cached = _COLL_STATS_CACHE.get(coll_name)
    if cached and (now - cached[0]) < _COLL_STATS_TTL:
        return cached[1]
    try:
        info = client.get_collection(coll_name)
        stats = {"points_count": info.points_count or 0}
        _COLL_STATS_CACHE[coll_name] = (now, stats)
        return stats
    except Exception:
        return {"points_count": 0}


# ---------------------------------------------------------------------------
# RRF (Reciprocal Rank Fusion)
# ---------------------------------------------------------------------------

def rrf(rank: int, k: int = RRF_K) -> float:
    """Reciprocal Rank Fusion score for a given rank."""
    return 1.0 / (k + rank)


def _scale_rrf_k(base_k: int, collection_size: int) -> int:
    """Scale RRF k parameter based on collection size.

    For large collections, increase k to spread score distribution.
    Uses logarithmic scaling: k_scaled = k * (1 + log10(size/threshold))
    Capped at MAX_RRF_K_SCALE * base_k.
    """
    if collection_size < LARGE_COLLECTION_THRESHOLD:
        return base_k
    ratio = collection_size / LARGE_COLLECTION_THRESHOLD
    scale = 1.0 + math.log10(max(1, ratio))
    scale = min(scale, MAX_RRF_K_SCALE)
    return int(base_k * scale)


def _adaptive_per_query(base_limit: int, collection_size: int, has_filters: bool) -> int:
    """Increase candidate retrieval for larger collections.

    Uses sublinear sqrt scaling to avoid excessive retrieval.
    Filters reduce the need for extra candidates.
    """
    if collection_size < LARGE_COLLECTION_THRESHOLD:
        return base_limit
    ratio = collection_size / LARGE_COLLECTION_THRESHOLD
    scale = math.sqrt(ratio)
    if has_filters:
        scale = max(1.0, scale * 0.7)
    scaled = int(base_limit * min(scale, 3.0))
    return max(base_limit, min(scaled, 200))


def _normalize_scores(score_map: Dict[str, Dict[str, Any]], collection_size: int) -> None:
    """Normalize scores using z-score + sigmoid for large collections.

    This spreads compressed score distributions to improve discrimination.
    Only applies when SCORE_NORMALIZE_ENABLED=true and collection is large.
    """
    if not SCORE_NORMALIZE_ENABLED:
        return
    if collection_size < LARGE_COLLECTION_THRESHOLD:
        return
    if len(score_map) < 3:
        return

    scores = [rec["s"] for rec in score_map.values()]
    mean_s = sum(scores) / len(scores)
    var_s = sum((s - mean_s) ** 2 for s in scores) / len(scores)
    std_s = math.sqrt(var_s) if var_s > 0 else 1.0

    if std_s < 1e-6:
        return

    for rec in score_map.values():
        z = (rec["s"] - mean_s) / std_s
        normalized = 1.0 / (1.0 + math.exp(-z * 0.5))
        rec["s"] = normalized


# ---------------------------------------------------------------------------
# Sinkhorn-Knopp normalization (prototype)
# ---------------------------------------------------------------------------

def _sinkhorn_knopp(
    matrix: List[List[float]],
    iters: int = 5,
    min_sum: float = 1e-6,
) -> List[List[float]]:
    """Approximate Sinkhorn-Knopp balancing for non-negative matrices."""
    if not matrix:
        return []
    rows = len(matrix)
    cols = len(matrix[0]) if rows else 0
    if cols == 0:
        return [list(row) for row in matrix]

    balanced = [[float(v) for v in row] for row in matrix]
    iters = max(1, int(iters or 1))
    for _ in range(iters):
        for i in range(rows):
            row = balanced[i]
            s = sum(row)
            if s > min_sum:
                inv = 1.0 / s
                for j in range(cols):
                    row[j] *= inv
        col_sums = [0.0] * cols
        for j in range(cols):
            col_sums[j] = sum(balanced[i][j] for i in range(rows))
        for j in range(cols):
            s = col_sums[j]
            if s > min_sum:
                inv = 1.0 / s
                for i in range(rows):
                    balanced[i][j] *= inv
    return balanced


# ---------------------------------------------------------------------------
# Sparse lexical scoring
# ---------------------------------------------------------------------------

def sparse_lex_score(raw_score: float, weight: float = LEX_VECTOR_WEIGHT) -> float:
    """Normalize sparse lexical vector score to RRF-equivalent range.

    Maps sparse similarity scores to the same range as RRF(rank) scores,
    preserving relative ordering while maintaining fusion balance.

    Formula: weight * (RRF_MIN + (clamped_score / max_score) * (RRF_MAX - RRF_MIN))
    - Sparse score 0 maps to RRF_MIN (like worst rank)
    - Sparse score max maps to RRF_MAX (like rank 1)
    - Quality ordering preserved, but doesn't dominate dense embedding scores
    """
    if raw_score <= 0:
        return 0.0
    clamped = min(raw_score, SPARSE_LEX_MAX_SCORE)
    ratio = clamped / SPARSE_LEX_MAX_SCORE
    rrf_equiv = SPARSE_RRF_MIN + ratio * (SPARSE_RRF_MAX - SPARSE_RRF_MIN)
    return weight * rrf_equiv


# ---------------------------------------------------------------------------
# Tokenization helpers
# ---------------------------------------------------------------------------

_STOP = {
    "the", "a", "an", "of", "in", "on", "for", "and", "or", "to",
    "with", "by", "is", "are", "be", "this", "that",
}


def _split_ident(s: str) -> List[str]:
    """Split snake_case and camelCase identifiers into tokens."""
    parts = re.split(r"[^A-Za-z0-9]+", s)
    out: List[str] = []
    for p in parts:
        if not p:
            continue
        segs = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+", p)
        out.extend([x for x in segs if x])
    return [x.lower() for x in out if x and x.lower() not in _STOP]


def tokenize_queries(phrases: List[str]) -> List[str]:
    """Tokenize phrases into unique identifier-aware tokens."""
    toks: List[str] = []
    for ph in phrases:
        toks.extend(_split_ident(ph))
    seen = set()
    out: List[str] = []
    for t in toks:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out


# ---------------------------------------------------------------------------
# Lexical scoring
# ---------------------------------------------------------------------------

def lexical_score(
    phrases: List[str],
    md: Dict[str, Any],
    token_weights: Dict[str, float] | None = None,
    bm25_weight: float | None = None
) -> float:
    """Smarter lexical: split identifiers, weight matches in symbol/path higher.
    
    If token_weights provided, apply a small BM25-style multiplicative factor per token:
        factor = 1 + bm25_weight * (w - 1) where w are normalized around 1.0
    """
    tokens = tokenize_queries(phrases)
    if not tokens:
        return 0.0
    path = str(md.get("path", "")).lower()
    path_segs = re.split(r"[/\\]", path)
    sym = str(md.get("symbol", "")).lower()
    symp = str(md.get("symbol_path", "")).lower()
    code = str(md.get("code", ""))[:2000].lower()
    pseudo = str(md.get("pseudo") or "").lower()
    tags_val = md.get("tags") or []
    if isinstance(tags_val, list):
        tags_text = " ".join(str(x) for x in tags_val).lower()
    else:
        tags_text = str(tags_val).lower()
    s = 0.0
    for t in tokens:
        if not t:
            continue
        contrib = 0.0
        if t in sym or t in symp:
            contrib += 2.0
        if any(t in seg for seg in path_segs):
            contrib += 0.8
            if path_segs and t in path_segs[-1]:
                contrib += 0.3
        if t in code:
            contrib += 1.0
        if PSEUDO_BOOST > 0.0:
            if pseudo and t in pseudo:
                contrib += PSEUDO_BOOST
            if tags_text and t in tags_text:
                contrib += 0.5 * PSEUDO_BOOST
        if contrib > 0 and token_weights and bm25_weight:
            w = float(token_weights.get(t, 1.0) or 1.0)
            contrib *= (1.0 + float(bm25_weight) * (w - 1.0))
        s += contrib
    return s


# ---------------------------------------------------------------------------
# Adaptive weights
# ---------------------------------------------------------------------------

def _compute_query_stats(queries: List[str]) -> Dict[str, Any]:
    """Compute statistics about query tokens for adaptive weighting."""
    toks = tokenize_queries(queries)
    total = len(toks)

    def _is_camel(t: str) -> bool:
        try:
            return any(c.isupper() for c in t[1:]) and any(c.islower() for c in t)
        except Exception:
            return False

    def _is_identifier_like(t: str) -> bool:
        try:
            return ("_" in t) or t.isupper() or any(ch.isdigit() for ch in t) or _is_camel(t)
        except Exception:
            return False

    id_like = sum(1 for t in toks if _is_identifier_like(t))
    avg_tok_len = (sum(len(t) for t in toks) / max(1, total)) if total else 0.0
    qchars = sum(len(q) for q in queries) if queries else 0
    has_question = any(("?" in q) for q in (queries or []))
    q0 = (queries[0].strip().lower() if queries else "")
    wh_start = q0.startswith(("how", "what", "why", "when", "where", "explain", "describe"))
    stats = {
        "total_tokens": total,
        "identifier_density": (id_like / max(1, total)),
        "avg_token_len": avg_tok_len,
        "avg_query_chars": (qchars / max(1, len(queries))) if queries else 0.0,
        "narrative_hint": bool(has_question or wh_start),
    }
    return stats


def _adaptive_weights(stats: Dict[str, Any]) -> Tuple[float, float, float]:
    """Return per-query weights (dense_w, lex_vec_w, lex_text_w) with gentle clamps.
    
    Dense/lex-vector vary within ±25%; lexical text component within ±20%.
    """
    base_d = DENSE_WEIGHT
    base_lv = LEX_VECTOR_WEIGHT
    base_lx = LEXICAL_WEIGHT

    id_density = float(stats.get("identifier_density", 0.0) or 0.0)
    total = int(stats.get("total_tokens", 0) or 0)
    narrative_hint = 1.0 if stats.get("narrative_hint") else 0.0
    longish = 1.0 if total >= 8 else 0.0

    narrative_score = 0.6 * narrative_hint + 0.4 * longish
    id_score = id_density
    delta = max(-1.0, min(1.0, narrative_score - id_score))

    dens_scale = 1.0 + 0.25 * delta
    lv_scale = 1.0 - 0.25 * delta
    lx_scale = 1.0 + 0.20 * (-delta)

    dens_scale = max(0.75, min(1.25, dens_scale))
    lv_scale = max(0.75, min(1.25, lv_scale))
    lx_scale = max(0.80, min(1.20, lx_scale))

    return base_d * dens_scale, base_lv * lv_scale, base_lx * lx_scale


def _bm25_token_weights_from_results(phrases: List[str], results: List[Any]) -> Dict[str, float]:
    """Compute lightweight per-token IDF-like weights from a small sample of lex results.
    
    Returns weights normalized to mean 1.0 over tokens present in phrases.
    """
    try:
        tokens = [t for t in tokenize_queries(phrases) if t]
        if not tokens or not results:
            return {}
        tok_set = set(tokens)
        N = max(1, len(results))
        df: Dict[str, int] = {t: 0 for t in tok_set}
        for p in results:
            try:
                md = (p.payload or {}).get("metadata") or {}
            except Exception:
                md = {}
            text = " ".join([
                str(md.get("symbol") or ""),
                str(md.get("symbol_path") or ""),
                str(md.get("path") or ""),
                str((md.get("code") or ""))[:2000],
            ]).lower()
            doc_toks = set(tokenize_queries([text]))
            for t in tok_set:
                if t in doc_toks:
                    df[t] += 1
        idf: Dict[str, float] = {t: math.log(1.0 + (N / float(df[t] + 1))) for t in tok_set}
        mean = sum(idf.values()) / max(1, len(idf))
        if mean <= 0:
            return {t: 1.0 for t in tok_set}
        return {t: (idf[t] / mean) for t in tok_set}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# MMR diversification
# ---------------------------------------------------------------------------

def _mmr_diversify(ranked: List[Dict[str, Any]], k: int = 60, lambda_: float = 0.7) -> List[Dict[str, Any]]:
    """Maximal Marginal Relevance over fused list.
    
    Preserves top-1 by relevance, then balances relevance vs. diversity by path/symbol.
    Returns a reordered list (top-k diversified, remainder appended in original order).
    """
    if not ranked:
        return []
    k = max(1, min(int(k or 1), len(ranked)))

    def _path(md: Dict[str, Any]) -> str:
        return str(md.get("path") or "")

    def _symp(md: Dict[str, Any]) -> str:
        return str(md.get("symbol_path") or md.get("symbol") or "")

    def _sim(a: Dict[str, Any], b: Dict[str, Any]) -> float:
        mda = (a["pt"].payload or {}).get("metadata") or {}
        mdb = (b["pt"].payload or {}).get("metadata") or {}
        pa, pb = _path(mda), _path(mdb)
        if pa and pb and pa == pb:
            return 1.0
        sa, sb = _symp(mda), _symp(mdb)
        if sa and sb and sa == sb:
            return 0.8
        if pa and pb:
            ta = set(re.split(r"[/\\]+", pa.lower()))
            tb = set(re.split(r"[/\\]+", pb.lower()))
            ta.discard("")
            tb.discard("")
            if ta and tb:
                inter = len(ta & tb)
                union = max(1, len(ta | tb))
                return 0.5 * (inter / union)
        return 0.0

    rel = [float(m.get("s", 0.0)) for m in ranked]
    selected_idx = [0]
    candidates = list(range(1, len(ranked)))
    while len(selected_idx) < k and candidates:
        best_idx = None
        best_score = -1e18
        for i in candidates:
            r = rel[i]
            if selected_idx:
                max_sim = max(_sim(ranked[i], ranked[j]) for j in selected_idx)
            else:
                max_sim = 0.0
            mmr = lambda_ * r - (1.0 - lambda_) * max_sim
            if mmr > best_score:
                best_score = mmr
                best_idx = i
        selected_idx.append(best_idx)
        candidates.remove(best_idx)

    sel_set = set(selected_idx)
    diversified = [ranked[i] for i in selected_idx]
    diversified.extend([ranked[i] for i in range(len(ranked)) if i not in sel_set])
    return diversified


def _birkhoff_diversify(
    ranked: List[Dict[str, Any]],
    k: int = 60,
    coverage_weight: float = 0.5,
    iters: int = 5,
) -> List[Dict[str, Any]]:
    """
    Birkhoff polytope diversification via doubly stochastic mixing.

    mHC-inspired alternative to MMR: uses Sinkhorn-Knopp to create a doubly
    stochastic doc×path affinity matrix, ensuring coverage across paths/symbols.

    The Birkhoff polytope view interprets the result as a convex combination
    of permutations, giving stronger theoretical coverage guarantees.

    Args:
        ranked: List of scored result dicts with 'pt' payload and 's' score
        k: Number of results to diversify (top-k)
        coverage_weight: Balance between relevance (0) and coverage (1)
        iters: Sinkhorn iterations for normalization

    Returns:
        Reordered list with diversified top-k, remainder appended
    """
    if not ranked or len(ranked) < 2:
        return ranked
    k = max(1, min(int(k or 1), len(ranked)))

    # Extract path groups for coverage analysis
    def _get_group(rec: Dict[str, Any]) -> str:
        md = (rec.get("pt") and rec["pt"].payload or {}).get("metadata") or {}
        path = str(md.get("path") or "")
        # Use directory as group for better coverage
        if "/" in path:
            return "/".join(path.split("/")[:-1])
        return path

    groups: List[str] = []
    group_set: Dict[str, int] = {}
    for rec in ranked:
        g = _get_group(rec)
        if g not in group_set:
            group_set[g] = len(groups)
            groups.append(g)

    n_docs = len(ranked)
    n_groups = len(groups)

    if n_groups < 2:
        # No diversity possible, return by relevance
        return ranked

    # Build doc×group affinity matrix
    # Each doc has affinity 1.0 to its group, 0 to others
    # We add relevance scores to bias toward high-scoring docs
    rel_scores = [float(m.get("s", 0.0)) for m in ranked]
    max_rel = max(rel_scores) if rel_scores else 1.0
    if max_rel <= 0:
        max_rel = 1.0

    affinity: List[List[float]] = []
    for i, rec in enumerate(ranked):
        g = _get_group(rec)
        gidx = group_set.get(g, 0)
        row = [0.0] * n_groups
        # Base affinity to own group
        row[gidx] = 1.0
        # Add normalized relevance to bias selection
        rel_norm = rel_scores[i] / max_rel
        row[gidx] += (1.0 - coverage_weight) * rel_norm
        affinity.append(row)

    # Apply Sinkhorn to get doubly stochastic matrix
    # This ensures each group gets "equal coverage mass"
    balanced = _sinkhorn_knopp(affinity, iters=iters, min_sum=1e-8)

    # Score each doc by its balanced row sum (coverage contribution)
    # Higher means this doc contributes more to overall coverage
    coverage_scores = [sum(row) for row in balanced]

    # Combine relevance and coverage for final ranking
    combined = []
    for i in range(n_docs):
        rel_norm = rel_scores[i] / max_rel
        cov_norm = coverage_scores[i] / max(coverage_scores) if max(coverage_scores) > 0 else 0
        score = (1.0 - coverage_weight) * rel_norm + coverage_weight * cov_norm
        combined.append((score, i))

    # Sort by combined score descending
    combined.sort(key=lambda x: -x[0])

    # Take top-k diversified, append rest in original order
    selected_idx = [idx for _, idx in combined[:k]]
    sel_set = set(selected_idx)
    diversified = [ranked[i] for i in selected_idx]
    diversified.extend([ranked[i] for i in range(n_docs) if i not in sel_set])
    return diversified


def _sinkhorn_query_doc_balance(
    query_doc_scores: List[List[float]],
    min_sum: float = 1e-6,
) -> List[List[float]]:
    """
    Normalize query×doc affinity by equalizing per-query contribution.

    For PRF/expansion with multiple queries, this ensures:
    - Each query contributes equal total mass (row sums normalized)

    Args:
        query_doc_scores: Matrix where [q][d] = affinity of query q to doc d
                          Rows are queries, columns are documents.
        min_sum: Epsilon for numerical stability

    Returns:
        Balanced matrix with same shape
    """
    if not query_doc_scores:
        return []
    rows = len(query_doc_scores)
    cols = len(query_doc_scores[0]) if rows else 0
    if cols == 0:
        return [list(row) for row in query_doc_scores]

    balanced = [[float(v) for v in row] for row in query_doc_scores]
    row_sums = [sum(row) for row in balanced]
    total = sum(row_sums)
    if total <= min_sum or rows <= 0:
        return balanced

    target_row_sum = total / rows
    for i in range(rows):
        s = row_sums[i]
        if s > min_sum:
            scale = target_row_sum / s
            row = balanced[i]
            for j in range(cols):
                row[j] *= scale
    return balanced


def _soft_rrf_fusion(
    score_components: Dict[str, Dict[str, float]],
    signal_weights: Optional[Dict[str, float]] = None,
    rrf_k: int = 60,
    boost_balance: float = 0.3,
) -> Dict[str, float]:
    """
    Weighted Rank-based Fusion with boost balancing.

    Addresses key issues:
    1. Preserves signal weights (dense/lexical dominate, boosts are secondary)
    2. Uses rank-based RRF for scale invariance
    3. Only balances boost signals among themselves, not against primaries
    4. Ignores signals with no variance (all zeros)

    Args:
        score_components: {doc_id: {signal_name: score}}
        signal_weights: {signal_name: weight}. Defaults to standard weights.
        rrf_k: RRF constant (higher = smoother ranking)
        boost_balance: How much to balance boost signals (0=none, 1=full equalization)

    Returns:
        {doc_id: fused_score}
    """
    if not score_components:
        return {}

    # Default weights: primary signals dominate
    if signal_weights is None:
        signal_weights = {
            "d": 1.0,      # Dense - primary
            "lx": 0.8,     # Lexical - primary
            "sym_sub": 0.15,  # Symbol substring - boost
            "sym_eq": 0.2,    # Symbol exact - boost
            "core": 0.1,      # Core file - boost
            "langb": 0.05,    # Language - boost
            "rec": 0.1,       # Recency - boost
            "impl": 0.1,      # Implementation - boost
        }

    doc_ids = list(score_components.keys())
    n_docs = len(doc_ids)
    if n_docs < 2:
        return {d: sum(score_components[d].values()) for d in doc_ids}

    # Collect signals present in data
    all_signals: set = set()
    for comps in score_components.values():
        all_signals.update(comps.keys())

    # Separate primary and boost signals
    primary_signals = {"d", "lx"}
    boost_signals = all_signals - primary_signals

    # Step 1: Compute rank-based RRF scores per signal
    signal_ranks: Dict[str, Dict[str, int]] = {}
    for sig in all_signals:
        # Get scores for this signal
        sig_scores = [(doc_id, score_components[doc_id].get(sig, 0.0)) for doc_id in doc_ids]
        # Check variance - skip if all same value
        vals = [s for _, s in sig_scores]
        if max(vals) - min(vals) < 1e-9:
            continue  # No signal, skip
        # Sort by score descending, assign ranks
        sig_scores.sort(key=lambda x: -x[1])
        signal_ranks[sig] = {doc_id: rank + 1 for rank, (doc_id, _) in enumerate(sig_scores)}

    if not signal_ranks:
        return {d: sum(score_components[d].values()) for d in doc_ids}

    # Step 2: Compute weighted RRF for primary signals
    primary_scores: Dict[str, float] = {d: 0.0 for d in doc_ids}
    for sig in primary_signals & set(signal_ranks.keys()):
        weight = signal_weights.get(sig, 0.5)
        ranks = signal_ranks[sig]
        for doc_id in doc_ids:
            rank = ranks.get(doc_id, n_docs)
            primary_scores[doc_id] += weight / (rrf_k + rank)

    # Step 3: Compute balanced boost contribution
    active_boosts = boost_signals & set(signal_ranks.keys())
    boost_scores: Dict[str, float] = {d: 0.0 for d in doc_ids}

    if active_boosts:
        n_boosts = len(active_boosts)
        for sig in active_boosts:
            weight = signal_weights.get(sig, 0.1)
            ranks = signal_ranks[sig]
            for doc_id in doc_ids:
                rank = ranks.get(doc_id, n_docs)
                boost_scores[doc_id] += weight / (rrf_k + rank)

        # Optional: balance boost contributions (equalize per-signal impact)
        if boost_balance > 0 and n_boosts > 1:
            # Compute per-signal contributions
            per_signal_contrib: Dict[str, Dict[str, float]] = {}
            for sig in active_boosts:
                weight = signal_weights.get(sig, 0.1)
                ranks = signal_ranks[sig]
                per_signal_contrib[sig] = {
                    doc_id: weight / (rrf_k + ranks.get(doc_id, n_docs))
                    for doc_id in doc_ids
                }

            # Normalize each signal to sum to 1, then weight equally
            balanced_boost: Dict[str, float] = {d: 0.0 for d in doc_ids}
            total_weight = sum(signal_weights.get(s, 0.1) for s in active_boosts)
            for sig in active_boosts:
                sig_total = sum(per_signal_contrib[sig].values())
                if sig_total > 1e-12:
                    sig_weight = signal_weights.get(sig, 0.1) / max(total_weight, 1e-6)
                    for doc_id in doc_ids:
                        balanced_boost[doc_id] += (per_signal_contrib[sig][doc_id] / sig_total) * sig_weight

            # Scale balanced back to original magnitude
            orig_total = sum(boost_scores.values())
            balanced_total = sum(balanced_boost.values())
            if balanced_total > 1e-12:
                scale = orig_total / balanced_total
                for doc_id in doc_ids:
                    balanced_boost[doc_id] *= scale

            # Blend original and balanced
            for doc_id in doc_ids:
                boost_scores[doc_id] = (1.0 - boost_balance) * boost_scores[doc_id] + boost_balance * balanced_boost[doc_id]

    # Step 4: Combine primary + boost
    result = {}
    for doc_id in doc_ids:
        result[doc_id] = primary_scores[doc_id] + boost_scores[doc_id]

    return result


# ---------------------------------------------------------------------------
# Adaptive Span Sizing: Symbol extent lookup
# ---------------------------------------------------------------------------

# Cache for symbol extents: {(collection, path, symbol): (start, end)}
_SYMBOL_EXTENT_CACHE: Dict[Tuple[str, str, str], Tuple[int, int]] = {}
_SYMBOL_EXTENT_CACHE_MAX = 500
_SYMBOL_EXTENT_CLIENT: Any = None


def _get_symbol_extent(
    path: str,
    symbol: str,
    collection: str = "",
    qdrant_client: Any = None,
) -> Tuple[int, int]:
    """Query Qdrant for the full extent of a symbol (function/class).

    Returns (start_line, end_line) for the complete symbol, or (0, 0) if not found.
    Uses cached lookups to minimize Qdrant queries.
    """
    if not symbol or not path:
        return (0, 0)

    cache_key = (collection, path, symbol)
    if cache_key in _SYMBOL_EXTENT_CACHE:
        return _SYMBOL_EXTENT_CACHE[cache_key]

    # Lazy import to avoid circular dependencies
    try:
        from qdrant_client import QdrantClient, models
    except ImportError:
        return (0, 0)

    if not collection:
        collection = os.environ.get("COLLECTION_NAME", "")
    if not collection:
        return (0, 0)

    try:
        global _SYMBOL_EXTENT_CLIENT
        if qdrant_client is None:
            # Reuse a single client instance to avoid repeated connection setup.
            if _SYMBOL_EXTENT_CLIENT is None:
                qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
                try:
                    timeout_s = float(os.environ.get("ADAPTIVE_SPAN_QDRANT_TIMEOUT", "1.0") or 1.0)
                except Exception:
                    timeout_s = 1.0
                _SYMBOL_EXTENT_CLIENT = QdrantClient(
                    url=qdrant_url,
                    api_key=os.environ.get("QDRANT_API_KEY"),
                    timeout=timeout_s,
                )
            qdrant_client = _SYMBOL_EXTENT_CLIENT

        # Query for all chunks with same path and symbol identifier.
        # Prefer symbol_path when provided, but also accept plain symbol for compatibility.
        filt = models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.path",
                    match=models.MatchValue(value=path),
                ),
            ],
            should=[
                models.FieldCondition(
                    key="metadata.symbol_path",
                    match=models.MatchValue(value=symbol),
                ),
                models.FieldCondition(
                    key="metadata.symbol",
                    match=models.MatchValue(value=symbol),
                ),
            ],
        )

        # Bound the amount of scrolling work; micro-chunking can produce many points.
        try:
            per_page = int(os.environ.get("ADAPTIVE_SPAN_SCROLL_LIMIT", "128") or 128)
        except Exception:
            per_page = 128
        per_page = max(16, min(per_page, 512))
        max_total = max(64, min(per_page * 4, 1024))

        points: list[Any] = []
        offset = None
        while len(points) < max_total:
            batch, offset = qdrant_client.scroll(
                collection_name=collection,
                scroll_filter=filt,
                with_payload=True,
                limit=min(per_page, max_total - len(points)),
                offset=offset,
            )
            if not batch:
                break
            points.extend(batch)

        if not points:
            _SYMBOL_EXTENT_CACHE[cache_key] = (0, 0)
            return (0, 0)

        # Find min start_line and max end_line across all chunks
        min_start = float("inf")
        max_end = 0
        for pt in points:
            md = (pt.payload or {}).get("metadata") or {}
            start = int(md.get("start_line") or 0)
            end = int(md.get("end_line") or 0)
            if start > 0 and end > 0:
                min_start = min(min_start, start)
                max_end = max(max_end, end)

        if min_start == float("inf") or max_end == 0:
            _SYMBOL_EXTENT_CACHE[cache_key] = (0, 0)
            return (0, 0)

        result = (int(min_start), int(max_end))

        # Cache management: evict oldest entries if cache is full
        if len(_SYMBOL_EXTENT_CACHE) >= _SYMBOL_EXTENT_CACHE_MAX:
            # Simple FIFO eviction - remove first 100 entries
            keys_to_remove = list(_SYMBOL_EXTENT_CACHE.keys())[:100]
            for k in keys_to_remove:
                _SYMBOL_EXTENT_CACHE.pop(k, None)

        _SYMBOL_EXTENT_CACHE[cache_key] = result
        return result

    except Exception as e:
        if os.environ.get("DEBUG_ADAPTIVE_SPAN"):
            logger.debug(f"Symbol extent lookup failed: {e}")
        return (0, 0)


# ---------------------------------------------------------------------------
# Micro-span budgeting
# ---------------------------------------------------------------------------

def _merge_and_budget_spans(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Given ranked items with metadata path/start_line/end_line, merge nearby spans
    per path and enforce a token budget using a simple tokens-per-line estimate.

    When ADAPTIVE_SPAN_SIZING is enabled, attempts to expand spans to full symbol
    boundaries (functions/classes) if the expanded span fits within budget.

    Returns a filtered/merged list preserving score order as much as possible.
    """
    try:
        merge_lines = int(os.environ.get("MICRO_MERGE_LINES", str(MICRO_MERGE_LINES)) or MICRO_MERGE_LINES)
    except (ValueError, TypeError):
        merge_lines = MICRO_MERGE_LINES
    try:
        budget_tokens = int(os.environ.get("MICRO_BUDGET_TOKENS", str(MICRO_BUDGET_TOKENS)) or MICRO_BUDGET_TOKENS)
    except (ValueError, TypeError):
        budget_tokens = MICRO_BUDGET_TOKENS
    try:
        tokens_per_line = int(os.environ.get("MICRO_TOKENS_PER_LINE", str(MICRO_TOKENS_PER_LINE)) or MICRO_TOKENS_PER_LINE)
    except (ValueError, TypeError):
        tokens_per_line = MICRO_TOKENS_PER_LINE
    try:
        out_max_spans = int(os.environ.get("MICRO_OUT_MAX_SPANS", str(MICRO_OUT_MAX_SPANS)) or MICRO_OUT_MAX_SPANS)
    except (ValueError, TypeError):
        out_max_spans = MICRO_OUT_MAX_SPANS

    # Adaptive span sizing - just use the hardcoded internal limits
    adaptive_enabled = ADAPTIVE_SPAN_SIZING
    expanded_count = 0
    collection = os.environ.get("COLLECTION_NAME", "")

    clusters: Dict[str, List[Dict[str, Any]]] = {}
    for m in items:
        md = {}
        symbol = ""
        try:
            if isinstance(m, dict):
                if m.get("path") or m.get("start_line") or m.get("end_line"):
                    md = {
                        "path": m.get("path"),
                        "start_line": m.get("start_line"),
                        "end_line": m.get("end_line"),
                        "symbol": m.get("symbol", ""),
                    }
                else:
                    pt = m.get("pt", {})
                    if hasattr(pt, "payload") and getattr(pt, "payload"):
                        md = (pt.payload or {}).get("metadata") or {}
                # Prefer symbol_path for uniqueness, but fall back to plain symbol.
                rels = m.get("relations") if isinstance(m.get("relations"), dict) else {}
                symbol = str(
                    m.get("symbol")
                    or (rels or {}).get("symbol_path")
                    or (md or {}).get("symbol_path")
                    or (md or {}).get("symbol")
                    or ""
                )
        except Exception:
            md = {}
        path = str((md or {}).get("path") or "")
        start_line = int((md or {}).get("start_line") or 0)
        end_line = int((md or {}).get("end_line") or 0)
        if not path or start_line <= 0 or end_line <= 0:
            continue
        lst = clusters.setdefault(path, [])
        merged = False
        item_score = float(m.get("raw_score") or m.get("score") or m.get("s") or 0.0)
        for c in lst:
            if (
                start_line <= c["end"] + merge_lines
                and end_line >= c["start"] - merge_lines
            ):
                cluster_score = float(c["m"].get("raw_score") or c["m"].get("score") or c["m"].get("s") or 0.0)
                if item_score > cluster_score:
                    c["m"] = m
                c["start"] = min(c["start"], start_line)
                c["end"] = max(c["end"], end_line)
                # Keep track of symbol if available
                if symbol and not c.get("symbol"):
                    c["symbol"] = symbol
                merged = True
                break
        if not merged:
            lst.append({"start": start_line, "end": end_line, "m": m, "p": path, "symbol": symbol})

    budget = budget_tokens
    out: List[Dict[str, Any]] = []
    per_path_counts: Dict[str, int] = {}

    def _line_tokens(s: int, e: int) -> int:
        return max(1, (e - s + 1) * tokens_per_line)

    flattened = []
    for lst in clusters.values():
        for c in lst:
            flattened.append(c)

    def _flat_key(c):
        m = c.get("m", {})
        path = str(c.get("p") or "")
        start = int(c.get("start") or 0)
        score = float(m.get("raw_score") or m.get("score") or m.get("s") or 0.0)
        if score < 0:
            return (score, path, start)
        else:
            return (-score, path, start)

    flattened.sort(key=_flat_key)

    for c in flattened:
        m = c["m"]
        path = str(c.get("p") or "")
        symbol = str(c.get("symbol") or "")
        if per_path_counts.get(path, 0) >= out_max_spans:
            continue

        span_start = c["start"]
        span_end = c["end"]
        expanded = False

        # Adaptive Span Sizing: try to expand to full symbol boundaries
        if adaptive_enabled and symbol and collection and expanded_count < _ADAPTIVE_MAX_EXPANDED:
            try:
                sym_start, sym_end = _get_symbol_extent(path, symbol, collection)
                if sym_start > 0 and sym_end > 0:
                    expand_lines = (sym_end - sym_start + 1) - (span_end - span_start + 1)
                    if 0 < expand_lines <= _ADAPTIVE_MAX_EXPAND_LINES:
                        expanded_need = _line_tokens(sym_start, sym_end)
                        max_for_single = int(budget * _ADAPTIVE_MAX_BUDGET_PCT)
                        if expanded_need <= budget and expanded_need <= max_for_single:
                            span_start = sym_start
                            span_end = sym_end
                            expanded = True
                            expanded_count += 1
                            if os.environ.get("DEBUG_ADAPTIVE_SPAN"):
                                logger.debug(f"Expanded '{symbol}' [{c['start']}-{c['end']}] -> [{sym_start}-{sym_end}]")
            except Exception as e:
                if os.environ.get("DEBUG_ADAPTIVE_SPAN"):
                    logger.debug(f"Span expansion failed: {e}")

        need = _line_tokens(span_start, span_end)
        if need <= budget:
            budget -= need
            per_path_counts[path] = per_path_counts.get(path, 0) + 1
            out.append({
                "m": m, "start": span_start, "end": span_end,
                "need_tokens": need, "_expanded": expanded,
            })
        elif budget > 0 and per_path_counts.get(path, 0) < out_max_spans:
            affordable_lines = max(1, budget // tokens_per_line)
            trim_end = span_start + affordable_lines - 1
            if trim_end >= span_start:
                trimmed_need = _line_tokens(span_start, trim_end)
                budget -= trimmed_need
                per_path_counts[path] = per_path_counts.get(path, 0) + 1
                out.append({"m": m, "start": span_start, "end": trim_end, "need_tokens": trimmed_need, "_trimmed": True})
        if budget <= 0:
            break

    result: List[Dict[str, Any]] = []
    for c in out:
        m = c["m"]
        m["start_line"] = c["start"]
        m["end_line"] = c["end"]
        m["text"] = ""
        m["_merged_start"] = c["start"]
        m["_merged_end"] = c["end"]
        m["_budget_tokens"] = c["need_tokens"]
        if c.get("_expanded"):
            m["_adaptive_expanded"] = True
        result.append(m)
    return result


# ---------------------------------------------------------------------------
# Intent detection
# ---------------------------------------------------------------------------

def _detect_implementation_intent(queries: List[str]) -> bool:
    """Detect if query signals user wants implementation code."""
    if not queries:
        return False
    joined = " ".join(queries).lower()
    for pattern in _IMPL_INTENT_PATTERNS:
        if pattern in joined:
            return True
    return False
