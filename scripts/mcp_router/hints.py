"""
mcp_router/hints.py - Query hint parsing and tool selection.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from .config import LANGS, HTTP_URL_INDEXER
from .client import tools_describe_cached
from .intent import _cosine, _embed_texts


def parse_repo_hints(q: str) -> Dict[str, Any]:
    """Extract light filters from the query: language, under, symbol, ext, path_glob, not_glob."""
    s = q.strip()
    low = s.lower()
    out: Dict[str, Any] = {}
    
    # language
    for lang in sorted(LANGS, key=len, reverse=True):
        if re.search(rf"\b{re.escape(lang)}\b", low):
            out["language"] = {"javascript": "js", "typescript": "ts", "c++": "cpp", "c#": "csharp"}.get(lang, lang)
            break
    
    # under / in folder
    m_under = re.search(r"\bunder\s+([\w./-]+)", low)
    m_in = re.search(r"\b(?:in|inside)\s+([\w./-]+)", low)
    m = m_under or m_in
    if m:
        cand = m.group(1)
        if len(cand) >= 2 and cand not in LANGS:
            out["under"] = cand
    
    # symbol-like tokens
    m2 = re.search(r"([A-Za-z_][A-Za-z0-9_]*\s*\(\))|([A-Za-z_][\w]*\.[A-Za-z_][\w]*)|([A-Za-z_][\w]*::[A-Za-z_][\w]*)", s)
    if m2:
        sym = m2.group(0)
        sym = re.sub(r"\s*\(\)\s*$", "", sym)
        out["symbol"] = sym
    
    # file extension
    m3 = re.search(r"\.(py|ts|tsx|js|jsx|go|java|rs|kt|rb|php|scala|swift)$", s)
    if m3:
        out["ext"] = m3.group(1)
    
    # glob inclusions
    globs: List[str] = []
    if re.search(r"\bonly\b", low):
        m_glob = re.search(r"\*\.[A-Za-z0-9]+", s)
        if m_glob:
            globs.append("**/" + m_glob.group(0))
        if "python" in low and "*.py" not in " ".join(globs):
            globs.append("**/*.py")
    if globs:
        out["path_glob"] = globs
    
    # exclusions
    not_glob: List[str] = []
    for ex in ["vendor", "node_modules", "dist", "build", "tests", "__pycache__"]:
        if re.search(rf"\bexclude\s+{re.escape(ex)}\b", low):
            not_glob.append(f"**/{ex}/**")
    if not_glob:
        out["not_glob"] = not_glob
    
    return out


def clean_query_and_dsl(q: str) -> Tuple[str, Dict[str, Any]]:
    """Strip DSL tokens from query and return (clean_query, dsl_filters)."""
    try:
        from scripts.hybrid_search import parse_query_dsl
        clean, extracted = parse_query_dsl([q])
        return (clean[0] if clean else ""), (extracted or {})
    except Exception:
        return q, {}


def _signature_text(t: dict) -> str:
    """Build signature text for tool similarity matching."""
    name = (t.get("name") or "").strip()
    desc = (t.get("description") or "").strip()
    params = []
    try:
        schema = t.get("inputSchema") or {}
        props = (schema.get("properties") or {}) if isinstance(schema, dict) else {}
        params = [k for k in props.keys()]
    except Exception:
        params = []
    ptxt = (" params:" + ",".join(params)) if params else ""
    return (name + "\n" + desc + ptxt).strip()


def select_best_search_tool_by_signature(q: str, tool_dict: dict[str, str], allow_network: bool = True) -> str | None:
    """Select best matching search tool based on signature similarity."""
    candidates = [n for n in tool_dict.keys() if n == "repo_search" or n.startswith("search_")]
    if not candidates:
        return None
    
    per_server: dict[str, list[dict]] = {}
    for base in set(tool_dict[t] for t in candidates):
        try:
            per_server[base] = tools_describe_cached(base, allow_network=allow_network)
        except Exception:
            per_server[base] = []
    
    sig_map: dict[str, str] = {}
    for tname in candidates:
        base = tool_dict.get(tname)
        descs = per_server.get(base, [])
        obj = None
        for td in descs:
            if (td.get("name") or "").strip() == tname:
                obj = td
                break
        sig_map[tname] = _signature_text(obj or {"name": tname, "description": ""})
    
    texts = [q] + [sig_map[n] for n in candidates]
    vecs = _embed_texts(texts)
    if not vecs or len(vecs) < 1 + len(candidates):
        return None
    
    qv = vecs[0]
    scores: list[tuple[str, float]] = []
    for i, name in enumerate(candidates):
        sv = vecs[1 + i]
        scores.append((name, _cosine(qv, sv)))
    scores.sort(key=lambda x: x[1], reverse=True)
    
    best, best_s = scores[0]
    repo_s = next((s for n, s in scores if n == "repo_search"), None)
    margin = 0.02
    if best == "repo_search" or repo_s is None:
        return best
    if best != "repo_search" and best_s >= (repo_s + margin):
        return best
    return "repo_search"
