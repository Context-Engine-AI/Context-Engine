"""
mcp_router/intent.py - Intent classification (rules + ML).
"""
from __future__ import annotations

import json
import os
import re
import sys
from typing import Any, Dict, List

# Intent constants
INTENT_ANSWER = "answer"
INTENT_SEARCH = "search"
INTENT_SEARCH_TESTS = "search_tests"
INTENT_SEARCH_CONFIG = "search_config"
INTENT_SEARCH_CALLERS = "search_callers"
INTENT_SEARCH_IMPORTERS = "search_importers"
INTENT_MEMORY_STORE = "memory_store"
INTENT_MEMORY_FIND = "memory_find"
INTENT_SYMBOL_GRAPH = "symbol_graph"
INTENT_INDEX = "index"
INTENT_PRUNE = "prune"
INTENT_STATUS = "status"
INTENT_LIST = "list"

# Debug state
_LAST_INTENT_DEBUG: Dict[str, Any] = {}


def get_last_intent_debug() -> Dict[str, Any]:
    """Get the last intent debug info."""
    return _LAST_INTENT_DEBUG


def _classify_intent_rules(q: str) -> str | None:
    s = q.lower()
    # Admin / maintenance first
    if any(w in s for w in ["reindex", "reset", "recreate", "index now", "fresh index"]):
        return INTENT_INDEX
    if any(w in s for w in ["prune", "pruning", "cleanup", "clean up"]):
        return INTENT_PRUNE
    if any(w in s for w in ["status", "health", "points", "stats"]):
        return INTENT_STATUS
    if any(w in s for w in ["list collections", "collections", "list qdrant"]):
        return INTENT_LIST

    # Search importers - check BEFORE tests to avoid "import" in test queries
    if any(w in s for w in ["import", "imports", "importers", "who imports", "imports this", "importing modules", "files that import"]):
        # Make sure it's not about "important" or similar
        if not any(w in s for w in ["important", "importance"]):
            return INTENT_SEARCH_IMPORTERS

    # Intent wrappers
    if any(w in s for w in ["tests", "pytest", "unit test", "test file", "where are tests"]):
        return INTENT_SEARCH_TESTS
    
    # Memory intents - be more specific to avoid false positives on "memory store implementation"
    # Check for actual user-intent memory storage, not code references
    memory_store_triggers = [
        "remember this", "save memory", "store memory", "remember that", 
        "save preference", "remember preference", "store a note", "save a note", "remember note"
    ]
    # IMPORTANT: "memory store" as a phrase often refers to code, not user intent
    if any(w in s for w in memory_store_triggers):
        # Exclude if it looks like a code search (has "implementation", "code", "function", etc)
        if not any(exc in s for exc in ["implementation", "code", "function", "class", "module", "file", "search for"]):
            return INTENT_MEMORY_STORE
    if any(w in s for w in [
        "find memory", "recall", "retrieve memory", "memory search", "what did we save",
        "recall notes", "find notes", "retrieve notes"
    ]):
        return INTENT_MEMORY_FIND

    # Symbol graph for callers - check BEFORE config to avoid false positives
    if any(w in s for w in ["who calls", "callers of", "call sites", "function calls"]):
        return INTENT_SYMBOL_GRAPH
    if re.search(r"calls?\s+(the\s+)?\w+\s*(function|method)?", s):
        return INTENT_SYMBOL_GRAPH

    # Config search - after callers check
    if any(w in s for w in ["config", "yaml", "toml", "ini", "settings file", "configuration"]):
        return INTENT_SEARCH_CONFIG
    
    # Fallback callers intent (used by search_callers_for)
    if any(w in s for w in ["used by", "usage sites", "references this function"]):
        return INTENT_SEARCH_CALLERS

    # Q&A-like prompts
    if re.match(r"^(what|how|why|explain|describe|summarize)(\b|\s)", s):
        return INTENT_ANSWER
    if any(w in s for w in ["recap", "design doc", "architecture", "adr", "retrospective", "postmortem", "summary of", "summarize the design"]):
        return INTENT_ANSWER
    return None


def _intent_prototypes() -> Dict[str, List[str]]:
    return {
        INTENT_ANSWER: [
            "explain, describe, summarize, recap, design, architecture, ADR, why/how",
            "summarize design decisions and architecture rationale",
        ],
        INTENT_SEARCH: [
            "find code references, search repository, locate files, find implementation",
            "code search in repo, general lookup, search for implementation",
            "find module, search function, locate class definition",
            "search for memory store implementation",  # Explicit example
        ],
        INTENT_MEMORY_STORE: [
            "remember this preference, save this note for later, store this memory",
            "save my preference, remember that for next time",
            # NOT: search for, find, implementation, code
        ],
        INTENT_MEMORY_FIND: [
            "what did we save, recall saved notes, retrieve memory, find my saved notes",
        ],
        INTENT_SEARCH_TESTS: [
            "find unit tests, test files, pytest, testing modules",
        ],
        INTENT_SEARCH_CONFIG: [
            "config files, configuration changes, yaml toml ini settings",
        ],
        INTENT_SEARCH_CALLERS: [
            "who calls this function, callers, usage sites, where is it used",
        ],
        INTENT_SEARCH_IMPORTERS: [
            "who imports this module, importers, importing modules, files that import",
        ],
        INTENT_SYMBOL_GRAPH: [
            "who calls this function, callers of, call graph, symbol callers",
        ],
    }


def _cosine(a: list[float], b: list[float]) -> float:
    """Lightweight cosine similarity."""
    try:
        s = 0.0
        na = 0.0
        nb = 0.0
        for i in range(min(len(a), len(b))):
            va = float(a[i])
            vb = float(b[i])
            s += va * vb
            na += va * va
            nb += vb * vb
        na = (na or 1.0) ** 0.5
        nb = (nb or 1.0) ** 0.5
        return s / (na * nb)
    except Exception:
        return 0.0


def _embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed texts using available embedding model."""
    if not texts:
        return []

    # Try centralized embedder factory first
    try:
        from scripts.embedder import get_embedding_model
        model_name = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
        em = get_embedding_model(model_name)
        raw = list(em.embed(texts))
        return [v.tolist() if hasattr(v, "tolist") else list(v) for v in raw]
    except ImportError:
        pass

    # Try fastembed directly
    try:
        from fastembed import TextEmbedding
        model_name = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
        em = TextEmbedding(model_name=model_name)
        raw = list(em.embed(texts))
        return [v.tolist() if hasattr(v, "tolist") else list(v) for v in raw]
    except Exception:
        pass

    # Fallback to lexical
    try:
        from scripts.utils import lex_hash_vector_text
        return [lex_hash_vector_text(t, dim=4096) for t in texts]
    except Exception:
        return [[float(len(t))] for t in texts]


def _classify_intent_ml(q: str) -> str:
    global _LAST_INTENT_DEBUG
    protos = _intent_prototypes()
    labels = list(protos.keys())
    texts = [q] + ["\n".join(protos[l]) for l in labels]
    vecs = _embed_texts(texts)
    if not vecs or len(vecs) < len(texts):
        _LAST_INTENT_DEBUG = {
            "strategy": "ml",
            "intent": INTENT_SEARCH,
            "confidence": 0.0,
            "query": q,
            "top_candidate": INTENT_SEARCH,
            "top_score": 0.0,
            "threshold": 0.25,
            "candidates": [],
            "reason": "embed_failed",
        }
        return INTENT_SEARCH
    qv = vecs[0]
    sims = []
    for i, lab in enumerate(labels):
        sims.append((lab, _cosine(qv, vecs[1 + i])))
    sims.sort(key=lambda x: x[1], reverse=True)
    top, score = sims[0]
    picked = top if score >= 0.25 else INTENT_SEARCH
    _LAST_INTENT_DEBUG = {
        "strategy": "ml",
        "intent": picked,
        "confidence": float(score),
        "query": q,
        "top_candidate": top,
        "top_score": float(score),
        "threshold": 0.25,
        "candidates": [(name, float(val)) for name, val in sims[:5]],
        "fallback": picked == INTENT_SEARCH and top != INTENT_SEARCH,
    }
    return picked


def classify_intent(q: str) -> str:
    """Classify user query into an intent."""
    global _LAST_INTENT_DEBUG
    ruled = _classify_intent_rules(q)
    if ruled is not None:
        _LAST_INTENT_DEBUG = {
            "strategy": "rules",
            "intent": ruled,
            "confidence": 1.0,
            "query": q,
        }
        return ruled
    picked = _classify_intent_ml(q)
    try:
        if os.environ.get("DEBUG_ROUTER") and isinstance(_LAST_INTENT_DEBUG, dict):
            if _LAST_INTENT_DEBUG.get("fallback"):
                print(json.dumps({"router": {"intent_fallback": _LAST_INTENT_DEBUG}}), file=sys.stderr)
    except Exception:
        pass
    return picked
