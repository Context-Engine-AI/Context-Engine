#!/usr/bin/env python3
"""
mcp/utils.py - Shared utility functions for MCP indexer server.

Extracted from mcp_indexer_server.py for better modularity.
Contains:
- Type coercion helpers (_coerce_bool, _coerce_int, _coerce_str)
- JSON/argument parsing helpers (_maybe_parse_jsonish, _parse_kv_string, etc.)
- String list normalization (_to_str_list_relaxed)
- Tokenization helpers (_split_ident, _tokens_from_queries)
"""

from __future__ import annotations

__all__ = [
    # Constants
    "_STOP",
    # Safe conversions
    "safe_int",
    "safe_float",
    "safe_bool",
    # Type coercion
    "_coerce_bool",
    "_coerce_int",
    "_coerce_str",
    "_coerce_value_string",
    # JSON/argument parsing
    "_maybe_parse_jsonish",
    "_looks_jsonish_string",
    "_parse_kv_string",
    "_extract_kwargs_payload",
    # String list normalization
    "_to_str_list_relaxed",
    # Tokenization
    "_split_ident",
    "_tokens_from_queries",
    # Environment helpers
    "_env_overrides",
    # Query helpers
    "_primary_identifier_from_queries",
]

import ast as _ast
import json
import logging
import re
import urllib.parse as _urlparse
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Stop words for tokenization
# ---------------------------------------------------------------------------
_STOP = {
    "the", "a", "an", "of", "in", "on", "for", "and", "or", "to",
    "with", "by", "is", "are", "be", "this", "that",
}


# ---------------------------------------------------------------------------
# Safe conversion helpers
# ---------------------------------------------------------------------------
def safe_int(value, default=0, logger=None, context=""):
    """Safely convert value to int with fallback."""
    try:
        if value is None or (isinstance(value, str) and value.strip() == ""):
            return default
        return int(value)
    except (ValueError, TypeError):
        return default


def safe_float(value, default=0.0, logger=None, context=""):
    """Safely convert value to float with fallback."""
    try:
        if value is None or (isinstance(value, str) and value.strip() == ""):
            return default
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_bool(value, default=False, logger=None, context=""):
    """Safely convert value to bool with fallback."""
    try:
        if value is None or (isinstance(value, str) and value.strip() == ""):
            return default
        if isinstance(value, bool):
            return value
        s = str(value).strip().lower()
        if s in {"1", "true", "yes", "on"}:
            return True
        if s in {"0", "false", "no", "off"}:
            return False
        return default
    except (ValueError, TypeError):
        return default


# ---------------------------------------------------------------------------
# Type coercion helpers
# ---------------------------------------------------------------------------
def _coerce_bool(x: Any, default: bool = False) -> bool:
    """Coerce value to boolean."""
    if isinstance(x, bool):
        return x
    if x is None:
        return default
    s = str(x).strip().lower()
    if s in {"1", "true", "yes", "on"}:
        return True
    if s in {"0", "false", "no", "off"}:
        return False
    return default


def _coerce_int(x: Any, default: int = 0) -> int:
    """Coerce value to integer."""
    try:
        if x is None or (isinstance(x, str) and x.strip() == ""):
            return default
        return int(x)
    except (ValueError, TypeError):
        return default


def _coerce_str(x: Any, default: str = "") -> str:
    """Coerce value to string."""
    if x is None:
        return default
    return str(x)


def _coerce_value_string(v: str) -> Any:
    """Coerce a string value, trying JSON then Python literal eval."""
    # Try JSON
    try:
        return json.loads(v)
    except json.JSONDecodeError:
        pass
    # Try Python literal (e.g., "['a','b']")
    try:
        return _ast.literal_eval(v)
    except (ValueError, SyntaxError):
        pass
    # As-is string
    return v


# ---------------------------------------------------------------------------
# JSON/argument parsing helpers
# ---------------------------------------------------------------------------
def _maybe_parse_jsonish(obj: Any) -> Optional[Dict[str, Any]]:
    """Attempt to parse object as JSON dict."""
    if isinstance(obj, dict):
        return obj
    if not isinstance(obj, str):
        return None
    s = obj.strip()
    if not s:
        return None
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    try:
        return json.loads("{" + s + "}")
    except json.JSONDecodeError:
        pass
    return None


def _looks_jsonish_string(s: Any) -> bool:
    """Check if string looks like JSON."""
    if not isinstance(s, str):
        return False
    t = s.strip()
    if not t:
        return False
    if t.startswith("{") and ":" in t:
        return True
    if t.endswith("}"):
        return True
    # quick heuristics for comma/colon pairs often seen when args are concatenated
    return ("," in t and ":" in t) or ('":' in t)


def _parse_kv_string(s: str) -> Dict[str, Any]:
    """Parse non-JSON strings like "a=1&b=2" or "query=[\"a\",\"b\"]" into a dict.
    
    Values are JSON-decoded when possible; else literal-eval; else kept as raw strings.
    """
    out: Dict[str, Any] = {}
    try:
        if not isinstance(s, str) or not s.strip():
            return out
        # Try query-string form first
        if ("=" in s) and ("{" not in s) and (":" not in s):
            qs = _urlparse.parse_qs(s, keep_blank_values=True)
            for k, vals in qs.items():
                v = vals[-1] if vals else ""
                out[k] = _coerce_value_string(v)
            return out
        # Fallback: split on commas for simple "k=v,k2=v2" forms
        if ("=" in s) and ("," in s):
            for part in s.split(","):
                if "=" not in part:
                    continue
                k, v = part.split("=", 1)
                out[k.strip()] = _coerce_value_string(v.strip())
            return out
    except Exception:
        return {}
    return out


def _extract_kwargs_payload(kwargs: Any) -> Dict[str, Any]:
    """Extract kwargs payload from potentially nested/stringified input."""
    try:
        # Handle kwargs being passed as a string "{}" by some MCP clients
        if isinstance(kwargs, str):
            parsed = _maybe_parse_jsonish(kwargs)
            if isinstance(parsed, dict):
                kwargs = parsed
            else:
                return {}

        if isinstance(kwargs, dict) and "kwargs" in kwargs:
            inner = kwargs.get("kwargs")
            if isinstance(inner, dict):
                return inner
            parsed = _maybe_parse_jsonish(inner)
            if isinstance(parsed, dict):
                return parsed
            # Fallback: accept query-string or k=v,k2=v2 strings
            if isinstance(inner, str):
                kv = _parse_kv_string(inner)
                if isinstance(kv, dict) and kv:
                    return kv
            return {}
    except Exception:
        return {}
    return {}


# ---------------------------------------------------------------------------
# String list normalization
# ---------------------------------------------------------------------------
def _to_str_list_relaxed(x: Any) -> List[str]:
    """Coerce various inputs to list[str]. Accepts JSON strings like "[\"a\",\"b\"]"."""
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        flat: List[str] = []
        for item in x:
            flat.extend(_to_str_list_relaxed(item))
        return [t for t in flat if t.strip()]
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []

        def _normalize_tokens(val: Any, depth: int = 0) -> List[str]:
            if depth > 10:
                text = str(val).strip()
                return [text] if text else []
            if isinstance(val, (list, tuple)):
                tokens: List[str] = []
                for item in val:
                    tokens.extend(_normalize_tokens(item, depth + 1))
                return tokens

            text = str(val).strip()
            if not text:
                return []

            seen: set[str] = set()
            current = text
            while True:
                if not current:
                    return []
                key = f"{depth}:{current}"
                if key in seen:
                    return [current]
                seen.add(key)

                if len(current) >= 2 and current[0] == current[-1] and current[0] in {'"', "'"}:
                    current = current[1:-1].strip()
                    continue

                changed = False
                if current.startswith('/"'):
                    current = current[2:].strip()
                    changed = True
                if current.endswith('"/'):
                    current = current[:-2].strip()
                    changed = True
                if current.endswith('/"'):
                    current = current[:-2].strip()
                    changed = True
                if changed:
                    continue

                parsed = None
                for parser in (json.loads, _ast.literal_eval):
                    try:
                        parsed = parser(current)
                    except Exception:
                        continue
                    else:
                        break
                if isinstance(parsed, (list, tuple)):
                    tokens: List[str] = []
                    for item in parsed:
                        tokens.extend(_normalize_tokens(item, depth + 1))
                    return tokens
                if isinstance(parsed, str):
                    current = parsed.strip()
                    continue
                if parsed is not None:
                    current = str(parsed).strip()
                    continue

                maybe = current.replace('\\"', '"').replace("\\'", "'")
                if maybe != current:
                    current = maybe.strip()
                    continue

                # Only split on commas if it looks like a simple list of identifiers,
                # NOT natural language prose. Heuristic: if any part has internal spaces
                # (multi-word), it's likely prose - don't split.
                if ',' in current:
                    parts = [p.strip() for p in current.split(',')]
                    has_prose = any(' ' in p for p in parts if p)
                    if not has_prose:
                        tokens: List[str] = []
                        for part in parts:
                            tokens.extend(_normalize_tokens(part, depth + 1))
                        return tokens

                return [current]

        return [t for t in _normalize_tokens(s) if t.strip()]
    return [str(x)]


# ---------------------------------------------------------------------------
# Tokenization helpers
# ---------------------------------------------------------------------------
def _split_ident(s: str) -> List[str]:
    """Split identifier into tokens (handles camelCase, snake_case, etc.)."""
    parts = re.split(r"[^A-Za-z0-9]+", s)
    out = []
    for p in parts:
        if not p:
            continue
        segs = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+", p)
        out.extend([x for x in segs if x])
    return [x.lower() for x in out if x and x.lower() not in _STOP]


def _tokens_from_queries(qs: List[str]) -> List[str]:
    """Extract unique tokens from a list of query strings."""
    toks = []
    for q in qs:
        toks.extend(_split_ident(q))
    seen = set()
    out = []
    for t in toks:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------
from contextlib import contextmanager
import os as _os


@contextmanager
def _env_overrides(pairs: Dict[str, str]):
    """Temporarily override environment variables."""
    old_vals = {}
    for k, v in pairs.items():
        old_vals[k] = _os.environ.get(k)
        if v is None:
            _os.environ.pop(k, None)
        else:
            _os.environ[k] = str(v)
    try:
        yield
    finally:
        for k, old_v in old_vals.items():
            if old_v is None:
                _os.environ.pop(k, None)
            else:
                _os.environ[k] = old_v


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------
def _primary_identifier_from_queries(qs: List[str]) -> str:
    """Best-effort extraction of the main CONSTANT_NAME or IDENTIFIER from queries.

    Catches ALL_CAPS, snake_case, camelCase, and lowercase identifiers.
    """
    try:
        cand: List[str] = []
        for q in qs:
            for t in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", q or ""):
                if len(t) < 2:
                    continue
                is_all_caps = t.isupper()
                has_underscore = "_" in t
                is_camel = any(c.isupper() for c in t[1:]) and any(c.islower() for c in t)
                is_longer_lower = t.islower() and len(t) >= 3
                if is_all_caps or has_underscore or is_camel or is_longer_lower:
                    cand.append(t)
        if not cand:
            return ""
        # Prefer stronger identifiers: ALL_CAPS > camelCase > snake_case > lowercase
        def _score(c: str):
            if c.isupper():
                return (3, len(c))
            if "_" in c:
                return (2, len(c))
            if any(ch.isupper() for ch in c[1:]):
                return (1, len(c))
            return (0, len(c))
        cand.sort(key=_score, reverse=True)
        return cand[0] if cand else ""
    except Exception:
        return ""

