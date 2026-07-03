#!/usr/bin/env python3
"""
mcp/query_expand.py - Query expansion helpers for MCP indexer server.

Extracted from mcp_indexer_server.py for better modularity.
Contains:
- _expand_query_impl: Main implementation (called by thin @mcp.tool() wrapper)
- _qe_* helper functions for expand_query

Note: The @mcp.tool() decorated expand_query function remains in mcp_indexer_server.py
as a thin wrapper that calls _expand_query_impl.
"""

from __future__ import annotations

__all__ = [
    "_expand_query_impl",
]

import os
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Imports from sibling modules
# ---------------------------------------------------------------------------
from scripts.mcp_impl.utils import _coerce_int, _to_str_list_relaxed

async def _expand_query_impl(query: Any = None, max_new: Any = None, session: Optional[str] = None) -> Dict[str, Any]:
    """LLM-assisted query expansion (local llama.cpp, if enabled).

    When to use:
    - Generate 1–2 compact alternates before repo_search/context_answer

    Parameters:
    - query: str or list[str]
    - max_new: int in [0,5] (default 3)

    Returns:
    - {"alternates": list[str]} or {"alternates": [], "hint": "..."} if decoder disabled
    """
    # NOTE: Do not print to stdout/stderr in MCP stdio mode; it can corrupt the JSON-RPC stream.
    # Use logger at DEBUG level when explicitly enabled.
    debug_expand = str(os.environ.get("DEBUG_EXPAND", "")).strip().lower() in ("1", "true", "yes")
    if debug_expand:
        logger.debug("expand_query called", extra={"query": repr(query), "max_new": repr(max_new)})
    try:
        qlist: list[str] = []
        if isinstance(query, (list, tuple)):
            qlist = [str(x) for x in query if str(x).strip()]
        elif query is not None:
            qlist = [str(query)] if str(query).strip() else []
        cap = 3
        if max_new not in (None, ""):
            try:
                cap = max(0, min(5, int(max_new)))
            except (ValueError, TypeError):
                cap = 3

        if not qlist or cap <= 0:
            return {
                "ok": True,
                "original_query": qlist[0] if qlist else "",
                "alternates": [],
                "total_queries": 1 if qlist else 0,
                "decoder_used": "none",
            }
        original_q = qlist[0] if qlist else ""
        # Select decoder runtime: explicit REFRAG_RUNTIME takes priority,
        # otherwise fall back to llamacpp (local).
        # GLM requires explicit REFRAG_RUNTIME=glm to avoid accidental API calls.
        runtime_kind = str(os.environ.get("REFRAG_RUNTIME", "")).strip().lower()
        if not runtime_kind:
            # Default to llamacpp when no runtime is specified
            runtime_kind = "llamacpp"

        # Only llama.cpp expansion requires the local decoder to be enabled.
        if runtime_kind == "llamacpp":
            from scripts.refrag_llamacpp import is_decoder_enabled  # type: ignore

            if not is_decoder_enabled():
                return {
                    "ok": True,
                    "original_query": original_q,
                    "alternates": [],
                    "total_queries": 1,
                    "decoder_used": "none",
                    "hint": "decoder disabled: set REFRAG_DECODER=1 to enable local llamacpp expansion",
                }

        # Build prompt per runtime - each model needs different prompting style
        extra_kwargs = {}
        use_force_json = False
        stop_tokens = ["\n\n"]
        if runtime_kind == "openai":
            from scripts.refrag_openai import OpenAIRefragClient  # type: ignore
            client = OpenAIRefragClient()
            # Use object schema {"queries": [...]} for OpenAI's response_format: json_object
            extra_kwargs["system"] = f'You rewrite code search queries. Output a JSON object with a "queries" key containing exactly {cap} alternative queries. Use different terminology - do NOT repeat any words from the original.'
            use_force_json = True
            prompt = f'Rewrite "{original_q}" using completely different technical terms. Return {{"queries": [...]}}:'
        elif runtime_kind == "minimax":
            from scripts.refrag_minimax import MiniMaxRefragClient  # type: ignore
            client = MiniMaxRefragClient()
            extra_kwargs["system"] = f"You rewrite code search queries. Output a JSON array with exactly {cap} alternative queries. Use different terminology - do NOT repeat any words from the original."
            prompt = f'Rewrite "{original_q}" using completely different technical terms:'
        elif runtime_kind == "glm":
            from scripts.refrag_glm import GLMRefragClient  # type: ignore
            client = GLMRefragClient()
            # no_thinking=True: use configured GLM_MODEL but skip thinking for fast response
            # All GLM models (4.5, 4.6, 4.7) support thinking: {type: "disabled"}
            extra_kwargs["no_thinking"] = True
            # Use object schema {"queries": [...]} for GLM's response_format: json_object
            use_force_json = True
            prompt = (
                f'For code search "{original_q}", suggest {cap} alternative queries focusing on:\n'
                f'- Library/package names (e.g., "nltk wordnet", "gensim vectors")\n'
                f'- Specific algorithms/techniques used internally\n'
                f'- Data structures or patterns\n\n'
                f'Output JSON object {{"queries": [...]}} with {cap} implementation-focused queries:'
            )
        else:
            from scripts.refrag_llamacpp import LlamaCppRefragClient  # type: ignore
            client = LlamaCppRefragClient()
            # llama.cpp / Granite: structured prompt with stop tokens
            stop_tokens = ["\n\n", "```", "]"]  # Stop after first array closes
            prompt = (
                f'Rewrite this code search query using different technical terms.\n'
                f'Do NOT repeat words from the original. Use alternative concepts.\n'
                f'Original: {original_q}\n\n'
                f'Output {cap} alternatives as JSON array:\n['
            )

        out = client.generate_with_soft_embeddings(
            prompt=prompt,
            max_tokens=int(os.environ.get("EXPAND_MAX_TOKENS", "256") or 256),
            # Some creativity needed for diverse expansions
            temperature=float(os.environ.get("EXPAND_TEMPERATURE", "0.7") or 0.7),
            top_k=int(os.environ.get("EXPAND_TOP_K", "40") or 40),
            top_p=float(os.environ.get("EXPAND_TOP_P", "0.9") or 0.9),
            stop=stop_tokens,
            force_json=use_force_json,
            **extra_kwargs,
        )
        import json as _json
        import re as _re

        if debug_expand:
            logger.debug("expand_query raw output", extra={"runtime": runtime_kind, "out": repr(out)})

        alts: list[str] = []
        seen = {q.strip() for q in qlist if isinstance(q, str)}
        max_len = int(os.environ.get("EXPAND_MAX_CHARS", "240") or 240)

        def _maybe_add(s: str) -> None:
            ss = (s or "").strip()
            if not ss:
                return
            # Trim extreme outputs and collapse whitespace
            ss = _re.sub(r"\s+", " ", ss)
            if max_len and len(ss) > max_len:
                ss = ss[:max_len].rstrip()
            if ss in seen:
                return
            seen.add(ss)
            alts.append(ss)

        # For llama.cpp, prompt ends with '[' so we need to prepend it and append ']'
        if runtime_kind == "llamacpp" and out:
            out = out.strip()
            if not out.startswith("["):
                out = "[" + out
            if not out.endswith("]"):
                out = out + "]"
        # Strip markdown code blocks if present
        out = _re.sub(r'^```(?:json)?\s*', '', out.strip())
        out = _re.sub(r'\s*```$', '', out)

        def _extract_strings_from_parsed(parsed: Any) -> None:
            """Extract strings from parsed JSON - handles list or dict with common keys."""
            nonlocal alts
            # If it's a list, iterate directly
            if isinstance(parsed, list):
                for s in parsed:
                    if isinstance(s, str):
                        _maybe_add(s)
                        if len(alts) >= cap:
                            return
            # If it's a dict, look for common keys that contain the array
            elif isinstance(parsed, dict):
                for key in ["answer", "alternatives", "queries", "results", "data"]:
                    if key in parsed and isinstance(parsed[key], list):
                        for s in parsed[key]:
                            if isinstance(s, str):
                                _maybe_add(s)
                                if len(alts) >= cap:
                                    return
                        return  # Found and processed a key
                # Last resort: try any list value in the dict
                for v in parsed.values():
                    if isinstance(v, list):
                        for s in v:
                            if isinstance(s, str):
                                _maybe_add(s)
                                if len(alts) >= cap:
                                    return
                        return  # Used first list found

        try:
            # First try direct JSON parse
            parsed = _json.loads(out)
            _extract_strings_from_parsed(parsed)
        except Exception as parse_err:
            logger.debug(f"expand_query direct parse failed: {parse_err}")
            # Fallback: try Python literal eval for single-quoted lists
            try:
                import ast
                parsed = ast.literal_eval(out)
                if isinstance(parsed, list):
                    for s in parsed:
                        if isinstance(s, str):
                            _maybe_add(s)
                            if len(alts) >= cap:
                                break
            except Exception:
                pass
            # Fallback: extract JSON array from text (model may prepend text like "Alternates:\n")
            if not alts:
                try:
                    # Find first [ and last ] to extract the array
                    match = _re.search(r'\[[\s\S]*\]', out)
                    if match:
                        arr_text = match.group(0)
                        logger.debug(f"expand_query found array: {repr(arr_text)}")
                        # Try JSON first, then Python literal eval
                        try:
                            parsed = _json.loads(arr_text)
                        except Exception:
                            import ast
                            parsed = ast.literal_eval(arr_text)
                        if isinstance(parsed, list):
                            for s in parsed:
                                if isinstance(s, str):
                                    _maybe_add(s)
                                    if len(alts) >= cap:
                                        break
                    else:
                        logger.debug("expand_query no array match found")
                        # Fallback: extract quoted strings from numbered lists
                        # Pattern matches: 1. "text" or - "text" or * "text"
                        quoted_matches = _re.findall(r'(?:^|\n)\s*(?:\d+\.|\-|\*)\s*["\']([^"\']+)["\']', out)
                        if quoted_matches:
                            logger.debug(f"expand_query found quoted strings: {quoted_matches}")
                            for s in quoted_matches:
                                _maybe_add(s)
                                if len(alts) >= cap:
                                    break
                except Exception as fallback_err:
                    logger.debug(f"expand_query fallback parse failed: {fallback_err}")
        if debug_expand:
            logger.debug("expand_query returning alts", extra={"alts": alts})
        capped = alts[:cap]
        return {
            "ok": True,
            "original_query": qlist[0] if qlist else "",
            "alternates": capped,
            "total_queries": 1 + len(capped),
            "decoder_used": runtime_kind,
        }
    except Exception as e:
        fallback_alts: list[str] = []
        for q in qlist:
            q = q.strip()
            if not q:
                continue
            for suffix in (" implementation", " usage", " example", " test"):
                cand = f"{q}{suffix}"
                if cand not in qlist and cand not in fallback_alts:
                    fallback_alts.append(cand)
                    if len(fallback_alts) >= cap:
                        break
            if len(fallback_alts) >= cap:
                break
        if fallback_alts:
            return {
                "ok": True,
                "original_query": qlist[0] if qlist else "",
                "alternates": fallback_alts[:cap],
                "total_queries": 1 + len(fallback_alts[:cap]),
                "decoder_used": "fallback",
                "hint": f"decoder error: {e}",
            }
        return {
            "ok": False,
            "original_query": qlist[0] if qlist else "",
            "alternates": [],
            "total_queries": 1,
            "decoder_used": "none",
            "error": str(e),
        }
