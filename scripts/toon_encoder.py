"""
TOON (Token-Oriented Object Notation) encoder for Context-Engine.

Uses the official python-toon library for spec-compliant encoding.
Provides helper functions for search result formatting.

Feature flag:
- TOON_ENABLED=1  Enable TOON encoding (default: 0)

Reference: https://github.com/toon-format/toon
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

# Use official python-toon library
from toon import encode as toon_encode


# -----------------------------------------------------------------------------
# Feature Flag
# -----------------------------------------------------------------------------

def is_toon_enabled() -> bool:
    """Check if TOON encoding is enabled via environment variable."""
    return os.environ.get("TOON_ENABLED", "0").lower() in ("1", "true", "yes")


# -----------------------------------------------------------------------------
# Core Encoding (delegates to python-toon)
# -----------------------------------------------------------------------------

def encode(
    data: Any,
    delimiter: str = ",",
    include_length: bool = True,
) -> str:
    """Encode any JSON-compatible data to TOON format using official library.

    Args:
        data: JSON-compatible data (dict, list, or primitive)
        delimiter: Field delimiter (default: ",")
        include_length: Ignored (python-toon always includes length markers)

    Returns:
        TOON-formatted string
    """
    options = {"delimiter": delimiter}
    return toon_encode(data, options)


# For backwards compatibility
def encode_object(
    obj: Dict[str, Any],
    delimiter: str = ",",
    indent: int = 0,
    include_length: bool = True,
) -> List[str]:
    """Encode an object to TOON format (backwards compat wrapper)."""
    result = encode(obj, delimiter=delimiter)
    # Add indentation if needed
    if indent > 0:
        prefix = "  " * indent
        return [prefix + line for line in result.split("\n")]
    return result.split("\n")


def encode_tabular(
    key: str,
    arr: List[Dict[str, Any]],
    delimiter: str = ",",
    indent: int = 0,
    include_length: bool = True,
) -> List[str]:
    """Encode array as tabular (backwards compat wrapper)."""
    result = encode({key: arr}, delimiter=delimiter)
    if indent > 0:
        prefix = "  " * indent
        return [prefix + line for line in result.split("\n")]
    return result.split("\n")


def encode_simple_array(
    key: str,
    arr: List[Any],
    delimiter: str = ",",
    indent: int = 0,
    include_length: bool = True,
) -> List[str]:
    """Encode simple array (backwards compat wrapper)."""
    result = encode({key: arr}, delimiter=delimiter)
    if indent > 0:
        prefix = "  " * indent
        return [prefix + line for line in result.split("\n")]
    return result.split("\n")


# For backwards compatibility
def _looks_numeric(s: str) -> bool:
    """Check if string looks like a number (would be parsed as numeric by TOON)."""
    if not s:
        return False
    # Leading zeros like "05" (but not "0" alone)
    if len(s) > 1 and s[0] == '0' and s[1].isdigit():
        return True
    # Try parsing as int/float
    try:
        float(s)
        return True
    except ValueError:
        pass
    # Scientific notation variants
    if 'e' in s.lower():
        try:
            float(s)
            return True
        except ValueError:
            pass
    return False


def _encode_value(value: Any, delimiter: str) -> str:
    """Encode a single value (used by search result helpers)."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        # Check if quoting needed per TOON spec
        needs_quote = (
            not value or
            value[0].isspace() or value[-1].isspace() or
            value in ("true", "false", "null") or
            value == "-" or value.startswith("-") or
            _looks_numeric(value) or
            any(c in value for c in (':', '"', '\\', '[', ']', '{', '}', '\n', '\r', '\t')) or
            delimiter in value
        )
        if needs_quote:
            escaped = value.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
            return f'"{escaped}"'
        return value
    # Nested objects/arrays: compact JSON
    import json
    return json.dumps(value, separators=(",", ":"))


def _is_uniform_array_of_objects(arr: List[Any]) -> bool:
    """Check if array is uniform objects (same keys)."""
    if not arr or len(arr) < 1:
        return False
    if not all(isinstance(item, dict) for item in arr):
        return False
    first_keys = set(arr[0].keys())
    return all(set(item.keys()) == first_keys for item in arr)


def _bracket_segment(length: int, delimiter: str, include_length: bool) -> str:
    """Build bracket segment for search results."""
    if not include_length:
        return ""
    # python-toon uses [N,] format for comma delimiter
    if delimiter == ",":
        return f"[{length}]"
    elif delimiter == "\t":
        return f"[{length}\t]"
    elif delimiter == "|":
        return f"[{length}|]"
    return f"[{length}]"


# -----------------------------------------------------------------------------
# Search Results Formatting (Context-Engine specific)
# -----------------------------------------------------------------------------

def encode_search_results(
    results: List[Dict[str, Any]],
    delimiter: str = ",",
    include_length: bool = True,
    compact: bool = True,
) -> str:
    """Encode search results to TOON tabular format.

    Dynamically includes all fields present in results to avoid dropping data.
    Core fields are ordered first, then any additional fields alphabetically.

    Args:
        results: List of search result dicts
        delimiter: Field delimiter (default: ",")
        include_length: Include [N] markers (default: True)
        compact: If True, only include core location fields (path/lines)

    Returns:
        TOON-formatted search results
    """
    bracket = _bracket_segment(len(results), delimiter, include_length)

    if not results:
        # Empty array per spec: key[0]: (nothing after colon)
        return f"results{bracket}:"

    # Determine fields based on compact mode
    if compact:
        fields = ["path", "start_line", "end_line"]
    else:
        # Collect all unique fields from all results
        all_fields: set = set()
        for r in results:
            all_fields.update(r.keys())

        # Core fields ordered first, then remaining alphabetically
        core_order = ["path", "start_line", "end_line", "score", "symbol", "kind",
                      "snippet", "information", "relevance_score", "why"]
        fields = [f for f in core_order if f in all_fields]
        # Add remaining fields not in core_order
        extra_fields = sorted(all_fields - set(core_order))
        fields.extend(extra_fields)

    # Build tabular output with delimiter in fields segment too
    fields_part = "{" + delimiter.join(fields) + "}"

    lines = [f"results{bracket}{fields_part}:"]
    for r in results:
        values = [_encode_value(r.get(f), delimiter) for f in fields]
        lines.append(f"  {delimiter.join(values)}")

    return "\n".join(lines)


def encode_context_results(
    results: List[Dict[str, Any]],
    delimiter: str = ",",
    include_length: bool = True,
    compact: bool = True,
) -> str:
    """Encode context_search results (code + memory) to TOON format.

    Dynamically includes all fields present to avoid dropping data.
    Handles mixed result types with source-aware encoding.

    Args:
        results: List of mixed search result dicts
        delimiter: Field delimiter (default: ",")
        include_length: Include [N] markers (default: True)
        compact: If True, use minimal core fields only

    Returns:
        TOON-formatted context results with source-aware encoding
    """
    bracket = _bracket_segment(len(results), delimiter, include_length)

    if not results:
        # Empty per spec: key[0]:
        return f"results{bracket}:"

    # Separate code and memory results
    code_results = [r for r in results if r.get("source") != "memory"]
    memory_results = [r for r in results if r.get("source") == "memory"]

    lines = []

    # Encode code results
    if code_results:
        code_bracket = _bracket_segment(len(code_results), delimiter, include_length)
        if compact:
            code_fields = ["path", "start_line", "end_line"]
        else:
            # Collect all fields from code results
            all_code_fields: set = set()
            for r in code_results:
                all_code_fields.update(r.keys())
            # Core order, then extras
            core_order = ["path", "start_line", "end_line", "score", "symbol", "kind",
                          "snippet", "information", "relevance_score", "why"]
            code_fields = [f for f in core_order if f in all_code_fields]
            extra = sorted(all_code_fields - set(core_order) - {"source"})
            code_fields.extend(extra)
        code_fields_part = "{" + delimiter.join(code_fields) + "}"
        lines.append(f"code{code_bracket}{code_fields_part}:")
        for r in code_results:
            values = [_encode_value(r.get(f), delimiter) for f in code_fields]
            lines.append(f"  {delimiter.join(values)}")

    # Encode memory results
    if memory_results:
        mem_bracket = _bracket_segment(len(memory_results), delimiter, include_length)
        if compact:
            mem_fields = ["content", "score"]
        else:
            # Collect all fields from memory results
            all_mem_fields: set = set()
            for r in memory_results:
                all_mem_fields.update(r.keys())
            core_order = ["content", "score", "id", "information"]
            mem_fields = [f for f in core_order if f in all_mem_fields]
            extra = sorted(all_mem_fields - set(core_order) - {"source"})
            mem_fields.extend(extra)
        mem_fields_part = "{" + delimiter.join(mem_fields) + "}"
        lines.append(f"memory{mem_bracket}{mem_fields_part}:")
        for r in memory_results:
            values = [_encode_value(r.get(f), delimiter) for f in mem_fields]
            lines.append(f"  {delimiter.join(values)}")

    # If no separation needed (all same type or empty), use unified format
    if not lines:
        # Fallback: use generic encode
        return encode_search_results(results, delimiter, include_length, compact)

    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Token Estimation
# -----------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """Rough token estimate (chars/4 heuristic, good enough for comparison)."""
    return len(text) // 4


def compare_formats(data: Any) -> Dict[str, Any]:
    """Compare TOON vs JSON token counts for given data.

    Returns:
        {
            "json_tokens": int,
            "json_compact_tokens": int,
            "toon_tokens": int,
            "toon_tab_tokens": int,
            "savings_vs_json": float,  # percentage
            "savings_vs_compact": float,
        }
    """
    import json

    json_pretty = json.dumps(data, indent=2)
    json_compact = json.dumps(data, separators=(",", ":"))
    toon_comma = encode(data, delimiter=",")
    toon_tab = encode(data, delimiter="\t")

    json_tokens = estimate_tokens(json_pretty)
    json_compact_tokens = estimate_tokens(json_compact)
    toon_tokens = estimate_tokens(toon_comma)
    toon_tab_tokens = estimate_tokens(toon_tab)

    return {
        "json_tokens": json_tokens,
        "json_compact_tokens": json_compact_tokens,
        "toon_tokens": toon_tokens,
        "toon_tab_tokens": toon_tab_tokens,
        "savings_vs_json": round((1 - toon_tokens / json_tokens) * 100, 1) if json_tokens else 0,
        "savings_vs_compact": round((1 - toon_tokens / json_compact_tokens) * 100, 1) if json_compact_tokens else 0,
    }

