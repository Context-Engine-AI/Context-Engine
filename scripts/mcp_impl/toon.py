#!/usr/bin/env python3
"""
mcp/toon.py - TOON (Token-Oriented Object Notation) support for MCP server.

Extracted from mcp_indexer_server.py for better modularity.
Contains:
- TOON format detection and enablement helpers
- Response formatting functions for TOON output
"""

from __future__ import annotations

__all__ = [
    "_is_toon_output_enabled",
    "_should_use_toon",
    "_format_results_as_toon",
    "_format_context_results_as_toon",
]

import logging
import os
from typing import Any, Dict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TOON feature flag helpers
# ---------------------------------------------------------------------------
def _is_toon_output_enabled() -> bool:
    """Check if TOON output format is enabled globally via TOON_ENABLED env var."""
    return os.environ.get("TOON_ENABLED", "0").lower() in ("1", "true", "yes")


def _should_use_toon(output_format: Any) -> bool:
    """Determine if TOON format should be used based on explicit param or env flag.
    
    Args:
        output_format: Explicit format request (e.g., "toon", "json", None)
        
    Returns:
        True if TOON format should be used
    """
    if output_format is not None:
        fmt = str(output_format).strip().lower()
        return fmt == "toon"
    return _is_toon_output_enabled()


# ---------------------------------------------------------------------------
# TOON response formatting
# ---------------------------------------------------------------------------
def _format_results_as_toon(response: Dict[str, Any], compact: bool = False) -> Dict[str, Any]:
    """Convert response to use TOON-formatted results string instead of JSON array.

    Replaces 'results' array with 'results' string in TOON format to save tokens.
    Always adds output_format marker when TOON is requested, even for empty results.
    
    Args:
        response: Search response dict with 'results' key
        compact: If True, use more compact TOON encoding
        
    Returns:
        Modified response with TOON-encoded results
    """
    try:
        from scripts.toon_encoder import encode_search_results

        results = response.get("results", [])
        if isinstance(results, list):
            # Replace JSON array with TOON string (handles empty arrays too)
            toon_results = encode_search_results(results, compact=compact)
            response["results"] = toon_results
        response["output_format"] = "toon"

        return response
    except ImportError:
        logger.warning("TOON encoder not available, returning JSON format")
        return response
    except Exception as e:
        logger.debug(f"TOON encoding failed: {e}")
        return response


def _format_context_results_as_toon(response: Dict[str, Any], compact: bool = False) -> Dict[str, Any]:
    """Convert context_search response to TOON format, handling mixed code/memory results.

    Uses encode_context_results which properly handles memory entries (content/score)
    vs code entries (path/line), avoiding blank rows or dropped content.
    
    Args:
        response: Context search response dict with 'results' key
        compact: If True, use more compact TOON encoding
        
    Returns:
        Modified response with TOON-encoded results
    """
    try:
        from scripts.toon_encoder import encode_context_results

        results = response.get("results", [])
        if isinstance(results, list):
            toon_results = encode_context_results(results, compact=compact)
            response["results"] = toon_results
        response["output_format"] = "toon"

        return response
    except ImportError:
        logger.warning("TOON encoder not available, returning JSON format")
        return response
    except Exception as e:
        logger.debug(f"TOON encoding failed: {e}")
        return response

