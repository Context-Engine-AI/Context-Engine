"""
Utility functions for parsing LLM responses.

These helpers handle common quirks like markdown code fences in JSON responses.
"""
from __future__ import annotations


def strip_markdown_fences(text: str) -> str:
    """Strip markdown code fences (```json ... ```) from text.
    
    Many LLMs (GLM, GPT, Claude, Gemini, etc.) sometimes wrap JSON responses
    in markdown code blocks. This function removes them so json.loads() can
    parse the content.
    
    Safe to call on any text:
    - If fences are present → strips them
    - If no fences → returns text unchanged
    
    Examples:
        >>> strip_markdown_fences('```json\\n{"key": "value"}\\n```')
        '{"key": "value"}'
        >>> strip_markdown_fences('{"key": "value"}')
        '{"key": "value"}'
    """
    text = text.strip()
    # Check for ```json or ``` at start
    if text.startswith("```"):
        # Remove opening fence (with optional language specifier like ```json)
        lines = text.split("\n", 1)
        if len(lines) > 1:
            text = lines[1]
        else:
            text = text[3:]  # Just "```" with no newline
        # Remove closing fence
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3].rstrip()
    return text
