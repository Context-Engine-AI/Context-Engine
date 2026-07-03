"""
Feature-flagged adapter for decoder-side ReFRAG using MiniMax M2.

Safe defaults:
- Only used when REFRAG_DECODER=1 and REFRAG_RUNTIME=minimax
- Requires MINIMAX_API_KEY; optionally configure MINIMAX_MODEL

MiniMax M2 API is OpenAI-compatible at https://api.minimax.io/v1
"""
from __future__ import annotations
import os
from typing import Any, Optional


class MiniMaxRefragClient:
    """MiniMax M2 client exposing generate_with_soft_embeddings(prompt, ...).

    Notes:
    - soft_embeddings are ignored (MiniMax does not support KV/soft-embed injection)
    - prompt-mode only; mirrors llama.cpp adapter surface
    - Uses OpenAI SDK with custom base_url for MiniMax API
    """

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None) -> None:
        self.api_key = api_key or os.environ.get("MINIMAX_API_KEY", "").strip()
        if not self.api_key:
            raise ValueError("MINIMAX_API_KEY is required when using REFRAG_RUNTIME=minimax")
        self.base_url = base_url or os.environ.get("MINIMAX_API_BASE", "https://api.minimax.io/v1")
        from openai import OpenAI
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def generate_with_soft_embeddings(
        self,
        prompt: str,
        soft_embeddings: Optional[list[list[float]]] = None,  # unused
        max_tokens: Optional[int] = None,
        **gen_kwargs: Any,
    ) -> str:
        model = os.environ.get("MINIMAX_MODEL", "MiniMax-M2")
        if max_tokens is None:
            try:
                max_tokens = int(os.environ.get("DECODER_MAX_TOKENS", "4096"))
            except ValueError:
                max_tokens = 4096
        # MiniMax M2 API requires temperature in (0.0, 1.0] - exclusive of 0.0
        # Per docs: "The temperature parameter range is (0.0, 1.0], recommended value: 1.0,
        # values outside this range will return an error"
        raw_temp = float(gen_kwargs.get("temperature", 1.0))
        temperature = max(0.01, min(1.0, raw_temp))  # MiniMax requires (0.0, 1.0]
        top_p = float(gen_kwargs.get("top_p", 0.95))
        # Ignore stop sequences - MiniMax M2 thinking models can be cut off prematurely
        gen_kwargs.pop("stop", None)
        gen_kwargs.pop("timeout", None)
        gen_kwargs.pop("force_json", None)
        system_prompt = gen_kwargs.pop("system", None)

        try:
            import re
            final_max_tokens = int(gen_kwargs.get("max_tokens", max_tokens))
            # Don't pass stop sequences to MiniMax - they can cut off thinking models
            # before they output the final answer
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=final_max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            content = response.choices[0].message.content or ""
            # MiniMax M2 wraps thinking in <think>...</think> blocks
            # First try to get content after </think>, if empty extract from inside think block
            after_think = re.sub(r'<think>[\s\S]*?</think>\s*', '', content).strip()
            if after_think:
                return after_think
            # If no content after think block, try to extract JSON from inside it
            think_match = re.search(r'<think>([\s\S]*?)</think>', content)
            if think_match:
                think_content = think_match.group(1)
                # Look for JSON array in the thinking
                json_match = re.search(r'\[[\s\S]*?\]', think_content)
                if json_match:
                    return json_match.group(0)
                # Fallback: extract quoted strings from numbered lists in thinking
                # Pattern matches: 1. "text" or - "text" or "text" on its own line
                quoted = re.findall(r'["\']([^"\']{5,})["\']', think_content)
                if len(quoted) >= 2:
                    # Return as JSON array
                    import json
                    return json.dumps(quoted[:2])
            return content.strip()
        except Exception as e:
            raise RuntimeError(f"MiniMax completion failed: {e}")

