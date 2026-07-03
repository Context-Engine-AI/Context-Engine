"""
Feature-flagged adapter for decoder-side ReFRAG using OpenAI API.

Safe defaults:
- Only used when REFRAG_DECODER=1 and REFRAG_RUNTIME=openai
- Requires OPENAI_API_KEY; optionally configure OPENAI_MODEL and OPENAI_MODEL_FAST

Model selection:
- OPENAI_MODEL: Used for context_answer (default: gpt-4.1)
- OPENAI_MODEL_FAST: Used for expand_query/simple tasks when disable_thinking=True (default: gpt-4.1-mini)

Model version compatibility:
- GPT-5.2: temp=1.0, top_p=1.0, max_output=64K, context=1M, reasoning support
- GPT-5.1: temp=1.0, top_p=1.0, max_output=32K, context=512K, reasoning support
- GPT-5: temp=1.0, top_p=1.0, max_output=32K, context=256K, reasoning support
- GPT-4.1: temp=1.0, top_p=1.0, max_output=32K, context=128K
- GPT-4.1-mini: temp=1.0, top_p=1.0, max_output=16K, context=128K (fast model)
- o3: temp=1.0, reasoning model
- o3-mini: temp=1.0, reasoning model (fast)
"""
from __future__ import annotations
import os
import re
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Model version configuration - backwards compatible with GPT-4.1 through GPT-5.2
# ---------------------------------------------------------------------------
# reasoning_effort: "low" | "medium" | "high" (for models that support it)
# use_max_completion_tokens: True for GPT-5.x and o3 models (they don't support max_tokens)
# supports_temperature: False for GPT-5.x and o3 (they don't support temperature param)
# supports_stop: False for GPT-5.x and o3 (they don't support stop sequences)
OPENAI_MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "gpt-5.2": {
        "temperature": 1.0,
        "top_p": 1.0,
        "max_output_tokens": 65536,  # 64K
        "max_context_tokens": 1048576,  # 1M
        "supports_reasoning": True,
        "supports_streaming": True,
        "supports_temperature": False,
        "supports_stop": False,
        "use_max_completion_tokens": True,
        "default_reasoning_effort": "medium",
    },
    "gpt-5.1": {
        "temperature": 1.0,
        "top_p": 1.0,
        "max_output_tokens": 32768,  # 32K
        "max_context_tokens": 524288,  # 512K
        "supports_reasoning": True,
        "supports_streaming": True,
        "supports_temperature": False,
        "supports_stop": False,
        "use_max_completion_tokens": True,
        "default_reasoning_effort": "medium",
    },
    "gpt-5": {
        "temperature": 1.0,
        "top_p": 1.0,
        "max_output_tokens": 32768,  # 32K
        "max_context_tokens": 262144,  # 256K
        "supports_reasoning": True,
        "supports_streaming": True,
        "supports_temperature": False,
        "supports_stop": False,
        "use_max_completion_tokens": True,
        "default_reasoning_effort": "medium",
    },
    "gpt-4.1": {
        "temperature": 1.0,
        "top_p": 1.0,
        "max_output_tokens": 32768,  # 32K
        "max_context_tokens": 131072,  # 128K
        "supports_reasoning": False,
        "supports_streaming": True,
        "supports_temperature": True,
        "supports_stop": True,
        "use_max_completion_tokens": False,
    },
    "gpt-4.1-mini": {
        "temperature": 1.0,
        "top_p": 1.0,
        "max_output_tokens": 16384,  # 16K
        "max_context_tokens": 131072,  # 128K
        "supports_reasoning": False,
        "supports_streaming": True,
        "supports_temperature": True,
        "supports_stop": True,
        "use_max_completion_tokens": False,
    },
    "o3": {
        "temperature": 1.0,
        "top_p": 1.0,
        "max_output_tokens": 100000,
        "max_context_tokens": 200000,
        "supports_reasoning": True,
        "supports_streaming": True,
        "supports_temperature": False,
        "supports_stop": False,
        "use_max_completion_tokens": True,
        "default_reasoning_effort": "medium",
    },
    "o3-mini": {
        "temperature": 1.0,
        "top_p": 1.0,
        "max_output_tokens": 65536,
        "max_context_tokens": 200000,
        "supports_reasoning": True,
        "supports_streaming": True,
        "supports_temperature": False,
        "supports_stop": False,
        "use_max_completion_tokens": True,
        "default_reasoning_effort": "low",  # fast model, use low by default
    },
}

# Default fallback config for unknown models
OPENAI_DEFAULT_CONFIG: dict[str, Any] = {
    "temperature": 1.0,
    "top_p": 1.0,
    "max_output_tokens": 8192,
    "max_context_tokens": 128000,
    "supports_reasoning": False,
    "supports_streaming": True,
}


def get_model_config(model: str) -> dict[str, Any]:
    """Get configuration for an OpenAI model version with backwards compatibility.
    
    Matches model names like 'gpt-5.2', 'gpt-4.1-mini', 'o3', etc.
    Falls back to default config for unknown models.
    """
    model_lower = model.lower()
    # Try exact match first
    if model_lower in OPENAI_MODEL_CONFIGS:
        return OPENAI_MODEL_CONFIGS[model_lower]
    # Try matching base version (e.g., 'gpt-5.2-turbo' -> 'gpt-5.2')
    for base_model, config in OPENAI_MODEL_CONFIGS.items():
        if model_lower.startswith(base_model):
            return config
    # Check for version pattern (gpt-5.X or gpt-4.X)
    match = re.match(r"gpt-(\d+)\.(\d+)", model_lower)
    if match:
        major = int(match.group(1))
        minor = int(match.group(2))
        if major >= 5:
            if minor >= 2:
                return OPENAI_MODEL_CONFIGS["gpt-5.2"]
            elif minor >= 1:
                return OPENAI_MODEL_CONFIGS["gpt-5.1"]
            else:
                return OPENAI_MODEL_CONFIGS["gpt-5"]
        elif major == 4:
            if "mini" in model_lower:
                return OPENAI_MODEL_CONFIGS["gpt-4.1-mini"]
            return OPENAI_MODEL_CONFIGS["gpt-4.1"]
    # o-series models
    if model_lower.startswith("o3-mini") or model_lower.startswith("o4-mini"):
        return OPENAI_MODEL_CONFIGS["o3-mini"]
    if model_lower.startswith("o3") or model_lower.startswith("o4"):
        return OPENAI_MODEL_CONFIGS["o3"]
    return OPENAI_DEFAULT_CONFIG


def detect_openai_runtime() -> bool:
    """Detect whether the OpenAI runtime should be considered active."""
    runtime = os.environ.get("REFRAG_RUNTIME", "").strip().lower()
    return runtime == "openai"


def get_openai_model_name() -> str:
    """Get the active OpenAI model name with consistent fallback."""
    model = os.environ.get("OPENAI_MODEL", "").strip()
    return model if model else "gpt-4.1"


class OpenAIRefragClient:
    """OpenAI client exposing generate_with_soft_embeddings(prompt, ...).

    Notes:
    - soft_embeddings are ignored (OpenAI does not support KV/soft-embed injection)
    - prompt-mode only; mirrors llama.cpp adapter surface
    - Uses OpenAI SDK with official API
    - Backwards compatible with GPT-4.1 through GPT-5.2 and o-series
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "").strip()
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required when using REFRAG_RUNTIME=openai")
        self.base_url = base_url or os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
        # Default timeout from env or 60s
        if timeout is None:
            try:
                timeout = float(os.environ.get("OPENAI_TIMEOUT", "60") or 60)
            except ValueError:
                timeout = 60.0
        self._timeout = timeout
        from openai import OpenAI
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=timeout)

    def _parse_prompt_to_messages(self, prompt: str) -> list[dict[str, str]]:
        """Parse Granite-style prompt into proper OpenAI messages array."""
        messages: list[dict[str, str]] = []
        pattern = r'<\|start_of_role\|>(\w+)<\|end_of_role\|>(.*?)(?:<\|end_of_text\|>|$)'
        matches = re.findall(pattern, prompt, re.DOTALL)
        if matches:
            for role, content in matches:
                content = content.strip()
                if content:
                    messages.append({"role": role, "content": content})
        if not messages:
            messages = [{"role": "user", "content": prompt}]
        return messages

    def generate_with_soft_embeddings(
        self,
        prompt: str,
        soft_embeddings: Optional[list[list[float]]] = None,  # unused
        max_tokens: int = 256,
        **gen_kwargs: Any,
    ) -> str:
        # Model selection priority:
        # 1. Explicit model= parameter
        # 2. disable_thinking=True -> OPENAI_MODEL_FAST
        # 3. OPENAI_MODEL env var
        # 4. Default: gpt-4.1
        disable_thinking = bool(gen_kwargs.pop("disable_thinking", False))
        explicit_model = gen_kwargs.pop("model", None)
        if explicit_model:
            model = explicit_model
        elif disable_thinking:
            model = os.environ.get("OPENAI_MODEL_FAST", "gpt-4.1-mini")
        else:
            model = os.environ.get("OPENAI_MODEL", "gpt-4.1")

        model_config = get_model_config(model)

        temperature = float(gen_kwargs.get("temperature", model_config["temperature"]))
        top_p = float(gen_kwargs.get("top_p", model_config["top_p"]))
        requested_max = int(gen_kwargs.get("max_tokens", max_tokens))
        # For reasoning models (GPT-5.x, o3), tokens are split between reasoning + response
        # Apply a multiplier to ensure enough tokens for both
        if model_config.get("supports_reasoning"):
            # Reasoning typically uses 3-10x more tokens than the response itself
            # Use a 5x multiplier with a minimum of 1000 tokens for reasoning models
            reasoning_multiplier = int(os.environ.get("OPENAI_REASONING_MULTIPLIER", "5"))
            requested_max = max(1000, requested_max * reasoning_multiplier)
        effective_max = min(requested_max, model_config["max_output_tokens"])
        stop = gen_kwargs.get("stop")
        # timeout is handled at client init, pop to avoid passing to create()
        gen_kwargs.pop("timeout", None)
        force_json = bool(gen_kwargs.pop("force_json", False))
        stream = bool(gen_kwargs.pop("stream", False))
        # Pop system message if provided
        system_msg = gen_kwargs.pop("system", None)
        # Pop unused params that other adapters might pass
        gen_kwargs.pop("no_thinking", None)
        gen_kwargs.pop("enable_thinking", None)

        try:
            # Build messages list
            messages = self._parse_prompt_to_messages(prompt)
            # Prepend system message if provided
            if system_msg:
                messages.insert(0, {"role": "system", "content": str(system_msg)})
            # OpenAI requires "json" in messages when using response_format: json_object
            if force_json:
                has_json_word = any("json" in m.get("content", "").lower() for m in messages)
                if not has_json_word:
                    # Append instruction to user message or add new one
                    if messages and messages[-1]["role"] == "user":
                        messages[-1]["content"] += " Respond with valid JSON."
                    else:
                        messages.append({"role": "user", "content": "Respond with valid JSON."})
            create_kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages,
            }
            # GPT-5.x and o3 use max_completion_tokens instead of max_tokens
            if model_config.get("use_max_completion_tokens"):
                create_kwargs["max_completion_tokens"] = effective_max
            else:
                create_kwargs["max_tokens"] = effective_max
            # GPT-5.x and o3 don't support temperature/top_p
            if model_config.get("supports_temperature", True):
                create_kwargs["temperature"] = temperature
                create_kwargs["top_p"] = top_p
            # GPT-5.x and o3 don't support stop sequences
            if stop and model_config.get("supports_stop", True):
                create_kwargs["stop"] = stop
            if stream:
                create_kwargs["stream"] = True
            if force_json:
                create_kwargs["response_format"] = {"type": "json_object"}
            # reasoning_effort for GPT-5.x, o3, o3-mini (values: "minimal", "low", "medium", "high")
            if model_config.get("supports_reasoning"):
                reasoning_effort = gen_kwargs.pop("reasoning_effort", None)
                if reasoning_effort is None:
                    reasoning_effort = os.environ.get("OPENAI_REASONING_EFFORT")
                if reasoning_effort is None:
                    reasoning_effort = model_config.get("default_reasoning_effort")
                if reasoning_effort in ("minimal", "low", "medium", "high"):
                    create_kwargs["reasoning_effort"] = reasoning_effort

            response = self.client.chat.completions.create(**create_kwargs)

            if stream:
                return self._handle_streaming_response(response)

            msg = response.choices[0].message
            return (msg.content or "").strip()
        except Exception as e:
            raise RuntimeError(f"OpenAI completion failed: {e}")

    def _handle_streaming_response(self, response: Any) -> str:
        """Handle streaming response, accumulating content."""
        content_parts: list[str] = []
        for chunk in response:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if hasattr(delta, 'content') and delta.content:
                content_parts.append(delta.content)
        return "".join(content_parts).strip()

    async def generate_batch_async(
        self,
        prompts: list[str],
        max_tokens: int = 96,
        concurrency: int = 4,
        **gen_kwargs: Any,
    ) -> list[str]:
        """Run multiple prompts concurrently using asyncio + ThreadPoolExecutor."""
        import asyncio
        from concurrent.futures import ThreadPoolExecutor

        if not prompts:
            return []

        gen_kwargs["disable_thinking"] = True
        gen_kwargs["max_tokens"] = max_tokens

        async def run_one(prompt: str) -> str:
            loop = asyncio.get_event_loop()
            try:
                return await loop.run_in_executor(
                    executor,
                    lambda: self.generate_with_soft_embeddings(prompt, **gen_kwargs)
                )
            except Exception:
                return ""

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            results = await asyncio.gather(*[run_one(p) for p in prompts])

        return list(results)


def generate_pseudo_tags_batch(
    texts: list[str],
    concurrency: int = 4,
) -> list[tuple[str, list[str]]]:
    """Batch generate pseudo+tags for multiple code chunks concurrently.

    Args:
        texts: List of code snippets to process
        concurrency: Number of concurrent OpenAI calls (default 4)

    Returns:
        List of (pseudo, tags) tuples in same order as input texts
    """
    import asyncio
    import json as _json
    from scripts.llm_utils import strip_markdown_fences as _strip_markdown_fences

    if not texts:
        return []

    # Build prompts
    prompts = []
    for text in texts:
        prompt = (
            "You are a JSON-only function that labels code spans for search enrichment.\n"
            "Respond with a single JSON object and nothing else (no prose, no markdown).\n"
            "Exact format: {\"pseudo\": string (<=20 tokens), \"tags\": [3-6 short strings]}.\n"
            "Code:\n" + text[:2000]
        )
        prompts.append(prompt)

    client = OpenAIRefragClient()

    async def _run_batch() -> list[str]:
        return await client.generate_batch_async(
            prompts,
            max_tokens=int(os.environ.get("PSEUDO_MAX_TOKENS", "96") or 96),
            concurrency=concurrency,
            temperature=float(os.environ.get("PSEUDO_TEMPERATURE", "0.10") or 0.10),
            top_p=float(os.environ.get("PSEUDO_TOP_P", "0.9") or 0.9),
            stop=["\n\n"],
            force_json=True,
        )

    # Handle both sync and async contexts safely
    try:
        # Check if we're already in a running event loop
        loop = asyncio.get_running_loop()
        # We're in an async context - create a task and use asyncio.run in a thread
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            results = pool.submit(asyncio.run, _run_batch()).result()
    except RuntimeError:
        # No running loop - safe to use asyncio.run directly
        results = asyncio.run(_run_batch())

    # Parse JSON responses
    parsed: list[tuple[str, list[str]]] = []
    for out in results:
        pseudo, tags = "", []
        try:
            obj = _json.loads(_strip_markdown_fences(out))
            if isinstance(obj, dict):
                p = obj.get("pseudo")
                t = obj.get("tags")
                if isinstance(p, str):
                    pseudo = p.strip()[:256]
                if isinstance(t, list):
                    tags = [str(x).strip() for x in t if str(x).strip()][:6]
        except Exception:
            pass
        parsed.append((pseudo, tags))

    return parsed


async def generate_pseudo_tags_batch_async(
    texts: list[str],
    concurrency: int = 4,
) -> list[tuple[str, list[str]]]:
    """Async variant of generate_pseudo_tags_batch for use in async contexts.

    Args:
        texts: List of code snippets to process
        concurrency: Number of concurrent OpenAI calls (default 4)

    Returns:
        List of (pseudo, tags) tuples in same order as input texts
    """
    import json as _json
    from scripts.llm_utils import strip_markdown_fences as _strip_markdown_fences

    if not texts:
        return []

    # Build prompts
    prompts = []
    for text in texts:
        prompt = (
            "You are a JSON-only function that labels code spans for search enrichment.\n"
            "Respond with a single JSON object and nothing else (no prose, no markdown).\n"
            "Exact format: {\"pseudo\": string (<=20 tokens), \"tags\": [3-6 short strings]}.\n"
            "Code:\n" + text[:2000]
        )
        prompts.append(prompt)

    client = OpenAIRefragClient()

    results = await client.generate_batch_async(
        prompts,
        max_tokens=int(os.environ.get("PSEUDO_MAX_TOKENS", "96") or 96),
        concurrency=concurrency,
        temperature=float(os.environ.get("PSEUDO_TEMPERATURE", "0.10") or 0.10),
        top_p=float(os.environ.get("PSEUDO_TOP_P", "0.9") or 0.9),
        stop=["\n\n"],
        force_json=True,
    )

    # Parse JSON responses
    parsed: list[tuple[str, list[str]]] = []
    for out in results:
        pseudo, tags = "", []
        try:
            obj = _json.loads(_strip_markdown_fences(out))
            if isinstance(obj, dict):
                p = obj.get("pseudo")
                t = obj.get("tags")
                if isinstance(p, str):
                    pseudo = p.strip()[:256]
                if isinstance(t, list):
                    tags = [str(x).strip() for x in t if str(x).strip()][:6]
        except Exception:
            pass
        parsed.append((pseudo, tags))

    return parsed
