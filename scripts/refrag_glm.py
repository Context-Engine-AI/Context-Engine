"""
Feature-flagged adapter for decoder-side ReFRAG using GLM (ZhipuAI).

Safe defaults:
- Only used when REFRAG_DECODER=1 and REFRAG_RUNTIME=glm
- Requires GLM_API_KEY; optionally configure GLM_MODEL and GLM_MODEL_FAST

Model selection:
- GLM_MODEL: Used for context_answer (default: glm-4.6)
- GLM_MODEL_FAST: Used for expand_query/simple tasks when disable_thinking=True (default: glm-4.5)

Model version compatibility:
- GLM-4.7: temp=1.0, top_p=0.95, max_output=128K, context=200K, tool_stream support, thinking
- GLM-4.6: temp=1.0, top_p=0.95, thinking support
- GLM-4.5: temp=1.0, top_p=0.95, fast model (no thinking)
"""
from __future__ import annotations
import os
import re
from typing import Any, Optional

from scripts.llm_utils import strip_markdown_fences as _strip_markdown_fences


# ---------------------------------------------------------------------------
# Model version configuration - backwards compatible with GLM 4.5, 4.6, 4.7
# ---------------------------------------------------------------------------
GLM_MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "glm-4.7": {
        "temperature": 1.0,
        "top_p": 0.95,
        "max_output_tokens": 131072,  # 128K
        "max_context_tokens": 204800,  # 200K
        "supports_thinking": True,
        "supports_tool_stream": True,
    },
    "glm-4.6": {
        "temperature": 1.0,
        "top_p": 0.95,
        "max_output_tokens": 50000,  # No official limit documented; 50K conservative cap
        "max_context_tokens": 204800,  # 200K (expanded from 128K per docs)
        "supports_thinking": True,
        "supports_tool_stream": False,
    },
    "glm-4.5": {
        "temperature": 1.0,
        "top_p": 0.95,
        "max_output_tokens": 96000,  # Per docs: 96K max output
        "max_context_tokens": 131072,  # 128K context
        "supports_thinking": True,  # Supports hybrid thinking modes with enabled/disabled
        "supports_tool_stream": False,
    },
}

# Default fallback config for unknown models
GLM_DEFAULT_CONFIG: dict[str, Any] = {
    "temperature": 1.0,
    "top_p": 0.95,
    "max_output_tokens": 8192,
    "max_context_tokens": 131072,
    "supports_thinking": False,
    "supports_tool_stream": False,
}


def get_model_config(model: str) -> dict[str, Any]:
    """Get configuration for a GLM model version with backwards compatibility.
    
    Matches model names like 'glm-4.7', 'glm-4.6-air', 'glm-4.5-flash', etc.
    Falls back to default config for unknown models.
    """
    model_lower = model.lower()
    # Try exact match first
    if model_lower in GLM_MODEL_CONFIGS:
        return GLM_MODEL_CONFIGS[model_lower]
    # Try matching base version (e.g., 'glm-4.7-air' -> 'glm-4.7')
    for base_model, config in GLM_MODEL_CONFIGS.items():
        if model_lower.startswith(base_model):
            return config
    # Check for version pattern (glm-4.X)
    # Note: This assumes GLM uses glm-4.X versioning. Future major versions
    # (e.g., glm-5.X) will fall through to GLM_DEFAULT_CONFIG until explicitly added.
    match = re.match(r"glm-4\.(\d+)", model_lower)
    if match:
        version = int(match.group(1))
        if version >= 7:
            return GLM_MODEL_CONFIGS["glm-4.7"]
        elif version == 6:
            return GLM_MODEL_CONFIGS["glm-4.6"]
        elif version == 5:
            return GLM_MODEL_CONFIGS["glm-4.5"]
        else:
            return GLM_DEFAULT_CONFIG
    return GLM_DEFAULT_CONFIG


def detect_glm_runtime() -> bool:
    """Detect whether the GLM runtime should be considered active.

    Shared helper to centralise the environment-variable logic used to decide
    if GLM-based behaviour should be enabled.

    Returns:
        True only if REFRAG_RUNTIME is explicitly set to 'glm' (case-insensitive).
        False otherwise (e.g., REFRAG_RUNTIME='llamacpp', 'minimax', or unset).

    Note:
        The presence of GLM_API_KEY alone does NOT auto-enable GLM.
        You must explicitly set REFRAG_RUNTIME=glm to use the GLM runtime.
    """
    runtime = os.environ.get("REFRAG_RUNTIME", "").strip().lower()
    return runtime == "glm"


def get_glm_model_name() -> str:
    """Get the active GLM model name with consistent fallback.
    
    Returns GLM_MODEL env var if set and non-empty, otherwise 'glm-4.6'.
    """
    model = os.environ.get("GLM_MODEL", "").strip()
    return model if model else "glm-4.6"


class GLMRefragClient:
    """GLM client exposing generate_with_soft_embeddings(prompt, ...).

    Notes:
    - soft_embeddings are ignored (GLM does not support KV/soft-embed injection)
    - prompt-mode only; mirrors llama.cpp adapter surface
    - Uses OpenAI SDK with custom base_url for GLM API
    - Backwards compatible with GLM 4.5, 4.6, and 4.7
    """

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None) -> None:
        self.api_key = api_key or os.environ.get("GLM_API_KEY", "").strip()
        if not self.api_key:
            raise ValueError("GLM_API_KEY is required when using REFRAG_RUNTIME=glm")
        self.base_url = base_url or os.environ.get("GLM_API_BASE", "https://api.z.ai/api/paas/v4/")
        from openai import OpenAI
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def _parse_prompt_to_messages(self, prompt: str) -> list[dict[str, str]]:
        """Parse Granite-style prompt into proper OpenAI messages array.

        Handles prompts like:
        <|start_of_role|>system<|end_of_role|>...<|end_of_text|>
        <|start_of_role|>user<|end_of_role|>...<|end_of_text|>
        <|start_of_role|>assistant<|end_of_role|>
        """
        import re
        messages: list[dict[str, str]] = []

        # Pattern to extract role blocks
        pattern = r'<\|start_of_role\|>(\w+)<\|end_of_role\|>(.*?)(?:<\|end_of_text\|>|$)'
        matches = re.findall(pattern, prompt, re.DOTALL)

        if matches:
            for role, content in matches:
                content = content.strip()
                if content:  # Skip empty content (like trailing assistant role)
                    messages.append({"role": role, "content": content})

        # Fallback: if no matches, treat entire prompt as user message
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
        # 2. disable_thinking=True -> GLM_MODEL_FAST (backwards compat for expand_query)
        # 3. GLM_MODEL env var
        # 4. Default: glm-4.6
        disable_thinking = bool(gen_kwargs.pop("disable_thinking", False))
        explicit_model = gen_kwargs.pop("model", None)
        if explicit_model:
            model = explicit_model
        elif disable_thinking:
            model = os.environ.get("GLM_MODEL_FAST", "glm-4.5")
        else:
            model = os.environ.get("GLM_MODEL", "glm-4.6")

        # no_thinking: disable thinking on current model WITHOUT switching models
        # Useful when you want to use GLM-4.7 but skip reasoning for simple tasks
        no_thinking = bool(gen_kwargs.pop("no_thinking", False))

        # Get model-specific configuration for backwards compatibility
        model_config = get_model_config(model)

        # Use model-specific defaults, allow override via gen_kwargs
        # For GLM-4.7: temp=1.0, top_p=0.95 (per migration guide)
        # For backwards compat: caller can still override with lower values for stable output
        temperature = float(gen_kwargs.get("temperature", model_config["temperature"]))
        top_p = float(gen_kwargs.get("top_p", model_config["top_p"]))

        # Cap max_tokens to model's output limit
        requested_max = int(gen_kwargs.get("max_tokens", max_tokens))
        effective_max = min(requested_max, model_config["max_output_tokens"])
        stop = gen_kwargs.get("stop")
        timeout = gen_kwargs.pop("timeout", None)
        force_json = bool(gen_kwargs.pop("force_json", False))

        # Streaming options (GLM-4.7+ supports tool_stream)
        stream = bool(gen_kwargs.pop("stream", False))
        tool_stream = bool(gen_kwargs.pop("tool_stream", False))
        tools = gen_kwargs.pop("tools", None)

        # Thinking/reasoning options
        enable_thinking = gen_kwargs.pop("enable_thinking", None)
        # Combine disable_thinking and no_thinking flags for thinking control
        should_disable_thinking = disable_thinking or no_thinking
        
        try:
            timeout_val = float(timeout) if timeout is not None else None
        except Exception:
            timeout_val = None

        try:
            create_kwargs: dict[str, Any] = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": effective_max,
                "temperature": temperature,
                "top_p": top_p,
                "stop": stop if stop else None,
                "timeout": timeout_val,
            }
            
            # Streaming support
            if stream:
                create_kwargs["stream"] = True
            
            # Tool calling support
            if tools:
                create_kwargs["tools"] = tools
                # GLM-4.7+ supports streaming tool call parameters
                if tool_stream and model_config["supports_tool_stream"]:
                    create_kwargs["extra_body"] = create_kwargs.get("extra_body", {})
                    create_kwargs["extra_body"]["tool_stream"] = True
            
            # When explicitly requested and supported by the backend, ask for
            # JSON-only responses. If the provider rejects this parameter, the
            # API call will raise and the caller will handle the failure.
            if force_json:
                create_kwargs["response_format"] = {"type": "json_object"}
            
            # Thinking/deep reasoning control (GLM-4.5/4.6/4.7 all support thinking param)
            # All three models support thinking: {type: "disabled"} for fast responses
            if model_config["supports_thinking"]:
                if should_disable_thinking:
                    create_kwargs["extra_body"] = create_kwargs.get("extra_body", {})
                    create_kwargs["extra_body"]["thinking"] = {"type": "disabled"}
                elif enable_thinking:
                    create_kwargs["extra_body"] = create_kwargs.get("extra_body", {})
                    create_kwargs["extra_body"]["thinking"] = {"type": "enabled"}

            response = self.client.chat.completions.create(**create_kwargs)
            
            # Handle streaming response
            if stream:
                return self._handle_streaming_response(response)
            
            msg = response.choices[0].message
            # GLM models may use either content or reasoning_content
            content = msg.content or ""
            # Fallback to reasoning_content if content is empty (thinking models)
            if not content.strip():
                content = getattr(msg, 'reasoning_content', None) or ""
            return content.strip()
        except Exception as e:
            raise RuntimeError(f"GLM completion failed: {e}")
    
    def _handle_streaming_response(
        self,
        response: Any,
    ) -> str:
        """Handle streaming response, accumulating content and reasoning.
        
        Supports:
        - delta.content: Regular content tokens
        - delta.reasoning_content: Thinking/reasoning tokens (GLM-4.6+)
        
        Note: Tool call streaming (GLM-4.7+) is parsed but not currently used
        since context_answer returns text, not tool calls.
        """
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        
        for chunk in response:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            
            # Accumulate reasoning content (thinking process)
            if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                reasoning_parts.append(delta.reasoning_content)
            
            # Accumulate regular content
            if hasattr(delta, 'content') and delta.content:
                content_parts.append(delta.content)
        
        # Return content, fallback to reasoning if content is empty
        content = "".join(content_parts).strip()
        if not content:
            content = "".join(reasoning_parts).strip()
        return content
    
    def generate_with_streaming(
        self,
        prompt: str,
        max_tokens: int = 256,
        on_content: Optional[Any] = None,
        on_reasoning: Optional[Any] = None,
        on_tool_call: Optional[Any] = None,
        **gen_kwargs: Any,
    ) -> dict[str, Any]:
        """Generate with streaming, providing callbacks for real-time output.
        
        Callbacks:
        - on_content(token: str): Called for each content token
        - on_reasoning(token: str): Called for each reasoning/thinking token
        - on_tool_call(idx: int, name: str, args: str): Called for tool call updates
        
        Returns:
            Dict with 'content', 'reasoning', and 'tool_calls' accumulated results
        """
        disable_thinking = bool(gen_kwargs.pop("disable_thinking", False))
        if disable_thinking:
            model = os.environ.get("GLM_MODEL_FAST", "glm-4.5")
        else:
            model = os.environ.get("GLM_MODEL", "glm-4.6")
        
        model_config = get_model_config(model)
        
        temperature = float(gen_kwargs.get("temperature", model_config["temperature"]))
        top_p = float(gen_kwargs.get("top_p", model_config["top_p"]))
        requested_max = int(gen_kwargs.get("max_tokens", max_tokens))
        effective_max = min(requested_max, model_config["max_output_tokens"])
        
        stop = gen_kwargs.get("stop")
        tools = gen_kwargs.pop("tools", None)
        tool_stream = bool(gen_kwargs.pop("tool_stream", False))
        enable_thinking = gen_kwargs.pop("enable_thinking", None)
        
        create_kwargs: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": effective_max,
            "temperature": temperature,
            "top_p": top_p,
            "stop": stop if stop else None,
            "stream": True,
        }
        
        if tools:
            create_kwargs["tools"] = tools
            if tool_stream and model_config["supports_tool_stream"]:
                create_kwargs["extra_body"] = create_kwargs.get("extra_body", {})
                create_kwargs["extra_body"]["tool_stream"] = True
        
        if model_config["supports_thinking"]:
            if disable_thinking:
                create_kwargs["extra_body"] = create_kwargs.get("extra_body", {})
                create_kwargs["extra_body"]["thinking"] = {"type": "disabled"}
            elif enable_thinking:
                create_kwargs["extra_body"] = create_kwargs.get("extra_body", {})
                create_kwargs["extra_body"]["thinking"] = {"type": "enabled"}
        
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls: dict[int, dict[str, str]] = {}
        
        response = self.client.chat.completions.create(**create_kwargs)
        
        for chunk in response:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            
            if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                reasoning_parts.append(delta.reasoning_content)
                if on_reasoning:
                    on_reasoning(delta.reasoning_content)
            
            if hasattr(delta, 'content') and delta.content:
                content_parts.append(delta.content)
                if on_content:
                    on_content(delta.content)
            
            if hasattr(delta, 'tool_calls') and delta.tool_calls:
                for tool_call in delta.tool_calls:
                    idx = tool_call.index
                    if idx not in tool_calls:
                        tool_calls[idx] = {
                            "name": getattr(tool_call.function, 'name', '') or '',
                            "arguments": getattr(tool_call.function, 'arguments', '') or '',
                        }
                    else:
                        if hasattr(tool_call.function, 'arguments') and tool_call.function.arguments:
                            tool_calls[idx]["arguments"] += tool_call.function.arguments
                    if on_tool_call:
                        on_tool_call(idx, tool_calls[idx]["name"], tool_calls[idx]["arguments"])
        
        return {
            "content": "".join(content_parts).strip(),
            "reasoning": "".join(reasoning_parts).strip(),
            "tool_calls": tool_calls,
        }

    async def generate_batch_async(
        self,
        prompts: list[str],
        max_tokens: int = 96,
        concurrency: int = 4,
        **gen_kwargs: Any,
    ) -> list[str]:
        """Run multiple prompts concurrently using asyncio + ThreadPoolExecutor.

        Args:
            prompts: List of prompts to process
            max_tokens: Max tokens per response
            concurrency: Number of concurrent requests (default 4)
            **gen_kwargs: Additional args passed to generate_with_soft_embeddings

        Returns:
            List of responses in same order as prompts
        """
        import asyncio
        from concurrent.futures import ThreadPoolExecutor

        if not prompts:
            return []

        # Always use fast model for batch operations (indexing)
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
        concurrency: Number of concurrent GLM calls (default 4)

    Returns:
        List of (pseudo, tags) tuples in same order as input texts
    """
    import asyncio
    import json as _json

    if not texts:
        return []

    # Build prompts - explicit JSON-only instruction for coding endpoint
    # (coding endpoint doesn't support response_format, so we rely on prompting)
    prompts = []
    for text in texts:
        prompt = (
            "Output ONLY a JSON object, no explanation, no markdown fences.\n"
            'Format: {"pseudo": "<20 token summary>", "tags": ["tag1", "tag2", ...]}\n'
            "Code:\n" + text[:2000]
        )
        prompts.append(prompt)

    # Run batch with thinking disabled for clean JSON output
    client = GLMRefragClient()

    async def _run_batch() -> list[str]:
        return await client.generate_batch_async(
            prompts,
            max_tokens=int(os.environ.get("PSEUDO_MAX_TOKENS", "96") or 96),
            concurrency=concurrency,
            temperature=float(os.environ.get("PSEUDO_TEMPERATURE", "0.10") or 0.10),
            top_p=float(os.environ.get("PSEUDO_TOP_P", "0.9") or 0.9),
            stop=["\n\n", "```"],
            disable_thinking=True,  # Skip reasoning for clean JSON
        )

    # Handle both sync and async contexts safely
    try:
        asyncio.get_running_loop()
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            results = pool.submit(asyncio.run, _run_batch()).result()
    except RuntimeError:
        results = asyncio.run(_run_batch())


    # Parse JSON responses - robust extraction for coding endpoint output
    parsed: list[tuple[str, list[str]]] = []
    for out in results:
        pseudo, tags = "", []
        try:
            # First try direct parse after stripping fences
            cleaned = _strip_markdown_fences(out)
            obj = _json.loads(cleaned)
        except Exception:
            # Fallback: extract first JSON object from response
            try:
                match = re.search(r'\{[^{}]*"pseudo"[^{}]*\}', out, re.DOTALL)
                if match:
                    obj = _json.loads(match.group(0))
                else:
                    obj = {}
            except Exception:
                obj = {}
        if isinstance(obj, dict):
            p = obj.get("pseudo")
            t = obj.get("tags")
            if isinstance(p, str):
                pseudo = p.strip()[:256]
            if isinstance(t, list):
                tags = [str(x).strip() for x in t if str(x).strip()][:6]
        parsed.append((pseudo, tags))

    return parsed
