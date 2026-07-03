#!/usr/bin/env python3
import re
import difflib
"""
Context-aware prompt enhancer CLI.

Retrieves relevant code context from the Context-Engine MCP server and enhances
your prompts with it using a local LLM decoder. Works with both questions and
commands/instructions. Outputs at least two detailed paragraphs.

Usage:
  ctx "how does hybrid search work?"              # Question → enhanced question
  ctx "refactor the caching logic"                # Command → enhanced instructions
  ctx --language python "explain the indexer"     # Filter by language
  ctx --detail "add error handling to ctx.py"     # Include code snippets

Examples:
  # Enhance questions with context
  ctx "how does the indexer work?"
  # Output: Two detailed question paragraphs with file/line references

  # Enhance commands with specific details
  ctx "refactor ctx.py to improve modularity"
  # Output: Two detailed instruction paragraphs with concrete steps

  # Detail mode: include short code snippets (slower but richer)
  ctx --detail "explain the caching logic"

  # Unicorn mode: staged 2-3 pass enhancement for best quality
  ctx --unicorn "refactor ctx.py"
  ctx --unicorn "what is ReFRAG and how does it work?"

  # Pipe to LLM
  ctx "fix the bug in watcher.py" | llm

  # Filter by language and path
  ctx --language python --under scripts/ "caching implementation"

Environment:
  MCP_INDEXER_URL       - MCP indexer endpoint (default: http://localhost:8003/mcp)
  CTX_LIMIT             - Default result limit (default: 5)
  CTX_CONTEXT_LINES     - Context lines for snippets (default: 0)
  CTX_REWRITE_MAX_TOKENS - Max tokens for LLM rewrite (default: 320)
  DECODER_URL           - Override decoder endpoint
  USE_GPU_DECODER       - Use GPU decoder on port 8081 (default: 0)
"""

import sys
import json
import os
import argparse
import subprocess
from urllib import request
from urllib.parse import urlparse
from urllib.error import HTTPError, URLError
import socket
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

# Load .env file if it exists (for local CLI usage)
def _load_env_file():
	"""Load .env file from workspace (if provided) or project root if it exists."""
	# Prefer an explicit workspace root (set by the hook) when available,
	# otherwise fall back to the original project-root behavior based on this file.
	script_dir = Path(__file__).resolve().parent
	candidates = []

	workspace_dir = os.environ.get("CTX_WORKSPACE_DIR")
	if workspace_dir:
		try:
			candidates.append(Path(workspace_dir) / ".env")
		except Exception:
			pass

	# Original project-root-based .env (for CLI / repo-local usage)
	candidates.append(script_dir.parent / ".env")

	for env_file in candidates:
		if not env_file.exists():
			continue
		with open(env_file) as f:
			for line in f:
				line = line.strip()
				if not line or line.startswith("#"):
					continue
				if "=" in line:
					key, value = line.split("=", 1)
					key = key.strip()
					value = value.strip().strip('"').strip("'")
					# Only set if not already in environment
					if key and key not in os.environ:
						os.environ[key] = value
		# Only load the first existing .env
		break

_load_env_file()

try:
    from scripts.mcp_router import call_tool_http  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - local execution fallback
    from mcp_router import call_tool_http  # type: ignore

# Configuration from environment
MCP_URL = os.environ.get("MCP_INDEXER_URL", "http://localhost:8003/mcp")
DEFAULT_LIMIT = int(os.environ.get("CTX_LIMIT", "5"))
DEFAULT_CONTEXT_LINES = int(os.environ.get("CTX_CONTEXT_LINES", "0"))
DEFAULT_REWRITE_TOKENS = int(os.environ.get("CTX_REWRITE_MAX_TOKENS", "320"))
DEFAULT_PER_PATH = int(os.environ.get("CTX_PER_PATH", "2"))

# User preferences config file
CTX_CONFIG_FILE = os.path.expanduser("~/.ctx_config.json")

# Local decoder configuration (llama.cpp server)
def resolve_decoder_url() -> str:
    """Resolve decoder endpoint, honoring overrides and Ollama/GLM options.

    Rules:
    - DECODER_URL wins
    - Otherwise, if OLLAMA_HOST is set, default to its /api/chat endpoint
    - Otherwise, fall back to llama.cpp URL (GPU override if requested)
    - Only append /completion for llama.cpp-style endpoints; leave Ollama/OpenAI paths untouched
    """
    override = os.environ.get("DECODER_URL", "").strip()
    if override:
        base = override
    else:
        ollama_host = os.environ.get("OLLAMA_HOST", "").strip()
        if ollama_host:
            base = ollama_host.rstrip("/")
            if "/api/" not in base:
                base = base + "/api/chat"
        else:
            use_gpu = str(os.environ.get("USE_GPU_DECODER", "0")).strip().lower()
            if use_gpu in {"1", "true", "yes", "on"}:
                host = "host.docker.internal" if os.path.exists("/.dockerenv") else "localhost"
                base = f"http://{host}:8081"
            else:
                base = os.environ.get("LLAMACPP_URL", "http://localhost:8080").strip()

    base = base or "http://localhost:11434/api/chat"
    parsed_base = urlparse(base)
    if parsed_base.hostname == "host.docker.internal" and not os.path.exists("/.dockerenv"):
        try:
            socket.gethostbyname(parsed_base.hostname)
        except socket.gaierror:
            base = base.replace("host.docker.internal", "localhost")
            sys.stderr.write("[DEBUG] decoder host.docker.internal not reachable; falling back to localhost\n")
            sys.stderr.flush()
    lowered = base.lower()
    if (
        "ollama" in lowered
        or "/api/chat" in lowered
        or "/api/generate" in lowered
        or "/v1/chat/completions" in lowered
    ):
        return base
    if base.endswith("/completion"):
        return base
    return base.rstrip("/") + "/completion"


DECODER_URL = resolve_decoder_url()
DECODER_TIMEOUT = int(os.environ.get("CTX_DECODER_TIMEOUT", "300"))


# Global session ID for MCP HTTP calls
_session_id: Optional[str] = None


def parse_sse_response(text: str) -> Dict[str, Any]:
    """Parse SSE format response (event: message\\ndata: {...})."""
    for line in text.strip().split('\n'):
        if line.startswith('data: '):
            return json.loads(line[6:])
    raise ValueError("No data line found in SSE response")


def get_session_id(timeout: int = 10) -> str:
    """Initialize MCP session and return session ID."""
    global _session_id
    if _session_id:
        return _session_id

    payload = {
        "jsonrpc": "2.0",
        "id": 0,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "ctx-cli", "version": "1.0.0"}
        }
    }

    try:
        req = request.Request(
            MCP_URL,
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream"
            }
        )
        with request.urlopen(req, timeout=timeout) as resp:
            session_id = resp.headers.get("mcp-session-id")
            if not session_id:
                raise RuntimeError("Server did not return session ID")
            # Read the initialization response to ensure session is fully established
            init_response = resp.read().decode('utf-8')
            # Wait a moment for session to be fully processed
            import time
            time.sleep(0.5)
            _session_id = session_id
            return session_id
    except Exception as e:
        raise RuntimeError(f"Failed to initialize MCP session: {e}")


def call_mcp_tool(tool_name: str, params: Dict[str, Any], timeout: int = 30) -> Dict[str, Any]:
    """Call MCP tool via HTTP JSON-RPC with session management."""
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": tool_name, "arguments": params}
    }

    # Debug output (opt-in to avoid leaking queries in normal use)
    debug_flag = os.environ.get("CTX_DEBUG", "").strip().lower()
    if debug_flag in {"1", "true", "yes", "on"}:
        sys.stderr.write(f"[DEBUG] Calling MCP tool '{tool_name}' at {MCP_URL}\n")
        sys.stderr.write(f"[DEBUG] Sending payload: {json.dumps(payload, indent=2)}\n")
        sys.stderr.flush()

    try:
        return call_tool_http(MCP_URL, tool_name, params, timeout=float(timeout))
    except Exception as e:
        sys.stderr.write(f"[ERROR] MCP call to '{tool_name}' at {MCP_URL} failed: {type(e).__name__}: {e}\n")
        sys.stderr.flush()
        return {"error": f"Request failed: {str(e)}"}


def parse_mcp_response(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Parse MCP response and extract the actual result.

    Supports both text and json content items from FastMCP.
    """
    if "error" in result:
        return None

    # FastMCP typically wraps results in a content array
    res = result.get("result", {})
    content = res.get("content", [])

    # Some servers may return a dict directly (no content array)
    if isinstance(res, dict) and content == [] and any(k in res for k in ("results", "answer", "total")):
        return res

    if not content:
        return None

    item = content[0] or {}

    # Prefer typed JSON content
    if isinstance(item, dict) and "json" in item:
        return item.get("json")

    # Fallback: parse text as JSON or return raw text
    text = item.get("text", "") if isinstance(item, dict) else ""
    if not text:
        return None

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"raw": text}


def _extract_tool_error(result: Dict[str, Any]) -> Optional[str]:
    """Best-effort extraction of a tool-level error message from MCP response.

    Handles FastMCP-style isError + content.text as well as structuredContent.result.error.
    Returns a human-readable error string when present, or None when no error detected.
    """
    try:
        res = result.get("result") or {}
        if isinstance(res, dict) and res.get("isError") is True:
            content = res.get("content") or []
            if isinstance(content, list) and content and isinstance(content[0], dict):
                text = content[0].get("text")
                if isinstance(text, str) and text.strip():
                    return text.strip()
        sc = res.get("structuredContent") or {}
        rs = sc.get("result") or {}
        err = rs.get("error")
        if isinstance(err, str) and err.strip():
            return err.strip()
    except Exception:
        return None
    return None


def _compress_snippet(snippet: str, max_lines: int = 6) -> str:
    """Compact, high-signal subset of a code snippet.

    Heuristics: prefer signatures, guards, returns/raises, asserts; fall back to head/tail.
    """
    try:
        raw_lines = [ln.rstrip() for ln in snippet.splitlines() if ln.strip()]
        if not raw_lines:
            return ""
        keys = ("def ", "class ", "return", "raise", "assert", "if ", "except", "try:")
        scored = [(sum(k in ln for k in keys), idx, ln) for idx, ln in enumerate(raw_lines)]
        keep_idx = sorted({idx for _, idx, _ in sorted(scored, key=lambda t: (-t[0], t[1]))[:max_lines]})
        kept = [raw_lines[i] for i in keep_idx]
        if not kept:
            head = raw_lines[: max(1, max_lines // 2)]
            tail = raw_lines[-(max_lines - len(head)) :]
            kept = head + tail
        return "\n".join(kept[:max_lines])
    except Exception:
        return (snippet or "").splitlines()[0][:160]


def format_search_results(results: List[Dict[str, Any]], include_snippets: bool = False) -> str:
    """Format search results succinctly for LLM rewrite.

    When include_snippets is False (default), only include headers with path and line ranges.
    This keeps prompts small and fast for Granite via llama.cpp.
    """
    lines: List[str] = []
    for hit in results:
        # Prefer the server-chosen display path; fall back to host/container paths
        raw_path = (
            hit.get("path")
            or hit.get("host_path")
            or hit.get("container_path")
            or "unknown"
        )
        path = raw_path
        start = hit.get("start_line", "?")
        end = hit.get("end_line", "?")
        language = hit.get("language") or ""
        symbol = hit.get("symbol") or ""
        snippet = (hit.get("snippet") or "").strip()

        # Only include line ranges when both start and end are known
        if start in (None, "?") or end in (None, "?"):
            header = f"- {path}"
        else:
            header = f"- {path}:{start}-{end}"
        meta: List[str] = []
        if language:
            meta.append(language)
        if symbol:
            meta.append(f"{symbol}")
        if meta:
            header += f" ({', '.join(meta)})"
        lines.append(header)

        if include_snippets and snippet:
            compact = _compress_snippet(snippet, max_lines=6)
            if compact:
                for ln in compact.splitlines():
                    # Inline compact snippet (no fences to keep token count small)
                    lines.append(f"    {ln}")

    return "\n".join(lines)



def _ensure_two_paragraph_questions(text: str) -> str:
    """Normalize to at least two paragraphs.

    - Collapse excessive whitespace
    - For questions: ensure each paragraph ends with '?'
    - For commands/instructions: ensure proper punctuation
    - If only one paragraph, split heuristically or add a generic follow-up
    """
    if not text:
        return ""
    # Normalize whitespace/newlines
    t = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    # Collapse triple+ newlines to double
    while "\n\n\n" in t:
        t = t.replace("\n\n\n", "\n\n")
    raw_paras = [p.strip() for p in t.split("\n\n") if p.strip()]

    # Deduplicate paragraphs (case/whitespace insensitive, tolerance for near-duplicates)
    paras: list[str] = []
    dedup_keys: list[str] = []
    for p in raw_paras:
        key = re.sub(r"\s+", " ", p).strip().lower()
        if any(difflib.SequenceMatcher(None, key, existing).ratio() >= 0.99 for existing in dedup_keys):
            continue
        dedup_keys.append(key)
        paras.append(p)

    def normalize_paragraph(s: str) -> str:
        """Ensure proper punctuation - keep questions as questions, commands as commands."""
        s = s.strip()
        if not s:
            return s
        # If already ends with proper punctuation, keep as-is
        if s[-1] in "?!.":
            return s
        # Check if it looks like a question (starts with question words or contains '?')
        question_starters = ("what", "how", "why", "when", "where", "who", "which", "can", "could", "would", "should", "is", "are", "does", "do")
        first_word = s.split()[0].lower() if s.split() else ""
        if first_word in question_starters or "?" in s:
            # It's a question - ensure it ends with '?'
            if s[-1] in ".!:":
                return s[:-1].rstrip() + "?"
            return s + "?"
        # It's a command/statement - ensure it ends with '.'
        if s[-1] in ":":
            return s[:-1].rstrip() + "."
        return s + "."

    max_paragraphs = 3
    if len(paras) >= 2:
        selected = [normalize_paragraph(p) for p in paras[:max_paragraphs]]
        return "\n\n".join(selected)

    # Single paragraph: try to split by sentence boundary
    p = paras[0] if paras else t
    # Naive sentence split
    sentences = [s.strip() for s in p.replace("?", ". ").replace("!", ". ").split(". ") if s.strip()]
    if len(sentences) > 1:
        half = max(1, len(sentences) // 2)
        p1 = ". ".join(sentences[:half]).strip()
        p2 = ". ".join(sentences[half:]).strip()
    else:
        p1 = p.strip()
        p2 = (
            "Detail the exact systems involved (e.g., files, classes, state machines), how data flows between them, and any validation before emitting updates."
        )
    return normalize_paragraph(p1) + "\n\n" + normalize_paragraph(p2)


# --- Grounding helpers to reduce hallucinated paths/symbols
from typing import Set

def extract_allowed_citations(context_text: str) -> tuple[Set[str], Set[str]]:
    """Extract allowed file paths and symbols from formatted context lines.

    Parses lines produced by format_search_results. Returns (paths, symbols).
    """
    allowed_paths: Set[str] = set()
    allowed_symbols: Set[str] = set()
    for raw in (context_text or "").splitlines():
        line = (raw or "").strip()
        if not line:
            continue
        if line.startswith("- "):
            header = line[2:].strip()
            header_main = header.split(" (")[0]
            path_part = header_main.split(":")[0]
            if path_part:
                allowed_paths.add(path_part)
            # symbols are inside parens, after optional language
            m = re.search(r"\(([^)]+)\)", header)
            if m:
                for part in m.group(1).split(","):
                    sym = part.strip()
                    if sym and sym.lower() not in {
                        "python", "typescript", "javascript", "go", "rust", "java", "c", "c++", "c#", "shell", "bash", "markdown", "json", "yaml", "toml"
                    }:
                        allowed_symbols.add(sym)
    return allowed_paths, allowed_symbols


def build_refined_query(original_query: str, allowed_paths: Set[str], allowed_symbols: Set[str], max_terms: int = 6) -> str:
    """Construct a grounded follow-up query using only known paths/symbols."""
    from os.path import basename
    terms: list[str] = []
    for p in list(allowed_paths)[: max_terms // 2]:
        base = basename(p)
        if base and base not in terms:
            terms.append(base)
    for s in list(allowed_symbols)[: max_terms - len(terms)]:
        if s and s not in terms:
            terms.append(s)
    return (original_query or "").strip() + (" " + " ".join(terms) if terms else "")


def _simple_tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[A-Za-z0-9_]+", text or "")
    return [t.lower() for t in tokens if t]


def _token_overlap_ratio(a: str, b: str) -> float:
    a_tokens = set(_simple_tokenize(a))
    b_tokens = set(_simple_tokenize(b))
    if not a_tokens or not b_tokens:
        return 0.0
    inter = len(a_tokens & b_tokens)
    union = len(a_tokens | b_tokens)
    if not union:
        return 0.0
    return inter / union


def _estimate_query_result_relevance(query: str, results: List[Dict[str, Any]]) -> float:
    q_tokens = set(_simple_tokenize(query))
    if not q_tokens or not results:
        return 0.0
    scores: List[float] = []
    for hit in results[:5]:
        parts: List[str] = []
        for key in ("path", "symbol", "snippet"):
            val = hit.get(key)
            if isinstance(val, str):
                parts.append(val)
        if not parts:
            continue
        r_tokens = set()
        for part in parts:
            r_tokens.update(_simple_tokenize(part))
        if not r_tokens:
            continue
        inter = len(q_tokens & r_tokens)
        union = len(q_tokens | r_tokens)
        if union:
            scores.append(inter / union)
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def sanitize_citations(text: str, allowed_paths: Set[str]) -> str:
    """Replace path-like strings not present in allowed_paths with a neutral phrase.

    Keeps exact paths and basenames that appear in allowed_paths; replaces others.
    """
    if not text:
        return text
    from os.path import basename
    allowed_set = set(allowed_paths or set())
    basename_to_paths: Dict[str, Set[str]] = {}
    for _p in allowed_set:
        _b = basename(_p)
        if _b:
            basename_to_paths.setdefault(_b, set()).add(_p)

    # For now, keep allowed paths exactly as they appear in the context refs.
    # Earlier versions tried to be clever by rewriting absolute paths to
    # workspace-relative forms (e.g., "Context-Engine/scripts/ctx.py"), which
    # could produce confusing hybrids when multiple workspace roots or
    # slugged/collection-hash directories were involved.  To simplify behavior
    # and avoid mixing host/container/hash paths, we preserve the original
    # full path strings for any citation that is known to come from the
    # formatted context.
    root = (os.environ.get("CTX_WORKSPACE_DIR") or "").strip()

    def _to_display_path(full_path: str) -> str:
        # Identity mapping: leave allowed paths as-is so the LLM sees the same
        # absolute/host paths that appeared in the Context refs.
        return full_path

    def _repl(m):
        p = m.group(0)
        if p in allowed_set:
            return _to_display_path(p)
        b = basename(p)
        paths = basename_to_paths.get(b) if b else None
        if paths:
            if len(paths) == 1:
                return _to_display_path(next(iter(paths)))
            return p
        return "the referenced file"

    cleaned = re.sub(r"/path/to/[^\s]+", "the referenced file", text)
    # Simple path-like matcher: segments with a slash and a dot-ext
    cleaned = re.sub(r"(?<!\w)([./\w-]+/[./\w-]+\.[A-Za-z0-9_-]+|[A-Za-z0-9_.-]+\.[A-Za-z0-9_-]+)", _repl, cleaned)
    return cleaned



def _load_user_preferences() -> dict:
    """Load user preferences from ~/.ctx_config.json if it exists.

    Example config:
    {
        "always_include_tests": true,
        "prefer_bullet_commands": true,
        "extra_instructions": "Always include error handling considerations",
        "default_mode": "unicorn",
        "streaming": true
    }
    """
    if not os.path.exists(CTX_CONFIG_FILE):
        return {}
    try:
        with open(CTX_CONFIG_FILE, 'r') as f:
            return json.load(f)
    except Exception:
        return {}


def _apply_user_preferences(system_msg: str, user_msg: str, prefs: dict) -> tuple[str, str]:
    """Apply user preferences to system and user messages.

    Allows personalization like:
    - Always include test-plan paragraph
    - Prefer bullet commands
    - Custom instructions
    """
    if not prefs:
        return system_msg, user_msg

    # Add extra instructions to system message
    if prefs.get("extra_instructions"):
        system_msg += f"\n\nUser preference: {prefs['extra_instructions']}"

    # Modify user message based on preferences
    if prefs.get("always_include_tests"):
        user_msg += "\n\nAlways include a paragraph about testing considerations and test cases."

    if prefs.get("prefer_bullet_commands"):
        user_msg += "\n\nFor commands/instructions, prefer bullet-point format for clarity."

    return system_msg, user_msg


def _adaptive_context_sizing(query: str, filters: dict) -> dict:
    """Adaptively adjust limit and context_lines based on query characteristics.

    - Short/vague queries → increase limit and context for richer grounding
    - Queries with file/function names → lighter settings for speed
    """
    import re
    adjusted = dict(filters)

    # Detect if query mentions specific files or functions
    has_file_ref = bool(re.search(r'\b\w+\.(py|js|ts|go|rs|java|cpp|c|h)\b', query))
    has_function_ref = bool(re.search(r'\b(function|class|def|func|fn|method)\s+\w+', query))
    is_specific = has_file_ref or has_function_ref

    # Query length heuristic
    word_count = len(query.split())
    is_short = word_count < 5

    # Adaptive sizing
    if is_short and not is_specific:
        # Short, vague query → need more context
        adjusted["limit"] = max(adjusted.get("limit", DEFAULT_LIMIT), 6)
        if adjusted.get("with_snippets"):
            adjusted["context_lines"] = max(adjusted.get("context_lines", DEFAULT_CONTEXT_LINES), 10)
    elif is_specific:
        # Specific query → can use lighter settings
        adjusted["limit"] = min(adjusted.get("limit", DEFAULT_LIMIT), 4)
        if adjusted.get("with_snippets"):
            adjusted["context_lines"] = min(adjusted.get("context_lines", DEFAULT_CONTEXT_LINES) or 8, 6)

    return adjusted


def enhance_prompt(query: str, **filters) -> str:
    """Retrieve context, invoke the LLM, and return a final enhanced prompt.

    Uses adaptive context sizing to balance quality and speed.
    """
    # Apply adaptive sizing
    filters = _adaptive_context_sizing(query, filters)

    context_text, context_note = fetch_context(query, **filters)

    require_ctx_flag = os.environ.get("CTX_REQUIRE_CONTEXT", "").strip().lower()
    if require_ctx_flag in {"1", "true", "yes", "on"}:
        has_real_context = bool((context_text or "").strip()) and not (
            context_note and (
                "failed" in context_note.lower()
                or "no relevant" in context_note.lower()
                or "no data" in context_note.lower()
            )
        )
        if not has_real_context:
            return (query or "").strip()

    rewrite_opts = filters.get("rewrite_options") or {}
    rewritten = rewrite_prompt(
        query,
        context_text,
        context_note,
        max_tokens=rewrite_opts.get("max_tokens"),
    )
    return rewritten.strip()


def _generate_plan(enhanced_prompt: str, context: str, note: str) -> str:
    """Generate a step-by-step execution plan for a command/instruction.

    Uses the LLM to create a concrete action plan based on the enhanced prompt and code context.
    Returns empty string if plan generation fails or is not applicable.
    """
    import sys

    # Detect if we have actual code context
    has_code_context = bool((context or "").strip() and not (note and ("failed" in note.lower() or "no relevant" in note.lower() or "no data" in note.lower())))

    if not has_code_context:
        # No code context - skip plan generation
        return ""

    system_msg = (
        "You are a technical planning assistant. Your job is to create a step-by-step execution plan. "
        "Given an enhanced prompt and code context, generate a numbered list of concrete steps to accomplish the task. "
        "Each step should be specific and actionable. "
        "Format: Start with 'EXECUTION PLAN:' followed by numbered steps (1., 2., 3., etc.). "
        "Keep it concise - aim for 3-7 steps maximum. "
        "Only reference files, functions, or code elements that appear in the provided context. "
        "Do NOT invent file paths or function names. "
        "Output format: plain text only, no markdown, no code fences."
    )

    user_msg = (
        f"Code context:\n{context}\n\n"
        f"Enhanced prompt:\n{enhanced_prompt}\n\n"
        "Generate a step-by-step execution plan to accomplish this task. "
        "Use only the files and functions mentioned in the code context above. "
        "Format as: EXECUTION PLAN: followed by numbered steps."
    )

    meta_prompt = (
        "<|start_of_role|>system<|end_of_role|>" + system_msg + "<|end_of_text|>\n"
        "<|start_of_role|>user<|end_of_role|>" + user_msg + "<|end_of_text|>\n"
        "<|start_of_role|>assistant<|end_of_role|>"
    )

    runtime_kind = str(os.environ.get("REFRAG_RUNTIME", "llamacpp")).strip().lower()

    # GLM path mirrors rewrite_prompt behavior
    if runtime_kind == "glm":
        try:
            from refrag_glm import GLMRefragClient  # type: ignore

            client = GLMRefragClient()
            response = client.client.chat.completions.create(
                model=os.environ.get("GLM_MODEL", "glm-4.6"),
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=200,
                temperature=0.3,
                stream=False,
            )
            plan = (
                (response.choices[0].message.content if response and response.choices else "")
                or ""
            ).strip()
            if not plan:
                # Fall through to llama.cpp path
                runtime_kind = "llamacpp"
            else:
                if "EXECUTION PLAN" not in plan.upper():
                    plan = "EXECUTION PLAN:\n" + plan
                return plan
        except Exception as e:
            sys.stderr.write(f"[DEBUG] Plan generation (GLM) failed, falling back to llama.cpp: {type(e).__name__}: {e}\n")
            sys.stderr.flush()
            runtime_kind = "llamacpp"

    decoder_url = DECODER_URL
    # Safety: restrict to local decoder hosts
    parsed = urlparse(decoder_url)
    if parsed.hostname not in {"localhost", "127.0.0.1", "host.docker.internal"}:
        return ""

    payload = {
        "prompt": meta_prompt,
        "n_predict": 200,  # Shorter for plan generation
        "temperature": 0.3,  # Lower temperature for more focused plans
        "stream": False,  # Silent plan generation
    }

    try:
        req = request.Request(
            decoder_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )

        # Use shorter timeout for plan generation (60 seconds instead of 300)
        plan_timeout = min(60, DECODER_TIMEOUT)
        with request.urlopen(req, timeout=plan_timeout) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
            data = json.loads(raw)

            plan = (
                (data.get("content") if isinstance(data, dict) else None)
                or ((data.get("choices") or [{}])[0].get("content") if isinstance(data, dict) else None)
                or ((data.get("choices") or [{}])[0].get("text") if isinstance(data, dict) else None)
                or (data.get("generated_text") if isinstance(data, dict) else None)
                or (data.get("text") if isinstance(data, dict) else None)
                or ""
            )

            plan = plan.strip()

            # Relaxed validation: return any non-empty plan; add header if missing
            if not plan:
                return ""
            if "EXECUTION PLAN" not in plan.upper():
                plan = "EXECUTION PLAN:\n" + plan
            return plan

    except Exception as e:
        # Plan generation failed - not critical, just skip it
        sys.stderr.write(f"[DEBUG] Plan generation failed: {type(e).__name__}: {e}\n")
        sys.stderr.flush()
        return ""


def _needs_polish(text: str) -> bool:
    """Enhanced QA heuristic to decide if a third polishing pass is needed.

    Checks for:
    - Too short output
    - Generic/vague language
    - Missing concrete details
    - Lack of code-specific references
    """
    if not text:
        return True
    t = text.strip()

    # Length check
    if len(t) < 180:
        return True

    # Generic language cues (expanded list)
    generic_cues = (
        "overall structure", "consider ", "ensure ", "improve its",
        "you should", "it is important", "make sure", "be sure to",
        "in general", "typically", "usually", "often"
    )
    generic_count = sum(1 for cue in generic_cues if cue in t.lower())
    if generic_count >= 3:
        return True

    # Check for concrete details (file paths, line numbers, function names, etc.)
    import re
    has_file_ref = bool(re.search(r'\b\w+\.(py|js|ts|go|rs|java|cpp|c|h)\b', t))
    has_line_ref = bool(re.search(r'\bline[s]?\s+\d+', t, re.IGNORECASE))
    has_function_ref = bool(re.search(r'\b(function|class|method|def|fn)\s+\w+', t))
    has_concrete = has_file_ref or has_line_ref or has_function_ref

    # If no concrete references and has generic language, needs polish
    if not has_concrete and generic_count >= 2:
        return True

    # Check paragraph structure (should have at least 2 paragraphs)
    paragraphs = [p.strip() for p in t.split('\n\n') if p.strip()]
    if len(paragraphs) < 2:
        return True

    return False


def _dedup_paragraphs(text: str, max_paragraphs: int = 3) -> str:
    """Deterministic paragraph-level deduplication and truncation.

    - Split on double-newline boundaries
    - Drop duplicate paragraphs beyond the first occurrence (case/whitespace insensitive)
    - Cap total paragraphs to max_paragraphs
    """
    if not text:
        return ""

    # Normalize newlines and split into paragraphs
    t = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    raw_paras = [p.strip() for p in t.split("\n\n") if p.strip()]
    if not raw_paras:
        return text.strip()

    seen_keys: set[str] = set()
    out: list[str] = []
    for p in raw_paras:
        key = re.sub(r"\s+", " ", p).strip().lower()
        if key in seen_keys:
            continue
        seen_keys.add(key)
        out.append(p)
        if len(out) >= max_paragraphs:
            break

    if not out:
        return text.strip()
    return "\n\n".join(out)


def enhance_unicorn(query: str, **filters) -> str:
    """Multi-pass staged enhancement for higher quality with optional plan generation.

    Pass 1: rich snippets to draft sharper intent
    Pass 2: refined retrieval using the draft, with even richer snippets to ground specifics
    Pass 3: polish if output looks short/generic
    Pass 4 (optional): generate execution plan if query is a command/instruction

    Falls back to single-pass enhance_prompt if no context is available.
    Stops immediately when repo search returns no hits to avoid hallucinated references.
    """
    # ---- Pass 1: draft (rich snippets for grounding)
    f1 = dict(filters)
    rewrite_opts = f1.get("rewrite_options") or {}
    try:
        max_budget = int(rewrite_opts.get("max_tokens", DEFAULT_REWRITE_TOKENS))
    except Exception:
        max_budget = DEFAULT_REWRITE_TOKENS
    f1.update({
        "with_snippets": True,
        "limit": max(1, min(int(f1.get("limit", DEFAULT_LIMIT) or 3), 3)),
        "per_path": 2,
        "context_lines": 8,  # Rich context for understanding
    })
    ctx1, note1 = fetch_context(query, **f1)

    # Early exit: if first pass has no context AND note indicates failure/no results, fall back immediately
    has_context1 = bool((ctx1 or "").strip())
    has_error1 = note1 and ("failed" in note1.lower() or "no relevant" in note1.lower() or "no data" in note1.lower())

    if not has_context1:
        # No context at all - fall back to single-pass with the diagnostic note
        return enhance_prompt(query, **filters)

    # Pass 1: silent (no streaming)
    draft = rewrite_prompt(
        query,
        ctx1,
        note1,
        max_tokens=min(180, max_budget),
        citation_policy="snippets",
        stream=False,
    )

    # Build a grounded follow-up query from original query + allowed paths/symbols
    allowed_paths1, allowed_symbols1 = extract_allowed_citations(ctx1)
    refined_query = build_refined_query(query, allowed_paths1, allowed_symbols1)

    overlap = _token_overlap_ratio(query, draft)
    sys.stderr.write(f"[DEBUG] Unicorn draft similarity={overlap:.3f}\n")
    sys.stderr.flush()
    gate_flag = os.environ.get("CTX_DRAFT_SIM_GATE", "").strip().lower()
    if gate_flag in {"1", "true", "yes", "on"}:
        try:
            min_sim = float(os.environ.get("CTX_MIN_DRAFT_SIM", "0.4"))
        except Exception:
            min_sim = 0.4
        if overlap < min_sim:
            sys.stderr.write(f"[DEBUG] Draft similarity below threshold {min_sim:.3f}; reusing original query for pass2.\n")
            sys.stderr.flush()
            refined_query = query

    # ---- Pass 2: refine (even richer snippets, focused results)
    f2 = dict(filters)
    f2.update({
        "with_snippets": True,
        "limit": 4,
        "per_path": 1,
        "context_lines": 12,  # Very rich context for detailed grounding
    })
    ctx2, note2 = fetch_context(refined_query, **f2)

    # Check if second pass has context
    has_context2 = bool((ctx2 or "").strip())

    # If second-pass retrieval is empty, reuse first-pass context to avoid invented refs
    if not has_context2:
        ctx2 = ctx1
        note2 = note1

    # Pass 2: silent (no streaming). Use paths policy for clearer file/line anchoring.
    final = rewrite_prompt(
        draft,
        ctx2,
        note2,
        max_tokens=min(300, max_budget),
        citation_policy="paths",
        stream=False,
    )

    # ---- Pass 3: polish if clearly needed (optional via CTX_UNICORN_POLISH)
    polish_flag = os.environ.get("CTX_UNICORN_POLISH", "1").strip().lower()
    if polish_flag in {"1", "true", "yes", "on"} and _needs_polish(final):
        # Polish pass: silent (no streaming yet)
        final = rewrite_prompt(final, ctx2, note2, max_tokens=140, citation_policy="snippets", stream=False)

    # ---- Pass 4: Generate execution plan if this is a command/instruction
    plan = ""
    is_command = not query.strip().endswith("?")

    # Only generate plan if we have actual code context (not just error notes)
    has_real_context = has_context1 and bool((ctx2 or "").strip())

    import sys as _sys
    _sys.stderr.write(f"[DEBUG] Plan generation: is_command={is_command}, has_real_context={has_real_context}\n")
    _sys.stderr.flush()

    if is_command and has_real_context:
        # Generate a step-by-step execution plan based on code context
        _sys.stderr.write("[DEBUG] Generating plan...\n")
        _sys.stderr.flush()
        plan = _generate_plan(final, ctx2, note2)
        _sys.stderr.write(f"[DEBUG] Plan length: {len(plan)} chars\n")
        _sys.stderr.flush()

    # Combine enhanced prompt with plan if available
    if plan:
        output = final + "\n\n" + plan
    else:
        output = final

    # Sanitize citations on the final output and return
    allowed_paths2, _ = extract_allowed_citations(ctx2)
    return sanitize_citations(output.strip(), allowed_paths1.union(allowed_paths2))


def fetch_context(query: str, **filters) -> Tuple[str, str]:
    """Fetch repository context text plus a note describing the status.

    Defaults to header-only refs for speed unless with_snippets=True is provided.
    Falls back to context_search (with memories) if repo_search returns no hits.
    """
    with_snippets = bool(filters.get("with_snippets", False))
    # Resolve collection: explicit filter wins, then env COLLECTION_NAME, then default "codebase"
    collection_name = filters.get("collection") or os.environ.get("COLLECTION_NAME", "codebase")

    params = {
        "query": query,
        "limit": filters.get("limit", DEFAULT_LIMIT),
        "include_snippet": with_snippets,
        "context_lines": filters.get("context_lines", DEFAULT_CONTEXT_LINES),
        "collection": collection_name,
    }
    for key in ["language", "under", "path_glob", "not_glob", "kind", "symbol", "ext"]:
        if filters.get(key):
            params[key] = filters[key]

    result = call_mcp_tool("repo_search", params)
    if "error" in result:
        error_msg = result.get('error', 'Unknown error')
        sys.stderr.write(f"[DEBUG] repo_search error: {error_msg}\n")
        sys.stderr.flush()
        return "", f"Context retrieval failed: {error_msg}"

    # Surface tool-level errors (including auth failures) explicitly instead of
    # silently treating them as "no context".
    tool_err = _extract_tool_error(result)
    if tool_err:
        sys.stderr.write(f"[DEBUG] repo_search tool error: {tool_err}\n")
        sys.stderr.flush()
        return "", f"Context retrieval failed: {tool_err}"

    data = parse_mcp_response(result)
    if not data:
        sys.stderr.write("[DEBUG] repo_search returned no data\n")
        sys.stderr.flush()
        return "", "Context retrieval returned no data."

    hits = data.get("results") or []
    relevance = _estimate_query_result_relevance(query, hits)
    sys.stderr.write(f"[DEBUG] repo_search returned {len(hits)} hits (relevance={relevance:.3f})\n")
    sys.stderr.flush()

    # Optional path-level debug: sample raw paths coming back from MCP
    debug_paths_flag = os.environ.get("CTX_DEBUG_PATHS", "").strip().lower()
    if debug_paths_flag in {"1", "true", "yes", "on"} and hits:
        try:
            sample = [
                {
                    "path": h.get("path"),
                    "host_path": h.get("host_path"),
                    "container_path": h.get("container_path"),
                    "start_line": h.get("start_line"),
                    "end_line": h.get("end_line"),
                    "symbol": h.get("symbol"),
                }
                for h in hits[:5]
            ]
            sys.stderr.write("[DEBUG] repo_search sample paths:\n" + json.dumps(sample, indent=2) + "\n")
            sys.stderr.flush()
        except Exception:
            pass

    gate_flag = os.environ.get("CTX_RELEVANCE_GATE", "").strip().lower()
    if hits and gate_flag in {"1", "true", "yes", "on"}:
        try:
            min_rel = float(os.environ.get("CTX_MIN_RELEVANCE", "0.15"))
        except Exception:
            min_rel = 0.15
        if relevance < min_rel:
            sys.stderr.write(f"[DEBUG] Relevance below threshold {min_rel:.3f}; treating as no relevant context.\n")
            sys.stderr.flush()
            return "", "No relevant context found for the prompt (low retrieval relevance)."

    if not hits:
        # Memory blending: try context_search with memories as fallback
        memory_params = {
            "query": query,
            "limit": filters.get("limit", DEFAULT_LIMIT),
            "include_memories": True,
            "include_snippet": with_snippets,
            "context_lines": filters.get("context_lines", DEFAULT_CONTEXT_LINES),
            "collection": collection_name,
        }
        memory_result = call_mcp_tool("context_search", memory_params)
        if "error" not in memory_result:
            memory_data = parse_mcp_response(memory_result)
            if memory_data:
                memory_hits = memory_data.get("results") or []
                if memory_hits:
                    return format_search_results(memory_hits, include_snippets=with_snippets), "Using memories and design docs"
        return "", "No relevant context found for the prompt."

    return format_search_results(hits, include_snippets=with_snippets), ""


def rewrite_prompt(original_prompt: str, context: str, note: str, max_tokens: Optional[int], citation_policy: str = "paths", stream: bool = True) -> str:
    """Use the configured decoder (GLM or llama.cpp) to rewrite the prompt with repository context.

    Returns ONLY the improved prompt text. Raises exception if decoder fails.
    If stream=True (default), prints tokens as they arrive for instant feedback.
    """
    import sys
    ctx = (context or "").strip()
    nt = (note or "").strip()
    effective_context = ctx if ctx else (nt or "No context available.")

    # Granite 4.0 chat template with explicit rewrite-only instruction
    if (citation_policy or "paths") == "snippets":
        policy_system = (
            "Use code snippets provided in Context refs to ground the rewrite. "
            "Do NOT include file paths or line numbers. "
            "You may quote very short code fragments directly from the snippets if essential, but never use markdown or code fences. "
            "Never invent identifiers not present in the snippets. "
        )
        policy_user = (
            "When relevant, reference concrete behaviors and small code fragments from the snippets above. "
            "Do not mention file paths or line numbers. "
        )
    else:
        policy_system = (
            "If context is provided, use it to make the prompt more concrete by citing specific file paths, line ranges, and symbols that appear in the Context refs. "
            "When you cite a file, use its full path exactly as it appears in the Context refs, including all directories and prefixes (for example, '/home/.../ctx.py'), rather than shortening it to just a filename. "
            "Never invent references - only cite what appears verbatim in the Context refs. "
        )
        policy_user = (
            "If the context above contains relevant references, cite concrete file paths, line ranges, and symbols in your rewrite. "
            "When mentioning a file, use the full path exactly as shown in the Context refs (including directories), not a shortened form like 'ctx.py'. "
        )

    # Detect if we have actual code context or just a diagnostic note
    has_code_context = bool((ctx or "").strip() and not (nt and ("failed" in nt.lower() or "no relevant" in nt.lower() or "no data" in nt.lower())))

    system_msg = (
        "You are a prompt rewriter. Your ONLY job is to rewrite prompts to be more specific and detailed. "
        "CRITICAL: You must NEVER answer questions or execute commands. You must ONLY rewrite the prompt to be better and more specific. "
        "ALWAYS enhance the prompt to be more detailed and actionable. "
        + policy_system
    )

    if has_code_context:
        # We have real code context - encourage using it
        system_msg += (
            "Use the provided context to make the prompt more concrete and specific. "
            "Your rewrite must be at least two short paragraphs separated by a single blank line. "
            "For questions: rewrite as more specific questions. For commands/instructions: rewrite as more detailed, specific instructions with concrete targets. "
            "Each paragraph should explore different aspects of the topic. "
            "Output format: plain text only, no markdown, no code fences, no answers, no explanations."
        )
    else:
        # No code context - stay generic and don't invent details
        system_msg += (
            "IMPORTANT: No code context is available for this query. "
            "Do NOT invent file paths, line numbers, function names, or other specific code references. "
            "Instead, rewrite the prompt to be more general and exploratory, asking about concepts, approaches, and best practices. "
            "Your rewrite must be at least two short paragraphs separated by a single blank line. "
            "For questions: expand into multiple related questions about the topic. For commands/instructions: expand into general guidance about the task. "
            "Stay generic - do not hallucinate specific files, functions, or code locations. "
            "Output format: plain text only, no markdown, no code fences, no answers, no explanations."
        )

    label = "with snippets" if "\n    " in effective_context else "headers only"
    user_msg = (
        f"Context refs ({label}):\n{effective_context}\n\n"
        f"Original prompt: {(original_prompt or '').strip()}\n\n"
        "Rewrite this as a more specific, detailed prompt using at least two short paragraphs separated by a blank line. "
        + policy_user
    )

    if has_code_context:
        user_msg += (
            "Use the context above to make the rewrite concrete and specific. "
            "For questions: make them more specific and multi-faceted (each paragraph should be a question ending with '?'). "
            "For commands/instructions: make them more detailed and concrete (specify exact functions, parameters, edge cases to handle). "
        )
    else:
        user_msg += (
            "Since no code context is available, keep the rewrite general and exploratory. "
            "Do NOT invent specific file paths, line numbers, or function names. "
            "For questions: expand into related conceptual questions. For commands/instructions: provide general guidance about the task. "
        )

    user_msg += (
        "Remember: ONLY rewrite the prompt - do NOT answer questions or execute commands. "
        "Avoid generic phrasing. No markdown or code fences."
    )

    # Apply user preferences if config exists
    prefs = _load_user_preferences()
    system_msg, user_msg = _apply_user_preferences(system_msg, user_msg, prefs)

    # Override stream setting from preferences if specified
    if prefs.get("streaming") is not None:
        stream = prefs.get("streaming")

    # Check which decoder runtime to use
    runtime_kind = str(os.environ.get("REFRAG_RUNTIME", "llamacpp")).strip().lower()

    if runtime_kind == "glm":
        from refrag_glm import GLMRefragClient  # type: ignore
        client = GLMRefragClient()

        # GLM uses OpenAI-style chat completions, convert context to user prompt format
        # Note: For GLM, we need to convert the meta_prompt format to simple user message
        user_msg = (
            f"Context refs:\n{effective_context}\n\n"
            f"Original prompt: {(original_prompt or '').strip()}\n\n"
            "Rewrite this as a more specific, detailed prompt using at least two short paragraphs separated by a blank line. "
        )

        if has_code_context:
            user_msg += (
                "Use the context above to make the rewrite concrete and specific. "
                "For questions: make them more specific and multi-faceted (each paragraph should be a question ending with '?'). "
                "For commands/instructions: make them more detailed and concrete (specify exact functions, parameters, edge cases to handle). "
            )
        else:
            user_msg += (
                "Since no code context is available, keep the rewrite general and exploratory. "
                "Do NOT invent specific file paths, line numbers, or function names. "
                "For questions: expand into related conceptual questions. For commands/instructions: provide general guidance about the task. "
            )

        # GLM API call
        response = client.client.chat.completions.create(
            model=os.environ.get("GLM_MODEL", "glm-4.6"),
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            max_tokens=int(max_tokens or DEFAULT_REWRITE_TOKENS),
            temperature=0.45,
            stream=stream
        )

        enhanced = ""
        if stream:
            # Streaming mode for GLM
            for chunk in response:
                if chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    sys.stdout.write(token)
                    sys.stdout.flush()
                    enhanced += token
            sys.stdout.write("\n")
            sys.stdout.flush()
        else:
            # Non-streaming mode for GLM
            enhanced = response.choices[0].message.content

    else:
        # Use local decoder (llama.cpp by default; Ollama supported when DECODER_URL points to /api/chat)
        meta_prompt = (
            "<|start_of_role|>system<|end_of_role|>" + system_msg + "<|end_of_text|>\n"
            "<|start_of_role|>user<|end_of_role|>" + user_msg + "<|end_of_text|>\n"
            "<|start_of_role|>assistant<|end_of_role|>"
        )

        decoder_url = DECODER_URL
        # Safety: only allow local decoder hosts
        parsed = urlparse(decoder_url)
        if parsed.hostname not in {"localhost", "127.0.0.1", "host.docker.internal"}:
            raise ValueError(f"Unsafe decoder host: {parsed.hostname}")

        lowered_url = decoder_url.lower()
        is_ollama = (
            "ollama" in lowered_url
            or "/api/chat" in lowered_url
            or "/api/generate" in lowered_url
            or "/v1/chat/completions" in lowered_url
        )

        enhanced = ""
        try:
            if is_ollama:
                model = (
                    os.environ.get("DECODER_MODEL", "").strip()
                    or os.environ.get("OLLAMA_MODEL", "").strip()
                    or "llama3"
                )
                payload = {
                    "model": model,
                    "stream": stream,
                    "options": {"temperature": 0.45},
                }
                if max_tokens:
                    payload["options"]["num_predict"] = int(max_tokens)
                if "/api/chat" in lowered_url or "/v1/chat/completions" in lowered_url:
                    payload["messages"] = [
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                    ]
                else:
                    payload["prompt"] = f"{system_msg}\n\n{user_msg}"

                req = request.Request(
                    decoder_url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                )

                if stream:
                    with request.urlopen(req, timeout=DECODER_TIMEOUT) as resp:
                        for line in resp:
                            line_str = line.decode("utf-8", errors="ignore").strip()
                            if not line_str or line_str.startswith(":"):
                                continue
                            if line_str.startswith("data: "):
                                line_str = line_str[6:]
                            try:
                                chunk = json.loads(line_str)
                            except json.JSONDecodeError:
                                continue
                            token = ""
                            if isinstance(chunk, dict):
                                token = (
                                    (chunk.get("message") or {}).get("content", "")
                                    or chunk.get("response", "")
                                )
                            if token:
                                sys.stdout.write(token)
                                sys.stdout.flush()
                                enhanced += token
                            if chunk.get("done") or chunk.get("stop"):
                                break
                    sys.stdout.write("\n")
                    sys.stdout.flush()
                else:
                    with request.urlopen(req, timeout=DECODER_TIMEOUT) as resp:
                        raw = resp.read().decode("utf-8", errors="ignore")
                        data = json.loads(raw or "{}")
                        if isinstance(data, dict):
                            enhanced = (
                                (data.get("message") or {}).get("content")
                                or data.get("response")
                                or ((data.get("choices") or [{}])[0].get("message") or {}).get("content")
                            )
                        else:
                            enhanced = None
            else:
                payload = {
                    "prompt": meta_prompt,
                    "n_predict": int(max_tokens or DEFAULT_REWRITE_TOKENS),
                    "temperature": 0.45,
                    "stream": stream,
                }

                req = request.Request(
                    decoder_url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                )

                if stream:
                    # Streaming mode: print tokens as they arrive for instant feedback
                    with request.urlopen(req, timeout=DECODER_TIMEOUT) as resp:
                        for line in resp:
                            line_str = line.decode("utf-8", errors="ignore").strip()
                            if not line_str or line_str.startswith(":"):
                                continue
                            if line_str.startswith("data: "):
                                line_str = line_str[6:]
                            try:
                                chunk = json.loads(line_str)
                                token = chunk.get("content", "")
                                if token:
                                    sys.stdout.write(token)
                                    sys.stdout.flush()
                                    enhanced += token
                                if chunk.get("stop", False):
                                    break
                            except json.JSONDecodeError as e:
                                # Warn once per malformed line but keep streaming the final output only
                                sys.stderr.write(f"[WARN] decoder stream JSON decode failed: {str(e)}\n")
                                sys.stderr.flush()
                                continue
                    sys.stdout.write("\n")
                    sys.stdout.flush()
                else:
                    # Non-streaming mode: wait for full response
                    with request.urlopen(req, timeout=DECODER_TIMEOUT) as resp:
                        raw = resp.read().decode("utf-8", errors="ignore")
                        data = json.loads(raw)

                        # Extract content from llama.cpp response
                        enhanced = (
                            (data.get("content") if isinstance(data, dict) else None)
                            or ((data.get("choices") or [{}])[0].get("content") if isinstance(data, dict) else None)
                            or ((data.get("choices") or [{}])[0].get("text") if isinstance(data, dict) else None)
                            or (data.get("generated_text") if isinstance(data, dict) else None)
                            or (data.get("text") if isinstance(data, dict) else None)
                        )
        except Exception as e:
            body_detail = ""
            if isinstance(e, HTTPError):
                try:
                    body_detail = e.read().decode("utf-8", errors="ignore").strip()
                except Exception:
                    body_detail = ""
            msg = f"[ERROR] Decoder call to {decoder_url} failed: {type(e).__name__}: {e}"
            if body_detail:
                msg += f" | body: {body_detail}"
            sys.stderr.write(msg + "\n")
            sys.stderr.flush()
            raise

    # Normalize and strip formatting / template artifacts from decoder output
    enhanced = (enhanced or "")
    enhanced = enhanced.replace("```", "").replace("`", "")
    # Remove stray chat-template tags like <|user|>, <|assistant|>, etc.
    enhanced = re.sub(r"<\|[^|>]+?\|>", "", enhanced)
    enhanced = enhanced.strip()

    if not enhanced:
        raise ValueError("Decoder returned empty response")

    # Enforce at least two question paragraphs, then deduplicate and cap paragraphs
    enhanced = _ensure_two_paragraph_questions(enhanced)
    enhanced = _dedup_paragraphs(enhanced, max_paragraphs=3)
    return enhanced





def build_final_output(
    rewritten_prompt: str, context: str, note: str, include_context: bool
) -> str:
    """Combine LLM rewrite with optional supporting context for downstream tools."""
    improved = rewritten_prompt.strip() or "No rewrite generated."
    if not include_context:
        return improved

    context_block = context.strip() if context.strip() else (note or "No supporting context.")

    return f"""# Improved Prompt
{improved}

---

# Supporting Context
{context_block}
"""


def main():
    parser = argparse.ArgumentParser(
        description="Context-aware prompt enhancer - rewrites questions and commands with codebase context",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Questions: enhanced with specific details
  ctx "how does hybrid search work?"

  # Commands: enhanced with concrete implementation steps
  ctx "refactor ctx.py to improve modularity"

  # Unicorn mode: staged 2–3 pass enhancement for best results
  ctx --unicorn "refactor ctx.py"

  # Detail mode: include code snippets (slower but richer)
  ctx --detail "explain the caching logic"

  # Pipe to LLM or clipboard
  ctx --cmd llm "explain the caching logic"
  ctx --cmd pbcopy --language python "fix bug in watcher"
        """
    )

    parser.add_argument("query", help="Your question or command to enhance")

    # Command execution
    parser.add_argument("--cmd", "-c", help="Command to pipe enhanced prompt to (e.g., llm, pbcopy)")
    parser.add_argument("--with-context", action="store_true",
                        help="Append supporting context after the improved prompt")
    parser.add_argument("--unicorn", action="store_true",
                        help="One-size 'amazing' mode: staged 2–3 calls for best prompts (keeps defaults unchanged)")

    # Search filters
    parser.add_argument("--language", "-l", help="Filter by language (e.g., python, typescript)")
    parser.add_argument("--under", "-u", help="Filter by path prefix (e.g., scripts/)")
    parser.add_argument("--path-glob", help="Filter by path glob pattern")
    parser.add_argument("--not-glob", help="Exclude paths matching glob pattern")
    parser.add_argument("--kind", help="Filter by symbol kind (e.g., function, class)")
    parser.add_argument("--symbol", help="Filter by symbol name")
    parser.add_argument("--ext", help="Filter by file extension")
    parser.add_argument("--collection", help="Override collection name (default: env COLLECTION_NAME)")

    # Output control
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT,
                       help=f"Max results (default: {DEFAULT_LIMIT})")
    parser.add_argument("--context-lines", type=int, default=DEFAULT_CONTEXT_LINES,
                       help=f"Context lines for snippets (default: {DEFAULT_CONTEXT_LINES})")
    parser.add_argument("--per-path", type=int,
                       help="Limit results per file (default: server setting)")
    parser.add_argument("--rewrite-max-tokens", type=int, default=DEFAULT_REWRITE_TOKENS,
                       help=f"Max tokens for LLM rewrite (default: {DEFAULT_REWRITE_TOKENS})")

    # Detail mode
    parser.add_argument("--detail", action="store_true",
                       help="Include short code snippets for richer rewrites (slower but more specific; auto-clamps to limit=4, per_path=1)")

    args = parser.parse_args()

    # Build filter dict
    filters = {
        "limit": args.limit,
        "context_lines": args.context_lines,
        "language": args.language,
        "under": args.under,
        "path_glob": args.path_glob,
        "not_glob": args.not_glob,
        "kind": args.kind,
        "symbol": args.symbol,
        "ext": args.ext,
        "collection": args.collection,
        "per_path": args.per_path,
        "with_snippets": args.detail,
        "rewrite_options": {
            "max_tokens": args.rewrite_max_tokens,
        },
    }

    # If detail mode is on and context_lines equals the default (0), bump to 1 for a short snippet
    if args.detail and args.context_lines == DEFAULT_CONTEXT_LINES:
        filters["context_lines"] = 1
    # Clamp result counts in detail mode for latency
    if args.detail:
        try:
            filters["limit"] = max(1, min(int(filters.get("limit", DEFAULT_LIMIT)), 4))
        except Exception:
            filters["limit"] = 4
        filters["per_path"] = 1

    # Remove None values
    filters = {k: v for k, v in filters.items() if v is not None}

    try:
        # Enhance prompt
        if args.unicorn:
            output = enhance_unicorn(args.query, **filters)
        else:
            context_text, context_note = fetch_context(args.query, **filters)

            # Derive allowed paths from the formatted context so we can validate/normalize
            # any file-like mentions in the final rewrite.
            allowed_paths, _ = extract_allowed_citations(context_text)

            require_ctx_flag = os.environ.get("CTX_REQUIRE_CONTEXT", "").strip().lower()
            if require_ctx_flag in {"1", "true", "yes", "on"}:
                has_real_context = bool((context_text or "").strip()) and not (
                    context_note and (
                        "failed" in context_note.lower()
                        or "no relevant" in context_note.lower()
                        or "no data" in context_note.lower()
                    )
                )
                if not has_real_context:
                    output = (args.query or "").strip()
                else:
                    rewritten = rewrite_prompt(args.query, context_text, context_note, max_tokens=args.rewrite_max_tokens)
                    output = sanitize_citations(rewritten.strip(), allowed_paths)
            else:
                rewritten = rewrite_prompt(args.query, context_text, context_note, max_tokens=args.rewrite_max_tokens)
                output = sanitize_citations(rewritten.strip(), allowed_paths)

        if args.cmd:
            subprocess.run(args.cmd, input=output.encode("utf-8"), shell=True, check=False)
        else:
            print(output)

    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
