"""
Feature-flagged adapter for decoder-side ReFRAG using llama.cpp.

Notes:
- This file is intentionally lightweight and safe: it does nothing unless
  REFRAG_DECODER=1 (or true/yes/on).
- Actual soft-embedding / KV injection requires a patched llama.cpp build;
  wiring will be added behind this adapter once the runtime is available.
"""

from __future__ import annotations
import os
from typing import Any, Dict, Optional
import threading
import contextlib
import time


def _bool_env(name: str, default: str = "0") -> bool:
    return str(os.environ.get(name, default)).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def is_decoder_enabled() -> bool:
    return _bool_env("REFRAG_DECODER", "0")


def get_runtime_kind() -> str:
    return str(os.environ.get("REFRAG_RUNTIME", "llamacpp")).strip().lower()


def get_phi_path() -> str:
    return str(os.environ.get("REFRAG_PHI_PATH", "")).strip()


def get_encoder_model() -> str:
    return str(os.environ.get("REFRAG_ENCODER_MODEL", "BAAI/bge-base-en-v1.5")).strip()


def get_sense_policy() -> str:
    return str(os.environ.get("REFRAG_SENSE", "heuristic")).strip().lower()


def _max_parallel() -> int:
    try:
        val = int(os.environ.get("LLAMACPP_MAX_PARALLEL", "").strip() or "2")
        return max(1, val)
    except Exception:
        return 2


_LLAMACPP_PARALLEL = _max_parallel()
_LLAMACPP_SLOT = threading.Semaphore(_LLAMACPP_PARALLEL)
_LLAMACPP_SLOT_LOCK = threading.Lock()


def _refresh_parallel_semaphore() -> None:
    """Refresh semaphore when LLAMACPP_MAX_PARALLEL changes at runtime."""
    try:
        desired = _max_parallel()
    except Exception:
        desired = 2
    with _LLAMACPP_SLOT_LOCK:
        global _LLAMACPP_SLOT, _LLAMACPP_PARALLEL
        if desired == _LLAMACPP_PARALLEL:
            return
        _LLAMACPP_PARALLEL = desired
        _LLAMACPP_SLOT = threading.Semaphore(desired)


@contextlib.contextmanager
def _parallel_slot():
    """Context manager honoring LLAMACPP_MAX_PARALLEL."""
    _refresh_parallel_semaphore()
    slot = globals().get("_LLAMACPP_SLOT")
    if not isinstance(slot, threading.Semaphore):
        slot = threading.Semaphore(_max_parallel())
        globals()["_LLAMACPP_SLOT"] = slot
    acquired = slot.acquire(timeout=float(os.environ.get("LLAMACPP_ACQUIRE_TIMEOUT", "30") or 30))
    if not acquired:
        raise RuntimeError("llama.cpp saturated: parallel limit reached")
    try:
        yield
    finally:
        slot.release()


_WARM_CHECKED = False


def _maybe_warm(base_url: str) -> None:
    global _WARM_CHECKED
    if _WARM_CHECKED:
        return
    _WARM_CHECKED = True
    if not _bool_env("LLAMACPP_AUTOWARM", "1"):
        return
    try:
        from urllib import request

        req = request.Request(base_url.rstrip("/") + "/health", method="GET")
        timeout = float(os.environ.get("LLAMACPP_WARM_TIMEOUT", "3") or 3)
        with request.urlopen(req, timeout=timeout):
            pass
    except Exception:
        # Ignore warm failures; decoder calls will raise later if truly unavailable
        return


class LlamaCppRefragClient:
    """Feature-flagged client for llama.cpp decoder.

    Modes:
      - prompt (default): builds a compressed textual prompt from merged spans.
      - soft: reserved for patched server with soft-embedding support (NotImplemented).
    """

    def __init__(self, base_url: Optional[str] = None) -> None:
        if base_url:
            self.base_url = base_url
        else:
            # Smart URL resolution: GPU vs Docker based on USE_GPU_DECODER flag
            use_gpu = str(os.environ.get("USE_GPU_DECODER", "0")).strip().lower()
            if use_gpu in {"1", "true", "yes", "on"}:
                # Use native GPU-accelerated server
                # Use localhost when running on host, host.docker.internal when in container
                if os.path.exists("/.dockerenv"):
                    self.base_url = "http://host.docker.internal:8081"
                else:
                    self.base_url = "http://localhost:8081"
            else:
                # Use configured LLAMACPP_URL (default: Docker CPU-only)
                self.base_url = os.environ.get("LLAMACPP_URL", "http://localhost:8080")
        _maybe_warm(self.base_url)
        if get_runtime_kind() != "llamacpp":
            raise ValueError(
                "REFRAG_RUNTIME must be 'llamacpp' for LlamaCppRefragClient"
            )

    def _post(self, path: str, json_payload: Dict[str, Any]) -> Dict[str, Any]:
        import json as _json
        from urllib import request

        req = request.Request(self.base_url.rstrip("/") + path, method="POST")
        req.add_header("Content-Type", "application/json")
        data = _json.dumps(json_payload).encode("utf-8")
        import os as _os
        _timeout = float(_os.environ.get("LLAMACPP_TIMEOUT_SEC", "60") or 60)
        with request.urlopen(req, data=data, timeout=_timeout) as resp:
            body = resp.read()
        return _json.loads(body.decode("utf-8"))

    def generate_with_soft_embeddings(
        self,
        prompt: str,
        soft_embeddings: Optional[list[list[float]]] = None,
        max_tokens: int = 256,
        **gen_kwargs: Any,
    ) -> str:
        if not is_decoder_enabled():
            raise RuntimeError("Decoder path disabled: set REFRAG_DECODER=1 to enable")
        mode = os.environ.get("REFRAG_DECODER_MODE", "prompt").strip().lower()
        if mode == "soft":
            if not soft_embeddings:
                raise ValueError("soft mode requires soft_embeddings")
            _rp = float(os.environ.get("DECODER_REPEAT_PENALTY", str(gen_kwargs.get("repeat_penalty", 1.1))))
            _rln = int(os.environ.get("DECODER_REPEAT_LAST_N", str(gen_kwargs.get("repeat_last_n", 128))))
            _pp = float(os.environ.get("DECODER_PRESENCE_PENALTY", str(gen_kwargs.get("presence_penalty", 0.0))))
            _fp = float(os.environ.get("DECODER_FREQUENCY_PENALTY", str(gen_kwargs.get("frequency_penalty", 0.0))))
            payload = {
                "prompt": prompt,
                "soft_embeddings": soft_embeddings,
                "scale": float(os.environ.get("REFRAG_SOFT_SCALE", "1.0") or 1.0),
                "n_predict": int(gen_kwargs.get("max_tokens", max_tokens)),
                "temperature": float(gen_kwargs.get("temperature", 0.2)),
                "top_k": int(gen_kwargs.get("top_k", 40)),
                "top_p": float(gen_kwargs.get("top_p", 0.95)),
                "repeat_penalty": _rp,
                "repeat_last_n": _rln,
                "presence_penalty": _pp,
                "frequency_penalty": _fp,
                "stop": gen_kwargs.get("stop") or [],
            }
            try:
                with _parallel_slot():
                    _start = time.time()
                    res = self._post("/soft_completion", payload)
                    elapsed = time.time() - _start
                    os.environ.setdefault(
                        "LLAMACPP_LAST_LATENCY_SEC", f"{elapsed:.3f}"
                    )
            except Exception as e:
                raise RuntimeError(f"llama.cpp soft_completion failed: {e}")
            return (res.get("content") or res.get("generation") or "").strip()
        # Prompt mode: send a normal completion request using a compressed prompt
        # Allow repetition controls; fall back to env if not passed
        _rp = float(os.environ.get("DECODER_REPEAT_PENALTY", str(gen_kwargs.get("repeat_penalty", 1.1))))
        _rln = int(os.environ.get("DECODER_REPEAT_LAST_N", str(gen_kwargs.get("repeat_last_n", 128))))
        _pp = float(os.environ.get("DECODER_PRESENCE_PENALTY", str(gen_kwargs.get("presence_penalty", 0.0))))
        _fp = float(os.environ.get("DECODER_FREQUENCY_PENALTY", str(gen_kwargs.get("frequency_penalty", 0.0))))
        payload = {
            "prompt": prompt,
            "n_predict": int(gen_kwargs.get("max_tokens", max_tokens)),
            # fast, deterministic-ish defaults; callers can override via gen_kwargs
            "temperature": float(gen_kwargs.get("temperature", 0.2)),
            "top_k": int(gen_kwargs.get("top_k", 40)),
            "top_p": float(gen_kwargs.get("top_p", 0.95)),
            "repeat_penalty": _rp,
            "repeat_last_n": _rln,
            "presence_penalty": _pp,
            "frequency_penalty": _fp,
            "stop": gen_kwargs.get("stop") or [],
        }
        try:
            with _parallel_slot():
                _start = time.time()
                res = self._post("/completion", payload)
                elapsed = time.time() - _start
                os.environ.setdefault(
                    "LLAMACPP_LAST_LATENCY_SEC", f"{elapsed:.3f}"
                )
        except Exception as e:
            raise RuntimeError(f"llama.cpp completion failed: {e}")
        # llama.cpp server returns { 'content': '...' } or { 'token': ... } streams; we expect non-stream
        txt = (res.get("content") or res.get("generation") or "").strip()
        return txt
