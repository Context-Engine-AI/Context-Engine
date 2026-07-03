# ---------------------------------------------------------------------------
# CRITICAL: OpenLit must be initialized BEFORE any qdrant_client imports
# to properly instrument vector DB calls.
# ---------------------------------------------------------------------------
import os
import sys as _sys

# Ensure repo roots are importable so 'scripts' resolves inside container
_roots_env = os.environ.get("WORK_ROOTS", "")
_roots = [p.strip() for p in _roots_env.split(",") if p.strip()] or ["/work", "/app"]
for _root in _roots:
    if _root and _root not in _sys.path:
        _sys.path.insert(0, _root)

# Now import OpenLit init (before any other scripts imports that may use qdrant)
try:
    from scripts import openlit_init  # noqa: F401 - triggers early instrumentation
except ImportError:
    pass  # OpenLit not available

import json
import threading
from datetime import datetime
from typing import Any, Dict, Optional, List, Union
from weakref import WeakKeyDictionary


# FastMCP server and request Context (ctx) for per-connection state
try:
    from mcp.server.fastmcp import FastMCP, Context  # type: ignore
    from mcp.server.transport_security import TransportSecuritySettings  # type: ignore
except Exception:
    from mcp.server.fastmcp import FastMCP  # type: ignore
    Context = Any  # type: ignore
    TransportSecuritySettings = None  # type: ignore

from scripts.mcp_auth import (
    require_auth_session as _require_auth_session,
    require_collection_access as _require_collection_access,
)

from qdrant_client import QdrantClient, models

# Import connection pooling for proper resource management
try:
    from scripts.qdrant_client_manager import (
        get_qdrant_client,
        return_qdrant_client,
        pooled_qdrant_client,
    )
    _POOL_AVAILABLE = True
except ImportError:
    _POOL_AVAILABLE = False

# Env
QDRANT_URL = os.environ.get("QDRANT_URL", "http://qdrant:6333")
DEFAULT_COLLECTION = (
    os.environ.get("DEFAULT_COLLECTION")
    or os.environ.get("COLLECTION_NAME")
    or "codebase"
)
LEX_VECTOR_NAME = os.environ.get("LEX_VECTOR_NAME", "lex")
LEX_VECTOR_DIM = int(os.environ.get("LEX_VECTOR_DIM", "4096") or 4096)
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")

# Minimal embedding via fastembed (CPU)

# Single-process embedding model cache (avoid re-initializing fastembed on each call)

# Use shared utils for consistent vector naming and lexical hashing
from scripts.utils import sanitize_vector_name as _sanitize_vector_name
from scripts.utils import lex_hash_vector_text as _lex_hash_vector_text

VECTOR_NAME = _sanitize_vector_name(EMBEDDING_MODEL)

# I/O-safety knobs for memory server behavior
# These env vars allow tuning startup latency vs. first-call latency, especially important
# on slow storage backends (e.g., Ceph + HDD). See comments below for rationale.
MEMORY_ENSURE_ON_START = str(os.environ.get("MEMORY_ENSURE_ON_START", "1")).strip().lower() in {"1", "true", "yes", "on"}
MEMORY_COLD_SKIP_DENSE = str(os.environ.get("MEMORY_COLD_SKIP_DENSE", "0")).strip().lower() in {"1", "true", "yes", "on"}
MEMORY_PROBE_EMBED_DIM = str(os.environ.get("MEMORY_PROBE_EMBED_DIM", "1")).strip().lower() in {"1", "true", "yes", "on"}
try:
    MEMORY_VECTOR_DIM = int(os.environ.get("MEMORY_VECTOR_DIM") or os.environ.get("EMBED_DIM") or "768")
except Exception:
    MEMORY_VECTOR_DIM = 768

# ---------------------------------------------------------------------------
# Embedding Model Management
# ---------------------------------------------------------------------------
# Use the centralized embedder from scripts.embedder for consistent caching.
# This eliminates duplicate model loading and ensures consistent behavior.

# Reference to the centralized embedder for cold-skip detection
try:
    from scripts.embedder import get_embedding_model as _centralized_get_embedding_model
    from scripts.embedder import is_model_cached as _is_model_cached
    _EMBEDDER_AVAILABLE = True
except ImportError:
    _EMBEDDER_AVAILABLE = False
    def _is_model_cached(model_name: str = "") -> bool:  # type: ignore[misc]
        return False  # Fallback: assume not cached

def _get_embedding_model():
    """Get the embedding model using the centralized embedder.

    Uses scripts.embedder.get_embedding_model() which provides:
    - Thread-safe lazy loading with double-checked locking
    - Qwen3 model support with feature flags
    - Automatic cache invalidation on corrupted downloads
    """
    if _EMBEDDER_AVAILABLE:
        return _centralized_get_embedding_model(EMBEDDING_MODEL)

    # Fallback for environments without centralized embedder (rare)
    from fastembed import TextEmbedding
    return TextEmbedding(model_name=EMBEDDING_MODEL)

# Track ensured collections to reduce redundant ensure calls.
# RATIONALE: Avoid repeated Qdrant network calls for the same collection.
_ENSURED = set()

def _ensure_once(name: str) -> bool:
    """Ensure collection exists, but only once per process (cached result)."""
    if name in _ENSURED:
        return True
    try:
        _ensure_collection(name)
        _ENSURED.add(name)
        return True
    except Exception:
        return False

# Disable DNS rebinding protection - breaks Docker internal networking (Host: mcp:8000)
_security_settings = (
    TransportSecuritySettings(enable_dns_rebinding_protection=False)
    if TransportSecuritySettings
    else None
)
mcp = FastMCP(name="memory-server", transport_security=_security_settings)

# Capture tool registry automatically by wrapping the decorator once
_TOOLS_REGISTRY: list[dict] = []
try:
    _orig_tool = mcp.tool
    def _tool_capture_wrapper(*dargs, **dkwargs):
        orig_deco = _orig_tool(*dargs, **dkwargs)
        def _inner(fn):
            try:
                _TOOLS_REGISTRY.append({
                    "name": dkwargs.get("name") or getattr(fn, "__name__", ""),
                    "description": (getattr(fn, "__doc__", None) or "").strip(),
                })
            except Exception:
                pass
            return orig_deco(fn)
        return _inner
    mcp.tool = _tool_capture_wrapper  # type: ignore
except Exception:
    pass


def _relax_var_kwarg_defaults() -> None:
    """Allow tools that rely on **kwargs compatibility shims to be invoked without
    callers supplying an explicit 'kwargs' or 'arguments' field.
    
    This patches Pydantic models to add default_factory=dict for kwargs/arguments,
    making them optional instead of required.
    """
    try:
        from pydantic_core import PydanticUndefined as _PydanticUndefined  # type: ignore
    except Exception:  # pragma: no cover - defensive
        class _Sentinel:  # type: ignore
            pass
        _PydanticUndefined = _Sentinel()  # type: ignore

    try:
        tool_manager = getattr(mcp, "_tool_manager", None)
        tools = getattr(tool_manager, "_tools", {}) if tool_manager is not None else {}
    except Exception:
        tools = {}

    for tool in tools.values():
        try:
            model = getattr(tool.fn_metadata, "arg_model", None)
            if model is None:
                continue
            fields = getattr(model, "model_fields", {})
            changed = False
            for key in ("kwargs", "arguments"):
                fld = fields.get(key)
                if fld is None:
                    continue
                default = getattr(fld, "default", None)
                default_factory = getattr(fld, "default_factory", None)
                if default is _PydanticUndefined and default_factory is None:
                    try:
                        fld.default_factory = dict  # type: ignore[attr-defined]
                    except Exception:
                        fld.default_factory = lambda: {}  # type: ignore
                    fld.default = None
                    changed = True
            if changed:
                try:
                    model.model_rebuild(force=True)
                except Exception:
                    pass
        except Exception:
            continue


HOST = os.environ.get("FASTMCP_HOST", "0.0.0.0")
PORT = int(os.environ.get("FASTMCP_PORT", "8000") or 8000)

# Lightweight readiness endpoint on a separate health port (non-MCP), optional
try:
    HEALTH_PORT = int(os.environ.get("FASTMCP_HEALTH_PORT", "18000") or 18000)
except Exception:
    HEALTH_PORT = 18000

# In-memory session defaults (legacy token-based)
_SESSION_LOCK = threading.Lock()
SESSION_DEFAULTS: Dict[str, Dict[str, Any]] = {}
# In-memory per-connection defaults keyed by ctx.session (no token required)
_SESSION_CTX_LOCK = threading.Lock()
SESSION_DEFAULTS_BY_SESSION: "WeakKeyDictionary[Any, Dict[str, Any]]" = WeakKeyDictionary()




def _start_readyz_server():
    try:
        from http.server import BaseHTTPRequestHandler, HTTPServer

        class H(BaseHTTPRequestHandler):
            def do_GET(self):
                try:
                    if self.path == "/readyz":
                        self.send_response(200)
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        payload = {"ok": True, "app": "memory-server"}
                        self.wfile.write((json.dumps(payload)).encode("utf-8"))
                    elif self.path == "/tools":
                        self.send_response(200)
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        payload = {"ok": True, "tools": _TOOLS_REGISTRY}
                        self.wfile.write((json.dumps(payload)).encode("utf-8"))
                    else:
                        self.send_response(404)
                        self.end_headers()
                except Exception:
                    try:
                        self.send_response(500)
                        self.end_headers()
                    except Exception:
                        pass

            def log_message(self, *args, **kwargs):
                return

        srv = HTTPServer((HOST, HEALTH_PORT), H)
        th = threading.Thread(target=srv.serve_forever, daemon=True)
        th.start()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Qdrant Client Management
# ---------------------------------------------------------------------------
# Use connection pooling when available, fallback to creating clients on-demand.
# This prevents socket exhaustion under load and improves connection reuse.

def _get_qdrant_client() -> QdrantClient:
    """Get a Qdrant client from pool or create one."""
    if _POOL_AVAILABLE:
        return get_qdrant_client(
            url=QDRANT_URL,
            api_key=os.environ.get("QDRANT_API_KEY")
        )
    return QdrantClient(url=QDRANT_URL, api_key=os.environ.get("QDRANT_API_KEY"))


def _return_qdrant_client(client: QdrantClient):
    """Return a client to the pool, or close it if pooling unavailable."""
    if client is None:
        return
    if _POOL_AVAILABLE:
        return_qdrant_client(client)
    else:
        # Fallback path: close client to avoid socket leak
        try:
            client.close()
        except Exception:
            pass  # Best effort cleanup


# Ensure collection exists with dual vectors


def _ensure_collection(name: str):
    """Create collection if missing.

    Default behavior mirrors the original implementation for PR compatibility:
    - Probe the embedding model to detect the dense vector dimension (MEMORY_PROBE_EMBED_DIM=1)
    - Eager ensure on startup (MEMORY_ENSURE_ON_START=1)

    For slow storage backends (e.g., Ceph + HDD), set the following in your env:
    - MEMORY_PROBE_EMBED_DIM=0  -> skip model probing; use MEMORY_VECTOR_DIM/EMBED_DIM
    - MEMORY_ENSURE_ON_START=0  -> ensure lazily on first tool call
    """
    client = _get_qdrant_client()
    try:
        client.get_collection(name)
        return True
    except Exception:
        pass
    finally:
        _return_qdrant_client(client)

    # Choose dense dimension based on config: probe (default) vs env-configured
    if MEMORY_PROBE_EMBED_DIM:
        try:
            # Probe dimension without populating the shared model cache.
            # This preserves the "cache loads on first tool call" behavior and
            # keeps MEMORY_COLD_SKIP_DENSE semantics unchanged.
            from fastembed import TextEmbedding
            _model_probe = TextEmbedding(model_name=EMBEDDING_MODEL)
            _dense_vec = next(_model_probe.embed(["probe"]))
            if hasattr(_dense_vec, "tolist"):
                dense_dim = len(_dense_vec.tolist())
            else:
                try:
                    dense_dim = len(_dense_vec)
                except Exception:
                    dense_dim = int(os.environ.get("MEMORY_VECTOR_DIM") or os.environ.get("EMBED_DIM") or "768")
        except Exception:
            # Fallback to env-configured dimension if probing fails
            try:
                dense_dim = int(os.environ.get("MEMORY_VECTOR_DIM") or os.environ.get("EMBED_DIM") or "768")
            except Exception:
                dense_dim = 768
    else:
        dense_dim = int(MEMORY_VECTOR_DIM or 768)

    vectors_cfg = {
        VECTOR_NAME: models.VectorParams(size=int(dense_dim or 768), distance=models.Distance.COSINE),
        LEX_VECTOR_NAME: models.VectorParams(size=LEX_VECTOR_DIM, distance=models.Distance.COSINE),
    }

    # Add mini vector for ReFRAG mode (same logic as ingest_code.py)
    try:
        if os.environ.get("REFRAG_MODE", "").strip().lower() in {
            "1", "true", "yes", "on"
        }:
            mini_vector_name = os.environ.get("MINI_VECTOR_NAME", "mini")
            mini_vec_dim = int(os.environ.get("MINI_VEC_DIM", "64"))
            vectors_cfg[mini_vector_name] = models.VectorParams(
                size=mini_vec_dim,
                distance=models.Distance.COSINE,
            )
    except Exception:
        pass

    # Get a fresh client for collection creation
    client = _get_qdrant_client()
    try:
        client.create_collection(
            collection_name=name,
            vectors_config=vectors_cfg,
            hnsw_config=models.HnswConfigDiff(m=16, ef_construct=256),
        )
        vector_names = list(vectors_cfg.keys())
        print(f"[MEMORY_SERVER] Created collection '{name}' with vectors: {vector_names}")
        return True
    finally:
        _return_qdrant_client(client)


# Optional eager collection ensure on startup (enabled by default for backward compatibility).
# Set MEMORY_ENSURE_ON_START=0 to defer ensure to first tool call (recommended on slow storage).
if MEMORY_ENSURE_ON_START:
    try:
        _ensure_collection(DEFAULT_COLLECTION)
    except Exception:
        pass

@mcp.tool()
def set_session_defaults(
    collection: Optional[str] = None,
    session: Optional[str] = None,
    mode: Optional[str] = None,
    language: Optional[str] = None,
    under: Optional[str] = None,
    ctx: Context = None,
    kwargs: Any = None,
) -> Dict[str, Any]:
    """Set defaults (e.g., collection, mode, language, under) for subsequent calls.

    Behavior:
    - If a request Context is provided (normal with FastMCP), store defaults per-connection
      so subsequent calls on the same MCP session automatically use them (no token needed).
    - Optionally, also supports a lightweight token for clients that prefer cross-connection reuse.

    Precedence everywhere: explicit collection > per-connection defaults > token defaults > env default.
    """
    # Handle kwargs payload from some clients
    try:
        _extra = kwargs or {}
        if isinstance(_extra, str):
            try:
                _extra = json.loads(_extra)
            except Exception:
                _extra = {}
        if isinstance(_extra, dict):
            if not collection and _extra.get("collection"):
                collection = _extra["collection"]
            if not mode and _extra.get("mode"):
                mode = _extra["mode"]
            if not language and _extra.get("language"):
                language = _extra["language"]
            if not under and _extra.get("under"):
                under = _extra["under"]
            if not session and _extra.get("session"):
                session = _extra["session"]
    except Exception:
        pass

    # Prepare defaults payload
    defaults: Dict[str, Any] = {}
    if isinstance(collection, str) and collection.strip():
        defaults["collection"] = collection.strip()
    if isinstance(mode, str) and mode.strip():
        defaults["mode"] = mode.strip()
    if isinstance(language, str) and language.strip():
        defaults["language"] = language.strip()
    if isinstance(under, str) and under.strip():
        defaults["under"] = under.strip()

    # Store per-connection (preferred, no token required)
    try:
        if ctx is not None and getattr(ctx, "session", None) is not None and defaults:
            with _SESSION_CTX_LOCK:
                existing = SESSION_DEFAULTS_BY_SESSION.get(ctx.session) or {}
                existing.update(defaults)
                SESSION_DEFAULTS_BY_SESSION[ctx.session] = existing
    except Exception:
        pass

    # Optional: also support legacy token
    sid = (str(session).strip() if session is not None else "") or None
    if not sid:
        import uuid as _uuid
        sid = _uuid.uuid4().hex[:12]
    try:
        if defaults:
            with _SESSION_LOCK:
                existing = SESSION_DEFAULTS.get(sid) or {}
                existing.update(defaults)
                SESSION_DEFAULTS[sid] = existing
    except Exception:
        pass

    return {
        "ok": True,
        "session": sid,
        "defaults": (SESSION_DEFAULTS.get(sid, {}) if sid else {}),
        "applied": ("connection" if (ctx is not None and getattr(ctx, "session", None) is not None) else "token"),
    }


@mcp.tool()
def memory_store(
    information: str,
    metadata: Optional[Dict[str, Any]] = None,
    collection: Optional[str] = None,
    session: Optional[str] = None,
    ctx: Context = None,
) -> Dict[str, Any]:
    """Store a memory entry into Qdrant (dual vectors consistent with indexer).

    First call may be slower because the embedding model loads lazily.
    """
    sess = _require_auth_session(session)
    coll = _resolve_collection(collection, session=session, ctx=ctx)
    _require_collection_access((sess or {}).get("user_id"), coll, "write")
    _ensure_once(coll)

    # Prepare metadata with defaults and timestamp as documented
    md = metadata.copy() if metadata else {}
    if "created_at" not in md:
        md["created_at"] = datetime.utcnow().isoformat() + "Z"
    if "kind" not in md:
        md["kind"] = "memory"
    if "source" not in md:
        md["source"] = "memory"

    model = _get_embedding_model()
    dense = next(model.embed([str(information)])).tolist()
    lex = _lex_hash_vector_text(str(information), LEX_VECTOR_DIM)

    # Use UUID to avoid point ID collisions under concurrent load
    import uuid
    pid = uuid.uuid4().hex
    payload = {
        "information": str(information),
        "metadata": md,
    }
    point = models.PointStruct(
        id=pid, vector={VECTOR_NAME: dense, LEX_VECTOR_NAME: lex}, payload=payload
    )
    # wait=True blocks until Qdrant confirms write; set MEMORY_UPSERT_WAIT=0 for async writes
    _upsert_wait = os.environ.get("MEMORY_UPSERT_WAIT", "1").strip().lower() in {"1", "true", "yes", "on"}

    # Use pooled client for upsert
    client = _get_qdrant_client()
    try:
        client.upsert(collection_name=coll, points=[point], wait=_upsert_wait)
    finally:
        _return_qdrant_client(client)

    return {
        "ok": True,
        "id": pid,
        "message": "Successfully stored information",
        "collection": coll,
        "vector": VECTOR_NAME
    }


@mcp.tool()
def memory_find(
    query: str,
    limit: Optional[int] = None,
    collection: Optional[str] = None,
    top_k: Optional[int] = None,
    session: Optional[str] = None,
    q: Optional[str] = None,
    kind: Optional[str] = None,
    language: Optional[str] = None,
    topic: Optional[str] = None,
    tags: Optional[Union[str, List[str]]] = None,
    priority_min: Optional[int] = None,
    ctx: Context = None,
) -> Dict[str, Any]:
    """Find memory-like entries by vector similarity (dense + lexical fusion).

    Cold-start option: set MEMORY_COLD_SKIP_DENSE=1 to skip dense embedding until the
    model is cached (useful on slow storage).
    """
    # Handle 'q' alias for query
    if not query and q:
        query = q

    sess = _require_auth_session(session)
    coll = _resolve_collection(collection, session=session, ctx=ctx)
    _require_collection_access((sess or {}).get("user_id") if sess else None, coll, "read")
    _ensure_once(coll)

    use_dense = True
    if MEMORY_COLD_SKIP_DENSE and not _is_model_cached(EMBEDDING_MODEL):
        use_dense = False
    if use_dense:
        model = _get_embedding_model()
        dense = next(model.embed([str(query)])).tolist()
    else:
        dense = None
    lex = _lex_hash_vector_text(str(query), LEX_VECTOR_DIM)

    # Harmonize alias: top_k -> limit
    lim = int(limit if limit is not None else (top_k if top_k is not None else 5))

    # Build Qdrant filter
    must = []
    if kind:
        must.append(models.FieldCondition(key="metadata.kind", match=models.MatchValue(value=kind)))
    if language:
        must.append(models.FieldCondition(key="metadata.language", match=models.MatchValue(value=language)))
    if topic:
        must.append(models.FieldCondition(key="metadata.topic", match=models.MatchValue(value=topic)))
    if priority_min is not None:
        must.append(models.FieldCondition(key="metadata.priority", range=models.Range(gte=float(priority_min))))
    if tags:
        if isinstance(tags, str):
            tag_list = [t.strip() for t in tags.split(",") if t.strip()]
        else:
            tag_list = tags
        if tag_list:
            must.append(models.FieldCondition(key="metadata.tags", match=models.MatchAny(any=tag_list)))

    flt = models.Filter(must=must) if must else None

    # Two searches (prefer query_points) then simple RRF-like merge
    # Use pooled client for all search operations
    client = _get_qdrant_client()
    try:
        if use_dense:
            try:
                qp_dense = client.query_points(
                    collection_name=coll,
                    query=dense,
                    using=VECTOR_NAME,
                    query_filter=flt,
                    limit=max(10, lim),
                    with_payload=True,
                )
                res_dense = getattr(qp_dense, "points", qp_dense)
            except AttributeError:
                res_dense = client.search(
                    collection_name=coll,
                    query_vector=(VECTOR_NAME, dense),
                    query_filter=flt,
                    limit=max(10, lim),
                    with_payload=True,
                )
        else:
            res_dense = []

        try:
            qp_lex = client.query_points(
                collection_name=coll,
                query=lex,
                using=LEX_VECTOR_NAME,
                query_filter=flt,
                limit=max(10, lim),
                with_payload=True,
            )
            res_lex = getattr(qp_lex, "points", qp_lex)
        except AttributeError:
            res_lex = client.search(
                collection_name=coll,
                query_vector=(LEX_VECTOR_NAME, lex),
                query_filter=flt,
                limit=max(10, lim),
                with_payload=True,
            )
    finally:
        _return_qdrant_client(client)

    def is_memory_like(payload: Dict[str, Any]) -> bool:
        md = (payload or {}).get("metadata") or {}
        path = md.get("path")
        m_kind = (md.get("kind") or "").lower()
        source = (md.get("source") or "").lower()
        return (
            (not path)
            or (m_kind in {"memory", "preference", "note", "policy", "chat"})
            or (source in {"memory", "chat"})
        )

    def get_highlights(text: str, q_text: str) -> List[str]:
        if not text or not q_text:
            return []
        import re
        # Take interesting tokens from query
        tokens = [t for t in re.split(r'[^A-Za-z0-9]+', q_text) if len(t) > 2]
        if not tokens:
            return []
        
        results = []
        for t in tokens[:3]: # Max 3 types of highlights
            pattern = re.compile(re.escape(t), re.IGNORECASE)
            match = pattern.search(text)
            if match:
                # Get small context
                start = max(0, match.start() - 30)
                end = min(len(text), match.end() + 30)
                snip = text[start:end]
                # Wrap all occurrences of this token in this snip
                highlighted = pattern.sub(lambda m: f"<<{m.group(0)}>>", snip)
                results.append(f"...{highlighted}...")
        return results

    scores: Dict[str, float] = {}
    items: Dict[str, Dict[str, Any]] = {}

    def add_hits(hits, weight: float):
        for r in hits:
            pid = str(getattr(r, "id", None))
            if not pid:
                continue
            pl = getattr(r, "payload", {}) or {}
            if not is_memory_like(pl):
                continue
            scores[pid] = scores.get(pid, 0.0) + weight / (
                1.0 + getattr(r, "score", 0.0)
            )
            inf = pl.get("information") or pl.get("content") or pl.get("text") or ""
            items[pid] = {
                "id": getattr(r, "id", None),
                "information": inf,
                "metadata": pl.get("metadata") or {},
                "score": getattr(r, "score", None),
                "highlights": get_highlights(inf, query)
            }

    add_hits(res_dense, 1.0)
    add_hits(res_lex, 0.9)

    ordered = sorted(
        items.values(), key=lambda x: scores.get(str(x["id"]), 0.0), reverse=True
    )[:lim]
    return {
        "ok": True, 
        "results": ordered, 
        "total": len(ordered), 
        "count": len(ordered),
        "query": query
    }


def _resolve_collection(
    collection: Optional[str],
    session: Optional[str] = None,
    ctx: Context = None,
    extra_kwargs: Any = None,
) -> str:
    """Resolve the collection name honoring explicit args, session defaults, and env fallbacks."""
    coll = (collection or "").strip()
    sid: Optional[str] = None

    # Extract overrides from nested kwargs payloads some clients send
    try:
        payload = extra_kwargs or {}
        if isinstance(payload, dict) and "kwargs" in payload:
            payload = payload.get("kwargs")
            if isinstance(payload, str):
                try:
                    payload = json.loads(payload)
                except Exception:
                    payload = {}
        if not coll and isinstance(payload, dict) and payload.get("collection") is not None:
            coll = str(payload.get("collection")).strip()
        if isinstance(payload, dict) and payload.get("session") is not None:
            sid = str(payload.get("session")).strip()
    except Exception:
        pass

    # Explicit session parameter wins over payload session
    try:
        if session is not None and str(session).strip():
            sid = str(session).strip()
    except Exception:
        pass

    # Per-connection defaults via Context session
    if not coll and ctx is not None and getattr(ctx, "session", None) is not None:
        try:
            with _SESSION_CTX_LOCK:
                defaults = SESSION_DEFAULTS_BY_SESSION.get(ctx.session) or {}
                candidate = str(defaults.get("collection") or "").strip()
                if candidate:
                    coll = candidate
        except Exception:
            pass

    # Legacy token-based session defaults
    if not coll and sid:
        try:
            with _SESSION_LOCK:
                defaults = SESSION_DEFAULTS.get(sid) or {}
                candidate = str(defaults.get("collection") or "").strip()
                if candidate:
                    coll = candidate
        except Exception:
            pass

    return coll or DEFAULT_COLLECTION


if __name__ == "__main__":
    transport = os.environ.get("FASTMCP_TRANSPORT", "sse").strip().lower()
    # Start lightweight /readyz health endpoint in background (best-effort)
    try:
        _start_readyz_server()
    except Exception:
        pass

    # Relax Pydantic model defaults for **kwargs compatibility
    # This must be called AFTER all tools are registered but BEFORE mcp.run()
    try:
        _relax_var_kwarg_defaults()
    except Exception:
        pass

    if transport == "stdio":
        # Run over stdio (for clients that don't support network transports)
        mcp.run(transport="stdio")
    elif transport in {"http", "streamable", "streamable_http", "streamable-http"}:
        # Streamable HTTP (recommended) — endpoint at /mcp (FastMCP default)
        try:
            mcp.settings.host = HOST
            mcp.settings.port = PORT
        except Exception:
            pass
        # Use the correct FastMCP transport name
        try:
            mcp.run(transport="streamable-http")
        except Exception:
            # Fallback to SSE only if HTTP truly unavailable
            mcp.settings.host = HOST
            mcp.settings.port = PORT
            mcp.run(transport="sse")
    else:
        # SSE (legacy) — endpoint at /sse
        mcp.settings.host = HOST
        mcp.settings.port = PORT
        mcp.run(transport="sse")
