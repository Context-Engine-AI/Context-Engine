#!/usr/bin/env python3
"""
HTTP Upload Service for Delta Bundles in Context-Engine.

This FastAPI service receives delta bundles from remote upload clients,
processes them, and integrates with the existing indexing pipeline.
"""

import os
import json
import tarfile
import tempfile
import asyncio
import logging
import re
import secrets
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

# ---------------------------------------------------------------------------
# OpenLit Observability (optional - only if OPENLIT_ENABLED=1)
# ---------------------------------------------------------------------------
_OPENLIT_ENABLED = os.environ.get("OPENLIT_ENABLED", "0").lower() in ("1", "true", "yes")
if _OPENLIT_ENABLED:
    try:
        import openlit
        _otel_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://openlit:4318")
        openlit.init(otlp_endpoint=_otel_endpoint)
        # Explicitly instrument Qdrant for vector DB operations
        try:
            from openlit.instrumentation.qdrant import QdrantInstrumentation
            QdrantInstrumentation().instrument()
            print(f"[OpenLit] Upload service Qdrant instrumentation enabled")
        except Exception as qe:
            print(f"[OpenLit] Qdrant instrumentation failed: {qe}")
        print(f"[OpenLit] Upload service observability enabled, exporting to {_otel_endpoint}")
    except ImportError:
        print("[OpenLit] SDK not installed, skipping observability")
    except Exception as e:
        print(f"[OpenLit] Failed to initialize: {e}")

import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request, status
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware

from scripts.upload_delta_bundle import get_workspace_key, process_delta_bundle

from scripts.indexing_admin import (
    build_admin_collections_view,
    resolve_collection_root,
    spawn_ingest_code,
    recreate_collection_qdrant,
)

try:
    from scripts.workspace_state import is_staging_enabled
except Exception:
    is_staging_enabled = None  # type: ignore


from pydantic import BaseModel, Field
from scripts.auth_backend import (
    AuthDisabledError,
    AuthInvalidToken,
    authenticate_user,
    create_session,
    create_session_for_token,
    create_user,
    has_any_users,
    has_collection_access,
    validate_session,
    AUTH_ENABLED,
    AUTH_SESSION_TTL_SECONDS,
    is_admin_user,
    list_users,
    list_collections,
    list_collection_acl,
    grant_collection_access,
    revoke_collection_access,
)

try:
    from scripts.collection_admin import delete_collection_everywhere, copy_collection_qdrant
except Exception:
    delete_collection_everywhere = None
    copy_collection_qdrant = None
try:
    from scripts.admin_ui import (
        render_admin_acl,
        render_admin_bootstrap,
        render_admin_error,
        render_admin_login,
    )
except Exception:

    def _admin_ui_unavailable(*args, **kwargs):
        raise HTTPException(status_code=500, detail="Admin UI unavailable")

    render_admin_acl = _admin_ui_unavailable
    render_admin_bootstrap = _admin_ui_unavailable
    render_admin_error = _admin_ui_unavailable
    render_admin_login = _admin_ui_unavailable

# Import staging/indexing admin helpers
try:
    from scripts.indexing_admin import (
        start_staging_rebuild,
        activate_staging_rebuild,
        abort_staging_rebuild,
    )
except ImportError:
    start_staging_rebuild = None  # type: ignore
    activate_staging_rebuild = None  # type: ignore
    abort_staging_rebuild = None  # type: ignore

# Import existing workspace state and indexing functions
try:
    from scripts.workspace_state import (
        log_activity,
        get_collection_name,
        get_cached_file_hash,
        set_cached_file_hash,
        _extract_repo_name_from_path,
        update_repo_origin,
        get_collection_mappings,
        find_collection_for_logical_repo,
        update_workspace_state,
        set_staging_state,
        update_staging_status,
        clear_staging_collection,
        logical_repo_reuse_enabled,
        get_collection_state_snapshot,
    )
except ImportError:
    # Fallback for testing without full environment
    log_activity = None
    get_collection_name = None
    get_cached_file_hash = None
    set_cached_file_hash = None
    _extract_repo_name_from_path = None
    update_repo_origin = None
    get_collection_mappings = None
    find_collection_for_logical_repo = None
    update_workspace_state = None
    set_staging_state = None
    update_staging_status = None
    clear_staging_collection = None
    def logical_repo_reuse_enabled() -> bool:  # type: ignore[no-redef]
        return False


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration from environment
QDRANT_URL = os.environ.get("QDRANT_URL", "http://qdrant:6333")
DEFAULT_COLLECTION = os.environ.get("COLLECTION_NAME", "codebase")
WORK_DIR = os.environ.get("WORK_DIR") or os.environ.get("WORKDIR") or "/work"
MAX_BUNDLE_SIZE_MB = int(os.environ.get("MAX_BUNDLE_SIZE_MB", "100"))
UPLOAD_TIMEOUT_SECS = int(os.environ.get("UPLOAD_TIMEOUT_SECS", "300"))
ADMIN_COLLECTION_DELETE_ENABLED = (
    str(os.environ.get("CTXCE_ADMIN_COLLECTION_DELETE_ENABLED", "0")).strip().lower()
    in {"1", "true", "yes", "on"}
)
try:
    ADMIN_COLLECTION_REFRESH_MS = int(
        str(os.environ.get("CTXCE_ADMIN_COLLECTION_REFRESH_MS", "5000")).strip()
        or "5000"
    )
except Exception:
    ADMIN_COLLECTION_REFRESH_MS = 5000
_SLUGGED_REPO_RE = re.compile(r"^.+-[0-9a-f]{16}(?:_old)?$")
CTXCE_MCP_ACL_ENFORCE = (
    str(os.environ.get("CTXCE_MCP_ACL_ENFORCE", "0")).strip().lower()
    in {"1", "true", "yes", "on"}
)
BRIDGE_STATE_TOKEN = (os.environ.get("CTXCE_BRIDGE_STATE_TOKEN") or "").strip()

# FastAPI app
app = FastAPI(
    title="Context-Engine Delta Upload Service",
    description="HTTP service for receiving and processing delta bundles",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory sequence tracking (in production, use persistent storage)
_sequence_tracker: Dict[str, int] = {}


def _int_env(name: str, default: int) -> int:
    """Read integer env vars safely; blank/invalid values fall back to default."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        trimmed = str(raw).strip()
        if not trimmed:
            return default
        return int(trimmed)
    except (TypeError, ValueError):
        return default

class UploadResponse(BaseModel):
    success: bool
    bundle_id: Optional[str] = None
    sequence_number: Optional[int] = None
    processed_operations: Optional[Dict[str, int]] = None
    processing_time_ms: Optional[int] = None
    next_sequence: Optional[int] = None
    error: Optional[Dict[str, Any]] = None

class StatusResponse(BaseModel):
    workspace_path: str
    collection_name: str
    last_sequence: int
    last_upload: Optional[str] = None
    pending_operations: int
    status: str
    server_info: Dict[str, Any]

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    qdrant_url: str
    work_dir: str


class BridgeCollectionStateResponse(BaseModel):
    workspace_path: str
    repo_name: Optional[str]
    active_collection: Optional[str]
    serving_collection: Optional[str]
    previous_collection: Optional[str]
    active_repo_slug: Optional[str]
    serving_repo_slug: Optional[str]
    indexing_status: Optional[Dict[str, Any]]
    staging: Optional[Dict[str, Any]]


class AuthLoginRequest(BaseModel):
    client: str
    workspace: Optional[str] = None
    token: Optional[str] = None


class AuthLoginResponse(BaseModel):
    session_id: str
    user_id: Optional[str] = None
    expires_at: Optional[int] = None


class AuthStatusResponse(BaseModel):
    enabled: bool
    has_users: Optional[bool] = None
    session_ttl_seconds: int


class AuthUserCreateRequest(BaseModel):
    username: str
    password: str


class AuthUserCreateResponse(BaseModel):
    user_id: str
    username: str


class PasswordLoginRequest(BaseModel):
    username: str
    password: str
    workspace: Optional[str] = None


ADMIN_SESSION_COOKIE_NAME = "ctxce_session"


def _get_session_candidate_from_request(request: Request) -> Dict[str, Any]:
    sid = (request.cookies.get(ADMIN_SESSION_COOKIE_NAME) or "").strip()
    if sid:
        return {"session_id": sid, "source": "cookie"}
    try:
        qp = request.query_params
        sid = (qp.get("session") or qp.get("session_id") or qp.get("sessionId") or "").strip()
    except Exception:
        sid = ""
    if sid:
        return {"session_id": sid, "source": "query"}
    sid = (
        (request.headers.get("X-Session-Id") or "").strip()
        or (request.headers.get("X-Auth-Session") or "").strip()
    )
    if sid:
        return {"session_id": sid, "source": "header"}
    return {"session_id": "", "source": ""}


def _set_admin_session_cookie(resp: Any, session_id: str) -> Any:
    sid = (session_id or "").strip()
    if not sid:
        return resp
    try:
        kwargs: Dict[str, Any] = {
            "key": ADMIN_SESSION_COOKIE_NAME,
            "value": sid,
            "httponly": True,
            "samesite": "lax",
            "path": "/",
        }
        ttl = int(AUTH_SESSION_TTL_SECONDS or 0)
        if ttl > 0:
            kwargs["max_age"] = ttl
        resp.set_cookie(**kwargs)
    except Exception:
        pass
    return resp


def _get_valid_session_record(request: Request) -> Optional[Dict[str, Any]]:
    sid = (_get_session_candidate_from_request(request).get("session_id") or "").strip()
    if not sid:
        return None
    try:
        return validate_session(sid)
    except AuthDisabledError:
        return None
    except Exception as e:
        logger.error(f"[upload_service] Failed to validate session cookie: {e}")
        raise HTTPException(status_code=500, detail="Failed to validate auth session")


def _require_admin_session(request: Request) -> Dict[str, Any]:
    if not AUTH_ENABLED:
        raise HTTPException(status_code=404, detail="Auth disabled")
    record = _get_valid_session_record(request)
    if record is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    user_id = str(record.get("user_id") or "").strip()
    if not user_id or not is_admin_user(user_id):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin required")
    return record


def _bridge_state_authorized(request: Request) -> None:
    header_token = (request.headers.get("X-Bridge-State-Token") or "").strip()
    if BRIDGE_STATE_TOKEN:
        if header_token:
            try:
                if secrets.compare_digest(header_token, BRIDGE_STATE_TOKEN):
                    return
            except Exception:
                pass
        try:
            _require_admin_session(request)
            return
        except HTTPException as exc:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="Unauthorized bridge request"
            ) from exc

    else:
        try:
            _require_admin_session(request)
            return
        except HTTPException as exc:
            raise HTTPException(
                status_code=exc.status_code, detail="Unauthorized bridge request"
            ) from exc


def _infer_repo_from_workspace(workspace_path: str) -> Optional[str]:
    try:
        candidate = Path(workspace_path).name
        if candidate and _SLUGGED_REPO_RE.match(candidate):
            return candidate
    except Exception:
        return None
    return None


def _resolve_bridge_state_target(
    *,
    collection: Optional[str],
    workspace: Optional[str],
    repo_name: Optional[str],
) -> Tuple[str, Optional[str]]:
    """Resolve workspace path + repo_name for bridge state lookups."""
    workspace_path = (workspace or "").strip()
    if not workspace_path:
        workspace_path = WORK_DIR

    repo = (repo_name or "").strip() or None

    if collection:
        if resolve_collection_root is None:
            raise HTTPException(status_code=400, detail="collection mapping unavailable")
        root, resolved_repo = resolve_collection_root(collection=collection, work_dir=WORK_DIR)
        if not root:
            raise HTTPException(status_code=404, detail="collection mapping not found")
        workspace_path = root
        repo = resolved_repo
    elif not repo:
        repo = _infer_repo_from_workspace(workspace_path)

    return workspace_path, repo


def get_next_sequence(workspace_path: str) -> int:
    """Get next sequence number for workspace."""
    key = get_workspace_key(workspace_path)
    current = _sequence_tracker.get(key, 0)
    next_seq = current + 1
    _sequence_tracker[key] = next_seq
    return next_seq

def get_last_sequence(workspace_path: str) -> int:
    """Get last sequence number for workspace."""
    key = get_workspace_key(workspace_path)
    return _sequence_tracker.get(key, 0)

def validate_bundle_format(bundle_path: Path) -> Dict[str, Any]:
    """Validate delta bundle format and return manifest."""
    try:
        with tarfile.open(bundle_path, "r:gz") as tar:
            # Check for required files
            required_files = ["manifest.json", "metadata/operations.json", "metadata/hashes.json"]
            members = tar.getnames()

            for req_file in required_files:
                if not any(req_file in member for member in members):
                    raise ValueError(f"Missing required file: {req_file}")

            # Extract and validate manifest
            manifest_member = None
            for member in members:
                if member.endswith("manifest.json"):
                    manifest_member = member
                    break

            if not manifest_member:
                raise ValueError("manifest.json not found in bundle")

            manifest_file = tar.extractfile(manifest_member)
            if not manifest_file:
                raise ValueError("Cannot extract manifest.json")

            manifest = json.loads(manifest_file.read().decode('utf-8'))

            # Validate manifest structure
            required_fields = ["version", "bundle_id", "workspace_path", "created_at", "sequence_number"]
            for field in required_fields:
                if field not in manifest:
                    raise ValueError(f"Missing required field in manifest: {field}")

            return manifest

    except Exception as e:
        raise ValueError(f"Invalid bundle format: {str(e)}")


async def _process_bundle_background(
    workspace_path: str,
    bundle_path: Path,
    manifest: Dict[str, Any],
    sequence_number: Optional[int],
    bundle_id: Optional[str],
) -> None:
    try:
        start_time = datetime.now()
        operations_count = await asyncio.to_thread(
            process_delta_bundle, workspace_path, bundle_path, manifest
        )
        if sequence_number is not None:
            key = get_workspace_key(workspace_path)
            _sequence_tracker[key] = sequence_number
        if log_activity:
            try:
                repo = _extract_repo_name_from_path(workspace_path) if _extract_repo_name_from_path else None
                log_activity(
                    repo_name=repo,
                    action="uploaded",
                    file_path=bundle_id,
                    details={
                        "bundle_id": bundle_id,
                        "operations": operations_count,
                        "source": "delta_upload",
                    },
                )
            except Exception as activity_err:
                logger.debug(f"[upload_service] Failed to log activity for bundle {bundle_id}: {activity_err}")
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        logger.info(
            f"[upload_service] Finished processing bundle {bundle_id} seq {sequence_number} in {int(processing_time)}ms"
        )
    except Exception as e:
        logger.error(f"[upload_service] Error in background processing for bundle {bundle_id}: {e}")
    finally:
        try:
            bundle_path.unlink()
        except Exception:
            pass


@app.get("/auth/status", response_model=AuthStatusResponse)
async def auth_status():
    try:
        if not AUTH_ENABLED:
            return AuthStatusResponse(enabled=False, has_users=None, session_ttl_seconds=0)
        try:
            users_exist = has_any_users()
        except AuthDisabledError:
            return AuthStatusResponse(
                enabled=False,
                has_users=None,
                session_ttl_seconds=AUTH_SESSION_TTL_SECONDS,
            )
        return AuthStatusResponse(
            enabled=True,
            has_users=users_exist,
            session_ttl_seconds=AUTH_SESSION_TTL_SECONDS,
        )
    except Exception as e:
        logger.error(f"[upload_service] Failed to report auth status: {e}")
        raise HTTPException(status_code=500, detail="Failed to read auth status")


@app.post("/auth/login", response_model=AuthLoginResponse)
async def auth_login(payload: AuthLoginRequest):
    try:
        session = create_session_for_token(
            client=payload.client,
            workspace=payload.workspace,
            token=payload.token,
        )
    except AuthDisabledError:
        raise HTTPException(status_code=404, detail="Auth disabled")
    except AuthInvalidToken:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid auth token")
    except Exception as e:
        logger.error(f"[upload_service] Failed to create auth session: {e}")
        raise HTTPException(status_code=500, detail="Failed to create auth session")
    return AuthLoginResponse(
        session_id=session.get("session_id"),
        user_id=session.get("user_id"),
        expires_at=session.get("expires_at"),
    )


@app.get("/admin")
async def admin_root(request: Request):
    if not AUTH_ENABLED:
        raise HTTPException(status_code=404, detail="Auth disabled")
    try:
        users_exist = has_any_users()
    except AuthDisabledError:
        raise HTTPException(status_code=404, detail="Auth disabled")
    except Exception as e:
        logger.error(f"[upload_service] Failed to inspect user state for admin UI: {e}")
        raise HTTPException(status_code=500, detail="Failed to inspect user state")

    if not users_exist:
        return RedirectResponse(url="/admin/bootstrap", status_code=302)

    candidate = _get_session_candidate_from_request(request)
    record = _get_valid_session_record(request)
    if record is None:
        return RedirectResponse(url="/admin/login", status_code=302)

    user_id = str(record.get("user_id") or "").strip()
    if user_id and is_admin_user(user_id):
        resp = RedirectResponse(url="/admin/acl", status_code=302)
        if candidate.get("source") and candidate.get("source") != "cookie":
            _set_admin_session_cookie(resp, str(candidate.get("session_id") or ""))
        return resp
    return RedirectResponse(url="/admin/login", status_code=302)


@app.get("/admin/bootstrap")
async def admin_bootstrap_form(request: Request):
    if not AUTH_ENABLED:
        raise HTTPException(status_code=404, detail="Auth disabled")
    try:
        users_exist = has_any_users()
    except AuthDisabledError:
        raise HTTPException(status_code=404, detail="Auth disabled")
    except Exception as e:
        logger.error(f"[upload_service] Failed to inspect user state for bootstrap: {e}")
        raise HTTPException(status_code=500, detail="Failed to inspect user state")
    if users_exist:
        return RedirectResponse(url="/admin/login", status_code=302)
    return render_admin_bootstrap(request)


@app.post("/admin/bootstrap")
async def admin_bootstrap_submit(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
):
    if not AUTH_ENABLED:
        raise HTTPException(status_code=404, detail="Auth disabled")
    try:
        users_exist = has_any_users()
    except AuthDisabledError:
        raise HTTPException(status_code=404, detail="Auth disabled")
    except Exception as e:
        logger.error(f"[upload_service] Failed to inspect user state for bootstrap submit: {e}")
        raise HTTPException(status_code=500, detail="Failed to inspect user state")
    if users_exist:
        return RedirectResponse(url="/admin/login", status_code=302)

    try:
        user = create_user(username, password)
    except Exception as e:
        return render_admin_error(
            request=request,
            title="Bootstrap Failed",
            message=str(e),
            back_href="/admin/bootstrap",
            status_code=400,
        )

    try:
        session = create_session(user_id=user.get("user_id"), metadata={"client": "admin_ui"})
    except Exception as e:
        logger.error(f"[upload_service] Failed to create session after bootstrap: {e}")
        raise HTTPException(status_code=500, detail="Failed to create auth session")

    resp = RedirectResponse(url="/admin/acl", status_code=302)
    _set_admin_session_cookie(resp, str(session.get("session_id") or ""))
    return resp


@app.get("/admin/login")
async def admin_login_form(request: Request):
    if not AUTH_ENABLED:
        raise HTTPException(status_code=404, detail="Auth disabled")
    return render_admin_login(request)


@app.post("/admin/login")
async def admin_login_submit(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
):
    if not AUTH_ENABLED:
        raise HTTPException(status_code=404, detail="Auth disabled")
    try:
        user = authenticate_user(username, password)
    except AuthDisabledError:
        raise HTTPException(status_code=404, detail="Auth disabled")
    except Exception as e:
        logger.error(f"[upload_service] Error authenticating user for admin UI: {e}")
        raise HTTPException(status_code=500, detail="Authentication error")

    if not user:
        return render_admin_login(
            request=request,
            error="Invalid credentials",
            status_code=401,
        )

    try:
        session = create_session(user_id=user.get("id"), metadata={"client": "admin_ui"})
    except Exception as e:
        logger.error(f"[upload_service] Failed to create session for admin UI: {e}")
        raise HTTPException(status_code=500, detail="Failed to create auth session")

    resp = RedirectResponse(url="/admin/acl", status_code=302)
    _set_admin_session_cookie(resp, str(session.get("session_id") or ""))
    return resp


@app.post("/admin/logout")
async def admin_logout():
    resp = RedirectResponse(url="/admin/login", status_code=302)
    resp.delete_cookie(key=ADMIN_SESSION_COOKIE_NAME, path="/")
    return resp


@app.get("/admin/acl")
async def admin_acl_page(request: Request):
    _require_admin_session(request)
    try:
        users = list_users()
        collections = list_collections(include_deleted=False)
        grants = list_collection_acl()
    except AuthDisabledError:
        raise HTTPException(status_code=404, detail="Auth disabled")
    except Exception as e:
        logger.error(f"[upload_service] Failed to load admin UI data: {e}")
        raise HTTPException(status_code=500, detail="Failed to load admin data")

    enriched = await asyncio.to_thread(
        build_admin_collections_view, collections=collections, work_dir=WORK_DIR
    )

    resp = render_admin_acl(
        request,
        users=users,
        collections=enriched,
        grants=grants,
        deletion_enabled=ADMIN_COLLECTION_DELETE_ENABLED,
        work_dir=WORK_DIR,
        refresh_ms=ADMIN_COLLECTION_REFRESH_MS,
    )
    candidate = _get_session_candidate_from_request(request)
    if candidate.get("source") and candidate.get("source") != "cookie":
        _set_admin_session_cookie(resp, str(candidate.get("session_id") or ""))
    return resp


@app.post("/admin/acl/grant")
async def admin_acl_grant(
    request: Request,
    user_id: str = Form(...),
    collection: str = Form(...),
    permission: str = Form("read"),
):
    _require_admin_session(request)
    try:
        grant_collection_access(user_id=user_id, qdrant_collection=collection, permission=permission)
    except AuthDisabledError:
        raise HTTPException(status_code=404, detail="Auth disabled")
    except Exception as e:
        return render_admin_error(request, title="Grant Failed", message=str(e), back_href="/admin/acl")
    return RedirectResponse(url="/admin/acl", status_code=302)


@app.get("/admin/collections/status")
async def admin_collections_status(request: Request):
    _require_admin_session(request)
    try:
        collections = list_collections(include_deleted=False)
    except AuthDisabledError:
        raise HTTPException(status_code=404, detail="Auth disabled")
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to load collections")

    enriched = await asyncio.to_thread(
        lambda: build_admin_collections_view(collections=collections, work_dir=WORK_DIR)
    )
    return JSONResponse({"collections": enriched})


@app.post("/admin/collections/reindex")
async def admin_reindex_collection(
    request: Request,
    collection: str = Form(...),
):
    _require_admin_session(request)
    name = (collection or "").strip()
    if not name:
        return render_admin_error(
            request,
            title="Reindex Failed",
            message="collection is required",
            back_href="/admin/acl",
        )

    root, repo_name = resolve_collection_root(collection=name, work_dir=WORK_DIR)
    if not root:
        return render_admin_error(
            request,
            title="Reindex Failed",
            message="No workspace mapping found for collection (no state.json mapping)",
            back_href="/admin/acl",
        )

    try:
        spawn_ingest_code(
            root=root,
            work_dir=WORK_DIR,
            collection=name,
            recreate=False,
            repo_name=repo_name,
            clear_caches=True,
        )
    except Exception as e:
        return render_admin_error(
            request,
            title="Reindex Failed",
            message=str(e),
            back_href="/admin/acl",
        )

    return RedirectResponse(url="/admin/acl", status_code=302)


@app.get("/bridge/state")
async def bridge_collection_state(
    request: Request,
    collection: Optional[str] = None,
    workspace: Optional[str] = None,
    repo_name: Optional[str] = None,
):
    if get_collection_state_snapshot is None:
        raise HTTPException(status_code=503, detail="workspace_state helper unavailable")

    _bridge_state_authorized(request)

    workspace_path, repo = _resolve_bridge_state_target(collection=collection, workspace=workspace, repo_name=repo_name)

    snapshot = get_collection_state_snapshot(workspace_path=workspace_path, repo_name=repo)
    if not snapshot:
        raise HTTPException(status_code=404, detail="Workspace state not found")

    if not (is_staging_enabled() if callable(is_staging_enabled) else False):
        # Classic mode: ignore any serving_* overrides from staging/migration.
        snapshot = dict(snapshot)
        snapshot.pop("serving_collection", None)
        snapshot.pop("serving_repo_slug", None)

    return BridgeCollectionStateResponse(
        workspace_path=str(snapshot.get("workspace_path") or workspace_path),
        repo_name=snapshot.get("repo_name") or repo,
        active_collection=snapshot.get("active_collection"),
        serving_collection=snapshot.get("serving_collection") or snapshot.get("active_collection"),
        previous_collection=snapshot.get("previous_collection"),
        active_repo_slug=snapshot.get("active_repo_slug") or repo,
        serving_repo_slug=snapshot.get("serving_repo_slug") or snapshot.get("active_repo_slug"),
        indexing_status=snapshot.get("indexing_status"),
        staging=snapshot.get("staging"),
    )


# Recreate previous admin endpoint that was displaced by /bridge/state
@app.post("/admin/collections/recreate")
async def admin_recreate_collection(
    request: Request,
    collection: str = Form(...),
):
    _require_admin_session(request)
    name = (collection or "").strip()
    if not name:
        return render_admin_error(
            request,
            title="Recreate Failed",
            message="collection is required",
            back_href="/admin/acl",
        )

    root, repo_name = resolve_collection_root(collection=name, work_dir=WORK_DIR)
    if not root:
        return render_admin_error(
            request,
            title="Recreate Failed",
            message="No workspace mapping found for collection (no state.json mapping)",
            back_href="/admin/acl",
        )

    try:
        recreate_collection_qdrant(
            qdrant_url=QDRANT_URL,
            api_key=os.environ.get("QDRANT_API_KEY") or None,
            collection=name,
        )
        spawn_ingest_code(
            root=root,
            work_dir=WORK_DIR,
            collection=name,
            recreate=False,
            repo_name=repo_name,
            clear_caches=True,
        )
    except Exception as e:
        return render_admin_error(
            request,
            title="Recreate Failed",
            message=str(e),
            back_href="/admin/acl",
        )

    return RedirectResponse(url="/admin/acl", status_code=302)


@app.post("/admin/collections/delete")
async def admin_delete_collection(
    request: Request,
    collection: str = Form(...),
    delete_fs: str = Form(""),
):
    _require_admin_session(request)
    if not ADMIN_COLLECTION_DELETE_ENABLED:
        try:
            return render_admin_error(
                request,
                title="Delete Collection Disabled",
                message="Collection deletion is disabled by server configuration",
                back_href="/admin/acl",
                status_code=403,
            )
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Collection deletion is disabled by server configuration",
            )
    name = (collection or "").strip()
    if not name:
        return render_admin_error(
            request,
            title="Delete Collection Failed",
            message="collection is required",
            back_href="/admin/acl",
        )

    if delete_collection_everywhere is None:
        return render_admin_error(
            request,
            title="Delete Collection Failed",
            message="Collection delete helper unavailable",
            back_href="/admin/acl",
        )

    # Default is Qdrant-only (no filesystem cleanup). Users must explicitly opt in.
    try:
        cleanup_fs = (delete_fs or "").strip().lower() in {"1", "true", "yes", "on"}
    except Exception:
        cleanup_fs = False

    try:
        delete_collection_everywhere(
            collection=name,
            work_dir=WORK_DIR,
            qdrant_url=QDRANT_URL,
            cleanup_fs=cleanup_fs,
        )
    except Exception as e:
        return render_admin_error(
            request,
            title="Delete Collection Failed",
            message=str(e),
            back_href="/admin/acl",
        )

    return RedirectResponse(url="/admin/acl", status_code=302)


@app.post("/admin/staging/start")
async def admin_start_staging(
    request: Request,
    collection: str = Form(...),
):
    _require_admin_session(request)
    if not (is_staging_enabled() if callable(is_staging_enabled) else False):
        return render_admin_error(
            request,
            title="Start Staging Failed",
            message="Staging is disabled (set CTXCE_STAGING_ENABLED=1 to enable)",
            back_href="/admin/acl",
        )

    name = (collection or "").strip()
    if not name:
        return render_admin_error(
            request,
            title="Start Staging Failed",
            message="collection is required",
            back_href="/admin/acl",
        )

    if start_staging_rebuild is None:
        return render_admin_error(
            request,
            title="Start Staging Failed",
            message="Staging helper unavailable",
            back_href="/admin/acl",
        )

    root: Optional[str] = None
    repo_name: Optional[str] = None
    try:
        root, repo_name = resolve_collection_root(collection=name, work_dir=WORK_DIR)
    except Exception:
        root, repo_name = None, None
    if not root:
        return render_admin_error(
            request,
            title="Start Staging Failed",
            message="No workspace mapping found for collection",
            back_href="/admin/acl",
        )

    staged_clone_name = f"{name}_old"
    request_id = secrets.token_hex(8)
    now = datetime.utcnow().isoformat()
    staging_payload = {
        "collection": staged_clone_name,
        "started_at": now,
        "updated_at": now,
        "env_hash": None,
        "workspace_path": root,
        "repo_name": repo_name,
        "status": {
            "state": "queued",
            "queued_at": now,
            "request_id": request_id,
        },
    }

    if set_staging_state:
        try:
            set_staging_state(workspace_path=root, repo_name=repo_name, staging=staging_payload)
        except Exception as set_err:
            logger.warning(f"[admin] Failed to persist queued staging state for {name}: {set_err}")
    elif update_workspace_state:
        try:
            update_workspace_state(
                workspace_path=root,
                repo_name=repo_name,
                updates={"staging": staging_payload},
            )
        except Exception as set_err:
            logger.warning(f"[admin] Failed to update workspace state for queued staging {name}: {set_err}")

    if update_staging_status:
        try:
            update_staging_status(
                workspace_path=root,
                repo_name=repo_name,
                status={"state": "queued", "queued_at": now, "request_id": request_id},
            )
        except Exception as status_err:
            logger.debug(f"[admin] Failed to mark staging status queued for {name}: {status_err}")

    try:
        async def _bg_start() -> None:
            try:
                staging_collection = await asyncio.to_thread(
                    start_staging_rebuild, collection=name, work_dir=WORK_DIR
                )
                logger.info(f"[admin] Started staging rebuild for {name} -> {staging_collection}")
            except Exception as e:
                logger.error(f"[admin] Background staging start failed for {name}: {e}")
                # Ensure we don't leave the workspace stuck in a queued staging state.
                try:
                    if clear_staging_collection:
                        clear_staging_collection(workspace_path=root, repo_name=repo_name)
                    elif update_workspace_state:
                        update_workspace_state(
                            workspace_path=root,
                            repo_name=repo_name,
                            updates={"staging": None},
                        )
                except Exception as clear_err:
                    logger.warning(
                        f"[admin] Failed to clear staging state after start failure for {name}: {clear_err}"
                    )

        asyncio.create_task(_bg_start())
    except Exception as e:
        return render_admin_error(
            request,
            title="Start Staging Failed",
            message=str(e),
            back_href="/admin/acl",
        )

    return RedirectResponse(url="/admin/acl", status_code=302)


@app.post("/admin/staging/activate")
async def admin_activate_staging(
    request: Request,
    collection: str = Form(...),
):
    _require_admin_session(request)
    if not (is_staging_enabled() if callable(is_staging_enabled) else False):
        return render_admin_error(
            request,
            title="Activate Staging Failed",
            message="Staging is disabled (set CTXCE_STAGING_ENABLED=1 to enable)",
            back_href="/admin/acl",
        )
    name = (collection or "").strip()
    if not name:
        return render_admin_error(
            request,
            title="Activate Staging Failed",
            message="collection is required",
            back_href="/admin/acl",
        )

    if activate_staging_rebuild is None:
        return render_admin_error(
            request,
            title="Activate Staging Failed",
            message="Staging helper unavailable",
            back_href="/admin/acl",
        )

    try:
        async def _bg_activate() -> None:
            try:
                await asyncio.to_thread(activate_staging_rebuild, collection=name, work_dir=WORK_DIR)
                logger.info(f"[admin] Activated staging for {name}")
            except Exception as e:
                logger.error(f"[admin] Background staging activate failed for {name}: {e}")

        asyncio.create_task(_bg_activate())
    except Exception as e:
        return render_admin_error(
            request,
            title="Activate Staging Failed",
            message=str(e),
            back_href="/admin/acl",
        )

    return RedirectResponse(url="/admin/acl", status_code=302)


@app.post("/admin/staging/abort")
async def admin_abort_staging(
    request: Request,
    collection: str = Form(...),
):
    _require_admin_session(request)
    name = (collection or "").strip()
    if not name:
        return render_admin_error(
            request,
            title="Abort Staging Failed",
            message="collection is required",
            back_href="/admin/acl",
        )

    root, repo_name = resolve_collection_root(collection=name, work_dir=WORK_DIR)
    if not root:
        return render_admin_error(
            request,
            title="Abort Staging Failed",
            message="No workspace mapping found for collection",
            back_href="/admin/acl",
        )

    try:
        if abort_staging_rebuild is not None:
            # Run abort to completion so we always clear staging metadata before returning.
            await asyncio.to_thread(
                abort_staging_rebuild,
                collection=name,
                work_dir=WORK_DIR,
                delete_collection=True,
            )
            logger.info(f"[admin] Aborted staging rebuild for {name}")
        elif clear_staging_collection:
            # Fallback for older deployments: clear staging metadata only.
            clear_staging_collection(workspace_path=root, repo_name=repo_name)
            logger.info(f"[admin] Aborted staging for {name} (metadata only)")
        else:
            raise RuntimeError("staging abort helpers unavailable")
    except Exception as e:
        return render_admin_error(
            request,
            title="Abort Staging Failed",
            message=str(e),
            back_href="/admin/acl",
        )

    return RedirectResponse(url="/admin/acl", status_code=302)


@app.post("/admin/staging/copy")
async def admin_copy_collection(
    request: Request,
    collection: str = Form(...),
    target: Optional[str] = Form(None),
    overwrite: Optional[str] = Form(""),
):
    _require_admin_session(request)
    name = (collection or "").strip()
    if not name:
        return render_admin_error(
            request,
            title="Copy Collection Failed",
            message="collection is required",
            back_href="/admin/acl",
        )

    if copy_collection_qdrant is None:
        return render_admin_error(
            request,
            title="Copy Collection Failed",
            message="copy helper unavailable",
            back_href="/admin/acl",
        )

    try:
        allow_overwrite = str(overwrite or "").strip().lower() in {"1", "true", "yes", "on"}
    except Exception:
        allow_overwrite = False

    try:
        new_name = copy_collection_qdrant(
            source=name,
            target=(target or None),
            qdrant_url=QDRANT_URL,
            overwrite=allow_overwrite,
        )
        logger.info(f"[admin] Copied collection {name} -> {new_name}")
    except Exception as e:
        return render_admin_error(
            request,
            title="Copy Collection Failed",
            message=str(e),
            back_href="/admin/acl",
        )

    return RedirectResponse(url="/admin/acl", status_code=302)


@app.post("/admin/users")
async def admin_create_user(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    role: str = Form("user"),
):
    _require_admin_session(request)
    try:
        create_user(username, password, role=role)
    except AuthDisabledError:
        raise HTTPException(status_code=404, detail="Auth disabled")
    except Exception as e:
        return render_admin_error(request, title="Create User Failed", message=str(e), back_href="/admin/acl")
    return RedirectResponse(url="/admin/acl", status_code=302)


@app.post("/admin/acl/revoke")
async def admin_acl_revoke(
    request: Request,
    user_id: str = Form(...),
    collection: str = Form(...),
):
    _require_admin_session(request)
    try:
        revoke_collection_access(user_id=user_id, qdrant_collection=collection)
    except AuthDisabledError:
        raise HTTPException(status_code=404, detail="Auth disabled")
    except Exception as e:
        return render_admin_error(request, title="Revoke Failed", message=str(e), back_href="/admin/acl")
    return RedirectResponse(url="/admin/acl", status_code=302)


@app.post("/auth/users", response_model=AuthUserCreateResponse)
async def auth_create_user(payload: AuthUserCreateRequest, request: Request):
    try:
        first_user = not has_any_users()
    except AuthDisabledError:
        raise HTTPException(status_code=404, detail="Auth disabled")
    except Exception as e:
        logger.error(f"[upload_service] Failed to check user state: {e}")
        raise HTTPException(status_code=500, detail="Failed to inspect user state")

    admin_token = os.environ.get("CTXCE_AUTH_ADMIN_TOKEN") or os.environ.get("CTXCE_AUTH_SHARED_TOKEN")
    if not first_user:
        if not admin_token:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin token not configured",
            )
        header = request.headers.get("X-Admin-Token")
        if not header or header != admin_token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid admin token",
            )

    try:
        user = create_user(payload.username, payload.password)
    except AuthDisabledError:
        raise HTTPException(status_code=404, detail="Auth disabled")
    except Exception as e:
        logger.error(f"[upload_service] Failed to create user: {e}")
        msg = str(e)
        if "UNIQUE" in msg or "unique" in msg:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Username already exists",
            )
        raise HTTPException(status_code=500, detail="Failed to create user")

    return AuthUserCreateResponse(user_id=user.get("user_id"), username=user.get("username"))


@app.post("/auth/login/password", response_model=AuthLoginResponse)
async def auth_login_password(payload: PasswordLoginRequest):
    try:
        user = authenticate_user(payload.username, payload.password)
    except AuthDisabledError:
        raise HTTPException(status_code=404, detail="Auth disabled")
    except Exception as e:
        logger.error(f"[upload_service] Error authenticating user: {e}")
        raise HTTPException(status_code=500, detail="Authentication error")

    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    meta: Optional[Dict[str, Any]] = None
    if payload.workspace:
        meta = {"workspace": payload.workspace}

    try:
        session = create_session(user_id=user.get("id"), metadata=meta)
    except AuthDisabledError:
        raise HTTPException(status_code=404, detail="Auth disabled")
    except Exception as e:
        logger.error(f"[upload_service] Failed to create session for user: {e}")
        raise HTTPException(status_code=500, detail="Failed to create auth session")

    return AuthLoginResponse(
        session_id=session.get("session_id"),
        user_id=session.get("user_id"),
        expires_at=session.get("expires_at"),
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0",
        qdrant_url=QDRANT_URL,
        work_dir=WORK_DIR
    )

@app.get("/api/v1/delta/status", response_model=StatusResponse)
async def get_status(workspace_path: str):
    """Get upload status for workspace."""
    try:
        # Get collection name
        if get_collection_name:
            repo_name = _extract_repo_name_from_path(workspace_path) if _extract_repo_name_from_path else None
            collection_name = get_collection_name(repo_name)
        else:
            collection_name = DEFAULT_COLLECTION

        # Get last sequence
        last_sequence = get_last_sequence(workspace_path)

        last_upload = None

        return StatusResponse(
            workspace_path=workspace_path,
            collection_name=collection_name,
            last_sequence=last_sequence,
            last_upload=last_upload,
            pending_operations=0,
            status="ready",
            server_info={
                "version": "1.0.0",
                "max_bundle_size_mb": MAX_BUNDLE_SIZE_MB,
                "supported_formats": ["tar.gz"]
            }
        )

    except Exception as e:
        logger.error(f"Error getting status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/delta/upload", response_model=UploadResponse)
async def upload_delta_bundle(
    request: Request,
    bundle: UploadFile = File(...),
    workspace_path: str = Form(...),
    collection_name: Optional[str] = Form(None),
    sequence_number: Optional[int] = Form(None),
    force: Optional[bool] = Form(False),
    source_path: Optional[str] = Form(None),
    logical_repo_id: Optional[str] = Form(None),
    session: Optional[str] = Form(None),
):
    """Upload and process delta bundle."""
    start_time = datetime.now()
    client_host = request.client.host if hasattr(request, 'client') and request.client else 'unknown'

    record: Optional[Dict[str, Any]] = None

    try:
        logger.info(f"[upload_service] Begin processing upload for workspace={workspace_path} from {client_host}")

        if AUTH_ENABLED:
            session_value = (session or "").strip()
            try:
                record = validate_session(session_value)
            except AuthDisabledError:
                record = None
            except Exception as e:
                logger.error(f"[upload_service] Failed to validate auth session for upload: {e}")
                raise HTTPException(
                    status_code=500,
                    detail="Failed to validate auth session",
                )
            if record is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid or expired session",
                )

        # Validate workspace path
        workspace = Path(workspace_path)
        if not workspace.is_absolute():
            workspace = Path(WORK_DIR) / workspace

        workspace_path = str(workspace.resolve())

        # Always derive repo_name from workspace_path for origin tracking
        repo_name = _extract_repo_name_from_path(workspace_path) if _extract_repo_name_from_path else None
        if not repo_name:
            repo_name = Path(workspace_path).name

        # Preserve any client-supplied collection name but allow server-side overrides
        client_collection_name = collection_name
        resolved_collection: Optional[str] = None

        # Resolve collection name, preferring server-side mapping for logical_repo_id when enabled
        if logical_repo_reuse_enabled() and logical_repo_id and find_collection_for_logical_repo:
            try:
                existing = find_collection_for_logical_repo(logical_repo_id, search_root=WORK_DIR)
            except Exception:
                existing = None
            if existing:
                resolved_collection = existing

        # Latent migration: when no explicit mapping exists yet for this logical_repo_id, but there is a
        # single existing collection mapping, prefer reusing it rather than creating a fresh collection.
        if logical_repo_reuse_enabled() and logical_repo_id and resolved_collection is None and get_collection_mappings:
            try:
                mappings = get_collection_mappings(search_root=WORK_DIR) or []
            except Exception:
                mappings = []

            if len(mappings) == 1:
                canonical = mappings[0]
                canonical_coll = canonical.get("collection_name")
                if canonical_coll:
                    resolved_collection = canonical_coll
                    if update_workspace_state:
                        try:
                            update_workspace_state(
                                workspace_path=canonical.get("container_path") or canonical.get("state_file"),
                                updates={"logical_repo_id": logical_repo_id},
                                repo_name=canonical.get("repo_name"),
                            )
                        except Exception as migrate_err:
                            logger.debug(
                                f"[upload_service] Failed to migrate logical_repo_id for existing mapping: {migrate_err}"
                            )

        # Finalize collection_name: prefer resolved server-side mapping, then client-supplied name,
        # then standard get_collection_name/DEFAULT_COLLECTION fallbacks.
        if resolved_collection is not None:
            collection_name = resolved_collection
        elif client_collection_name:
            collection_name = client_collection_name
        else:
            if get_collection_name and repo_name:
                collection_name = get_collection_name(repo_name)
            else:
                collection_name = DEFAULT_COLLECTION

        # Enforce collection write access for uploads when auth is enabled.
        # Semantics: "write" is sufficient for uploading/indexing content.
        if AUTH_ENABLED and CTXCE_MCP_ACL_ENFORCE:
            uid = str((record or {}).get("user_id") or "").strip()
            if not uid:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid or expired session",
                )
            try:
                allowed = has_collection_access(uid, str(collection_name), "write")
            except AuthDisabledError:
                allowed = True
            except Exception as e:
                logger.error(f"[upload_service] Failed to check collection access for upload: {e}")
                raise HTTPException(status_code=500, detail="Failed to check collection access")
            if not allowed:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Forbidden: write access to collection '{collection_name}' denied",
                )

        # Persist origin metadata for remote lookups (including client source_path)
        # Use slugged repo name (repo+16) for state so it matches ingest/watch_index usage
        try:
            if repo_name:
                workspace_leaf = Path(workspace_path).name
                if _SLUGGED_REPO_RE.match(workspace_leaf or ""):
                    slug_repo_name = workspace_leaf
                else:
                    workspace_key = get_workspace_key(workspace_path)
                    slug_repo_name = f"{repo_name}-{workspace_key}"
                container_workspace = str(Path(WORK_DIR) / slug_repo_name)

                try:
                    marker_dir = Path(WORK_DIR) / ".codebase" / "repos" / slug_repo_name
                    marker_dir.mkdir(parents=True, exist_ok=True)
                    (marker_dir / ".ctxce_managed_upload").write_text("1\n")
                except Exception:
                    pass

                # Persist logical_repo_id mapping for this slug/workspace when provided (feature-gated)
                if logical_repo_reuse_enabled() and logical_repo_id and update_workspace_state:
                    try:
                        update_workspace_state(
                            workspace_path=container_workspace,
                            updates={
                                "logical_repo_id": logical_repo_id,
                                "qdrant_collection": collection_name,
                            },
                            repo_name=slug_repo_name,
                        )
                    except Exception as state_err:
                        logger.debug(
                            f"[upload_service] Failed to persist logical_repo_id mapping: {state_err}"
                        )

                if update_repo_origin:
                    update_repo_origin(
                        workspace_path=container_workspace,
                        repo_name=slug_repo_name,
                        container_path=container_workspace,
                        source_path=source_path or workspace_path,
                        collection_name=collection_name,
                    )
        except Exception as origin_err:
            logger.debug(f"[upload_service] Failed to persist origin info: {origin_err}")

        # Validate bundle size
        if bundle.size and bundle.size > MAX_BUNDLE_SIZE_MB * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail=f"Bundle too large. Max size: {MAX_BUNDLE_SIZE_MB}MB"
            )

        # Save bundle to temporary file
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as temp_file:
            bundle_path = Path(temp_file.name)

            max_bytes = MAX_BUNDLE_SIZE_MB * 1024 * 1024
            if bundle.size and bundle.size > max_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=f"Bundle too large. Max size: {MAX_BUNDLE_SIZE_MB}MB"
                )

            # Stream upload to file while enforcing size
            total = 0
            chunk_size = 1024 * 1024
            while True:
                chunk = await bundle.read(chunk_size)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    try:
                        temp_file.close()
                        bundle_path.unlink(missing_ok=True)
                    except Exception:
                        pass
                    raise HTTPException(
                        status_code=413,
                        detail=f"Bundle too large. Max size: {MAX_BUNDLE_SIZE_MB}MB"
                    )
                temp_file.write(chunk)

        handed_off = False

        try:
            # Validate bundle format
            manifest = validate_bundle_format(bundle_path)
            bundle_id = manifest.get("bundle_id")
            manifest_sequence = manifest.get("sequence_number")

            # Check sequence number
            last_sequence = get_last_sequence(workspace_path)
            if sequence_number is None:
                if manifest_sequence is not None:
                    sequence_number = manifest_sequence
                else:
                    sequence_number = last_sequence + 1

            if not force and sequence_number is not None:
                if sequence_number != last_sequence + 1:
                    return UploadResponse(
                        success=False,
                        error={
                            "code": "SEQUENCE_MISMATCH",
                            "message": f"Expected sequence {last_sequence + 1}, got {sequence_number}",
                            "expected_sequence": last_sequence + 1,
                            "received_sequence": sequence_number,
                            "retry_after": 5000
                        }
                    )

            handed_off = True

            asyncio.create_task(
                _process_bundle_background(
                    workspace_path=workspace_path,
                    bundle_path=bundle_path,
                    manifest=manifest,
                    sequence_number=sequence_number,
                    bundle_id=bundle_id,
                )
            )

            return UploadResponse(
                success=True,
                bundle_id=bundle_id,
                sequence_number=sequence_number,
                processed_operations=None,
                processing_time_ms=None,
                next_sequence=sequence_number + 1 if sequence_number else None
            )

        finally:
            if not handed_off:
                try:
                    bundle_path.unlink()
                except Exception:
                    pass

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing upload: {e}")
        return UploadResponse(
            success=False,
            error={
                "code": "PROCESSING_ERROR",
                "message": f"Error processing bundle: {str(e)}"
            }
        )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": {
                "code": "INTERNAL_ERROR",
                "message": "Internal server error"
            }
        }
    )

def main():
    """Main entry point for the upload service."""
    host = os.environ.get("UPLOAD_SERVICE_HOST", "0.0.0.0")
    port = _int_env("UPLOAD_SERVICE_PORT", 8002)

    logger.info(f"Starting upload service on {host}:{port}")
    logger.info(f"Qdrant URL: {QDRANT_URL}")
    logger.info(f"Work directory: {WORK_DIR}")
    logger.info(f"Max bundle size: {MAX_BUNDLE_SIZE_MB}MB")

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        access_log=True
    )

if __name__ == "__main__":
    main()
