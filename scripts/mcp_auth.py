import os
from typing import Any, Dict, Optional

try:
    from scripts.logger import ValidationError
except Exception:

    class ValidationError(Exception):
        pass


try:
    from scripts.auth_backend import (
        AUTH_ENABLED as AUTH_ENABLED_AUTH,
        ACL_ALLOW_ALL as ACL_ALLOW_ALL_AUTH,
        validate_session as _auth_validate_session,
        has_collection_access as _has_collection_access,
    )
except Exception as _auth_backend_import_exc:
    _AUTH_BACKEND_IMPORT_ERROR = repr(_auth_backend_import_exc)
    AUTH_ENABLED_AUTH = (
        str(os.environ.get("CTXCE_AUTH_ENABLED", "0")).strip().lower() in {"1", "true", "yes", "on"}
    )
    ACL_ALLOW_ALL_AUTH = (
        str(os.environ.get("CTXCE_ACL_ALLOW_ALL", "0")).strip().lower() in {"1", "true", "yes", "on"}
    )

    def _auth_validate_session(session_id: str):  # type: ignore[no-redef]
        if AUTH_ENABLED_AUTH:
            raise ValidationError(
                f"Auth backend unavailable (import failed): {_AUTH_BACKEND_IMPORT_ERROR}"
            )
        return None

    def _has_collection_access(
        user_id: str, qdrant_collection: str, permission: str = "read"
    ) -> bool:  # type: ignore[no-redef]
        if AUTH_ENABLED_AUTH:
            raise ValidationError(
                f"Auth backend unavailable (import failed): {_AUTH_BACKEND_IMPORT_ERROR}"
            )
        return True


ACL_ENFORCE = (
    str(os.environ.get("CTXCE_MCP_ACL_ENFORCE", "0")).strip().lower()
    in {"1", "true", "yes", "on"}
)


def require_auth_session(session: Optional[str]) -> Optional[Dict[str, Any]]:
    if not AUTH_ENABLED_AUTH:
        return None
    sid = (session or "").strip()
    if not sid:
        raise ValidationError("Missing session for authorized operation")
    info = _auth_validate_session(sid)
    if not info:
        raise ValidationError("Invalid or expired session")
    return info


def require_collection_access(user_id: Optional[str], collection: str, perm: str) -> None:
    if not ACL_ENFORCE or not AUTH_ENABLED_AUTH:
        return
    if ACL_ALLOW_ALL_AUTH:
        return
    uid = (user_id or "").strip()
    if not uid:
        raise ValidationError("Not authorized: missing user id")
    if not _has_collection_access(uid, collection, perm):
        raise ValidationError(
            f"Forbidden: {perm} access to collection '{collection}' denied"
        )
