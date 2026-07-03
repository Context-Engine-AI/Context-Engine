#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import Request
from starlette.templating import Jinja2Templates
from jinja2 import select_autoescape

try:
    from scripts.workspace_state import is_staging_enabled
except Exception:
    is_staging_enabled = None  # type: ignore

_TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
_templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))
_templates.env.autoescape = select_autoescape(enabled_extensions=("html", "xml"), default=True)


def render_admin_login(
    request: Request,
    error: Optional[str] = None,
    status_code: int = 200,
) -> Any:
    return _templates.TemplateResponse(
        "admin/login.html",
        {"request": request, "title": "CTXCE Admin Login", "error": error},
        status_code=status_code,
    )


def render_admin_bootstrap(
    request: Request,
    error: Optional[str] = None,
    status_code: int = 200,
) -> Any:
    return _templates.TemplateResponse(
        "admin/bootstrap.html",
        {"request": request, "title": "CTXCE Admin Bootstrap", "error": error},
        status_code=status_code,
    )


def render_admin_acl(
    request: Request,
    users: Any,
    collections: Any,
    grants: Any,
    deletion_enabled: bool = False,
    work_dir: str = "/work",
    refresh_ms: int = 5000,
    status_code: int = 200,
) -> Any:
    return _templates.TemplateResponse(
        "admin/acl.html",
        {
            "request": request,
            "title": "CTXCE Admin ACL",
            "users": users,
            "collections": collections,
            "grants": grants,
            "deletion_enabled": bool(deletion_enabled),
            "work_dir": work_dir,
            "staging_enabled": bool(is_staging_enabled() if callable(is_staging_enabled) else False),
            "refresh_ms": int(refresh_ms) if refresh_ms is not None else 5000,
        },
        status_code=status_code,
    )


def render_admin_error(
    request: Request,
    title: str,
    message: str,
    back_href: str = "/admin",
    status_code: int = 400,
) -> Any:
    return _templates.TemplateResponse(
        "admin/error.html",
        {
            "request": request,
            "title": title,
            "message": message,
            "back_href": back_href,
        },
        status_code=status_code,
    )
