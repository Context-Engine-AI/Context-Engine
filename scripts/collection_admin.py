import os
import json
import re
import shutil
import time
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional, List

from scripts.auth_backend import mark_collection_deleted

try:
    from qdrant_client import QdrantClient
    from qdrant_client import models as qmodels
except Exception:
    QdrantClient = None  # type: ignore
    qmodels = None  # type: ignore

try:
    from scripts.qdrant_client_manager import pooled_qdrant_client
except Exception:
    pooled_qdrant_client = None

try:
    from scripts.workspace_state import get_collection_mappings
except Exception:
    get_collection_mappings = None


_SLUGGED_REPO_RE = re.compile(r"^.+-[0-9a-f]{16}(?:_old)?$")
_MARKER_NAME = ".ctxce_managed_upload"


def _resolve_work_root(work_dir: Optional[str] = None) -> Path:
    return Path(
        work_dir
        or os.environ.get("WORK_DIR")
        or os.environ.get("WORKDIR")
        or "/work"
    ).resolve()


def _resolve_codebase_root(work_root: Path) -> Path:
    env_root = (
        os.environ.get("CTXCE_CODEBASE_ROOT")
        or os.environ.get("CODEBASE_ROOT")
        or ""
    ).strip()

    candidates = []
    if env_root:
        candidates.append(Path(env_root))
    candidates.append(work_root)
    candidates.append(work_root.parent)

    for candidate in candidates:
        try:
            base = candidate.resolve()
        except Exception:
            base = candidate
        try:
            if (base / ".codebase" / "repos").exists():
                return base
        except Exception:
            continue

    return work_root


def _delete_path_tree(p: Path) -> bool:
    try:
        if not p.exists():
            return False
        if p.is_dir():
            shutil.rmtree(p)
            return True
        p.unlink()
        return True
    except Exception:
        return False


def _read_state_collection(state_path: Path) -> str:
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return str(data.get("qdrant_collection") or "").strip()
    except Exception:
        pass
    return ""


def _managed_upload_marker_path(
    *,
    work_root: Path,
    slug_name: str,
    marker_root: Optional[Path] = None,
) -> Path:
    # Marker is stored with per-repo metadata, not inside the repo workspace tree.
    base = marker_root or work_root
    return base / ".codebase" / "repos" / slug_name / _MARKER_NAME


def _is_managed_upload_workspace_dir(
    p: Path,
    *,
    work_root: Path,
    marker_root: Optional[Path] = None,
) -> bool:
    try:
        if not p.is_dir():
            return False
        if p.parent.resolve() != work_root:
            return False
        if not _SLUGGED_REPO_RE.match(p.name or ""):
            return False
        return _managed_upload_marker_path(
            work_root=work_root,
            marker_root=marker_root,
            slug_name=p.name,
        ).exists()
    except Exception:
        return False


def _cleanup_state_files_for_mapping(
    mapping: Dict[str, Any],
    *,
    work_root: Optional[Path] = None,
    codebase_root: Optional[Path] = None,
) -> int:
    removed = 0
    state_file = mapping.get("state_file")
    if isinstance(state_file, str) and state_file.strip():
        p = Path(state_file)
        state_dir = p.parent
        try:
            base = (
                codebase_root
                or work_root
                or Path(os.environ.get("WORK_DIR") or os.environ.get("WORKDIR") or "/work")
            ).resolve()
        except Exception:
            base = codebase_root or work_root or Path(
                os.environ.get("WORK_DIR") or os.environ.get("WORKDIR") or "/work"
            )

        # Multi-repo mode stores per-repo metadata under /work/.codebase/repos/<repo>/
        try:
            repos_root = (base / ".codebase" / "repos").resolve()
            sd = state_dir.resolve()
            if sd != repos_root and str(sd).startswith(str(repos_root) + os.sep):
                if _delete_path_tree(sd):
                    removed += 1
                return removed
        except Exception:
            pass

        # Single-repo mode stores metadata under <workspace>/.codebase/
        if _delete_path_tree(p):
            removed += 1
        _delete_path_tree(state_dir / "cache.json")
        try:
            for sym in state_dir.glob("symbols_*.json"):
                _delete_path_tree(sym)
        except Exception:
            pass
    return removed


def delete_collection_everywhere(
    *,
    collection: str,
    work_dir: Optional[str] = None,
    qdrant_url: Optional[str] = None,
    cleanup_fs: bool = True,
) -> Dict[str, Any]:
    enabled = (
        str(os.environ.get("CTXCE_ADMIN_COLLECTION_DELETE_ENABLED", "0")).strip().lower()
        in {"1", "true", "yes", "on"}
    )
    if not enabled:
        raise PermissionError("Collection deletion is disabled by server configuration")

    name = (collection or "").strip()
    if not name:
        raise ValueError("collection is required")

    work_root = _resolve_work_root(work_dir)
    codebase_root = _resolve_codebase_root(work_root)

    out: Dict[str, Any] = {
        "collection": name,
        "qdrant_deleted": False,
        "registry_marked_deleted": False,
        "deleted_state_files": 0,
        "deleted_managed_workspaces": 0,
    }

    target_is_old = name.endswith("_old")

    # 1) Delete Qdrant collection
    try:
        if pooled_qdrant_client is not None:
            with pooled_qdrant_client(url=qdrant_url, api_key=os.environ.get("QDRANT_API_KEY")) as cli:
                try:
                    cli.delete_collection(collection_name=name)
                    out["qdrant_deleted"] = True
                except Exception:
                    out["qdrant_deleted"] = False
    except Exception:
        out["qdrant_deleted"] = False

    # 2) Mark deleted in registry DB
    try:
        mark_collection_deleted(name)
        out["registry_marked_deleted"] = True
    except Exception:
        out["registry_marked_deleted"] = False

    # 3) Cleanup workspace state metadata + managed upload workspaces
    if not cleanup_fs:
        return out

    mappings = []
    try:
        if get_collection_mappings is not None:
            mappings = get_collection_mappings(search_root=str(codebase_root)) or []
    except Exception:
        mappings = []

    # NOTE: logically linked worktrees still share the
    # primary repo's qdrant_collection in their state.json files. When the UI targets the lineage
    # collection directly, no mapping below matches, so filesystem cleanup is a no-op and we keep
    # both the metadata (`.codebase/repos/<slug>`) and the workspace on disk. This conservative
    # behavior is intentional until we have branch-aware deletion semantics—we do not want to
    # cascade-delete shared worktrees that may host future branch/version state.
    for m in mappings:
        try:
            if str(m.get("collection_name") or "").strip() != name:
                continue

            # Safety: when targeting a staging clone collection ("*_old"), never delete
            # a non-"*_old" workspace on disk even if its state temporarily points at the clone.
            if target_is_old:
                repo = str(m.get("repo_name") or "").strip()
                if not repo.endswith("_old"):
                    continue

            container_path = m.get("container_path")
            if isinstance(container_path, str) and container_path.strip():
                p = Path(container_path)
                try:
                    p = p.resolve()
                except Exception:
                    pass

                if target_is_old:
                    # For staging clone workspaces, delete the workspace dir directly when it
                    # is under the expected work_root and ends with "_old".
                    try:
                        if p.parent.resolve() == work_root and (p.name or "").endswith("_old"):
                            if _delete_path_tree(p):
                                out["deleted_managed_workspaces"] += 1
                    except Exception:
                        pass
                else:
                    if _is_managed_upload_workspace_dir(p, work_root=work_root, marker_root=codebase_root):
                        if _delete_path_tree(p):
                            out["deleted_managed_workspaces"] += 1

            # Cleanup state metadata after workspace deletion so the marker still exists
            # when authorizing the filesystem delete.
            out["deleted_state_files"] += _cleanup_state_files_for_mapping(
                m,
                work_root=work_root,
                codebase_root=codebase_root,
            )
        except Exception:
            continue

    return out


def _normalize_qdrant_url(qdrant_url: Optional[str]) -> str:
    url = (qdrant_url or os.environ.get("QDRANT_URL") or "http://qdrant:6333").strip()
    if not url:
        url = "http://qdrant:6333"
    return url.rstrip("/")


def copy_collection_qdrant(
    *,
    source: str,
    target: Optional[str] = None,
    qdrant_url: Optional[str] = None,
    overwrite: bool = False,
) -> str:
    """Copy a Qdrant collection using the pooled client.

    Returns the target collection name.
    """
    src = (source or "").strip()
    if not src:
        raise ValueError("source collection is required")

    dest = (target or f"{src}__copy__{datetime.utcnow().strftime('%Y%m%d%H%M%S')}").strip()
    if not dest:
        raise ValueError("target collection is required")

    base_url = _normalize_qdrant_url(qdrant_url)
    api_key = os.environ.get("QDRANT_API_KEY") or ""
    headers = {"api-key": api_key} if api_key else {}

    def _copy_client_timeout_seconds() -> Optional[float]:
        try:
            raw = (
                os.environ.get("CTXCE_COPY_COLLECTION_TIMEOUT")
                or os.environ.get("QDRANT_TIMEOUT")
                or ""
            )
            raw = str(raw).strip()
            if not raw:
                # Default: no timeout. Staging clone is background and may take a long time.
                return None
            if raw.lower() in {"0", "none", "null", "false", "off", "disabled"}:
                return None
            return float(raw)
        except Exception:
            return None

    copied = False

    def _manual_copy_points() -> None:
        if QdrantClient is None or qmodels is None:
            raise RuntimeError("QdrantClient unavailable for manual collection copy")
        cli = QdrantClient(url=base_url, api_key=api_key or None, timeout=_copy_client_timeout_seconds())
        try:
            if overwrite:
                try:
                    cli.delete_collection(collection_name=dest)
                except Exception:
                    pass

            try:
                src_info = cli.get_collection(collection_name=src)
            except Exception as exc:
                raise RuntimeError(f"Failed to fetch source collection config for {src}: {exc}") from exc

            vectors_config = None
            sparse_vectors_config = None
            try:
                params = getattr(getattr(src_info, "config", None), "params", None)
                if params is not None:
                    vectors_config = getattr(params, "vectors", None)
                    sparse_vectors_config = getattr(params, "sparse_vectors", None)
            except Exception:
                vectors_config = None
                sparse_vectors_config = None

            if vectors_config is None:
                raise RuntimeError(f"Cannot determine vectors config for source collection {src}")

            try:
                cli.create_collection(
                    collection_name=dest,
                    vectors_config=vectors_config,
                    sparse_vectors_config=sparse_vectors_config,
                )
            except Exception as exc:
                # Allow clone to proceed if collection already exists.
                if "already exists" not in str(exc).lower():
                    raise RuntimeError(
                        f"Failed to create destination collection {dest}: {exc}"
                    ) from exc

            # Allow transient network hiccups when verifying the destination collection.
            verify_attempts = max(1, int(os.environ.get("CTXCE_COPY_VERIFY_RETRIES", "3") or "3"))
            verify_delay = float(os.environ.get("CTXCE_COPY_VERIFY_DELAY", "2") or "2")
            last_err: Optional[Exception] = None
            for _ in range(verify_attempts):
                try:
                    cli.get_collection(collection_name=dest)
                    last_err = None
                    break
                except Exception as exc:
                    last_err = exc
                    time.sleep(max(0.1, verify_delay))
            if last_err is not None:
                raise RuntimeError(
                    f"Destination collection {dest} unavailable after creation: {last_err}"
                ) from last_err

            offset = None
            batch_limit = int(os.environ.get("CTXCE_COPY_COLLECTION_BATCH", "512") or "512")
            while True:
                try:
                    points, next_offset = cli.scroll(
                        collection_name=src,
                        limit=batch_limit,
                        offset=offset,
                        with_payload=True,
                        with_vectors=True,
                    )
                except Exception as exc:
                    raise RuntimeError(f"Failed to scroll points from {src}: {exc}") from exc

                if points:
                    structured: List[qmodels.PointStruct] = []
                    for record in points:
                        if record is None:
                            continue
                        point_id = getattr(record, "id", None)
                        payload = getattr(record, "payload", None)
                        vector = None
                        if hasattr(record, "vector") and getattr(record, "vector") is not None:
                            vector = getattr(record, "vector")
                        elif hasattr(record, "vectors") and getattr(record, "vectors") is not None:
                            vector = getattr(record, "vectors")
                        structured.append(
                            qmodels.PointStruct(id=point_id, vector=vector, payload=payload)
                        )
                    if structured:
                        try:
                            cli.upsert(collection_name=dest, points=structured)
                        except Exception as exc:
                            raise RuntimeError(f"Failed to upsert points into {dest}: {exc}") from exc

                if next_offset is None:
                    break
                offset = next_offset
        finally:
            try:
                cli.close()
            except Exception:
                pass

    def _count_points(name: str) -> Optional[int]:
        if QdrantClient is None:
            return None
        cli = QdrantClient(url=base_url, api_key=api_key or None, timeout=_copy_client_timeout_seconds())
        try:
            res = cli.count(collection_name=name, exact=True)
            return int(getattr(res, "count", 0))
        except Exception:
            return None
        finally:
            try:
                cli.close()
            except Exception:
                pass

    source_count = _count_points(src)

    if pooled_qdrant_client is not None:
        with pooled_qdrant_client(url=base_url, api_key=api_key or None) as cli:
            if overwrite:
                try:
                    cli.delete_collection(collection_name=dest)
                except Exception:
                    # Fall back to HTTP delete below if needed
                    pass
            try:
                copy_method = getattr(cli, "copy_collection", None)
                if callable(copy_method):
                    copy_method(collection_name=src, new_collection_name=dest)
                    copied = True
            except AttributeError:
                copied = False
            except Exception as exc:
                raise RuntimeError(f"Failed to copy collection {src} -> {dest}: {exc}") from exc

    if not copied:
        # Always run the manual scroll+upsert copy. Many Qdrant deployments (including ours)
        # either lack /clone entirely or return success while creating an empty collection.
        # The manual path guarantees the destination gets the exact same points/payloads/vectors.
        _manual_copy_points()

    return dest
