import os
import json
import tarfile
import hashlib
import re
import logging
from pathlib import Path
from typing import Dict, Any, Optional


try:
    from scripts.workspace_state import (
        _extract_repo_name_from_path,
        get_staging_targets,
        get_collection_state_snapshot,
        is_staging_enabled,
    )
except ImportError as exc:
    raise ImportError(
        "upload_delta_bundle requires scripts.workspace_state; ensure the module is available"
    ) from exc


logger = logging.getLogger(__name__)

WORK_DIR = os.environ.get("WORK_DIR") or os.environ.get("WORKDIR") or "/work"
_SLUGGED_REPO_RE = re.compile(r"^.+-[0-9a-f]{16}(?:_old)?$")


def get_workspace_key(workspace_path: str) -> str:
    """Generate 16-char hash for collision avoidance in remote uploads.

    Remote uploads may have identical folder names from different users,
    so uses longer hash than local indexing (8-chars) to ensure uniqueness.

    Both host paths (/home/user/project/repo) and container paths (/work/repo)
    should generate the same key for the same repository.
    """
    repo_name = Path(workspace_path).name
    if _SLUGGED_REPO_RE.match(repo_name):
        return repo_name[-16:]
    return hashlib.sha256(repo_name.encode("utf-8")).hexdigest()[:16]


def _cleanup_empty_dirs(path: Path, stop_at: Path) -> None:
    """Recursively remove empty directories up to stop_at (exclusive)."""
    try:
        path = path.resolve()
        stop_at = stop_at.resolve()
    except Exception:
        pass
    while True:
        try:
            if path == stop_at or not path.exists() or not path.is_dir():
                break
            if any(path.iterdir()):
                break
            path.rmdir()
            path = path.parent
        except Exception:
            break


def process_delta_bundle(workspace_path: str, bundle_path: Path, manifest: Dict[str, Any]) -> Dict[str, int]:
    """Process delta bundle and return operation counts."""
    operations_count = {
        "created": 0,
        "updated": 0,
        "deleted": 0,
        "moved": 0,
        "skipped": 0,
        "failed": 0,
    }

    try:
        # CRITICAL: Always materialize writes under WORK_DIR using a slugged repo directory.
        # Do NOT write directly into the client-supplied workspace_path, since that may be a host
        # path (e.g. /home/user/repo) that is not mounted/visible to the watcher/indexer.
        workspace_leaf = Path(workspace_path).name

        repo_name_for_state: Optional[str] = None

        serving_slug: Optional[str] = None
        active_slug: Optional[str] = None
        if _extract_repo_name_from_path and get_collection_state_snapshot:
            try:
                repo_name_for_state = _extract_repo_name_from_path(workspace_path)
                if repo_name_for_state:
                    snapshot = get_collection_state_snapshot(workspace_path=None, repo_name=repo_name_for_state)  # type: ignore[arg-type]
                    serving_slug = snapshot.get("serving_repo_slug")
                    active_slug = snapshot.get("active_repo_slug")
            except Exception:
                serving_slug = None
                active_slug = None

        slug_order: list[str] = []
        serving_candidate: Optional[str] = None
        if serving_slug and _SLUGGED_REPO_RE.match(serving_slug):
            serving_candidate = serving_slug
        if active_slug and _SLUGGED_REPO_RE.match(active_slug) and active_slug not in slug_order:
            slug_order.append(active_slug)

        # If staging is active, we must mirror uploads into BOTH the canonical slug and
        # the "*_old" slug. Relying purely on snapshot detection is brittle (e.g. when
        # the client workspace_path is a host path). When we can infer a canonical slug,
        # force both targets.
        staging_active = False
        staging_gate = bool(is_staging_enabled() if callable(is_staging_enabled) else False)
        try:
            if serving_slug and str(serving_slug).endswith("_old"):
                staging_active = True
        except Exception:
            staging_active = False

        if not staging_gate:
            staging_active = False

        def _append_slug(slug: Optional[str]) -> None:
            if slug and _SLUGGED_REPO_RE.match(slug) and slug not in slug_order:
                slug_order.append(slug)

        if repo_name_for_state and _SLUGGED_REPO_RE.match(repo_name_for_state):
            canonical_slug = repo_name_for_state[:-4] if repo_name_for_state.endswith("_old") else repo_name_for_state
            old_slug_candidate = (
                repo_name_for_state if repo_name_for_state.endswith("_old") else f"{canonical_slug}_old"
            )
            if staging_active:
                slug_order = []
                _append_slug(canonical_slug)
                _append_slug(old_slug_candidate)
            elif not slug_order:
                _append_slug(canonical_slug)
                old_slug_path = Path(WORK_DIR) / old_slug_candidate
                if old_slug_path.exists():
                    _append_slug(old_slug_candidate)

        if not slug_order:
            if _SLUGGED_REPO_RE.match(workspace_leaf):
                slug_order.append(workspace_leaf)
            else:
                if _extract_repo_name_from_path:
                    repo_name = _extract_repo_name_from_path(workspace_path) or workspace_leaf
                else:
                    repo_name = workspace_leaf
                workspace_key = get_workspace_key(workspace_path)
                slug_order.append(f"{repo_name}-{workspace_key}")

        # Best-effort: if staging is active according to workspace_state, ensure we mirror to
        # both the canonical slug and its *_old slug.
        if staging_gate and (not staging_active) and get_staging_targets and _extract_repo_name_from_path:
            try:
                repo_name_for_staging = _extract_repo_name_from_path(workspace_path) or slug_order[0]
                targets = get_staging_targets(workspace_path=workspace_path, repo_name=repo_name_for_staging)
                if isinstance(targets, dict) and targets.get("staging"):
                    staging_active = True
            except Exception as staging_err:
                logger.debug(f"[upload_service] Failed to detect staging: {staging_err}")

        def _slug_exists(slug: str) -> bool:
            try:
                return (
                    (Path(WORK_DIR) / slug).exists()
                    or (Path(WORK_DIR) / ".codebase" / "repos" / slug).exists()
                )
            except Exception:
                return False

        if staging_gate and (not staging_active) and slug_order:
            primary = slug_order[0]
            if _SLUGGED_REPO_RE.match(primary):
                canonical = primary[:-4] if primary.endswith("_old") else primary
                inferred_old = primary if primary.endswith("_old") else f"{canonical}_old"
                if _slug_exists(inferred_old):
                    staging_active = True

        if staging_gate and staging_active and slug_order:
            primary = slug_order[0]
            if _SLUGGED_REPO_RE.match(primary):
                canonical = primary[:-4] if primary.endswith("_old") else primary
                old_slug = primary if primary.endswith("_old") else f"{canonical}_old"
                desired = [canonical, old_slug]
                slug_order = [s for s in desired if _SLUGGED_REPO_RE.match(s)]
        elif staging_gate and not staging_active and serving_candidate:
            # Ignore serving slugs when staging is disabled; keep deterministic non-staging writes.
            if serving_candidate in slug_order:
                slug_order = [s for s in slug_order if s != serving_candidate]

        if staging_gate:
            try:
                logger.info(f"[upload_service] Delta bundle targets (staging={staging_active}): {slug_order}")
            except Exception:
                pass

        replica_roots: Dict[str, Path] = {}
        for slug in slug_order:
            path = Path(WORK_DIR) / slug
            path.mkdir(parents=True, exist_ok=True)
            try:
                marker_dir = Path(WORK_DIR) / ".codebase" / "repos" / slug
                marker_dir.mkdir(parents=True, exist_ok=True)
                (marker_dir / ".ctxce_managed_upload").write_text("1\n")
            except Exception:
                pass
            replica_roots[slug] = path.resolve()

        primary_slug = slug_order[0]
        workspace_root = replica_roots[primary_slug]

        def _safe_join(base: Path, rel: str) -> Path:
            # SECURITY: Prevent path traversal / absolute-path writes by ensuring the resolved
            # candidate path stays within the intended workspace root.
            rp = Path(str(rel))
            if str(rp) in {".", ""}:
                raise ValueError("Invalid operation path")
            if rp.is_absolute():
                raise ValueError(f"Absolute paths are not allowed: {rel}")
            base_resolved = base.resolve()
            candidate = (base_resolved / rp).resolve()
            try:
                ok = candidate.is_relative_to(base_resolved)
            except Exception:
                ok = os.path.commonpath([str(base_resolved), str(candidate)]) == str(base_resolved)
            if not ok:
                raise ValueError(f"Path escapes workspace: {rel}")
            return candidate

        with tarfile.open(bundle_path, "r:gz") as tar:
            ops_member = None
            for member in tar.getnames():
                if member.endswith("metadata/operations.json"):
                    ops_member = member
                    break

            if not ops_member:
                raise ValueError("operations.json not found in bundle")

            ops_file = tar.extractfile(ops_member)
            if not ops_file:
                raise ValueError("Cannot extract operations.json")

            operations_data = json.loads(ops_file.read().decode("utf-8"))
            operations = operations_data.get("operations", [])

            # Best-effort: extract git history metadata for watcher to ingest
            try:
                git_member = None
                for member in tar.getnames():
                    if member.endswith("metadata/git_history.json"):
                        git_member = member
                        break
                if git_member:
                    git_file = tar.extractfile(git_member)
                    if git_file:
                        history_bytes = git_file.read()
                        bundle_id = manifest.get("bundle_id") or "unknown"
                        for root in replica_roots.values():
                            try:
                                history_dir = root / ".remote-git"
                                history_dir.mkdir(parents=True, exist_ok=True)
                                history_path = history_dir / f"git_history_{bundle_id}.json"
                                history_path.write_bytes(history_bytes)
                            except Exception as write_err:
                                logger.debug(
                                    f"[upload_service] Failed to write git history manifest for {root}: {write_err}",
                                )
            except Exception as git_err:
                logger.debug(f"[upload_service] Error extracting git history metadata: {git_err}")

            def _apply_operation_to_workspace(workspace_root: Path) -> bool:
                """Apply a single file operation to a workspace. Returns True on success."""
                nonlocal operations_count, op_type, rel_path, tar
                
                target_path = _safe_join(workspace_root, rel_path)

                safe_source_path = None
                source_rel_path = None
                if op_type == "moved":
                    source_rel_path = operation.get("source_path") or operation.get("source_relative_path")
                    if source_rel_path:
                        safe_source_path = _safe_join(workspace_root, source_rel_path)

                try:
                    if op_type == "created":
                        file_member = None
                        for member in tar.getnames():
                            if member.endswith(f"files/created/{rel_path}"):
                                file_member = member
                                break

                        if file_member:
                            file_content = tar.extractfile(file_member)
                            if file_content:
                                target_path.parent.mkdir(parents=True, exist_ok=True)
                                target_path.write_bytes(file_content.read())
                                return True
                            else:
                                return False
                        else:
                            return False

                    elif op_type == "updated":
                        file_member = None
                        for member in tar.getnames():
                            if member.endswith(f"files/updated/{rel_path}"):
                                file_member = member
                                break

                        if file_member:
                            file_content = tar.extractfile(file_member)
                            if file_content:
                                target_path.parent.mkdir(parents=True, exist_ok=True)
                                target_path.write_bytes(file_content.read())
                                return True
                            else:
                                return False
                        else:
                            return False

                    elif op_type == "deleted":
                        if target_path.exists():
                            target_path.unlink(missing_ok=True)
                            return True
                        else:
                            return True  # Already deleted

                    elif op_type == "moved":
                        if safe_source_path and safe_source_path.exists():
                            target_path.parent.mkdir(parents=True, exist_ok=True)
                            safe_source_path.rename(target_path)
                            return True
                        # Remote uploads may not have the source file on the server (e.g. staging
                        # mirrors). In that case, clients can embed the destination content under
                        # files/moved/<dest>.
                        file_member = None
                        for member in tar.getnames():
                            if member.endswith(f"files/moved/{rel_path}"):
                                file_member = member
                                break
                        if file_member:
                            file_content = tar.extractfile(file_member)
                            if file_content:
                                target_path.parent.mkdir(parents=True, exist_ok=True)
                                target_path.write_bytes(file_content.read())
                                return True
                            return False
                        return False

                    else:
                        logger.warning(f"[upload_service] Unknown operation type: {op_type}")
                        return False
                except Exception as e:
                    logger.debug(f"[upload_service] Failed to apply {op_type} to {rel_path} in {workspace_root}: {e}")
                    return False

            for operation in operations:
                op_type = operation.get("operation")
                rel_path = operation.get("path")

                if not rel_path:
                    operations_count["skipped"] += 1
                    continue

                sanitized_path = rel_path
                skipped_due_to_exact_slug = False
                for slug in replica_roots.keys():
                    if sanitized_path == slug:
                        skipped_due_to_exact_slug = True
                        break
                    prefix = f"{slug}/"
                    if sanitized_path.startswith(prefix):
                        sanitized_path = sanitized_path[len(prefix):]
                        break

                if skipped_due_to_exact_slug or not sanitized_path:
                    logger.debug(
                        f"[upload_service] Skipping operation {op_type} for path {rel_path}: "
                        "appears to reference slug root directly.",
                    )
                    operations_count["skipped"] += 1
                    continue

                rel_path = sanitized_path

                replica_results: Dict[str, bool] = {}
                for slug, root in replica_roots.items():
                    replica_results[slug] = _apply_operation_to_workspace(root)

                success_any = any(replica_results.values())
                success_all = all(replica_results.values())
                if success_any:
                    operations_count.setdefault(op_type, 0)
                    operations_count[op_type] = operations_count.get(op_type, 0) + 1
                    if not success_all:
                        logger.debug(
                            f"[upload_service] Partial success for {op_type} {rel_path}: {replica_results}"
                        )
                else:
                    operations_count["failed"] += 1

        return operations_count

    except Exception as e:
        logger.error(f"Error processing delta bundle: {e}")
        raise
