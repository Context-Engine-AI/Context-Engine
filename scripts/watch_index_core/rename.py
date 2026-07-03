"""Move/rename fast-path helpers."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Optional, Tuple

from qdrant_client import QdrantClient, models

import scripts.ingest_code as idx

from .config import LOGGER


def _rename_in_store(
    client: QdrantClient,
    src_collection: str,
    src: Path,
    dest: Path,
    dest_collection: Optional[str] = None,
) -> Tuple[int, Optional[str]]:
    """Best-effort: if dest content hash matches previously indexed src hash,
    update points in-place to the new path without re-embedding.

    Returns number of points moved, or -1 if not applicable/failure.
    """

    if dest_collection is None:
        dest_collection = src_collection
    try:
        if not dest.exists() or dest.is_dir():
            return -1, None
        try:
            text = dest.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return -1, None
        dest_hash = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()
        prev = idx.get_indexed_file_hash(client, src_collection, str(src))
        LOGGER.debug(
            "rename fast-path candidate src=%s dest=%s prev_hash=%s dest_hash=%s",
            str(src),
            str(dest),
            prev,
            dest_hash,
        )
        if not prev or prev != dest_hash:
            return -1, prev if prev else None

        moved = 0
        next_offset = None
        while True:
            filt = models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.path", match=models.MatchValue(value=str(src))
                    )
                ]
            )
            points, next_offset = client.scroll(
                collection_name=src_collection,
                scroll_filter=filt,
                with_payload=True,
                with_vectors=True,
                limit=256,
                offset=next_offset,
            )
            if not points:
                break
            new_points = []
            for rec in points:
                payload = rec.payload or {}
                md = payload.get("metadata") or {}
                code = md.get("code") or ""
                try:
                    start_line = int(md.get("start_line") or 1)
                    end_line = int(md.get("end_line") or start_line)
                except Exception:
                    start_line, end_line = 1, 1
                new_id = idx.hash_id(code, str(dest), start_line, end_line)

                # Update metadata path fields
                new_md = dict(md)
                new_md["path"] = str(dest)
                new_md["path_prefix"] = str(dest.parent)
                # Recompute dual-path hints
                cur_path = str(dest)
                host_root = (
                    str(os.environ.get("HOST_INDEX_PATH") or "").strip().rstrip("/")
                )
                # Detect Windows drive letter using splitdrive (not raw colon check)
                drive, _ = os.path.splitdrive(host_root)
                if drive:  # Non-empty drive means Windows path like "C:"
                    host_root = ""
                host_path = None
                container_path = None
                try:
                    if cur_path.startswith("/work/") and host_root:
                        rel = cur_path[len("/work/") :]
                        host_path = os.path.realpath(os.path.join(host_root, rel))
                        container_path = cur_path
                    else:
                        host_path = cur_path
                        if host_root and cur_path.startswith(host_root + "/"):
                            rel = cur_path[len(host_root) + 1 :]
                            container_path = "/work/" + rel
                except Exception:
                    host_path = cur_path
                    container_path = cur_path if cur_path.startswith("/work/") else None
                new_md["host_path"] = host_path
                new_md["container_path"] = container_path

                new_payload = dict(payload)
                new_payload["metadata"] = new_md

                vec = rec.vector  # Named or unnamed vector(s)
                try:
                    new_points.append(
                        models.PointStruct(id=new_id, vector=vec, payload=new_payload)
                    )
                except Exception as e:
                    LOGGER.warning(
                        "[rename] Failed to create point id=%s for %s: %s",
                        new_id, str(dest), e,
                    )
                    continue
            if new_points:
                LOGGER.debug(
                    "rename fast-path upserting %d chunk(s) %s -> %s into %s",
                    len(new_points),
                    str(src),
                    str(dest),
                    dest_collection,
                )
                idx.upsert_points(client, dest_collection, new_points)
                moved += len(new_points)
            if next_offset is None:
                break

        try:
            idx.delete_points_by_path(client, src_collection, str(src))
        except Exception as e:
            LOGGER.warning(
                "[rename] Failed to delete source points from %s for %s -> %s: %s",
                src_collection, str(src), str(dest), e,
            )
            # Return partial success - points were moved but source cleanup failed
            return moved, dest_hash
        return moved, dest_hash
    except Exception as exc:  # pragma: no cover - defensive logging
        try:
            LOGGER.warning(
                "[rename_debug] rename failed for %s -> %s: %s",
                str(src),
                str(dest),
                exc,
            )
        except Exception:
            pass
        return -1, None


__all__ = ["_rename_in_store"]
