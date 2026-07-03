#!/usr/bin/env python3
import os
import hashlib
from pathlib import Path
from typing import Tuple

from qdrant_client import QdrantClient, models

COLLECTION = os.environ.get("COLLECTION_NAME", "codebase")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
API_KEY = os.environ.get("QDRANT_API_KEY")
ROOT = Path(os.environ.get("PRUNE_ROOT", ".")).resolve()
GRAPH_COLLECTION = os.environ.get("GRAPH_COLLECTION_NAME", f"{COLLECTION}_graph")


def sha1_file(path: Path) -> str:
    try:
        data = path.read_bytes()
    except Exception:
        return ""
    return hashlib.sha1(data).hexdigest()


def delete_by_path(client: QdrantClient, path_str: str) -> int:
    flt = models.Filter(
        must=[
            models.FieldCondition(
                key="metadata.path", match=models.MatchValue(value=path_str)
            )
        ]
    )
    try:
        res = client.delete(
            collection_name=COLLECTION,
            points_selector=models.FilterSelector(filter=flt),
        )
        return 1
    except Exception:
        return 0


def delete_graph_edges_by_path(client: QdrantClient, path_str: str) -> int:
    """Best-effort deletion for graph-edge collections (if present).

    Some deployments store symbol-graph edges in a separate Qdrant collection
    (commonly `${COLLECTION}_graph`). Those points may reference a file path as
    either caller or callee; delete both to prevent stale graph results.
    """
    if not path_str:
        return 0

    flt = models.Filter(
        should=[
            models.FieldCondition(
                key="caller_path", match=models.MatchValue(value=path_str)
            ),
            models.FieldCondition(
                key="callee_path", match=models.MatchValue(value=path_str)
            ),
        ]
    )
    try:
        res = client.delete(
            collection_name=GRAPH_COLLECTION,
            points_selector=models.FilterSelector(filter=flt),
        )
        # Qdrant responses vary by client version; return 1 as "success" when count isn't available.
        deleted_count = None
        result_attr = getattr(res, "result", None)
        if isinstance(result_attr, dict):
            v = result_attr.get("deleted")
            if isinstance(v, int):
                deleted_count = v
        if deleted_count is None:
            v = getattr(res, "deleted", None)
            if isinstance(v, int):
                deleted_count = v
        if deleted_count is None:
            deleted_count = 1
        return deleted_count
    except Exception:
        # Non-fatal: graph collection may not exist in this deployment.
        return 0


def main():
    client = QdrantClient(url=QDRANT_URL, api_key=API_KEY or None)

    seen = set()
    removed_missing = 0
    removed_mismatch = 0
    removed_graph_edges = 0

    next_page = None
    while True:
        points, next_page = client.scroll(
            collection_name=COLLECTION,
            with_payload=True,
            limit=256,
            offset=next_page,
            scroll_filter=None,
        )
        if not points:
            break
        for p in points:
            md = (p.payload or {}).get("metadata") or {}
            path_str = md.get("path")
            file_hash = md.get("file_hash")
            if not path_str or path_str in seen:
                continue
            seen.add(path_str)
            abs_path = (
                ROOT / Path(path_str).relative_to("/work")
                if path_str.startswith("/work/")
                else ROOT / path_str
            )
            if not abs_path.exists():
                removed_missing += delete_by_path(client, path_str)
                removed_graph_edges += delete_graph_edges_by_path(client, path_str)
                print(f"[prune] removed missing file points: {path_str}")
                continue
            current_hash = sha1_file(abs_path)
            if file_hash and current_hash and current_hash != file_hash:
                removed_mismatch += delete_by_path(client, path_str)
                removed_graph_edges += delete_graph_edges_by_path(client, path_str)
                print(f"[prune] removed outdated points (hash mismatch): {path_str}")

        if next_page is None:
            break

    print(
        "Prune complete. "
        f"removed_missing={removed_missing}, "
        f"removed_mismatch={removed_mismatch}, "
        f"removed_graph_edges={removed_graph_edges}"
    )


if __name__ == "__main__":
    main()
