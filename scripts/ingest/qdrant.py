#!/usr/bin/env python3
"""
ingest/qdrant.py - Qdrant schema and I/O operations.

This module provides functions for Qdrant collection management, point operations,
and vector schema handling for the indexing pipeline.
"""
from __future__ import annotations

import os
import time
import hashlib
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional

from qdrant_client import QdrantClient, models

from scripts.ingest.config import (
    LEX_VECTOR_NAME,
    LEX_VECTOR_DIM,
    LEX_SPARSE_NAME,
    LEX_SPARSE_MODE,
    MINI_VECTOR_NAME,
    MINI_VEC_DIM,
    logical_repo_reuse_enabled,
)


# ---------------------------------------------------------------------------
# Collection tracking (thread-safe)
# ---------------------------------------------------------------------------
_ensured_lock = threading.Lock()
ENSURED_COLLECTIONS: set[str] = set()
ENSURED_COLLECTIONS_LAST_CHECK: dict[str, float] = {}

# TTL for ensured collections cache (default 1 hour). After TTL expires, entries are
# eligible for eviction to bound memory in long-running processes with many collections.
ENSURED_COLLECTIONS_TTL = float(os.environ.get("ENSURED_COLLECTIONS_TTL", "3600") or 3600)
# Maximum number of collections to track (prevents unbounded growth)
ENSURED_COLLECTIONS_MAX = int(os.environ.get("ENSURED_COLLECTIONS_MAX", "500") or 500)


def _cleanup_stale_ensured_collections_unlocked() -> int:
    """Remove stale entries from ENSURED_COLLECTIONS based on TTL.

    Returns number of entries removed.

    NOTE: Caller MUST hold _ensured_lock before calling this function.
    """
    if not ENSURED_COLLECTIONS_LAST_CHECK:
        return 0
    now = time.time()
    stale = []
    for coll, last_check in list(ENSURED_COLLECTIONS_LAST_CHECK.items()):
        if (now - last_check) > ENSURED_COLLECTIONS_TTL:
            stale.append(coll)
    for coll in stale:
        ENSURED_COLLECTIONS.discard(coll)
        ENSURED_COLLECTIONS_LAST_CHECK.pop(coll, None)
    return len(stale)


class CollectionNeedsRecreateError(Exception):
    """Raised when a collection needs to be recreated to add new vector types."""
    pass


PATTERN_VECTOR_NAME = "pattern_vector"
PATTERN_VECTOR_DIM = 64  # Structural pattern embedding dimension


def ensure_collection(client: QdrantClient, name: str, dim: int, vector_name: str):
    """Ensure collection exists with named vectors.

    Always includes dense (vector_name) and lexical (LEX_VECTOR_NAME).
    When REFRAG_MODE=1, also includes a compact mini vector (MINI_VECTOR_NAME).
    When PATTERN_VECTORS=1, also includes pattern_vector for structural similarity.
    """
    if not name:
        print("[BUG] ensure_collection called with name=None! Fix the caller - collection name is required.", flush=True)
        return
    backup_file = None
    try:
        info = client.get_collection(name)
        try:
            cfg = getattr(info.config.params, "vectors", None)
            sparse_cfg = getattr(info.config.params, "sparse_vectors", None)
            if isinstance(cfg, dict):
                has_lex = LEX_VECTOR_NAME in cfg
                has_mini = MINI_VECTOR_NAME in cfg
                has_sparse = sparse_cfg and LEX_SPARSE_NAME in (sparse_cfg if isinstance(sparse_cfg, dict) else {})

                if LEX_SPARSE_MODE and not has_sparse:
                    print(f"[COLLECTION_INFO] Collection {name} lacks sparse vector '{LEX_SPARSE_NAME}' - recreating...")
                    backup_file = _backup_memories_before_recreate(name)
                    try:
                        client.delete_collection(name)
                        print(f"[COLLECTION_INFO] Deleted existing collection {name}")
                    except Exception:
                        pass
                    raise CollectionNeedsRecreateError(f"Collection {name} needs sparse vectors")

                missing = {}
                if not has_lex:
                    missing[LEX_VECTOR_NAME] = models.VectorParams(
                        size=LEX_VECTOR_DIM, distance=models.Distance.COSINE
                    )

                try:
                    refrag_on = os.environ.get("REFRAG_MODE", "").strip().lower() in {
                        "1", "true", "yes", "on",
                    }
                except Exception:
                    refrag_on = False

                if refrag_on and not has_mini:
                    missing[MINI_VECTOR_NAME] = models.VectorParams(
                        size=int(os.environ.get("MINI_VEC_DIM", MINI_VEC_DIM) or MINI_VEC_DIM),
                        distance=models.Distance.COSINE,
                    )

                # Check for pattern vector
                try:
                    pattern_on = os.environ.get("PATTERN_VECTORS", "").strip().lower() in {
                        "1", "true", "yes", "on",
                    }
                    has_pattern = PATTERN_VECTOR_NAME in cfg
                except Exception:
                    pattern_on = False
                    has_pattern = False

                if pattern_on and not has_pattern:
                    missing[PATTERN_VECTOR_NAME] = models.VectorParams(
                        size=PATTERN_VECTOR_DIM,
                        distance=models.Distance.COSINE,
                    )

                if missing:
                    try:
                        client.update_collection(
                            collection_name=name, vectors_config=missing
                        )
                        print(f"[COLLECTION_SUCCESS] Successfully updated collection {name} with missing vectors")
                    except Exception as update_e:
                        print(f"[COLLECTION_WARNING] Cannot add missing vectors to {name} ({update_e}). Recreating collection...")
                        backup_file = _backup_memories_before_recreate(name)
                        try:
                            client.delete_collection(name)
                            print(f"[COLLECTION_INFO] Deleted existing collection {name}")
                        except Exception:
                            pass
                        raise CollectionNeedsRecreateError(f"Collection {name} needs recreation for new vectors")
        except CollectionNeedsRecreateError:
            print(f"[COLLECTION_INFO] Collection {name} needs recreation - proceeding...")
            raise
        except Exception as e:
            print(f"[COLLECTION_ERROR] Failed to update collection {name}: {e}")
            pass
        return
    except Exception as e:
        print(f"[COLLECTION_INFO] Creating new collection {name}: {type(e).__name__}")
        pass

    vectors_cfg = {
        vector_name: models.VectorParams(size=dim, distance=models.Distance.COSINE),
        LEX_VECTOR_NAME: models.VectorParams(
            size=LEX_VECTOR_DIM, distance=models.Distance.COSINE
        ),
    }
    try:
        if os.environ.get("REFRAG_MODE", "").strip().lower() in {"1", "true", "yes", "on"}:
            vectors_cfg[MINI_VECTOR_NAME] = models.VectorParams(
                size=int(os.environ.get("MINI_VEC_DIM", MINI_VEC_DIM) or MINI_VEC_DIM),
                distance=models.Distance.COSINE,
            )
    except Exception:
        pass
    try:
        if os.environ.get("PATTERN_VECTORS", "").strip().lower() in {"1", "true", "yes", "on"}:
            vectors_cfg[PATTERN_VECTOR_NAME] = models.VectorParams(
                size=PATTERN_VECTOR_DIM,
                distance=models.Distance.COSINE,
            )
    except Exception:
        pass

    sparse_cfg = None
    if LEX_SPARSE_MODE:
        sparse_cfg = {
            LEX_SPARSE_NAME: models.SparseVectorParams(
                index=models.SparseIndexParams(full_scan_threshold=5000)
            )
        }
    client.create_collection(
        collection_name=name,
        vectors_config=vectors_cfg,
        sparse_vectors_config=sparse_cfg,
        hnsw_config=models.HnswConfigDiff(m=16, ef_construct=256),
    )
    sparse_info = f", sparse: [{LEX_SPARSE_NAME}]" if sparse_cfg else ""
    print(f"[COLLECTION_INFO] Successfully created new collection {name} with vectors: {list(vectors_cfg.keys())}{sparse_info}")

    _restore_memories_after_recreate(name, backup_file)


def _backup_memories_before_recreate(name: str) -> Optional[str]:
    """Backup memories before recreating a collection."""
    backup_file = None
    try:
        import tempfile
        import subprocess
        import sys
        with tempfile.NamedTemporaryFile(mode='w', suffix='_memories_backup.json', delete=False) as f:
            backup_file = f.name
        print(f"[MEMORY_BACKUP] Backing up memories from {name} to {backup_file}")
        backup_script = Path(__file__).parent.parent / "memory_backup.py"
        result = subprocess.run([
            sys.executable, str(backup_script),
            "--collection", name,
            "--output", backup_file
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent.parent)
        if result.returncode != 0:
            print(f"[MEMORY_BACKUP_WARNING] Backup script failed: {result.stderr}")
            backup_file = None
    except Exception as backup_e:
        print(f"[MEMORY_BACKUP_WARNING] Failed to backup memories: {backup_e}")
        backup_file = None
    return backup_file


def _restore_memories_after_recreate(name: str, backup_file: Optional[str]):
    """Restore memories after recreating a collection."""
    strict_restore = False
    try:
        val = os.environ.get("STRICT_MEMORY_RESTORE", "")
        strict_restore = str(val or "").strip().lower() in {"1", "true", "yes", "on"}
    except Exception:
        strict_restore = False

    try:
        if backup_file and os.path.exists(backup_file):
            print(f"[MEMORY_RESTORE] Restoring memories from {backup_file}")
            import subprocess
            import sys

            restore_script = Path(__file__).parent.parent / "memory_restore.py"
            result = subprocess.run(
                [
                    sys.executable,
                    str(restore_script),
                    "--backup",
                    backup_file,
                    "--collection",
                    name,
                    "--skip-collection-creation",
                ],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent.parent,
            )

            if result.returncode == 0:
                print(f"[MEMORY_RESTORE] Successfully restored memories using {restore_script.name}")
            else:
                print(f"[MEMORY_RESTORE_WARNING] Restore script failed (exit {result.returncode})")
                if result.stdout:
                    print(f"[MEMORY_RESTORE_STDOUT] {result.stdout}")
                if result.stderr:
                    print(f"[MEMORY_RESTORE_STDERR] {result.stderr}")
                if strict_restore:
                    msg = result.stderr or result.stdout or f"exit code {result.returncode}"
                    raise RuntimeError(f"Memory restore failed for collection {name}: {msg}")

            try:
                os.unlink(backup_file)
                print(f"[MEMORY_RESTORE] Cleaned up backup file {backup_file}")
            except Exception:
                pass

        elif backup_file:
            print(f"[MEMORY_RESTORE_WARNING] Backup file {backup_file} not found")

    except Exception as restore_e:
        print(f"[MEMORY_RESTORE_ERROR] Failed to restore memories: {restore_e}")
        if strict_restore:
            raise


def recreate_collection(client: QdrantClient, name: str, dim: int, vector_name: str):
    """Drop and recreate collection with named vectors."""
    if not name:
        print("[BUG] recreate_collection called with name=None! Fix the caller - collection name is required.", flush=True)
        return
    try:
        client.delete_collection(name)
    except Exception:
        pass
    vectors_cfg = {
        vector_name: models.VectorParams(size=dim, distance=models.Distance.COSINE),
        LEX_VECTOR_NAME: models.VectorParams(
            size=LEX_VECTOR_DIM, distance=models.Distance.COSINE
        ),
    }
    try:
        if os.environ.get("REFRAG_MODE", "").strip().lower() in {"1", "true", "yes", "on"}:
            vectors_cfg[MINI_VECTOR_NAME] = models.VectorParams(
                size=int(os.environ.get("MINI_VEC_DIM", MINI_VEC_DIM) or MINI_VEC_DIM),
                distance=models.Distance.COSINE,
            )
    except Exception:
        pass
    try:
        if os.environ.get("PATTERN_VECTORS", "").strip().lower() in {"1", "true", "yes", "on"}:
            vectors_cfg[PATTERN_VECTOR_NAME] = models.VectorParams(
                size=PATTERN_VECTOR_DIM,
                distance=models.Distance.COSINE,
            )
    except Exception:
        pass
    sparse_cfg = None
    if LEX_SPARSE_MODE:
        sparse_cfg = {
            LEX_SPARSE_NAME: models.SparseVectorParams(
                index=models.SparseIndexParams(full_scan_threshold=5000)
            )
        }
    client.create_collection(
        collection_name=name,
        vectors_config=vectors_cfg,
        sparse_vectors_config=sparse_cfg,
        hnsw_config=models.HnswConfigDiff(m=16, ef_construct=256),
    )


def ensure_payload_indexes(client: QdrantClient, collection: str):
    """Create helpful payload indexes if they don't exist (idempotent)."""
    for field in (
        "metadata.language",
        "metadata.path_prefix",
        "metadata.repo_id",
        "metadata.repo_rel_path",
        "metadata.repo",
        "metadata.kind",
        "metadata.symbol",
        "metadata.symbol_path",
        "metadata.imports",
        "metadata.calls",
        "metadata.file_hash",
        "metadata.ingested_at",
        "metadata.last_modified_at",
        "metadata.churn_count",
        "metadata.author_count",
        "pid_str",
    ):
        try:
            client.create_payload_index(
                collection_name=collection,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        except Exception:
            pass


def ensure_collection_and_indexes_once(
    client: QdrantClient,
    collection: str,
    dim: int,
    vector_name: str | None,
) -> None:
    """Ensure collection and indexes exist (cached per-process).

    Thread-safe: uses _ensured_lock to protect cache mutations.
    Periodically cleans up stale entries based on TTL to prevent unbounded growth.
    """
    if not collection:
        return

    # Fast path: check if already ensured (read under lock for consistency)
    with _ensured_lock:
        if collection in ENSURED_COLLECTIONS:
            try:
                ping_seconds = float(os.environ.get("ENSURED_COLLECTION_PING_SECONDS", "0") or 0)
            except Exception:
                ping_seconds = 0.0

            if ping_seconds <= 0:
                return

            now = time.time()
            last = ENSURED_COLLECTIONS_LAST_CHECK.get(collection, 0.0)
            if (now - last) < ping_seconds:
                return
            # Need to ping - release lock for network call
            need_ping = True
        else:
            need_ping = False

    # Ping outside lock to avoid holding lock during network I/O
    if need_ping:
        try:
            client.get_collection(collection)
            with _ensured_lock:
                ENSURED_COLLECTIONS_LAST_CHECK[collection] = time.time()
            return
        except Exception:
            with _ensured_lock:
                ENSURED_COLLECTIONS.discard(collection)
                ENSURED_COLLECTIONS_LAST_CHECK.pop(collection, None)
            # Fall through to re-ensure

    # Slow path: need to ensure collection exists
    with _ensured_lock:
        # Double-check after acquiring lock (another thread may have ensured it)
        if collection in ENSURED_COLLECTIONS:
            return

        # Periodic cleanup: remove stale entries when cache is large
        if len(ENSURED_COLLECTIONS) > ENSURED_COLLECTIONS_MAX // 2:
            _cleanup_stale_ensured_collections_unlocked()

        # Enforce max size: evict oldest entries if at capacity
        while len(ENSURED_COLLECTIONS) >= ENSURED_COLLECTIONS_MAX:
            _cleanup_stale_ensured_collections_unlocked()
            if len(ENSURED_COLLECTIONS) >= ENSURED_COLLECTIONS_MAX:
                # Still at capacity after cleanup - evict oldest
                oldest_coll = None
                oldest_time = float('inf')
                for c, t in ENSURED_COLLECTIONS_LAST_CHECK.items():
                    if t < oldest_time:
                        oldest_time = t
                        oldest_coll = c
                if oldest_coll:
                    ENSURED_COLLECTIONS.discard(oldest_coll)
                    ENSURED_COLLECTIONS_LAST_CHECK.pop(oldest_coll, None)
                else:
                    break  # Safety: avoid infinite loop

    # Create collection outside lock to avoid holding lock during network I/O
    ensure_collection(client, collection, dim, vector_name)
    ensure_payload_indexes(client, collection)

    # Mark as ensured
    with _ensured_lock:
        ENSURED_COLLECTIONS.add(collection)
        ENSURED_COLLECTIONS_LAST_CHECK[collection] = time.time()


def get_indexed_file_hash(
    client: QdrantClient,
    collection: str,
    file_path: str,
    *,
    repo_id: str | None = None,
    repo_rel_path: str | None = None,
) -> str:
    """Return previously indexed file hash for this logical path, or empty string."""
    if not collection:
        print("[BUG] get_indexed_file_hash called with collection=None! Fix the caller.", flush=True)
        return ""
    if logical_repo_reuse_enabled() and repo_id and repo_rel_path:
        try:
            filt = models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.repo_id", match=models.MatchValue(value=repo_id)
                    ),
                    models.FieldCondition(
                        key="metadata.repo_rel_path",
                        match=models.MatchValue(value=repo_rel_path),
                    ),
                ]
            )
            points, _ = client.scroll(
                collection_name=collection,
                scroll_filter=filt,
                with_payload=True,
                limit=1,
            )
            if points:
                md = (points[0].payload or {}).get("metadata") or {}
                fh = md.get("file_hash")
                if fh:
                    return str(fh)
        except Exception:
            pass

    try:
        filt = models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.path", match=models.MatchValue(value=file_path)
                )
            ]
        )
        points, _ = client.scroll(
            collection_name=collection,
            scroll_filter=filt,
            with_payload=True,
            limit=1,
        )
        if points:
            md = (points[0].payload or {}).get("metadata") or {}
            fh = md.get("file_hash")
            if fh:
                return str(fh)
    except Exception:
        return ""
    return ""


def delete_points_by_path(client: QdrantClient, collection: str, file_path: str):
    """Delete all points for a given file path."""
    if not collection:
        print("[BUG] delete_points_by_path called with collection=None! Fix the caller.", flush=True)
        return
    try:
        filt = models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.path", match=models.MatchValue(value=file_path)
                )
            ]
        )
        client.delete(
            collection_name=collection,
            points_selector=models.FilterSelector(filter=filt),
            wait=True,
        )
    except Exception:
        pass


def upsert_points(
    client: QdrantClient, collection: str, points: List[models.PointStruct]
):
    """Upsert points with retry and batching."""
    if not points:
        return
    if not collection:
        print("[BUG] upsert_points called with collection=None! Fix the caller.", flush=True)
        return
    try:
        bsz = int(os.environ.get("INDEX_UPSERT_BATCH", "256") or 256)
    except Exception:
        bsz = 256
    try:
        retries = int(os.environ.get("INDEX_UPSERT_RETRIES", "3") or 3)
    except Exception:
        retries = 3
    try:
        backoff = float(os.environ.get("INDEX_UPSERT_BACKOFF", "0.5") or 0.5)
    except Exception:
        backoff = 0.5

    for i in range(0, len(points), max(1, bsz)):
        batch = points[i : i + max(1, bsz)]
        attempt = 0
        while True:
            try:
                client.upsert(collection_name=collection, points=batch, wait=True)
                break
            except Exception:
                attempt += 1
                if attempt >= retries:
                    sub_size = max(1, bsz // 4)
                    for j in range(0, len(batch), sub_size):
                        sub = batch[j : j + sub_size]
                        try:
                            client.upsert(
                                collection_name=collection, points=sub, wait=True
                            )
                        except Exception:
                            pass
                    break
                else:
                    try:
                        time.sleep(backoff * attempt)
                    except Exception:
                        pass


def hash_id(text: str, path: str, start: int, end: int) -> int:
    """Generate a stable hash ID for a chunk."""
    h = hashlib.sha1(
        f"{path}:{start}-{end}\n{text}".encode("utf-8", errors="ignore")
    ).hexdigest()
    return int(h[:16], 16)


def embed_batch(model, texts: List[str]) -> List[List[float]]:
    """Embed a batch of texts using the embedding model."""
    return [vec.tolist() for vec in model.embed(texts)]
