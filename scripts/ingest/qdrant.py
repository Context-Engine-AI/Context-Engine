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
# Collection tracking
# ---------------------------------------------------------------------------
ENSURED_COLLECTIONS: set[str] = set()
ENSURED_COLLECTIONS_LAST_CHECK: dict[str, float] = {}


class CollectionNeedsRecreateError(Exception):
    """Raised when a collection needs to be recreated to add new vector types."""
    pass


PATTERN_VECTOR_NAME = "pattern_vector"
PATTERN_VECTOR_DIM = 64  # Structural pattern embedding dimension

PAYLOAD_INDEX_FIELDS = (
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
)

_SCHEMA_MODES = {"legacy", "validate", "create", "migrate"}

# ---------------------------------------------------------------------------
# Quantization configuration
# ---------------------------------------------------------------------------
def _get_quantization_config() -> models.QuantizationConfig | None:
    """Get quantization config based on QDRANT_QUANTIZATION env var.

    Options:
        - none: No quantization (default)
        - scalar: INT8 scalar quantization (4x smaller, 2-4x faster)
        - binary: Binary quantization (32x smaller, very fast but less accurate)
    """
    quant_mode = os.environ.get("QDRANT_QUANTIZATION", "none").strip().lower()

    if quant_mode in {"none", "", "0", "false", "off"}:
        return None

    if quant_mode == "scalar":
        return models.ScalarQuantization(
            scalar=models.ScalarQuantizationConfig(
                type=models.ScalarType.INT8,
                quantile=0.99,  # Clip outliers for better distribution
                always_ram=True,  # Keep quantized vectors in RAM for speed
            )
        )

    if quant_mode == "binary":
        return models.BinaryQuantization(
            binary=models.BinaryQuantizationConfig(
                always_ram=True,
            )
        )

    print(f"[COLLECTION_WARNING] Unknown QDRANT_QUANTIZATION={quant_mode!r}; using none")
    return None


def _normalize_schema_mode(schema_mode: str | None) -> str:
    mode = (schema_mode or "").strip().lower()
    if not mode:
        return "legacy"
    if mode in _SCHEMA_MODES:
        return mode
    print(f"[COLLECTION_WARNING] Unknown schema_mode={schema_mode!r}; using legacy behavior")
    return "legacy"


def _desired_vector_configs(
    dim: int,
    vector_name: str,
) -> tuple[Dict[str, models.VectorParams], Dict[str, models.SparseVectorParams] | None]:
    vectors_cfg: Dict[str, models.VectorParams] = {
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
    return vectors_cfg, sparse_cfg


def _as_vector_params_diff(vec: models.VectorParams) -> Any:
    diff_cls = getattr(models, "VectorParamsDiff", None)
    if diff_cls is None:
        return vec
    try:
        return diff_cls(
            size=getattr(vec, "size", None) or getattr(vec, "dim", None),
            distance=getattr(vec, "distance", None),
            hnsw_config=getattr(vec, "hnsw_config", None),
            quantization_config=getattr(vec, "quantization_config", None),
            on_disk=getattr(vec, "on_disk", None),
            datatype=getattr(vec, "datatype", None),
            multivector_config=getattr(vec, "multivector_config", None),
        )
    except Exception:
        return vec


def _prepare_vector_update_config(
    vectors_cfg: Dict[str, models.VectorParams],
) -> Dict[str, Any]:
    return {name: _as_vector_params_diff(params) for name, params in vectors_cfg.items()}


def _missing_payload_indexes(info: Any) -> list[str]:
    schema = getattr(info, "payload_schema", None)
    if isinstance(schema, dict):
        existing = set(schema.keys())
    else:
        existing = set()
    return [field for field in PAYLOAD_INDEX_FIELDS if field not in existing]


def get_collection_vector_names(
    client: QdrantClient,
    name: str,
) -> tuple[set[str] | None, set[str] | None]:
    """Return (dense_vector_names, sparse_vector_names) for a collection.

    Returns (None, None) if the schema cannot be inspected.
    """
    if not name:
        return None, None
    try:
        info = client.get_collection(name)
    except Exception:
        return None, None

    vectors_cfg = getattr(info.config.params, "vectors", None)
    if isinstance(vectors_cfg, dict):
        dense_names = set(vectors_cfg.keys())
    else:
        dense_names = None

    sparse_cfg = getattr(info.config.params, "sparse_vectors", None)
    if isinstance(sparse_cfg, dict):
        sparse_names = set(sparse_cfg.keys())
    else:
        sparse_names = set() if dense_names is not None else None

    return dense_names, sparse_names


def _ensure_collection_with_mode(
    client: QdrantClient,
    name: str,
    dim: int,
    vector_name: str,
    mode: str,
) -> None:
    if not name:
        print("[BUG] ensure_collection called with name=None! Fix the caller - collection name is required.", flush=True)
        return

    vectors_cfg, sparse_cfg = _desired_vector_configs(dim, vector_name)
    desired_vectors = set(vectors_cfg.keys())
    desired_sparse = set(sparse_cfg.keys()) if sparse_cfg else set()

    try:
        info = client.get_collection(name)
    except Exception as e:
        if mode == "validate":
            raise RuntimeError(
                f"Collection {name} does not exist; run with --schema-mode create "
                "or --recreate to create it."
            ) from e
        quant_cfg = _get_quantization_config()
        client.create_collection(
            collection_name=name,
            vectors_config=vectors_cfg,
            sparse_vectors_config=sparse_cfg,
            hnsw_config=models.HnswConfigDiff(m=16, ef_construct=256),
            quantization_config=quant_cfg,
        )
        sparse_info = f", sparse: [{LEX_SPARSE_NAME}]" if sparse_cfg else ""
        quant_info = f", quantization: {os.environ.get('QDRANT_QUANTIZATION', 'none')}" if quant_cfg else ""
        print(
            f"[COLLECTION_INFO] Successfully created new collection {name} with vectors: "
            f"{list(vectors_cfg.keys())}{sparse_info}{quant_info}"
        )
        ensure_payload_indexes(client, name)
        return

    cfg = getattr(info.config.params, "vectors", None)
    if not isinstance(cfg, dict):
        if mode == "validate":
            raise RuntimeError(
                f"Collection {name} uses unnamed vector schema; cannot validate. "
                "Recreate the collection with named vectors."
            )
        return

    sparse_existing_cfg = getattr(info.config.params, "sparse_vectors", None)
    existing_vectors = set(cfg.keys())
    existing_sparse = (
        set(sparse_existing_cfg.keys())
        if isinstance(sparse_existing_cfg, dict)
        else set()
    )

    missing_vectors = sorted(desired_vectors - existing_vectors)
    missing_sparse = sorted(desired_sparse - existing_sparse)
    missing_indexes = _missing_payload_indexes(info)

    if mode in {"validate", "create"}:
        if missing_vectors or missing_sparse or missing_indexes:
            parts = []
            if missing_vectors:
                parts.append(f"vectors={missing_vectors}")
            if missing_sparse:
                parts.append(f"sparse={missing_sparse}")
            if missing_indexes:
                parts.append(f"payload_indexes={missing_indexes}")
            detail = ", ".join(parts) if parts else "unknown schema mismatch"
            raise RuntimeError(
                f"Collection {name} schema mismatch ({detail}). "
                "Run with --schema-mode migrate or --recreate to update it."
            )
        return

    # migrate: allow additive changes but avoid destructive recreation
    if missing_sparse:
        raise RuntimeError(
            f"Collection {name} missing sparse vectors {missing_sparse}. "
            "Recreate the collection (e.g., --recreate) to add sparse vectors, "
            "or disable LEX_SPARSE_MODE."
        )

    if missing_vectors:
        missing_cfg = {k: vectors_cfg[k] for k in missing_vectors if k in vectors_cfg}
        missing_cfg = _prepare_vector_update_config(missing_cfg)
        try:
            client.update_collection(collection_name=name, vectors_config=missing_cfg)
            print(f"[COLLECTION_SUCCESS] Successfully updated collection {name} with missing vectors")
        except Exception as update_e:
            raise RuntimeError(
                f"Cannot add missing vectors to {name} ({update_e}). "
                "Recreate the collection (e.g., --recreate) to apply schema changes."
            ) from update_e

    if missing_indexes:
        ensure_payload_indexes(client, name)


def ensure_collection(
    client: QdrantClient,
    name: str,
    dim: int,
    vector_name: str,
    *,
    schema_mode: str | None = None,
):
    """Ensure collection exists with named vectors.

    Always includes dense (vector_name) and lexical (LEX_VECTOR_NAME).
    When REFRAG_MODE=1, also includes a compact mini vector (MINI_VECTOR_NAME).
    When PATTERN_VECTORS=1, also includes pattern_vector for structural similarity.
    """
    mode = _normalize_schema_mode(schema_mode)
    if mode != "legacy":
        _ensure_collection_with_mode(client, name, dim, vector_name, mode)
        return
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
                    print(
                        f"[COLLECTION_WARNING] Collection {name} lacks sparse vector '{LEX_SPARSE_NAME}'. "
                        "Sparse indexing will be skipped for this run."
                    )

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
                        update_cfg = _prepare_vector_update_config(missing)
                        client.update_collection(
                            collection_name=name, vectors_config=update_cfg
                        )
                        print(f"[COLLECTION_SUCCESS] Successfully updated collection {name} with missing vectors")
                    except Exception as update_e:
                        print(
                            f"[COLLECTION_WARNING] Cannot add missing vectors to {name} ({update_e}). "
                            "Continuing without them for this run."
                        )
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
    quant_cfg = _get_quantization_config()
    client.create_collection(
        collection_name=name,
        vectors_config=vectors_cfg,
        sparse_vectors_config=sparse_cfg,
        hnsw_config=models.HnswConfigDiff(m=16, ef_construct=256),
        quantization_config=quant_cfg,
    )
    sparse_info = f", sparse: [{LEX_SPARSE_NAME}]" if sparse_cfg else ""
    quant_info = f", quantization: {os.environ.get('QDRANT_QUANTIZATION', 'none')}" if quant_cfg else ""
    print(f"[COLLECTION_INFO] Successfully created new collection {name} with vectors: {list(vectors_cfg.keys())}{sparse_info}{quant_info}")

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
    quant_cfg = _get_quantization_config()
    client.create_collection(
        collection_name=name,
        vectors_config=vectors_cfg,
        sparse_vectors_config=sparse_cfg,
        hnsw_config=models.HnswConfigDiff(m=16, ef_construct=256),
        quantization_config=quant_cfg,
    )


def ensure_payload_indexes(client: QdrantClient, collection: str):
    """Create helpful payload indexes if they don't exist (idempotent)."""
    for field in PAYLOAD_INDEX_FIELDS:
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
    *,
    schema_mode: str | None = None,
) -> None:
    """Ensure collection and indexes exist (cached per-process)."""
    if not collection:
        return
    mode = _normalize_schema_mode(schema_mode)
    if mode not in {"validate", "create"} and collection in ENSURED_COLLECTIONS:
        try:
            ping_seconds = float(os.environ.get("ENSURED_COLLECTION_PING_SECONDS", "0") or 0)
        except Exception:
            ping_seconds = 0.0

        if ping_seconds <= 0:
            return

        try:
            now = time.time()
            last = ENSURED_COLLECTIONS_LAST_CHECK.get(collection, 0.0)
            if (now - last) < ping_seconds:
                return
            client.get_collection(collection)
            ENSURED_COLLECTIONS_LAST_CHECK[collection] = now
            return
        except Exception:
            try:
                ENSURED_COLLECTIONS.discard(collection)
            except Exception:
                pass
            try:
                ENSURED_COLLECTIONS_LAST_CHECK.pop(collection, None)
            except Exception:
                pass
    ensure_collection(client, collection, dim, vector_name, schema_mode=mode)
    if mode in {"legacy", "migrate"}:
        ensure_payload_indexes(client, collection)
    if mode != "validate":
        ENSURED_COLLECTIONS.add(collection)
        try:
            ENSURED_COLLECTIONS_LAST_CHECK[collection] = time.time()
        except Exception:
            pass


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
