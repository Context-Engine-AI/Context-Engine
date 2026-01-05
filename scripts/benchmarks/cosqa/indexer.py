#!/usr/bin/env python3
"""
CoSQA Corpus Indexer for Context-Engine Benchmarks.

Indexes the CoSQA code corpus into a dedicated Qdrant collection using
our existing embedding pipeline (BGE + lexical vectors).

Features:
- **AST-aware processing**: Uses tree-sitter/AST symbol extraction like main pipeline
- **Semantic chunking**: Uses chunk_semantic for code-aware chunking
- Progress tracking with resume capability
- Uses standard Context-Engine embedding pipeline
- Stores metadata for result mapping back to original IDs
- Config-aware fingerprinting for smart collection reuse
- Automatic invalidation on model/config changes
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from qdrant_client import QdrantClient, models

from scripts.ingest.chunking import chunk_by_tokens, chunk_lines, chunk_semantic
from scripts.ingest.pipeline import build_information
from scripts.ingest.vectors import project_mini, extract_pattern_vector
from scripts.ingest.qdrant import (
    hash_id,
    embed_batch,
    get_collection_vector_names,
    PATTERN_VECTOR_NAME,
    PATTERN_VECTOR_DIM,
)
from scripts.utils import (
    lex_hash_vector_text as _lex_hash_vector_text,
    lex_sparse_vector_text as _lex_sparse_vector_text,
)
from scripts.ingest.config import (
    LEX_VECTOR_NAME,
    LEX_VECTOR_DIM,
    MINI_VECTOR_NAME,
    MINI_VEC_DIM,
    LEX_SPARSE_NAME,
    LEX_SPARSE_MODE,
)

# Import Context-Engine's AST/symbol extraction (optional)
try:
    from scripts.ingest.symbols import _extract_symbols, _choose_symbol_for_chunk
    from scripts.ingest.metadata import _get_imports_calls
    _AST_AVAILABLE = True
except ImportError:
    _AST_AVAILABLE = False
    def _extract_symbols(_lang, _text):  # noqa: unused args for fallback
        return []
    def _choose_symbol_for_chunk(_start, _end, _symbols):  # noqa: unused args for fallback
        return "", "", ""
    def _get_imports_calls(_lang, _text):  # noqa: unused args for fallback
        return [], []

from scripts.embedder import DEFAULT_MODEL

# Default configuration
DEFAULT_COLLECTION = "cosqa-corpus"
DEFAULT_BATCH_SIZE = 256  # Larger batches for faster embedding throughput
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", DEFAULT_MODEL)

# Metadata key for corpus fingerprint (stored in collection)
CORPUS_FINGERPRINT_KEY = "_corpus_fingerprint"


def get_qdrant_client() -> QdrantClient:
    """Get Qdrant client with standard configuration."""
    url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    api_key = os.environ.get("QDRANT_API_KEY")
    timeout = int(os.environ.get("QDRANT_TIMEOUT", "60"))
    return QdrantClient(url=url, api_key=api_key or None, timeout=timeout)


def get_embedding_model():
    """Get embedding model using standard Context-Engine factory."""
    try:
        from scripts.embedder import get_embedding_model as _get_model
        return _get_model()
    except ImportError:
        from fastembed import TextEmbedding
        model_name = os.environ.get("EMBEDDING_MODEL", DEFAULT_MODEL)
        return TextEmbedding(model_name=model_name)


def get_model_dimension(model) -> int:
    """Get embedding dimension from model."""
    try:
        from scripts.embedder import get_model_dimension as _get_dim
        model_name = os.environ.get("EMBEDDING_MODEL", DEFAULT_MODEL)
        return _get_dim(model_name)
    except ImportError:
        # Probe dimension
        vec = list(model.embed(["test"]))[0]
        return len(vec)


def _get_config_fingerprint() -> str:
    """Get fingerprint of current embedding/vector configuration.

    Includes all settings that would require reindexing if changed:
    - Embedding model
    - Lexical vector dimensions
    - ReFRAG/mini vector settings
    - Chunking settings (semantic vs micro)
    - Sparse/pattern vector toggles
    - AST enrichment (v2 = enriched with symbols/imports/calls)
    """
    from scripts.ingest.config import LEX_VECTOR_DIM, MINI_VEC_DIM

    config_parts = [
        f"model:{EMBEDDING_MODEL}",
        f"lex_dim:{LEX_VECTOR_DIM}",
        f"refrag_mode:{os.environ.get('REFRAG_MODE', '0')}",
        f"mini_dim:{MINI_VEC_DIM}",
        f"mini_seed:{os.environ.get('MINI_VEC_SEED', '1337')}",
        f"lex_sparse:{os.environ.get('LEX_SPARSE_MODE', '0')}",
        f"pattern_vectors:{os.environ.get('PATTERN_VECTORS', '0')}",
        f"index_micro:{os.environ.get('INDEX_MICRO_CHUNKS', '0')}",
        f"micro_tokens:{os.environ.get('MICRO_CHUNK_TOKENS', '16')}",
        f"micro_stride:{os.environ.get('MICRO_CHUNK_STRIDE', '')}",
        f"semantic_chunks:{os.environ.get('INDEX_SEMANTIC_CHUNKS', '1')}",
        f"chunk_lines:{os.environ.get('INDEX_CHUNK_LINES', '120')}",
        f"chunk_overlap:{os.environ.get('INDEX_CHUNK_OVERLAP', '20')}",
        f"use_tree_sitter:{os.environ.get('USE_TREE_SITTER', '1')}",
        f"enhanced_ast:{os.environ.get('INDEX_USE_ENHANCED_AST', '1')}",
        f"ast_enriched:v2",  # Triggers reindex when AST enrichment changes
    ]
    return hashlib.sha256("|".join(config_parts).encode()).hexdigest()[:8]


def compute_corpus_fingerprint(corpus_entries: List[Dict[str, Any]]) -> str:
    """Compute a stable fingerprint for a corpus + config to detect changes.
    
    Includes:
    - Sorted code_ids + all content that affects embeddings (text, docstring, func_name)
    - Embedding model and vector config (invalidates on model/config change)
    - ReFRAG/mini vector settings (invalidates on processing change)
    """
    hasher = hashlib.sha256()
    
    # Config fingerprint first - ensures reindex on config change
    config_fp = _get_config_fingerprint()
    hasher.update(f"config:{config_fp}".encode())
    
    # Corpus content fingerprint - include all fields that affect embeddings
    for entry in sorted(corpus_entries, key=lambda e: e.get("code_id", "")):
        code_id = entry.get("code_id", "")
        text = entry.get("text", "")
        docstring = entry.get("docstring", "")
        func_name = entry.get("func_name", "")
        # Hash the same content that gets embedded (docstring + text, func_name for lexical)
        # Use full content for accurate fingerprinting (truncation caused stale reuse)
        combined = f"{func_name}\n{docstring}\n{text}"
        hasher.update(f"{code_id}:{combined}".encode("utf-8", errors="ignore"))
    # Note: We intentionally don't include count in fingerprint
    # This allows resume to work when some entries fail to index
    
    return hasher.hexdigest()[:16]


def get_collection_fingerprint(collection: str) -> Optional[str]:
    """Get the stored corpus fingerprint from a collection, if any."""
    try:
        client = get_qdrant_client()
        points, _ = client.scroll(
            collection_name=collection,
            limit=1,
            with_payload=[CORPUS_FINGERPRINT_KEY],
        )
        if points and points[0].payload:
            return points[0].payload.get(CORPUS_FINGERPRINT_KEY)
    except Exception:
        pass
    return None


def collection_matches_corpus(collection: str, corpus_entries: List[Dict[str, Any]]) -> bool:
    """Check if an existing collection matches the given corpus.
    
    Note: We intentionally don't require exact point count match.
    If fingerprint matches, the resume logic will fill in any missing entries.
    """
    try:
        client = get_qdrant_client()
        info = client.get_collection(collection)
        if info.points_count == 0:
            return False  # Empty collection, needs indexing
        # Fingerprint check (content-based, not count-based)
        stored_fp = get_collection_fingerprint(collection)
        if stored_fp:
            return stored_fp == compute_corpus_fingerprint(corpus_entries)
        return False
    except Exception:
        return False


def create_collection(
    client: QdrantClient,
    collection: str,
    dim: int,
    recreate: bool = False,
) -> None:
    """Create or recreate the CoSQA collection with proper vector config."""
    from scripts.ingest.config import (
        LEX_VECTOR_NAME,
        LEX_VECTOR_DIM,
        MINI_VECTOR_NAME,
        MINI_VEC_DIM,
        LEX_SPARSE_NAME,
        LEX_SPARSE_MODE,
    )
    from scripts.ingest.qdrant import PATTERN_VECTOR_NAME, PATTERN_VECTOR_DIM

    # Use sanitized model name for vector name (matches hybrid_search)
    from scripts.utils import sanitize_vector_name
    model_name = os.environ.get("EMBEDDING_MODEL", DEFAULT_MODEL)
    vector_name = sanitize_vector_name(model_name)

    if recreate:
        try:
            client.delete_collection(collection)
            print(f"Deleted existing collection: {collection}")
        except Exception:
            pass

    # Check if exists
    try:
        info = client.get_collection(collection)
        print(f"Collection {collection} exists with {info.points_count} points")
        return
    except Exception:
        pass

    # Create with named vectors (dense + lexical)
    vectors_config = {
        vector_name: models.VectorParams(size=dim, distance=models.Distance.COSINE),
        LEX_VECTOR_NAME: models.VectorParams(size=LEX_VECTOR_DIM, distance=models.Distance.COSINE),
    }
    refrag_on = os.environ.get("REFRAG_MODE", "").strip().lower() in {"1", "true", "yes", "on"}
    if refrag_on:
        vectors_config[MINI_VECTOR_NAME] = models.VectorParams(
            size=MINI_VEC_DIM, distance=models.Distance.COSINE
        )
    pattern_on = os.environ.get("PATTERN_VECTORS", "").strip().lower() in {"1", "true", "yes", "on"}
    if pattern_on:
        vectors_config[PATTERN_VECTOR_NAME] = models.VectorParams(
            size=PATTERN_VECTOR_DIM, distance=models.Distance.COSINE
        )

    sparse_cfg = None
    if LEX_SPARSE_MODE:
        sparse_cfg = {
            LEX_SPARSE_NAME: models.SparseVectorParams(
                index=models.SparseIndexParams(full_scan_threshold=5000)
            )
        }

    client.create_collection(
        collection_name=collection,
        vectors_config=vectors_config,
        sparse_vectors_config=sparse_cfg,
        hnsw_config=models.HnswConfigDiff(m=16, ef_construct=256),
    )
    print(f"Created collection: {collection} (dim={dim})")

    # Create payload indexes for filtering
    for field in ["code_id", "language", "source"]:
        try:
            client.create_payload_index(
                collection_name=collection,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        except Exception:
            pass


def generate_point_id(code_id: str) -> str:
    """Generate stable Qdrant point ID from code_id."""
    return hashlib.md5(code_id.encode()).hexdigest()


def create_lexical_vector(text: str, dim: int | None = None) -> List[float]:
    """Create lexical hash vector for hybrid search."""
    try:
        return _lex_hash_vector_text(text, dim)
    except Exception:
        # Fallback: simple hash-based vector
        import re
        dim = int(dim or 256)
        vec = [0.0] * dim
        tokens = re.findall(r'\w+', text.lower())
        for t in tokens:
            h = int(hashlib.md5(t.encode()).hexdigest()[:8], 16)
            vec[h % dim] += 1.0
        norm = (sum(v*v for v in vec) ** 0.5) or 1.0
        return [v / norm for v in vec]


def index_corpus(
    corpus_entries: List[Dict[str, Any]],
    collection: str = DEFAULT_COLLECTION,
    batch_size: int = DEFAULT_BATCH_SIZE,
    recreate: bool = False,
    resume: bool = True,
    progress_callback: Optional[callable] = None,
    force: bool = False,
) -> Dict[str, Any]:
    """Index CoSQA corpus into Qdrant.

    Supports smart reuse: skips indexing if the collection already contains
    the same corpus with matching config (detected via fingerprinting).

    Args:
        corpus_entries: List of corpus entries from dataset.get_corpus_for_indexing()
        collection: Qdrant collection name
        batch_size: Batch size for upsert operations
        recreate: Whether to recreate the collection (deprecated, use force=True)
        resume: Whether to skip already-indexed entries (queries Qdrant)
        progress_callback: Optional callback(indexed, total) for progress updates
        force: Force reindexing even if collection matches corpus

    Returns:
        Stats dict with indexed count, time, errors, reused
    """
    # force implies recreate (drop & full reindex)
    if force:
        recreate = True

    # Compute fingerprint for this corpus upfront
    corpus_fingerprint = compute_corpus_fingerprint(corpus_entries)
    client = get_qdrant_client()

    # Check if we can reuse existing collection (fingerprint match)
    if not recreate:
        if collection_matches_corpus(collection, corpus_entries):
            print(f"Reusing existing collection: {collection} ({len(corpus_entries)} entries, fingerprint match)")
            return {
                "indexed": len(corpus_entries),
                "skipped": 0,
                "errors": 0,
                "reused": True,
            }
        
        # Check if collection exists with different fingerprint
        # Instead of recreating, we'll try to resume and fill in missing entries
        stored_fp = get_collection_fingerprint(collection)
        if stored_fp is not None and stored_fp != corpus_fingerprint:
            print(f"Fingerprint mismatch for {collection} (stored={stored_fp}, current={corpus_fingerprint})")
            print(f"  → Will resume and fill in missing entries (not recreating)")
            # Don't set recreate=True - let resume logic handle it
        elif stored_fp is None:
            # Legacy collection without fingerprint - check if it has points
            try:
                info = client.get_collection(collection)
                if info.points_count > 0:
                    print(f"Collection {collection} has no fingerprint but {info.points_count} points; recreating for AST-enriched index")
                    recreate = True
            except Exception:
                pass  # Collection doesn't exist, will be created fresh

    model = get_embedding_model()
    dim = get_model_dimension(model)
    # Use sanitized model name for vector name (matches hybrid_search/repo_search)
    from scripts.utils import sanitize_vector_name
    model_name = os.environ.get("EMBEDDING_MODEL", DEFAULT_MODEL)
    vector_name = sanitize_vector_name(model_name)

    # Create collection (recreate=True will drop existing)
    create_collection(client, collection, dim, recreate=recreate)

    # Clear cached vector names so search uses fresh collection schema
    try:
        from scripts.hybrid.qdrant import clear_ensured_collections
        clear_ensured_collections()
    except ImportError:
        pass

    allowed_vectors, allowed_sparse = get_collection_vector_names(client, collection)
    allow_lex = allowed_vectors is None or LEX_VECTOR_NAME in allowed_vectors
    allow_mini = allowed_vectors is None or MINI_VECTOR_NAME in allowed_vectors
    allow_pattern = allowed_vectors is None or PATTERN_VECTOR_NAME in allowed_vectors
    allow_sparse = allowed_sparse is None or LEX_SPARSE_NAME in allowed_sparse

    refrag_on = os.environ.get("REFRAG_MODE", "").strip().lower() in {"1", "true", "yes", "on"}
    pattern_on = os.environ.get("PATTERN_VECTORS", "").strip().lower() in {"1", "true", "yes", "on"}
    use_mini = refrag_on and allow_mini
    use_pattern = pattern_on and allow_pattern
    use_sparse = LEX_SPARSE_MODE and allow_sparse

    if refrag_on and not allow_mini:
        print(
            f"[COLLECTION_WARNING] Collection {collection} lacks mini vector "
            f"'{MINI_VECTOR_NAME}'. ReFRAG vectors will be skipped."
        )
    if pattern_on and not allow_pattern:
        print(
            f"[COLLECTION_WARNING] Collection {collection} lacks pattern vector "
            f"'{PATTERN_VECTOR_NAME}'. Pattern vectors will be skipped."
        )
    if LEX_SPARSE_MODE and not allow_sparse:
        print(
            f"[COLLECTION_WARNING] Collection {collection} lacks sparse vector "
            f"'{LEX_SPARSE_NAME}'. Sparse vectors will be skipped."
        )

    use_micro = os.environ.get("INDEX_MICRO_CHUNKS", "0").strip().lower() in {"1", "true", "yes", "on"}
    use_semantic = os.environ.get("INDEX_SEMANTIC_CHUNKS", "1").strip().lower() in {"1", "true", "yes", "on"}
    chunk_lines = int(os.environ.get("INDEX_CHUNK_LINES", "120") or 120)
    chunk_overlap = int(os.environ.get("INDEX_CHUNK_OVERLAP", "20") or 20)

    def _entry_path(entry: Dict[str, Any]) -> str:
        meta = entry.get("metadata") or {}
        return meta.get("path") or f"cosqa/{entry.get('code_id', 'unknown')}.py"

    def _chunk_entry(entry: Dict[str, Any]) -> List[Dict[str, Any]]:
        text = entry.get("text", "") or ""
        if not text:
            return []
        language = entry.get("language") or "python"
        if use_micro:
            try:
                cap = int(os.environ.get("MAX_MICRO_CHUNKS_PER_FILE", "200") or 200)
                base_tokens = int(os.environ.get("MICRO_CHUNK_TOKENS", "128") or 128)
                base_stride = int(os.environ.get("MICRO_CHUNK_STRIDE", "64") or 64)
                chunks = chunk_by_tokens(text, k_tokens=base_tokens, stride_tokens=base_stride)
                if cap > 0 and len(chunks) > cap:
                    scale = (len(chunks) / cap) * 1.1
                    new_tokens = max(base_tokens, int(base_tokens * scale))
                    new_stride = max(base_stride, int(base_stride * scale))
                    chunks = chunk_by_tokens(text, k_tokens=new_tokens, stride_tokens=new_stride)
            except Exception:
                chunks = chunk_by_tokens(text)
        elif use_semantic:
            chunks = chunk_semantic(text, language, chunk_lines, chunk_overlap)
        else:
            chunks = chunk_lines(text, chunk_lines, chunk_overlap)
        if not chunks:
            return [{"text": text, "start": 1, "end": 1}]
        return chunks

    # Build set of already-indexed point IDs by querying Qdrant (skip if recreating)
    indexed_ids: set[str] = set()
    if resume and not recreate:
        try:
            next_offset = None
            page = 0
            while True:
                points, next_offset = client.scroll(
                    collection_name=collection,
                    limit=1000,
                    offset=next_offset,
                    with_payload=["pid_str"],
                )
                if not points:
                    break
                for pt in points:
                    if pt.payload and pt.payload.get("pid_str"):
                        indexed_ids.add(str(pt.payload["pid_str"]))
                page += 1
                if next_offset is None:
                    break
            if indexed_ids:
                print(f"Found {len(indexed_ids)} already-indexed chunks in Qdrant (scanned {page} pages)")
        except Exception as e:
            print(f"Warning: Could not query existing entries: {e}")

    to_index: List[Dict[str, Any]] = []
    skipped = 0
    for entry in corpus_entries:
        chunks = _chunk_entry(entry)
        if not chunks:
            continue
        if indexed_ids:
            path = _entry_path(entry)
            if all(str(hash_id(ch["text"], path, ch["start"], ch["end"])) in indexed_ids for ch in chunks):
                skipped += 1
                continue
        to_index.append({"entry": entry, "chunks": chunks})

    if not to_index:
        print(f"All {len(corpus_entries)} entries already indexed")
        return {"indexed": 0, "skipped": len(corpus_entries), "errors": 0, "reused": True}

    print(f"Indexing {len(to_index)} entries (skipping {skipped} already indexed)")

    stats = {"indexed": 0, "indexed_entries": 0, "skipped": skipped, "errors": 0, "batches": 0}
    start_time = time.time()
    total = len(to_index)
    last_pct = 0

    # Split into batches
    batches = [to_index[i:i + batch_size] for i in range(0, len(to_index), batch_size)]

    # Parallel upsert workers (embedding is sequential, upsert is parallel)
    N_WORKERS = 10

    def upsert_points(points_list, worker_client):
        """Upsert pre-built points to Qdrant."""
        worker_client.upsert(collection_name=collection, points=points_list)
        return len(points_list)

    n_chunks = (len(batches) + N_WORKERS - 1) // N_WORKERS
    print(f"  Processing {len(batches)} batches in {n_chunks} chunks of {N_WORKERS}...")

    # Process in chunks: embed sequentially (not thread-safe), upsert in parallel
    processed_entries = 0
    for chunk_idx, chunk_start in enumerate(range(0, len(batches), N_WORKERS)):
        chunk = batches[chunk_start:chunk_start + N_WORKERS]
        chunk_points = []

        # Embed each batch sequentially (fastembed not thread-safe)
        for batch_idx, batch in enumerate(chunk):
            total_chunks = sum(len(item["chunks"]) for item in batch)
            print(
                f"  Chunk {chunk_idx+1}/{n_chunks} batch {batch_idx+1}/{len(chunk)}: "
                f"embedding {total_chunks} chunks...",
                end=" ",
                flush=True,
            )

            chunk_records = []
            for item in batch:
                entry = item["entry"]
                chunks = item["chunks"]
                code = entry.get("text", "") or ""
                language = entry.get("language") or "python"

                symbols, imports, calls = [], [], []
                if _AST_AVAILABLE and code:
                    try:
                        symbols = _extract_symbols(language, code) or []
                        imports, calls = _get_imports_calls(language, code) or ([], [])
                    except Exception:
                        pass

                symbol_names = [s.get("name", "") for s in symbols if s.get("name")]
                import_names = [
                    getattr(imp, "module", str(imp)) if hasattr(imp, "module") else str(imp)
                    for imp in imports[:10]
                ]
                call_names = [
                    getattr(c, "name", str(c)) if hasattr(c, "name") else str(c)
                    for c in calls[:10]
                ]

                path = _entry_path(entry)
                for ch_idx, ch in enumerate(chunks):
                    first_line = ch["text"].splitlines()[0] if ch.get("text") else ""
                    info = build_information(language, Path(path), ch["start"], ch["end"], first_line)
                    kind, sym, sym_path = _choose_symbol_for_chunk(ch["start"], ch["end"], symbols)
                    if ch.get("kind"):
                        kind = ch.get("kind") or kind
                    if ch.get("symbol"):
                        sym = ch.get("symbol") or sym
                    if ch.get("symbol_path"):
                        sym_path = ch.get("symbol_path") or sym_path
                    if not sym:
                        sym = entry.get("func_name") or entry.get("code_id") or ""
                    if not sym_path:
                        sym_path = sym
                    if not kind:
                        kind = entry.get("kind") or "function"

                    pid = hash_id(ch["text"], path, ch["start"], ch["end"])
                    lex_text = ch.get("text") or ""
                    entry_with_fp = {
                        **entry,
                        CORPUS_FINGERPRINT_KEY: corpus_fingerprint,
                        "symbols": symbol_names[:20],
                        "imports": import_names,
                        "calls": call_names,
                        "document": info,
                        "information": info,
                        "pid_str": str(pid),
                    }
                    meta = dict(entry_with_fp.get("metadata") or {})
                    meta.update({
                        "path": path,
                        "path_prefix": str(Path(path).parent),
                        "language": language,
                        "kind": kind,
                        "symbol": sym,
                        "symbol_path": sym_path,
                        "start_line": ch["start"],
                        "end_line": ch["end"],
                        "text": lex_text,
                        "code": lex_text,
                        "imports": import_names,
                        "calls": call_names,
                        "source": entry.get("source", "cosqa"),
                        "chunk_idx": ch_idx,
                    })
                    entry_with_fp["metadata"] = meta

                    chunk_records.append({
                        "id": pid,
                        "info": info,
                        "lex_text": lex_text,
                        "payload": entry_with_fp,
                        "code_text": lex_text,
                        "language": language,
                    })

            embeddings = embed_batch(model, [r["info"] for r in chunk_records])
            print("done", flush=True)

            points = []
            for rec, emb in zip(chunk_records, embeddings):
                dense_vec = emb
                vecs = {vector_name: dense_vec}
                if allow_lex:
                    vecs[LEX_VECTOR_NAME] = create_lexical_vector(rec["lex_text"], LEX_VECTOR_DIM)
                if use_mini:
                    try:
                        vecs[MINI_VECTOR_NAME] = project_mini(list(dense_vec), MINI_VEC_DIM)
                    except Exception:
                        pass
                if use_pattern:
                    try:
                        pv = extract_pattern_vector(rec["code_text"], rec["language"])
                        if pv:
                            vecs[PATTERN_VECTOR_NAME] = pv
                    except Exception:
                        pass
                if use_sparse and rec["lex_text"]:
                    try:
                        sparse_vec = _lex_sparse_vector_text(rec["lex_text"])
                        if sparse_vec.get("indices"):
                            vecs[LEX_SPARSE_NAME] = models.SparseVector(**sparse_vec)
                    except Exception:
                        pass

                points.append(models.PointStruct(
                    id=rec["id"],
                    vector=vecs,
                    payload=rec["payload"],
                ))
            chunk_points.append(points)

        # Upsert in parallel (Qdrant client is thread-safe with separate instances)
        with ThreadPoolExecutor(max_workers=len(chunk_points)) as executor:
            worker_clients = [get_qdrant_client() for _ in chunk_points]
            futures = {
                executor.submit(upsert_points, pts, wc): len(pts)
                for pts, wc in zip(chunk_points, worker_clients)
            }
            for future in as_completed(futures):
                batch_size = futures[future]
                try:
                    count = future.result()
                    stats["indexed"] += count
                    stats["batches"] += 1
                except Exception as e:
                    print(f"  Batch error (size={batch_size}): {e}")
                    stats["errors"] += batch_size

        # Progress after each chunk (always log)
        processed_entries += sum(len(b) for b in chunk)
        stats["indexed_entries"] = processed_entries
        elapsed = time.time() - start_time
        rate = stats["indexed"] / elapsed if elapsed > 0 else 0
        pct = int(100 * processed_entries / total) if total else 100
        print(
            f"  Chunk {chunk_idx+1}/{n_chunks} done: {processed_entries}/{total} entries, "
            f"{stats['indexed']} points ({pct}%, {rate:.1f}/s)"
        )
        last_pct = pct

    elapsed = time.time() - start_time
    stats["elapsed_seconds"] = round(elapsed, 2)
    stats["rate_per_second"] = round(stats["indexed"] / elapsed, 2) if elapsed > 0 else 0
    stats["reused"] = False

    print(
        f"\nIndexing complete: {stats['indexed_entries']} entries "
        f"({stats['indexed']} points) in {elapsed:.1f}s"
    )
    return stats


def verify_collection(collection: str = DEFAULT_COLLECTION) -> Dict[str, Any]:
    """Verify the indexed collection."""
    client = get_qdrant_client()
    try:
        info = client.get_collection(collection)
        return {
            "exists": True,
            "points_count": info.points_count,
            "status": str(info.status),
        }
    except Exception as e:
        return {"exists": False, "error": str(e)}
