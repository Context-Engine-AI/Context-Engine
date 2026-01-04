#!/usr/bin/env python3
"""
CoIR Corpus Indexer - Uses Context-Engine's real indexing pipeline.

Indexes CoIR benchmark corpora into Qdrant using the same pipeline as
production code indexing (hybrid vectors, same models, same settings).

This ensures benchmark results reflect actual system performance.

Features:
- Corpus fingerprinting for smart reuse (avoids reindexing unchanged corpora)
- Consistent collection naming per task
- Optional cleanup of benchmark collections
"""
from __future__ import annotations

import hashlib
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from qdrant_client import QdrantClient, models

# Reuse shared Qdrant client + embedder helpers from cosqa (handles API key, timeout)
from scripts.benchmarks.cosqa.indexer import (
    get_qdrant_client,
    get_embedding_model,
    get_model_dimension,
)
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

from scripts.embedder import DEFAULT_MODEL

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", DEFAULT_MODEL)

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

# Collection prefix for CoIR benchmarks
COIR_COLLECTION_PREFIX = "coir-bench-"

# Metadata key for corpus fingerprint (stored in collection)
CORPUS_FINGERPRINT_KEY = "_corpus_fingerprint"


def get_temp_collection(task_name: str = "default") -> str:
    """Get collection name for a CoIR task."""
    return f"{COIR_COLLECTION_PREFIX}{task_name}"


def get_corpus_collection(corpus: List[Dict[str, Any]]) -> str:
    """Get a stable collection name for a specific corpus.
    
    Uses corpus fingerprint to ensure each unique corpus+config gets its own
    collection. This enables automatic reuse across benchmark runs.
    """
    fp = compute_corpus_fingerprint(corpus)
    return f"{COIR_COLLECTION_PREFIX}{fp}"


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
        f"ast_enriched:v2",
        "text_dense_mode:chunk_text_v1",
    ]
    return hashlib.sha256("|".join(config_parts).encode()).hexdigest()[:8]


def compute_corpus_fingerprint(corpus: List[Dict[str, Any]]) -> str:
    """Compute a stable fingerprint for a corpus + config to detect changes.
    
    Includes:
    - Sorted doc IDs + all content that affects embeddings (title + text)
    - Embedding model and vector config (invalidates on model/config change)
    - ReFRAG/chunking settings (invalidates on processing change)
    """
    hasher = hashlib.sha256()
    
    # Config fingerprint first - ensures reindex on config change
    config_fp = _get_config_fingerprint()
    hasher.update(f"config:{config_fp}".encode())
    
    # Corpus content fingerprint - include all fields that affect embeddings
    for doc in sorted(corpus, key=lambda d: d.get("_id", "")):
        doc_id = doc.get("_id", "")
        title = doc.get("title", "")
        text = doc.get("text", "")
        language = doc.get("language", "")
        # Hash the same content that gets embedded (title + text)
        # Use full content for accurate fingerprinting (truncation caused stale reuse)
        combined = f"{language}\n{title}\n{text}"
        hasher.update(f"{doc_id}:{combined}".encode("utf-8", errors="ignore"))
    # Note: We intentionally don't include count in fingerprint
    # This allows resume to work when some entries fail to index
    
    return hasher.hexdigest()[:16]


def get_collection_fingerprint(collection: str) -> Optional[str]:
    """Get the stored corpus fingerprint from a collection, if any."""
    try:
        client = get_qdrant_client()
        # Scroll for a single point with the fingerprint payload
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


def collection_matches_corpus(
    collection: str,
    corpus: List[Dict[str, Any]],
    corpus_fingerprint: Optional[str] = None,
) -> bool:
    """Check if an existing collection matches the given corpus.

    Note: We intentionally don't require exact point count match.
    If fingerprint matches, the resume logic will fill in any missing entries.
    """
    try:
        client = get_qdrant_client()
        info = client.get_collection(collection)
        if info.points_count == 0:
            return False  # Empty collection, needs indexing
        # Fingerprint check
        stored_fp = get_collection_fingerprint(collection)
        if stored_fp:
            current_fp = corpus_fingerprint or compute_corpus_fingerprint(corpus)
            return stored_fp == current_fp
        return False
    except Exception:
        return False


def cleanup_coir_collections(task_names: Optional[List[str]] = None) -> int:
    """Delete CoIR benchmark collections.
    
    Args:
        task_names: Specific tasks to clean up, or None for all coir-bench-* collections
    
    Returns:
        Number of collections deleted
    """
    client = get_qdrant_client()
    deleted = 0
    try:
        collections = client.get_collections().collections
        for col in collections:
            name = getattr(col, "name", "")
            if not name.startswith(COIR_COLLECTION_PREFIX):
                continue
            if task_names is not None:
                task = name[len(COIR_COLLECTION_PREFIX):]
                if task not in task_names:
                    continue
            try:
                client.delete_collection(name)
                deleted += 1
            except Exception:
                pass
    except Exception:
        pass
    return deleted


def create_collection(
    client: QdrantClient,
    collection: str,
    dim: int,
    recreate: bool = False,
) -> None:
    """Create or recreate the CoIR collection with proper vector config."""
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
    for field in ["_id", "language", "source", "doc_type"]:
        try:
            client.create_payload_index(
                collection_name=collection,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        except Exception:
            pass


def create_lexical_vector(text: str, dim: int | None = None) -> List[float]:
    """Create lexical hash vector for hybrid search."""
    try:
        return _lex_hash_vector_text(text, dim)
    except Exception:
        # Fallback: simple hash-based vector
        import re
        dim = int(dim or 256)
        vec = [0.0] * dim
        tokens = re.findall(r"\w+", text.lower())
        for t in tokens:
            h = int(hashlib.md5(t.encode()).hexdigest()[:8], 16)
            vec[h % dim] += 1.0
        norm = (sum(v * v for v in vec) ** 0.5) or 1.0
        return [v / norm for v in vec]


def index_coir_corpus(
    corpus: List[Dict[str, Any]],
    collection: str,
    batch_size: int = 100,
    recreate: bool = False,
    show_progress: bool = True,
    force: bool = False,
) -> Dict[str, Any]:
    """
    Index a CoIR corpus using Context-Engine's production pipeline.
    
    Uses the same embedding model and vector configuration as production.
    Supports smart reuse: skips indexing if the collection already contains
    the same corpus (detected via fingerprinting).
    
    Args:
        corpus: List of {"_id": str, "text": str, "title": str (optional)}
        collection: Qdrant collection name
        batch_size: Batch size for indexing
        recreate: Drop and recreate collection (deprecated, use force=True)
        show_progress: Show progress bar
        force: Force reindexing even if collection matches corpus
    
    Returns:
        {"indexed": int, "collection": str, "time_s": float, "reused": bool}
    """
    # force implies recreate (drop & full reindex)
    if force:
        recreate = True

    # Compute fingerprint for this corpus (needed for both matching and storing)
    corpus_fingerprint = compute_corpus_fingerprint(corpus)
    client = get_qdrant_client()

    # Check if we can reuse existing collection
    if not recreate:
        if collection_matches_corpus(collection, corpus, corpus_fingerprint=corpus_fingerprint):
            if show_progress:
                print(f"  Reusing existing collection: {collection} ({len(corpus)} docs, fingerprint match)")
            return {
                "indexed": len(corpus),
                "collection": collection,
                "time_s": 0.0,
                "reused": True,
            }
        # Check if collection exists with different data (fingerprint mismatch)
        stored_fp = get_collection_fingerprint(collection)
        if stored_fp is not None and stored_fp != corpus_fingerprint:
            if show_progress:
                print(f"Fingerprint mismatch for {collection} (stored={stored_fp}, current={corpus_fingerprint})")
                print("  → Will resume and fill in missing entries (not recreating)")
        elif stored_fp is None:
            # Legacy collection without fingerprint - check if it has points
            try:
                existing = client.get_collection(collection)
                if existing.points_count > 0:
                    if show_progress:
                        print(
                            f"Collection {collection} has no fingerprint but {existing.points_count} points; "
                            "recreating for full-pipeline index"
                        )
                    recreate = True
            except Exception:
                pass

    model = get_embedding_model()
    dim = get_model_dimension(model)
    # Use sanitized model name for vector name (matches hybrid_search/repo_search)
    from scripts.utils import sanitize_vector_name
    model_name = os.environ.get("EMBEDDING_MODEL", DEFAULT_MODEL)
    vector_name = sanitize_vector_name(model_name)
    start_time = time.time()
    
    # Create collection (recreate=True will drop existing)
    create_collection(client, collection, dim, recreate=recreate)

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

    if show_progress and refrag_on and not allow_mini:
        print(
            f"[COLLECTION_WARNING] Collection {collection} lacks mini vector "
            f"'{MINI_VECTOR_NAME}'. ReFRAG vectors will be skipped."
        )
    if show_progress and pattern_on and not allow_pattern:
        print(
            f"[COLLECTION_WARNING] Collection {collection} lacks pattern vector "
            f"'{PATTERN_VECTOR_NAME}'. Pattern vectors will be skipped."
        )
    if show_progress and LEX_SPARSE_MODE and not allow_sparse:
        print(
            f"[COLLECTION_WARNING] Collection {collection} lacks sparse vector "
            f"'{LEX_SPARSE_NAME}'. Sparse vectors will be skipped."
        )

    use_micro = os.environ.get("INDEX_MICRO_CHUNKS", "0").strip().lower() in {"1", "true", "yes", "on"}
    use_semantic = os.environ.get("INDEX_SEMANTIC_CHUNKS", "1").strip().lower() in {"1", "true", "yes", "on"}
    chunk_lines_count = int(os.environ.get("INDEX_CHUNK_LINES", "120") or 120)
    chunk_overlap = int(os.environ.get("INDEX_CHUNK_OVERLAP", "20") or 20)

    def _doc_path(doc: Dict[str, Any]) -> str:
        meta = doc.get("metadata") or {}
        return meta.get("path") or f"coir/{collection}/{doc.get('_id', 'unknown')}"

    def _chunk_doc(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
        text = doc.get("text", "") or ""
        if not text:
            return []
        language = doc.get("language") or "text"
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
            chunks = chunk_semantic(text, language, chunk_lines_count, chunk_overlap)
        else:
            chunks = chunk_lines(text, chunk_lines_count, chunk_overlap)
        if not chunks:
            return [{"text": text, "start": 1, "end": 1}]
        return chunks

    # Build set of already-indexed point IDs by querying Qdrant (skip if recreating)
    indexed_ids: set[str] = set()
    if not recreate:
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
            if indexed_ids and show_progress:
                print(f"Found {len(indexed_ids)} already-indexed chunks in Qdrant (scanned {page} pages)")
        except Exception as e:
            if show_progress:
                print(f"Warning: Could not query existing entries: {e}")

    to_index: List[Dict[str, Any]] = []
    skipped = 0
    for doc in corpus:
        chunks = _chunk_doc(doc)
        if not chunks:
            continue
        if indexed_ids:
            path = _doc_path(doc)
            if all(str(hash_id(ch["text"], path, ch["start"], ch["end"])) in indexed_ids for ch in chunks):
                skipped += 1
                continue
        to_index.append({"doc": doc, "chunks": chunks})

    if not to_index:
        if show_progress:
            print(f"All {len(corpus)} entries already indexed")
        return {
            "indexed": 0,
            "indexed_entries": 0,
            "indexed_points": 0,
            "skipped": len(corpus),
            "errors": 0,
            "collection": collection,
            "time_s": 0.0,
            "reused": True,
        }

    if show_progress:
        print(f"Indexing {len(to_index)} entries (skipping {skipped} already indexed)")

    stats = {"indexed_entries": 0, "indexed_points": 0, "skipped": skipped, "errors": 0, "batches": 0}
    total = len(to_index)

    # Split into batches
    batches = [to_index[i:i + batch_size] for i in range(0, len(to_index), batch_size)]

    # Parallel upsert workers (embedding is sequential, upsert is parallel)
    N_WORKERS = 10

    def upsert_points(points_list, worker_client):
        """Upsert pre-built points to Qdrant."""
        worker_client.upsert(collection_name=collection, points=points_list)
        return len(points_list)

    n_chunks = (len(batches) + N_WORKERS - 1) // N_WORKERS
    if show_progress:
        print(f"  Processing {len(batches)} batches in {n_chunks} chunks of {N_WORKERS}...")

    # Process in chunks: embed sequentially (not thread-safe), upsert in parallel
    processed_entries = 0
    for chunk_idx, chunk_start in enumerate(range(0, len(batches), N_WORKERS)):
        chunk = batches[chunk_start:chunk_start + N_WORKERS]
        chunk_points = []

        # Embed each batch sequentially (fastembed not thread-safe)
        for batch_idx, batch in enumerate(chunk):
            total_chunks = sum(len(item["chunks"]) for item in batch)
            if show_progress:
                print(
                    f"  Chunk {chunk_idx+1}/{n_chunks} batch {batch_idx+1}/{len(chunk)}: "
                    f"embedding {total_chunks} chunks...",
                    end=" ",
                    flush=True,
                )

            chunk_records = []
            for item in batch:
                doc = item["doc"]
                chunks = item["chunks"]
                code = doc.get("text", "") or ""
                raw_language = doc.get("language")
                language = raw_language or "text"
                title = doc.get("title", "") or ""
                is_text_corpus = not raw_language or raw_language == "text"

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

                path = _doc_path(doc)
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
                        sym = title or doc.get("_id") or ""
                    if not sym_path:
                        sym_path = sym
                    if not kind:
                        kind = doc.get("kind") or "document"

                    pid = hash_id(ch["text"], path, ch["start"], ch["end"])
                    lex_text = ch.get("text") or ""
                    dense_text = lex_text if is_text_corpus and lex_text else info
                    entry_with_fp = dict(doc)
                    entry_with_fp.update(
                        {
                            CORPUS_FINGERPRINT_KEY: corpus_fingerprint,
                            "doc_type": "coir_corpus",
                            "document": info,
                            "information": info,
                            "pid_str": str(pid),
                            "symbols": symbol_names[:20],
                            "imports": import_names,
                            "calls": call_names,
                            "language": entry_with_fp.get("language") or language,
                        }
                    )
                    entry_with_fp.setdefault("source", doc.get("source", "coir"))
                    entry_with_fp.setdefault("title", title)
                    entry_with_fp.setdefault("text", doc.get("text", ""))

                    meta = dict(entry_with_fp.get("metadata") or {})
                    meta.update(
                        {
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
                            "source": entry_with_fp.get("source", "coir"),
                            "chunk_idx": ch_idx,
                        }
                    )
                    entry_with_fp["metadata"] = meta

                    chunk_records.append(
                        {
                            "id": pid,
                            "info": info,
                            "dense_text": dense_text,
                            "lex_text": lex_text,
                            "payload": entry_with_fp,
                            "code_text": lex_text,
                            "language": language,
                        }
                    )

            embeddings = embed_batch(model, [r["dense_text"] for r in chunk_records])
            if show_progress:
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

                points.append(
                    models.PointStruct(
                        id=rec["id"],
                        vector=vecs,
                        payload=rec["payload"],
                    )
                )
            chunk_points.append(points)

        # Upsert in parallel (Qdrant client is thread-safe with separate instances)
        with ThreadPoolExecutor(max_workers=len(chunk_points)) as executor:
            worker_clients = [get_qdrant_client() for _ in chunk_points]
            futures = {
                executor.submit(upsert_points, pts, wc): len(pts)
                for pts, wc in zip(chunk_points, worker_clients)
            }
            for future in as_completed(futures):
                pts_count = futures[future]
                try:
                    count = future.result()
                    stats["indexed_points"] += count
                    stats["batches"] += 1
                except Exception as e:
                    if show_progress:
                        print(f"  Batch error (size={pts_count}): {e}")
                    stats["errors"] += pts_count

        # Progress after each chunk (always log)
        processed_entries += sum(len(b) for b in chunk)
        stats["indexed_entries"] = processed_entries
        elapsed = time.time() - start_time
        rate = stats["indexed_points"] / elapsed if elapsed > 0 else 0
        pct = int(100 * processed_entries / total) if total else 100
        if show_progress:
            print(
                f"  Chunk {chunk_idx+1}/{n_chunks} done: {processed_entries}/{total} entries, "
                f"{stats['indexed_points']} points ({pct}%, {rate:.1f}/s)"
            )

    elapsed = time.time() - start_time
    stats["elapsed_seconds"] = round(elapsed, 2)
    stats["rate_per_second"] = round(stats["indexed_points"] / elapsed, 2) if elapsed > 0 else 0
    stats["reused"] = False
    stats["indexed"] = stats["indexed_entries"]
    stats["collection"] = collection
    stats["time_s"] = elapsed

    if show_progress:
        print(
            f"\nIndexing complete: {stats['indexed_entries']} entries "
            f"({stats['indexed_points']} points) in {elapsed:.1f}s"
        )

    return stats
