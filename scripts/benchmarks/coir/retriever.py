#!/usr/bin/env python3
"""
Context-Engine Retriever Adapter for CoIR Benchmark.

Wraps our embedding and search pipeline as a CoIR-compatible retriever
for standardized evaluation.

**Uses these .env settings:**
- QDRANT_URL: Qdrant server URL (default: http://localhost:6333)
- EMBEDDING_MODEL: Dense embedding model (default: BAAI/bge-base-en-v1.5)
- VECTOR_NAME: Dense vector field name (default: dense)
- LEXICAL_VECTOR_NAME: Lexical vector field name (default: lexical)
- RERANKER_ENABLED: Enable ONNX reranker (default: true)
- RERANKER_MODEL: Reranker model path/name
- DENSE_DIM / LEXICAL_DIM: Vector dimensions

CoIR expects retrievers to implement:
- encode_queries(queries: List[str]) -> np.ndarray
- encode_corpus(corpus: List[Dict]) -> np.ndarray
- Or: search(corpus, queries, top_k) -> Dict[str, Dict[str, float]]

We implement the search() method to use our full hybrid+rerank pipeline.
"""
from __future__ import annotations

import asyncio
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Shared utilities
from scripts.benchmarks.qdrant_utils import probe_pseudo_tags, verify_config_compatibility, get_qdrant_client

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Read .env settings
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
RERANKER_ENABLED = os.environ.get("RERANKER_ENABLED", "true").lower() in ("true", "1", "yes")


class ContextEngineRetriever:
    """
    CoIR-compatible retriever wrapping Context-Engine's embedding pipeline.

    Can operate in two modes:
    1. Embedding mode: encode queries/corpus separately (for CoIR's internal scoring)
    2. Search mode: use our hybrid search directly (more accurate, uses reranker)
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        use_hybrid_search: bool = True,
        rerank_enabled: bool = True,
        batch_size: int = 32,
        **kwargs,
    ):
        """
        Initialize the retriever.

        Args:
            model_name: Embedding model name (default: from env or BGE-base)
            use_hybrid_search: Use our hybrid search instead of pure embedding
            rerank_enabled: Enable reranker when using hybrid search
            batch_size: Batch size for embedding
        
        Note: Collection naming is automatic based on corpus fingerprint.
        Each unique corpus+config gets its own named collection for reuse.
        """
        self.model_name = model_name or os.environ.get("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
        self.use_hybrid_search = use_hybrid_search
        self.rerank_enabled = rerank_enabled
        self.batch_size = batch_size
        self._model = None
        self._corpus_index = {}  # doc_id -> embedding
        self._corpus_doc_ids: set = set()  # Track which doc IDs are in current corpus
        self._indexed_collections: set = set()  # Track all collections we've created

    def reset_corpus_cache(self) -> None:
        """Reset the corpus embedding cache.

        Call this before processing a new corpus to avoid stale embeddings
        or KeyError from doc ID mismatches.
        """
        self._corpus_index = {}
        self._corpus_doc_ids = set()

    def _get_model(self):
        """Lazy load embedding model."""
        if self._model is None:
            try:
                from scripts.embedder import get_embedding_model
                self._model = get_embedding_model(self.model_name)
            except ImportError:
                from fastembed import TextEmbedding
                self._model = TextEmbedding(model_name=self.model_name)
        return self._model

    def encode_queries(
        self,
        queries: List[str],
        batch_size: Optional[int] = None,
        **kwargs,
    ) -> np.ndarray:
        """Encode queries to embeddings."""
        model = self._get_model()
        bs = batch_size or self.batch_size

        all_embeddings = []
        for i in range(0, len(queries), bs):
            batch = queries[i:i + bs]
            embeddings = list(model.embed(batch))
            all_embeddings.extend(embeddings)

        return np.array(all_embeddings)

    def encode_corpus(
        self,
        corpus: List[Dict[str, Any]],
        batch_size: Optional[int] = None,
        **kwargs,
    ) -> np.ndarray:
        """Encode corpus documents to embeddings.

        CoIR corpus format: [{"_id": str, "text": str, "title": str}, ...]
        """
        model = self._get_model()
        bs = batch_size or self.batch_size

        # Extract text (combine title + text if available)
        texts = []
        for doc in corpus:
            title = doc.get("title", "")
            text = doc.get("text", "")
            combined = f"{title}\n{text}" if title else text
            texts.append(combined)

        all_embeddings = []
        for i in range(0, len(texts), bs):
            batch = texts[i:i + bs]
            embeddings = list(model.embed(batch))
            all_embeddings.extend(embeddings)

        # Cache for search (track which doc IDs belong to this corpus)
        # Note: coir-eval may use "_id" or "id" depending on the code path
        self._corpus_doc_ids = set()
        for idx, doc in enumerate(corpus):
            doc_id = doc.get("_id") or doc.get("id") or str(idx)
            self._corpus_index[doc_id] = all_embeddings[idx]
            self._corpus_doc_ids.add(doc_id)

        return np.array(all_embeddings)

    def search(
        self,
        corpus: Dict[str, Dict[str, Any]],
        queries: Dict[str, str],
        top_k: int,
        **kwargs,
    ) -> Dict[str, Dict[str, float]]:
        """
        Search corpus for each query, return ranked results.

        This method uses our hybrid search for better quality when available.

        Args:
            corpus: {doc_id: {"text": str, "title": str}, ...}
            queries: {query_id: query_text, ...}
            top_k: Number of results per query

        Returns:
            {query_id: {doc_id: score, ...}, ...}
        """
        if self.use_hybrid_search:
            return self._search_hybrid(corpus, queries, top_k)
        else:
            return self._search_embedding(corpus, queries, top_k)

    def _search_embedding(
        self,
        corpus: Dict[str, Dict[str, Any]],
        queries: Dict[str, str],
        top_k: int,
    ) -> Dict[str, Dict[str, float]]:
        """Pure embedding-based search (cosine similarity)."""
        # Check if corpus has changed (different doc IDs = different corpus)
        current_doc_ids = set(corpus.keys())
        corpus_changed = current_doc_ids != self._corpus_doc_ids

        # Encode corpus if not cached or corpus changed
        if not self._corpus_index or corpus_changed:
            if corpus_changed and self._corpus_index:
                # Clear stale embeddings before encoding new corpus
                self.reset_corpus_cache()
            corpus_list = [{"_id": k, **v} for k, v in corpus.items()]
            self.encode_corpus(corpus_list)

        # Encode queries
        query_ids = list(queries.keys())
        query_texts = [queries[qid] for qid in query_ids]
        query_embeddings = self.encode_queries(query_texts)

        # Build corpus matrix
        doc_ids = list(corpus.keys())
        corpus_embeddings = np.array([self._corpus_index[did] for did in doc_ids])

        # Compute similarities
        results = {}
        for qidx, qid in enumerate(query_ids):
            q_emb = query_embeddings[qidx]
            # Cosine similarity (embeddings are normalized)
            scores = np.dot(corpus_embeddings, q_emb)

            # Get top-k
            top_indices = np.argsort(scores)[::-1][:top_k]
            results[qid] = {doc_ids[idx]: float(scores[idx]) for idx in top_indices}

        return results

    def _search_hybrid(
        self,
        corpus: Dict[str, Dict[str, Any]],
        queries: Dict[str, str],
        top_k: int,
    ) -> Dict[str, Dict[str, float]]:
        """
        Hybrid search using our full pipeline.

        For CoIR benchmarks, we need to index the corpus first, then search.
        This is slower but more representative of real performance.
        """
        # For CoIR tasks, the corpus is provided per-task.
        # Use a background thread if we're already inside a running event loop.
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self._search_hybrid_async(corpus, queries, top_k))

        def _run():
            return asyncio.run(self._search_hybrid_async(corpus, queries, top_k))

        with ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(_run).result()

    async def _search_hybrid_async(
        self,
        corpus: Dict[str, Dict[str, Any]],
        queries: Dict[str, str],
        top_k: int,
    ) -> Dict[str, Dict[str, float]]:
        """Async hybrid search using Context-Engine's full pipeline.

        NO FALLBACKS - benchmarks Context-Engine or fails.
        
        Supports collection reuse: each unique corpus+config gets its own
        named collection (via fingerprinting). If collection exists and matches,
        indexing is skipped for faster runs.
        """
        from scripts.benchmarks.coir.indexer import index_coir_corpus, get_corpus_collection
        from scripts.mcp_indexer_server import repo_search  # NO FALLBACK

        # Get corpus-specific collection name (based on fingerprint)
        corpus_list = [{"_id": k, **v} for k, v in corpus.items()]
        collection = get_corpus_collection(corpus_list)
        self._indexed_collections.add(collection)
        
        # index_coir_corpus checks fingerprint and skips if unchanged
        index_result = await asyncio.to_thread(index_coir_corpus, corpus_list, collection)
        if index_result.get("reused"):
            pass  # Collection reused, no indexing needed

        # Verify config compatibility (Fail-Fast)
        # Checks that the collection we are about to search matches our current env config
        try:
            # Create a throwaway client if needed, or use a shared one if available. 
            # get_qdrant_client creates a new one, which is fine for a one-off check.
            verify_config_compatibility(get_qdrant_client(), collection)
        except Exception as e:
            print(f"  [error] Configuration mismatch for {collection}: {e}")
            raise

        # Probe collection for pseudo/tags presence (validation)
        try:
            await asyncio.to_thread(probe_pseudo_tags, collection)
        except Exception as e:
            print(f"  [warn] Probe failed: {e}")

        # Search queries in parallel batches using Context-Engine
        # Use larger candidate pool for reranking (100 candidates -> top_k)
        query_items = list(queries.items())
        batch_size = self.batch_size or 32  # Parallel batch size
        results = {}

        async def search_one(qid: str, query_text: str) -> tuple:
            result = await repo_search(
                query=query_text,
                limit=top_k,
                per_path=1,
                collection=collection,
                rerank_enabled=self.rerank_enabled,
                rerank_top_n=100 if self.rerank_enabled else None,
                rerank_return_m=top_k if self.rerank_enabled else None,
            )
            # Extract scores
            doc_scores = {}
            for r in result.get("results", []):
                doc_id = r.get("doc_id") or r.get("code_id") or r.get("_id") or (r.get("payload") or {}).get("_id")
                score = r.get("score", 0.0)
                if doc_id:
                    score_val = float(score)
                    prev = doc_scores.get(doc_id)
                    doc_scores[doc_id] = score_val if prev is None else max(prev, score_val)
            return qid, doc_scores

        # Process in batches
        total_queries = len(query_items)
        for batch_start in range(0, total_queries, batch_size):
            batch_end = min(batch_start + batch_size, total_queries)
            batch = query_items[batch_start:batch_end]

            # Run batch in parallel
            tasks = [search_one(qid, qtext) for qid, qtext in batch]
            batch_results = await asyncio.gather(*tasks)

            for qid, doc_scores in batch_results:
                results[qid] = doc_scores

            # Progress update
            print(f"[coir] Searched {batch_end}/{total_queries} queries", flush=True)

        return results
    
    def cleanup(self, delete_collections: bool = False) -> int:
        """Explicitly cleanup collections created by this retriever.
        
        Args:
            delete_collections: If True, delete all indexed collections.
                               If False (default), just clear tracking (collections remain for reuse).
        
        Returns:
            Number of collections deleted (0 if delete_collections=False)
        """
        deleted = 0
        if delete_collections and self._indexed_collections:
            try:
                from scripts.benchmarks.coir.indexer import get_qdrant_client
                client = get_qdrant_client()
                for coll in self._indexed_collections:
                    try:
                        client.delete_collection(coll)
                        deleted += 1
                    except Exception:
                        pass
            except Exception:
                pass
        self._indexed_collections.clear()
        return deleted


class ContextEngineRetrieverDense(ContextEngineRetriever):
    """Dense-only retriever (no hybrid, no rerank) for ablation studies."""

    def __init__(self, **kwargs):
        super().__init__(use_hybrid_search=False, rerank_enabled=False, **kwargs)


class ContextEngineRetrieverHybrid(ContextEngineRetriever):
    """Hybrid retriever without reranker for ablation studies."""

    def __init__(self, **kwargs):
        super().__init__(use_hybrid_search=True, rerank_enabled=False, **kwargs)


class ContextEngineRetrieverFull(ContextEngineRetriever):
    """Full pipeline: hybrid + reranker."""

    def __init__(self, **kwargs):
        super().__init__(use_hybrid_search=True, rerank_enabled=True, **kwargs)
