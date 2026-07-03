"""
RecursiveReranker, ONNXRecursiveReranker, SessionAwareReranker - Main reranking pipelines.

Implements TRM-style iterative refinement:
1. Initialize latent state z from query
2. For each iteration: score, refine z, check early stopping
3. Return final ranking
"""
import os
import threading
import time
from typing import Any, Dict, List, Optional

import numpy as np

# Safe ONNX imports
try:
    import onnxruntime as ort
    from tokenizers import Tokenizer
    HAS_ONNX = True
except ImportError:
    ort = None
    Tokenizer = None
    HAS_ONNX = False

# Use centralized reranker factory (supports FastEmbed + ONNX backends)
try:
    from scripts.reranker import (
        get_reranker_model as _get_reranker_model,
        rerank_pairs as _rerank_pairs,
        is_reranker_available as _is_reranker_available,
        RERANKER_MODEL,
    )
    HAS_RERANKER_FACTORY = True
except ImportError:
    HAS_RERANKER_FACTORY = False
    _get_reranker_model = None
    _rerank_pairs = None
    _is_reranker_available = None
    RERANKER_MODEL = None

# Legacy: direct FastEmbed imports (fallback when factory unavailable)
try:
    from fastembed.rerank.cross_encoder import TextCrossEncoder
    HAS_FASTEMBED_RERANK = True
except ImportError:
    TextCrossEncoder = None
    HAS_FASTEMBED_RERANK = False

from scripts.rerank_recursive.state import RefinementState
from scripts.rerank_recursive.scorer import TinyScorer
from scripts.rerank_recursive.refiner import LatentRefiner
from scripts.rerank_recursive.projection import LearnedProjection
from scripts.rerank_recursive.confidence import ConfidenceEstimator
from scripts.rerank_recursive.utils import (
    _compute_fname_boost,
    _get_cached_embedding,
    _cache_embedding,
)


class RecursiveReranker:
    """
    Main recursive reranking pipeline.

    Key insight: Multiple passes through tiny networks > one pass through large network
    """

    def __init__(
        self,
        n_iterations: int = 3,
        dim: int = 256,
        hidden_dim: int = 512,
        early_stop: bool = True,
        blend_with_initial: float = 0.3,
        alpha_scheduler: Optional[Any] = None,
    ):
        self.n_iterations = n_iterations
        self.dim = dim
        self.early_stop = early_stop
        self.blend_with_initial = blend_with_initial

        # Alpha scheduler: if None, use CosineAlphaScheduler by default
        if alpha_scheduler is None:
            from scripts.rerank_recursive.alpha_scheduler import CosineAlphaScheduler
            self.alpha_scheduler = CosineAlphaScheduler(n_iterations=n_iterations)
        else:
            self.alpha_scheduler = alpha_scheduler

        self.scorer = TinyScorer(dim=dim, hidden_dim=hidden_dim)
        self.refiner = LatentRefiner(dim=dim)

        from scripts.embedder import get_model_dimension
        embed_dim = get_model_dimension()
        self._learned_projection = LearnedProjection(input_dim=embed_dim, output_dim=dim, lr=0.0)
        self._embedder = None
        self._embedder_lock = threading.Lock()
        self._proj_cache: Dict[int, np.ndarray] = {}
        self._proj_cache_lock = threading.Lock()

        # Observability counters for learning status
        self._proj_learned_count = 0
        self._proj_fallback_count = 0
        self._fallback_warned = False

    def _get_embedder(self):
        if self._embedder is not None:
            return self._embedder
        with self._embedder_lock:
            if self._embedder is not None:
                return self._embedder
            try:
                from scripts.embedder import get_embedding_model
                model_name = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
                self._embedder = get_embedding_model(model_name)
            except Exception:
                self._embedder = None
            return self._embedder

    def _encode(self, texts: List[str]) -> np.ndarray:
        cached_results = []
        texts_to_encode = []
        text_indices = []

        for i, text in enumerate(texts):
            cached = _get_cached_embedding(text)
            if cached is not None:
                cached_results.append((i, cached))
            else:
                texts_to_encode.append(text)
                text_indices.append(i)

        new_embeddings = []
        if texts_to_encode:
            embedder = self._get_embedder()
            if embedder is not None:
                try:
                    embeddings = list(embedder.embed(texts_to_encode))
                    if len(embeddings) != len(texts_to_encode):
                        raise ValueError("Embedding count mismatch")
                    for text, emb in zip(texts_to_encode, embeddings):
                        emb_arr = np.array(emb, dtype=np.float32)
                        if emb_arr.shape[0] != self.dim:
                            emb_arr = self._project_to_dim(emb_arr.reshape(1, -1))[0]
                        _cache_embedding(text, emb_arr)
                        new_embeddings.append(emb_arr)
                except Exception:
                    new_embeddings = []

            if not new_embeddings:
                import hashlib
                fallback_dim = self.dim
                if cached_results:
                    fallback_dim = cached_results[0][1].shape[0]
                for text in texts_to_encode:
                    text_hash = hashlib.sha256(text.encode("utf-8", errors="replace")).digest()
                    seed = int.from_bytes(text_hash[:4], "big")
                    rng = np.random.RandomState(seed)
                    vec = rng.randn(fallback_dim).astype(np.float32)
                    vec = vec / (np.linalg.norm(vec) + 1e-8)
                    _cache_embedding(text, vec)
                    new_embeddings.append(vec)

        result = [None] * len(texts)
        for i, emb in cached_results:
            result[i] = emb
        for i, idx in enumerate(text_indices):
            result[idx] = new_embeddings[i]
        return np.array(result, dtype=np.float32)

    def _encode_raw(self, texts: List[str]) -> np.ndarray:
        from scripts.embedder import get_model_dimension
        fallback_dim = get_model_dimension()
        embedder = self._get_embedder()
        if embedder is None:
            import hashlib
            result = []
            for text in texts:
                text_hash = hashlib.sha256(text.encode("utf-8", errors="replace")).digest()
                seed = int.from_bytes(text_hash[:4], "big")
                rng = np.random.RandomState(seed)
                vec = rng.randn(fallback_dim).astype(np.float32)
                vec = vec / (np.linalg.norm(vec) + 1e-8)
                result.append(vec)
            return np.array(result, dtype=np.float32)
        try:
            embeddings = list(embedder.embed(texts))
            result = [np.array(emb, dtype=np.float32) for emb in embeddings]
            return np.array(result, dtype=np.float32)
        except Exception:
            import hashlib
            result = []
            for text in texts:
                text_hash = hashlib.sha256(text.encode("utf-8", errors="replace")).digest()
                seed = int.from_bytes(text_hash[:4], "big")
                rng = np.random.RandomState(seed)
                vec = rng.randn(fallback_dim).astype(np.float32)
                vec = vec / (np.linalg.norm(vec) + 1e-8)
                result.append(vec)
            return np.array(result, dtype=np.float32)

    def _project_to_dim(self, embeddings: np.ndarray) -> np.ndarray:
        if embeddings.shape[-1] == self.dim:
            return embeddings
        input_dim = embeddings.shape[-1]
        # Try to use learned projection if available (hot-reloads from worker)
        if hasattr(self, '_learned_projection') and self._learned_projection.input_dim == input_dim:
            # forward() calls maybe_reload_weights() internally
            if self._learned_projection._weights_loaded:
                self._proj_learned_count += 1
                return self._learned_projection.forward(embeddings)
            # Try one reload attempt if not yet loaded
            self._learned_projection.maybe_reload_weights()
            if self._learned_projection._weights_loaded:
                self._proj_learned_count += 1
                return self._learned_projection.forward(embeddings)
        # Fallback to random projection
        self._proj_fallback_count += 1
        if not self._fallback_warned and self._proj_fallback_count >= 10:
            from scripts.logger import get_logger
            get_logger(__name__).warning(
                f"LearnedProjection not loaded after {self._proj_fallback_count} calls; "
                f"using random projection (learning may be disabled or weights missing)"
            )
            self._fallback_warned = True
        with self._proj_cache_lock:
            if input_dim not in self._proj_cache:
                rng = np.random.RandomState(44)
                proj_matrix = rng.randn(input_dim, self.dim).astype(np.float32) * np.float32(0.01)
                self._proj_cache[input_dim] = proj_matrix
            proj_matrix = self._proj_cache[input_dim]
        projected = embeddings @ proj_matrix
        norms = np.linalg.norm(projected, axis=-1, keepdims=True) + 1e-8
        return projected / norms

    def get_learning_status(self) -> Dict[str, Any]:
        """Return observability info for learning components."""
        scorer_metrics = self.scorer.get_metrics()
        proj_loaded = (
            hasattr(self, '_learned_projection') and
            self._learned_projection._weights_loaded
        )
        return {
            "projection_loaded": proj_loaded,
            "projection_version": self._learned_projection._version if proj_loaded else 0,
            "projection_learned_calls": self._proj_learned_count,
            "projection_fallback_calls": self._proj_fallback_count,
            "scorer_version": scorer_metrics.get("version", 0),
            "scorer_converged": scorer_metrics.get("converged", False),
            "scorer_avg_loss": scorer_metrics.get("avg_loss", 0.0),
            "scorer_update_count": scorer_metrics.get("update_count", 0),
            "refiner_version": self.refiner._version,
        }

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        initial_scores: Optional[List[float]] = None,
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        confidence = ConfidenceEstimator()
        n_docs = len(candidates)

        doc_texts = []
        for c in candidates:
            text_parts = []
            if c.get("symbol"):
                text_parts.append(str(c["symbol"]))
            if c.get("path"):
                text_parts.append(str(c["path"]))
            code = c.get("code") or c.get("snippet") or c.get("text") or ""
            if code:
                text_parts.append(str(code)[:500])
            doc_texts.append(" ".join(text_parts) if text_parts else "empty")

        query_emb = self._encode([query])[0]
        doc_embs = self._encode(doc_texts)
        query_emb = self._project_to_dim(query_emb.reshape(1, -1))[0]
        doc_embs = self._project_to_dim(doc_embs)

        z = query_emb.copy()
        if initial_scores is not None:
            scores = np.array(initial_scores, dtype=np.float32)
        else:
            scores = np.zeros(n_docs, dtype=np.float32)

        state = RefinementState(z=z, scores=scores, iteration=0)
        state.score_history.append(scores.copy())
        alpha_trajectory = []  # Track alpha values used

        for i in range(self.n_iterations):
            state.iteration = i + 1
            new_scores = self.scorer.forward(query_emb, doc_embs, state.z)
            alpha = self.alpha_scheduler.get_alpha(i)
            alpha_trajectory.append(alpha)
            state.scores = alpha * new_scores + (1 - alpha) * state.scores
            state.score_history.append(state.scores.copy())
            state.z = self.refiner.refine(state.z, query_emb, doc_embs, state.scores)
            if self.early_stop and confidence.should_stop(state):
                break

        final_scores = state.scores
        if initial_scores is not None and self.blend_with_initial > 0:
            init_arr = np.array(initial_scores, dtype=np.float32)
            std = final_scores.std()
            if std > 1e-6:
                final_norm = (final_scores - final_scores.mean()) / std
            else:
                final_norm = final_scores - final_scores.mean()
            std = init_arr.std()
            if std > 1e-6:
                init_norm = (init_arr - init_arr.mean()) / std
            else:
                init_norm = init_arr - init_arr.mean()
            final_scores = (1 - self.blend_with_initial) * final_norm + self.blend_with_initial * init_norm

        ranked_indices = np.argsort(-final_scores)
        reranked = []
        fname_boost_factor = float(os.environ.get("FNAME_BOOST", "0.15") or 0.15)

        for rank, idx in enumerate(ranked_indices):
            candidate = candidates[idx].copy()
            candidate["recursive_score"] = float(final_scores[idx])
            candidate["recursive_rank"] = rank
            candidate["recursive_iterations"] = state.iteration
            trajectory = [float(h[idx]) for h in state.score_history]
            candidate["score_trajectory"] = trajectory
            candidate["alpha_trajectory"] = alpha_trajectory[:state.iteration]  # Alpha values used
            fname_boost = _compute_fname_boost(query, candidate, fname_boost_factor)
            candidate["score"] = float(final_scores[idx]) + fname_boost
            if fname_boost > 0:
                candidate["fname_boost"] = fname_boost
            reranked.append(candidate)

        if fname_boost_factor > 0 and any(c.get("fname_boost", 0) > 0 for c in reranked):
            reranked.sort(key=lambda x: -x["score"])

        return reranked


class ONNXRecursiveReranker(RecursiveReranker):
    """Recursive reranker using ONNX cross-encoder for scoring."""

    def __init__(
        self,
        n_iterations: int = 3,
        onnx_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        **kwargs
    ):
        super().__init__(n_iterations=n_iterations, **kwargs)
        self.onnx_path = onnx_path or os.environ.get("RERANKER_ONNX_PATH", "")
        self.tokenizer_path = tokenizer_path or os.environ.get("RERANKER_TOKENIZER_PATH", "")
        self._session = None
        self._tokenizer = None
        self._onnx_lock = threading.Lock()

    def _get_onnx_session(self):
        if self._session is not None:
            return self._session, self._tokenizer
        if not HAS_ONNX or not self.onnx_path or not self.tokenizer_path:
            return None, None
        with self._onnx_lock:
            if self._session is not None:
                return self._session, self._tokenizer
            try:
                tok = Tokenizer.from_file(self.tokenizer_path)
                try:
                    tok.enable_truncation(max_length=512)
                except Exception:
                    pass
                sess = ort.InferenceSession(self.onnx_path, providers=["CPUExecutionProvider"])
                self._session, self._tokenizer = sess, tok
            except Exception:
                self._session, self._tokenizer = None, None
            return self._session, self._tokenizer

    def _onnx_score(self, query: str, docs: List[str]) -> Optional[np.ndarray]:
        sess, tok = self._get_onnx_session()
        if sess is None or tok is None:
            return None
        try:
            pairs = [(query, doc) for doc in docs]
            enc = tok.encode_batch(pairs)
            input_ids = [e.ids for e in enc]
            attn = [e.attention_mask for e in enc]
            max_len = max((len(ids) for ids in input_ids), default=0)
            if max_len == 0:
                return None
            pad_id = 0
            try:
                pad_token_id = tok.token_to_id("[PAD]")
                if pad_token_id is not None:
                    pad_id = int(pad_token_id)
            except Exception:
                pad_id = 0

            def pad(seq, pad_val):
                return seq + [pad_val] * (max_len - len(seq))

            input_ids_padded = [pad(s, pad_id) for s in input_ids]
            attn_padded = [pad(s, 0) for s in attn]
            input_ids_arr = np.array(input_ids_padded, dtype=np.int64)
            attn_arr = np.array(attn_padded, dtype=np.int64)
            input_names = [i.name for i in sess.get_inputs()]
            feeds = {}
            if "input_ids" in input_names:
                feeds["input_ids"] = input_ids_arr
            if "attention_mask" in input_names:
                feeds["attention_mask"] = attn_arr
            if "token_type_ids" in input_names:
                token_type_arr = np.zeros((len(input_ids_padded), max_len), dtype=np.int64)
                feeds["token_type_ids"] = token_type_arr
            out = sess.run(None, feeds)
            logits = out[0]
            scores = []
            for row in logits:
                try:
                    if hasattr(row, "__len__") and len(row) >= 2:
                        scores.append(float(row[1]))
                    elif hasattr(row, "__len__") and len(row) == 1:
                        scores.append(float(row[0]))
                    else:
                        scores.append(float(row))
                except Exception:
                    scores.append(0.0)
            return np.array(scores, dtype=np.float32)
        except Exception:
            return None

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        initial_scores: Optional[List[float]] = None,
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        confidence = ConfidenceEstimator()

        doc_texts = []
        for c in candidates:
            parts = []
            if c.get("symbol"):
                parts.append(str(c["symbol"]))
            if c.get("path"):
                parts.append(str(c["path"]))
            code = c.get("code") or c.get("snippet") or c.get("text") or ""
            if code:
                parts.append(str(code)[:400])
            doc_texts.append(" ".join(parts) if parts else "empty")

        onnx_scores = self._onnx_score(query, doc_texts)
        if onnx_scores is None:
            return super().rerank(query, candidates, initial_scores)

        scores = onnx_scores.copy()
        query_emb = self._encode([query])[0]
        doc_embs = self._encode(doc_texts)
        query_emb = self._project_to_dim(query_emb.reshape(1, -1))[0]
        doc_embs = self._project_to_dim(doc_embs)

        z = query_emb.copy()
        state = RefinementState(z=z, scores=scores, iteration=0)
        state.score_history.append(scores.copy())

        for i in range(self.n_iterations - 1):
            state.iteration = i + 1
            state.z = self.refiner.refine(state.z, query_emb, doc_embs, state.scores)
            adjustment = self.scorer.forward(query_emb, doc_embs, state.z)
            try:
                metrics = self.scorer.get_metrics()
                if metrics.get("converged", False) and metrics.get("avg_loss", 1.0) < 0.3:
                    alpha = 0.5
                elif metrics.get("update_count", 0) > 100:
                    alpha = 0.35
                else:
                    alpha = 0.2
            except Exception:
                alpha = 0.2
            state.scores = (1 - alpha) * state.scores + alpha * adjustment
            state.score_history.append(state.scores.copy())
            if self.early_stop and confidence.should_stop(state):
                break

        final_scores = state.scores
        if initial_scores is not None and self.blend_with_initial > 0:
            init_arr = np.array(initial_scores, dtype=np.float32)
            std = final_scores.std()
            if std > 1e-6:
                final_norm = (final_scores - final_scores.mean()) / std
            else:
                final_norm = final_scores - final_scores.mean()
            std = init_arr.std()
            if std > 1e-6:
                init_norm = (init_arr - init_arr.mean()) / std
            else:
                init_norm = init_arr - init_arr.mean()
            final_scores = (1 - self.blend_with_initial) * final_norm + self.blend_with_initial * init_norm

        ranked_indices = np.argsort(-final_scores)
        reranked = []
        for rank, idx in enumerate(ranked_indices):
            candidate = candidates[idx].copy()
            candidate["recursive_score"] = float(final_scores[idx])
            candidate["onnx_score"] = float(onnx_scores[idx])
            candidate["recursive_rank"] = rank
            candidate["recursive_iterations"] = state.iteration + 1
            candidate["score_trajectory"] = [float(h[idx]) for h in state.score_history]
            candidate["score"] = float(final_scores[idx])
            reranked.append(candidate)
        return reranked


class FastEmbedRecursiveReranker(RecursiveReranker):
    """Recursive reranker using FastEmbed cross-encoder (via reranker factory).

    Uses the centralized reranker factory which supports both RERANKER_MODEL
    (FastEmbed auto-download) and legacy RERANKER_ONNX_PATH configs.
    """

    def __init__(self, n_iterations: int = 3, **kwargs):
        super().__init__(n_iterations=n_iterations, **kwargs)
        self._reranker_model = None
        self._model_lock = threading.Lock()

    def _get_model(self):
        """Get cached reranker model from factory."""
        if self._reranker_model is not None:
            return self._reranker_model
        if not HAS_RERANKER_FACTORY or _get_reranker_model is None:
            return None
        with self._model_lock:
            if self._reranker_model is not None:
                return self._reranker_model
            self._reranker_model = _get_reranker_model()
            return self._reranker_model

    def _factory_score(self, query: str, docs: List[str]) -> Optional[np.ndarray]:
        """Score documents using reranker factory."""
        model = self._get_model()
        if model is None or _rerank_pairs is None:
            return None
        try:
            pairs = [(query, doc) for doc in docs]
            scores = _rerank_pairs(pairs, model=model)
            return np.array(scores, dtype=np.float32)
        except Exception:
            return None

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        initial_scores: Optional[List[float]] = None,
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        confidence = ConfidenceEstimator()

        doc_texts = []
        for c in candidates:
            parts = []
            if c.get("symbol"):
                parts.append(str(c["symbol"]))
            if c.get("path"):
                parts.append(str(c["path"]))
            code = c.get("code") or c.get("snippet") or c.get("text") or ""
            if code:
                parts.append(str(code)[:400])
            doc_texts.append(" ".join(parts) if parts else "empty")

        factory_scores = self._factory_score(query, doc_texts)
        if factory_scores is None:
            return super().rerank(query, candidates, initial_scores)

        scores = factory_scores.copy()
        query_emb = self._encode([query])[0]
        doc_embs = self._encode(doc_texts)
        query_emb = self._project_to_dim(query_emb.reshape(1, -1))[0]
        doc_embs = self._project_to_dim(doc_embs)

        z = query_emb.copy()
        state = RefinementState(z=z, scores=scores, iteration=0)
        state.score_history.append(scores.copy())

        for i in range(self.n_iterations - 1):
            state.iteration = i + 1
            state.z = self.refiner.refine(state.z, query_emb, doc_embs, state.scores)
            adjustment = self.scorer.forward(query_emb, doc_embs, state.z)
            try:
                metrics = self.scorer.get_metrics()
                if metrics.get("converged", False) and metrics.get("avg_loss", 1.0) < 0.3:
                    alpha = 0.5
                elif metrics.get("update_count", 0) > 100:
                    alpha = 0.35
                else:
                    alpha = 0.2
            except Exception:
                alpha = 0.2
            state.scores = (1 - alpha) * state.scores + alpha * adjustment
            state.score_history.append(state.scores.copy())
            if self.early_stop and confidence.should_stop(state):
                break

        final_scores = state.scores
        if initial_scores is not None and self.blend_with_initial > 0:
            init_arr = np.array(initial_scores, dtype=np.float32)
            std = final_scores.std()
            if std > 1e-6:
                final_norm = (final_scores - final_scores.mean()) / std
            else:
                final_norm = final_scores - final_scores.mean()
            std = init_arr.std()
            if std > 1e-6:
                init_norm = (init_arr - init_arr.mean()) / std
            else:
                init_norm = init_arr - init_arr.mean()
            final_scores = (1 - self.blend_with_initial) * final_norm + self.blend_with_initial * init_norm

        ranked_indices = np.argsort(-final_scores)
        reranked = []
        for rank, idx in enumerate(ranked_indices):
            candidate = candidates[idx].copy()
            candidate["recursive_score"] = float(final_scores[idx])
            candidate["factory_score"] = float(factory_scores[idx])
            candidate["recursive_rank"] = rank
            candidate["recursive_iterations"] = state.iteration + 1
            candidate["score_trajectory"] = [float(h[idx]) for h in state.score_history]
            candidate["score"] = float(final_scores[idx])
            reranked.append(candidate)
        return reranked


class SessionAwareReranker:
    """Session-aware recursive reranker with latent state carryover."""

    def __init__(
        self,
        n_iterations: int = 3,
        dim: int = 256,
        session_decay: float = 0.9,
        max_session_age: float = 3600.0,
        max_sessions: int = 1000,
    ):
        self.n_iterations = n_iterations
        self.dim = dim
        self.session_decay = session_decay
        self.max_session_age = max_session_age
        self.max_sessions = max_sessions
        self.reranker = RecursiveReranker(n_iterations=n_iterations, dim=dim)
        self._sessions: Dict[str, tuple] = {}
        self._session_lock = threading.Lock()

    def _cleanup_old_sessions(self):
        now = time.time()
        expired = [sid for sid, (_, last_access) in self._sessions.items() if now - last_access > self.max_session_age]
        for sid in expired:
            del self._sessions[sid]
        if len(self._sessions) > self.max_sessions:
            sorted_sessions = sorted(self._sessions.items(), key=lambda x: x[1][1])
            to_remove = len(self._sessions) - self.max_sessions
            for sid, _ in sorted_sessions[:to_remove]:
                del self._sessions[sid]

    def get_session_latent(self, session_id: str) -> Optional[np.ndarray]:
        with self._session_lock:
            if session_id not in self._sessions:
                return None
            latent, last_access = self._sessions[session_id]
            if time.time() - last_access > self.max_session_age:
                del self._sessions[session_id]
                return None
            return latent

    def update_session_latent(self, session_id: str, new_latent: np.ndarray):
        with self._session_lock:
            self._cleanup_old_sessions()
            if session_id in self._sessions:
                old_latent, _ = self._sessions[session_id]
                blended = self.session_decay * old_latent + (1 - self.session_decay) * new_latent
                blended = blended / (np.linalg.norm(blended) + 1e-8)
                self._sessions[session_id] = (blended, time.time())
            else:
                self._sessions[session_id] = (new_latent.copy(), time.time())

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        session_id: Optional[str] = None,
        initial_scores: Optional[List[float]] = None,
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        session_latent = None
        if session_id:
            session_latent = self.get_session_latent(session_id)

        query_emb = self.reranker._encode([query])[0]
        query_emb = self.reranker._project_to_dim(query_emb.reshape(1, -1))[0]

        if session_latent is not None:
            initial_z = 0.7 * query_emb + 0.3 * session_latent
            initial_z = initial_z / (np.linalg.norm(initial_z) + 1e-8)
        else:
            initial_z = query_emb.copy()

        doc_texts = []
        for c in candidates:
            text_parts = []
            if c.get("symbol"):
                text_parts.append(str(c["symbol"]))
            if c.get("path"):
                text_parts.append(str(c["path"]))
            code = c.get("code") or c.get("snippet") or c.get("text") or ""
            if code:
                text_parts.append(str(code)[:500])
            doc_texts.append(" ".join(text_parts) if text_parts else "empty")

        doc_embs = self.reranker._encode(doc_texts)
        doc_embs = self.reranker._project_to_dim(doc_embs)

        if initial_scores is None:
            initial_scores = [c.get("score", 0.0) for c in candidates]

        state = RefinementState(z=initial_z, scores=np.array(initial_scores, dtype=np.float32))
        state.score_history.append(state.scores.copy())

        confidence = ConfidenceEstimator()
        for i in range(self.n_iterations):
            state.iteration = i + 1
            new_scores = self.reranker.scorer.forward(query_emb, doc_embs, state.z)
            alpha = 0.5
            state.scores = alpha * new_scores + (1 - alpha) * state.scores
            state.score_history.append(state.scores.copy())
            state.z = self.reranker.refiner.refine(state.z, query_emb, doc_embs, state.scores)
            if self.reranker.early_stop and confidence.should_stop(state):
                break

        if session_id:
            self.update_session_latent(session_id, state.z)

        final_scores = state.scores
        if self.reranker.blend_with_initial > 0:
            init_arr = np.array(initial_scores, dtype=np.float32)
            std = final_scores.std()
            if std > 1e-6:
                final_norm = (final_scores - final_scores.mean()) / std
            else:
                final_norm = final_scores - final_scores.mean()
            std = init_arr.std()
            if std > 1e-6:
                init_norm = (init_arr - init_arr.mean()) / std
            else:
                init_norm = init_arr - init_arr.mean()
            final_scores = (1 - self.reranker.blend_with_initial) * final_norm + self.reranker.blend_with_initial * init_norm

        ranked_indices = np.argsort(-final_scores)
        reranked = []
        for rank, idx in enumerate(ranked_indices):
            candidate = candidates[idx].copy()
            candidate["recursive_score"] = float(final_scores[idx])
            candidate["recursive_rank"] = rank
            candidate["recursive_iterations"] = state.iteration
            candidate["session_aware"] = session_id is not None
            candidate["score_trajectory"] = [float(h[idx]) for h in state.score_history]
            candidate["score"] = float(final_scores[idx])
            reranked.append(candidate)
        return reranked

    def clear_session(self, session_id: str):
        with self._session_lock:
            if session_id in self._sessions:
                del self._sessions[session_id]

    def get_session_count(self) -> int:
        with self._session_lock:
            return len(self._sessions)


# Convenience functions
def rerank_recursive(
    query: str,
    candidates: List[Dict[str, Any]],
    n_iterations: int = 3,
    blend_with_initial: float = 0.3,
) -> List[Dict[str, Any]]:
    """Convenience wrapper for recursive reranking."""
    reranker = RecursiveReranker(n_iterations=n_iterations, blend_with_initial=blend_with_initial)
    initial_scores = [c.get("score", 0.0) for c in candidates]
    return reranker.rerank(query, candidates, initial_scores)


def rerank_recursive_inprocess(
    query: str,
    candidates: List[Dict[str, Any]],
    limit: int = 12,
    n_iterations: int = 3,
) -> List[Dict[str, Any]]:
    """In-process recursive reranking for MCP server integration."""
    reranked = rerank_recursive(query=query, candidates=candidates, n_iterations=n_iterations)
    return reranked[:limit]


# Per-collection learning rerankers
_LEARNING_RERANKERS: Dict[str, RecursiveReranker] = {}
_LEARNING_RERANKERS_LOCK = threading.Lock()


def _get_learning_reranker(
    n_iterations: int = 3,
    dim: int = 256,
    collection: str = "default",
) -> RecursiveReranker:
    """Get or create a learning reranker for a specific collection."""
    with _LEARNING_RERANKERS_LOCK:
        if collection not in _LEARNING_RERANKERS:
            reranker = RecursiveReranker(n_iterations=n_iterations, dim=dim)
            reranker.scorer.set_collection(collection)
            reranker.refiner.set_collection(collection)
            reranker._learned_projection.set_collection(collection)
            _LEARNING_RERANKERS[collection] = reranker
        return _LEARNING_RERANKERS[collection]


def rerank_with_learning(
    query: str,
    candidates: List[Dict[str, Any]],
    limit: int = 12,
    n_iterations: int = 3,
    learn_from_onnx: bool = True,
    collection: str = "default",
) -> List[Dict[str, Any]]:
    """Learning-enabled reranking for MCP server integration."""
    reranker = _get_learning_reranker(n_iterations=n_iterations, collection=collection)
    initial_scores = [c.get("score", 0) for c in candidates]

    if learn_from_onnx and candidates:
        teacher_scores = None
        if str(os.environ.get("RERANK_TEACHER_INLINE", "")).strip().lower() in {"1", "true", "yes", "on"}:
            try:
                from scripts.rerank_local import rerank_local
            except ImportError:
                try:
                    from rerank_local import rerank_local
                except ImportError:
                    rerank_local = None
            if rerank_local is not None:
                try:
                    pairs = []
                    for c in candidates:
                        doc = c.get("code") or c.get("snippet") or ""
                        if not doc:
                            parts = []
                            if c.get("symbol"):
                                parts.append(str(c["symbol"]))
                            if c.get("path"):
                                parts.append(str(c["path"]))
                            doc = " ".join(parts) if parts else "empty"
                        pairs.append((query, doc[:1000]))
                    teacher_scores = rerank_local(pairs)
                except Exception:
                    teacher_scores = None
        try:
            try:
                from rerank_events import log_training_event
            except ImportError:
                from scripts.rerank_events import log_training_event
            log_training_event(
                query=query,
                candidates=candidates,
                initial_scores=initial_scores,
                teacher_scores=(list(teacher_scores) if teacher_scores is not None else None),
                collection=collection,
                metadata={"teacher_inline": bool(teacher_scores is not None)},
            )
        except Exception:
            pass

    reranked = reranker.rerank(query, candidates, initial_scores)
    return reranked[:limit]


def get_recursive_reranker(n_iterations: int = 3, **kwargs) -> RecursiveReranker:
    """Get the best available recursive reranker.

    Priority:
    1. RERANKER_MODEL (FastEmbed) via factory -> FastEmbedRecursiveReranker
    2. RERANKER_ONNX_PATH + RERANKER_TOKENIZER_PATH -> ONNXRecursiveReranker
    3. Fallback -> base RecursiveReranker (no neural reranking)

    Backwards compatible: existing ONNX configs continue to work.
    """
    # Priority 1: Use factory if RERANKER_MODEL is set
    if HAS_RERANKER_FACTORY and _is_reranker_available is not None:
        if _is_reranker_available():
            return FastEmbedRecursiveReranker(n_iterations=n_iterations, **kwargs)

    # Priority 2: Legacy ONNX path (backwards compatibility)
    onnx_path = os.environ.get("RERANKER_ONNX_PATH", "")
    tokenizer_path = os.environ.get("RERANKER_TOKENIZER_PATH", "")
    if HAS_ONNX and onnx_path and tokenizer_path:
        return ONNXRecursiveReranker(n_iterations=n_iterations, **kwargs)

    # Priority 3: Base reranker (no neural scoring)
    return RecursiveReranker(n_iterations=n_iterations, **kwargs)


def rerank_with_session(
    query: str,
    candidates: List[Dict[str, Any]],
    session_id: str,
    n_iterations: int = 3,
) -> List[Dict[str, Any]]:
    """Session-aware reranking (stateless convenience wrapper)."""
    reranker = SessionAwareReranker(n_iterations=n_iterations)
    return reranker.rerank(query, candidates, session_id=session_id)
