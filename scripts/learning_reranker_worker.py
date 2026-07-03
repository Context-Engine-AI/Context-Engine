#!/usr/bin/env python3
"""
Background Learning Reranker Worker.

Consumes training events from the event log and updates weights per-collection.
This keeps the MCP serving path fast and deterministic.

Features:
- Reads events from NDJSON log files (one per collection)
- Batches teacher scoring to amortize ONNX overhead
- Updates weights atomically (write to .tmp, rename)
- Supports multiple collections with isolated weights
- Can run as a daemon or one-shot

Usage:
    # Run continuously (daemon mode)
    python scripts/learning_reranker_worker.py --daemon

    # Process pending events once and exit
    python scripts/learning_reranker_worker.py --once

    # Process specific collection
    python scripts/learning_reranker_worker.py --collection my-repo
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.rerank_events import (
    RERANK_EVENTS_DIR,
    RERANK_EVENTS_RETENTION_DAYS,
    read_events,
    list_event_files,
    cleanup_old_events,
)
from scripts.rerank_recursive import (
    TinyScorer,
    LatentRefiner,
    RecursiveReranker,
    VICReg,
    LearnedProjection,
    LearnedHybridWeights,
    QueryExpander,
)

# Configuration
BATCH_SIZE = int(os.environ.get("RERANK_LEARNING_BATCH_SIZE", "32"))
POLL_INTERVAL = float(os.environ.get("RERANK_LEARNING_POLL_INTERVAL", "30"))
LEARNING_RATE = float(os.environ.get("RERANK_LEARNING_RATE", "0.001"))

# VICReg configuration
VICREG_WEIGHT = float(os.environ.get("RERANK_VICREG_WEIGHT", "0.1"))
VICREG_MIN_BATCH = int(os.environ.get("RERANK_VICREG_MIN_BATCH", "4"))  # Need 4+ samples for covariance

# GLM teacher configuration (online learning with LLM judgments)
LLM_TEACHER_ENABLED = os.environ.get("RERANK_LLM_TEACHER", "0") == "1"
LLM_TEACHER_SAMPLE_RATE = float(os.environ.get("RERANK_LLM_SAMPLE_RATE", "1.0"))  # 100% when enabled (background anyway)


def get_logger():
    """Get logger for worker."""
    try:
        from scripts.logger import get_logger as _get_logger
        return _get_logger(__name__)
    except Exception:
        import logging
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)


logger = get_logger()


class CollectionLearner:
    """Handles learning for a single collection."""

    def __init__(self, collection: str):
        self.collection = collection
        self._lock_file = None

        # Acquire exclusive lock to prevent multiple workers on same collection
        self._acquire_lock()

        self.scorer = TinyScorer(lr=LEARNING_RATE)
        self.scorer.set_collection(collection)
        self.refiner = LatentRefiner(dim=self.scorer.dim, lr=LEARNING_RATE)
        self.refiner.set_collection(collection)

        # Learned projection: raw embedding dim → working dim (256)
        # Lower LR than scorer/refiner - projection is more sensitive
        from scripts.embedder import get_model_dimension
        embed_dim = get_model_dimension()  # Respects EMBEDDING_MODEL env
        self.projection = LearnedProjection(
            input_dim=embed_dim,
            output_dim=self.scorer.dim,
            lr=LEARNING_RATE * 0.5,  # Half the learning rate
        )
        self.projection.set_collection(collection)

        self._last_processed_ts = self._load_checkpoint()

        # Reuse the serving reranker's embed + project code path 1:1.
        # We only use its private feature helpers, not its scoring loop.
        self._feature_reranker = RecursiveReranker(
            n_iterations=1,
            dim=self.scorer.dim,
            early_stop=False,
            blend_with_initial=0.0,
        )

        # VICReg for residual regularization (prevents collapse, decorrelates)
        self.vicreg = VICReg(
            lambda_var=1.0,
            lambda_cov=0.04,
            lambda_inv=0.1,
        ) if VICREG_WEIGHT > 0 else None

        # Learned hybrid weights: dense vs. lexical balance
        self.hybrid_weights = LearnedHybridWeights(lr=0.01)
        self.hybrid_weights.set_collection(collection)

        # Query expander: learns synonyms/related terms from usage
        self.query_expander = QueryExpander(lr=0.1)
        self.query_expander.set_collection(collection)

        # LLM teacher for higher-quality supervision (optional)
        # Supports llama.cpp or GLM API - auto-detects based on env
        self._llm_client = None
        self._llm_runtime = None
        self._llm_calls = 0
        if LLM_TEACHER_ENABLED:
            try:
                # Auto-detect runtime: GLM_API_KEY -> glm, else -> llamacpp
                runtime = os.environ.get("REFRAG_RUNTIME", "").strip().lower()
                if not runtime:
                    if os.environ.get("GLM_API_KEY", "").strip():
                        runtime = "glm"
                    else:
                        runtime = "llamacpp"

                if runtime == "glm":
                    from scripts.refrag_glm import GLMRefragClient
                    self._llm_client = GLMRefragClient()
                else:
                    from scripts.refrag_llamacpp import LlamaCppRefragClient, is_decoder_enabled
                    if is_decoder_enabled():
                        self._llm_client = LlamaCppRefragClient()
                    else:
                        logger.info(f"[{collection}] LLM teacher skipped (decoder disabled)")

                if self._llm_client:
                    self._llm_runtime = runtime
                    logger.info(f"[{collection}] LLM teacher enabled ({runtime}, sample_rate={LLM_TEACHER_SAMPLE_RATE})")
            except Exception as e:
                logger.warning(f"[{collection}] LLM teacher unavailable: {e}")

        # Metrics tracking for logging
        self._vicreg_loss_sum = 0.0
        self._vicreg_count = 0
        self._proj_grad_norm_sum = 0.0
        self._proj_grad_count = 0
        self._hybrid_updates = 0
        self._expander_updates = 0

    @staticmethod
    def _sanitize_collection(collection: str) -> str:
        """Sanitize collection name to prevent path traversal."""
        return "".join(c if c.isalnum() or c in "-_" else "_" for c in collection)

    def _acquire_lock(self):
        """Acquire exclusive lock to prevent multiple workers on same collection."""
        import fcntl
        safe_name = self._sanitize_collection(self.collection)
        lock_path = Path(TinyScorer.WEIGHTS_DIR) / f"{safe_name}.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock_file = open(lock_path, "w")
        try:
            fcntl.flock(self._lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            logger.debug(f"[{self.collection}] Acquired exclusive lock")
        except OSError:
            logger.error(f"[{self.collection}] Another worker already running. Exiting.")
            sys.exit(1)

    def _release_lock(self):
        """Release the exclusive lock."""
        if self._lock_file:
            import fcntl
            try:
                fcntl.flock(self._lock_file, fcntl.LOCK_UN)
                self._lock_file.close()
            except Exception:
                pass
            self._lock_file = None

    def _load_checkpoint(self) -> float:
        """Load last processed timestamp from checkpoint file."""
        safe_name = self._sanitize_collection(self.collection)
        checkpoint_path = Path(TinyScorer.WEIGHTS_DIR) / f"checkpoint_{safe_name}.json"
        try:
            if checkpoint_path.exists():
                with open(checkpoint_path) as f:
                    return json.load(f).get("last_ts", 0)
        except Exception as e:
            logger.warning(f"[{self.collection}] Failed to load checkpoint: {e}")
        return 0

    def _save_checkpoint(self, ts: float):
        """Save last processed timestamp to checkpoint file atomically."""
        safe_name = self._sanitize_collection(self.collection)
        checkpoint_path = Path(TinyScorer.WEIGHTS_DIR) / f"checkpoint_{safe_name}.json"
        tmp_path = checkpoint_path.with_suffix(".json.tmp")
        try:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp_path, "w") as f:
                json.dump({"last_ts": ts, "collection": self.collection}, f)
            os.replace(tmp_path, checkpoint_path)
        except Exception as e:
            logger.warning(f"[{self.collection}] Failed to save checkpoint: {e}")
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except Exception:
                    pass

    def _encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to BGE embeddings (768-dim, raw without projection)."""
        return self._feature_reranker._encode_raw(texts)

    def _encode_project(self, texts: List[str]) -> np.ndarray:
        """Encode + project via learned projection (for inference/serving compat)."""
        embs = self._encode(texts)
        return self.projection.forward(embs)

    def _encode_project_with_cache(
        self, texts: List[str]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Encode + project with cache for gradient backprop through projection.

        Returns:
            projected: (batch, output_dim) projected embeddings
            cache: dict with raw_embs and projection cache for backward pass
        """
        raw_embs = self._encode(texts)  # (batch, 768)
        projected, proj_cache = self.projection.forward_with_cache(raw_embs)
        cache = {"raw_embs": raw_embs, "proj_cache": proj_cache}
        return projected, cache

    @staticmethod
    def _pack_doc(candidate: Dict[str, Any], max_chars: int = 500) -> str:
        """Pack candidate into doc text (shared by learning and teacher scoring)."""
        parts = []
        if candidate.get("symbol"):
            parts.append(str(candidate["symbol"]))
        if candidate.get("path"):
            parts.append(str(candidate["path"]))
        code = candidate.get("code") or candidate.get("snippet") or candidate.get("text") or ""
        if code:
            parts.append(str(code)[:max_chars])
        return " ".join(parts) if parts else "empty"

    def _llm_judge(self, query: str, candidates: List[Dict[str, Any]], top_k: int = 5) -> Optional[np.ndarray]:
        """Get LLM relevance judgments for top candidates.

        Works with llama.cpp or GLM API (auto-detected at init).

        Returns:
            scores: (n_candidates,) array with LLM-derived scores, or None if failed
        """
        if not self._llm_client or not candidates:
            return None

        # Only judge top-k to save API cost
        n = len(candidates)
        top_k = min(top_k, n)

        # Build prompt for GLM to rate relevance
        prompt_parts = [f"Rate code search relevance 0-10 for query: \"{query}\"\n"]
        for i in range(top_k):
            c = candidates[i]
            doc = self._pack_doc(c, max_chars=800)
            prompt_parts.append(f"[{i}] {doc}")

        prompt_parts.append(
            "\nRespond with JSON: {\"scores\": [N, N, ...]} where each N is 0-10 relevance."
        )
        prompt = "\n".join(prompt_parts)

        try:
            import json
            response = self._llm_client.generate_with_soft_embeddings(
                prompt=prompt,
                max_tokens=64,
                temperature=0.1,
                force_json=True,
                disable_thinking=True,  # Fast mode
            )
            data = json.loads(response)
            llm_scores = data.get("scores", [])

            if len(llm_scores) >= top_k:
                # Normalize to 0-1 and extend to all candidates
                scores = np.zeros(n, dtype=np.float32)
                for i in range(top_k):
                    scores[i] = float(llm_scores[i]) / 10.0
                # Lower candidates get decaying scores
                for i in range(top_k, n):
                    scores[i] = max(0, scores[top_k - 1] * 0.5 ** (i - top_k + 1))

                self._llm_calls += 1
                return scores
        except Exception as e:
            logger.debug(f"[{self.collection}] LLM judge failed: {e}")

        return None

    def process_events(self, limit: int = 1000) -> int:
        """Process pending events and return count processed."""
        events = read_events(self.collection, since_ts=self._last_processed_ts, limit=limit)
        if not events:
            return 0

        processed = 0
        batch_events = []

        for event in events:
            batch_events.append(event)
            if len(batch_events) >= BATCH_SIZE:
                self._learn_from_batch(batch_events)
                processed += len(batch_events)
                batch_events = []

        # Process remaining
        if batch_events:
            self._learn_from_batch(batch_events)
            processed += len(batch_events)

        # Update checkpoint
        if events:
            self._last_processed_ts = max(e.get("ts", 0) for e in events)
            self._save_checkpoint(self._last_processed_ts)

        # Save weights after processing batch (both scorer and refiner)
        if processed > 0:
            self.scorer._save_weights(checkpoint=True)
            self.refiner._save_weights(checkpoint=True)
            metrics = self.scorer.get_metrics()

            # VICReg + projection info
            extra_info = ""
            if self._vicreg_count > 0:
                avg_vicreg = self._vicreg_loss_sum / self._vicreg_count
                extra_info += f" | vicreg={avg_vicreg:.4f}"
                self._vicreg_loss_sum = 0.0
                self._vicreg_count = 0

            if self._proj_grad_count > 0:
                avg_proj_grad = self._proj_grad_norm_sum / self._proj_grad_count
                extra_info += f" | proj_grad={avg_proj_grad:.4f}"
                self._proj_grad_norm_sum = 0.0
                self._proj_grad_count = 0

            # Hybrid weight, expander, and GLM stats
            hw_info = f" dense_w={self.hybrid_weights.dense_weight:.2f}"
            exp_stats = self.query_expander.get_stats()
            exp_info = f" terms={exp_stats['terms']}"
            llm_info = f" llm={self._llm_calls}({self._llm_runtime})" if self._llm_client else ""

            logger.info(
                f"[{self.collection}] Processed {processed} events | "
                f"scorer_v{metrics['version']} refiner_v{self.refiner._version} proj_v{self.projection._version}{hw_info}{exp_info}{llm_info} | "
                f"lr={metrics['learning_rate']:.6f} | "
                f"avg_loss={metrics['avg_loss']:.4f}{extra_info} | converged={metrics['converged']}"
            )

        return processed

    def _learn_from_batch(self, events: List[Dict[str, Any]]):
        """Learn from batch with 3-pass deep supervision + VICReg + projection learning.

        Full end-to-end learning:
        - Projection: BGE (768) → working dim (256), learns domain-specific subspace
        - Scorer: learns to rank with current z
        - Refiner: learns to improve z toward teacher-optimal state
        - VICReg: regularizes residuals to prevent collapse

        Deep Supervision (TRM-style):
        - Each refinement pass gets a loss signal toward teacher_z
        - Later passes have decaying weight (pass 1: 1.0, pass 2: 0.7, pass 3: 0.5)
        """
        # Fill missing teacher scores in batch (amortizes ONNX overhead)
        self._maybe_fill_teacher_scores(events)

        # Accumulate data for end-of-batch updates
        vicreg_data: List[tuple] = []  # [(z, z_refined, refiner_cache), ...]
        projection_grads: List[Tuple[np.ndarray, Dict[str, Any]]] = []  # [(grad, proj_cache), ...]

        # Deep supervision weights: earlier passes contribute more
        pass_weights = [1.0, 0.7, 0.5]
        n_passes = 3

        for event in events:
            try:
                query = event.get("query", "")
                candidates = event.get("candidates", [])
                teacher_scores = event.get("teacher_scores")

                if not query or not candidates or not teacher_scores:
                    continue

                # Validate alignment between candidates and teacher scores
                if len(teacher_scores) != len(candidates):
                    logger.warning(f"[{self.collection}] Skipping event: mismatched lengths")
                    continue

                # Build doc texts from candidates
                doc_texts = [self._pack_doc(c) for c in candidates]
                teacher_arr = np.array(teacher_scores, dtype=np.float32)

                # ===== LLM TEACHER: Higher-quality supervision (sampled) =====
                if self._llm_client and np.random.random() < LLM_TEACHER_SAMPLE_RATE:
                    llm_scores = self._llm_judge(query, candidates, top_k=5)
                    if llm_scores is not None:
                        # Blend LLM with ONNX: LLM is more reliable, weight it higher
                        teacher_arr = 0.3 * teacher_arr + 0.7 * llm_scores

                # ===== ENCODE WITH CACHE FOR PROJECTION LEARNING =====
                # Query embedding with cache for gradient backprop
                query_proj, query_proj_cache = self._encode_project_with_cache([query])
                query_emb = query_proj[0]

                # Doc embeddings with cache for gradient backprop
                doc_embs, doc_proj_cache = self._encode_project_with_cache(doc_texts)

                # Compute teacher-weighted document summary as target z
                teacher_weights = np.exp(teacher_arr - teacher_arr.max())
                teacher_weights = teacher_weights / (teacher_weights.sum() + 1e-8)
                teacher_z = (teacher_weights[:, None] * doc_embs).sum(axis=0)
                teacher_z = teacher_z / (np.linalg.norm(teacher_z) + 1e-8)

                # Initialize latent state from query
                z = query_emb.copy()

                # ===== PROJECTION GRADIENT: Contrastive alignment loss =====
                # Goal: projection should produce embeddings where query is close to
                # high-scoring docs and far from low-scoring docs
                # Gradient w.r.t. query: weighted by teacher scores
                # Push query toward high-teacher-score docs
                query_grad = (teacher_weights[:, None] * doc_embs).sum(axis=0) - query_emb
                query_grad = query_grad / (np.linalg.norm(query_grad) + 1e-8)

                # Gradient w.r.t. docs: each doc pulled/pushed based on teacher score
                # High score docs: pull toward query, Low score docs: push away
                centered_weights = teacher_weights - teacher_weights.mean()
                doc_grad = centered_weights[:, None] * (query_emb - doc_embs)

                projection_grads.append((query_grad.reshape(1, -1), query_proj_cache))
                projection_grads.append((doc_grad, doc_proj_cache))

                # ===== LEARN HYBRID WEIGHTS (dense vs. lexical) =====
                # Extract dense/lexical scores if available in candidates
                if candidates and "dense_score" in candidates[0] and "lexical_score" in candidates[0]:
                    dense_scores = np.array([c.get("dense_score", 0) for c in candidates], dtype=np.float32)
                    lexical_scores = np.array([c.get("lexical_score", 0) for c in candidates], dtype=np.float32)
                    self.hybrid_weights.learn_from_teacher(dense_scores, lexical_scores, teacher_arr)
                    self._hybrid_updates += 1

                # ===== LEARN QUERY EXPANSIONS =====
                # Learn term associations from high-scoring docs
                self.query_expander.learn_from_teacher(query, doc_texts, teacher_arr)
                self._expander_updates += 1

                # ===== 3-PASS DEEP SUPERVISION =====
                for pass_idx in range(n_passes):
                    # Score with current z
                    scores = self.scorer.forward(query_emb, doc_embs, z)

                    # Train scorer at this pass (learns to rank with current z)
                    self.scorer.learn_from_teacher(query_emb, doc_embs, z, teacher_arr)

                    # Train refiner: z → z' toward teacher_z
                    if self.vicreg is not None:
                        # Get cache for VICReg backprop
                        _, z_orig, z_refined, cache = self.refiner.learn_from_teacher_with_cache(
                            z, query_emb, doc_embs, scores, teacher_z
                        )
                        cache["pass_weight"] = pass_weights[pass_idx]
                        vicreg_data.append((z_orig, z_refined, cache))
                    else:
                        self.refiner.learn_from_teacher(z, query_emb, doc_embs, scores, teacher_z)

                    # Update z for next pass (refinement chain)
                    z = self.refiner.refine(z, query_emb, doc_embs, scores)

            except Exception as e:
                logger.warning(f"[{self.collection}] Error processing event: {e}")
                continue

        # ===== PROJECTION LEARNING: batch update =====
        if projection_grads:
            try:
                for grad, cache in projection_grads:
                    self.projection.backward(grad, cache["proj_cache"], weight=0.1)
                    self._proj_grad_norm_sum += np.linalg.norm(grad)
                    self._proj_grad_count += 1
            except Exception as e:
                logger.warning(f"[{self.collection}] Projection update failed: {e}")

        # ===== VICReg: batch-level residual regularization =====
        if self.vicreg is not None and len(vicreg_data) >= VICREG_MIN_BATCH:
            try:
                z_batch = np.vstack([item[0] for item in vicreg_data])
                z_refined_batch = np.vstack([item[1] for item in vicreg_data])

                vicreg_loss, vicreg_grad, _ = self.vicreg.forward(z_batch, z_refined_batch)

                self._vicreg_loss_sum += vicreg_loss
                self._vicreg_count += 1

                for i, (_, _, cache) in enumerate(vicreg_data):
                    pass_weight = cache.get("pass_weight", 1.0)
                    self.refiner.apply_vicreg_gradient(
                        vicreg_grad[i], cache, weight=VICREG_WEIGHT * pass_weight
                    )

            except Exception as e:
                logger.warning(f"[{self.collection}] VICReg failed: {e}")

    def _maybe_fill_teacher_scores(self, events: List[Dict[str, Any]]):
        """Compute teacher scores for events that don't already have them."""
        try:
            from scripts.rerank_local import rerank_local
        except Exception:
            rerank_local = None

        if rerank_local is None:
            return

        # Gather pairs across events so we can call rerank_local once.
        all_pairs: List[tuple] = []
        slices: List[tuple] = []  # (event_index, start, end)

        for event_index, event in enumerate(events):
            if event.get("teacher_scores"):
                continue

            query = event.get("query", "")
            candidates = event.get("candidates", [])
            if not query or not candidates:
                continue

            start = len(all_pairs)
            for c in candidates:
                doc_text = self._pack_doc(c)  # Same packing as learning path
                all_pairs.append((query, doc_text))
            end = len(all_pairs)
            if end > start:
                slices.append((event_index, start, end))

        if not slices:
            return

        try:
            scores = rerank_local(all_pairs)
        except Exception as e:
            logger.warning(f"[{self.collection}] Teacher scoring failed for {len(all_pairs)} pairs: {e}")
            return

        # Map scores back to each event.
        for event_index, start, end in slices:
            try:
                events[event_index]["teacher_scores"] = list(scores[start:end])
            except Exception as e:
                logger.warning(f"[{self.collection}] Failed to map teacher scores for event {event_index}: {e}")
                continue


def discover_collections() -> List[str]:
    """Discover collections with pending events."""
    events_dir = Path(RERANK_EVENTS_DIR)
    if not events_dir.exists():
        return []

    import re

    collections: List[str] = []
    seen = set()
    for f in events_dir.glob("events_*.ndjson"):
        stem = f.stem  # events_<collection>_<YYYYMMDDHH>
        if not stem.startswith("events_"):
            continue

        rest = stem[len("events_") :]
        # Strip the hour suffix if present (10 digits)
        m = re.match(r"^(?P<name>.+)_(?P<hour>\d{10})$", rest)
        name = m.group("name") if m else rest

        if name and name not in seen:
            seen.add(name)
            collections.append(name)

    return collections


def run_once():
    """Process all pending events once and exit."""
    collections = discover_collections()
    if not collections:
        logger.info("No collections with pending events")
        return

    total = 0
    for coll in collections:
        learner = CollectionLearner(coll)
        processed = learner.process_events()
        total += processed

    logger.info(f"Processed {total} events across {len(collections)} collections")


def run_daemon():
    """Run continuously, polling for new events."""
    logger.info(f"Starting learning worker daemon (poll interval: {POLL_INTERVAL}s)")
    learners: Dict[str, CollectionLearner] = {}
    last_cleanup = 0
    cleanup_interval = 3600  # Cleanup old events hourly

    while True:
        try:
            collections = discover_collections()
            for coll in collections:
                if coll not in learners:
                    learners[coll] = CollectionLearner(coll)
                learners[coll].process_events()

            # Periodic cleanup of old event files
            now = time.time()
            if RERANK_EVENTS_RETENTION_DAYS > 0 and now - last_cleanup > cleanup_interval:
                for coll in collections:
                    deleted = cleanup_old_events(coll, RERANK_EVENTS_RETENTION_DAYS)
                    if deleted > 0:
                        logger.info(f"[{coll}] Cleaned up {deleted} old event files")
                last_cleanup = now

        except KeyboardInterrupt:
            logger.info("Shutting down")
            break
        except Exception as e:
            logger.error(f"Error in daemon loop: {e}")

        time.sleep(POLL_INTERVAL)


def main():
    parser = argparse.ArgumentParser(description="Background learning reranker worker")
    parser.add_argument("--daemon", action="store_true", help="Run continuously")
    parser.add_argument("--once", action="store_true", help="Process once and exit")
    parser.add_argument("--collection", type=str, help="Process specific collection only")
    args = parser.parse_args()

    if args.collection:
        learner = CollectionLearner(args.collection)
        processed = learner.process_events()
        logger.info(f"Processed {processed} events for {args.collection}")
    elif args.daemon:
        run_daemon()
    else:
        run_once()


if __name__ == "__main__":
    main()
