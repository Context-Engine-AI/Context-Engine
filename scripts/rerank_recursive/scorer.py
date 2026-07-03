"""
TinyScorer - 2-layer MLP for scoring query-document pairs.

Inspired by TRM: minimal parameters, maximum iterations.
"""
import os
import time
from typing import Any, Dict, List, Tuple

import numpy as np


class TinyScorer:
    """
    Tiny 2-layer MLP for scoring query-document pairs.

    Inspired by TRM: minimal parameters, maximum iterations.
    Production-ready with:
    - Collection-aware weights with atomic loading
    - Checkpoint versioning (keep last N versions)
    - Training metrics (loss, sample count, convergence)
    - Learning rate decay
    - Hot reload from background worker updates
    """

    # Class-level configuration
    WEIGHTS_DIR = os.environ.get("RERANKER_WEIGHTS_DIR", "/tmp/rerank_weights")
    WEIGHTS_RELOAD_INTERVAL = float(os.environ.get("RERANKER_WEIGHTS_RELOAD_INTERVAL", "60"))
    MAX_CHECKPOINTS = int(os.environ.get("RERANKER_MAX_CHECKPOINTS", "5"))
    LR_DECAY_STEPS = int(os.environ.get("RERANKER_LR_DECAY_STEPS", "1000"))
    LR_DECAY_RATE = float(os.environ.get("RERANKER_LR_DECAY_RATE", "0.95"))
    MIN_LR = float(os.environ.get("RERANKER_MIN_LR", "0.0001"))

    def __init__(self, dim: int = 256, hidden_dim: int = 512, lr: float = 0.001):
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.base_lr = lr
        self.lr = lr
        self._collection = "default"
        self._weights_path = self._get_weights_path("default")
        self._weights_mtime = 0.0
        self._last_reload_check = 0.0

        # Training metrics
        self._update_count = 0
        self._total_samples = 0
        self._cumulative_loss = 0.0
        self._recent_losses: List[float] = []  # Rolling window for convergence detection
        self._version = 0

        # Try to load saved weights, otherwise init random
        if os.path.exists(self._weights_path):
            try:
                self._load_weights()
                return
            except Exception as e:
                from scripts.logger import get_logger
                get_logger(__name__).warning(f"TinyScorer: failed to load {self._weights_path}: {e}, using random init")

        self._init_random_weights()

    def _init_random_weights(self):
        """Initialize weights randomly using He initialization (local RNG, deterministic)."""
        rng = np.random.RandomState(42)
        scale = np.float32(np.sqrt(2.0 / (self.dim * 3)))
        self.W1 = rng.randn(self.dim * 3, self.hidden_dim).astype(np.float32) * scale
        self.b1 = np.zeros(self.hidden_dim, dtype=np.float32)
        w2_scale = np.float32(np.sqrt(2.0 / self.hidden_dim))
        self.W2 = rng.randn(self.hidden_dim, 1).astype(np.float32) * w2_scale
        self.b2 = np.zeros(1, dtype=np.float32)

        # Momentum for SGD
        self._momentum_W1 = np.zeros_like(self.W1)
        self._momentum_b1 = np.zeros_like(self.b1)
        self._momentum_W2 = np.zeros_like(self.W2)
        self._momentum_b2 = np.zeros_like(self.b2)

    def _update_learning_rate(self):
        """Decay learning rate based on update count."""
        if self._update_count > 0 and self._update_count % self.LR_DECAY_STEPS == 0:
            self.lr = max(self.MIN_LR, self.lr * self.LR_DECAY_RATE)

    def get_metrics(self) -> Dict[str, Any]:
        """Get current training metrics."""
        avg_loss = self._cumulative_loss / max(1, self._update_count)
        recent_avg = np.mean(self._recent_losses) if self._recent_losses else 0.0
        return {
            "collection": self._collection,
            "version": self._version,
            "update_count": self._update_count,
            "total_samples": self._total_samples,
            "cumulative_loss": self._cumulative_loss,
            "avg_loss": avg_loss,
            "recent_avg_loss": float(recent_avg),
            "learning_rate": self.lr,
            "converged": self._is_converged(),
        }

    def _is_converged(self, window: int = 100, threshold: float = 0.01) -> bool:
        """Check if training has converged (loss not improving)."""
        if len(self._recent_losses) < window:
            return False
        recent = self._recent_losses[-window:]
        first_half = np.mean(recent[:window // 2])
        second_half = np.mean(recent[window // 2:])
        return abs(first_half - second_half) < threshold * first_half

    def _get_weights_path(self, collection: str) -> str:
        """Get weights file path for a collection."""
        os.makedirs(self.WEIGHTS_DIR, exist_ok=True)
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in collection)
        return os.path.join(self.WEIGHTS_DIR, f"weights_{safe_name}.npz")

    def set_collection(self, collection: str):
        """Set collection and load corresponding weights."""
        self._collection = collection
        self._weights_path = self._get_weights_path(collection)
        if os.path.exists(self._weights_path):
            try:
                self._load_weights()
            except Exception:
                pass

    def maybe_reload_weights(self):
        """Check if weights file changed and reload if needed (hot reload)."""
        # Fast path: skip if reload disabled (interval <= 0)
        if self.WEIGHTS_RELOAD_INTERVAL <= 0:
            return
        now = time.time()
        if now - self._last_reload_check < self.WEIGHTS_RELOAD_INTERVAL:
            return
        self._last_reload_check = now

        try:
            if os.path.exists(self._weights_path):
                mtime = os.path.getmtime(self._weights_path)
                if mtime > self._weights_mtime:
                    self._load_weights_safe()
        except Exception:
            pass

    def _load_weights_safe(self):
        """Load weights with advisory file locking."""
        import fcntl
        lock_path = self._weights_path + ".lock"
        try:
            os.makedirs(os.path.dirname(lock_path) or ".", exist_ok=True)
            with open(lock_path, "w") as lock_file:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_SH)
                try:
                    self._load_weights()
                finally:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        except Exception:
            self._load_weights()

    def forward(self, query_emb: np.ndarray, doc_emb: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Score documents given query and latent state."""
        self.maybe_reload_weights()

        n_docs = doc_emb.shape[0]
        q_broadcast = np.tile(query_emb, (n_docs, 1))
        z_broadcast = np.tile(z, (n_docs, 1))
        x = np.concatenate([q_broadcast, doc_emb, z_broadcast], axis=1)
        h = np.maximum(0, x @ self.W1 + self.b1)
        scores = (h @ self.W2 + self.b2).squeeze(-1)
        return scores

    def forward_with_cache(self, x: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Forward pass with cached activations for backprop."""
        z1 = x @ self.W1 + self.b1
        h1 = np.maximum(0, z1)
        z2 = h1 @ self.W2 + self.b2
        scores = z2.squeeze(-1)
        cache = {"x": x, "z1": z1, "h1": h1}
        return scores, cache

    def backward(self, dscores: np.ndarray, cache: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Backward pass to compute gradients."""
        batch_size = dscores.shape[0]
        dz2 = dscores.reshape(-1, 1)
        dW2 = cache["h1"].T @ dz2
        db2 = dz2.sum(axis=0)
        dh1 = dz2 @ self.W2.T
        dz1 = dh1 * (cache["z1"] > 0).astype(np.float32)
        dW1 = cache["x"].T @ dz1
        db1 = dz1.sum(axis=0)
        return {"W1": dW1 / batch_size, "b1": db1 / batch_size, "W2": dW2 / batch_size, "b2": db2 / batch_size}

    def learn_from_teacher(
        self,
        query_emb: np.ndarray,
        doc_embs: np.ndarray,
        z: np.ndarray,
        teacher_scores: np.ndarray,
        margin: float = 0.5,
    ) -> float:
        """Online learning: update weights to match ONNX teacher ranking."""
        n_docs = doc_embs.shape[0]
        if n_docs < 2:
            return 0.0

        q_broadcast = np.tile(query_emb, (n_docs, 1))
        z_broadcast = np.tile(z, (n_docs, 1))
        x = np.concatenate([q_broadcast, doc_embs, z_broadcast], axis=1)
        our_scores, cache = self.forward_with_cache(x)
        teacher_order = np.argsort(-teacher_scores)

        n_pairs = min(5, n_docs // 2)
        total_loss = 0.0
        dscores = np.zeros(n_docs, dtype=np.float32)

        for i in range(n_pairs):
            pos_idx = teacher_order[i]
            neg_idx = teacher_order[-(i + 1)]
            diff = our_scores[pos_idx] - our_scores[neg_idx]
            if diff < margin:
                loss = margin - diff
                total_loss += loss
                dscores[pos_idx] -= 1.0
                dscores[neg_idx] += 1.0

        self._total_samples += n_docs
        self._cumulative_loss += total_loss
        self._recent_losses.append(total_loss)
        if len(self._recent_losses) > 200:
            self._recent_losses = self._recent_losses[-200:]

        if total_loss > 0:
            grads = self.backward(dscores, cache)
            momentum = 0.9
            self._momentum_W1 = momentum * self._momentum_W1 - self.lr * grads["W1"]
            self._momentum_b1 = momentum * self._momentum_b1 - self.lr * grads["b1"]
            self._momentum_W2 = momentum * self._momentum_W2 - self.lr * grads["W2"]
            self._momentum_b2 = momentum * self._momentum_b2 - self.lr * grads["b2"]
            self.W1 += self._momentum_W1
            self.b1 += self._momentum_b1
            self.W2 += self._momentum_W2
            self.b2 += self._momentum_b2
            self._update_count += 1
            self._update_learning_rate()

        return total_loss

    def _save_weights(self, checkpoint: bool = False):
        """Save weights to disk atomically."""
        import fcntl
        try:
            self._version += 1
            tmp_base = self._weights_path.replace(".npz", ".tmp")
            # Keep only last 200 losses for convergence (avoid unbounded growth)
            recent_losses_to_save = self._recent_losses[-200:] if self._recent_losses else []
            np.savez(
                tmp_base,
                W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2,
                momentum_W1=self._momentum_W1, momentum_b1=self._momentum_b1,
                momentum_W2=self._momentum_W2, momentum_b2=self._momentum_b2,
                update_count=self._update_count,
                total_samples=self._total_samples,
                cumulative_loss=self._cumulative_loss,
                learning_rate=self.lr,
                version=self._version,
                collection=self._collection,
                recent_losses=np.array(recent_losses_to_save, dtype=np.float32),
            )
            tmp_path = tmp_base + ".npz"
            lock_path = self._weights_path + ".lock"
            os.makedirs(os.path.dirname(lock_path) or ".", exist_ok=True)
            with open(lock_path, "w") as lock_file:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                try:
                    os.replace(tmp_path, self._weights_path)
                finally:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            if checkpoint or self._version % 100 == 0:
                self._save_checkpoint()
        except Exception:
            pass

    def _save_checkpoint(self):
        """Save a versioned checkpoint and prune old ones."""
        try:
            import shutil
            checkpoint_path = self._weights_path.replace(".npz", f"_v{self._version}.npz")
            shutil.copy2(self._weights_path, checkpoint_path)
            self._prune_old_checkpoints()
        except Exception:
            pass

    def _prune_old_checkpoints(self):
        """Remove old checkpoints keeping only the most recent MAX_CHECKPOINTS."""
        try:
            import glob
            pattern = self._weights_path.replace(".npz", "_v*.npz")
            checkpoints = sorted(glob.glob(pattern))
            if len(checkpoints) > self.MAX_CHECKPOINTS:
                for old_cp in checkpoints[:-self.MAX_CHECKPOINTS]:
                    try:
                        os.remove(old_cp)
                    except Exception:
                        pass
        except Exception:
            pass

    def _load_weights(self):
        """Load weights from disk with dimension validation."""
        from scripts.logger import get_logger
        logger = get_logger(__name__)

        data = np.load(self._weights_path, allow_pickle=True)

        def _get(key: str, default):
            return data[key] if key in data.files else default

        w1_loaded = data["W1"]
        w2_loaded = data["W2"]
        b1_loaded = data["b1"]
        b2_loaded = data["b2"]

        expected_w1 = (self.dim * 3, self.hidden_dim)
        expected_w2 = (self.hidden_dim, 1)
        expected_b1 = (self.hidden_dim,)
        expected_b2 = (1,)

        shape_ok = (
            w1_loaded.shape == expected_w1 and
            w2_loaded.shape == expected_w2 and
            b1_loaded.shape == expected_b1 and
            b2_loaded.shape == expected_b2
        )

        if not shape_ok:
            logger.warning(f"TinyScorer: shape mismatch, falling back to random init.")
            data.close()
            self._init_random_weights()
            return

        self.W1 = w1_loaded.astype(np.float32, copy=False)
        self.b1 = b1_loaded.astype(np.float32, copy=False)
        self.W2 = w2_loaded.astype(np.float32, copy=False)
        self.b2 = b2_loaded.astype(np.float32, copy=False)
        self._update_count = int(_get("update_count", 0))
        self._total_samples = int(_get("total_samples", 0))
        self._cumulative_loss = float(_get("cumulative_loss", 0.0))
        self._version = int(_get("version", 0))

        if "learning_rate" in data.files:
            self.lr = float(data["learning_rate"])

        if "momentum_W1" in data.files:
            self._momentum_W1 = data["momentum_W1"].astype(np.float32, copy=False)
            self._momentum_b1 = data["momentum_b1"].astype(np.float32, copy=False)
            self._momentum_W2 = data["momentum_W2"].astype(np.float32, copy=False)
            self._momentum_b2 = data["momentum_b2"].astype(np.float32, copy=False)
        else:
            self._momentum_W1 = np.zeros_like(self.W1)
            self._momentum_b1 = np.zeros_like(self.b1)
            self._momentum_W2 = np.zeros_like(self.W2)
            self._momentum_b2 = np.zeros_like(self.b2)

        # Restore recent losses for convergence detection (survives restarts)
        if "recent_losses" in data.files:
            self._recent_losses = list(data["recent_losses"].astype(np.float32))
        else:
            self._recent_losses = []

        self._weights_mtime = os.path.getmtime(self._weights_path)
        data.close()

    def rollback_to_checkpoint(self, version: int) -> bool:
        """Rollback to a specific checkpoint version."""
        try:
            import shutil
            checkpoint_path = self._weights_path.replace(".npz", f"_v{version}.npz")
            if os.path.exists(checkpoint_path):
                shutil.copy2(checkpoint_path, self._weights_path)
                self._load_weights()
                return True
        except Exception:
            pass
        return False
