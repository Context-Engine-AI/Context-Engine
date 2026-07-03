"""
LatentRefiner - Refines latent state based on current ranking results.

From TRM paper: z encodes "what we've learned about the query so far"
and gets updated based on the current answer (scores).
"""
import os
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np


class LatentRefiner:
    """
    Refines the latent state z based on current results.

    Supports:
    - Per-collection weight persistence
    - Hot-reload from background worker updates
    - Online learning via learn_from_teacher()
    """

    WEIGHTS_DIR = os.environ.get("RERANKER_WEIGHTS_DIR", "/tmp/rerank_weights")
    WEIGHTS_RELOAD_INTERVAL = float(os.environ.get("RERANKER_WEIGHTS_RELOAD_INTERVAL", "60"))

    def __init__(self, dim: int = 256, hidden_dim: int = 256, lr: float = 0.001):
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.base_lr = lr
        self.lr = lr
        self._collection = "default"
        self._weights_path = self._get_weights_path("default")
        self._weights_mtime = 0.0
        self._last_reload_check = 0.0
        self._weights_loaded = False

        self._update_count = 0
        self._version = 0

        self._momentum_W1: Optional[np.ndarray] = None
        self._momentum_b1: Optional[np.ndarray] = None
        self._momentum_W2: Optional[np.ndarray] = None
        self._momentum_b2: Optional[np.ndarray] = None

        if os.path.exists(self._weights_path):
            try:
                self._load_weights()
                return
            except Exception as e:
                from scripts.logger import get_logger
                get_logger(__name__).warning(f"LatentRefiner: failed to load {self._weights_path}: {e}")

        self._init_random_weights()

    @staticmethod
    def _sanitize_collection(collection: str) -> str:
        return "".join(c if c.isalnum() or c in "-_" else "_" for c in collection)

    def _get_weights_path(self, collection: str) -> str:
        safe_name = self._sanitize_collection(collection)
        return os.path.join(self.WEIGHTS_DIR, f"refiner_{safe_name}.npz")

    def set_collection(self, collection: str):
        self._collection = collection
        self._weights_path = self._get_weights_path(collection)
        if os.path.exists(self._weights_path):
            try:
                self._load_weights()
            except Exception:
                pass

    def maybe_reload_weights(self):
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

    def _init_random_weights(self):
        rng = np.random.RandomState(43)
        scale = np.float32(np.sqrt(2.0 / (self.dim * 3)))
        self.W1 = rng.randn(self.dim * 3, self.hidden_dim).astype(np.float32) * scale
        self.b1 = np.zeros(self.hidden_dim, dtype=np.float32)
        w2_scale = np.float32(np.sqrt(2.0 / self.hidden_dim))
        self.W2 = rng.randn(self.hidden_dim, self.dim).astype(np.float32) * w2_scale
        self.b2 = np.zeros(self.dim, dtype=np.float32)

        self._momentum_W1 = np.zeros_like(self.W1)
        self._momentum_b1 = np.zeros_like(self.b1)
        self._momentum_W2 = np.zeros_like(self.W2)
        self._momentum_b2 = np.zeros_like(self.b2)

    def _load_weights(self) -> bool:
        from scripts.logger import get_logger
        logger = get_logger(__name__)
        try:
            data = np.load(self._weights_path, allow_pickle=True)

            def _get(key: str, default):
                return data[key] if key in data.files else default

            w1 = _get("W1", None)
            w2 = _get("W2", None)
            b1 = _get("b1", None)
            b2 = _get("b2", None)

            if w1 is None or w2 is None:
                data.close()
                return False

            expected_w1 = (self.dim * 3, self.hidden_dim)
            expected_w2 = (self.hidden_dim, self.dim)

            if w1.shape != expected_w1 or w2.shape != expected_w2:
                logger.warning(f"LatentRefiner: shape mismatch")
                data.close()
                return False

            self.W1 = w1.astype(np.float32, copy=False)
            self.b1 = b1.astype(np.float32, copy=False) if b1 is not None else np.zeros(self.hidden_dim, dtype=np.float32)
            self.W2 = w2.astype(np.float32, copy=False)
            self.b2 = b2.astype(np.float32, copy=False) if b2 is not None else np.zeros(self.dim, dtype=np.float32)
            self._update_count = int(_get("update_count", 0))
            self._version = int(_get("version", 0))

            if self._momentum_W1 is None or self._momentum_W1.shape != self.W1.shape:
                self._momentum_W1 = np.zeros_like(self.W1)
                self._momentum_b1 = np.zeros_like(self.b1)
                self._momentum_W2 = np.zeros_like(self.W2)
                self._momentum_b2 = np.zeros_like(self.b2)

            self._weights_loaded = True
            self._weights_mtime = os.path.getmtime(self._weights_path)
            data.close()
            return True
        except Exception as e:
            logger.warning(f"LatentRefiner: failed to load weights: {e}")
            return False

    def refine(
        self,
        z: np.ndarray,
        query_emb: np.ndarray,
        doc_embs: np.ndarray,
        scores: np.ndarray,
        alpha: float = 0.5
    ) -> np.ndarray:
        """Refine latent state based on current ranking."""
        self.maybe_reload_weights()

        weights = np.exp(scores - scores.max())
        weights = weights / (weights.sum() + 1e-8)
        doc_summary = (weights[:, None] * doc_embs).sum(axis=0)
        x = np.concatenate([z, query_emb, doc_summary])
        h = np.maximum(0, x @ self.W1 + self.b1)
        z_new = h @ self.W2 + self.b2
        z_refined = alpha * z_new + (1 - alpha) * z
        z_refined = z_refined / (np.linalg.norm(z_refined) + 1e-8)
        return z_refined

    def refine_with_cache(
        self,
        z: np.ndarray,
        query_emb: np.ndarray,
        doc_embs: np.ndarray,
        scores: np.ndarray,
        alpha: float = 0.5
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Refine with cache for backprop."""
        weights = np.exp(scores - scores.max())
        weights = weights / (weights.sum() + 1e-8)
        doc_summary = (weights[:, None] * doc_embs).sum(axis=0)
        x = np.concatenate([z, query_emb, doc_summary])
        h = np.maximum(0, x @ self.W1 + self.b1)
        z_new = h @ self.W2 + self.b2
        z_refined = alpha * z_new + (1 - alpha) * z
        z_refined = z_refined / (np.linalg.norm(z_refined) + 1e-8)
        cache = {"x": x, "h": h, "z": z, "z_new": z_new, "alpha": alpha, "weights": weights}
        return z_refined, cache

    def learn_from_teacher(
        self,
        z: np.ndarray,
        query_emb: np.ndarray,
        doc_embs: np.ndarray,
        scores: np.ndarray,
        teacher_z: np.ndarray,
    ) -> float:
        """Online learning: update weights so refined z moves toward teacher_z."""
        z_refined, cache = self.refine_with_cache(z, query_emb, doc_embs, scores)
        diff = z_refined - teacher_z
        loss = float(np.sum(diff ** 2))

        if loss < 1e-8:
            return 0.0

        dz_refined = 2.0 * diff
        dz_new = cache["alpha"] * dz_refined
        dW2 = np.outer(cache["h"], dz_new)
        db2 = dz_new
        dh = dz_new @ self.W2.T
        dh = dh * (cache["h"] > 0).astype(np.float32)
        dW1 = np.outer(cache["x"], dh)
        db1 = dh

        momentum = 0.9
        if self._momentum_W1 is None:
            self._momentum_W1 = np.zeros_like(self.W1)
            self._momentum_b1 = np.zeros_like(self.b1)
            self._momentum_W2 = np.zeros_like(self.W2)
            self._momentum_b2 = np.zeros_like(self.b2)

        self._momentum_W1 = momentum * self._momentum_W1 - self.lr * dW1
        self._momentum_b1 = momentum * self._momentum_b1 - self.lr * db1
        self._momentum_W2 = momentum * self._momentum_W2 - self.lr * dW2
        self._momentum_b2 = momentum * self._momentum_b2 - self.lr * db2

        self.W1 += self._momentum_W1
        self.b1 += self._momentum_b1
        self.W2 += self._momentum_W2
        self.b2 += self._momentum_b2
        self._update_count += 1
        return loss

    def learn_from_teacher_with_cache(
        self,
        z: np.ndarray,
        query_emb: np.ndarray,
        doc_embs: np.ndarray,
        scores: np.ndarray,
        teacher_z: np.ndarray,
    ) -> Tuple[float, np.ndarray, np.ndarray, Dict[str, Any]]:
        """Online learning with cache for VICReg backprop."""
        z_refined, cache = self.refine_with_cache(z, query_emb, doc_embs, scores)
        diff = z_refined - teacher_z
        loss = float(np.sum(diff ** 2))

        if loss >= 1e-8:
            dz_refined = 2.0 * diff
            dz_new = cache["alpha"] * dz_refined
            dW2 = np.outer(cache["h"], dz_new)
            db2 = dz_new
            dh = dz_new @ self.W2.T
            dh = dh * (cache["h"] > 0).astype(np.float32)
            dW1 = np.outer(cache["x"], dh)
            db1 = dh

            momentum = 0.9
            if self._momentum_W1 is None:
                self._momentum_W1 = np.zeros_like(self.W1)
                self._momentum_b1 = np.zeros_like(self.b1)
                self._momentum_W2 = np.zeros_like(self.W2)
                self._momentum_b2 = np.zeros_like(self.b2)

            self._momentum_W1 = momentum * self._momentum_W1 - self.lr * dW1
            self._momentum_b1 = momentum * self._momentum_b1 - self.lr * db1
            self._momentum_W2 = momentum * self._momentum_W2 - self.lr * dW2
            self._momentum_b2 = momentum * self._momentum_b2 - self.lr * db2

            self.W1 += self._momentum_W1
            self.b1 += self._momentum_b1
            self.W2 += self._momentum_W2
            self.b2 += self._momentum_b2
            self._update_count += 1

        return loss, z, z_refined, cache

    def apply_vicreg_gradient(self, grad_z_refined: np.ndarray, cache: Dict[str, Any], weight: float = 0.1):
        """Apply VICReg gradient to refiner weights."""
        dz_new = cache["alpha"] * grad_z_refined * weight
        dW2 = np.outer(cache["h"], dz_new)
        db2 = dz_new
        dh = dz_new @ self.W2.T
        dh = dh * (cache["h"] > 0).astype(np.float32)
        dW1 = np.outer(cache["x"], dh)
        db1 = dh

        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    def _save_weights(self, checkpoint: bool = False):
        """Save weights to disk atomically."""
        import fcntl
        os.makedirs(self.WEIGHTS_DIR, exist_ok=True)
        self._version += 1

        tmp_base = self._weights_path.replace(".npz", ".tmp")
        tmp_path = tmp_base + ".npz"
        lock_path = self._weights_path + ".lock"

        try:
            with open(lock_path, "w") as lock_file:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                try:
                    np.savez(
                        tmp_base,
                        W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2,
                        update_count=self._update_count,
                        version=self._version,
                        dim=self.dim,
                        hidden_dim=self.hidden_dim,
                    )
                    os.replace(tmp_path, self._weights_path)
                finally:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        except Exception:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
            raise
