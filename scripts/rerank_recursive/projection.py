"""
LearnedProjection - Learnable linear projection from embedding dim to working dim.
"""
import os
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np


class LearnedProjection:
    """
    Learnable linear projection from embedding dim to working dim.

    Replaces fixed random projection with a learnable layer that adapts
    to domain-specific semantics.
    """

    WEIGHTS_DIR = os.environ.get("RERANKER_WEIGHTS_DIR", "/tmp/rerank_weights")
    WEIGHTS_RELOAD_INTERVAL = float(os.environ.get("RERANKER_WEIGHTS_RELOAD_INTERVAL", "60"))

    def __init__(self, input_dim: int = 768, output_dim: int = 256, lr: float = 0.0005):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.base_lr = lr
        self.lr = lr
        self._collection = "default"
        self._weights_path = self._get_weights_path("default")
        self._weights_mtime = 0.0
        self._last_reload_check = 0.0
        self._weights_loaded = False

        self._update_count = 0
        self._version = 0
        self._momentum_W: Optional[np.ndarray] = None
        self._momentum = 0.9

        if os.path.exists(self._weights_path):
            try:
                self._load_weights()
                return
            except Exception as e:
                from scripts.logger import get_logger
                get_logger(__name__).warning(f"LearnedProjection: failed to load {self._weights_path}: {e}")

        self._init_random_weights()

    @staticmethod
    def _sanitize_collection(collection: str) -> str:
        return "".join(c if c.isalnum() or c in "-_" else "_" for c in collection)

    def _get_weights_path(self, collection: str) -> str:
        safe_name = self._sanitize_collection(collection)
        return os.path.join(self.WEIGHTS_DIR, f"projection_{safe_name}.npz")

    def _init_random_weights(self):
        scale = np.sqrt(2.0 / (self.input_dim + self.output_dim))
        rng = np.random.RandomState(44)
        self.W = (rng.randn(self.input_dim, self.output_dim) * scale).astype(np.float32)
        self._momentum_W = np.zeros_like(self.W)

    def init_from_pca(self, embeddings: np.ndarray):
        """Initialize projection using PCA on corpus embeddings.

        Provides a much better cold-start than random initialization by using
        the principal components of the actual data distribution.

        Args:
            embeddings: (n_samples, input_dim) array of corpus embeddings
        """
        if embeddings.shape[0] < self.output_dim:
            from scripts.logger import get_logger
            get_logger(__name__).warning(
                f"LearnedProjection: {embeddings.shape[0]} samples < {self.output_dim} dims, using random init"
            )
            self._init_random_weights()
            return

        # Center the data
        mean = embeddings.mean(axis=0, keepdims=True)
        centered = embeddings - mean

        # Use SVD for numerical stability
        # X = U @ S @ Vt, where Vt rows are principal components
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)

        # Take top output_dim principal components
        # Vt is (min(n_samples, input_dim), input_dim)
        components = Vt[:self.output_dim, :].T  # (input_dim, output_dim)

        self.W = components.astype(np.float32)
        self._momentum_W = np.zeros_like(self.W)

        # Compute explained variance for logging
        total_var = np.sum(S ** 2)
        explained_var = np.sum(S[:self.output_dim] ** 2) / total_var if total_var > 0 else 0

        from scripts.logger import get_logger
        get_logger(__name__).info(
            f"LearnedProjection: PCA init on {embeddings.shape[0]} samples "
            f"({self.input_dim}→{self.output_dim}), explained variance: {explained_var:.1%}"
        )

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
                    self._load_weights()
        except Exception:
            pass

    def _load_weights(self):
        import fcntl
        lock_path = self._weights_path + ".lock"
        try:
            os.makedirs(os.path.dirname(lock_path) or ".", exist_ok=True)
            with open(lock_path, "w") as lock_file:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_SH)
                data = np.load(self._weights_path)
                self.W = data["W"].astype(np.float32)
                self._version = int(data.get("version", 0))
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            self._weights_mtime = os.path.getmtime(self._weights_path)
            self._weights_loaded = True
            self._momentum_W = np.zeros_like(self.W)
        except Exception as e:
            from scripts.logger import get_logger
            get_logger(__name__).warning(f"LearnedProjection: load failed: {e}")

    def _save_weights(self):
        import fcntl
        os.makedirs(os.path.dirname(self._weights_path) or ".", exist_ok=True)
        lock_path = self._weights_path + ".lock"
        base_path = self._weights_path.rsplit(".npz", 1)[0]
        tmp_path = base_path + ".tmp.npz"
        try:
            with open(lock_path, "w") as lock_file:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                np.savez(tmp_path, W=self.W, version=self._version)
                os.replace(tmp_path, self._weights_path)
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            self._weights_mtime = os.path.getmtime(self._weights_path)
        except Exception as e:
            from scripts.logger import get_logger
            get_logger(__name__).warning(f"LearnedProjection: save failed: {e}")

    def forward(self, embeddings: np.ndarray) -> np.ndarray:
        """Project embeddings to output dim (normalized)."""
        self.maybe_reload_weights()  # Hot-reload from worker updates
        squeeze = embeddings.ndim == 1
        if squeeze:
            embeddings = embeddings.reshape(1, -1)
        projected = embeddings @ self.W
        norms = np.linalg.norm(projected, axis=-1, keepdims=True) + 1e-8
        projected = projected / norms
        if squeeze:
            projected = projected[0]
        return projected

    def forward_with_cache(self, embeddings: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Forward pass with cache for backprop."""
        squeeze = embeddings.ndim == 1
        if squeeze:
            embeddings = embeddings.reshape(1, -1)
        pre_norm = embeddings @ self.W
        norms = np.linalg.norm(pre_norm, axis=-1, keepdims=True) + 1e-8
        projected = pre_norm / norms
        cache = {"input": embeddings, "pre_norm": pre_norm, "norms": norms, "projected": projected}
        if squeeze:
            projected = projected[0]
        return projected, cache

    def backward(self, grad_output: np.ndarray, cache: Dict[str, Any], weight: float = 1.0):
        """Backprop gradient through projection and update weights."""
        if grad_output.ndim == 1:
            grad_output = grad_output.reshape(1, -1)
        embeddings = cache["input"]
        norms = cache["norms"]
        batch_size = embeddings.shape[0]
        projected = cache["projected"]
        dot = np.sum(grad_output * projected, axis=-1, keepdims=True)
        grad_pre_norm = (grad_output - projected * dot) / norms
        dW = embeddings.T @ grad_pre_norm / batch_size
        dW = dW * weight
        self._momentum_W = self._momentum * self._momentum_W + dW
        self.W -= self.lr * self._momentum_W
        self._update_count += 1
        if self._update_count % 100 == 0:
            self._version += 1
            self._save_weights()
