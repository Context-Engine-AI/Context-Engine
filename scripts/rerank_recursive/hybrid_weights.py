"""
LearnedHybridWeights - Learns optimal dense vs. lexical balance per-collection.
"""
import os

import numpy as np


class LearnedHybridWeights:
    """Learns optimal dense vs. lexical balance per-collection."""

    WEIGHTS_DIR = os.environ.get("RERANKER_WEIGHTS_DIR", "/tmp/rerank_weights")

    def __init__(self, lr: float = 0.01):
        self.lr = lr
        self._collection = "default"
        self._weights_path = self._get_weights_path("default")
        self.alpha = 0.0
        self._momentum_alpha = 0.0
        self._momentum = 0.9
        self._update_count = 0
        self._version = 0

        if os.path.exists(self._weights_path):
            try:
                self._load_weights()
            except Exception:
                pass

    @staticmethod
    def _sanitize_collection(collection: str) -> str:
        return "".join(c if c.isalnum() or c in "-_" else "_" for c in collection)

    def _get_weights_path(self, collection: str) -> str:
        safe_name = self._sanitize_collection(collection)
        return os.path.join(self.WEIGHTS_DIR, f"hybrid_{safe_name}.npz")

    def set_collection(self, collection: str):
        self._collection = collection
        self._weights_path = self._get_weights_path(collection)
        if os.path.exists(self._weights_path):
            try:
                self._load_weights()
            except Exception:
                pass

    def _load_weights(self):
        import fcntl
        lock_path = self._weights_path + ".lock"
        os.makedirs(os.path.dirname(lock_path) or ".", exist_ok=True)
        with open(lock_path, "w") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_SH)
            data = np.load(self._weights_path)
            self.alpha = float(data["alpha"])
            self._version = int(data.get("version", 0))
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _save_weights(self):
        import fcntl
        os.makedirs(os.path.dirname(self._weights_path) or ".", exist_ok=True)
        lock_path = self._weights_path + ".lock"
        base_path = self._weights_path.rsplit(".npz", 1)[0]
        tmp_base = base_path + ".tmp"
        tmp_path = tmp_base + ".npz"
        with open(lock_path, "w") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            np.savez(tmp_base, alpha=self.alpha, version=self._version)
            os.replace(tmp_path, self._weights_path)
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    @property
    def dense_weight(self) -> float:
        return 1.0 / (1.0 + np.exp(-self.alpha))

    @property
    def lexical_weight(self) -> float:
        return 1.0 - self.dense_weight

    def blend(self, dense_scores: np.ndarray, lexical_scores: np.ndarray) -> np.ndarray:
        w = self.dense_weight
        return w * dense_scores + (1 - w) * lexical_scores

    def learn_from_teacher(
        self,
        dense_scores: np.ndarray,
        lexical_scores: np.ndarray,
        teacher_scores: np.ndarray,
    ):
        w = self.dense_weight
        blended = self.blend(dense_scores, lexical_scores)
        teacher_norm = (teacher_scores - teacher_scores.mean()) / (teacher_scores.std() + 1e-8)
        blended_norm = (blended - blended.mean()) / (blended.std() + 1e-8)
        dense_norm = (dense_scores - dense_scores.mean()) / (dense_scores.std() + 1e-8)
        lexical_norm = (lexical_scores - lexical_scores.mean()) / (lexical_scores.std() + 1e-8)
        error = teacher_norm - blended_norm
        modality_diff = dense_norm - lexical_norm
        sigmoid_grad = w * (1 - w)
        grad = (error * modality_diff).mean() * sigmoid_grad
        self._momentum_alpha = self._momentum * self._momentum_alpha + grad
        self.alpha += self.lr * self._momentum_alpha
        self.alpha = np.clip(self.alpha, -5.0, 5.0)
        self._update_count += 1
        if self._update_count % 50 == 0:
            self._version += 1
            self._save_weights()
