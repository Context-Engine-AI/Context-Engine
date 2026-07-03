"""
Alpha Scheduler - Learnable and scheduled alpha for score blending.

From TRM paper insight: the blending factor α between new scores and previous
scores should vary per iteration. Early iterations need more exploration (higher α),
later iterations need more exploitation (lower α).

Two strategies:
1. CosineAlphaScheduler: Fixed cosine decay schedule
2. LearnedAlphaWeights: Per-iteration learnable weights with persistence
"""
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class CosineAlphaScheduler:
    """Fixed cosine schedule for alpha values.
    
    α = alpha_min + (alpha_max - alpha_min) * (1 + cos(π * i / (n-1))) / 2
    
    This gives higher α early (trust new scores more) and lower α later
    (rely more on refined estimates).
    """
    
    def __init__(
        self,
        n_iterations: int = 3,
        alpha_max: float = 0.7,
        alpha_min: float = 0.3,
    ):
        self.n_iterations = max(1, n_iterations)
        self.alpha_max = alpha_max
        self.alpha_min = alpha_min
        # Pre-compute schedule
        self._schedule = self._compute_schedule()
    
    def _compute_schedule(self) -> List[float]:
        """Compute the full schedule."""
        if self.n_iterations == 1:
            return [(self.alpha_max + self.alpha_min) / 2]
        
        schedule = []
        for i in range(self.n_iterations):
            # Cosine decay from alpha_max to alpha_min
            progress = i / (self.n_iterations - 1)
            alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * (1 + np.cos(np.pi * progress)) / 2
            schedule.append(float(alpha))
        return schedule
    
    def get_alpha(self, iteration: int) -> float:
        """Get alpha for a specific iteration (0-indexed)."""
        idx = max(0, min(iteration, len(self._schedule) - 1))
        return self._schedule[idx]
    
    def get_schedule(self) -> List[float]:
        """Get the full schedule."""
        return self._schedule.copy()


class LearnedAlphaWeights:
    """Learnable per-iteration alpha weights with persistence.
    
    Uses sigmoid(raw_weight) to keep alpha in (0, 1).
    Learns optimal blending through gradient descent on ranking loss.
    """
    
    WEIGHTS_DIR = os.environ.get("RERANKER_WEIGHTS_DIR", "/tmp/rerank_weights")
    WEIGHTS_RELOAD_INTERVAL = float(os.environ.get("RERANKER_WEIGHTS_RELOAD_INTERVAL", "60"))
    
    def __init__(
        self,
        n_iterations: int = 3,
        init_alpha: float = 0.5,
        lr: float = 0.01,
    ):
        self.n_iterations = max(1, n_iterations)
        self.lr = lr
        self._collection = "default"
        self._weights_path = self._get_weights_path("default")
        self._weights_mtime = 0.0
        self._last_reload_check = 0.0
        self._weights_loaded = False
        self._update_count = 0
        self._version = 0
        
        # Initialize raw weights such that sigmoid(raw) = init_alpha
        # sigmoid(x) = init_alpha => x = logit(init_alpha)
        init_raw = np.log(init_alpha / (1 - init_alpha + 1e-8))
        self.raw_weights = np.full(self.n_iterations, init_raw, dtype=np.float32)
        self._momentum = np.zeros_like(self.raw_weights)
        
        # Try to load saved weights
        if os.path.exists(self._weights_path):
            try:
                self._load_weights()
            except Exception as e:
                from scripts.logger import get_logger
                get_logger(__name__).warning(f"LearnedAlphaWeights: failed to load: {e}")
    
    @staticmethod
    def _sanitize_collection(collection: str) -> str:
        return "".join(c if c.isalnum() or c in "-_" else "_" for c in collection)
    
    def _get_weights_path(self, collection: str) -> str:
        safe_name = self._sanitize_collection(collection)
        return os.path.join(self.WEIGHTS_DIR, f"alpha_{safe_name}.npz")
    
    def set_collection(self, collection: str):
        """Set collection and load corresponding weights."""
        self._collection = collection
        self._weights_path = self._get_weights_path(collection)
        if os.path.exists(self._weights_path):
            try:
                self._load_weights()
            except Exception:
                pass
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid."""
        return np.where(
            x >= 0,
            1 / (1 + np.exp(-x)),
            np.exp(x) / (1 + np.exp(x))
        )
    
    def _sigmoid_grad(self, sigmoid_val: np.ndarray) -> np.ndarray:
        """Gradient of sigmoid: σ(x) * (1 - σ(x))."""
        return sigmoid_val * (1 - sigmoid_val)
    
    def get_alpha(self, iteration: int) -> float:
        """Get alpha for a specific iteration (0-indexed)."""
        self.maybe_reload_weights()
        idx = max(0, min(iteration, len(self.raw_weights) - 1))
        return float(self._sigmoid(self.raw_weights[idx]))
    
    def get_schedule(self) -> List[float]:
        """Get alpha values for all iterations."""
        return [float(a) for a in self._sigmoid(self.raw_weights)]
    
    def maybe_reload_weights(self):
        """Hot-reload weights if changed on disk."""
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
    
    def learn_from_ranking_loss(
        self,
        iteration: int,
        blended_scores: np.ndarray,
        teacher_scores: np.ndarray,
        new_scores: Optional[np.ndarray] = None,
        prev_scores: Optional[np.ndarray] = None,
    ) -> float:
        """Learn alpha to minimize ranking difference with teacher.
        
        Uses pairwise ranking loss: if teacher prefers A over B, reduce loss
        when our blended scores also prefer A over B.
        
        Args:
            iteration: Which iteration's alpha to update
            blended_scores: (n_docs,) our blended scores at this iteration
            teacher_scores: (n_docs,) ground truth scores
            new_scores: (n_docs,) optional - the "new" signal before blending
            prev_scores: (n_docs,) optional - the "previous" signal before blending
            
        Returns:
            Gradient magnitude (for logging)
        """
        if len(blended_scores) < 2:
            return 0.0
        
        idx = max(0, min(iteration, len(self.raw_weights) - 1))
        alpha = self._sigmoid(self.raw_weights[idx:idx+1])[0]
        
        # Compute ranking agreement
        our_order = np.argsort(-blended_scores)
        teacher_order = np.argsort(-teacher_scores)
        
        # Count top-k mismatches
        k = min(3, len(blended_scores))
        n_mismatches = np.sum(our_order[:k] != teacher_order[:k])
        
        if n_mismatches == 0:
            return 0.0  # Perfect match, no update needed
        
        # Determine gradient direction based on which signal would have helped
        grad_direction = 1.0  # Default: increase alpha (trust new scores more)
        
        if new_scores is not None and prev_scores is not None:
            # Compare: would higher alpha (more new_scores) or lower alpha (more prev_scores) help?
            new_order = np.argsort(-new_scores)
            prev_order = np.argsort(-prev_scores)
            
            # Count how many top-k matches each signal has with teacher
            new_matches = np.sum(new_order[:k] == teacher_order[:k])
            prev_matches = np.sum(prev_order[:k] == teacher_order[:k])
            
            if prev_matches > new_matches:
                # Previous scores are better aligned with teacher → decrease alpha
                grad_direction = -1.0
            elif new_matches > prev_matches:
                # New scores are better aligned with teacher → increase alpha
                grad_direction = 1.0
            else:
                # Tie: small nudge toward middle (alpha=0.5)
                grad_direction = 0.5 - alpha  # Pull toward 0.5
        
        # Gradient through sigmoid, scaled by mismatches and direction
        sigmoid_grad = self._sigmoid_grad(np.array([alpha]))[0]
        grad = n_mismatches * 0.1 * sigmoid_grad * grad_direction
        
        # Momentum SGD
        momentum = 0.9
        self._momentum[idx] = momentum * self._momentum[idx] + grad
        self.raw_weights[idx] -= self.lr * self._momentum[idx]
        
        # Clamp to reasonable range (alpha between ~0.1 and ~0.9)
        self.raw_weights[idx] = np.clip(self.raw_weights[idx], -2.2, 2.2)
        
        self._update_count += 1
        if self._update_count % 50 == 0:
            self._save_weights()
        
        return abs(grad)
    
    def _load_weights(self):
        """Load weights from disk."""
        import fcntl
        lock_path = self._weights_path + ".lock"
        try:
            os.makedirs(os.path.dirname(lock_path) or ".", exist_ok=True)
            with open(lock_path, "w") as lock_file:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_SH)
                data = np.load(self._weights_path)
                loaded_weights = data["raw_weights"]
                # Handle dimension mismatch gracefully
                if len(loaded_weights) == len(self.raw_weights):
                    self.raw_weights = loaded_weights.astype(np.float32)
                else:
                    # Resize: copy what we can, init rest
                    min_len = min(len(loaded_weights), len(self.raw_weights))
                    self.raw_weights[:min_len] = loaded_weights[:min_len].astype(np.float32)
                self._version = int(data.get("version", 0))
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            self._weights_mtime = os.path.getmtime(self._weights_path)
            self._weights_loaded = True
            self._momentum = np.zeros_like(self.raw_weights)
        except Exception as e:
            from scripts.logger import get_logger
            get_logger(__name__).warning(f"LearnedAlphaWeights: load failed: {e}")
    
    def _save_weights(self):
        """Save weights atomically."""
        import fcntl
        os.makedirs(self.WEIGHTS_DIR, exist_ok=True)
        lock_path = self._weights_path + ".lock"
        tmp_path = self._weights_path.replace(".npz", ".tmp.npz")
        self._version += 1
        try:
            with open(lock_path, "w") as lock_file:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                np.savez(
                    tmp_path,
                    raw_weights=self.raw_weights,
                    version=self._version,
                    n_iterations=self.n_iterations,
                )
                os.replace(tmp_path, self._weights_path)
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            self._weights_mtime = os.path.getmtime(self._weights_path)
        except Exception as e:
            from scripts.logger import get_logger
            get_logger(__name__).warning(f"LearnedAlphaWeights: save failed: {e}")
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics for logging."""
        return {
            "collection": self._collection,
            "version": self._version,
            "update_count": self._update_count,
            "alphas": self.get_schedule(),
        }
