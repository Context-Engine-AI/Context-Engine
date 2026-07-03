"""
ConfidenceEstimator - Estimates confidence to enable early stopping.

From TRM: Q-learning inspired halting - stop when improvement is minimal.
"""
import numpy as np

from scripts.rerank_recursive.state import RefinementState


class ConfidenceEstimator:
    """
    Estimates confidence to enable early stopping.

    Uses patience to avoid stopping on noisy single-step improvements.
    """

    def __init__(self, patience: int = 1, min_improvement: float = 0.01):
        self.patience = patience
        self.min_improvement = min_improvement
        self._stable_count = 0

    def reset(self):
        """Reset state for a new query."""
        self._stable_count = 0

    def should_stop(self, state: RefinementState) -> bool:
        """Check if we should stop refining based on score stability."""
        if len(state.score_history) < 2:
            return False

        prev_scores = state.score_history[-2]
        curr_scores = state.scores

        prev_order = np.argsort(-prev_scores)
        curr_order = np.argsort(-curr_scores)

        is_stable = False
        k = min(5, len(prev_order))
        if np.array_equal(prev_order[:k], curr_order[:k]):
            is_stable = True

        improvement = np.abs(curr_scores - prev_scores).mean()
        if improvement < self.min_improvement:
            is_stable = True

        if is_stable:
            self._stable_count += 1
            if self._stable_count >= self.patience:
                return True
        else:
            self._stable_count = 0

        return False
