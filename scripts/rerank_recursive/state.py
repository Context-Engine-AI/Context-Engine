"""
RefinementState dataclass - carries latent state between refinement iterations.
"""
from dataclasses import dataclass, field
from typing import List
import numpy as np


@dataclass
class RefinementState:
    """Carries latent state between refinement iterations."""
    z: np.ndarray  # Latent representation (query understanding)
    scores: np.ndarray  # Current score estimates
    iteration: int = 0
    confidence: float = 0.0  # For early stopping

    # Track per-iteration improvements for analysis
    score_history: List[np.ndarray] = field(default_factory=list)
