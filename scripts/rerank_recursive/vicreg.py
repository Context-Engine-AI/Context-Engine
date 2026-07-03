"""
VICReg - Variance-Invariance-Covariance Regularization for Latent Refinement.

Adapted from VICReg (Bardes et al., 2021) for online 3-pass reranking.
"""
from typing import Dict, Tuple

import numpy as np


class VICReg:
    """
    VICReg regularization for refinement residuals.

    Regularizes the refiner's residual (z_refined - z) to have:
    - Unit variance per dimension (prevents collapse)
    - Decorrelated dimensions (prevents redundancy)
    - Bounded magnitude (stable updates)
    """

    def __init__(
        self,
        lambda_var: float = 1.0,
        lambda_cov: float = 0.04,
        lambda_inv: float = 0.1,
        var_target: float = 1.0,
    ):
        self.lambda_var = lambda_var
        self.lambda_cov = lambda_cov
        self.lambda_inv = lambda_inv
        self.var_target = var_target

    def forward(
        self, z_batch: np.ndarray, z_refined_batch: np.ndarray
    ) -> Tuple[float, np.ndarray, Dict[str, float]]:
        """
        Compute VICReg loss and gradient w.r.t. z_refined.

        Args:
            z_batch: (N, dim) original latent states
            z_refined_batch: (N, dim) refined latent states

        Returns:
            (total_loss, grad_z_refined, loss_components)
        """
        N, dim = z_batch.shape
        eps = 1e-8

        residual = z_refined_batch - z_batch
        mean_res = residual.mean(axis=0, keepdims=True)
        residual_centered = residual - mean_res

        # Variance loss
        std = residual.std(axis=0) + eps
        var_diff = self.var_target - std
        var_loss = float(np.maximum(0, var_diff).mean())
        hinge_mask = (var_diff > 0).astype(np.float32)
        d_var = -hinge_mask[None, :] * residual_centered / (N * std[None, :] * dim)

        # Covariance loss
        cov = (residual_centered.T @ residual_centered) / (N - 1 + eps)
        off_diag_mask = 1.0 - np.eye(dim, dtype=np.float32)
        off_diag = cov * off_diag_mask
        cov_loss = float((off_diag ** 2).sum() / dim)
        d_cov = 4 * residual_centered @ (off_diag * off_diag_mask) / ((N - 1 + eps) * dim)

        # Invariance loss
        inv_loss = float((residual ** 2).mean())
        d_inv = 2 * residual / (N * dim)

        # Total
        total_loss = self.lambda_var * var_loss + self.lambda_cov * cov_loss + self.lambda_inv * inv_loss
        grad = (self.lambda_var * d_var + self.lambda_cov * d_cov + self.lambda_inv * d_inv).astype(np.float32)

        components = {"var_loss": var_loss, "cov_loss": cov_loss, "inv_loss": inv_loss}
        return total_loss, grad, components
