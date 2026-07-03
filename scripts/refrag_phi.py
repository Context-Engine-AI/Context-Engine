"""
Projection φ utilities for decoder-path ReFRAG.

- File format: JSON 2D array [[...], [...], ...] with shape (d_in, d_model)
- Operation: y = x @ φ  (x: length d_in)
- No numpy dependency (pure Python) for portability.
"""

from __future__ import annotations
import json
from typing import List

Matrix = List[List[float]]


def load_phi_matrix(path: str) -> Matrix:
    with open(path, "r", encoding="utf-8") as f:
        m = json.load(f)
    if not isinstance(m, list) or not m or not isinstance(m[0], list):
        raise ValueError("φ must be a 2D JSON array")
    # Validate rectangular shape
    cols = len(m[0])
    for row in m:
        if len(row) != cols:
            raise ValueError("φ must be rectangular (consistent number of columns)")
    return m  # shape (d_in, d_model)


def project(vec: List[float], phi: Matrix) -> List[float]:
    d_in = len(phi)
    if len(vec) != d_in:
        raise ValueError(f"vec length {len(vec)} != φ rows {d_in}")
    d_model = len(phi[0])
    out = [0.0] * d_model
    # y_j = sum_i x_i * φ_{i,j}
    for i in range(d_in):
        xi = float(vec[i])
        row = phi[i]
        for j in range(d_model):
            out[j] += xi * float(row[j])
    return out


def project_batch(vectors: List[List[float]], phi: Matrix) -> List[List[float]]:
    return [project(v, phi) for v in vectors]
