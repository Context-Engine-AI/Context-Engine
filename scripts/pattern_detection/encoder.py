"""
Pattern Encoder - Convert pattern signatures to dense vectors.

Encoding dimensions (64 total):
- MinHash n-grams: 16-dim
- Weisfeiler-Lehman kernel: 8-dim
- Control flow features: 16-dim
- CFG fingerprint: 8-dim
- SimHash bits: 8-dim
- Spectral features: 8-dim
"""

from typing import List, Dict, Tuple, Optional, Any
from collections import Counter
import hashlib
import math

from .extractor import PatternSignature


class PatternEncoder:
    """Encode pattern signatures as dense vectors for similarity search."""

    TOTAL_DIM = 64
    MINHASH_DIM = 16
    WL_DIM = 8
    CONTROL_DIM = 16
    CFG_DIM = 8
    SIMHASH_DIM = 8
    SPECTRAL_DIM = 8

    HASH_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]

    def __init__(self, normalize: bool = True):
        self.normalize = normalize

    def encode(self, signature: PatternSignature) -> List[float]:
        """Encode pattern signature as 64-dim vector."""
        vector = []

        vector.extend(self._encode_minhash(signature.structural_ngrams))
        vector.extend(self._encode_wl_kernel(signature.wl_labels))
        vector.extend(self._encode_control_flow(signature.control_flow))
        vector.extend(self._encode_cfg(signature.cfg_nodes, signature.cfg_edges))
        vector.extend(self._encode_simhash(signature.simhash))
        vector.extend(self._encode_spectral(signature.spectral_features))

        if self.normalize:
            vector = self._l2_normalize(vector)

        return vector

    # =========================================================================
    # MinHash for structural n-grams (Jaccard similarity)
    # =========================================================================

    def _encode_minhash(self, ngrams: Counter) -> List[float]:
        """MinHash encoding for Jaccard similarity preservation."""
        if not ngrams:
            return [0.0] * self.MINHASH_DIM

        ngram_set = set(ngrams.keys())
        minhash = []

        for i, prime in enumerate(self.HASH_PRIMES[:self.MINHASH_DIM]):
            min_h = float('inf')
            for ngram in ngram_set:
                h = self._hash_item(str(ngram), prime, i)
                min_h = min(min_h, h)
            minhash.append((min_h % 1000) / 1000.0 if min_h != float('inf') else 0.0)

        return minhash

    def _hash_item(self, item: str, prime: int, seed: int) -> int:
        h = hashlib.md5(item.encode()).digest()
        return (int.from_bytes(h[:4], 'big') * prime + seed) % (2**31)

    # =========================================================================
    # Weisfeiler-Lehman Graph Kernel encoding
    # =========================================================================

    def _encode_wl_kernel(self, wl_labels: Dict[int, List[str]]) -> List[float]:
        """Encode WL labels into fixed-size vector via feature hashing."""
        vec = [0.0] * self.WL_DIM

        if not wl_labels:
            return vec

        label_counts = Counter()
        for node_id, labels in wl_labels.items():
            for label in labels:
                label_counts[label] += 1

        total = sum(label_counts.values())
        if total == 0:
            return vec

        for label, count in label_counts.items():
            bucket = int(hashlib.md5(label.encode()).hexdigest()[:4], 16) % self.WL_DIM
            vec[bucket] += count / total

        return vec

    # =========================================================================
    # Control Flow encoding
    # =========================================================================

    def _encode_control_flow(self, cf: Dict[str, Any]) -> List[float]:
        """Encode control flow features (16-dim)."""
        vec = [0.0] * self.CONTROL_DIM

        vec[0] = self._log_scale(cf.get("max_loop_depth", 0), 5)
        vec[1] = self._log_scale(cf.get("loop_count", 0), 10)
        vec[2] = 1.0 if cf.get("loop_count", 0) >= 1 else 0.0
        vec[3] = 1.0 if cf.get("try_in_loop", False) else 0.0  # retry pattern

        vec[4] = self._log_scale(cf.get("branch_count", 0), 20)
        vec[5] = 1.0 if cf.get("branch_count", 0) >= 1 else 0.0
        vec[6] = self._log_scale(cf.get("match_count", 0), 5)
        vec[7] = 1.0 if cf.get("branch_in_loop", False) else 0.0  # filter pattern

        vec[8] = self._log_scale(cf.get("try_count", 0), 5)
        vec[9] = 1.0 if cf.get("try_count", 0) >= 1 else 0.0
        vec[10] = 1.0 if cf.get("has_catch", False) or cf.get("has_except", False) else 0.0
        vec[11] = 1.0 if cf.get("has_finally", False) else 0.0

        vec[12] = 1.0 if cf.get("has_resource_guard", False) else 0.0
        vec[13] = self._log_scale(cf.get("max_nesting_depth", 0), 8)
        vec[14] = self._log_scale(cf.get("func_count", 0), 10)
        vec[15] = self._log_scale(cf.get("class_count", 0), 5)

        return vec

    def _log_scale(self, count: int, max_val: float = 10.0) -> float:
        if count <= 0:
            return 0.0
        return min(1.0, math.log1p(count) / math.log1p(max_val))

    # =========================================================================
    # CFG Fingerprint encoding
    # =========================================================================

    def _encode_cfg(self, cfg_nodes: Dict[int, str], cfg_edges: List[Tuple[int, int, str]]) -> List[float]:
        """Encode CFG structure via edge type distribution and graph properties."""
        vec = [0.0] * self.CFG_DIM

        if not cfg_edges:
            return vec

        edge_types = Counter(e[2] for e in cfg_edges)
        # Exclude loop_entry from ratio calculation - it's structural, not semantic
        # This keeps vector meaning stable after adding loop_entry edges
        ratio_edges = len(cfg_edges) - edge_types.get("loop_entry", 0)
        if ratio_edges <= 0:
            ratio_edges = 1  # Avoid division by zero

        vec[0] = edge_types.get("sequential", 0) / ratio_edges
        vec[1] = edge_types.get("branch_true", 0) / ratio_edges
        vec[2] = edge_types.get("branch_false", 0) / ratio_edges
        vec[3] = edge_types.get("loop_back", 0) / ratio_edges
        vec[4] = edge_types.get("exception", 0) / ratio_edges

        # Graph density (use total_edges including loop_entry for topology)
        total_edges = len(cfg_edges)
        n_nodes = len(cfg_nodes)
        if n_nodes > 1:
            vec[5] = min(1.0, total_edges / (n_nodes * 2))

        # Cyclomatic complexity approximation: E - N + 2P
        vec[6] = self._log_scale(total_edges - n_nodes + 2, 20)

        # Node type entropy
        if cfg_nodes:
            node_types = Counter(cfg_nodes.values())
            total_nodes = len(cfg_nodes)
            entropy = -sum((c/total_nodes) * math.log2(c/total_nodes + 1e-10)
                          for c in node_types.values())
            vec[7] = min(1.0, entropy / 4.0)

        return vec

    # =========================================================================
    # SimHash encoding (LSH bits)
    # =========================================================================

    def _encode_simhash(self, simhash: int) -> List[float]:
        """Extract 8 representative bits from 64-bit SimHash."""
        vec = [0.0] * self.SIMHASH_DIM

        # Sample bits at regular intervals
        for i in range(self.SIMHASH_DIM):
            bit_pos = i * 8
            vec[i] = 1.0 if (simhash >> bit_pos) & 1 else 0.0

        return vec

    # =========================================================================
    # Spectral Features encoding
    # =========================================================================

    def _encode_spectral(self, spectral_features: List[float]) -> List[float]:
        """Encode spectral features (eigenvalues)."""
        if not spectral_features:
            return [0.0] * self.SPECTRAL_DIM

        result = spectral_features[:self.SPECTRAL_DIM]
        while len(result) < self.SPECTRAL_DIM:
            result.append(0.0)

        return result

    # =========================================================================
    # Utilities
    # =========================================================================

    def _l2_normalize(self, vec: List[float]) -> List[float]:
        norm = math.sqrt(sum(x*x for x in vec))
        if norm < 1e-10:
            return vec
        return [x / norm for x in vec]

    def similarity(self, vec_a: List[float], vec_b: List[float]) -> float:
        """Cosine similarity."""
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a*a for a in vec_a))
        norm_b = math.sqrt(sum(b*b for b in vec_b))
        if norm_a < 1e-10 or norm_b < 1e-10:
            return 0.0
        return dot / (norm_a * norm_b)

    def hamming_distance(self, simhash_a: int, simhash_b: int) -> int:
        """Hamming distance between two SimHash values."""
        return bin(simhash_a ^ simhash_b).count('1')

    def tree_edit_distance_approx(self, paths_a: List[int], paths_b: List[int]) -> float:
        """Approximate tree edit distance via Jaccard of path hashes."""
        if not paths_a or not paths_b:
            return 1.0
        set_a = set(paths_a)
        set_b = set(paths_b)
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return 1.0 - (intersection / union) if union > 0 else 1.0
