"""
AROMA-style Prune Algorithm - Greedy maximal subtree extraction.

From the AROMA paper (Section 3.3.2):
The Prune algorithm takes two code snippets m1 (query) and m2 (candidate),
and finds a MAXIMAL SUBTREE of m2 that is most similar to m1.

Key insight: By identifying which leaf nodes of m2's parse tree should be
retained, we get a maximal subtree by keeping all nodes on paths from root
to those leaves.

Algorithm (greedy):
1. Start with empty set R (leaf nodes to retain)
2. Start with empty feature set F
3. Iteratively find leaf node n from m2 that maximizes:
   SimScore(F(m1), F ∪ F(n))
4. If adding n increases similarity, add to R, update F
5. Stop when no improvement possible
6. Return subtree formed by R + all ancestors

SimScore(m1, m2) = |F(m1) ∩ F(m2)| (feature overlap cardinality)

Uses:
- Reranking: Prune each search result w.r.t. query, rank by pruned similarity
- Intersection: Extract common structure from query + cluster members
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Any, Iterator
from collections import Counter
import hashlib

from .extractor import PatternSignature, PatternExtractor


@dataclass
class PrunedResult:
    """Result of pruning m2 with respect to m1."""
    
    # Retained leaf node IDs
    retained_leaves: Set[int] = field(default_factory=set)
    
    # Retained internal node IDs (ancestors of leaves)
    retained_nodes: Set[int] = field(default_factory=set)
    
    # Features of the pruned subtree
    pruned_features: Set[str] = field(default_factory=set)
    
    # Similarity score after pruning
    similarity_score: float = 0.0
    
    # Original m2 feature count (before pruning)
    original_feature_count: int = 0
    
    # Pruned feature count
    pruned_feature_count: int = 0
    
    # Ratio of m2 retained
    retention_ratio: float = 0.0


class AromaPruner:
    """
    AROMA-style greedy pruning for maximal similar subtree extraction.
    
    This implements the core AROMA algorithm that enables:
    1. Better reranking of search results
    2. Precise intersection for pattern discovery
    """
    
    def __init__(self, extractor: Optional[PatternExtractor] = None):
        self.extractor = extractor or PatternExtractor()
    
    def prune(
        self,
        query_sig: PatternSignature,
        candidate_sig: PatternSignature,
        query_tree: Optional[Any] = None,
        candidate_tree: Optional[Any] = None,
    ) -> PrunedResult:
        """
        Find maximal subtree of candidate that is most similar to query.
        
        Uses feature-based approximation when AST not available,
        or full tree-based algorithm when AST is provided.
        """
        if candidate_tree is not None and query_tree is not None:
            return self._prune_tree_based(query_sig, candidate_sig, query_tree, candidate_tree)
        else:
            return self._prune_feature_based(query_sig, candidate_sig)
    
    def _prune_feature_based(
        self,
        query_sig: PatternSignature,
        candidate_sig: PatternSignature,
    ) -> PrunedResult:
        """
        Feature-based pruning approximation.
        
        When we don't have AST access, we use features as proxy for leaves:
        - Each structural n-gram acts like a "leaf feature"
        - Greedily add features that increase overlap with query
        """
        result = PrunedResult()
        
        # Extract feature sets
        query_features = self._signature_to_features(query_sig)
        candidate_features = self._signature_to_features(candidate_sig)
        
        result.original_feature_count = len(candidate_features)
        
        if not query_features or not candidate_features:
            return result
        
        # Greedy selection: add candidate features that are in query
        # This is the feature-based approximation of tree pruning
        retained: Set[str] = set()
        candidate_list = list(candidate_features)
        
        # Sort by whether feature is in query (prefer matches)
        candidate_list.sort(key=lambda f: (f in query_features, f), reverse=True)
        
        current_score = 0
        for feature in candidate_list:
            # Would adding this feature improve overlap?
            new_retained = retained | {feature}
            new_score = len(new_retained & query_features)
            
            if new_score > current_score:
                retained.add(feature)
                current_score = new_score
            # Stop if we've matched all query features
            if current_score == len(query_features):
                break
        
        result.retained_leaves = set(range(len(retained)))  # Placeholder IDs
        result.pruned_features = retained
        result.pruned_feature_count = len(retained)
        result.similarity_score = current_score / max(len(query_features), 1)
        result.retention_ratio = len(retained) / max(len(candidate_features), 1)

        return result

    def _prune_tree_based(
        self,
        query_sig: PatternSignature,
        candidate_sig: PatternSignature,
        query_tree: Any,
        candidate_tree: Any,
    ) -> PrunedResult:
        """
        Full tree-based AROMA pruning algorithm.

        Algorithm:
        1. Collect all leaf nodes of candidate tree
        2. For each leaf, compute its feature contribution
        3. Greedily select leaves that maximize SimScore(query, retained)
        4. Compute ancestors of retained leaves for full subtree
        """
        result = PrunedResult()

        # Get query features
        query_features = self._signature_to_features(query_sig)
        if not query_features:
            return result

        # Collect leaf nodes with their features
        leaves: List[Tuple[Any, int, Set[str]]] = []  # (node, node_id, features)
        self._collect_leaves_with_features(candidate_tree, 0, leaves)

        result.original_feature_count = sum(len(f) for _, _, f in leaves)

        if not leaves:
            return result

        # Greedy selection
        retained_leaves: Set[int] = set()
        current_features: Set[str] = set()
        current_score = 0

        # Track which leaves haven't been considered yet
        available = set(range(len(leaves)))

        while available:
            best_leaf_idx = -1
            best_new_score = current_score
            best_new_features: Set[str] = current_features

            # Find leaf that maximizes score increase
            for idx in available:
                _, _, leaf_features = leaves[idx]
                candidate_features = current_features | leaf_features
                new_score = len(candidate_features & query_features)

                if new_score > best_new_score:
                    best_new_score = new_score
                    best_leaf_idx = idx
                    best_new_features = candidate_features

            # If no improvement, stop
            if best_leaf_idx == -1 or best_new_score <= current_score:
                break

            # Add best leaf
            retained_leaves.add(best_leaf_idx)
            current_features = best_new_features
            current_score = best_new_score
            available.remove(best_leaf_idx)

        # Compute ancestors of retained leaves
        retained_nodes = self._compute_ancestors(candidate_tree, retained_leaves, leaves)

        result.retained_leaves = retained_leaves
        result.retained_nodes = retained_nodes
        result.pruned_features = current_features
        result.pruned_feature_count = len(current_features)
        result.similarity_score = current_score / max(len(query_features), 1)
        result.retention_ratio = len(retained_nodes) / max(self._count_nodes(candidate_tree), 1)

        return result

    def _collect_leaves_with_features(
        self,
        node: Any,
        node_id: int,
        leaves: List[Tuple[Any, int, Set[str]]],
        parent_path: Optional[List[str]] = None,
    ) -> int:
        """Collect leaf nodes with their feature contributions."""
        if parent_path is None:
            parent_path = []

        node_type = getattr(node, 'type', str(type(node).__name__))
        current_path = parent_path + [node_type]

        children = getattr(node, 'children', [])

        if not children:
            # This is a leaf - compute its features
            features: Set[str] = set()

            # Path feature (root to this leaf)
            features.add(f"path:{'→'.join(current_path)}")

            # Parent feature
            if len(current_path) >= 2:
                features.add(f"parent:{current_path[-2]}→{current_path[-1]}")

            # Grandparent feature
            if len(current_path) >= 3:
                features.add(f"gp:{current_path[-3]}→{current_path[-2]}→{current_path[-1]}")

            # Node type feature
            features.add(f"type:{node_type}")

            leaves.append((node, node_id, features))
            return node_id + 1

        # Recurse to children
        next_id = node_id + 1
        for child in children:
            next_id = self._collect_leaves_with_features(child, next_id, leaves, current_path)

        return next_id

    def _compute_ancestors(
        self,
        root: Any,
        retained_leaf_indices: Set[int],
        leaves: List[Tuple[Any, int, Set[str]]],
    ) -> Set[int]:
        """Compute all ancestor node IDs for retained leaves."""
        # Build parent map
        parent_map: Dict[int, int] = {}
        self._build_parent_map(root, 0, parent_map)

        # Get actual node IDs of retained leaves
        retained_node_ids = {leaves[idx][1] for idx in retained_leaf_indices}

        # Trace up to root for each retained leaf
        ancestors: Set[int] = set()
        for node_id in retained_node_ids:
            current = node_id
            while current in parent_map:
                ancestors.add(current)
                current = parent_map[current]
            ancestors.add(current)  # Add root

        return ancestors

    def _build_parent_map(self, node: Any, node_id: int, parent_map: Dict[int, int]) -> int:
        """Build mapping from node ID to parent ID."""
        children = getattr(node, 'children', [])
        next_id = node_id + 1

        for child in children:
            parent_map[next_id] = node_id
            next_id = self._build_parent_map(child, next_id, parent_map)

        return next_id

    def _count_nodes(self, node: Any) -> int:
        """Count total nodes in tree."""
        children = getattr(node, 'children', [])
        return 1 + sum(self._count_nodes(c) for c in children)

    def _signature_to_features(self, sig: PatternSignature) -> Set[str]:
        """Convert PatternSignature to feature set for overlap computation."""
        features: Set[str] = set()

        # Structural n-grams as features
        for ngram, count in sig.structural_ngrams.items():
            if isinstance(ngram, tuple):
                features.add(f"ngram:{'>'.join(str(x) for x in ngram)}")
            else:
                features.add(f"ngram:{ngram}")

        # AST paths as features
        for parent, path, child, depth in sig.ast_paths:
            features.add(f"path:{parent}→{path}→{child}")

        # Control flow features
        cf = sig.control_flow
        if cf.get("loop_count", 0) > 0:
            features.add("cf:has_loop")
        if cf.get("branch_count", 0) > 0:
            features.add("cf:has_branch")
        if cf.get("try_count", 0) > 0:
            features.add("cf:has_try")
        if cf.get("has_catch"):
            features.add("cf:has_catch")
        if cf.get("has_finally"):
            features.add("cf:has_finally")
        if cf.get("has_resource_guard"):
            features.add("cf:has_resource_guard")
        if cf.get("has_retry"):
            features.add("cf:has_retry")

        # Control flow signature
        if cf.get("signature"):
            features.add(f"cfsig:{cf['signature']}")

        return features

    # =========================================================================
    # AROMA Phase II: Rerank search results using pruned similarity
    # =========================================================================

    def rerank_results(
        self,
        query_sig: PatternSignature,
        candidates: List[Tuple[PatternSignature, float, Any]],  # (sig, original_score, metadata)
        top_k: int = 10,
        alpha: float = 0.5,  # Weight for pruned vs original score
    ) -> List[Tuple[PatternSignature, float, PrunedResult, Any]]:
        """
        Rerank search results using AROMA pruning.

        For each candidate:
        1. Prune it w.r.t. query to find maximal similar subtree
        2. Score = alpha * pruned_similarity + (1-alpha) * original_score
        3. Return top-k by combined score

        Args:
            query_sig: Query pattern signature
            candidates: List of (signature, original_score, metadata) tuples
            top_k: Number of results to return
            alpha: Weight for pruned similarity (0-1)

        Returns:
            List of (signature, combined_score, prune_result, metadata) sorted by score
        """
        reranked = []

        for sig, orig_score, metadata in candidates:
            prune_result = self.prune(query_sig, sig)

            # Combined score: pruned similarity + original embedding score
            combined = alpha * prune_result.similarity_score + (1 - alpha) * orig_score

            reranked.append((sig, combined, prune_result, metadata))

        # Sort by combined score descending
        reranked.sort(key=lambda x: -x[1])

        return reranked[:top_k]

    # =========================================================================
    # AROMA Phase III: Intersect for pattern discovery
    # =========================================================================

    def intersect_signatures(
        self,
        signatures: List[PatternSignature],
        min_support: float = 0.8,  # Feature must appear in this fraction of signatures
    ) -> PatternSignature:
        """
        Extract common structure from multiple signatures (AROMA intersection).

        This is used after clustering to find the COMMON pattern across examples.

        Args:
            signatures: List of similar pattern signatures
            min_support: Minimum fraction of signatures a feature must appear in

        Returns:
            New PatternSignature containing only common features
        """
        if not signatures:
            return PatternSignature()

        if len(signatures) == 1:
            return signatures[0]

        # Count feature occurrences across all signatures
        feature_counts: Counter = Counter()
        all_features_by_sig: List[Set[str]] = []

        for sig in signatures:
            features = self._signature_to_features(sig)
            all_features_by_sig.append(features)
            feature_counts.update(features)

        # Keep features with sufficient support
        threshold = int(len(signatures) * min_support)
        common_features = {f for f, count in feature_counts.items() if count >= threshold}

        # Build intersected signature
        result = PatternSignature(language="multi")  # Cross-language pattern

        # Intersect structural n-grams
        for sig in signatures:
            for ngram, count in sig.structural_ngrams.items():
                ngram_feature = f"ngram:{'>'.join(str(x) for x in ngram)}" if isinstance(ngram, tuple) else f"ngram:{ngram}"
                if ngram_feature in common_features:
                    result.structural_ngrams[ngram] = min(
                        result.structural_ngrams.get(ngram, float('inf')),
                        count
                    )

        # Fix inf values to actual counts
        result.structural_ngrams = Counter({
            k: v for k, v in result.structural_ngrams.items()
            if v != float('inf')
        })

        # Intersect AST paths
        path_counts: Counter = Counter()
        for sig in signatures:
            for path in sig.ast_paths:
                path_key = (path[0], path[1], path[2])  # Exclude depth
                path_counts[path_key] += 1

        for path_key, count in path_counts.items():
            if count >= threshold:
                result.ast_paths.append((*path_key, 0))  # Add with depth=0

        # Intersect control flow
        cf_features = {
            "loop_count": [], "branch_count": [], "try_count": [],
            "has_catch": [], "has_finally": [], "has_resource_guard": [],
            "has_retry": [], "max_nesting_depth": [],
        }

        for sig in signatures:
            cf = sig.control_flow
            for key in cf_features:
                if key in cf:
                    cf_features[key].append(cf[key])

        result.control_flow = {}
        for key, values in cf_features.items():
            if len(values) >= threshold:
                if isinstance(values[0], bool):
                    # For booleans, require all True
                    result.control_flow[key] = all(values)
                else:
                    # For counts, take minimum
                    result.control_flow[key] = min(values)

        # Collect languages
        languages = {sig.language for sig in signatures if sig.language != "unknown"}
        if languages:
            result.language = ",".join(sorted(languages))

        return result

    def compute_extension_score(
        self,
        query_sig: PatternSignature,
        candidate_sig: PatternSignature,
        prune_result: PrunedResult,
        tau2: float = 1.5,  # Candidate should be at least tau2 * query size
        tau3: float = 0.9,  # Minimum similarity threshold
    ) -> Tuple[float, bool]:
        """
        AROMA commonality constraints (Section 3.3.3).

        A recommendation should:
        1. Be similar enough to query (SimScore >= tau3)
        2. Contain significant extension beyond query (size >= tau2 * query_size)

        Returns:
            (extension_score, passes_constraints)
        """
        query_features = self._signature_to_features(query_sig)
        candidate_features = self._signature_to_features(candidate_sig)

        query_size = len(query_features)
        candidate_size = len(candidate_features)

        # Check similarity threshold
        passes_similarity = prune_result.similarity_score >= tau3

        # Check extension threshold
        passes_extension = candidate_size >= tau2 * query_size

        # Extension score: how much extra the candidate provides
        common = len(prune_result.pruned_features & query_features)
        extension = candidate_size - common
        extension_score = extension / max(query_size, 1)

        passes = passes_similarity and passes_extension

        return extension_score, passes

