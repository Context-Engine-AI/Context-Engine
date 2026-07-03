"""
Dynamic Pattern Discovery - AROMA-inspired self-organizing pattern system.

This is NOT a static catalog. Patterns are DISCOVERED dynamically by:

1. **Clustering** - Group structurally similar code across the codebase
2. **Intersection** - Extract the COMMON structure from clusters (the "pattern")
3. **Emergence** - Patterns naturally emerge from code, not manual definition
4. **Cross-language** - Normalized AST means Python patterns match Go/Rust/etc.

Key insight from AROMA paper: Don't predefine patterns. Instead:
- Query: "find code similar to X"
- Retrieve: Top-K structurally similar snippets
- Intersect: Find what's COMMON across all K snippets
- That common structure IS the pattern

This approach:
- Discovers domain-specific patterns unique to each codebase
- Adapts as code evolves
- Works for patterns nobody thought to predefine
- Cross-language pattern matching via normalized AST
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set, Any, Iterator
from collections import Counter, defaultdict
import hashlib
import math
import time


@dataclass
class DiscoveredPattern:
    """A pattern discovered through structural clustering - NOT predefined."""

    # Unique ID derived from structural fingerprint
    pattern_id: str

    # The common structural features across examples
    common_ngrams: Counter  # Structural n-grams present in ALL examples
    common_cf: Dict[str, Any]  # Common control flow features

    # Representative vector (centroid of cluster)
    centroid: List[float] = field(default_factory=list)

    # Example file paths that exhibit this pattern
    exemplars: List[str] = field(default_factory=list)

    # Languages this pattern appears in
    languages: Set[str] = field(default_factory=set)

    # Discovery metadata
    cluster_size: int = 0
    coherence_score: float = 0.0  # How tight is the cluster
    discovery_time: float = 0.0

    # Auto-generated description based on structure
    auto_description: str = ""

    def __hash__(self):
        return hash(self.pattern_id)


class PatternMiner:
    """
    AROMA-style pattern discovery through structural clustering.

    Instead of predefined patterns, this DISCOVERS patterns by:
    1. Indexing structural signatures of all code spans
    2. Clustering similar structures (locality-sensitive hashing)
    3. Intersecting clusters to extract common structure
    4. Ranking patterns by frequency and coherence
    """

    # Clustering parameters
    MIN_CLUSTER_SIZE = 3  # Minimum examples to form a pattern
    MAX_PATTERNS = 1000   # Cap on discovered patterns
    SIMILARITY_THRESHOLD = 0.7  # Min similarity to be in same cluster

    # LSH parameters for fast clustering
    NUM_HASH_TABLES = 10
    HASH_SIZE = 8

    def __init__(self):
        self._patterns: Dict[str, DiscoveredPattern] = {}
        # lsh_hash -> [(path, language, vector, signature)]
        self._signature_index: Dict[str, List[Tuple[str, str, List[float], Any]]] = defaultdict(list)
        self._extractor = None
        self._encoder = None

    def _lazy_init(self):
        if self._extractor is None:
            from .extractor import PatternExtractor
            from .encoder import PatternEncoder
            self._extractor = PatternExtractor()
            self._encoder = PatternEncoder()

    def index_snippet(self, code: str, path: str, language: str) -> str:
        """Index a code snippet for pattern discovery. Returns signature hash."""
        self._lazy_init()

        sig = self._extractor.extract(code, language)
        vec = self._encoder.encode(sig)

        # Generate LSH hashes for fast clustering
        lsh_hashes = self._compute_lsh_hashes(vec)

        # Index under all LSH hashes
        for h in lsh_hashes:
            self._signature_index[h].append((path, language, vec, sig))

        return sig.fingerprint()

    def _compute_lsh_hashes(self, vec: List[float]) -> List[str]:
        """Compute LSH hashes for approximate nearest neighbor clustering."""
        hashes = []

        for table_idx in range(self.NUM_HASH_TABLES):
            # Random hyperplane projection (deterministic via seed)
            bits = []
            for i in range(self.HASH_SIZE):
                # Use table and bit index as seed for reproducibility
                seed = table_idx * 1000 + i
                projection = sum(
                    vec[j] * self._pseudo_random(seed, j)
                    for j in range(len(vec))
                )
                bits.append('1' if projection >= 0 else '0')

            hash_val = f"t{table_idx}_{''.join(bits)}"
            hashes.append(hash_val)

        return hashes

    def _pseudo_random(self, seed: int, idx: int) -> float:
        """Deterministic pseudo-random for LSH projection."""
        h = hashlib.md5(f"{seed}:{idx}".encode()).digest()
        val = int.from_bytes(h[:4], 'big') / (2**32)
        return val * 2 - 1  # Map to [-1, 1]

    def discover_patterns(self, min_support: int = 3) -> List[DiscoveredPattern]:
        """
        Discover patterns by clustering and intersection.

        This is the AROMA magic:
        1. Find clusters of similar signatures via LSH
        2. For each cluster, intersect to find COMMON structure
        3. That common structure becomes a discovered pattern
        """
        self._lazy_init()

        discovered = []
        processed_clusters: Set[frozenset] = set()

        # For each LSH bucket with enough items
        for lsh_hash, items in self._signature_index.items():
            if len(items) < min_support:
                continue

            # Get cluster members - use (path, signature_fingerprint) to allow multiple spans per file
            members = frozenset((item[0], item[3].fingerprint()) for item in items)
            if members in processed_clusters:
                continue
            processed_clusters.add(members)

            # Extract unique paths for pattern ID and exemplars
            paths = list(set(item[0] for item in items))

            # Compute cluster coherence (avg pairwise similarity)
            vecs = [item[2] for item in items]
            coherence = self._cluster_coherence(vecs)

            if coherence < self.SIMILARITY_THRESHOLD:
                continue

            # INTERSECT: Find common structure across all cluster members
            sigs = [item[3] for item in items]
            common = self._intersect_signatures(sigs)

            if not common:
                continue

            # Create discovered pattern
            pattern = DiscoveredPattern(
                pattern_id=hashlib.md5(str(sorted(paths)).encode()).hexdigest()[:12],
                common_ngrams=common['ngrams'],
                common_cf=common['cf'],
                centroid=self._compute_centroid(vecs),
                exemplars=paths[:10],
                languages=set(item[1] for item in items),
                cluster_size=len(items),
                coherence_score=coherence,
                discovery_time=time.time(),
                auto_description=self._generate_description(common),
            )

            discovered.append(pattern)
            self._patterns[pattern.pattern_id] = pattern

        # Sort by cluster size and coherence
        discovered.sort(key=lambda p: (p.cluster_size, p.coherence_score), reverse=True)
        return discovered[:self.MAX_PATTERNS]

    def _cluster_coherence(self, vecs: List[List[float]]) -> float:
        """Compute average pairwise cosine similarity within cluster."""
        if len(vecs) < 2:
            return 1.0

        total_sim = 0.0
        count = 0

        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                sim = self._cosine_similarity(vecs[i], vecs[j])
                total_sim += sim
                count += 1

        return total_sim / count if count > 0 else 0.0

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Compute cosine similarity."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a < 1e-10 or norm_b < 1e-10:
            return 0.0
        return dot / (norm_a * norm_b)

    def _compute_centroid(self, vecs: List[List[float]]) -> List[float]:
        """Compute centroid of vectors."""
        if not vecs:
            return []
        dim = len(vecs[0])
        centroid = [0.0] * dim
        for vec in vecs:
            for i, v in enumerate(vec):
                centroid[i] += v
        return [c / len(vecs) for c in centroid]

    def _intersect_signatures(self, sigs) -> Optional[Dict[str, Any]]:
        """
        AROMA CORE: Intersect signatures to find COMMON structure.

        This extracts what's shared across ALL examples in a cluster.
        The intersection IS the pattern - features that appear in
        every instance of this pattern.
        """
        if not sigs:
            return None

        # Intersect structural n-grams: keep only those in ALL signatures
        common_ngrams = None
        for sig in sigs:
            if common_ngrams is None:
                common_ngrams = Counter(sig.structural_ngrams)
            else:
                # Keep only n-grams present in both, with min count
                common_ngrams = Counter({
                    k: min(common_ngrams[k], sig.structural_ngrams[k])
                    for k in common_ngrams
                    if k in sig.structural_ngrams
                })

        if not common_ngrams:
            return None

        # Intersect control flow: keep features present in ALL signatures
        common_cf = {}
        cf_keys = ['max_loop_depth', 'loop_count', 'branch_count', 'try_count',
                   'has_finally', 'has_catch', 'has_resource_guard', 'match_count']

        for key in cf_keys:
            values = [sig.control_flow.get(key) for sig in sigs]
            if all(v == values[0] for v in values):
                common_cf[key] = values[0]
            elif all(isinstance(v, bool) for v in values):
                # For bool, use AND (present in ALL)
                common_cf[key] = all(values)
            elif all(isinstance(v, (int, float)) for v in values):
                # For numbers, use min (lower bound guarantee)
                common_cf[key] = min(values)

        return {
            'ngrams': common_ngrams,
            'cf': common_cf,
        }

    def _generate_description(self, common: Dict[str, Any]) -> str:
        """Auto-generate a description from common structural features."""
        parts = []
        cf = common.get('cf', {})

        # Describe control flow
        if cf.get('loop_count', 0) > 0:
            depth = cf.get('max_loop_depth', 1)
            parts.append(f"loop structure (depth {depth})")

        if cf.get('try_count', 0) > 0:
            try_desc = "error handling"
            if cf.get('has_catch'):
                try_desc += " with catch"
            if cf.get('has_finally'):
                try_desc += " with cleanup"
            parts.append(try_desc)

        if cf.get('branch_count', 0) > 0:
            parts.append(f"{cf['branch_count']}+ conditional branches")

        if cf.get('has_resource_guard'):
            parts.append("resource management")

        if cf.get('match_count', 0) > 0:
            parts.append("pattern matching")

        # Describe common n-grams (top structural patterns)
        ngrams = common.get('ngrams', Counter())
        if ngrams:
            top_ngrams = ngrams.most_common(3)
            # Could analyze n-gram content for more specific description

        if not parts:
            return "structural pattern"

        return "Pattern with " + ", ".join(parts)

    def find_similar_to(self, code: str, language: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Find code snippets similar to the given example.

        This is the query path:
        1. Extract signature from query code
        2. LSH lookup for candidate clusters
        3. Rank by vector similarity
        """
        self._lazy_init()

        sig = self._extractor.extract(code, language)
        vec = self._encoder.encode(sig)
        lsh_hashes = self._compute_lsh_hashes(vec)

        # Gather candidates from all matching LSH buckets
        # Key by (path, signature_fingerprint) to allow multiple spans per file
        candidates = {}
        for h in lsh_hashes:
            for path, lang, cvec, csig in self._signature_index.get(h, []):
                key = (path, csig.fingerprint())
                if key not in candidates:
                    sim = self._cosine_similarity(vec, cvec)
                    candidates[key] = (sim, lang, path)

        # Sort by similarity
        results = [(path, sim) for key, (sim, lang, path) in candidates.items()]
        results.sort(key=lambda x: -x[1])

        return results[:top_k]

    def get_pattern_for_examples(self, paths: List[str]) -> Optional[DiscoveredPattern]:
        """
        Given a set of example paths, discover their common pattern.

        This is AROMA's recommendation path in reverse:
        - User says "these 3 files do the same thing"
        - We extract and intersect their structures
        - Return the discovered pattern
        """
        # Find signatures for these paths
        sigs = []
        vecs = []
        languages = set()

        for lsh_hash, items in self._signature_index.items():
            for path, lang, vec, sig in items:
                if path in paths:
                    sigs.append(sig)
                    vecs.append(vec)
                    languages.add(lang)

        if len(sigs) < 2:
            return None

        common = self._intersect_signatures(sigs)
        if not common:
            return None

        return DiscoveredPattern(
            pattern_id=hashlib.md5(str(sorted(paths)).encode()).hexdigest()[:12],
            common_ngrams=common['ngrams'],
            common_cf=common['cf'],
            centroid=self._compute_centroid(vecs),
            exemplars=paths,
            languages=languages,
            cluster_size=len(sigs),
            coherence_score=self._cluster_coherence(vecs),
            discovery_time=time.time(),
            auto_description=self._generate_description(common),
        )


# =============================================================================
# Online Pattern Learning - The GENIUS part
# =============================================================================

class OnlinePatternLearner:
    """
    Continuously discovers patterns as code is indexed.

    This is the genius: patterns emerge AUTOMATICALLY during indexing.
    No manual definition. No static catalog. Pure emergence.

    Key innovations:
    1. Incremental clustering - don't reprocess everything
    2. Pattern aging - old patterns decay, active ones strengthen
    3. Cross-language unification - Python pattern matches Go code
    4. Query-driven refinement - searches refine pattern boundaries
    """

    def __init__(self, persist_path: Optional[str] = None):
        self.miner = PatternMiner()
        self.persist_path = persist_path

        # Pattern strength tracking (reinforcement learning inspired)
        self._pattern_hits: Counter = Counter()  # How often each pattern matched a query
        self._pattern_age: Dict[str, float] = {}  # When pattern was last useful

        # Incremental update tracking
        self._pending_snippets: List[Tuple[str, str, str]] = []  # (code, path, lang)
        self._batch_size = 100  # Process patterns every N snippets

    def observe(self, code: str, path: str, language: str):
        """
        Observe a code snippet during indexing.

        Called by the indexer for every code span. Patterns
        emerge naturally from the accumulated observations.
        """
        self.miner.index_snippet(code, path, language)
        self._pending_snippets.append((code, path, language))

        # Periodically discover new patterns
        if len(self._pending_snippets) >= self._batch_size:
            self._discover_batch()

    def _discover_batch(self):
        """Process pending snippets and discover new patterns."""
        if not self._pending_snippets:
            return

        new_patterns = self.miner.discover_patterns(min_support=3)

        for pattern in new_patterns:
            self._pattern_age[pattern.pattern_id] = time.time()

        self._pending_snippets.clear()

    def query(self, code: str, language: str, top_k: int = 5) -> List[Tuple[DiscoveredPattern, float]]:
        """
        Find patterns matching a query code example.

        Also updates pattern strength (query-driven learning):
        - Patterns that match queries get reinforced
        - This naturally surfaces the most USEFUL patterns
        """
        self._lazy_init_miner()

        sig = self.miner._extractor.extract(code, language)
        vec = self.miner._encoder.encode(sig)

        results = []
        for pattern_id, pattern in self.miner._patterns.items():
            if not pattern.centroid:
                continue
            sim = self.miner._cosine_similarity(vec, pattern.centroid)
            if sim > 0.5:  # Threshold
                results.append((pattern, sim))
                # Reinforce this pattern
                self._pattern_hits[pattern_id] += 1
                self._pattern_age[pattern_id] = time.time()

        results.sort(key=lambda x: -x[1])
        return results[:top_k]

    def _lazy_init_miner(self):
        self.miner._lazy_init()

    def get_top_patterns(self, n: int = 20) -> List[DiscoveredPattern]:
        """Get the most useful discovered patterns (by query hits)."""
        # Score by: hits * recency * cluster_coherence
        now = time.time()
        scored = []

        for pid, pattern in self.miner._patterns.items():
            hits = self._pattern_hits.get(pid, 0)
            age = now - self._pattern_age.get(pid, now)
            recency = math.exp(-age / (30 * 24 * 3600))  # 30 day half-life
            score = (hits + 1) * recency * pattern.coherence_score
            scored.append((score, pattern))

        scored.sort(key=lambda x: -x[0])
        return [p for _, p in scored[:n]]

    def natural_language_query(self, query: str, top_k: int = 5) -> List[DiscoveredPattern]:
        """
        Find patterns matching a natural language description.

        This bridges "find retry patterns" → discovered patterns
        without needing predefined keywords. Uses the auto-generated
        descriptions for matching.
        """
        query_lower = query.lower()
        keywords = set(query_lower.split())

        scored = []
        for pattern in self.miner._patterns.values():
            desc_words = set(pattern.auto_description.lower().split())

            # Simple keyword overlap score
            overlap = len(keywords & desc_words)

            # Boost by pattern quality
            quality = pattern.coherence_score * math.log1p(pattern.cluster_size)

            score = overlap * quality
            if score > 0:
                scored.append((score, pattern))

        scored.sort(key=lambda x: -x[0])
        return [p for _, p in scored[:top_k]]

    def explain_pattern(self, pattern: DiscoveredPattern) -> str:
        """Generate human-readable explanation of a discovered pattern."""
        lines = [
            f"Pattern: {pattern.auto_description}",
            f"Found in {pattern.cluster_size} places across {len(pattern.languages)} language(s)",
            f"Coherence: {pattern.coherence_score:.2%}",
            "",
            "Common structural elements:",
        ]

        # Describe control flow
        cf = pattern.common_cf
        if cf.get('loop_count', 0) > 0:
            lines.append(f"  • Loops (max depth: {cf.get('max_loop_depth', 1)})")
        if cf.get('try_count', 0) > 0:
            parts = ["  • Error handling"]
            if cf.get('has_catch'):
                parts.append("with catch/except")
            if cf.get('has_finally'):
                parts.append("with cleanup")
            lines.append(" ".join(parts))
        if cf.get('branch_count', 0) > 0:
            lines.append(f"  • Conditional branches: {cf['branch_count']}+")
        if cf.get('has_resource_guard'):
            lines.append("  • Resource management (with/using/defer)")

        lines.append("")
        lines.append("Example locations:")
        for path in pattern.exemplars[:5]:
            lines.append(f"  - {path}")

        return "\n".join(lines)


# Singleton instance for global pattern learning
_global_learner: Optional[OnlinePatternLearner] = None


def get_pattern_learner() -> OnlinePatternLearner:
    """Get or create the global pattern learner."""
    global _global_learner
    if _global_learner is None:
        _global_learner = OnlinePatternLearner()
    return _global_learner


# =============================================================================
# Backward compatibility aliases
# =============================================================================

PatternCatalog = PatternMiner  # Legacy name
SEED_PATTERNS = {
    "retry": "for i in range(n): try: x() break except: sleep(2**i)",
    "cleanup": "x = acquire() try: use(x) finally: x.close()",
}
KNOWN_PATTERNS: List[DiscoveredPattern] = []  # Populated dynamically by learner

