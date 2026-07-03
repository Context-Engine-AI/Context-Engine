# Pattern Detection System
# ========================
#
# A DYNAMIC structural code similarity system with automatic pattern discovery.
# Based on AROMA (Meta) + code2vec AST path techniques.
#
# KEY FEATURES:
#   1. Works across ALL supported languages (16+ languages)
#   2. Patterns EMERGE automatically - not predefined
#   3. Cross-language matching - Python pattern matches Go/Rust/Java/etc.
#   4. Online learning - patterns improve as codebase is indexed
#   5. TOON output support for token-efficient responses
#
# Architecture:
#   1. PatternExtractor - Extract AST paths + control flow (normalized across languages)
#   2. PatternEncoder   - Convert features to 64-dim vector via TF-IDF + LSH
#   3. PatternMiner     - AROMA-style clustering to DISCOVER patterns dynamically
#   4. OnlinePatternLearner - Continuous learning as code is indexed
#   5. PatternSearch    - Qdrant-backed similarity search with TOON support
#
# Usage:
#   from scripts.pattern_detection import PatternExtractor, PatternEncoder
#
#   extractor = PatternExtractor()
#   encoder = PatternEncoder()
#
#   # Extract pattern from code (works for ANY supported language)
#   signature = extractor.extract(code, language="python")  # or "go", "rust", etc.
#   vector = encoder.encode(signature)
#
#   # Search for similar code patterns
#   from scripts.pattern_detection import pattern_search
#   results = pattern_search(code, "python", output_format="toon")  # Token-efficient
#
#   # Dynamic pattern discovery
#   from scripts.pattern_detection import get_pattern_learner
#   learner = get_pattern_learner()
#   learner.observe(code, path, language)  # Called during indexing
#   patterns = learner.query(example_code, "python")  # Find matching patterns

from .extractor import PatternExtractor, PatternSignature
from .encoder import PatternEncoder
from .catalog import PatternMiner, OnlinePatternLearner, get_pattern_learner, DiscoveredPattern
from .prune import AromaPruner, PrunedResult
from .search import (
    pattern_search,
    find_similar_patterns,
    search_by_pattern_description,
    search_similar_code,
    find_code_like,
    PatternSearchResult,
    PatternSearchResponse,
    encode_pattern_results,
)

# Backward compatibility aliases
PatternCatalog = PatternMiner
KNOWN_PATTERNS = []  # Now populated dynamically

__all__ = [
    # Core extraction
    "PatternExtractor",
    "PatternSignature",
    "PatternEncoder",
    # AROMA pruning (Phase II & III)
    "AromaPruner",
    "PrunedResult",
    # Dynamic discovery
    "PatternMiner",
    "OnlinePatternLearner",
    "get_pattern_learner",
    "DiscoveredPattern",
    # Search (with TOON support)
    "pattern_search",
    "find_similar_patterns",
    "search_by_pattern_description",
    "search_similar_code",
    "find_code_like",
    "PatternSearchResult",
    "PatternSearchResponse",
    "encode_pattern_results",
    # Backward compatibility
    "PatternCatalog",
    "KNOWN_PATTERNS",
]

