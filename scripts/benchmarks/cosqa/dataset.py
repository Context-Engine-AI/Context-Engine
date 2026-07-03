#!/usr/bin/env python3
"""
CoSQA Dataset Handler for Context-Engine Benchmarks.

Downloads and processes the CoSQA (Code Search and Question Answering) dataset
from HuggingFace for evaluating code search quality.

The CoSQA dataset contains:
- Natural language queries (web queries from search engine logs)
- Python code snippets with docstrings
- Relevance labels (0/1) for query-code pairs

References:
- Paper: https://arxiv.org/abs/2105.13239
- Dataset: https://huggingface.co/datasets/gonglinyuan/CoSQA
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Iterator

# Cache directory for downloaded datasets
CACHE_DIR = Path(os.environ.get("COSQA_CACHE_DIR", Path.home() / ".cache" / "cosqa"))

# Cache version - increment when ID generation scheme changes
# v1: Initial version with positional IDs
# v2: Content-hash based IDs for proper query aggregation
# v3: Enhanced synthetic paths for prod-like FNAME_BOOST testing
# v5: Recouple filename to func_name to exercise FNAME_BOOST like production
CACHE_VERSION = 5


@dataclass
class CoSQAExample:
    """A single CoSQA query-code pair."""
    query_id: str
    query: str
    code_id: str
    code: str
    docstring: str
    label: int  # 1 = relevant, 0 = not relevant
    url: str = ""
    func_name: str = ""

    @property
    def is_relevant(self) -> bool:
        return self.label == 1


@dataclass
class CoSQACorpusEntry:
    """A single code entry in the CoSQA corpus (for indexing)."""
    code_id: str
    code: str
    docstring: str
    func_name: str
    url: str = ""
    language: str = "python"

    def to_index_payload(self) -> Dict[str, Any]:
        """Convert to payload suitable for Qdrant indexing.

        Creates synthetic but realistic file paths and metadata so Context-Engine's
        full pipeline (FNAME_BOOST, symbol boost, path filters) is properly exercised.
        """
        # NOTE: Context-Engine's retrieval stack (`hybrid_search` / `repo_search`) expects
        # codebase-like points that carry `payload["metadata"]["path"]` + line bounds.
        # For the CoSQA corpus (which is not a real filesystem), we synthesize realistic
        # metadata so we can evaluate retrieval using the FULL production pipeline.
        #
        # This keeps the canonical identifier in `code_id` for MRR computations.

        # Build realistic synthetic path from function name
        # e.g., func_name="parse_json_config" -> "cosqa/parse/parse_json_config__d123.py"
        # NOTE: Include code_id in filename to avoid path collisions (CoSQA has duplicate func_names)
        # This prevents per_path limiting from dropping the expected doc
        module_hint = ""
        func_name = self.func_name or ""
        filename = ""
        if func_name:
            # Extract module hint from function name (first word before underscore)
            parts = func_name.split("_")
            if len(parts) >= 2:
                module_hint = parts[0]
            symbol_name = func_name
            # Include code_id to ensure unique paths for duplicate func_names
            filename = f"{func_name}__{self.code_id}.py"
        else:
            # Fallback: extract potential function name from code
            import re as _re
            match = _re.search(r"def\s+(\w+)\s*\(", self.code)
            if match:
                extracted_name = match.group(1)
                parts = extracted_name.split("_")
                if len(parts) >= 2:
                    module_hint = parts[0]
                symbol_name = extracted_name
                filename = f"{extracted_name}__{self.code_id}.py"
            else:
                symbol_name = self.code_id
                filename = f"{self.code_id}.py"

        if module_hint:
            _synthetic_path = f"cosqa/{module_hint}/{filename}"
        else:
            _synthetic_path = f"cosqa/{filename}"

        rerank_text = self.code
        if self.docstring and self.docstring.strip():
            rerank_text = f"{self.docstring.strip()}\n\n{self.code}"

        # Count lines for realistic line bounds
        code_lines = self.code.count("\n") + 1

        return {
            "code_id": self.code_id,
            "text": self.code,
            "docstring": self.docstring,
            "func_name": func_name or symbol_name,
            "url": self.url,
            "language": self.language,
            "kind": "function",
            "source": "cosqa",
            "metadata": {
                "path": _synthetic_path,
                "path_prefix": "cosqa",
                "symbol": symbol_name,
                "symbol_path": f"{_synthetic_path}::{symbol_name}",
                "kind": "function",
                "language": self.language,
                # Realistic line bounds based on code length
                "start_line": 1,
                "end_line": code_lines,
                # Provide text inline so snippet/keyword bump doesn't try to read files.
                "text": self.code,
                # Reranker expects metadata.code for cross-encoder scoring (include docstring).
                "code": rerank_text,
                # Docstring for pseudo fallback in core_indexer
                "docstring": self.docstring or "",
            },
        }


@dataclass
class CoSQAQuery:
    """A CoSQA query with its relevant code IDs."""
    query_id: str
    query: str
    relevant_code_ids: List[str] = field(default_factory=list)


@dataclass
class CoSQADataset:
    """Container for the full CoSQA dataset."""
    corpus: Dict[str, CoSQACorpusEntry] = field(default_factory=dict)
    queries: Dict[str, CoSQAQuery] = field(default_factory=dict)
    qrels: Dict[str, Dict[str, int]] = field(default_factory=dict)  # query_id -> {code_id: label}
    split: str = "test"
    cache_version: int = CACHE_VERSION
    fallback_used: bool = False  # True if MTEB format failed and fell back to original

    def __len__(self) -> int:
        return len(self.queries)

    def corpus_size(self) -> int:
        return len(self.corpus)

    def iter_corpus(self) -> Iterator[CoSQACorpusEntry]:
        yield from self.corpus.values()

    def iter_queries(self) -> Iterator[CoSQAQuery]:
        yield from self.queries.values()


def normalize_code(code: str) -> str:
    """Normalize code for consistent indexing.

    - Strip leading/trailing whitespace
    - Normalize line endings
    - Remove excessive blank lines
    """
    if not code:
        return ""
    # Normalize line endings
    code = code.replace("\r\n", "\n").replace("\r", "\n")
    # Remove trailing whitespace from lines
    lines = [line.rstrip() for line in code.split("\n")]
    # Collapse multiple blank lines to one
    result = []
    prev_blank = False
    for line in lines:
        is_blank = not line.strip()
        if is_blank and prev_blank:
            continue
        result.append(line)
        prev_blank = is_blank
    return "\n".join(result).strip()


def normalize_query(query: str) -> str:
    """Normalize query for consistent matching.

    - Lowercase
    - Strip whitespace
    - Remove special characters that might affect matching
    """
    if not query:
        return ""
    query = query.strip().lower()
    # Keep alphanumeric, spaces, and common programming symbols
    query = re.sub(r"[^\w\s\-_\.]", " ", query)
    # Collapse multiple spaces
    query = re.sub(r"\s+", " ", query)
    return query.strip()


def generate_code_id(code: str, idx: int = 0) -> str:
    """Generate a stable ID for a code snippet.

    Uses content hash only (no idx) so identical code dedupes properly.
    idx is kept for API compatibility but ignored.
    """
    del idx  # Unused - kept for API compat
    content_hash = hashlib.md5(code.encode("utf-8", errors="ignore")).hexdigest()[:12]
    return f"cosqa-{content_hash}"


def generate_query_id(query: str, idx: int = 0) -> str:
    """Generate a stable ID for a query.

    Uses content-only hash so identical queries aggregate their relevance
    judgments properly. The idx parameter is ignored (kept for API compat).

    This is important because:
    - Same query appearing multiple times should have same ID
    - Relevance judgments for the same query text should aggregate
    - Without this, repeated queries get different IDs and metrics are skewed
    """
    del idx  # Unused - kept for API compat
    content_hash = hashlib.md5(query.encode("utf-8", errors="ignore")).hexdigest()[:12]
    return f"q-{content_hash}"


def load_from_huggingface(
    dataset_name: str = "gonglinyuan/CoSQA",
    split: str = "test",
    cache_dir: Optional[Path] = None,
) -> CoSQADataset:
    """Load CoSQA dataset from HuggingFace.

    Args:
        dataset_name: HuggingFace dataset identifier.
        split: Dataset split to load ("train", "validation", "test").
        cache_dir: Local cache directory for downloaded data.

    Returns:
        CoSQADataset with corpus, queries, and relevance judgments.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "datasets library required. Install with: pip install datasets"
        )

    cache_path = cache_dir or CACHE_DIR
    cache_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading CoSQA dataset from {dataset_name}...")

    corpus: Dict[str, CoSQACorpusEntry] = {}
    queries: Dict[str, CoSQAQuery] = {}
    qrels: Dict[str, Dict[str, int]] = {}

    # Try mteb/cosqa format first (has separate queries/corpus/qrels)
    try:
        queries_ds = load_dataset("mteb/cosqa", "queries", split=split, cache_dir=str(cache_path))
        corpus_ds = load_dataset("mteb/cosqa", "corpus", split="test", cache_dir=str(cache_path))
        qrels_ds = load_dataset("mteb/cosqa", split=split, cache_dir=str(cache_path))

        print(f"Loaded mteb/cosqa: {len(queries_ds)} queries, {len(corpus_ds)} corpus, {len(qrels_ds)} qrels")

        # Map original IDs (d-xxxx) to our stable content-hash IDs
        # This handles deduplication: multiple original IDs -> single content ID
        id_map: Dict[str, str] = {}  # original_id -> stable_id

        # Parse corpus
        for item in corpus_ds:
            original_id = item["_id"]
            code = item.get("text", "")
            
            # Normalize and generate stable ID for deduplication
            code_norm = normalize_code(code)
            if not code_norm:
                continue
                
            # Use generate_code_id to get consistent cosqa-{hash} format
            stable_id = generate_code_id(code_norm)
            id_map[original_id] = stable_id

            # Only add to corpus if not already present (first win / dedup)
            if stable_id not in corpus:
                # Extract function name
                func_name = ""
                match = re.search(r"def\s+(\w+)\s*\(", code)
                if match:
                    func_name = match.group(1)
                
                corpus[stable_id] = CoSQACorpusEntry(
                    code_id=stable_id,
                    code=code,
                    docstring="",
                    func_name=func_name,
                    url="",
                )

        # Parse queries
        for item in queries_ds:
            query_id = item["_id"]
            query_text = item.get("text", "")
            queries[query_id] = CoSQAQuery(
                query_id=query_id,
                query=query_text,
                relevant_code_ids=[],  # Populated from qrels
            )

        # Parse qrels (relevance judgments)
        for item in qrels_ds:
            query_id = item["query-id"]
            original_corpus_id = item["corpus-id"]
            score = item.get("score", 1)
            
            # Remap ID for deduped corpus
            stable_corpus_id = id_map.get(original_corpus_id)
            if not stable_corpus_id:
                # ID referenced in qrels but not in corpus? Skip or warn.
                continue
                
            if query_id not in qrels:
                qrels[query_id] = {}
            qrels[query_id][stable_corpus_id] = score

        # Update queries with relevant code IDs
        for query_id, rels in qrels.items():
            if query_id in queries:
                queries[query_id] = CoSQAQuery(
                    query_id=query_id,
                    query=queries[query_id].query,
                    relevant_code_ids=[cid for cid, score in rels.items() if score > 0],
                )

        dataset = CoSQADataset(
            corpus=corpus,
            queries=queries,
            qrels=qrels,
            split=split,
            fallback_used=False,
        )

        # Cache result using centralized function (includes version)
        save_dataset_cache(dataset, cache_path / f"cosqa_{split}_processed.json")

        return dataset

    except Exception as e:
        print(f"[WARN] mteb/cosqa format failed ({e}), falling back to original format...")

    # Fallback to original gonglinyuan/CoSQA format
    try:
        ds = load_dataset(dataset_name, split=split, cache_dir=str(cache_path))
    except Exception as e2:
        raise RuntimeError(f"Failed to load CoSQA dataset: {e2}")

    # Process examples from original format
    for idx, example in enumerate(ds):
        # Extract fields (handle different schema versions)
        query = example.get("query") or example.get("doc") or ""
        code = example.get("code") or example.get("code_tokens", "")
        if isinstance(code, list):
            code = " ".join(code)
        docstring = example.get("docstring") or example.get("docstring_tokens", "")
        if isinstance(docstring, list):
            docstring = " ".join(docstring)

        label = int(example.get("label", 1))
        func_name = example.get("func_name", "")
        url = example.get("url", "")

        # Normalize
        code = normalize_code(code)
        # query_normalized not used currently but kept for future use
        _ = normalize_query(query)

        if not code or not query:
            continue

        # Generate IDs
        code_id = generate_code_id(code, idx)
        query_id = generate_query_id(query, idx)

        # Add to corpus (dedupe by code content)
        if code_id not in corpus:
            corpus[code_id] = CoSQACorpusEntry(
                code_id=code_id,
                code=code,
                docstring=docstring,
                func_name=func_name,
                url=url,
            )

        # Add to queries
        if query_id not in queries:
            queries[query_id] = CoSQAQuery(
                query_id=query_id,
                query=query,  # Keep original case
                relevant_code_ids=[],
            )

        # Add relevance judgment
        if label == 1:
            if query_id not in qrels:
                qrels[query_id] = {}
            qrels[query_id][code_id] = label
            queries[query_id].relevant_code_ids.append(code_id)

    print(f"Loaded {len(corpus)} code entries, {len(queries)} queries (fallback format)")

    return CoSQADataset(
        corpus=corpus,
        queries=queries,
        qrels=qrels,
        split=split,
        fallback_used=True,
    )



def save_dataset_cache(dataset: CoSQADataset, cache_path: Optional[Path] = None) -> Path:
    """Save processed dataset to local cache for faster loading."""
    path = cache_path or (CACHE_DIR / f"cosqa_{dataset.split}_processed.json")
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "cache_version": CACHE_VERSION,
        "split": dataset.split,
        "fallback_used": dataset.fallback_used,
        "corpus": {k: asdict(v) for k, v in dataset.corpus.items()},
        "queries": {k: asdict(v) for k, v in dataset.queries.items()},
        "qrels": dataset.qrels,
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved dataset cache (v{CACHE_VERSION}) to {path}")
    return path


def load_dataset_cache(cache_path: Optional[Path] = None, split: str = "test") -> Optional[CoSQADataset]:
    """Load processed dataset from local cache.

    Returns None if cache doesn't exist or has incompatible version.
    """
    path = cache_path or (CACHE_DIR / f"cosqa_{split}_processed.json")

    if not path.exists():
        return None

    try:
        with open(path) as f:
            data = json.load(f)

        # Check cache version - invalidate if different
        cached_version = data.get("cache_version", 1)
        if cached_version != CACHE_VERSION:
            print(f"Cache version mismatch (cached={cached_version}, current={CACHE_VERSION}), re-downloading...")
            return None

        corpus = {
            k: CoSQACorpusEntry(**v) for k, v in data.get("corpus", {}).items()
        }
        queries = {
            k: CoSQAQuery(**v) for k, v in data.get("queries", {}).items()
        }

        return CoSQADataset(
            corpus=corpus,
            queries=queries,
            qrels=data.get("qrels", {}),
            split=data.get("split", split),
            cache_version=cached_version,
            fallback_used=data.get("fallback_used", False),
        )
    except Exception as e:
        print(f"Failed to load cache: {e}")
        return None


def load_cosqa(
    split: str = "test",
    use_cache: bool = True,
    force_download: bool = False,
) -> CoSQADataset:
    """Load CoSQA dataset with optional caching.

    Args:
        split: Dataset split ("train", "validation", "test").
        use_cache: Whether to use local cache if available.
        force_download: Force re-download from HuggingFace.

    Returns:
        CoSQADataset ready for indexing and evaluation.
    """
    # Try cache first
    if use_cache and not force_download:
        cached = load_dataset_cache(split=split)
        if cached is not None:
            print(f"Loaded {len(cached.corpus)} entries from cache")
            return cached

    # Download from HuggingFace
    dataset = load_from_huggingface(split=split)

    # Cache for next time
    if use_cache:
        save_dataset_cache(dataset)

    return dataset


def get_corpus_for_indexing(dataset: CoSQADataset) -> List[Dict[str, Any]]:
    """Get corpus entries formatted for Qdrant indexing."""
    return [entry.to_index_payload() for entry in dataset.iter_corpus()]


def get_queries_for_evaluation(
    dataset: CoSQADataset,
    limit: Optional[int] = None,
) -> List[Tuple[str, str, List[str]]]:
    """Get queries with their relevant code IDs for evaluation.

    Returns:
        List of (query_id, query_text, relevant_code_ids) tuples.
    """
    result = []
    for query in dataset.iter_queries():
        if query.relevant_code_ids:  # Only queries with known relevant docs
            result.append((query.query_id, query.query, query.relevant_code_ids))

    if limit:
        result = result[:limit]

    return result
