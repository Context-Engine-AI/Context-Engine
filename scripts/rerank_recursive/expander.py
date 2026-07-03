"""
QueryExpander - Learns query expansions (synonyms/related terms) from usage patterns.
"""
import os
import re
from typing import Any, Dict, List, Tuple

import numpy as np

from scripts.rerank_recursive.utils import _COMMON_TOKENS


class QueryExpander:
    """
    Learns query expansions (synonyms/related terms) from usage patterns.

    Observes which terms co-occur with successful retrievals and builds
    a lightweight term→expansion mapping per-collection.
    """

    WEIGHTS_DIR = os.environ.get("RERANKER_WEIGHTS_DIR", "/tmp/rerank_weights")
    MAX_EXPANSIONS_PER_TERM = 5
    MIN_CONFIDENCE = 0.3
    DECAY_RATE = 0.995

    def __init__(self, lr: float = 0.1):
        self.lr = lr
        self._collection = "default"
        self._weights_path = self._get_weights_path("default")
        self.expansions: Dict[str, Dict[str, float]] = {}
        self._update_count = 0
        self._version = 0

        if os.path.exists(self._weights_path):
            try:
                self._load_weights()
            except Exception:
                pass

    @staticmethod
    def _sanitize_collection(collection: str) -> str:
        return "".join(c if c.isalnum() or c in "-_" else "_" for c in collection)

    def _get_weights_path(self, collection: str) -> str:
        safe_name = self._sanitize_collection(collection)
        return os.path.join(self.WEIGHTS_DIR, f"expander_{safe_name}.json")

    def set_collection(self, collection: str):
        self._collection = collection
        self._weights_path = self._get_weights_path(collection)
        if os.path.exists(self._weights_path):
            try:
                self._load_weights()
            except Exception:
                pass

    def _load_weights(self):
        import json
        import fcntl
        lock_path = self._weights_path + ".lock"
        os.makedirs(os.path.dirname(lock_path) or ".", exist_ok=True)
        with open(lock_path, "w") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_SH)
            with open(self._weights_path, "r") as f:
                data = json.load(f)
            self.expansions = data.get("expansions", {})
            self._version = data.get("version", 0)
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _save_weights(self):
        import json
        import fcntl
        os.makedirs(os.path.dirname(self._weights_path) or ".", exist_ok=True)
        lock_path = self._weights_path + ".lock"
        tmp_path = self._weights_path + ".tmp"
        with open(lock_path, "w") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            with open(tmp_path, "w") as f:
                json.dump({"expansions": self.expansions, "version": self._version}, f)
            os.replace(tmp_path, self._weights_path)
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _tokenize(self, text: str) -> List[str]:
        tokens = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', text.lower())
        return [t for t in tokens if len(t) > 2 and t not in _COMMON_TOKENS]

    def expand(self, query: str, max_expansions: int = 3) -> List[str]:
        query_tokens = set(self._tokenize(query))
        candidates: List[Tuple[str, float]] = []
        for token in query_tokens:
            if token in self.expansions:
                for exp_term, conf in self.expansions[token].items():
                    if exp_term not in query_tokens and conf >= self.MIN_CONFIDENCE:
                        candidates.append((exp_term, conf))
        candidates.sort(key=lambda x: -x[1])
        return [term for term, _ in candidates[:max_expansions]]

    def learn_from_teacher(
        self,
        query: str,
        doc_texts: List[str],
        teacher_scores: np.ndarray,
    ):
        query_tokens = set(self._tokenize(query))
        if not query_tokens:
            return

        weights = np.exp(teacher_scores - teacher_scores.max())
        weights = weights / (weights.sum() + 1e-8)

        doc_term_weights: Dict[str, float] = {}
        for doc_text, weight in zip(doc_texts, weights):
            for token in self._tokenize(doc_text):
                if token not in query_tokens:
                    doc_term_weights[token] = doc_term_weights.get(token, 0.0) + weight

        for query_term in query_tokens:
            if query_term not in self.expansions:
                self.expansions[query_term] = {}

            term_expansions = self.expansions[query_term]

            for exp in list(term_expansions.keys()):
                term_expansions[exp] = float(term_expansions[exp] * self.DECAY_RATE)
                if term_expansions[exp] < 0.01:
                    del term_expansions[exp]

            for doc_term, weight in doc_term_weights.items():
                if weight > 0.1:
                    old_conf = term_expansions.get(doc_term, 0.0)
                    new_conf = old_conf + self.lr * (weight - old_conf)
                    term_expansions[doc_term] = float(min(new_conf, 1.0))

            if len(term_expansions) > self.MAX_EXPANSIONS_PER_TERM * 2:
                sorted_exp = sorted(term_expansions.items(), key=lambda x: -x[1])
                self.expansions[query_term] = dict(sorted_exp[:self.MAX_EXPANSIONS_PER_TERM])

        self._update_count += 1
        if self._update_count % 20 == 0:
            self._version += 1
            self._save_weights()

    def get_stats(self) -> Dict[str, Any]:
        total_terms = len(self.expansions)
        total_expansions = sum(len(v) for v in self.expansions.values())
        avg_expansions = total_expansions / max(total_terms, 1)
        return {"terms": total_terms, "expansions": total_expansions, "avg_per_term": avg_expansions, "version": self._version}
