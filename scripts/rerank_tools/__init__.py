"""
rerank_tools - Reranker training, evaluation, and benchmarking utilities.

This package contains tools for training, evaluating, and benchmarking the
recursive reranker system. The core reranker logic is in scripts/rerank_recursive/.

Modules:
- eval: Offline evaluation pipeline (MRR/Recall/latency)
- events: Training event logging for background processing
- train: Training infrastructure for TinyScorer
- query: Query reranking utilities
- local: Local reranking CLI
- ab_test: A/B testing framework
- benchmark: Real-world benchmark harness
"""
from __future__ import annotations

# Core evaluation
from .eval import (
    EvalResult,
    EvalSummary,
    get_candidates,
    compute_mrr,
    compute_recall_at_k,
    run_eval,
    print_summary,
)

# Event logging
from .events import (
    log_training_event,
    list_event_files,
    read_events,
    cleanup_old_events,
    RERANK_EVENTS_DIR,
    RERANK_EVENTS_ENABLED,
    RERANK_EVENTS_SAMPLE_RATE,
    RERANK_EVENTS_RETENTION_DAYS,
)

# Training
from .train import (
    TrainingExample,
    TrainingConfig,
    TrainableTinyScorer,
    TrainableLatentRefiner,
    RecursiveRerankerTrainer,
    margin_ranking_loss,
    deep_supervision_loss,
)

__all__ = [
    # Eval
    "EvalResult",
    "EvalSummary",
    "get_candidates",
    "compute_mrr",
    "compute_recall_at_k",
    "run_eval",
    "print_summary",
    # Events
    "log_training_event",
    "list_event_files",
    "read_events",
    "cleanup_old_events",
    "RERANK_EVENTS_DIR",
    "RERANK_EVENTS_ENABLED",
    "RERANK_EVENTS_SAMPLE_RATE",
    "RERANK_EVENTS_RETENTION_DAYS",
    # Training
    "TrainingExample",
    "TrainingConfig",
    "TrainableTinyScorer",
    "TrainableLatentRefiner",
    "RecursiveRerankerTrainer",
    "margin_ranking_loss",
    "deep_supervision_loss",
]
