#!/usr/bin/env python3
"""
Training infrastructure for Tiny Recursive Reranker.

Implements TRM-style deep supervision training:
1. Generate training data from search logs or synthetic examples
2. Train with loss at each refinement iteration (deep supervision)
3. Use pairwise ranking loss (margin ranking)
4. Support weight saving/loading

Usage:
    # Generate synthetic training data
    python scripts/rerank_train.py --generate-data --output data/rerank_train.jsonl

    # Train the model
    python scripts/rerank_train.py --train --data data/rerank_train.jsonl --epochs 100

    # Evaluate
    python scripts/rerank_train.py --evaluate --data data/rerank_test.jsonl
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import argparse


@dataclass
class TrainingExample:
    """A single training example for pairwise ranking."""
    query: str
    doc_positive: str  # Should rank higher
    doc_negative: str  # Should rank lower
    # Optional: relevance scores for regression loss
    score_positive: float = 1.0
    score_negative: float = 0.0


@dataclass
class TrainingConfig:
    """Configuration for training."""
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    margin: float = 0.5  # Margin for pairwise loss
    n_iterations: int = 3  # Refinement iterations
    deep_supervision_weight: float = 0.5  # Weight for intermediate losses
    dim: int = 256
    hidden_dim: int = 512
    weight_decay: float = 0.01
    save_every: int = 10  # Save checkpoint every N epochs


class SimpleGradientDescent:
    """
    Simple gradient descent optimizer with momentum.

    We implement our own to avoid PyTorch/TensorFlow dependency.
    """

    def __init__(self, lr: float = 0.001, momentum: float = 0.9, weight_decay: float = 0.01):
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocities: Dict[str, np.ndarray] = {}

    def step(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Update parameters using gradients."""
        updated = {}
        for name, param in params.items():
            if name not in grads:
                updated[name] = param
                continue

            grad = grads[name]

            # Weight decay (L2 regularization)
            grad = grad + self.weight_decay * param

            # Momentum
            if name not in self.velocities:
                self.velocities[name] = np.zeros_like(param)

            self.velocities[name] = self.momentum * self.velocities[name] - self.lr * grad
            updated[name] = param + self.velocities[name]

        return updated


class TrainableTinyScorer:
    """
    Tiny scorer with gradient computation for training.

    Uses numerical gradients for simplicity (analytical gradients would be faster).
    """

    def __init__(self, dim: int = 256, hidden_dim: int = 512, seed: int = 42):
        self.dim = dim
        self.hidden_dim = hidden_dim

        # Initialize weights
        np.random.seed(seed)
        scale = np.sqrt(2.0 / (dim * 3))  # He initialization
        self.params = {
            "W1": np.random.randn(dim * 3, hidden_dim).astype(np.float32) * scale,
            "b1": np.zeros(hidden_dim, dtype=np.float32),
            "W2": np.random.randn(hidden_dim, 1).astype(np.float32) * np.sqrt(2.0 / hidden_dim),
            "b2": np.zeros(1, dtype=np.float32),
        }

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Forward pass with cached activations for backprop.

        x: (batch, dim*3) concatenated [query, doc, latent]
        returns: (batch,) scores, cache dict
        """
        # Layer 1: Linear + ReLU
        z1 = x @ self.params["W1"] + self.params["b1"]  # (batch, hidden)
        h1 = np.maximum(0, z1)  # ReLU

        # Layer 2: Linear
        z2 = h1 @ self.params["W2"] + self.params["b2"]  # (batch, 1)
        scores = z2.squeeze(-1)  # (batch,)

        cache = {"x": x, "z1": z1, "h1": h1}
        return scores, cache

    def backward(self, dscores: np.ndarray, cache: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Backward pass to compute gradients.

        dscores: (batch,) gradient of loss w.r.t. scores
        returns: dict of gradients for each parameter
        """
        batch_size = dscores.shape[0]

        # Reshape for matrix ops
        dz2 = dscores.reshape(-1, 1)  # (batch, 1)

        # Layer 2 gradients
        dW2 = cache["h1"].T @ dz2  # (hidden, 1)
        db2 = dz2.sum(axis=0)  # (1,)
        dh1 = dz2 @ self.params["W2"].T  # (batch, hidden)

        # ReLU backward
        dz1 = dh1 * (cache["z1"] > 0).astype(np.float32)  # (batch, hidden)

        # Layer 1 gradients
        dW1 = cache["x"].T @ dz1  # (dim*3, hidden)
        db1 = dz1.sum(axis=0)  # (hidden,)

        # Average over batch
        grads = {
            "W1": dW1 / batch_size,
            "b1": db1 / batch_size,
            "W2": dW2 / batch_size,
            "b2": db2 / batch_size,
        }
        return grads

    def save(self, path: str):
        """Save model weights to file."""
        np.savez(path, **self.params)

    def load(self, path: str):
        """Load model weights from file."""
        data = np.load(path)
        for key in self.params:
            if key in data:
                self.params[key] = data[key]


class TrainableLatentRefiner:
    """Latent refiner with gradient support."""

    def __init__(self, dim: int = 256, hidden_dim: int = 256, seed: int = 43):
        self.dim = dim
        self.hidden_dim = hidden_dim

        np.random.seed(seed)
        scale = np.sqrt(2.0 / (dim * 3))
        self.params = {
            "W1": np.random.randn(dim * 3, hidden_dim).astype(np.float32) * scale,
            "b1": np.zeros(hidden_dim, dtype=np.float32),
            "W2": np.random.randn(hidden_dim, dim).astype(np.float32) * np.sqrt(2.0 / hidden_dim),
            "b2": np.zeros(dim, dtype=np.float32),
        }

    def forward(self, z: np.ndarray, query: np.ndarray, doc_summary: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """Refine latent state."""
        x = np.concatenate([z, query, doc_summary], axis=-1)
        h = np.maximum(0, x @ self.params["W1"] + self.params["b1"])
        z_new = h @ self.params["W2"] + self.params["b2"]
        z_refined = alpha * z_new + (1 - alpha) * z
        # Normalize
        norm = np.linalg.norm(z_refined, axis=-1, keepdims=True) + 1e-8
        return z_refined / norm

    def save(self, path: str):
        np.savez(path, **self.params)

    def load(self, path: str):
        data = np.load(path)
        for key in self.params:
            if key in data:
                self.params[key] = data[key]


def margin_ranking_loss(score_pos: np.ndarray, score_neg: np.ndarray, margin: float = 0.5) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Pairwise margin ranking loss.

    Loss = max(0, margin - (score_pos - score_neg))

    Returns: (loss_value, grad_pos, grad_neg)
    """
    diff = score_pos - score_neg
    violations = margin - diff
    loss = np.maximum(0, violations)

    # Gradient: -1 for pos, +1 for neg when violating
    mask = (violations > 0).astype(np.float32)
    grad_pos = -mask
    grad_neg = mask

    return loss.mean(), grad_pos, grad_neg


def deep_supervision_loss(
    scores_per_iter: List[np.ndarray],
    labels: np.ndarray,
    weight: float = 0.5
) -> Tuple[float, List[np.ndarray]]:
    """
    Deep supervision: compute loss at each iteration.

    From TRM paper: train model to improve answer at each step.
    Later iterations get higher weight.

    scores_per_iter: List of (batch,) score arrays, one per iteration
    labels: (batch,) ground truth relevance
    weight: how much to weight intermediate losses vs final

    Returns: (total_loss, list of gradients per iteration)
    """
    n_iters = len(scores_per_iter)
    total_loss = 0.0
    grads = []

    for i, scores in enumerate(scores_per_iter):
        # Later iterations get higher weight
        iter_weight = (i + 1) / n_iters
        if i < n_iters - 1:
            iter_weight *= weight

        # MSE loss for simplicity
        diff = scores - labels
        loss = (diff ** 2).mean()
        grad = 2 * diff / len(diff) * iter_weight

        total_loss += loss * iter_weight
        grads.append(grad)

    return total_loss, grads


class RecursiveRerankerTrainer:
    """
    Trainer for the recursive reranker with deep supervision.
    """

    def __init__(self, config: TrainingConfig):
        self.config = config

        # Initialize models
        self.scorer = TrainableTinyScorer(dim=config.dim, hidden_dim=config.hidden_dim)
        self.refiner = TrainableLatentRefiner(dim=config.dim)

        # Optimizers
        self.scorer_opt = SimpleGradientDescent(
            lr=config.learning_rate, weight_decay=config.weight_decay
        )
        self.refiner_opt = SimpleGradientDescent(
            lr=config.learning_rate, weight_decay=config.weight_decay
        )

        # Training state
        self.epoch = 0
        self.losses: List[float] = []

    def _encode_text(self, texts: List[str]) -> np.ndarray:
        """Simple bag-of-chars encoding (placeholder for real embeddings)."""
        result = []
        for text in texts:
            # Hash-based pseudo-embedding
            np.random.seed(hash(text) % (2**32))
            vec = np.random.randn(self.config.dim).astype(np.float32)
            vec = vec / (np.linalg.norm(vec) + 1e-8)
            result.append(vec)
        return np.array(result, dtype=np.float32)

    def train_step(self, examples: List[TrainingExample]) -> float:
        """Single training step on a batch of examples."""
        batch_size = len(examples)

        # Encode texts
        queries = self._encode_text([ex.query for ex in examples])
        docs_pos = self._encode_text([ex.doc_positive for ex in examples])
        docs_neg = self._encode_text([ex.doc_negative for ex in examples])

        # Initialize latent states
        z_pos = queries.copy()
        z_neg = queries.copy()

        # Collect scores per iteration for deep supervision
        scores_pos_per_iter = []
        scores_neg_per_iter = []
        caches_pos = []
        caches_neg = []

        # Forward pass through all iterations
        for i in range(self.config.n_iterations):
            # Score positive docs
            x_pos = np.concatenate([queries, docs_pos, z_pos], axis=1)
            s_pos, cache_pos = self.scorer.forward(x_pos)
            scores_pos_per_iter.append(s_pos)
            caches_pos.append(cache_pos)

            # Score negative docs
            x_neg = np.concatenate([queries, docs_neg, z_neg], axis=1)
            s_neg, cache_neg = self.scorer.forward(x_neg)
            scores_neg_per_iter.append(s_neg)
            caches_neg.append(cache_neg)

            # Refine latent states
            if i < self.config.n_iterations - 1:
                z_pos = self.refiner.forward(z_pos, queries, docs_pos)
                z_neg = self.refiner.forward(z_neg, queries, docs_neg)

        # Compute loss with deep supervision
        total_loss = 0.0
        scorer_grads_accumulated = {k: np.zeros_like(v) for k, v in self.scorer.params.items()}

        for i in range(self.config.n_iterations):
            # Pairwise margin loss at each iteration
            iter_weight = (i + 1) / self.config.n_iterations
            if i < self.config.n_iterations - 1:
                iter_weight *= self.config.deep_supervision_weight

            loss, grad_pos, grad_neg = margin_ranking_loss(
                scores_pos_per_iter[i],
                scores_neg_per_iter[i],
                self.config.margin
            )
            total_loss += loss * iter_weight

            # Backprop through scorer
            grads_pos = self.scorer.backward(grad_pos * iter_weight, caches_pos[i])
            grads_neg = self.scorer.backward(grad_neg * iter_weight, caches_neg[i])

            for k in scorer_grads_accumulated:
                scorer_grads_accumulated[k] += grads_pos[k] + grads_neg[k]

        # Update scorer parameters
        self.scorer.params = self.scorer_opt.step(self.scorer.params, scorer_grads_accumulated)

        return total_loss

    def train(self, examples: List[TrainingExample], val_examples: Optional[List[TrainingExample]] = None):
        """Full training loop."""
        n_batches = (len(examples) + self.config.batch_size - 1) // self.config.batch_size

        for epoch in range(self.config.epochs):
            self.epoch = epoch
            epoch_loss = 0.0

            # Shuffle examples
            indices = np.random.permutation(len(examples))

            for batch_idx in range(n_batches):
                start = batch_idx * self.config.batch_size
                end = min(start + self.config.batch_size, len(examples))
                batch_indices = indices[start:end]
                batch = [examples[i] for i in batch_indices]

                loss = self.train_step(batch)
                epoch_loss += loss

            epoch_loss /= n_batches
            self.losses.append(epoch_loss)

            # Logging
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: loss={epoch_loss:.4f}")

            # Save checkpoint
            if (epoch + 1) % self.config.save_every == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch+1}")

    def save_checkpoint(self, name: str):
        """Save model checkpoint."""
        save_dir = Path("models/rerank_recursive")
        save_dir.mkdir(parents=True, exist_ok=True)

        self.scorer.save(str(save_dir / f"{name}_scorer.npz"))
        self.refiner.save(str(save_dir / f"{name}_refiner.npz"))

        # Save training state
        state = {
            "epoch": self.epoch,
            "losses": self.losses,
            "config": {
                "dim": self.config.dim,
                "hidden_dim": self.config.hidden_dim,
                "n_iterations": self.config.n_iterations,
            }
        }
        with open(save_dir / f"{name}_state.json", "w") as f:
            json.dump(state, f)

        print(f"Saved checkpoint: {name}")

    def load_checkpoint(self, name: str):
        """Load model checkpoint."""
        save_dir = Path("models/rerank_recursive")
        self.scorer.load(str(save_dir / f"{name}_scorer.npz"))
        self.refiner.load(str(save_dir / f"{name}_refiner.npz"))

        with open(save_dir / f"{name}_state.json") as f:
            state = json.load(f)
        self.epoch = state["epoch"]
        self.losses = state["losses"]
        print(f"Loaded checkpoint: {name}")



def generate_synthetic_data(n_examples: int = 1000, output_path: str = "data/rerank_train.jsonl"):
    """
    Generate synthetic training data for the reranker.

    Creates pairwise examples where:
    - Positive doc contains query terms
    - Negative doc is unrelated
    """
    # Code-like queries and documents
    queries = [
        "hybrid search implementation",
        "recursive reranker training",
        "embedding model initialization",
        "cache manager eviction policy",
        "file watcher debounce",
        "MCP server tool registration",
        "vector similarity search",
        "document indexing pipeline",
        "query expansion techniques",
        "relevance scoring function",
    ]

    positive_templates = [
        "def {keyword}(query): # Implements {keyword} for search",
        "class {Keyword}Manager: # Handles {keyword} operations",
        "async def run_{keyword}(): # Execute {keyword} pipeline",
        "# {keyword} configuration and setup\nconfig = load_{keyword}_config()",
        "def test_{keyword}(): # Unit tests for {keyword}",
    ]

    negative_templates = [
        "def unrelated_function(): return 42",
        "class DatabaseConnection: # Connect to database",
        "import os, sys, json  # Standard imports",
        "# Configuration file for logging\nLOG_LEVEL = 'INFO'",
        "def helper_util(): pass  # Utility function",
    ]

    examples = []

    for i in range(n_examples):
        # Pick a query
        query = queries[i % len(queries)]
        keyword = query.split()[0].lower()
        Keyword = keyword.capitalize()

        # Generate positive doc (contains query terms)
        pos_template = positive_templates[i % len(positive_templates)]
        pos_doc = pos_template.format(keyword=keyword, Keyword=Keyword)

        # Generate negative doc (unrelated)
        neg_doc = negative_templates[i % len(negative_templates)]

        example = {
            "query": query,
            "doc_positive": pos_doc,
            "doc_negative": neg_doc,
            "score_positive": 1.0,
            "score_negative": 0.0,
        }
        examples.append(example)

    # Save to file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    print(f"Generated {n_examples} training examples -> {output_path}")
    return examples


def load_training_data(path: str) -> List[TrainingExample]:
    """Load training examples from JSONL file."""
    examples = []
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            examples.append(TrainingExample(
                query=data["query"],
                doc_positive=data["doc_positive"],
                doc_negative=data["doc_negative"],
                score_positive=data.get("score_positive", 1.0),
                score_negative=data.get("score_negative", 0.0),
            ))
    return examples


def export_trained_weights(checkpoint_name: str, output_dir: str = "models/rerank_recursive"):
    """
    Export trained weights to the rerank_recursive module.

    Copies the .npz files to where RecursiveReranker can load them.
    """
    import shutil

    src_dir = Path("models/rerank_recursive")
    dst_dir = Path(output_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    # Copy scorer weights
    scorer_src = src_dir / f"{checkpoint_name}_scorer.npz"
    scorer_dst = dst_dir / "scorer_weights.npz"
    if scorer_src.exists():
        shutil.copy(scorer_src, scorer_dst)
        print(f"Exported scorer weights -> {scorer_dst}")

    # Copy refiner weights
    refiner_src = src_dir / f"{checkpoint_name}_refiner.npz"
    refiner_dst = dst_dir / "refiner_weights.npz"
    if refiner_src.exists():
        shutil.copy(refiner_src, refiner_dst)
        print(f"Exported refiner weights -> {refiner_dst}")


def main():
    parser = argparse.ArgumentParser(description="Train Tiny Recursive Reranker")
    parser.add_argument("--generate-data", action="store_true", help="Generate synthetic training data")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the model")
    parser.add_argument("--export", type=str, help="Export checkpoint weights")
    parser.add_argument("--data", type=str, default="data/rerank_train.jsonl", help="Training data path")
    parser.add_argument("--output", type=str, default="data/rerank_train.jsonl", help="Output path for generated data")
    parser.add_argument("--n-examples", type=int, default=1000, help="Number of examples to generate")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--checkpoint", type=str, help="Checkpoint to load/export")

    args = parser.parse_args()

    if args.generate_data:
        generate_synthetic_data(n_examples=args.n_examples, output_path=args.output)

    elif args.train:
        # Load data
        if not Path(args.data).exists():
            print(f"Training data not found: {args.data}")
            print("Run with --generate-data first")
            return

        examples = load_training_data(args.data)
        print(f"Loaded {len(examples)} training examples")

        # Create trainer
        config = TrainingConfig(
            epochs=args.epochs,
            learning_rate=args.lr,
            batch_size=args.batch_size,
        )
        trainer = RecursiveRerankerTrainer(config)

        # Load checkpoint if specified
        if args.checkpoint:
            try:
                trainer.load_checkpoint(args.checkpoint)
            except Exception as e:
                print(f"Could not load checkpoint: {e}")

        # Train
        trainer.train(examples)
        trainer.save_checkpoint("final")
        print("Training complete!")

    elif args.export:
        export_trained_weights(args.export or "final")

    elif args.evaluate:
        if not Path(args.data).exists():
            print(f"Evaluation data not found: {args.data}")
            return

        examples = load_training_data(args.data)

        # Load trained model
        config = TrainingConfig()
        trainer = RecursiveRerankerTrainer(config)

        try:
            trainer.load_checkpoint(args.checkpoint or "final")
        except Exception as e:
            print(f"Could not load checkpoint: {e}")
            print("Training with random weights...")

        # Evaluate: compute accuracy (positive > negative)
        correct = 0
        total = 0

        for ex in examples[:100]:  # Sample
            q = trainer._encode_text([ex.query])[0]
            d_pos = trainer._encode_text([ex.doc_positive])[0]
            d_neg = trainer._encode_text([ex.doc_negative])[0]

            z = q.copy()

            # Run through iterations
            for _ in range(config.n_iterations):
                x_pos = np.concatenate([q, d_pos, z])
                x_neg = np.concatenate([q, d_neg, z])

                s_pos, _ = trainer.scorer.forward(x_pos.reshape(1, -1))
                s_neg, _ = trainer.scorer.forward(x_neg.reshape(1, -1))

                z = trainer.refiner.forward(z.reshape(1, -1), q.reshape(1, -1), d_pos.reshape(1, -1))[0]

            if s_pos[0] > s_neg[0]:
                correct += 1
            total += 1

        accuracy = correct / total if total > 0 else 0
        print(f"Pairwise accuracy: {accuracy:.2%} ({correct}/{total})")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()

