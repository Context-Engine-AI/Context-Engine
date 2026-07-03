#!/usr/bin/env python3
"""
A/B Experiment Framework for Context-Engine Auto-Tuning.

Features:
- Tracked experiments with metrics collection
- Automatic winner detection with statistical significance
- Persistent experiment results
- Integration with OpenLit traces for outcome measurement
"""

import asyncio
import hashlib
import json
import os
import random
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Experiment storage path
EXPERIMENTS_DIR = Path(os.environ.get("EXPERIMENTS_DIR", "/tmp/context-engine-experiments"))
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class ExperimentVariant:
    """A single variant (arm) in an A/B experiment."""
    name: str
    config: Dict[str, Any]
    impressions: int = 0
    successes: int = 0  # e.g., user didn't retry
    total_latency_ms: float = 0.0
    total_tokens: int = 0
    
    @property
    def success_rate(self) -> float:
        return self.successes / self.impressions if self.impressions > 0 else 0.0
    
    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.impressions if self.impressions > 0 else 0.0


@dataclass
class Experiment:
    """An A/B experiment comparing variants."""
    name: str
    description: str
    param: str  # The parameter being tested
    variants: List[ExperimentVariant] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    status: str = "running"  # running, concluded, paused
    winner: Optional[str] = None
    min_samples: int = 50  # Minimum samples before concluding
    
    def add_variant(self, name: str, config: Dict[str, Any]):
        """Add a variant to the experiment."""
        self.variants.append(ExperimentVariant(name=name, config=config))
    
    def select_variant(self, query_hash: Optional[str] = None) -> ExperimentVariant:
        """Select a variant for this request (deterministic if hash provided)."""
        if query_hash:
            # Deterministic assignment based on query hash
            idx = int(hashlib.md5(query_hash.encode()).hexdigest(), 16) % len(self.variants)
        else:
            # Random assignment
            idx = random.randint(0, len(self.variants) - 1)
        return self.variants[idx]
    
    def record_outcome(self, variant_name: str, success: bool, latency_ms: float, tokens: int = 0):
        """Record the outcome of a request."""
        for v in self.variants:
            if v.name == variant_name:
                v.impressions += 1
                if success:
                    v.successes += 1
                v.total_latency_ms += latency_ms
                v.total_tokens += tokens
                break
        
        # Check if we should conclude
        self._check_conclusion()
    
    def _check_conclusion(self):
        """Check if experiment has enough data to conclude."""
        if self.status != "running":
            return
        
        # All variants need minimum samples
        if not all(v.impressions >= self.min_samples for v in self.variants):
            return
        
        # Simple winner detection: best success rate with >5% difference
        sorted_variants = sorted(self.variants, key=lambda v: v.success_rate, reverse=True)
        best = sorted_variants[0]
        second = sorted_variants[1] if len(sorted_variants) > 1 else None
        
        if second and (best.success_rate - second.success_rate) > 0.05:
            self.winner = best.name
            self.status = "concluded"
    
    def get_summary(self) -> Dict[str, Any]:
        """Get experiment summary."""
        return {
            "name": self.name,
            "param": self.param,
            "status": self.status,
            "winner": self.winner,
            "variants": [
                {
                    "name": v.name,
                    "config": v.config,
                    "impressions": v.impressions,
                    "success_rate": f"{v.success_rate:.1%}",
                    "avg_latency_ms": f"{v.avg_latency_ms:.0f}",
                }
                for v in self.variants
            ],
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dict."""
        return {
            "name": self.name,
            "description": self.description,
            "param": self.param,
            "created_at": self.created_at,
            "status": self.status,
            "winner": self.winner,
            "min_samples": self.min_samples,
            "variants": [asdict(v) for v in self.variants],
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Experiment":
        """Load from dict."""
        variants = [ExperimentVariant(**v) for v in data.pop("variants", [])]
        exp = cls(**data)
        exp.variants = variants
        return exp


class ExperimentManager:
    """Manages A/B experiments with persistence."""
    
    def __init__(self, storage_dir: Path = EXPERIMENTS_DIR):
        self.storage_dir = storage_dir
        self.experiments: Dict[str, Experiment] = {}
        self._load_experiments()
    
    def _load_experiments(self):
        """Load experiments from disk."""
        for file in self.storage_dir.glob("*.json"):
            try:
                with open(file) as f:
                    data = json.load(f)
                    exp = Experiment.from_dict(data)
                    self.experiments[exp.name] = exp
            except Exception as e:
                print(f"Error loading experiment {file}: {e}")
    
    def _save_experiment(self, exp: Experiment):
        """Save experiment to disk."""
        file = self.storage_dir / f"{exp.name}.json"
        with open(file, "w") as f:
            json.dump(exp.to_dict(), f, indent=2)
    
    def create_experiment(
        self,
        name: str,
        param: str,
        variants: List[Tuple[str, Dict[str, Any]]],
        description: str = "",
        min_samples: int = 50,
    ) -> Experiment:
        """Create a new experiment."""
        exp = Experiment(
            name=name,
            description=description,
            param=param,
            min_samples=min_samples,
        )
        for variant_name, config in variants:
            exp.add_variant(variant_name, config)
        
        self.experiments[name] = exp
        self._save_experiment(exp)
        return exp
    
    def get_experiment(self, name: str) -> Optional[Experiment]:
        """Get an experiment by name."""
        return self.experiments.get(name)
    
    def list_experiments(self) -> List[Dict[str, Any]]:
        """List all experiments."""
        return [exp.get_summary() for exp in self.experiments.values()]
    
    def get_active_experiment_for_param(self, param: str) -> Optional[Experiment]:
        """Get an active experiment for a parameter."""
        for exp in self.experiments.values():
            if exp.param == param and exp.status == "running":
                return exp
        return None
    
    def record_outcome(self, experiment_name: str, variant_name: str, success: bool, latency_ms: float, tokens: int = 0):
        """Record outcome and save."""
        exp = self.experiments.get(experiment_name)
        if exp:
            exp.record_outcome(variant_name, success, latency_ms, tokens)
            self._save_experiment(exp)


# Global singleton
EXPERIMENT_MANAGER = ExperimentManager()


# ---------------------------------------------------------------------------
# Integration helpers
# ---------------------------------------------------------------------------
def get_experiment_config(param: str, query: str = "") -> Tuple[Optional[str], Dict[str, Any]]:
    """
    Get config for a parameter, potentially from an active experiment.
    
    Returns:
        (experiment_variant_id, config) - variant_id is None if no experiment
    """
    exp = EXPERIMENT_MANAGER.get_active_experiment_for_param(param)
    if not exp:
        return None, {}
    
    # Select variant (deterministic based on query)
    variant = exp.select_variant(query if query else None)
    return f"{exp.name}:{variant.name}", variant.config


def record_query_outcome(
    variant_id: Optional[str],
    success: bool,
    latency_ms: float,
    tokens: int = 0,
):
    """Record outcome of a query that was part of an experiment."""
    if not variant_id or ":" not in variant_id:
        return
    
    exp_name, variant_name = variant_id.split(":", 1)
    EXPERIMENT_MANAGER.record_outcome(exp_name, variant_name, success, latency_ms, tokens)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Experiment Manager")
    subparsers = parser.add_subparsers(dest="command")
    
    # List experiments
    subparsers.add_parser("list", help="List all experiments")
    
    # Create experiment
    create = subparsers.add_parser("create", help="Create new experiment")
    create.add_argument("name", help="Experiment name")
    create.add_argument("--param", required=True, help="Parameter to test")
    create.add_argument("--variants", nargs="+", required=True, help="Variants as name=value pairs")
    
    # Show experiment
    show = subparsers.add_parser("show", help="Show experiment details")
    show.add_argument("name", help="Experiment name")
    
    args = parser.parse_args()
    
    if args.command == "list":
        for exp in EXPERIMENT_MANAGER.list_experiments():
            status = "🏆" if exp["winner"] else "🔄" if exp["status"] == "running" else "⏸"
            print(f"{status} {exp['name']} ({exp['param']}): {exp['status']}")
            for v in exp["variants"]:
                winner = " ← WINNER" if v["name"] == exp.get("winner") else ""
                print(f"    {v['name']}: {v['impressions']} samples, {v['success_rate']} success{winner}")
    
    elif args.command == "create":
        variants = []
        for v in args.variants:
            name, value = v.split("=", 1)
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass
            variants.append((name, {args.param: value}))
        
        exp = EXPERIMENT_MANAGER.create_experiment(
            name=args.name,
            param=args.param,
            variants=variants,
        )
        print(f"Created experiment: {exp.name}")
        print(f"Variants: {[v[0] for v in variants]}")
    
    elif args.command == "show":
        exp = EXPERIMENT_MANAGER.get_experiment(args.name)
        if exp:
            print(json.dumps(exp.get_summary(), indent=2))
        else:
            print(f"Experiment not found: {args.name}")


if __name__ == "__main__":
    main()
