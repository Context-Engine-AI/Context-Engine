# Context-Engine Benchmark Suite

Reproducible retrieval quality benchmarks with Hit@k, MRR, and latency metrics.

## Quick Start

### 1. Clone the public snapshot

```bash
python bench/clone_snapshot.py --manifest bench/datasets/public_v1.json
```

This clones kubernetes@v1.28.0, vscode@1.86.0, and transformers@v4.37.0 to `bench/data/`.

### 2. Index into Qdrant

```bash
# Index each repo (adjust paths as needed)
python scripts/ingest_code.py --root bench/data/kubernetes_kubernetes --collection ctx-bench-k8s
python scripts/ingest_code.py --root bench/data/microsoft_vscode --collection ctx-bench-vscode
python scripts/ingest_code.py --root bench/data/huggingface_transformers --collection ctx-bench-hf

# Or create a combined collection
python scripts/ingest_code.py --root bench/data --collection ctx-bench-public-v1
```

### 3. Run quality evaluation

```bash
# Single model, single config
python bench/eval_quality.py \
  --collection ctx-bench-public-v1 \
  --model "BAAI/bge-base-en-v1.5" \
  --gold-file bench/gold/public_v1.queries.jsonl
```

**Output:**
```
MRR: 0.5800
Hit@1: 0.4500
Hit@5: 0.7200
Hit@10: 0.8000
```

### 4. Run full matrix comparison

```bash
# Compare models
python bench/run_matrix.py \
  --src ctx-bench-public-v1 \
  --gold-file bench/gold/public_v1.queries.jsonl \
  --models "BAAI/bge-base-en-v1.5,sentence-transformers/all-MiniLM-L6-v2"

# Compare models × configs (A/B testing)
python bench/run_matrix.py \
  --src ctx-bench-public-v1 \
  --gold-file bench/gold/public_v1.queries.jsonl \
  --config-file bench/configs/ctx_ab_configs.json \
  --models "BAAI/bge-base-en-v1.5" \
  --output results/bench_matrix.json
```

## Files

| File | Purpose |
|------|---------|
| `datasets/public_v1.json` | Snapshot manifest (repos + pinned refs) |
| `gold/public_v1.queries.jsonl` | 30 gold queries with expected file paths |
| `configs/ctx_ab_configs.json` | A/B config variants (rerank on/off, RRF weights, etc.) |
| `clone_snapshot.py` | Clone repos at pinned commits |
| `eval_quality.py` | Compute Hit@k, MRR against gold set |
| `eval.py` | Latency harness (p50/p95, Jaccard overlap) |
| `run_matrix.py` | Orchestrate (model × config) comparison |
| `copy-coll.py` | Clone/re-embed collections for model comparison |

## Gold Query Format

```jsonl
{"id": "k8s-eviction-001", "query": "where is pod eviction logic", "repo": "kubernetes/kubernetes", "relevant": [{"path": "pkg/kubelet/eviction/eviction_manager.go"}], "task_type": "navigation"}
```

## Metrics

- **Hit@k**: Did any of the top-k results match expected files?
- **MRR**: Mean Reciprocal Rank (1/rank of first correct result)
- **Latency p50/p95**: Response time percentiles

## Adding Your Own Queries

Edit `bench/gold/public_v1.queries.jsonl` or create a new file:

```bash
python bench/eval_quality.py \
  --collection my-collection \
  --model "BAAI/bge-base-en-v1.5" \
  --gold-file bench/gold/my_queries.jsonl
```
