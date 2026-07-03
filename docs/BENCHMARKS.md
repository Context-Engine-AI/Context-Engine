# Context Engine Retrieval Benchmarks

This document describes the retrieval evaluation methodology for Context Engine across three complementary benchmarks: **CoSQA**, **CoIR**, and **SWE-bench Retrieval**.

## Overview

| Benchmark | Task | Granularity | Query Type | Corpus |
|-----------|------|-------------|------------|--------|
| CoSQA | Text→Code | Function | Natural language | 20K Python functions |
| CoIR | Multi-task | Mixed | NL + Code | Multiple datasets |
| SWE-bench | Issue→Files | File/Function | GitHub issues | Real repositories |

## 1. CoSQA (Code Search Question Answering)

### Description

CoSQA evaluates natural language to code retrieval using web search queries paired with Python code answers. Unlike synthetic benchmarks, queries come from real Bing search logs.

### Dataset

- **Source**: `mteb/cosqa` on HuggingFace (MTEB-formatted version)
- **Queries**: 500 web search questions (test split)
- **Corpus**: 20,604 Python functions (full corpus, `split="test"`)
- **Labels**: Binary relevance (1 correct answer per query)
- **Collection**: `cosqa-corpus` (auto-created on first run)

**Note**: This is the **full mteb/cosqa corpus** (~20K snippets), not an in-repo subset. The corpus is downloaded once and cached in `~/.cache/cosqa/`.

### Methodology

```
1. Index corpus into Qdrant collection
2. For each query:
   a. Run hybrid search (dense + lexical + rerank)
   b. Retrieve top-K results
   c. Compare against ground truth
3. Compute metrics
```

### Metrics

| Metric | Description |
|--------|-------------|
| MRR | Mean Reciprocal Rank - average of 1/rank of first correct result |
| NDCG@K | Normalized Discounted Cumulative Gain at K |
| Recall@K | Fraction of relevant items found in top K |
| Hit@K | Binary: was the correct answer in top K? |

### Published Baselines

| Model | MRR |
|-------|-----|
| BoW | 0.065 |
| BM25 (Lucene) | 0.167 |
| CodeT5+ embedding | 0.266 |
| UniXcoder | 0.319 |
| CodeBERT | 0.392 |
| text-embedding-3-large | 0.393 |

### Usage

```bash
# Full benchmark (comparable to paper baselines)
python -m scripts.benchmarks.cosqa.runner --limit 500

# Dense-only mode (pure vector search, no hybrid scoring)
python -m scripts.benchmarks.cosqa.runner --limit 500 --dense-mode

# Enable LLM-based pseudo/tags generation during indexing
python -m scripts.benchmarks.cosqa.runner --limit 500 --enable-llm
```

#### CLI Flags

| Flag | Description |
|------|-------------|
| `--limit N` | Max results per query (default: 10) |
| `--corpus-limit N` | Index only N docs (0 = full corpus) |
| `--query-limit N` | Evaluate only N queries (0 = all) |
| `--no-rerank` | Disable cross-encoder reranking |
| `--dense-mode` | Pure vector search (disables lexical, symbol boosts, heuristics) |
| `--enable-llm` | Enable LLM pseudo/tags generation during indexing |
| `--debug` | Print detailed per-query debug output |
| `--output FILE` | Output JSON report path |

#### CoSQA ablation runs (refrag/mini, rerank, learning)

Use the helper script to run a consistent matrix across:
- rerank vs no rerank
- ReFRAG/mini vectors vs no ReFRAG
- learning vs no learning

```bash
# Default: 50/50 subset per run, new collection per variant
bash scripts/benchmarks/cosqa/run_ablation.sh

# Full corpus/queries (set 0 to disable limits)
CORPUS_LIMIT=0 QUERY_LIMIT=0 RUN_TAG=full \
  bash scripts/benchmarks/cosqa/run_ablation.sh
```

Outputs:
- JSON reports: `bench_results/cosqa/<RUN_TAG>/cosqa_<label>.json`
- Metadata: `bench_results/cosqa/<RUN_TAG>/cosqa_<label>_meta.json` (env snapshot, git sha, platform)
- Logs: `bench_results/cosqa/<RUN_TAG>/cosqa_<label>.log` (override with `LOG_DIR=...`)

Useful knobs (env):
- `QDRANT_URL`, `LEX_VECTOR_DIM`, `HYBRID_EXPAND`, `SEMANTIC_EXPANSION_ENABLED`
- `RUN_TAG`, `COLL_PREFIX`, `OUT_DIR`, `LOG_DIR`
- `COSQA_ENABLE_LEARNING` - Enable learning reranker (default: off for determinism)
- `EMBEDDING_SEED` - Seed for deterministic embeddings

#### Quick subset runs (smoke tests)

When you use `--corpus-limit`, the runner builds a corpus subset that includes
the relevant docs for the selected queries. This keeps small runs meaningful,
but results are **not** comparable to paper baselines. The report will include
`NOTE: subset evaluation (...)` when limits are active.

```bash
# Subset run (smoke test)
QDRANT_URL=http://localhost:6333 python3.11 -m scripts.benchmarks.cosqa.runner \
    --limit 500 --corpus-limit 500 --query-limit 500 --output cosqa_no_rerank.json --no-rerank

# Same subset with reranking enabled (for comparison)
QDRANT_URL=http://localhost:6333 python3.11 -m scripts.benchmarks.cosqa.runner \
    --limit 500 --corpus-limit 500 --query-limit 500 --output cosqa_rerank.json
```

#### Determinism & Reproducibility

CoSQA benchmarks are hardened for reproducibility:

- **Learning reranker disabled** by default (`RERANK_LEARNING=0`) to avoid score drift
- **Content-hash deduplication** prevents duplicate corpus entries across runs
- **Schema validation** fails fast on LEX_DIM mismatches (avoids silent scoring bugs)
- **Warmup query** runs before timing loop to exclude cold-start latency
- **Config fingerprinting** auto-detects stale collections when settings change
- **Metadata audit** saves full env snapshot, git sha, and platform info per run

---

## 2. CoIR (Code Information Retrieval)

### Description

CoIR is a comprehensive benchmark suite covering multiple code retrieval tasks across different query-document modalities.

### Tasks

| Task | Query | Document | Example |
|------|-------|----------|---------|
| Text→Code | Natural language | Code snippet | "sort list in python" → implementation |
| Code→Code | Code snippet | Code snippet | Find similar implementations |
| Code→Text | Code snippet | Documentation | Find docs for function |
| Hybrid | NL + Code | NL + Code | StackOverflow Q&A |

### Datasets

CoIR includes multiple constituent datasets:
- CodeSearchNet (6 languages)
- StackOverflow QA
- CodeFeedback
- Apps (competitive programming)
- SyntheticText2SQL

### Metrics

| Metric | Description |
|--------|-------------|
| NDCG@10 | Primary ranking metric |
| MRR | Mean Reciprocal Rank |
| Recall@K | Coverage at various K |

### Published Baselines

| Model | Avg NDCG@10 |
|-------|-------------|
| BM25 | 0.421 |
| CodeSage-large | 0.512 |
| Jina-Code-v2 | 0.534 |
| Voyage-Code-002 | 0.561 |
| CodeRankEmbed | 0.589 |

### Usage

```bash
# Run default tasks (cosqa, codesearchnet-python)
python -m scripts.benchmarks.coir.runner

# Run specific tasks
python -m scripts.benchmarks.coir.runner --tasks cosqa codesearchnet-python apps

# Quick test with subset
python -m scripts.benchmarks.coir.runner --tasks cosqa --query-limit 100 --corpus-limit 100

# Ablation sweep (rerank ± ReFRAG ± micro-chunks)
./scripts/benchmarks/coir/run_ablation.sh \
  TASKS="cosqa codesearchnet-python" \
  QUERY_LIMIT=200 \
  CORPUS_LIMIT=200 \
  OUT_DIR=bench_results/coir/$(date +%Y%m%d-%H%M%S)
```

**Note**: Requires `pip install coir-eval` for the external evaluation harness.
The CoIR runner uses Context-Engine's hybrid search directly (not the coir-eval
embedding-only DRES path), so rerank/expansion settings are honored.

---

## 3. SWE-bench Retrieval

### Description

SWE-bench Retrieval evaluates file localization for real GitHub issues. Given an issue description, can the retrieval system find the files that need to be modified?

This is a **prerequisite task** for the full SWE-bench: agents cannot fix issues in files they don't retrieve.

### Dataset

- **Source**: SWE-bench (Princeton)
- **Subsets**: Full (2,294), Lite (300), Verified (500)
- **Repositories**: Django, Astropy, Scikit-learn, Matplotlib, etc.
- **Ground Truth**: Files modified in the actual fix (extracted from patch)

### Methodology

```
For each instance:
1. Clone repository at base_commit
2. Index all source files into Qdrant
3. Search using issue text (problem_statement)
4. Compare retrieved files against patch files
5. Compute metrics
```

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Per-commit collections | Different commits have different file contents |
| File-level granularity | Matches practical agent workflows |
| Full repo indexing | Tests realistic retrieval conditions |

### Indexing Details

- **Collection naming**: `swe-bench-{repo}-{base_commit[:8]}` (commit-keyed)
- **Index reuse**: Collections are reused if same commit hash exists (`recreate=False`)
- **Cache**: Repos cached in `~/.cache/swe-bench/repos/`
- **Ground truth**: Extracted from patch via `--- a/path` and `+++ b/path` parsing

### Metrics

| Metric | Description |
|--------|-------------|
| Recall@K | Fraction of ground truth files found in top K |
| Precision@K | Fraction of top K that are ground truth |
| MRR | Reciprocal rank of first correct file |
| Pass@K | % of instances where ALL ground truth files are in top K |

### Related Work

**CoRNStack (UIUC/Nomic, 2024)** performs similar evaluation:
- 274 SWE-bench-Lite instances
- Function-level localization
- Reports File@1-3 and Function@5-10

| Model | File@1 | File@3 |
|-------|--------|--------|
| Agentless | 39.4% | 56.6% |
| CodeRankEmbed | ~55% | ~70% |
| + Reranker | ~58% | ~73% |

### Usage

```bash
# Quick test
python -m scripts.benchmarks.swe.runner --subset lite --limit 10

# Full evaluation  
python -m scripts.benchmarks.swe.runner --subset lite -o results.json

# With tuning (CLI flags for major features)
python -m scripts.benchmarks.swe.runner --subset lite --refrag --micro-chunks

# Optional weight tuning (env)
HYBRID_SYMBOL_BOOST=0.25 python -m scripts.benchmarks.swe.runner --subset lite
```

---

## Context Engine Features Tested

Each benchmark exercises different aspects of the retrieval pipeline:

| Feature | CoSQA | CoIR | SWE-bench |
|---------|-------|------|-----------|
| Dense embeddings | ✓ | ✓ | ✓ |
| Lexical (BM25-style) | ✓ | ✓ | ✓ |
| Hybrid RRF fusion | ✓ | ✓ | ✓ |
| Reranking | ✓ | ✓ | ✓ |
| AST chunking | - | - | ✓ |
| Symbol extraction | - | - | ✓ |
| Path scoring | - | - | ✓ |
| Multi-file context | - | - | ✓ |

---

## Tuning Knobs

All benchmarks respect Context Engine's configuration:

```bash
# Embedding
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5

# Hybrid search weights
HYBRID_DENSE_WEIGHT=0.7
HYBRID_LEXICAL_WEIGHT=0.3
HYBRID_SYMBOL_BOOST=0.15

# Reranking
RERANK_ENABLED=1
RERANK_TOP_N=50
RERANK_RETURN_M=20

# Indexing
INDEX_BATCH_SIZE=256
INDEX_CHUNK_LINES=120
```

Benchmark reports always log the effective config snapshot, including embedding
model, rerank on/off, expansion on/off, micro/semantic chunking, and ReFRAG.

---

## Reranker Configuration

### Candidate Pool Size

The reranker's effectiveness depends critically on the **candidate pool size**. By default, benchmarks retrieve 100 candidates and rerank down to the final limit (e.g., 10):

```python
# In CoSQA/CoIR search calls:
rerank_top_n=100    # Retrieve 100 candidates from hybrid search
rerank_return_m=10  # Rerank and return top 10
```

**Why this matters**: If you retrieve 10 and rerank 10, the reranker can only hurt—it has no room to promote better candidates. The candidate pool should be 5-20x the final result count.

| Setting | Candidate Pool | Expected Impact |
|---------|----------------|-----------------|
| `rerank_top_n=10` | 10 | No benefit, may hurt MRR |
| `rerank_top_n=50` | 50 | Moderate improvement |
| `rerank_top_n=100` | 100 | Good balance |
| `rerank_top_n=200` | 200 | Diminishing returns, slower |

### Reranker Input Text

The reranker (ONNX cross-encoder) scores `(query, document)` pairs. The document text is constructed from:

```python
# From scripts/rerank_local.py:prepare_pairs()
header = f"[{language}/{kind}] {symbol_path} — {path}"
doc = header + "\n" + metadata.code[:600]
```

**Key fields used**:
- `metadata.code` - The actual code content (truncated to 600 chars)
- `metadata.symbol_path` or `metadata.symbol` - Function/class name
- `metadata.path` - File path
- `metadata.language` - Programming language
- `metadata.kind` - Entity type (function, class, etc.)

**Common issue**: If `metadata.code` is missing or empty, the reranker sees only the header with no code content, degrading performance significantly.

### Verifying Reranker Input

To debug what the reranker sees:

```python
from scripts.rerank_local import prepare_pairs
from qdrant_client import QdrantClient

client = QdrantClient()
points, _ = client.scroll("cosqa-corpus", limit=5, with_payload=True)

pairs = prepare_pairs("sort a list", points)
for query, doc in pairs:
    print(f"Query: {query}")
    print(f"Doc: {doc[:200]}...")
    print("---")
```

---

## Ablation Experiments

### Recommended Ablation Runs

To isolate the impact of each pipeline component:

```bash
# 1. Dense only (no lexical, no rerank)
HYBRID_LEXICAL_WEIGHT=0 python -m scripts.benchmarks.cosqa.runner \
    --no-rerank --limit 500 -o ablation_dense_only.json

# 2. Hybrid (dense + lexical, no rerank)
python -m scripts.benchmarks.cosqa.runner \
    --no-rerank --limit 500 -o ablation_hybrid_no_rerank.json

# 3. Hybrid + Rerank (full pipeline)
python -m scripts.benchmarks.cosqa.runner \
    --limit 500 -o ablation_full.json

# 4. Rerank with varying candidate pools
for n in 25 50 100 200; do
    RERANK_TOP_N=$n python -m scripts.benchmarks.cosqa.runner \
        --limit 500 -o ablation_rerank_top_$n.json
done
```

### Re-indexing for Ablations

If you change indexing settings (embedding model, chunk size, etc.), you must re-index:

```bash
# Delete existing collection
python -c "from qdrant_client import QdrantClient; QdrantClient().delete_collection('cosqa-corpus')"

# Re-run benchmark (auto-reindexes)
python -m scripts.benchmarks.cosqa.runner --limit 500
```

For CoIR, use `--force-reindex`:
```bash
python -m scripts.benchmarks.coir.runner --tasks cosqa --force-reindex
```

---

## Running All Benchmarks

```bash
# 1. CoSQA (fast, ~10 min)
python -m scripts.benchmarks.cosqa.runner -o cosqa_results.json

# 2. CoIR (medium, ~30 min per task)
python -m scripts.benchmarks.coir.runner -o coir_results.json

# 3. SWE-bench (slow, ~2-3 hours for lite)
python3.11 -m scripts.benchmarks.swe.runner --subset lite -o swe_results.json
```

---

## Interpreting Results

### What good looks like

| Benchmark | Target | Interpretation |
|-----------|--------|----------------|
| CoSQA MRR | >0.35 | Competitive with CodeBERT |
| CoIR NDCG@10 | >0.50 | Above BM25 baseline |
| SWE Recall@10 | >0.60 | Find ground truth files 60%+ of time |
| SWE Pass@20 | >0.40 | Find ALL files in 40%+ of instances |

### What the benchmarks DON'T measure

- Agent reasoning quality
- Patch generation accuracy
- Test pass rate (full SWE-bench)
- Multi-turn retrieval refinement
- Memory/notes integration

---

## Context Engine Approach

### Architecture

Context Engine uses a **hybrid retrieval pipeline**:

```
Query → [Dense Encoder] → Vector Search ─┐
     → [Lexical Hash]  → Sparse Search ──┼→ RRF Fusion → Rerank → Results
     → [Symbol Boost]  → Metadata Filter ┘
```

### Key Differentiators

| Component | Standard RAG | Context Engine |
|-----------|--------------|----------------|
| Chunking | Fixed-size | AST-aware semantic |
| Embeddings | Single vector | Dense + lexical + mini |
| Search | Dense only | Hybrid RRF fusion |
| Ranking | Embedding similarity | Cross-encoder rerank |
| Metadata | None | Symbols, paths, language |

### Hypothesis

We hypothesize Context Engine will:
- **Outperform** pure BM25/dense on SWE-bench (hybrid helps with code identifiers)
- **Match** code-specific models on CoSQA (despite using general embeddings)
- **Excel** at multi-file localization (path/symbol metadata)

### Limitations

1. **General embeddings**: Using BGE rather than code-specific (CodeBERT, UniXcoder)
2. **Chunk granularity**: AST chunks may not align with function boundaries
3. **Reranker**: General-purpose, not code-trained

---

## Future Work

1. **Code-specific embeddings**: Evaluate CodeRankEmbed, Voyage-Code-002
2. **Function-level SWE-bench**: Match CoRNStack methodology exactly
3. **Agent integration**: Full SWE-bench with Claude + Context Engine
4. **Ablation studies**: Isolate impact of each pipeline component

---

## References

1. **CoSQA**: Huang et al., "CoSQA: 20,000+ Web Queries for Code Search and Question Answering" (ACL 2021)
2. **CoIR**: Li et al., "CoIR: A Comprehensive Benchmark for Code Information Retrieval Models" (2024)
3. **SWE-bench**: Jimenez et al., "SWE-bench: Can Language Models Resolve Real-World GitHub Issues?" (ICLR 2024)
4. **CoRNStack**: Suresh et al., "CoRNStack: High-Quality Contrastive Data for Better Code Ranking" (2024)
5. **Agentless**: Xia et al., "Agentless: Demystifying LLM-based Software Engineering Agents" (2024)
