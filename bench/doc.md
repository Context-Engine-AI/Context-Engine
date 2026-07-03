
# Bench: Copy/Re-embed Collections

This folder contains small utilities to help benchmark different embedding models against the same underlying code/payload.

## What we did (repeatable)

We took an existing Qdrant collection (example: `Context-Engine-41e67959`) and produced two benchmark collections:

- `bench-bge`
  - Recreated with the **BGE** dense vector params
  - Populated by **cloning** points (payload + dense vectors) from the source collection
  - Lexical (`lex`) + mini (`mini`) vectors are copied when present

- `bench-minilm`
  - Recreated with the **MiniLM** dense vector params (384-d)
  - Populated by **re-embedding** dense vectors from the source payload text using `sentence-transformers/all-MiniLM-L6-v2`
  - Lexical (`lex`) + mini (`mini`) vectors are copied when present (or optionally regenerated)

This produces apples-to-apples retrieval comparisons where only the **dense embedding model** differs.

## Script

- `bench/copy-coll.py`
  - Recreates destination collections with correct vector schemas
  - Clones or re-embeds data in batches with progress output
  - Uses the same vector-name conventions as the core code (`scripts.utils.sanitize_vector_name`)

## Prereqs

- Qdrant running (the dev-remote compose stack is fine)
- The `mcp_indexer` container has Python deps (`qdrant-client`, `fastembed`, `onnxruntime`)
- Source collection exists in Qdrant

## How to run (dev-remote docker compose)

### Rebuild (required after updating `bench/`)

If you changed anything under `bench/` (including `copy-coll.py`), rebuild the indexer image so `/app/bench` exists in the container:

```bash
docker compose -f docker-compose.dev-remote.yml up -d --build mcp_indexer
```

Run inside the `mcp_indexer` container so model caches and deps match indexing/search:

```bash
docker compose -f docker-compose.dev-remote.yml exec -T mcp_indexer sh -lc '
  export QDRANT_URL=http://qdrant:6333
  export QDRANT_TIMEOUT=180
  python /app/bench/copy-coll.py \
    --src Context-Engine-41e67959 \
    --bge-dest bench-bge \
    --minilm-dest bench-minilm
'
```

If your source collection has missing `lex`/`mini` vectors and you want to fill them:

```bash
docker compose -f docker-compose.dev-remote.yml exec -T mcp_indexer sh -lc '
  export QDRANT_URL=http://qdrant:6333
  export QDRANT_TIMEOUT=180
  python /app/bench/copy-coll.py \
    --src Context-Engine-41e67959 \
    --fill-missing-lex \
    --fill-missing-mini
'
```

### Useful knobs

- `--scroll-batch N`
  - Qdrant scroll page size
- `--embed-batch N`
  - MiniLM embed batch size
- `--upsert-batch N`
  - Upsert batch size
- `--upsert-wait`
  - Wait for each upsert; slower but easier to reason about during debugging

## Why MiniLM can take longer here

Even though MiniLM is a smaller model (384-d vs 768-d), in this workflow MiniLM has to:

- Scroll every point
- Extract payload text
- Run the embedding model for every point
- Upsert new dense vectors

Whereas BGE cloning is mostly:

- Scroll points
- Upsert vectors/payload as-is

So **copying** will usually beat **re-embedding**, regardless of which model you re-embed with.

If you want a fair “model speed” comparison, measure:

- Embedding throughput alone (texts/sec)
- Query latency on already-built collections

## Notes on `lex` and `mini`

- `lex` is a fixed-size lexical hashing vector (default dim `4096`).
- `mini` is a compact vector used for gating in ReFRAG-style mode (default dim `64`).
- Some source points may not have `lex`/`mini` populated (we saw this in `Context-Engine-41e67959`).
  - The script safely skips `None` vectors.
  - Optionally you can fill missing vectors via `--fill-missing-lex` / `--fill-missing-mini`.

## Evaluation loop (BGE vs MiniLM)

The simplest way to compare the two collections is to run the same query against each and measure latency.

Example (run in `mcp_indexer`):

```bash
docker compose -f docker-compose.dev-remote.yml exec -T mcp_indexer sh -lc '
  export QDRANT_URL=http://qdrant:6333
  export QDRANT_TIMEOUT=180

  for C in bench-bge bench-minilm; do
    echo "\n== $C =="
    /usr/bin/time -f "elapsed=%Es cpu=%P maxrss=%MKB" \
      python /app/scripts/hybrid_search.py \
        --collection "$C" \
        --limit 8 \
        -q "how does repo_search combine dense and lexical" \
        -q "where is sanitize_vector_name defined" \
        -q "how is multi repo collection name derived" \
        --json \
      >/tmp/${C}.json
    echo "saved /tmp/${C}.json"
  done
'
```

Notes:

- This uses the existing hybrid search path (dense + lexical + optional boosts).
- For more stable numbers, run multiple times and take p50/p95.

### Preferred: `bench/eval.py` (warmup + repeats + overlap)

`hybrid_search.py` uses `EMBEDDING_MODEL` for **query embedding**, so for a fair comparison we run it twice:

- `bench-bge` with `EMBEDDING_MODEL=BAAI/bge-base-en-v1.5`
- `bench-minilm` with `EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2`

The harness below does:

- warmup runs (discarded)
- N measured runs per target
- p50/p95 latency
- top-k overlap (Jaccard on result paths) between collections

```bash
docker compose -f docker-compose.dev-remote.yml exec -T mcp_indexer sh -lc '
  export QDRANT_URL=http://qdrant:6333
  export QDRANT_TIMEOUT=180
  python /app/bench/eval.py \
    --target bench-bge BAAI/bge-base-en-v1.5 \
    --target bench-minilm sentence-transformers/all-MiniLM-L6-v2 \
    --query "how does repo_search combine dense and lexical" \
    --query "where is sanitize_vector_name defined" \
    --query "how is multi repo collection name derived" \
    --repeats 10 \
    --warmup 1 \
    --limit 8 \
    --topk 8 \
    --per-path 1 \
    --json-out /tmp/bench_eval.json
'
```

To load queries from a file (one query per line):

```bash
python /app/bench/eval.py ... --query-file /work/.codebase/bench_queries.txt
```

To pass extra flags through to `hybrid_search.py`, you must append them after `--` (otherwise `eval.py` errors):

```bash
python /app/bench/eval.py ... -- --expand
```

## How to decide “ship MiniLM for low-end devices”

Treat this as a multi-metric decision with explicit gates.

### Performance gates

- **Query p50/p95**
  - Compare `bench-bge` vs `bench-minilm` on a warmed system (warmup then repeats)
  - Goal for “low-end”: meaningfully lower p95 (tail latency matters most)

- **Embedding throughput (queries/sec)**
  - Measure standalone query embedding speed for BGE vs MiniLM (no Qdrant)
  - Goal: MiniLM should be faster per query on CPU

- **Memory footprint**
  - Measure RSS of the process serving search (and the embedding model) on a constrained machine
  - Goal: MiniLM should reduce RSS enough to matter

### Quality gates

- **Hit@k against a small gold set**
  - Write ~30–100 representative queries and expected file/symbol hints
  - Score “did an expected path appear in top-k?”
  - Gate: MiniLM should not regress more than an agreed threshold (e.g., <= 5% absolute hit@k loss)

- **Overlap sanity check**
  - The harness reports top-k path overlap. Overlap being high is good, but it’s not a substitute for a gold set.

### Recommendation for next iteration

- Create a stable `bench_queries.txt` and run `bench/eval.py --query-file ...` with `--repeats 20 --warmup 2`.
- Add a separate embedding-only microbench (we can add `bench/embed_bench.py`) to isolate model speed from Qdrant/search overhead.

## Future work (not doing now)

The work so far was **plumbing / dirty validation**:

- Can we build separate collections per embedding model?
- Can we run the same query set against both and see stable-ish timings?

To make a defensible call like “MiniLM is good/fast enough to ship for low-end devices”, add a proper evaluation workflow:

### 1) Separate benchmarks by component

- **Embedding-only throughput**
  - Measure texts/sec and p50/p95 embed latency for each model (no Qdrant).
  - This is the true model CPU cost; it should be the main win for MiniLM.

- **Query-time latency (warm)**
  - Use `bench/eval.py` with warmups and higher repeats (e.g. `--warmup 2 --repeats 50`).
  - Report p50/p95, not just p50.

- **Index-time throughput**
  - Measure indexing wall time, chunks/sec, and peak memory.

### 2) Memory + footprint gates

- **Process RSS**
  - Measure RSS for the embedding process (and MCP server) with each model.
  - Low-end devices are often memory-bound.

- **Model size / cache impact**
  - Track on-disk cache size and first-run download time.

### 3) Quality gates

- **Gold-set hit@k**
  - Maintain a query file plus expected paths/symbols.
  - Score hit@k (e.g. hit@8) for each model.
  - Define an acceptable regression threshold.

- **Manual review slice**
  - Pick a smaller set of representative “hard” queries and manually compare top-k.

### 4) Decide with explicit thresholds

- **Latency**
  - “Ship for low-end” target might be: p95 improves by X% and doesn’t exceed Y seconds.
- **Quality**
  - hit@k within threshold; no major regressions on critical queries.
- **Footprint**
  - RSS reduction is meaningful on the target hardware.

### Exact rebuild+eval command used (copy/paste)

(Legacy; the preferred approach is `bench/eval.py` above.)

This is the exact command that rebuilt `mcp_indexer`, verified `/app/bench`, and ran a 3-repeat eval against both `bench-bge` and `bench-minilm`:

```bash
docker compose -f docker-compose.dev-remote.yml up -d --build mcp_indexer && \
docker compose -f docker-compose.dev-remote.yml exec -T mcp_indexer sh -lc '
set -e
ls -al /app/bench
export QDRANT_URL=http://qdrant:6333
export QDRANT_TIMEOUT=180
export BENCH_REPEATS=${BENCH_REPEATS:-3}
python - <<"PY"
import os, subprocess, sys, time, statistics

collections = ["bench-bge", "bench-minilm"]
queries = [
    "how does repo_search combine dense and lexical",
    "where is sanitize_vector_name defined",
    "how is multi repo collection name derived",
]
repeats = int(os.environ.get("BENCH_REPEATS", "3") or 3)

print("QDRANT_URL", os.environ.get("QDRANT_URL"))
print("repeats", repeats)

for coll in collections:
    times = []
    for r in range(repeats):
        cmd = [sys.executable, "/app/scripts/hybrid_search.py", "--collection", coll, "--limit", "8", "--json"]
        for q in queries:
            cmd.extend(["-q", q])
        out_path = f"/tmp/{coll}.run{r}.json"
        t0 = time.time()
        with open(out_path, "w", encoding="utf-8") as f:
            subprocess.run(cmd, check=True, stdout=f, env=os.environ.copy())
        dt = time.time() - t0
        times.append(dt)
        print(f"{coll} run{r}: {dt:.3f}s -> {out_path}")

    p50 = statistics.median(times)
    print(f"{coll} summary: n={len(times)} p50={p50:.3f}s min={min(times):.3f}s max={max(times):.3f}s")
PY
'
```


