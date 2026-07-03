#!/usr/bin/env bash
set -euo pipefail

# CoIR ablation runner (mirrors the CoSQA ablation helper)
# Runs multiple configurations against CoIR tasks and saves JSON + logs under bench_results/coir/<RUN_TAG>

PYTHON_BIN="${PYTHON_BIN:-python3.11}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
OUT_DIR="${OUT_DIR:-bench_results/coir/${RUN_TAG}}"
LOG_DIR="${LOG_DIR:-${OUT_DIR}}"
TASKS="${TASKS:-cosqa codesearchnet-python}"
QUERY_LIMIT="${QUERY_LIMIT:-200}"
CORPUS_LIMIT="${CORPUS_LIMIT:-200}"
BATCH_SIZE="${BATCH_SIZE:-64}"

BASE_ENV=(
  "LOG_LEVEL=${LOG_LEVEL:-INFO}"
  "HYBRID_EXPAND=${HYBRID_EXPAND:-1}"
  "SEMANTIC_EXPANSION_ENABLED=${SEMANTIC_EXPANSION_ENABLED:-1}"
  "HYBRID_IN_PROCESS=${HYBRID_IN_PROCESS:-1}"
  "RERANK_IN_PROCESS=${RERANK_IN_PROCESS:-1}"
  "LEX_VECTOR_DIM=${LEX_VECTOR_DIM:-4096}"
  "QDRANT_URL=${QDRANT_URL:-http://localhost:6333}"
)

BUF_CMD=()
if command -v stdbuf >/dev/null 2>&1; then
  BUF_CMD=(stdbuf -oL -eL)
fi

run_one() {
  local label="$1"
  local refrag="$2"
  local micro="$3"
  local rerank="$4"

  mkdir -p "${OUT_DIR}" "${LOG_DIR}"
  local output="${OUT_DIR}/coir_${label}.json"
  local log="${LOG_DIR}/coir_${label}.log"

  # Build task arguments
  read -r -a task_arr <<< "${TASKS}"
  local args=(
    "-m" "scripts.benchmarks.coir.runner"
    "--batch-size" "${BATCH_SIZE}"
    "--output-folder" "${OUT_DIR}"
    "--output" "${output}"
    "--tasks"
  )
  args+=("${task_arr[@]}")
  if [ "${QUERY_LIMIT}" -gt 0 ]; then
    args+=("--query-limit" "${QUERY_LIMIT}")
  fi
  if [ "${CORPUS_LIMIT}" -gt 0 ]; then
    args+=("--corpus-limit" "${CORPUS_LIMIT}")
  fi
  if [ "${rerank}" = "0" ]; then
    args+=("--no-rerank")
  fi

  echo "[run] ${label} (refrag=${refrag}, micro=${micro}, rerank=${rerank})"
  echo "      json: ${output}"
  echo "      log:  ${log}"
  (
    export "${BASE_ENV[@]}"
    export PYTHONUNBUFFERED=1
    export REFRAG_MODE="${refrag}"
    export INDEX_MICRO_CHUNKS="${micro}"
    "${PYTHON_BIN}" "${args[@]}"
  ) 2>&1 \
    | "${BUF_CMD[@]}" tr '\r' '\n' \
    | "${PYTHON_BIN}" -u -c $'import sys,time\nfor line in sys.stdin:\n    sys.stdout.write(time.strftime("[%H:%M:%S] ") + line)\n    sys.stdout.flush()' \
    | tee "${log}"

  echo "[ok] ${output}"
  echo "     log: ${log}"
}

run_one "norerank_norefrag" 0 0 0
run_one "norerank_refrag" 1 1 0
run_one "rerank_norefrag" 0 0 1
run_one "rerank_refrag" 1 1 1
