#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3.11}"
COLL_PREFIX="${COLL_PREFIX:-cosqa-ablate}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
OUT_DIR="${OUT_DIR:-bench_results/cosqa/${RUN_TAG}}"
LOG_DIR="${LOG_DIR:-${OUT_DIR}}"
CORPUS_LIMIT="${CORPUS_LIMIT:-50}"
QUERY_LIMIT="${QUERY_LIMIT:-50}"
RECREATE="${RECREATE:-1}"

BASE_ENV=(
  "LOG_LEVEL=${LOG_LEVEL:-INFO}"
  "DEBUG_HYBRID_SEARCH=${DEBUG_HYBRID_SEARCH:-0}"
  "HYBRID_EXPAND=${HYBRID_EXPAND:-1}"
  "SEMANTIC_EXPANSION_ENABLED=${SEMANTIC_EXPANSION_ENABLED:-1}"
  "HYBRID_SYMBOL_BOOST=${HYBRID_SYMBOL_BOOST:-0.35}"
  "RERANK_IN_PROCESS=${RERANK_IN_PROCESS:-1}"
  "LEX_VECTOR_DIM=${LEX_VECTOR_DIM:-4096}"
  "QDRANT_URL=${QDRANT_URL:-http://localhost:6333}"
)

run_one() {
  local label="$1"
  local refrag="$2"
  local micro="$3"
  local rerank="$4"
  local learning="$5"
  local collection="${COLL_PREFIX}-${label}-${RUN_TAG}"
  local output="${OUT_DIR}/cosqa_${label}.json"
  local log="${LOG_DIR}/cosqa_${label}.log"

  local args=(
    "-m" "scripts.benchmarks.cosqa.runner"
    "--collection" "${collection}"
    "--output" "${output}"
  )
  if [ "${CORPUS_LIMIT}" -gt 0 ]; then
    args+=("--corpus-limit" "${CORPUS_LIMIT}")
  fi
  if [ "${QUERY_LIMIT}" -gt 0 ]; then
    args+=("--query-limit" "${QUERY_LIMIT}")
  fi
  if [ "${RECREATE}" = "1" ]; then
    args+=("--recreate")
  fi
  if [ "${rerank}" = "0" ]; then
    args+=("--no-rerank")
  fi
  if [ "${learning}" = "1" ]; then
    args+=("--learning-worker")
  fi

  mkdir -p "${OUT_DIR}" "${LOG_DIR}"
  (
    export "${BASE_ENV[@]}"
    export REFRAG_MODE="${refrag}"
    export INDEX_MICRO_CHUNKS="${micro}"
    export RERANK_LEARNING="${learning}"
    export RERANK_EVENTS_ENABLED="${learning}"
    "${PYTHON_BIN}" "${args[@]}"
  ) > "${log}" 2>&1

  echo "[ok] ${output}"
  echo "     log: ${log}"
}

run_one "norerank_norefrag" 0 0 0 0
run_one "norerank_refrag" 1 1 0 0
run_one "rerank_norefrag" 0 0 1 0
run_one "rerank_refrag" 1 1 1 0
run_one "rerank_norefrag_learning" 0 0 1 1
run_one "rerank_refrag_learning" 1 1 1 1
