#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# OPD-V evaluation script
#
# Supported benchmarks:
#   vstar, zoombench, hrbench-4k, hrbench-8k, mme-realworld, mme-realworld-cn
#
# Usage:
#   API_BASE="http://localhost:8000/v1/" \
#   OPENAI_MODEL_ID="OPD-V" \
#   BENCHMARK="vstar,zoombench,hrbench-4k,hrbench-8k,mme-realworld,mme-realworld-cn" \
#   bash eval/run_eval.sh
# =============================================================================

export MKL_SERVICE_FORCE_INTEL=1
PYTHON="${PYTHON:-python3}"

# --- Required ---
API_BASE="${API_BASE:?ERROR: API_BASE must be set (e.g. http://localhost:8000/v1/)}"
OPENAI_MODEL_ID="${OPENAI_MODEL_ID:?ERROR: OPENAI_MODEL_ID must be set}"

# --- Optional ---
BENCHMARK="${BENCHMARK:-vstar}"
API_KEY="${OPENAI_API_KEY:-EMPTY}"
MODEL_NAME="${MODEL_NAME:-${OPENAI_MODEL_ID//\//_}}"
SEED="${SEED:-42}"
MAX_TOKENS="${MAX_TOKENS:-32768}"
OUT_DIR="${OUT_DIR:-model_answer}"
JUDGE_DIR="${JUDGE_DIR:-judge}"
EVAL_DATA_DIR="${EVAL_DATA_DIR:-}"
MAX_RETRIES="${MAX_RETRIES:-3}"
PARALLEL_WORKERS="${PARALLEL_WORKERS:-256}"
ENABLE_THINKING="${ENABLE_THINKING:-}"

JUDGE_API_BASE="${JUDGE_API_BASE:-}"
JUDGE_API_KEY="${JUDGE_API_KEY:-}"
JUDGE_MODEL="${JUDGE_MODEL:-}"
JUDGE_MODEL_PATH="${JUDGE_MODEL_PATH:-}"
JUDGE_MAX_TOKENS="${JUDGE_MAX_TOKENS:-2048}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DATA_DIR="${EVAL_DATA_DIR:-${SCRIPT_DIR}}"
mkdir -p "${EVAL_DATA_DIR}"
cd "${EVAL_DATA_DIR}"
echo "Eval data dir: ${EVAL_DATA_DIR}"

# Benchmark JSON mapping
declare -A BENCHMARK_JSON_MAP=(
  [zoombench]="zoombench.json"
  [vstar]="vstar.json"
  [hrbench-4k]="hr_bench_4k.json"
  [hrbench-8k]="hr_bench_8k.json"
  [mme-realworld]="MME_RealWorld.json"
  [mme-realworld-cn]="MME_RealWorld_CN.json"
)

# Parse comma-separated benchmarks
IFS=',' read -r -a BENCHMARKS_TO_RUN <<< "${BENCHMARK}"

run_single_benchmark() {
  local bench="$1"
  local bench_json="${BENCHMARK_JSON_MAP[$bench]:-}"
  if [[ -z "${bench_json}" ]]; then
    echo "ERROR: Unsupported benchmark: ${bench}"
    exit 1
  fi

  echo "=========================================="
  echo "Benchmark: ${bench}"
  echo "API base: ${API_BASE}"
  echo "Model: ${OPENAI_MODEL_ID}"
  echo "=========================================="

  local model_tag="${MODEL_NAME}_seed${SEED}"

  # [1/4] Prepare data
  echo "[1/4] Preparing data..."
  "${PYTHON}" "${SCRIPT_DIR}/prepare_data.py" --benchmark "${bench}" --data_dir "${EVAL_DATA_DIR}"

  # [2/4] Inference
  echo "[2/4] Running inference..."
  local -a INFER_ARGS=(
    --benchmark "${bench}"
    --benchmark_json "${EVAL_DATA_DIR}/${bench_json}"
    --out_dir "${OUT_DIR}"
    --model_name "${model_tag}"
    --seed "${SEED}"
    --api_base "${API_BASE}"
    --api_key "${API_KEY}"
    --model_id "${OPENAI_MODEL_ID}"
    --max_tokens "${MAX_TOKENS}"
    --max_retries "${MAX_RETRIES}"
    --parallel_workers "${PARALLEL_WORKERS}"
  )
  [[ -n "${ENABLE_THINKING}" ]] && INFER_ARGS+=(--enable_thinking "${ENABLE_THINKING}")

  "${PYTHON}" "${SCRIPT_DIR}/infer.py" "${INFER_ARGS[@]}"

  # [3/4] Judge
  echo "[3/4] Running judge..."
  local -a JUDGE_ARGS=()
  [[ -n "${JUDGE_API_BASE}" ]] && JUDGE_ARGS+=(--api_base "${JUDGE_API_BASE}")
  [[ -n "${JUDGE_API_KEY}" ]] && JUDGE_ARGS+=(--api_key "${JUDGE_API_KEY}")
  [[ -n "${JUDGE_MODEL}" ]] && JUDGE_ARGS+=(--judge_model "${JUDGE_MODEL}")
  [[ -n "${JUDGE_MODEL_PATH}" ]] && JUDGE_ARGS+=(--judge_model_path "${JUDGE_MODEL_PATH}")
  [[ -n "${JUDGE_MAX_TOKENS}" ]] && JUDGE_ARGS+=(--judge_max_tokens "${JUDGE_MAX_TOKENS}")

  local judge_benchmark="${bench}"
  local judge_model_tag="${model_tag}"
  local judge_json="${JUDGE_DIR}/${bench}/${model_tag}_answer.jsonl"

  "${PYTHON}" "${SCRIPT_DIR}/judge_qwenlm.py" \
    --benchmark "${judge_benchmark}" \
    --model "${judge_model_tag}" \
    --answer_dir "${OUT_DIR}" \
    --judge_dir "${JUDGE_DIR}" \
    "${JUDGE_ARGS[@]}"

  # [4/4] Accuracy
  echo "[4/4] Calculating accuracy..."
  "${PYTHON}" "${SCRIPT_DIR}/cal_acc.py" \
    --benchmark "${bench}" \
    --judge_json "${judge_json}" \
    --benchmark_json "${EVAL_DATA_DIR}/${bench_json}"

  echo "Done: ${bench}"
}

# Main loop
echo "Benchmarks to run: ${BENCHMARKS_TO_RUN[*]}"
total="${#BENCHMARKS_TO_RUN[@]}"
for idx in "${!BENCHMARKS_TO_RUN[@]}"; do
  bench="${BENCHMARKS_TO_RUN[$idx]}"
  bench="$(echo "${bench}" | xargs)"  # trim spaces
  [[ -z "${bench}" ]] && continue
  echo
  echo "########## [$((idx + 1))/${total}] ${bench} ##########"
  run_single_benchmark "${bench}"
done
