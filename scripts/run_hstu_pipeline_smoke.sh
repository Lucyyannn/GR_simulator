#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

GENERATOR="${REPO_ROOT}/scripts/generate_hstu_baseline_trace.py"
SIMULATOR="${REPO_ROOT}/build/bin/Simulator"
CONFIG="${REPO_ROOT}/configs/systolic_ws_256x256_c8_simple_noc_ascend910c_hbm128_ddr5_12ch_768gb.json"

CASE_NAME="hstu_pipeline_smoke_l4_h512_hist1024_cand512_mb256_u1_upb1"
TRACE_OUTPUT="${REPO_ROOT}/example/trace_tests/${CASE_NAME}"
MODELS_LIST="${REPO_ROOT}/example/${CASE_NAME}_models_list.json"
RESULT_DIR="${REPO_ROOT}/results/hstu_pipeline_smoke/$(date +"%Y%m%d_%H%M%S")"

mkdir -p "${RESULT_DIR}"

echo "Generating smoke trace: ${CASE_NAME}"
python3 "${GENERATOR}" \
  --pipeline \
  --shared-trace \
  --compact-json \
  --layers 4 \
  --hidden 512 \
  --history-len 1024 \
  --vocab 65536 \
  --num-users 1 \
  --users-per-batch 1 \
  --candidates-per-user 512 \
  --macro-batch-size 256 \
  --op-modeling split=skip,view=skip,concat=skip \
  --output "${TRACE_OUTPUT}" \
  --models-list "${MODELS_LIST}"

run_case() {
  local pipeline_preload="$1"
  local label="$2"
  local log_path="${RESULT_DIR}/${label}.log"
  local time_path="${RESULT_DIR}/${label}.time"
  local report_path="${RESULT_DIR}/${label}_memory_report.json"

  echo "Running ${label} pipeline_preload=${pipeline_preload}"
    "${SIMULATOR}" \
      --config "${CONFIG}" \
      --models_list "${MODELS_LIST}" \
      --mode trace \
      --log_level info \
      --pipeline_preload "${pipeline_preload}" \
      --memory_report_json "${report_path}" \
    > "${log_path}" 2>&1 &
}

run_case false "baseline_no_pipeline"
run_case true "pipeline_preload"

# echo "DONE: ${RESULT_DIR}"
# echo "Wall-clock summary:"
# cat "${RESULT_DIR}/baseline_no_pipeline.time"
# cat "${RESULT_DIR}/pipeline_preload.time"
