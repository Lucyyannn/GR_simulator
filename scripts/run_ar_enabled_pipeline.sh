#!/usr/bin/env bash
set -euo pipefail

ROOT="results/ar_enabled_20260903"
WAIT_PID=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root) ROOT="$2"; shift 2 ;;
    --wait-pid) WAIT_PID="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

CAL_MATRIX="${ROOT}/calibration_matrix"
CALIBRATION="${ROOT}/item_kv_calib.json"
WBOTH_ROOT="${ROOT}/w_both_matrix"
DSE_ROOT="${ROOT}/npu_dse"

wait_for_pid() {
  local pid="$1"
  while kill -0 "${pid}" 2>/dev/null; do sleep 30; done
}

run_calibration_matrix() {
  python3 scripts/run_hstu_calibration_matrix.py \
    --result-root "${CAL_MATRIX}" \
    --calibration configs/item_kv_calib.json \
    --base-methods w_AR --resume \
    --max-concurrent 196 --max-simulator-rss-gib 460 \
    --memory-headroom-gib 10 --log-level warn
}

run_wboth_matrix() {
  python3 scripts/run_meta_hstu_full_matrix.py \
    --result-root "${WBOTH_ROOT}" \
    --chips 910A 910B 910C MTIA2 --models small middle large \
    --seq-lens 4096 6144 8192 --batch-sizes 1 2 4 \
    --users hot cold --methods w_both \
    --calibration "${CALIBRATION}" \
    --max-concurrent 196 --max-total-simulators 190 \
    --max-simulator-rss-gib 460 --memory-headroom-gib 10 \
    --lock-prefix ar_enabled_wboth --log-level warn
}

run_pilot_matrix() {
  python3 scripts/run_npu_guard_matrix.py \
    --candidate-config-root "${DSE_ROOT}/pilot_candidates/configs" \
    --result-root "${DSE_ROOT}/pilot_results" \
    --calibration "${CALIBRATION}" --workload-set pilot \
    --models small middle large --max-concurrent 196 \
    --max-total-simulators 190 --max-simulator-rss-gib 460 \
    --memory-headroom-gib 10 --log-level warn
}

run_final_matrix() {
  python3 scripts/run_npu_guard_matrix.py \
    --candidate-config-root "${DSE_ROOT}/finalists/configs" \
    --result-root "${DSE_ROOT}/full_results" \
    --calibration "${DSE_ROOT}/item_kv_calib_npu.json" \
    --workload-set full --models small middle large \
    --max-concurrent 196 --max-total-simulators 190 \
    --max-simulator-rss-gib 460 --memory-headroom-gib 10 \
    --log-level warn
}

retry_three() {
  local label="$1"
  shift
  for attempt in 1 2 3; do
    if "$@"; then return 0; fi
    echo "${label}: attempt ${attempt} failed" >&2
  done
  echo "${label}: failed after three attempts" >&2
  return 1
}

mkdir -p "${ROOT}"
if [[ -n "${WAIT_PID}" ]]; then wait_for_pid "${WAIT_PID}"; fi

retry_three calibration_matrix run_calibration_matrix

python3 scripts/calibrate_item_kv_hardware.py \
  "${CAL_MATRIX}/w_both_ratio" --output "${CALIBRATION}" \
  --workers 18

retry_three w_both_matrix run_wboth_matrix

python3 scripts/search_npu_reconfiguration.py \
  --baselines configs/910A.json configs/910B.json configs/910C.json configs/MTIA2.json \
  --calibration "${CALIBRATION}" \
  --output-root "${DSE_ROOT}/analytic_initial" \
  --nc-min 1 --nc-max 64 --nv-min 1 --nv-max 128 \
  --wv-min 1024 --wv-max 8192 --wv-step 1024 \
  --cube-compression 1 --top-k 100 --min-area-utilization 0.95 \
  --ignore-power --models small middle large \
  --seq-lens 4096 6144 8192 --batch-sizes 1 2 4 \
  --users hot cold --min-predicted-speedup 0.95

python3 scripts/prepare_npu_w_both_candidates.py \
  --search-root "${DSE_ROOT}/analytic_initial" \
  --output-root "${DSE_ROOT}/pilot_candidates" \
  --per-chip 30 --top-overall 10 --top-vector-multiple 10 \
  --balance-bins 9 --cube-compression 1

retry_three pilot_matrix run_pilot_matrix

python3 scripts/calibrate_npu_reconfiguration.py \
  --base-calibration "${CALIBRATION}" \
  --candidate-config-root "${DSE_ROOT}/pilot_candidates/configs" \
  --result-root "${DSE_ROOT}/pilot_results" \
  --output "${DSE_ROOT}/item_kv_calib_npu.json"

python3 scripts/search_npu_reconfiguration.py \
  --baselines configs/910A.json configs/910B.json configs/910C.json configs/MTIA2.json \
  --calibration "${DSE_ROOT}/item_kv_calib_npu.json" \
  --output-root "${DSE_ROOT}/analytic_calibrated" \
  --nc-min 1 --nc-max 64 --nv-min 1 --nv-max 128 \
  --wv-min 1024 --wv-max 8192 --wv-step 1024 \
  --cube-compression 1 --top-k 100 --min-area-utilization 0.95 \
  --ignore-power --models small middle large \
  --seq-lens 4096 6144 8192 --batch-sizes 1 2 4 \
  --users hot cold --min-predicted-speedup 0.95

python3 scripts/prepare_npu_w_both_candidates.py \
  --search-root "${DSE_ROOT}/analytic_calibrated" \
  --output-root "${DSE_ROOT}/finalists" \
  --per-chip 6 --top-overall 3 --top-vector-multiple 2 \
  --balance-bins 1 --cube-compression 1

retry_three final_matrix run_final_matrix

python3 scripts/summarize_npu_guard.py \
  --baseline-root "${WBOTH_ROOT}" \
  --guard-root "${DSE_ROOT}/full_results" \
  --config-root "${DSE_ROOT}/finalists/configs" \
  --output "${DSE_ROOT}/final_candidate_summary.csv" \
  --expected-cases 54 --min-speedup 0.95 \
  --selected-config-root "${DSE_ROOT}/selected_configs"

printf '{"status":"complete","ar_reduce_attention_compute":true}\n' \
  > "${ROOT}/pipeline_complete.json"
