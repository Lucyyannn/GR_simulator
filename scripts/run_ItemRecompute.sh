#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

SIMULATOR_BIN="${SIMULATOR_BIN:-build/bin/Simulator}"
CALIBRATION="${CALIBRATION:-scripts/recompute_ratio_calibration.json}"
RESULT_ROOT="${RESULT_ROOT:-results/ItemRecompute}"
KV_LEN=4096
ITEM_COUNT=$(((KV_LEN + 1) / 2))

ratio_to_len() {
  python3 - "$1" "${ITEM_COUNT}" <<'PY'
import sys
ratio = float(sys.argv[1])
item_count = int(sys.argv[2])
print(int(round(item_count * ratio / 100.0)))
PY
}

optimal_len() {
  python3 scripts/recompute_ratio_cost_model_new.py \
    --config configs/910C.json \
    --calibration "${CALIBRATION}" \
    --user cold \
    --layers 4 \
    --hidden 256 \
    --kv-len "${KV_LEN}" \
    --batch-size 1 \
    --field len
}

run_case() {
  local index_mode="$1"
  local label="$2"
  local recompute_len="$3"
  local result_dir="${RESULT_ROOT}/${index_mode}_${label}"
  echo "[task2] ${index_mode}_${label}: history_recompute_len=${recompute_len}"
  SIMULATOR_BIN="${SIMULATOR_BIN}" bash scripts/run_hstu.sh \
    --source-medium ssd \
    --embedding-source-medium ssd \
    --history-recompute-source-medium ssd \
    --base-config configs/910C.json \
    --result-dir "${result_dir}" \
    --layers 4 \
    --hidden 256 \
    --kv-len "${KV_LEN}" \
    --history-recompute-len "${recompute_len}" \
    --history-recompute-index-mode "${index_mode}" \
    --num-users 1 \
    --users-per-batch 1 \
    --candidates-per-user 128 \
    --macro-batch-size 128 \
    --vocab 262144 \
    --attention-modeling fused \
    --log-level info
}

run_case continuous 0 0
opt_len="$(optimal_len)"

for index_mode in continuous random; do
  for ratio in 20 40 60 80 100; do
    run_case "${index_mode}" "${ratio}" "$(ratio_to_len "${ratio}")"
  done
  run_case "${index_mode}" optimal "${opt_len}"
done
