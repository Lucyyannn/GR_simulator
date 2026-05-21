#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

SIMULATOR_BIN="${SIMULATOR_BIN:-build/bin/Simulator}"
RESULT_ROOT="${RESULT_ROOT:-results/ActionReuse}"

reuse_ratio_for() {
  case "$1 $2" in
    "64 1") echo 0.1819 ;; "64 2") echo 0.2905 ;; "64 3") echo 0.3528 ;; "64 4") echo 0.3830 ;; "64 5") echo 0.3964 ;;
    "128 1") echo 0.1787 ;; "128 2") echo 0.2942 ;; "128 3") echo 0.3655 ;; "128 4") echo 0.4044 ;; "128 5") echo 0.4248 ;;
    "256 1") echo 0.1742 ;; "256 2") echo 0.2919 ;; "256 3") echo 0.3688 ;; "256 4") echo 0.4135 ;; "256 5") echo 0.4389 ;;
    "512 1") echo 0.1702 ;; "512 2") echo 0.2890 ;; "512 3") echo 0.3691 ;; "512 4") echo 0.4176 ;; "512 5") echo 0.4474 ;;
    *) echo "unknown task1 parameter group: window_size=$1 top_k=$2" >&2; return 2 ;;
  esac
}

for window_size in 64 128 256 512; do
  for top_k in 1 2 3 4 5; do
    result_dir="${RESULT_ROOT}/${window_size}_${top_k}"
    reuse_ratio="$(reuse_ratio_for "${window_size}" "${top_k}")"
    echo "[task1] ${window_size}_${top_k}: reuse_ratio=${reuse_ratio}"
    SIMULATOR_BIN="${SIMULATOR_BIN}" bash scripts/run_hstu.sh \
      --source-medium ssd \
      --embedding-source-medium ssd \
      --history-recompute-source-medium ssd \
      --base-config configs/910C.json \
      --result-dir "${result_dir}" \
      --layers 4 \
      --hidden 256 \
      --kv-len 4096 \
      --history-recompute-len 0 \
      --num-users 1 \
      --users-per-batch 1 \
      --candidates-per-user 128 \
      --macro-batch-size 128 \
      --vocab 262144 \
      --attention-modeling fused \
      --enable-kv-reuse \
      --kv-reuse-variant window_topk \
      --kv-reuse-window-size "${window_size}" \
      --kv-reuse-topk "${top_k}" \
      --kv-reuse-ratio "${reuse_ratio}" \
      --log-level info
  done
done
