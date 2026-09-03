#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

SIMULATOR_BIN="${SIMULATOR_BIN:-build/bin/Simulator}"
BASE_CONFIG="${BASE_CONFIG:-configs/910C.json}"
CALIBRATION="${CALIBRATION:-}"
KV_REUSE_RATIO="${KV_REUSE_RATIO:-0.4802}"
RESULT_ROOT="${RESULT_ROOT:-results/OoO_pipeline_ablation}"
METHOD_FILTER="${METHOD_FILTER:-without_OoO with_OoO}"

[[ -f "${BASE_CONFIG}" ]] || { echo "Missing BASE_CONFIG: ${BASE_CONFIG}" >&2; exit 2; }
if [[ -z "${CALIBRATION}" || ! -f "${CALIBRATION}" ]]; then
  echo "Set CALIBRATION to a current paper_cost_model schema-v2 calibration JSON." >&2
  exit 2
fi

contains_word() {
  local needle="$1"
  local words="$2"
  for word in ${words}; do
    [[ "${word}" == "${needle}" ]] && return 0
  done
  return 1
}

source_for_user() {
  if [[ "$1" == "hot" ]]; then
    echo ddr
  else
    echo ssd
  fi
}

optimal_recompute_len() {
  local user="$1" batch_size="$2" kv_len="$3" no_ar_reduce="$4"
  local args=(
    --config "${BASE_CONFIG}"
    --calibration "${CALIBRATION}"
    --user "${user}"
    --layers 4
    --hidden 256
    --kv-len "${kv_len}"
    --batch-size "${batch_size}"
    --enable-kv-reuse
    --kv-reuse-ratio "${KV_REUSE_RATIO}"
    --kv-reuse-reduce-npu
    --field len
  )
  if [[ "${no_ar_reduce}" == "1" ]]; then
    args+=(--without-ooo-pipeline)
  fi
  python3 scripts/recompute_ratio_cost_model_new.py "${args[@]}"
}

run_without_ooo() {
  local user="$1" batch_size="$2" kv_len="$3" src="$4"
  local recompute_len
  recompute_len="$(optimal_recompute_len "${user}" "${batch_size}" "${kv_len}" 1)"
  local result_dir="${RESULT_ROOT}/${user}/without_OoO/bs${batch_size}_kv${kv_len}"
  echo "method1 baseline+AR+IR,without out-of-order pipeline user=${user} bs=${batch_size} kv_len=${kv_len} history_recompute_len=${recompute_len}"
  SIMULATOR_BIN="${SIMULATOR_BIN}" bash scripts/run_hstu.sh \
    --source-medium "${src}" \
    --embedding-source-medium ssd \
    --history-recompute-source-medium "${src}" \
    --base-config "${BASE_CONFIG}" \
    --result-dir "${result_dir}" \
    --layers 4 \
    --hidden 256 \
    --kv-len "${kv_len}" \
    --history-recompute-len "${recompute_len}" \
    --history-recompute-index-mode continuous \
    --num-users "${batch_size}" \
    --users-per-batch "${batch_size}" \
    --candidates-per-user 128 \
    --macro-batch-size 128 \
    --vocab 262144 \
    --attention-modeling fused \
    --without-ooo-pipeline \
    --enable-kv-reuse \
    --enable-ar-reduce-attention-compute \
    --kv-reuse-variant window_topk \
    --kv-reuse-window-size 1024 \
    --kv-reuse-topk 4 \
    --kv-reuse-ratio "${KV_REUSE_RATIO}" \
    --log-level info \
    --npu-count 1
}

run_with_ooo() {
  local user="$1" batch_size="$2" kv_len="$3" src="$4"
  local recompute_len
  recompute_len="$(optimal_recompute_len "${user}" "${batch_size}" "${kv_len}" 0)"
  local result_dir="${RESULT_ROOT}/${user}/with_OoO/bs${batch_size}_kv${kv_len}"
  echo "method2 baseline+AR+IR,with out-of-order pipeline user=${user} bs=${batch_size} kv_len=${kv_len} history_recompute_len=${recompute_len}"
  SIMULATOR_BIN="${SIMULATOR_BIN}" bash scripts/run_hstu.sh \
    --source-medium "${src}" \
    --embedding-source-medium ssd \
    --history-recompute-source-medium "${src}" \
    --base-config "${BASE_CONFIG}" \
    --result-dir "${result_dir}" \
    --layers 4 \
    --hidden 256 \
    --kv-len "${kv_len}" \
    --history-recompute-len "${recompute_len}" \
    --history-recompute-index-mode continuous \
    --num-users "${batch_size}" \
    --users-per-batch "${batch_size}" \
    --candidates-per-user 128 \
    --macro-batch-size 128 \
    --vocab 262144 \
    --attention-modeling fused \
    --enable-kv-reuse \
    --enable-ar-reduce-attention-compute \
    --kv-reuse-variant window_topk \
    --kv-reuse-window-size 1024 \
    --kv-reuse-topk 4 \
    --kv-reuse-ratio "${KV_REUSE_RATIO}" \
    --log-level info \
    --npu-count 1
}

for user in cold hot; do
  src="$(source_for_user "${user}")"
  for batch_size in 1 4 8; do
    for kv_len in 4096 8192 16384; do
      if contains_word without_OoO "${METHOD_FILTER}"; then
        run_without_ooo "${user}" "${batch_size}" "${kv_len}" "${src}"
      fi
      if contains_word with_OoO "${METHOD_FILTER}"; then
        run_with_ooo "${user}" "${batch_size}" "${kv_len}" "${src}"
      fi
    done
  done
done
