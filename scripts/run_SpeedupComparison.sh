#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

SIMULATOR_BIN="${SIMULATOR_BIN:-build/bin/Simulator}"
BASE_CONFIG="${BASE_CONFIG:-configs/910C.json}"
CALIBRATION="${CALIBRATION:-}"
KV_REUSE_RATIO="${KV_REUSE_RATIO:-0.4802}"
RESULT_ROOT="${RESULT_ROOT:-results/SpeedupComparison}"
MODEL_FILTER="${MODEL_FILTER:-small middle large}"
SEQ_FILTER="${SEQ_FILTER:-4096 8192 16384}"
BATCH_FILTER="${BATCH_FILTER:-1 4 8}"
USER_FILTER="${USER_FILTER:-hot cold}"
METHOD_FILTER="${METHOD_FILTER:-Recompute FullCache W_AR W_IR W_both}"

[[ -f "${BASE_CONFIG}" ]] || { echo "Missing BASE_CONFIG: ${BASE_CONFIG}" >&2; exit 2; }
if [[ " ${METHOD_FILTER} " == *" W_IR "* || " ${METHOD_FILTER} " == *" W_both "* ]]; then
  if [[ -z "${CALIBRATION}" || ! -f "${CALIBRATION}" ]]; then
    echo "W_IR/W_both require CALIBRATION pointing to a current paper_cost_model schema-v2 JSON." >&2
    exit 2
  fi
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

model_shape() {
  case "$1" in
    small) echo "4 256" ;;
    middle) echo "8 512" ;;
    large) echo "12 1024" ;;
    *) echo "unknown HSTU model: $1" >&2; return 2 ;;
  esac
}

optimal_recompute_len() {
  local user="$1" layers="$2" hidden="$3" kv_len="$4" batch_size="$5" enable_kv_reuse="$6"
  local args=(
    --config "${BASE_CONFIG}"
    --calibration "${CALIBRATION}"
    --user "${user}"
    --layers "${layers}"
    --hidden "${hidden}"
    --kv-len "${kv_len}"
    --batch-size "${batch_size}"
    --field len
  )
  if [[ "${enable_kv_reuse}" == "1" ]]; then
    args+=(--enable-kv-reuse --kv-reuse-ratio "${KV_REUSE_RATIO}" --kv-reuse-reduce-npu)
  fi
  python3 scripts/recompute_ratio_cost_model_new.py "${args[@]}"
}

run_method() {
  local method="$1" model="$2" layers="$3" hidden="$4" kv_len="$5" batch_size="$6" user="$7"
  local src result_dir recompute_len
  src="$(source_for_user "${user}")"
  result_dir="${RESULT_ROOT}/${method}/HSTU-${model}_seq${kv_len}_bs${batch_size}_${user}"
  echo "[task3] ${method} HSTU-${model} seq=${kv_len} bs=${batch_size} user=${user}"

  case "${method}" in
    Recompute)
      SIMULATOR_BIN="${SIMULATOR_BIN}" bash scripts/run_hstu.sh \
        --source-medium "${src}" \
        --embedding-source-medium ssd \
        --history-recompute-source-medium "${src}" \
        --base-config "${BASE_CONFIG}" \
        --result-dir "${result_dir}" \
        --layers "${layers}" \
        --hidden "${hidden}" \
        --kv-len "${kv_len}" \
        --history-recompute-len "${kv_len}" \
        --history-recompute-index-mode random \
        --num-users "${batch_size}" \
        --users-per-batch "${batch_size}" \
        --candidates-per-user 128 \
        --macro-batch-size 128 \
        --vocab 262144 \
        --attention-modeling fused \
        --log-level info
      ;;
    FullCache)
      SIMULATOR_BIN="${SIMULATOR_BIN}" bash scripts/run_hstu.sh \
        --source-medium "${src}" \
        --embedding-source-medium ssd \
        --history-recompute-source-medium "${src}" \
        --base-config "${BASE_CONFIG}" \
        --result-dir "${result_dir}" \
        --layers "${layers}" \
        --hidden "${hidden}" \
        --kv-len "${kv_len}" \
        --history-recompute-len 0 \
        --num-users "${batch_size}" \
        --users-per-batch "${batch_size}" \
        --candidates-per-user 128 \
        --macro-batch-size 128 \
        --vocab 262144 \
        --attention-modeling fused \
        --log-level info
      ;;
    W_AR)
      SIMULATOR_BIN="${SIMULATOR_BIN}" bash scripts/run_hstu.sh \
        --source-medium "${src}" \
        --embedding-source-medium ssd \
        --history-recompute-source-medium "${src}" \
        --base-config "${BASE_CONFIG}" \
        --result-dir "${result_dir}" \
        --layers "${layers}" \
        --hidden "${hidden}" \
        --kv-len "${kv_len}" \
        --history-recompute-len 0 \
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
        --log-level info
      ;;
    W_IR)
      recompute_len="$(optimal_recompute_len "${user}" "${layers}" "${hidden}" "${kv_len}" "${batch_size}" 0)"
      SIMULATOR_BIN="${SIMULATOR_BIN}" bash scripts/run_hstu.sh \
        --source-medium "${src}" \
        --embedding-source-medium ssd \
        --history-recompute-source-medium "${src}" \
        --base-config "${BASE_CONFIG}" \
        --result-dir "${result_dir}" \
        --layers "${layers}" \
        --hidden "${hidden}" \
        --kv-len "${kv_len}" \
        --history-recompute-len "${recompute_len}" \
        --history-recompute-index-mode continuous \
        --num-users "${batch_size}" \
        --users-per-batch "${batch_size}" \
        --candidates-per-user 128 \
        --macro-batch-size 128 \
        --vocab 262144 \
        --attention-modeling fused \
        --log-level info
      ;;
    W_both)
      recompute_len="$(optimal_recompute_len "${user}" "${layers}" "${hidden}" "${kv_len}" "${batch_size}" 1)"
      SIMULATOR_BIN="${SIMULATOR_BIN}" bash scripts/run_hstu.sh \
        --source-medium "${src}" \
        --embedding-source-medium ssd \
        --history-recompute-source-medium "${src}" \
        --base-config "${BASE_CONFIG}" \
        --result-dir "${result_dir}" \
        --layers "${layers}" \
        --hidden "${hidden}" \
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
        --log-level info
      ;;
    *)
      echo "unknown task3 method: ${method}" >&2
      return 2
      ;;
  esac
}

for model in small middle large; do
  contains_word "${model}" "${MODEL_FILTER}" || continue
  read -r layers hidden <<<"$(model_shape "${model}")"
  for kv_len in 4096 8192 16384; do
    contains_word "${kv_len}" "${SEQ_FILTER}" || continue
    for batch_size in 1 4 8; do
      contains_word "${batch_size}" "${BATCH_FILTER}" || continue
      for user in hot cold; do
        contains_word "${user}" "${USER_FILTER}" || continue
        for method in Recompute FullCache W_AR W_IR W_both; do
          contains_word "${method}" "${METHOD_FILTER}" || continue
          run_method "${method}" "${model}" "${layers}" "${hidden}" "${kv_len}" "${batch_size}" "${user}"
        done
      done
    done
  done
done
