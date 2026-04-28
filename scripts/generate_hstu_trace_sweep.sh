#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

GENERATOR="${REPO_ROOT}/scripts/generate_hstu_baseline_trace.py"
OUTPUT_DIR="${REPO_ROOT}/example/trace_tests/baseline_intro_sweep"
MODELS_LIST_DIR="${REPO_ROOT}/example/trace_tests/baseline_intro_sweep_model_lists"

LAYERS_LIST=(4 8 12)
HIDDEN_LIST=(256 512 1024)
HISTORY_LEN_LIST=(1024 2048 4096)
MACRO_BATCH_SIZE_LIST=(256 512 1024)
USERS_PER_BATCH_LIST=(1 2 4)

VOCAB=65536
CANDIDATES_PER_USER=10240
OP_MODELING="split=skip,view=skip,concat=skip"

rm -rf "${OUTPUT_DIR}" "${MODELS_LIST_DIR}"
mkdir -p "${OUTPUT_DIR}" "${MODELS_LIST_DIR}"

for layers in "${LAYERS_LIST[@]}"; do
  for hidden in "${HIDDEN_LIST[@]}"; do
    for history_len in "${HISTORY_LEN_LIST[@]}"; do
      for macro_batch_size in "${MACRO_BATCH_SIZE_LIST[@]}"; do
        for users_per_batch in "${USERS_PER_BATCH_LIST[@]}"; do
          num_users="${users_per_batch}"
          name="hstu_l${layers}_h${hidden}_hist${history_len}_cand${CANDIDATES_PER_USER}_mb${macro_batch_size}_u${num_users}_upb${users_per_batch}"
          output_path="${OUTPUT_DIR}/${name}"
          models_list_path="${MODELS_LIST_DIR}/${name}_models_list.json"

          echo "Generating ${name}"
          python3 "${GENERATOR}" \
            --pipeline \
            --shared-trace \
            --compact-json \
            --layers "${layers}" \
            --hidden "${hidden}" \
            --history-len "${history_len}" \
            --vocab "${VOCAB}" \
            --num-users "${num_users}" \
            --users-per-batch "${users_per_batch}" \
            --candidates-per-user "${CANDIDATES_PER_USER}" \
            --macro-batch-size "${macro_batch_size}" \
            --op-modeling "${OP_MODELING}" \
            --output "${output_path}" \
            --models-list "${models_list_path}"
        done
      done
    done
  done
done

echo "DONE: generated HSTU trace sweep under ${OUTPUT_DIR}"
echo "DONE: generated model lists under ${MODELS_LIST_DIR}"
