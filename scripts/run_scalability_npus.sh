#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

usage() {
  cat <<'EOF'
Usage: bash scripts/run_scalability_npus.sh [options]

Batch-run multi-NPU HSTU-middle scalability experiments. Each case keeps the
logical batch size at 4 and asks the simulator to shard it across 2 or 4
same-config NPUs. All DRAM/SSD, chip, scheme, and NPU-count cases are launched
in parallel as detached jobs.

Options:
  --result-root PATH       Output root. Default: results/hstu_scalability_npus
  --docker-container NAME  Launch cases with docker exec -d in this container
  --container-workdir PATH Container repo path. Default: /workspace/GR_simulator
  --local                  Force local detached launch instead of docker exec -d
  --dry-run                Print run_hstu.sh commands and backup action without running
  -h, --help               Show this message
EOF
}

RESULT_ROOT="${RESULT_ROOT:-results/hstu_scalability_npus}"
DOCKER_CONTAINER="${DOCKER_CONTAINER:-}"
CONTAINER_WORKDIR="${CONTAINER_WORKDIR:-/workspace/GR_simulator}"
LOCAL_LAUNCH=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --result-root)
      RESULT_ROOT="$2"
      shift 2
      ;;
    --docker-container)
      DOCKER_CONTAINER="$2"
      shift 2
      ;;
    --container-workdir)
      CONTAINER_WORKDIR="$2"
      shift 2
      ;;
    --local)
      LOCAL_LAUNCH=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

RESULT_ROOT="${RESULT_ROOT%/}"
if [[ -z "${RESULT_ROOT}" ]]; then
  echo "--result-root must not be empty or /" >&2
  exit 1
fi
CASES_ROOT="${RESULT_ROOT}/cases"
LOG_DIR="${RESULT_ROOT}/logs"

RUN_IN_DOCKER=0
if [[ "${LOCAL_LAUNCH}" != "1" ]]; then
  if [[ -z "${DOCKER_CONTAINER}" && ! -f /.dockerenv ]] && command -v docker >/dev/null 2>&1; then
    if docker inspect gr-simulator-mini >/dev/null 2>&1; then
      DOCKER_CONTAINER="gr-simulator-mini"
    fi
  fi
  if [[ -n "${DOCKER_CONTAINER}" ]]; then
    RUN_IN_DOCKER=1
  fi
fi

LAYERS="${LAYERS:-8}"
HIDDEN="${HIDDEN:-512}"
KV_LEN="${KV_LEN:-4096}"
NUM_USERS="${NUM_USERS:-4}"
USERS_PER_BATCH="${USERS_PER_BATCH:-4}"
CANDIDATES_PER_USER="${CANDIDATES_PER_USER:-128}"
MACRO_BATCH_SIZE="${MACRO_BATCH_SIZE:-128}"
KV_REUSE_RATIO="${KV_REUSE_RATIO:-0.4791}"
FULL_RECOMPUTE_LEN="${FULL_RECOMPUTE_LEN:-${KV_LEN}}"
RECOMPUTE_MODEL="${RECOMPUTE_MODEL:-scripts/recompute_ratio_cost_model_new.py}"
RECOMPUTE_CALIBRATION="${RECOMPUTE_CALIBRATION:-scripts/recompute_ratio_calibration.json}"

NPU_COUNTS=(1 2 4)
CHIPS=(910A 910B 910C)
SOURCE_LABELS=(DRAM SSD)
SOURCE_MEDIA=(ddr ssd)
SCHEMES=(Full_Cache Full_Recompute w_AR w_IR w_both)

MAX_STATUS_WAIT_SECONDS="${MAX_STATUS_WAIT_SECONDS:-0}"

RUN_ARGS=()

config_for() {
  local chip="$1"
  printf 'configs/%s.json\n' "${chip}"
}

user_mode_for_source() {
  local source_medium="$1"

  case "${source_medium}" in
    ddr)
      printf 'hot\n'
      ;;
    ssd)
      printf 'cold\n'
      ;;
    *)
      echo "Unknown source medium: ${source_medium}" >&2
      exit 1
      ;;
  esac
}

compute_recompute_len() {
  local npu_count="$1"
  local chip="$2"
  local source_medium="$3"
  local enable_kv_reuse="$4"
  local cfg
  local user_mode
  local shard_batch_size
  local -a model_args

  cfg=$(config_for "${chip}")
  user_mode=$(user_mode_for_source "${source_medium}")
  shard_batch_size=$((USERS_PER_BATCH / npu_count))
  model_args=(
    --config "${cfg}"
    --calibration "${RECOMPUTE_CALIBRATION}"
    --user "${user_mode}"
    --layers "${LAYERS}"
    --hidden "${HIDDEN}"
    --kv-len "${KV_LEN}"
    --batch-size "${shard_batch_size}"
    --candidates "${CANDIDATES_PER_USER}"
    --embedding-source "${source_medium}"
    --field len
  )
  if [[ "${enable_kv_reuse}" == "1" ]]; then
    model_args+=(--enable-kv-reuse --kv-reuse-ratio "${KV_REUSE_RATIO}")
  fi

  python3 "${RECOMPUTE_MODEL}" "${model_args[@]}"
}

build_run_args() {
  local npu_count="$1"
  local source_label="$2"
  local source_medium="$3"
  local chip="$4"
  local scheme="$5"
  local cfg
  local case_dir
  local recompute_len

  cfg=$(config_for "${chip}")
  case_dir="${CASES_ROOT}/NPU${npu_count}/${source_label}/${chip}/${scheme}"
  RUN_ARGS=(
    --base-config "${cfg}"
    --result-dir "${case_dir}"
    --source-medium "${source_medium}"
    --embedding-source-medium "${source_medium}"
    --layers "${LAYERS}"
    --hidden "${HIDDEN}"
    --kv-len "${KV_LEN}"
    --num-users "${NUM_USERS}"
    --users-per-batch "${USERS_PER_BATCH}"
    --candidates-per-user "${CANDIDATES_PER_USER}"
    --macro-batch-size "${MACRO_BATCH_SIZE}"
    --npu-count "${npu_count}"
  )

  case "${scheme}" in
    Full_Cache)
      RUN_ARGS+=(--history-recompute-len 0)
      ;;
    Full_Recompute)
      RUN_ARGS+=(--history-recompute-len "${FULL_RECOMPUTE_LEN}")
      ;;
    w_AR)
      RUN_ARGS+=(--history-recompute-len 0 --enable-kv-reuse --kv-reuse-ratio "${KV_REUSE_RATIO}")
      ;;
    w_IR)
      recompute_len=$(compute_recompute_len "${npu_count}" "${chip}" "${source_medium}" 0)
      RUN_ARGS+=(--history-recompute-len "${recompute_len}")
      ;;
    w_both)
      recompute_len=$(compute_recompute_len "${npu_count}" "${chip}" "${source_medium}" 1)
      RUN_ARGS+=(
        --history-recompute-len "${recompute_len}"
        --enable-kv-reuse
        --kv-reuse-ratio "${KV_REUSE_RATIO}"
      )
      ;;
    *)
      echo "Unknown scheme: ${scheme}" >&2
      exit 1
      ;;
  esac
}

print_command() {
  local -a args=("$@")
  local cmd
  local arg

  printf -v cmd 'MPLBACKEND=%q bash scripts/run_hstu.sh' "${MPLBACKEND:-Agg}"
  for arg in "${args[@]}"; do
    printf -v cmd '%s %q' "${cmd}" "${arg}"
  done

  if [[ "${RUN_IN_DOCKER}" == "1" ]]; then
    printf 'docker exec -d -w %q %q bash -lc %q\n' \
      "${CONTAINER_WORKDIR}" "${DOCKER_CONTAINER}" "${cmd}"
    return
  fi

  printf '%s\n' "${cmd}"
}

validate_inputs() {
  local chip
  local cfg

  if [[ ! -f scripts/run_hstu.sh ]]; then
    echo "Runner not found: scripts/run_hstu.sh" >&2
    exit 1
  fi
  if [[ ! -f "${RECOMPUTE_MODEL}" ]]; then
    echo "Recompute model not found: ${RECOMPUTE_MODEL}" >&2
    exit 1
  fi
  if [[ ! -f "${RECOMPUTE_CALIBRATION}" ]]; then
    echo "Recompute calibration not found: ${RECOMPUTE_CALIBRATION}" >&2
    exit 1
  fi
  for chip in "${CHIPS[@]}"; do
    cfg=$(config_for "${chip}")
    if [[ ! -f "${cfg}" ]]; then
      echo "Config not found: ${cfg}" >&2
      exit 1
    fi
  done
  for npu_count in "${NPU_COUNTS[@]}"; do
    if (( USERS_PER_BATCH % npu_count != 0 )); then
      echo "USERS_PER_BATCH (${USERS_PER_BATCH}) must be divisible by NPU count ${npu_count}" >&2
      exit 1
    fi
  done
}

backup_existing_result_root() {
  if [[ "${RUN_IN_DOCKER}" == "1" ]]; then
    local backup_cmd
    printf -v backup_cmd 'set -euo pipefail; RESULT_ROOT=%q; if [[ ! -e "${RESULT_ROOT}" ]]; then exit 0; fi; if [[ ! -d "${RESULT_ROOT}" ]]; then echo "Result root exists but is not a directory: ${RESULT_ROOT}" >&2; exit 1; fi; backup_parent=results/backups; timestamp=$(date +"%%Y%%m%%d_%%H%%M%%S"); root_name=$(basename "${RESULT_ROOT}"); backup_dir="${backup_parent}/${root_name}_${timestamp}"; suffix=1; while [[ -e "${backup_dir}" ]]; do backup_dir="${backup_parent}/${root_name}_${timestamp}_${suffix}"; suffix=$((suffix + 1)); done; mkdir -p "${backup_parent}"; mv "${RESULT_ROOT}" "${backup_dir}"; echo "Backed up existing result root: ${RESULT_ROOT} -> ${backup_dir}"' \
      "${RESULT_ROOT}"
    docker exec -w "${CONTAINER_WORKDIR}" "${DOCKER_CONTAINER}" bash -lc "${backup_cmd}"
    return
  fi

  if [[ ! -e "${RESULT_ROOT}" ]]; then
    return
  fi
  if [[ ! -d "${RESULT_ROOT}" ]]; then
    echo "Result root exists but is not a directory: ${RESULT_ROOT}" >&2
    exit 1
  fi

  local backup_parent="results/backups"
  local timestamp
  local root_name
  local backup_dir
  local suffix

  timestamp=$(date +"%Y%m%d_%H%M%S")
  root_name=$(basename "${RESULT_ROOT}")
  backup_dir="${backup_parent}/${root_name}_${timestamp}"
  suffix=1
  while [[ -e "${backup_dir}" ]]; do
    backup_dir="${backup_parent}/${root_name}_${timestamp}_${suffix}"
    suffix=$((suffix + 1))
  done

  mkdir -p "${backup_parent}"
  mv "${RESULT_ROOT}" "${backup_dir}"
  echo "Backed up existing result root: ${RESULT_ROOT} -> ${backup_dir}"
}

launch_case() {
  local label="$1"
  local status_path="${LOG_DIR}/${label}.status"
  local state_path="${LOG_DIR}/${label}.state"
  local pid_path="${LOG_DIR}/${label}.pid"
  local stdout_path="${LOG_DIR}/${label}.stdout.log"
  local stderr_path="${LOG_DIR}/${label}.stderr.log"
  local runner_path="${LOG_DIR}/${label}.run.sh"
  local launcher_log="${LOG_DIR}/${label}.launcher.log"
  local launcher_pid

  if [[ "${RUN_IN_DOCKER}" == "1" ]]; then
    local docker_cmd
    local arg
    printf -v docker_cmd 'set +e; cd %q; export ONNXIM_HOME=%q; export MPLBACKEND=%q; echo "running $(date -Is)" > %q; bash scripts/run_hstu.sh' \
      "${CONTAINER_WORKDIR}" "${CONTAINER_WORKDIR}" "${MPLBACKEND:-Agg}" "${state_path}"
    for arg in "${RUN_ARGS[@]}"; do
      printf -v docker_cmd '%s %q' "${docker_cmd}" "${arg}"
    done
    printf -v docker_cmd '%s > %q 2> %q; rc=$?; echo "${rc}" > %q; if [[ "${rc}" == "0" ]]; then echo "done $(date -Is)" > %q; else echo "failed $(date -Is) rc=${rc}" > %q; fi; exit "${rc}"' \
      "${docker_cmd}" "${stdout_path}" "${stderr_path}" "${status_path}" "${state_path}" "${state_path}"

    docker exec -d -w "${CONTAINER_WORKDIR}" "${DOCKER_CONTAINER}" bash -lc "${docker_cmd}"
    echo "START ${label} docker_exec=detached"
    return
  fi

  {
    printf '#!/usr/bin/env bash\n'
    printf 'set +e\n'
    printf 'cd %q\n' "${REPO_ROOT}"
    printf 'export ONNXIM_HOME=%q\n' "${REPO_ROOT}"
    printf 'export MPLBACKEND=%q\n' "${MPLBACKEND:-Agg}"
    printf 'echo "running $(date -Is)" > %q\n' "${state_path}"
    printf 'bash scripts/run_hstu.sh'
    printf ' %q' "${RUN_ARGS[@]}"
    printf ' > %q 2> %q\n' "${stdout_path}" "${stderr_path}"
    printf 'rc=$?\n'
    printf 'echo "${rc}" > %q\n' "${status_path}"
    printf 'if [[ "${rc}" == "0" ]]; then\n'
    printf '  echo "done $(date -Is)" > %q\n' "${state_path}"
    printf 'else\n'
    printf '  echo "failed $(date -Is) rc=${rc}" > %q\n' "${state_path}"
    printf 'fi\n'
    printf 'exit "${rc}"\n'
  } > "${runner_path}"
  chmod +x "${runner_path}"

  : > "${stdout_path}"
  : > "${stderr_path}"
  : > "${launcher_log}"
  rm -f "${status_path}"
  echo "launched $(date -Is)" > "${state_path}"

  if command -v setsid >/dev/null 2>&1; then
    nohup setsid bash "${runner_path}" < /dev/null > "${launcher_log}" 2>&1 &
  else
    nohup bash "${runner_path}" < /dev/null > "${launcher_log}" 2>&1 &
  fi

  launcher_pid=$!
  echo "${launcher_pid}" > "${pid_path}"
  echo "START ${label} launcher_pid=${launcher_pid}"
}

wait_for_case_status() {
  local label="$1"
  local start_ts
  local now
  local elapsed=0

  if [[ "${MAX_STATUS_WAIT_SECONDS}" != "0" ]]; then
    start_ts=$(date +%s)
  fi

  while [[ ! -f "${LOG_DIR}/${label}.status" ]]; do
    sleep 2
    if [[ "${MAX_STATUS_WAIT_SECONDS}" != "0" ]]; then
      now=$(date +%s)
      elapsed=$((now - start_ts))
      if (( elapsed >= MAX_STATUS_WAIT_SECONDS )); then
        echo "Timeout while waiting for ${label} status file after ${elapsed}s" >&2
        return 1
      fi
    fi
  done
}

wait_batch_done() {
  local -n batch_labels_ref="$1"
  local label
  local start_ts
  local now
  local elapsed=0

  if [[ "${MAX_STATUS_WAIT_SECONDS}" != "0" ]]; then
    start_ts=$(date +%s)
  fi

  for label in "${batch_labels_ref[@]}"; do
    echo "Waiting for status file of ${label}..."
    if ! wait_for_case_status "${label}"; then
      return 1
    fi
  done

  if [[ "${MAX_STATUS_WAIT_SECONDS}" != "0" ]]; then
    local completed=0
    while :; do
      completed=0
      for label in "${batch_labels_ref[@]}"; do
        if [[ -f "${LOG_DIR}/${label}.status" ]]; then
          completed=$((completed + 1))
        fi
      done
      if (( completed == ${#batch_labels_ref[@]} )); then
        return 0
      fi
      sleep 2
      now=$(date +%s)
      elapsed=$((now - start_ts))
      if (( elapsed >= MAX_STATUS_WAIT_SECONDS )); then
        echo "Timeout while waiting for completion of batch: ${#batch_labels_ref[@]} cases after ${elapsed}s" >&2
        return 1
      fi
    done
  fi
}

trap 'echo "Launch driver interrupted; already launched HSTU multi-NPU cases remain detached" >&2; exit 130' INT TERM

validate_inputs

if [[ "${DRY_RUN}" == "1" ]]; then
  if [[ -d "${RESULT_ROOT}" ]]; then
    timestamp_preview=$(date +"%Y%m%d_%H%M%S")
    echo "Would back up existing result root: ${RESULT_ROOT} -> results/backups/$(basename "${RESULT_ROOT}")_${timestamp_preview}"
  elif [[ -e "${RESULT_ROOT}" ]]; then
    echo "Result root exists but is not a directory: ${RESULT_ROOT}" >&2
    exit 1
  fi

  total=0
  for npu_count in "${NPU_COUNTS[@]}"; do
    for source_idx in "${!SOURCE_LABELS[@]}"; do
      source_label="${SOURCE_LABELS[${source_idx}]}"
      source_medium="${SOURCE_MEDIA[${source_idx}]}"
      for chip in "${CHIPS[@]}"; do
        for scheme in "${SCHEMES[@]}"; do
          build_run_args "${npu_count}" "${source_label}" "${source_medium}" "${chip}" "${scheme}"
          print_command "${RUN_ARGS[@]}"
          total=$((total + 1))
        done
      done
    done
  done
  echo "Dry run: ${total} cases"
  exit 0
fi

backup_existing_result_root
if [[ "${RUN_IN_DOCKER}" == "1" ]]; then
  mkdir_cmd=$(printf 'mkdir -p %q %q' "${CASES_ROOT}" "${LOG_DIR}")
  docker exec -w "${CONTAINER_WORKDIR}" "${DOCKER_CONTAINER}" bash -lc "${mkdir_cmd}"
else
  mkdir -p "${CASES_ROOT}" "${LOG_DIR}"
fi

for npu_count in "${NPU_COUNTS[@]}"; do
  batch_labels=()
  for source_idx in "${!SOURCE_LABELS[@]}"; do
    source_label="${SOURCE_LABELS[${source_idx}]}"
    source_medium="${SOURCE_MEDIA[${source_idx}]}"
    for chip in "${CHIPS[@]}"; do
      for scheme in "${SCHEMES[@]}"; do
        label="NPU${npu_count}__${source_label}__${chip}__${scheme}"
        build_run_args "${npu_count}" "${source_label}" "${source_medium}" "${chip}" "${scheme}"
        launch_case "${label}"
        batch_labels+=("${label}")
      done
    done
  done
  if (( ${#batch_labels[@]} > 0 )); then
    echo "Launched ${#batch_labels[@]} cases for NPU${npu_count}; waiting for completion..."
    wait_batch_done batch_labels
    echo "Completed NPU${npu_count} batch."
  fi
done

echo "All HSTU multi-NPU scalability cases launched as detached jobs."
echo "Result root: ${RESULT_ROOT}"
echo "Logs: ${LOG_DIR}"
echo "Completion status files will appear as: ${LOG_DIR}/*.status"
