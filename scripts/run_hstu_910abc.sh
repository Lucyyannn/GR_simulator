#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

usage() {
  cat <<'EOF'
Usage: bash scripts/run_hstu_910abc.sh [options]

Batch-run HSTU-middle on 910A/B/C configs.
The script launches all 15 cases in parallel and calls scripts/run_hstu.sh
for each individual case. When run on the host with Docker available, each
case is launched by an independent docker exec -d task and writes its own
status file.

Options:
  --result-root PATH   Output root. Default: results/hstu_910abc_sweep
  --docker-container NAME  Launch cases with docker exec -d in this container
  --container-workdir PATH Container repo path. Default: /workspace/GR_simulator
  --local             Force local detached launch instead of docker exec -d
  --dry-run            Print run_hstu.sh commands and backup action without running
  -h, --help           Show this message
EOF
}

RESULT_ROOT="${RESULT_ROOT:-results/hstu_910abc_sweep}"
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

LAYERS=8
HIDDEN=512
KV_LEN="${KV_LEN:-4096}"
USERS_PER_BATCH="${USERS_PER_BATCH:-4}"
KV_REUSE_RATIO="${KV_REUSE_RATIO:-0.4791}"
FULL_RECOMPUTE_LEN="${FULL_RECOMPUTE_LEN:-$(((KV_LEN + 1) / 2))}"
IR_RECOMPUTE_LEN=""
BOTH_RECOMPUTE_LEN=""

CHIPS=(910A 910B 910C)
SCHEMES=(Full_Cache Full_Recompute W_AR W_IR W_both)

RUN_ARGS=()

config_for() {
  local chip="$1"

  printf 'configs/%s.json\n' "${chip}"
}

build_run_args() {
  local chip="$1"
  local scheme="$2"
  local cfg
  local case_dir

  cfg=$(config_for "${chip}")
  case_dir="${CASES_ROOT}/${chip}/${scheme}"
  RUN_ARGS=(
    --base-config "${cfg}"
    --result-dir "${case_dir}"
    --layers "${LAYERS}"
    --hidden "${HIDDEN}"
    --kv-len "${KV_LEN}"
    --users-per-batch "${USERS_PER_BATCH}"
  )

  case "${scheme}" in
    Full_Cache)
      RUN_ARGS+=(--history-recompute-len 0)
      ;;
    Full_Recompute)
      RUN_ARGS+=(--history-recompute-len "${FULL_RECOMPUTE_LEN}")
      ;;
    W_AR)
      RUN_ARGS+=(--history-recompute-len 0 --enable-kv-reuse --kv-reuse-ratio "${KV_REUSE_RATIO}")
      ;;
    W_IR)
      RUN_ARGS+=(--history-recompute-len "${IR_RECOMPUTE_LEN}")
      ;;
    W_both)
      RUN_ARGS+=(
        --history-recompute-len "${BOTH_RECOMPUTE_LEN}"
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

  if [[ "${RUN_IN_DOCKER}" == "1" ]]; then
    printf 'docker exec -d -w %q %q bash -lc ' "${CONTAINER_WORKDIR}" "${DOCKER_CONTAINER}"
  fi
  printf 'MPLBACKEND=%q bash scripts/run_hstu.sh' "${MPLBACKEND:-Agg}"
  printf ' %q' "${args[@]}"
  printf '\n'
}

compute_auto_recompute_lens() {
  local user_mode="hot"
  if [[ "${SOURCE_MEDIUM:-ddr}" == "ssd" ]]; then
    user_mode="cold"
  fi

  IR_RECOMPUTE_LEN=$(python3 scripts/recompute_ratio_model.py \
    --user "${user_mode}" \
    --layers "${LAYERS}" \
    --hidden "${HIDDEN}" \
    --kv-len "${KV_LEN}" \
    --batch-size "${USERS_PER_BATCH}" \
    --field len)
  BOTH_RECOMPUTE_LEN=$(python3 scripts/recompute_ratio_model.py \
    --user "${user_mode}" \
    --layers "${LAYERS}" \
    --hidden "${HIDDEN}" \
    --kv-len "${KV_LEN}" \
    --batch-size "${USERS_PER_BATCH}" \
    --enable-kv-reuse \
    --kv-reuse-ratio "${KV_REUSE_RATIO}" \
    --field len)
  if (( BOTH_RECOMPUTE_LEN <= 0 && IR_RECOMPUTE_LEN > 0 )); then
    BOTH_RECOMPUTE_LEN="${IR_RECOMPUTE_LEN}"
  fi
}

validate_configs() {
  local chip
  local cfg

  for chip in "${CHIPS[@]}"; do
    cfg=$(config_for "${chip}")
    if [[ ! -f "${cfg}" ]]; then
      echo "Config not found: ${cfg}" >&2
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
    printf -v docker_cmd 'set +e; cd %q; export ONNXIM_HOME=%q; export MPLBACKEND=%q; echo "running $(date -Is)" > %q; bash scripts/run_hstu.sh' \
      "${CONTAINER_WORKDIR}" "${CONTAINER_WORKDIR}" "${MPLBACKEND:-Agg}" "${state_path}"
    printf -v docker_cmd '%s' "${docker_cmd}"
    printf -v docker_cmd '%s' "${docker_cmd}"
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

trap 'echo "Launch driver interrupted; already launched HSTU cases remain detached" >&2; exit 130' INT TERM

validate_configs
compute_auto_recompute_lens

if [[ "${DRY_RUN}" == "1" ]]; then
  if [[ -d "${RESULT_ROOT}" ]]; then
    timestamp_preview=$(date +"%Y%m%d_%H%M%S")
    echo "Would back up existing result root: ${RESULT_ROOT} -> results/backups/$(basename "${RESULT_ROOT}")_${timestamp_preview}"
  elif [[ -e "${RESULT_ROOT}" ]]; then
    echo "Result root exists but is not a directory: ${RESULT_ROOT}" >&2
    exit 1
  fi

  total=0
  for chip in "${CHIPS[@]}"; do
    for scheme in "${SCHEMES[@]}"; do
      build_run_args "${chip}" "${scheme}"
      print_command "${RUN_ARGS[@]}"
      total=$((total + 1))
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

for chip in "${CHIPS[@]}"; do
  for scheme in "${SCHEMES[@]}"; do
    label="${chip}__${scheme}"
    build_run_args "${chip}" "${scheme}"
    launch_case "${label}"
  done
done

echo "All HSTU cases launched as detached jobs."
echo "Result root: ${RESULT_ROOT}"
echo "Logs: ${LOG_DIR}"
echo "Completion status files will appear as: ${LOG_DIR}/*.status"
