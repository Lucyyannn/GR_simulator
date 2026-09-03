#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

usage() {
  cat <<'EOF'
Usage: bash scripts/run_scalability.sh [options]

Run single-NPU, single-user HSTU-small/middle/large scalability experiments.

Options:
  --result-root PATH             Output root. Default: results/hstu_modelsize_scalability_<timestamp>
  --calibration PATH             Required paper_cost_model schema-v2 calibration JSON
  --calibration-cache-root PATH  Persistent cache root. Default: MISC/hstu_modelsize_calibration_cache
  --skip-calibration             Use --calibration directly and skip memory calibration
  --force-calibration            Re-run calibration even when cache exists
  --max-concurrent N             Maximum active scalability cases. Default: 45
  --calibration-max-concurrent N Maximum active calibration cases. Default: 30
  --poll-interval SECONDS        Scheduler polling interval. Default: 10
  --docker-container NAME        Launch cases with docker exec in this container
  --container-workdir PATH       Container repo path. Default: /workspace/GR_simulator
  --local                        Force local launch instead of docker exec
  --summary-only                 Regenerate summary files for an existing result root
  --no-clean-results             Do not move old results into results/backups before running
  --keep-intermediates           Keep generated trace directories after successful cases
  --skip-validation              Skip the pre-run Full_Recompute chip-order validation
  --schemes LIST                 Comma-separated schemes to run. Default: all schemes
  --dry-run                      Print planned commands without running
  -h, --help                     Show this message
EOF
}

timestamp() {
  date +"%Y%m%d_%H%M%S"
}

join_csv() {
  local -n values="$1"
  local joined=""
  local value
  for value in "${values[@]}"; do
    if [[ -n "${joined}" ]]; then
      joined+=","
    fi
    joined+="${value}"
  done
  printf '%s' "${joined}"
}

quote_cmd() {
  printf '%q ' "$@"
}

parse_csv_into_array() {
  local raw="$1"
  local -n out_ref="$2"
  local value
  out_ref=()
  raw="${raw//,/ }"
  for value in ${raw}; do
    [[ -n "${value}" ]] || continue
    out_ref+=("${value}")
  done
}

RESULT_ROOT="${RESULT_ROOT:-results/hstu_modelsize_scalability_$(timestamp)}"
BASE_CALIBRATION="${RECOMPUTE_CALIBRATION:-}"
CALIBRATION_CACHE_ROOT="${CALIBRATION_CACHE_ROOT:-MISC/hstu_modelsize_calibration_cache}"
SCHEMES_OVERRIDE="${SCHEMES_OVERRIDE:-}"
DOCKER_CONTAINER="${DOCKER_CONTAINER:-}"
CONTAINER_WORKDIR="${CONTAINER_WORKDIR:-/workspace/GR_simulator}"
MAX_CONCURRENT_JOBS="${MAX_CONCURRENT_JOBS:-45}"
CALIBRATION_MAX_CONCURRENT_JOBS="${CALIBRATION_MAX_CONCURRENT_JOBS:-30}"
POLL_INTERVAL_SECONDS="${POLL_INTERVAL_SECONDS:-10}"
LOCAL_LAUNCH=0
DRY_RUN=0
SKIP_CALIBRATION=0
FORCE_CALIBRATION=0
SUMMARY_ONLY=0
CLEAN_RESULTS=1
KEEP_INTERMEDIATES=0
SKIP_VALIDATION=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --result-root)
      RESULT_ROOT="$2"
      shift 2
      ;;
    --calibration)
      BASE_CALIBRATION="$2"
      shift 2
      ;;
    --calibration-cache-root)
      CALIBRATION_CACHE_ROOT="$2"
      shift 2
      ;;
    --skip-calibration)
      SKIP_CALIBRATION=1
      shift
      ;;
    --force-calibration)
      FORCE_CALIBRATION=1
      shift
      ;;
    --max-concurrent)
      MAX_CONCURRENT_JOBS="$2"
      shift 2
      ;;
    --calibration-max-concurrent)
      CALIBRATION_MAX_CONCURRENT_JOBS="$2"
      shift 2
      ;;
    --poll-interval)
      POLL_INTERVAL_SECONDS="$2"
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
    --summary-only)
      SUMMARY_ONLY=1
      shift
      ;;
    --no-clean-results)
      CLEAN_RESULTS=0
      shift
      ;;
    --keep-intermediates)
      KEEP_INTERMEDIATES=1
      shift
      ;;
    --skip-validation)
      SKIP_VALIDATION=1
      shift
      ;;
    --schemes)
      SCHEMES_OVERRIDE="$2"
      shift 2
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
CALIBRATION_CACHE_ROOT="${CALIBRATION_CACHE_ROOT%/}"
CASES_ROOT="${RESULT_ROOT}/cases"
LOG_DIR="${RESULT_ROOT}/logs"
RUN_METADATA="${RESULT_ROOT}/run_metadata.json"

if [[ -z "${RESULT_ROOT}" || "${RESULT_ROOT}" == "/" ]]; then
  echo "--result-root must not be empty or /" >&2
  exit 1
fi
if (( ! SUMMARY_ONLY )) && { [[ -z "${BASE_CALIBRATION}" ]] || [[ ! -f "${BASE_CALIBRATION}" ]]; }; then
  echo "Provide --calibration with a current paper_cost_model schema-v2 JSON." >&2
  exit 1
fi
if ! [[ "${MAX_CONCURRENT_JOBS}" =~ ^[0-9]+$ ]] || (( MAX_CONCURRENT_JOBS < 1 )); then
  echo "--max-concurrent must be a positive integer: ${MAX_CONCURRENT_JOBS}" >&2
  exit 1
fi
if [[ -z "${CALIBRATION_MAX_CONCURRENT_JOBS}" ]]; then
  CALIBRATION_MAX_CONCURRENT_JOBS="${MAX_CONCURRENT_JOBS}"
fi
if ! [[ "${CALIBRATION_MAX_CONCURRENT_JOBS}" =~ ^[0-9]+$ ]] || (( CALIBRATION_MAX_CONCURRENT_JOBS < 1 )); then
  echo "--calibration-max-concurrent must be a positive integer: ${CALIBRATION_MAX_CONCURRENT_JOBS}" >&2
  exit 1
fi

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

MODEL_SIZES=(small middle large)
CHIPS=(910A 910B 910C)
SOURCE_LABELS=(DRAM SSD)
SOURCE_MEDIA=(ddr ssd)
SCHEMES=(Full_Cache Full_Recompute w_AR w_IR w_both)
if [[ -n "${SCHEMES_OVERRIDE}" ]]; then
  parse_csv_into_array "${SCHEMES_OVERRIDE}" SCHEMES
  if (( ${#SCHEMES[@]} == 0 )); then
    echo "--schemes must contain at least one scheme" >&2
    exit 1
  fi
fi

declare -A MODEL_LAYERS=(["small"]=4 ["middle"]=8 ["large"]=12)
declare -A MODEL_HIDDEN=(["small"]=256 ["middle"]=512 ["large"]=1024)
declare -A MODEL_CALIBRATION=()

if [[ -n "${MODEL_LAYERS_OVERRIDE:-}" ]]; then
  if ! [[ "${MODEL_LAYERS_OVERRIDE}" =~ ^[0-9]+$ ]] || (( MODEL_LAYERS_OVERRIDE < 1 )); then
    echo "MODEL_LAYERS_OVERRIDE must be a positive integer: ${MODEL_LAYERS_OVERRIDE}" >&2
    exit 1
  fi
  for size in "${MODEL_SIZES[@]}"; do
    MODEL_LAYERS["${size}"]="${MODEL_LAYERS_OVERRIDE}"
  done
fi

KV_LEN="${KV_LEN:-4096}"
NUM_USERS="${NUM_USERS:-1}"
USERS_PER_BATCH="${USERS_PER_BATCH:-1}"
CANDIDATES_PER_USER="${CANDIDATES_PER_USER:-128}"
MACRO_BATCH_SIZE="${MACRO_BATCH_SIZE:-128}"
KV_REUSE_RATIO="${KV_REUSE_RATIO:-0.4802}"
FULL_RECOMPUTE_LEN="${FULL_RECOMPUTE_LEN:-${KV_LEN}}"
RECOMPUTE_MODEL="${RECOMPUTE_MODEL:-scripts/recompute_ratio_cost_model_new.py}"
IR_OBJECTIVE="${IR_OBJECTIVE:-balance}"
LOG_LEVEL="${LOG_LEVEL:-warn}"
BANDWIDTH_PATTERNS="${BANDWIDTH_PATTERNS:-contiguous,random_512b_index}"
BANDWIDTH_ACCESS_TYPES="${BANDWIDTH_ACCESS_TYPES:-read}"
BANDWIDTH_SIZES_BYTES="${BANDWIDTH_SIZES_BYTES:-512,1024,2048}"
BANDWIDTH_BURST_COUNTS="${BANDWIDTH_BURST_COUNTS:-1,2,4,8}"

CHIPS_JOINED=$(join_csv CHIPS)
SOURCES_JOINED=$(join_csv SOURCE_MEDIA)
MODEL_SIZES_JOINED=$(join_csv MODEL_SIZES)
SCHEMES_JOINED=$(join_csv SCHEMES)

CALIBRATION_ROOT=""
MEMORY_CALIBRATION=""
RUN_ARGS=()

config_for() {
  printf 'configs/%s.json\n' "$1"
}

user_mode_for_source() {
  case "$1" in
    ddr) printf 'hot\n' ;;
    ssd) printf 'cold\n' ;;
    *)
      echo "Unknown source medium: $1" >&2
      exit 1
      ;;
  esac
}

source_label_for_medium() {
  case "$1" in
    ddr) printf 'DRAM\n' ;;
    ssd) printf 'SSD\n' ;;
    *)
      echo "Unknown source medium: $1" >&2
      exit 1
      ;;
  esac
}

validate_inputs() {
  local chip
  if [[ ! -f scripts/run_hstu.sh || ! -f "${RECOMPUTE_MODEL}" ]]; then
    echo "Required runner/model script is missing" >&2
    exit 1
  fi
  if [[ ! -f scripts/calibrate_memory_bandwidth.py ]]; then
    echo "Memory calibration helper is missing" >&2
    exit 1
  fi
  if [[ ! -f scripts/summarize_scalability_results.py ]]; then
    echo "Summary helper is missing" >&2
    exit 1
  fi
  for chip in "${CHIPS[@]}"; do
    if [[ ! -f "$(config_for "${chip}")" ]]; then
      echo "Config not found: $(config_for "${chip}")" >&2
      exit 1
    fi
  done
}

backup_existing_result_root() {
  if [[ ! -e "${RESULT_ROOT}" ]]; then
    return
  fi
  local backup_parent="results/backups"
  local root_name
  local backup_dir
  root_name=$(basename "${RESULT_ROOT}")
  backup_dir="${backup_parent}/${root_name}_$(timestamp)"
  mkdir -p "${backup_parent}"
  mv "${RESULT_ROOT}" "${backup_dir}"
  echo "Backed up existing result root: ${RESULT_ROOT} -> ${backup_dir}"
}

clean_old_results() {
  if (( ! CLEAN_RESULTS || SUMMARY_ONLY || DRY_RUN )); then
    return
  fi
  if [[ ! -d results ]]; then
    return
  fi
  local backup_dir="results/backups/cleanup_$(timestamp)"
  local moved=0
  local path
  shopt -s nullglob
  for path in results/*; do
    if [[ "$(basename "${path}")" == "backups" ]]; then
      continue
    fi
    mkdir -p "${backup_dir}"
    mv "${path}" "${backup_dir}/"
    moved=1
  done
  shopt -u nullglob
  if (( moved )); then
    echo "Moved old result entries to ${backup_dir}"
  fi
}

calibration_cache_key() {
  {
    printf 'schema=hstu_modelsize_scalability_memory_calibration_v2\n'
    printf 'base_calibration '; sha256sum "${BASE_CALIBRATION}"
    printf 'memory_runner '; sha256sum scripts/calibrate_memory_bandwidth.py
    printf 'sources=%s\n' "${SOURCES_JOINED}"
    printf 'chips=%s\n' "${CHIPS_JOINED}"
    printf 'bandwidth=%s|%s|%s|%s\n' "${BANDWIDTH_PATTERNS}" "${BANDWIDTH_ACCESS_TYPES}" "${BANDWIDTH_SIZES_BYTES}" "${BANDWIDTH_BURST_COUNTS}"
    local chip
    for chip in "${CHIPS[@]}"; do
      printf 'config_%s ' "${chip}"
      sha256sum "$(config_for "${chip}")"
    done
  } | sha256sum | cut -c1-16
}

resolve_calibration_paths() {
  local key
  key=$(calibration_cache_key)
  CALIBRATION_ROOT="${CALIBRATION_CACHE_ROOT}/single_npu_single_user_hstu_sizes_${key}"
  MEMORY_CALIBRATION="${CALIBRATION_ROOT}/memory/recompute_ratio_calibration_memory_merged.json"
}

run_memory_calibration() {
  local -a cal_args=(
    python3 scripts/calibrate_memory_bandwidth.py
    --result-root "${CALIBRATION_ROOT}/memory"
    --calibration "${BASE_CALIBRATION}"
    --merged-calibration-output "${MEMORY_CALIBRATION}"
    --chips "${CHIPS_JOINED}"
    --patterns "${BANDWIDTH_PATTERNS}"
    --access-types "${BANDWIDTH_ACCESS_TYPES}"
    --sizes-bytes "${BANDWIDTH_SIZES_BYTES}"
    --burst-counts "${BANDWIDTH_BURST_COUNTS}"
    --max-concurrent "${CALIBRATION_MAX_CONCURRENT_JOBS}"
    --poll-interval "${POLL_INTERVAL_SECONDS}"
    --log-level "${LOG_LEVEL}"
  )
  if (( DRY_RUN )); then
    cal_args+=(--dry-run)
  fi
  printf '[memory-cal] %s\n' "$(quote_cmd "${cal_args[@]}")"
  if (( RUN_IN_DOCKER )); then
    local cal_cmd
    cal_cmd=$(quote_cmd "${cal_args[@]}")
    docker exec -w "${CONTAINER_WORKDIR}" "${DOCKER_CONTAINER}" bash -lc "cd ${CONTAINER_WORKDIR@Q}; ${cal_cmd}"
  else
    "${cal_args[@]}"
  fi
}

prepare_calibration() {
  resolve_calibration_paths
  if (( SKIP_CALIBRATION )); then
    local size
    for size in "${MODEL_SIZES[@]}"; do
      MODEL_CALIBRATION["${size}"]="${BASE_CALIBRATION}"
    done
    return
  fi

  if (( FORCE_CALIBRATION )) || [[ ! -f "${MEMORY_CALIBRATION}" ]]; then
    run_memory_calibration
  else
    echo "Reusing memory calibration: ${MEMORY_CALIBRATION}"
  fi

  local size
  for size in "${MODEL_SIZES[@]}"; do
    MODEL_CALIBRATION["${size}"]="${MEMORY_CALIBRATION}"
  done
}

compute_recompute_len() {
  local size="$1"
  local chip="$2"
  local source_medium="$3"
  local enable_kv_reuse="$4"
  local diag_path="${5:-}"
  local model_output
  local -a model_args=(
    --config "$(config_for "${chip}")"
    --calibration "${MODEL_CALIBRATION[${size}]}"
    --user "$(user_mode_for_source "${source_medium}")"
    --layers "${MODEL_LAYERS[${size}]}"
    --hidden "${MODEL_HIDDEN[${size}]}"
    --kv-len "${KV_LEN}"
    --batch-size "${USERS_PER_BATCH}"
    --candidates "${CANDIDATES_PER_USER}"
    --embedding-source "${source_medium}"
    --objective "${IR_OBJECTIVE}"
    --field json
  )
  if [[ "${enable_kv_reuse}" == "1" ]]; then
    model_args+=(--enable-kv-reuse --kv-reuse-ratio "${KV_REUSE_RATIO}" --kv-reuse-reduce-npu)
  fi
  model_output=$(python3 "${RECOMPUTE_MODEL}" "${model_args[@]}")
  if [[ -n "${diag_path}" ]]; then
    mkdir -p "$(dirname "${diag_path}")"
    printf '%s\n' "${model_output}" > "${diag_path}"
  fi
  python3 -c 'import json,sys; print(json.load(sys.stdin)["history_recompute_len"])' <<<"${model_output}"
}

build_run_args() {
  local size="$1"
  local source_label="$2"
  local source_medium="$3"
  local chip="$4"
  local scheme="$5"
  local case_dir="${CASES_ROOT}/${size}/${source_label}/${chip}/${scheme}"
  local recompute_len

  RUN_ARGS=(
    --base-config "$(config_for "${chip}")"
    --result-dir "${case_dir}"
    --source-medium "${source_medium}"
    --embedding-source-medium "${source_medium}"
    --layers "${MODEL_LAYERS[${size}]}"
    --hidden "${MODEL_HIDDEN[${size}]}"
    --kv-len "${KV_LEN}"
    --num-users "${NUM_USERS}"
    --users-per-batch "${USERS_PER_BATCH}"
    --candidates-per-user "${CANDIDATES_PER_USER}"
    --macro-batch-size "${MACRO_BATCH_SIZE}"
    --npu-count "1"
    --log-level "${LOG_LEVEL}"
  )

  case "${scheme}" in
    Full_Cache)
      RUN_ARGS+=(--history-recompute-len 0)
      ;;
    Full_Recompute)
      RUN_ARGS+=(--history-recompute-len "${FULL_RECOMPUTE_LEN}" --history-recompute-index-mode random)
      ;;
    w_AR)
      RUN_ARGS+=(--history-recompute-len 0 --enable-kv-reuse --enable-ar-reduce-attention-compute --kv-reuse-ratio "${KV_REUSE_RATIO}")
      ;;
    w_IR)
      recompute_len=$(compute_recompute_len "${size}" "${chip}" "${source_medium}" 0 "${case_dir}/ir_selection.json")
      RUN_ARGS+=(--history-recompute-len "${recompute_len}")
      ;;
    w_both)
      recompute_len=$(compute_recompute_len "${size}" "${chip}" "${source_medium}" 1 "${case_dir}/ir_selection.json")
      RUN_ARGS+=(--history-recompute-len "${recompute_len}" --enable-kv-reuse --enable-ar-reduce-attention-compute --kv-reuse-ratio "${KV_REUSE_RATIO}")
      ;;
    *)
      echo "Unknown scheme: ${scheme}" >&2
      exit 1
      ;;
  esac
}

case_dir_for_label() {
  local label="$1"
  local model source chip scheme
  IFS='|' read -r model source chip scheme <<<"${label//__/|}"
  printf '%s/%s/%s/%s/%s\n' "${CASES_ROOT}" "${model}" "${source}" "${chip}" "${scheme}"
}

launch_case() {
  local label="$1"
  local status_path="${LOG_DIR}/${label}.status"
  local state_path="${LOG_DIR}/${label}.state"
  local stdout_path="${LOG_DIR}/${label}.stdout.log"
  local stderr_path="${LOG_DIR}/${label}.stderr.log"
  local runner_path="${LOG_DIR}/${label}.run.sh"
  local launcher_log="${LOG_DIR}/${label}.launcher.log"

  mkdir -p "${LOG_DIR}"
  rm -f "${status_path}" "${state_path}"

  if (( RUN_IN_DOCKER )); then
    local docker_cmd
    local arg
    printf -v docker_cmd 'set +e; cd %q; export ONNXIM_HOME=%q; export MPLBACKEND=%q; echo "running $(date -Is)" > %q; bash scripts/run_hstu.sh' \
      "${CONTAINER_WORKDIR}" "${CONTAINER_WORKDIR}" "${MPLBACKEND:-Agg}" "${state_path}"
    for arg in "${RUN_ARGS[@]}"; do
      printf -v docker_cmd '%s %q' "${docker_cmd}" "${arg}"
    done
    printf -v docker_cmd '%s > %q 2> %q; rc=$?; echo "${rc}" > %q; if [[ "${rc}" == "0" ]]; then echo "done $(date -Is)" > %q; else echo "failed $(date -Is) rc=${rc}" > %q; fi; exit "${rc}"' \
      "${docker_cmd}" "${stdout_path}" "${stderr_path}" "${status_path}" "${state_path}" "${state_path}"
    if (( DRY_RUN )); then
      printf 'docker exec -d -w %q %q bash -lc %q\n' "${CONTAINER_WORKDIR}" "${DOCKER_CONTAINER}" "${docker_cmd}"
    else
      docker exec -d -w "${CONTAINER_WORKDIR}" "${DOCKER_CONTAINER}" bash -lc "${docker_cmd}"
    fi
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
    printf 'if [[ "${rc}" == "0" ]]; then echo "done $(date -Is)" > %q; else echo "failed $(date -Is) rc=${rc}" > %q; fi\n' "${state_path}" "${state_path}"
    printf 'exit "${rc}"\n'
  } > "${runner_path}"
  chmod +x "${runner_path}"
  : > "${stdout_path}"
  : > "${stderr_path}"
  : > "${launcher_log}"
  if (( DRY_RUN )); then
    printf '%s\n' "${runner_path}"
  elif command -v setsid >/dev/null 2>&1; then
    nohup setsid bash "${runner_path}" < /dev/null > "${launcher_log}" 2>&1 &
  else
    nohup bash "${runner_path}" < /dev/null > "${launcher_log}" 2>&1 &
  fi
  echo "START ${label}"
}

count_completed_cases() {
  local -n labels_ref="$1"
  local label
  local completed=0
  for label in "${labels_ref[@]}"; do
    if [[ -f "${LOG_DIR}/${label}.status" ]]; then
      completed=$((completed + 1))
    fi
  done
  printf '%s\n' "${completed}"
}

count_active_cases() {
  local -n labels_ref="$1"
  local label
  local active=0
  for label in "${labels_ref[@]}"; do
    if [[ ! -f "${LOG_DIR}/${label}.status" ]]; then
      active=$((active + 1))
    fi
  done
  printf '%s\n' "${active}"
}

cleanup_successful_cases() {
  if (( KEEP_INTERMEDIATES || DRY_RUN )); then
    return
  fi
  local -n labels_ref="$1"
  local label status case_dir cleaned_marker
  for label in "${labels_ref[@]}"; do
    [[ -f "${LOG_DIR}/${label}.status" ]] || continue
    status=$(<"${LOG_DIR}/${label}.status")
    case_dir=$(case_dir_for_label "${label}")
    cleaned_marker="${case_dir}/.intermediates_cleaned"
    [[ "${status}" == "0" && ! -f "${cleaned_marker}" ]] || continue
    rm -rf "${case_dir}/traces"
    echo "cleaned $(date -Is)" > "${cleaned_marker}"
  done
}

run_validation() {
  if (( SKIP_VALIDATION || SUMMARY_ONLY || DRY_RUN )); then
    return
  fi
  local size="small"
  local source_medium source_label chip case_dir time_us
  local -A times=()
  local -a val_args=()
  echo "Running Full_Recompute chip-order validation..."
  for source_medium in "${SOURCE_MEDIA[@]}"; do
    source_label=$(source_label_for_medium "${source_medium}")
    for chip in "${CHIPS[@]}"; do
      case_dir="${RESULT_ROOT}/validation/${size}/${source_label}/${chip}/Full_Recompute"
      val_args=(
        bash scripts/run_hstu.sh
        --base-config "$(config_for "${chip}")" \
        --result-dir "${case_dir}" \
        --source-medium "${source_medium}" \
        --embedding-source-medium "${source_medium}" \
        --layers "${MODEL_LAYERS[${size}]}" \
        --hidden "${MODEL_HIDDEN[${size}]}" \
        --kv-len "${KV_LEN}" \
        --history-recompute-len "${FULL_RECOMPUTE_LEN}" \
        --history-recompute-index-mode random \
        --num-users "${NUM_USERS}" \
        --users-per-batch "${USERS_PER_BATCH}" \
        --candidates-per-user "${CANDIDATES_PER_USER}" \
        --macro-batch-size "${MACRO_BATCH_SIZE}" \
        --npu-count 1 \
        --log-level "${LOG_LEVEL}"
      )
      if (( RUN_IN_DOCKER )); then
        local val_cmd
        val_cmd="MPLBACKEND=${MPLBACKEND:-Agg} $(quote_cmd "${val_args[@]}") >/dev/null"
        docker exec -w "${CONTAINER_WORKDIR}" "${DOCKER_CONTAINER}" bash -lc "cd ${CONTAINER_WORKDIR@Q}; ${val_cmd}"
      else
        MPLBACKEND="${MPLBACKEND:-Agg}" "${val_args[@]}" >/dev/null
      fi
      time_us=$(python3 - "${case_dir}/hardware_summary.csv" <<'PY'
import csv
import sys
from pathlib import Path
path = Path(sys.argv[1])
with path.open(newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if row.get("component") == "NPU" and row.get("scope") == "overall":
            print(row.get("sim_time_us", ""))
            break
PY
)
      times["${source_label}_${chip}"]="${time_us}"
    done
    python3 - "${source_label}" "${times[${source_label}_910A]}" "${times[${source_label}_910B]}" "${times[${source_label}_910C]}" <<'PY'
import sys
source, a, b, c = sys.argv[1:]
a, b, c = float(a), float(b), float(c)
if not (a > b > c):
    raise SystemExit(f"{source} Full_Recompute validation failed: expected 910A slower than 910B slower than 910C, got 910A={a:.2f}us 910B={b:.2f}us 910C={c:.2f}us")
print(f"{source} validation: 910A={a:.2f}us 910B={b:.2f}us 910C={c:.2f}us")
PY
  done
}

write_metadata_file() {
  local expanded_command
  expanded_command=$(quote_cmd \
    bash scripts/run_scalability.sh \
    --result-root "${RESULT_ROOT}" \
    --calibration "${BASE_CALIBRATION}" \
    --calibration-cache-root "${CALIBRATION_CACHE_ROOT}" \
    --schemes "${SCHEMES_JOINED}" \
    --max-concurrent "${MAX_CONCURRENT_JOBS}" \
    --calibration-max-concurrent "${CALIBRATION_MAX_CONCURRENT_JOBS}" \
    --poll-interval "${POLL_INTERVAL_SECONDS}")
  if (( SKIP_CALIBRATION )); then expanded_command+="--skip-calibration "; fi
  if (( FORCE_CALIBRATION )); then expanded_command+="--force-calibration "; fi
  if (( RUN_IN_DOCKER )); then expanded_command+="$(quote_cmd --docker-container "${DOCKER_CONTAINER}" --container-workdir "${CONTAINER_WORKDIR}")"; fi
  if (( KEEP_INTERMEDIATES )); then expanded_command+="--keep-intermediates "; fi
  if (( SKIP_VALIDATION )); then expanded_command+="--skip-validation "; fi

  mkdir -p "$(dirname "${RUN_METADATA}")"
  python3 - \
    "${RUN_METADATA}" \
    "${RESULT_ROOT}" \
    "${BASE_CALIBRATION}" \
    "${CALIBRATION_CACHE_ROOT}" \
    "${CALIBRATION_ROOT}" \
    "${MEMORY_CALIBRATION}" \
    "${MODEL_CALIBRATION[small]}" \
    "${MODEL_CALIBRATION[middle]}" \
    "${MODEL_CALIBRATION[large]}" \
    "${MAX_CONCURRENT_JOBS}" \
    "${CALIBRATION_MAX_CONCURRENT_JOBS}" \
    "${POLL_INTERVAL_SECONDS}" \
    "${KV_LEN}" \
    "${NUM_USERS}" \
    "${USERS_PER_BATCH}" \
    "${CANDIDATES_PER_USER}" \
    "${MACRO_BATCH_SIZE}" \
    "${KV_REUSE_RATIO}" \
    "${IR_OBJECTIVE}" \
    "${MODEL_SIZES_JOINED}" \
    "${CHIPS_JOINED}" \
    "${SOURCES_JOINED}" \
    "${SCHEMES_JOINED}" \
    "${expanded_command}" <<'PY'
import json
import sys
from pathlib import Path

(
    path,
    result_root,
    base_calibration,
    calibration_cache_root,
    calibration_root,
    memory_calibration,
    cal_small,
    cal_middle,
    cal_large,
    max_concurrent,
    calibration_max_concurrent,
    poll_interval,
    kv_len,
    num_users,
    users_per_batch,
    candidates_per_user,
    macro_batch_size,
    kv_reuse_ratio,
    ir_objective,
    model_sizes,
    chips,
    sources,
    schemes,
    expanded_command,
) = sys.argv[1:]

data = {
    "kind": "hstu_modelsize_scalability",
    "result_root": result_root,
    "base_calibration": base_calibration,
    "calibration_cache_root": calibration_cache_root,
    "calibration_root": calibration_root,
    "memory_calibration": memory_calibration,
    "model_calibrations": {
        "small": cal_small,
        "middle": cal_middle,
        "large": cal_large,
    },
    "max_concurrent": int(max_concurrent),
    "calibration_max_concurrent": int(calibration_max_concurrent),
    "poll_interval_seconds": int(poll_interval),
    "kv_len": int(kv_len),
    "num_users": int(num_users),
    "users_per_batch": int(users_per_batch),
    "candidates_per_user": int(candidates_per_user),
    "macro_batch_size": int(macro_batch_size),
    "kv_reuse_ratio": float(kv_reuse_ratio),
    "ir_objective": ir_objective,
    "model_sizes": [v for v in model_sizes.split(",") if v],
    "chips": [v for v in chips.split(",") if v],
    "sources": [v for v in sources.split(",") if v],
    "schemes": [v for v in schemes.split(",") if v],
    "expanded_command": expanded_command.strip(),
}
Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY
}

run_summary() {
  python3 scripts/summarize_scalability_results.py \
    --layout modelsize \
    --result-root "${RESULT_ROOT}" \
    --metadata "${RUN_METADATA}"
}

run_all_cases() {
  local -a pending=()
  local -a launched=()
  local size source_idx source_label source_medium chip scheme label
  for size in "${MODEL_SIZES[@]}"; do
    for source_idx in "${!SOURCE_MEDIA[@]}"; do
      source_medium="${SOURCE_MEDIA[${source_idx}]}"
      source_label="${SOURCE_LABELS[${source_idx}]}"
      for chip in "${CHIPS[@]}"; do
        for scheme in "${SCHEMES[@]}"; do
          pending+=("${size}|${source_label}|${source_medium}|${chip}|${scheme}")
        done
      done
    done
  done

  if (( DRY_RUN )); then
    for item in "${pending[@]}"; do
      IFS='|' read -r size source_label source_medium chip scheme <<<"${item}"
      label="${size}__${source_label}__${chip}__${scheme}"
      build_run_args "${size}" "${source_label}" "${source_medium}" "${chip}" "${scheme}"
      launch_case "${label}"
    done
    return
  fi

  local next=0
  local completed active item
  while (( next < ${#pending[@]} || $(count_completed_cases launched) < ${#pending[@]} )); do
    active=$(count_active_cases launched)
    while (( next < ${#pending[@]} && active < MAX_CONCURRENT_JOBS )); do
      item="${pending[${next}]}"
      IFS='|' read -r size source_label source_medium chip scheme <<<"${item}"
      label="${size}__${source_label}__${chip}__${scheme}"
      build_run_args "${size}" "${source_label}" "${source_medium}" "${chip}" "${scheme}"
      launch_case "${label}"
      launched+=("${label}")
      next=$((next + 1))
      active=$((active + 1))
    done
    cleanup_successful_cases launched
    completed=$(count_completed_cases launched)
    active=$(count_active_cases launched)
    echo "Progress: completed=${completed}/${#pending[@]}, active=${active}, launched=${#launched[@]}"
    if (( completed == ${#pending[@]} )); then
      break
    fi
    sleep "${POLL_INTERVAL_SECONDS}"
  done
  cleanup_successful_cases launched
}

main() {
  validate_inputs
  if (( SUMMARY_ONLY )); then
    run_summary
    return
  fi
  backup_existing_result_root
  clean_old_results
  prepare_calibration
  mkdir -p "${CASES_ROOT}" "${LOG_DIR}"
  write_metadata_file
  run_validation
  run_all_cases
  run_summary
}

main "$@"
