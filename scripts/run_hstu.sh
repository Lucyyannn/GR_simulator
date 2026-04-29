#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

usage() {
  cat <<'EOF'
Usage: bash scripts/run_hstu.sh [options]

Options:
  --source-medium {ddr|ssd}   Initial source medium for weights/KV/embedding rows. Default: ddr
  --result-dir PATH           Output directory. Default: results/run_hstu_<source-medium>
  --layers N                  Default: 4
  --hidden N                  Default: 256
  --kv-len N                  Default: 1024
  --num-users N               Default: 8
  --users-per-batch N         Default: 4
  --candidates-per-user N     Default: 1024
  --macro-batch-size N        Default: 256
  --vocab N                   Default: 128
  --seed N                    Default: 1234
  --op-modeling SPEC          Default: split=materialize,view=materialize,concat=materialize
  -h, --help                  Show this message
EOF
}

SOURCE_MEDIUM="ddr"
RESULT_DIR=""
LAYERS=4
HIDDEN=256
KV_LEN=1024
NUM_USERS=8
USERS_PER_BATCH=4
CANDIDATES_PER_USER=1024
MACRO_BATCH_SIZE=256
VOCAB=128
SEED=1234
OP_MODELING="split=materialize,view=materialize,concat=materialize"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-medium)
      SOURCE_MEDIUM="$2"
      shift 2
      ;;
    --result-dir)
      RESULT_DIR="$2"
      shift 2
      ;;
    --layers)
      LAYERS="$2"
      shift 2
      ;;
    --hidden)
      HIDDEN="$2"
      shift 2
      ;;
    --kv-len|--history-len)
      KV_LEN="$2"
      shift 2
      ;;
    --num-users)
      NUM_USERS="$2"
      shift 2
      ;;
    --users-per-batch)
      USERS_PER_BATCH="$2"
      shift 2
      ;;
    --candidates-per-user)
      CANDIDATES_PER_USER="$2"
      shift 2
      ;;
    --macro-batch-size)
      MACRO_BATCH_SIZE="$2"
      shift 2
      ;;
    --vocab)
      VOCAB="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --op-modeling)
      OP_MODELING="$2"
      shift 2
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

case "${SOURCE_MEDIUM}" in
  ddr|ssd) ;;
  *)
    echo "Invalid --source-medium: ${SOURCE_MEDIUM}" >&2
    exit 1
    ;;
esac

BASE_CONFIG="configs/910c_mini_${SOURCE_MEDIUM}.json"
if [[ ! -f "${BASE_CONFIG}" ]]; then
  echo "Base config not found: ${BASE_CONFIG}" >&2
  exit 1
fi

if [[ -z "${RESULT_DIR}" ]]; then
  RESULT_DIR="results/run_hstu_${SOURCE_MEDIUM}"
fi

TRACE_DIR="${RESULT_DIR}/traces"
MODELS_JSON="${RESULT_DIR}/models.json"
RUNTIME_CONFIG="${RESULT_DIR}/runtime_config.json"
BREAKDOWN_CSV="${RESULT_DIR}/layer_breakdown.csv"
TIMELINE_PNG="${RESULT_DIR}/layer_timeline.png"
LOG_PATH="${RESULT_DIR}/layer.log"

mkdir -p "${RESULT_DIR}"

python3 scripts/generate_hstu_baseline_trace.py \
  --pipeline --shared-trace --compact-json \
  --source-medium "${SOURCE_MEDIUM}" \
  --layers "${LAYERS}" \
  --hidden "${HIDDEN}" \
  --kv-len "${KV_LEN}" \
  --vocab "${VOCAB}" \
  --seed "${SEED}" \
  --num-users "${NUM_USERS}" \
  --users-per-batch "${USERS_PER_BATCH}" \
  --candidates-per-user "${CANDIDATES_PER_USER}" \
  --macro-batch-size "${MACRO_BATCH_SIZE}" \
  --op-modeling "${OP_MODELING}" \
  --output "${TRACE_DIR}" \
  --models-list "${MODELS_JSON}"

python3 - "${BASE_CONFIG}" "${RUNTIME_CONFIG}" "${BREAKDOWN_CSV}" <<'PY'
import json
import sys
from pathlib import Path

base_config = Path(sys.argv[1])
runtime_config = Path(sys.argv[2])
breakdown_csv = sys.argv[3]

cfg = json.loads(base_config.read_text(encoding="utf-8"))
pipeline = cfg.setdefault("pipeline", {})
pipeline["max_preloading_models"] = 1
pipeline["layer_preload_enabled"] = True
pipeline["layer_preload_lookahead"] = pipeline.get("layer_preload_lookahead", 1)
pipeline["hbm_residency_capacity_bytes"] = pipeline.get(
    "hbm_residency_capacity_bytes", 0
)
pipeline["breakdown_csv"] = breakdown_csv
runtime_config.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
PY

./build/bin/Simulator \
  --config "${RUNTIME_CONFIG}" \
  --models_list "${MODELS_JSON}" \
  --mode trace \
  --log_level info \
  > "${LOG_PATH}" 2>&1

python3 scripts/plot_pipeline_timeline.py \
  "${BREAKDOWN_CSV}" \
  "${TIMELINE_PNG}"

echo "Result dir: ${RESULT_DIR}"
echo "Config: ${RUNTIME_CONFIG}"
echo "Models: ${MODELS_JSON}"
echo "Breakdown: ${BREAKDOWN_CSV}"
echo "Timeline: ${TIMELINE_PNG}"
echo "Log: ${LOG_PATH}"
