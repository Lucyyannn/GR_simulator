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
  --base-config PATH           Simulator base config. Default: configs/910c_mini_<source-medium>.json
  --result-dir PATH           Output directory. Default: results/run_hstu_<source-medium>
  --layers N                   HSTU layer count
  --hidden N                   Hidden dimension
  --kv-len N                   Historical KV length
  --history-recompute-len N     Tail history rows recomputed from embedding instead of KV cache. Default: 0
  --num-users N                Number of users in the generated workload
  --users-per-batch N          Users per batch
  --candidates-per-user N      Candidates per user
  --macro-batch-size N         Candidate macro batch size
  --vocab N                    Embedding vocabulary size
  --seed N                     Random seed
  --op-modeling SPEC           Operator modeling modes, e.g. split=materialize,view=materialize,concat=materialize
  --attention-modeling MODE     Attention modeling mode: decomposed or fused. Default: decomposed
  --enable-kv-reuse             Enable KV row reuse metadata in generated traces
  --kv-reuse-variant MODE       KV reuse variant: global or window_topk. Default: window_topk
  --log-level LEVEL            Simulator log level. Default: info
  -h, --help                  Show this message
EOF
}

# Edit the standard experiment settings here when you want to change workload size.
SOURCE_MEDIUM="${SOURCE_MEDIUM:-ddr}"
BASE_CONFIG="${BASE_CONFIG:-}"
RESULT_DIR=""
LAYERS="${LAYERS:-4}"
HIDDEN="${HIDDEN:-256}"
KV_LEN="${KV_LEN:-1024}"
HISTORY_RECOMPUTE_LEN="${HISTORY_RECOMPUTE_LEN:-0}"
NUM_USERS="${NUM_USERS:-8}"
USERS_PER_BATCH="${USERS_PER_BATCH:-4}"
CANDIDATES_PER_USER="${CANDIDATES_PER_USER:-2048}"
MACRO_BATCH_SIZE="${MACRO_BATCH_SIZE:-1024}"
VOCAB="${VOCAB:-65536}"
SEED="${SEED:-1234}"
OP_MODELING="${OP_MODELING:-split=materialize,view=materialize,concat=materialize}"
ATTENTION_MODELING="${ATTENTION_MODELING:-decomposed}"
ENABLE_KV_REUSE="${ENABLE_KV_REUSE:-0}"
KV_REUSE_VARIANT="${KV_REUSE_VARIANT:-window_topk}"
KV_REUSE_ACTION_COUNT="${KV_REUSE_ACTION_COUNT:-4}"
KV_REUSE_WINDOW_SIZE="${KV_REUSE_WINDOW_SIZE:-1024}"
KV_REUSE_TOPK="${KV_REUSE_TOPK:-4}"
KV_REUSE_HOT_SHARE="${KV_REUSE_HOT_SHARE:-0.75}"
KV_REUSE_ACTION_OFFSET="${KV_REUSE_ACTION_OFFSET:-1}"
KV_REUSE_ACTION_STRIDE="${KV_REUSE_ACTION_STRIDE:-2}"
LOG_LEVEL="${LOG_LEVEL:-info}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-medium)
      SOURCE_MEDIUM="$2"
      shift 2
      ;;
    --base-config)
      BASE_CONFIG="$2"
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
    --history-recompute-len)
      HISTORY_RECOMPUTE_LEN="$2"
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
    --attention-modeling)
      ATTENTION_MODELING="$2"
      shift 2
      ;;
    --enable-kv-reuse)
      ENABLE_KV_REUSE=1
      shift
      ;;
    --kv-reuse-variant)
      KV_REUSE_VARIANT="$2"
      shift 2
      ;;
    --kv-reuse-action-count)
      KV_REUSE_ACTION_COUNT="$2"
      shift 2
      ;;
    --kv-reuse-window-size)
      KV_REUSE_WINDOW_SIZE="$2"
      shift 2
      ;;
    --kv-reuse-topk)
      KV_REUSE_TOPK="$2"
      shift 2
      ;;
    --kv-reuse-hot-share)
      KV_REUSE_HOT_SHARE="$2"
      shift 2
      ;;
    --kv-reuse-action-offset)
      KV_REUSE_ACTION_OFFSET="$2"
      shift 2
      ;;
    --kv-reuse-action-stride)
      KV_REUSE_ACTION_STRIDE="$2"
      shift 2
      ;;
    --log-level)
      LOG_LEVEL="$2"
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

case "${ATTENTION_MODELING}" in
  decomposed|fused) ;;
  *)
    echo "Invalid --attention-modeling: ${ATTENTION_MODELING}" >&2
    exit 1
    ;;
esac

if [[ -z "${BASE_CONFIG}" ]]; then
  BASE_CONFIG="configs/910c_mini_${SOURCE_MEDIUM}.json"
fi
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

if [[ -d "${RESULT_DIR}" ]]; then
  rm -rf "${RESULT_DIR}"
fi
mkdir -p "${RESULT_DIR}"

TRACE_ARGS=(
  --pipeline
  --compact-json
  --source-medium "${SOURCE_MEDIUM}"
  --layers "${LAYERS}"
  --hidden "${HIDDEN}"
  --kv-len "${KV_LEN}"
  --history-recompute-len "${HISTORY_RECOMPUTE_LEN}"
  --vocab "${VOCAB}"
  --seed "${SEED}"
  --num-users "${NUM_USERS}"
  --users-per-batch "${USERS_PER_BATCH}"
  --candidates-per-user "${CANDIDATES_PER_USER}"
  --macro-batch-size "${MACRO_BATCH_SIZE}"
  --op-modeling "${OP_MODELING}"
  --attention-modeling "${ATTENTION_MODELING}"
  --kv-reuse-variant "${KV_REUSE_VARIANT}"
  --kv-reuse-action-count "${KV_REUSE_ACTION_COUNT}"
  --kv-reuse-window-size "${KV_REUSE_WINDOW_SIZE}"
  --kv-reuse-topk "${KV_REUSE_TOPK}"
  --kv-reuse-hot-share "${KV_REUSE_HOT_SHARE}"
  --kv-reuse-action-offset "${KV_REUSE_ACTION_OFFSET}"
  --kv-reuse-action-stride "${KV_REUSE_ACTION_STRIDE}"
  --output "${TRACE_DIR}"
  --models-list "${MODELS_JSON}"
)

if [[ "${ENABLE_KV_REUSE}" == "1" ]]; then
  TRACE_ARGS+=(--enable-kv-reuse)
fi

python3 scripts/generate_hstu_baseline_trace.py "${TRACE_ARGS[@]}"

python3 - "${BASE_CONFIG}" "${RUNTIME_CONFIG}" "${BREAKDOWN_CSV}" <<'PY'
import json
import sys
from pathlib import Path

base_config = Path(sys.argv[1])
runtime_config = Path(sys.argv[2])
breakdown_csv = sys.argv[3]

cfg = json.loads(base_config.read_text(encoding="utf-8"))
pipeline = cfg.setdefault("pipeline", {})
hbm_cfg = cfg.get("hbm", {})
hbm_capacity = int(hbm_cfg.get("capacity_bytes", 0) or 0)
if hbm_capacity == 0:
    hbm_capacity = int(hbm_cfg.get("size_gb", 0) or 0) * (1024 ** 3)
default_residency_cap = hbm_capacity // 4 if hbm_capacity > 0 else 0
pipeline["max_preloading_models"] = 1
pipeline["layer_preload_enabled"] = True
pipeline["layer_preload_lookahead"] = pipeline.get("layer_preload_lookahead", 1)
pipeline["hbm_residency_capacity_bytes"] = (
    pipeline.get("hbm_residency_capacity_bytes", 0) or default_residency_cap
)
pipeline["breakdown_csv"] = breakdown_csv
runtime_config.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
PY

./build/bin/Simulator \
  --config "${RUNTIME_CONFIG}" \
  --models_list "${MODELS_JSON}" \
  --mode trace \
  --log_level "${LOG_LEVEL}" \
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
