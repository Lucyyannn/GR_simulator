#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
export ONNXIM_HOME="${PWD}"
export LD_LIBRARY_PATH="${PWD}/build/lib:${LD_LIBRARY_PATH-}"
BASE_CONFIG="${BASE_CONFIG:-configs/910C.json}"
[[ -f "${BASE_CONFIG}" ]] || { echo "Missing BASE_CONFIG: ${BASE_CONFIG}" >&2; exit 2; }

MODELS_JSON="${MODELS_JSON:-/tmp/test_gemm_models.json}"
cat > "${MODELS_JSON}" <<'JSON'
{"models":[{"name":"test_gemm","trace_path":"example/trace_tests/test_gemm.json"}]}
JSON

./build/bin/Simulator \
  --config "${BASE_CONFIG}" \
  --models_list "${MODELS_JSON}" \
  --mode trace \
  --log_level "${LOG_LEVEL:-info}"
