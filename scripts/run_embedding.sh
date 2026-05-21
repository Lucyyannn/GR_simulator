#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
export ONNXIM_HOME="${PWD}"
export LD_LIBRARY_PATH="${PWD}/build/lib:${LD_LIBRARY_PATH-}"

MODELS_JSON="${MODELS_JSON:-/tmp/test_embedding_models.json}"
cat > "${MODELS_JSON}" <<'JSON'
{"models":[{"name":"test_embedding","trace_path":"example/trace_tests/test_embedding.json"}]}
JSON

./build/bin/Simulator \
  --config configs/910C.json \
  --models_list "${MODELS_JSON}" \
  --mode trace \
  --log_level "${LOG_LEVEL:-info}"
