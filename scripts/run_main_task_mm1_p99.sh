#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

INPUT_ROOT="results/main_task"
OUTPUT_ROOT="results"
OUTPUT_CSV="${OUTPUT_ROOT}/p99_mm1_HSTU_middle_seq16384_bs1_cold.csv"
OUTPUT_PNG="${OUTPUT_ROOT}/p99_mm1_HSTU_middle_seq16384_bs1_cold.png"

if [[ ! -d "${INPUT_ROOT}" ]]; then
  echo "Missing input result directory: ${INPUT_ROOT}" >&2
  echo "Copy or generate main_task results under results/main_task first." >&2
  exit 1
fi

echo "Input result directory: ${INPUT_ROOT}"
echo "Output directory: ${OUTPUT_ROOT}"
python3 scripts/compute_mm1_p99.py

if [[ ! -f "${OUTPUT_CSV}" || ! -f "${OUTPUT_PNG}" ]]; then
  echo "Expected output files were not generated:" >&2
  echo "  ${OUTPUT_CSV}" >&2
  echo "  ${OUTPUT_PNG}" >&2
  exit 1
fi

echo "Wrote ${OUTPUT_CSV}"
echo "Wrote ${OUTPUT_PNG}"
