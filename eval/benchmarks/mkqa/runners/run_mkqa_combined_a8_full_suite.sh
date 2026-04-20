#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

MODEL_ROOT="${MKQA_MODEL_ROOT:-/work/training/FlexMoRE/models}"
PYTHON_BIN="${MKQA_PYTHON_BIN:-python3}"
CONFIG_TEMPLATE="${MKQA_CONFIG_TEMPLATE:-$SCRIPT_DIR/../configs/mkqa_analysis_config.combined_native.a8.example.json}"

if [[ ! -f "$CONFIG_TEMPLATE" ]]; then
  echo "Config template not found: $CONFIG_TEMPLATE" >&2
  exit 1
fi

TMP_CONFIG="$(mktemp /tmp/mkqa_combined_a8_full_config.XXXXXX.json)"
trap 'rm -f "$TMP_CONFIG"' EXIT

sed -e "s|__MODEL_ROOT__|$MODEL_ROOT|g" "$CONFIG_TEMPLATE" > "$TMP_CONFIG"

cd "$PROJECT_ROOT"
"$PYTHON_BIN" eval/benchmarks/mkqa/runners/run_mkqa_analysis_suite.py --config "$TMP_CONFIG" "$@"
