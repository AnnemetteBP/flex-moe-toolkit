#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <config.env> [--dry-run]" >&2
  exit 1
fi

CONFIG_PATH="$1"
shift || true

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config file not found: $CONFIG_PATH" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$CONFIG_PATH"

: "${UCLOUD_HOST:?Set UCLOUD_HOST in the config file.}"
: "${UCLOUD_USER:?Set UCLOUD_USER in the config file.}"
: "${REMOTE_EVAL_SCRIPT:?Set REMOTE_EVAL_SCRIPT in the config file.}"
: "${REMOTE_OUTPUT_ROOT:?Set REMOTE_OUTPUT_ROOT in the config file.}"

SSH_TARGET="${UCLOUD_USER}@${UCLOUD_HOST}"
SSH_PORT="${UCLOUD_PORT:-22}"
REMOTE_PYTHON_BIN="${REMOTE_PYTHON_BIN:-python3}"
REMOTE_ENV_ACTIVATE="${REMOTE_ENV_ACTIVATE:-}"
REMOTE_WORKDIR="${REMOTE_WORKDIR:-}"
REMOTE_TOKENIZER_PATH="${REMOTE_TOKENIZER_PATH:-}"
REMOTE_DEVICE="${REMOTE_DEVICE:-cuda}"
REMOTE_DTYPE="${REMOTE_DTYPE:-auto}"
REMOTE_MAX_LENGTH="${REMOTE_MAX_LENGTH:-1024}"
REMOTE_MAX_EXAMPLES="${REMOTE_MAX_EXAMPLES:-}"
REMOTE_SUMMARY_JSONL="${REMOTE_SUMMARY_JSONL:-}"
MODEL_LIST_FILE="${MODEL_LIST_FILE:-}"
DATASET_LIST_FILE="${DATASET_LIST_FILE:-}"
MODEL_PATHS="${MODEL_PATHS:-}"
DATASET_PATHS="${DATASET_PATHS:-}"

build_repeated_args() {
  local flag="$1"
  local list_file="$2"
  local inline_values="$3"
  local output=""

  if [[ -n "$list_file" ]]; then
    if [[ ! -f "$list_file" ]]; then
      echo "List file not found: $list_file" >&2
      exit 1
    fi
    while IFS= read -r line; do
      [[ -z "$line" || "$line" =~ ^# ]] && continue
      output+=" $(printf '%q' "$flag") $(printf '%q' "$line")"
    done < "$list_file"
  fi

  if [[ -n "$inline_values" ]]; then
    while IFS= read -r line; do
      [[ -z "$line" || "$line" =~ ^# ]] && continue
      output+=" $(printf '%q' "$flag") $(printf '%q' "$line")"
    done <<< "$(printf '%s\n' "$inline_values" | tr ',' '\n')"
  fi

  printf '%s' "$output"
}

MODEL_ARGS="$(build_repeated_args --model-path "$MODEL_LIST_FILE" "$MODEL_PATHS")"
DATASET_ARGS="$(build_repeated_args --dataset "$DATASET_LIST_FILE" "$DATASET_PATHS")"

if [[ -z "$MODEL_ARGS" ]]; then
  echo "No model paths configured. Set MODEL_LIST_FILE or MODEL_PATHS." >&2
  exit 1
fi

if [[ -z "$DATASET_ARGS" ]]; then
  echo "No dataset paths configured. Set DATASET_LIST_FILE or DATASET_PATHS." >&2
  exit 1
fi

REMOTE_CMD=""
if [[ -n "$REMOTE_WORKDIR" ]]; then
  REMOTE_CMD+="cd $(printf '%q' "$REMOTE_WORKDIR") && "
fi
if [[ -n "$REMOTE_ENV_ACTIVATE" ]]; then
  REMOTE_CMD+="source $(printf '%q' "$REMOTE_ENV_ACTIVATE") && "
fi

REMOTE_CMD+="$(printf '%q' "$REMOTE_PYTHON_BIN") $(printf '%q' "$REMOTE_EVAL_SCRIPT")"
REMOTE_CMD+=" --output-root $(printf '%q' "$REMOTE_OUTPUT_ROOT")"
REMOTE_CMD+=" --device $(printf '%q' "$REMOTE_DEVICE")"
REMOTE_CMD+=" --dtype $(printf '%q' "$REMOTE_DTYPE")"
REMOTE_CMD+=" --max-length $(printf '%q' "$REMOTE_MAX_LENGTH")"
REMOTE_CMD+="$MODEL_ARGS"
REMOTE_CMD+="$DATASET_ARGS"

if [[ -n "$REMOTE_TOKENIZER_PATH" ]]; then
  REMOTE_CMD+=" --tokenizer-path $(printf '%q' "$REMOTE_TOKENIZER_PATH")"
fi

if [[ -n "$REMOTE_MAX_EXAMPLES" ]]; then
  REMOTE_CMD+=" --max-examples $(printf '%q' "$REMOTE_MAX_EXAMPLES")"
fi

if [[ -n "$REMOTE_SUMMARY_JSONL" ]]; then
  REMOTE_CMD+=" --summary-jsonl $(printf '%q' "$REMOTE_SUMMARY_JSONL")"
fi

echo "SSH target: $SSH_TARGET"
echo "Remote command:"
echo "$REMOTE_CMD"

if [[ "$DRY_RUN" -eq 1 ]]; then
  exit 0
fi

ssh -p "$SSH_PORT" "$SSH_TARGET" "$REMOTE_CMD"
