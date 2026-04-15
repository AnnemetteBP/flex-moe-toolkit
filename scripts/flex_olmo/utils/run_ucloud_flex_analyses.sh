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
: "${REMOTE_RUNNER_SCRIPT:?Set REMOTE_RUNNER_SCRIPT in the config file.}"
: "${REMOTE_RUN_CONFIG:?Set REMOTE_RUN_CONFIG in the config file.}"

SSH_TARGET="${UCLOUD_USER}@${UCLOUD_HOST}"
SSH_PORT="${UCLOUD_PORT:-22}"
REMOTE_PYTHON_BIN="${REMOTE_PYTHON_BIN:-python3}"
REMOTE_ENV_ACTIVATE="${REMOTE_ENV_ACTIVATE:-}"
REMOTE_WORKDIR="${REMOTE_WORKDIR:-}"

REMOTE_CMD=""
if [[ -n "$REMOTE_WORKDIR" ]]; then
  REMOTE_CMD+="cd $(printf '%q' "$REMOTE_WORKDIR") && "
fi
if [[ -n "$REMOTE_ENV_ACTIVATE" ]]; then
  REMOTE_CMD+="source $(printf '%q' "$REMOTE_ENV_ACTIVATE") && "
fi

REMOTE_CMD+="$(printf '%q' "$REMOTE_PYTHON_BIN") $(printf '%q' "$REMOTE_RUNNER_SCRIPT")"
REMOTE_CMD+=" --config $(printf '%q' "$REMOTE_RUN_CONFIG")"

if [[ "$DRY_RUN" -eq 1 ]]; then
  REMOTE_CMD+=" --dry-run"
fi

echo "SSH target: $SSH_TARGET"
echo "Remote command:"
echo "$REMOTE_CMD"

if [[ "$DRY_RUN" -eq 1 ]]; then
  exit 0
fi

ssh -p "$SSH_PORT" "$SSH_TARGET" "$REMOTE_CMD"
