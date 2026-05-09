#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
source "$PROJECT_ROOT/test_scripts/_common.sh"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
DEVICE="${DEVICE:-cuda}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/Track1/A-V-P/binary}"
DATA_ROOT="${DATA_ROOT:-$(dataset_root_for_split "Elder")}"
PERSONALITY_NPY="${PERSONALITY_NPY:-$(resolve_personality_npy "Elder")}"
LOGS_DIR="${LOGS_DIR:-logs/test}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python interpreter not found: $PYTHON_BIN" >&2
  exit 1
fi

if [[ ! -e "$CHECKPOINT_DIR" ]]; then
  echo "Checkpoint path not found: $CHECKPOINT_DIR" >&2
  exit 1
fi

if [[ ! -f "$PERSONALITY_NPY" ]]; then
  echo "Personality embeddings not found. Set PERSONALITY_NPY explicitly." >&2
  exit 1
fi

echo "[Track1][A-V-P][binary] $CHECKPOINT_DIR"
"$PYTHON_BIN" test.py \
  --checkpoint "$CHECKPOINT_DIR" \
  --data_root "$DATA_ROOT" \
  --personality_npy "$PERSONALITY_NPY" \
  --device "$DEVICE" \
  --logs_dir "$LOGS_DIR" \
  "$@"

