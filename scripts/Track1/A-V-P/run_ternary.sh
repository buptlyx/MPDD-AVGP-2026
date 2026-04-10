#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-3407}"
EPOCHS="${EPOCHS:-120}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LR="${LR:-3e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
HIDDEN_DIM="${HIDDEN_DIM:-64}"
DROPOUT="${DROPOUT:-0.4}"
PATIENCE="${PATIENCE:-25}"
MIN_DELTA="${MIN_DELTA:-1e-4}"
TARGET_T="${TARGET_T:-128}"
AUDIO_FEATURES=(${AUDIO_FEATURES:-mfcc opensmile wav2vec})
VIDEO_FEATURES=(${VIDEO_FEATURES:-densenet resnet openface})

for AUDIO_FEATURE in "${AUDIO_FEATURES[@]}"; do
  for VIDEO_FEATURE in "${VIDEO_FEATURES[@]}"; do
    echo "[Track1][A-V+P][ternary] audio=${AUDIO_FEATURE} video=${VIDEO_FEATURE}"
    "$PYTHON_BIN" "$PROJECT_ROOT/train.py" \
      --track Track1 \
      --task ternary \
      --subtrack A-V+P \
      --encoder_type bilstm_mean \
      --audio_feature "$AUDIO_FEATURE" \
      --video_feature "$VIDEO_FEATURE" \
      --seed "$SEED" \
      --epochs "$EPOCHS" \
      --batch_size "$BATCH_SIZE" \
      --lr "$LR" \
      --weight_decay "$WEIGHT_DECAY" \
      --hidden_dim "$HIDDEN_DIM" \
      --dropout "$DROPOUT" \
      --patience "$PATIENCE" \
      --min_delta "$MIN_DELTA" \
      --target_t "$TARGET_T" \
      --device "$DEVICE" \
      "$@"
  done
done
