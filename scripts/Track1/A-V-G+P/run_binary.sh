#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-checkpoints}"
LOGS_DIR="${LOGS_DIR:-logs}"
DATA_ROOT="${DATA_ROOT:-MPDD-AVG-2026-trainval/Elder}"
SPLIT_CSV="${SPLIT_CSV:-MPDD-AVG-2026-trainval/Elder/split_labels_train.csv}"
PERSONALITY_NPY="${PERSONALITY_NPY:-MPDD-AVG-2026-trainval/Elder/descriptions_embeddings_with_ids.npy}"
SEED="${SEED:-42}"
VAL_RATIO="${VAL_RATIO:-0.1}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LR="${LR:-7e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
HIDDEN_DIM="${HIDDEN_DIM:-64}"
DROPOUT="${DROPOUT:-0.5}"
PATIENCE="${PATIENCE:-20}"
MIN_DELTA="${MIN_DELTA:-1e-4}"
TARGET_T="${TARGET_T:-128}"
AUDIO_FEATURE="${AUDIO_FEATURE:-mfcc}"
VIDEO_FEATURE="${VIDEO_FEATURE:-resnet}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-track1_elder_final6_avgp_binary_mfcc_resnet}"

echo "[Track1][A-V-G+P][binary] audio=${AUDIO_FEATURE} video=${VIDEO_FEATURE} seed=${SEED} target_t=${TARGET_T}"
cd "$PROJECT_ROOT"
"$PYTHON_BIN" "$PROJECT_ROOT/train.py" \
  --track Track1 \
  --task binary \
  --subtrack A-V-G+P \
  --encoder_type bilstm_mean \
  --audio_feature "$AUDIO_FEATURE" \
  --video_feature "$VIDEO_FEATURE" \
  --experiment_name "$EXPERIMENT_NAME" \
  --data_root "$DATA_ROOT" \
  --split_csv "$SPLIT_CSV" \
  --personality_npy "$PERSONALITY_NPY" \
  --seed "$SEED" \
  --val_ratio "$VAL_RATIO" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --lr "$LR" \
  --weight_decay "$WEIGHT_DECAY" \
  --hidden_dim "$HIDDEN_DIM" \
  --dropout "$DROPOUT" \
  --patience "$PATIENCE" \
  --min_delta "$MIN_DELTA" \
  --checkpoints_dir "$CHECKPOINTS_DIR" \
  --logs_dir "$LOGS_DIR" \
  --target_t "$TARGET_T" \
  --device "$DEVICE" \
  "$@"
