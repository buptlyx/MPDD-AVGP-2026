#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(CDPATH= cd "$SCRIPT_DIR/../../.." && pwd)

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-checkpoints}"
LOGS_DIR="${LOGS_DIR:-logs}"
DATA_ROOT="${DATA_ROOT:-MPDD-AVG2026-trainval/Elder}"
SPLIT_CSV="${SPLIT_CSV:-MPDD-AVG2026-trainval/Elder/split_labels_train.csv}"
PERSONALITY_NPY="${PERSONALITY_NPY:-MPDD-AVG2026-trainval/Elder/descriptions_embeddings_with_ids.npy}"
SEED="${SEED:-42}"
VAL_RATIO="${VAL_RATIO:-0.1}"
EPOCHS="${EPOCHS:-320}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LR="${LR:-8e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
HIDDEN_DIM="${HIDDEN_DIM:-128}"
DROPOUT="${DROPOUT:-0.3}"
PATIENCE="${PATIENCE:-90}"
MIN_DELTA="${MIN_DELTA:-1e-4}"
TARGET_T="${TARGET_T:-128}"
ENCODER_TYPE="${ENCODER_TYPE:-bilstm_mean}"
AUDIO_FEATURE="${AUDIO_FEATURE:-mfcc}"
VIDEO_FEATURE="${VIDEO_FEATURE:-densenet}"
SELECTION_METRIC="${SELECTION_METRIC:-kappa}"
CLS_LOSS_WEIGHT="${CLS_LOSS_WEIGHT:-3.0}"
REG_LOSS_WEIGHT="${REG_LOSS_WEIGHT:-0.1}"
WEIGHTED_SAMPLER="${WEIGHTED_SAMPLER:-1}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-track1_elder_tuned_avgp_${AUDIO_FEATURE}_${VIDEO_FEATURE}_ternary_kappa_v1}"

WEIGHTED_SAMPLER_ARG=""
case "$WEIGHTED_SAMPLER" in
  1|true|TRUE|yes|YES|on|ON)
    WEIGHTED_SAMPLER_ARG="--weighted_sampler"
    ;;
esac

echo "[Track1][A-V-G+P][ternary] seed=${SEED} target_t=${TARGET_T}"
cd "$PROJECT_ROOT"
"$PYTHON_BIN" "$PROJECT_ROOT/train.py" \
  --track Track1 \
  --task ternary \
  --subtrack "A-V-G+P" \
  --encoder_type "$ENCODER_TYPE" \
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
  --num_workers "$NUM_WORKERS" \
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
  --selection_metric "$SELECTION_METRIC" \
  --cls_loss_weight "$CLS_LOSS_WEIGHT" \
  --reg_loss_weight "$REG_LOSS_WEIGHT" \
  $WEIGHTED_SAMPLER_ARG \
  "$@"
