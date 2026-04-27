#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Required for real training. Override with a HuggingFace model name or local model directory.
MODEL_PATH="${MODEL_PATH:-/root/workspace/berts/chinese-roberta-wwm-ext}"

EXCEL_FILE="${EXCEL_FILE:-test_data_0413.xlsx}"
DATA_DIR="${DATA_DIR:-excel_ner_data}"
LABEL_FILE="${LABEL_FILE:-${DATA_DIR}/labels.json}"

PRETRAIN_OUTPUT_DIR="${PRETRAIN_OUTPUT_DIR:-outputs/pretrain}"
FEWSHOT_OUTPUT_DIR="${FEWSHOT_OUTPUT_DIR:-outputs/fewshot}"

MAX_LENGTH="${MAX_LENGTH:-128}"
PRETRAIN_BATCH_SIZE="${PRETRAIN_BATCH_SIZE:-16}"
FEWSHOT_BATCH_SIZE="${FEWSHOT_BATCH_SIZE:-16}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-100}"
FEWSHOT_EPOCHS="${FEWSHOT_EPOCHS:-100}"
PRETRAIN_LR="${PRETRAIN_LR:-1e-5}"
FEWSHOT_LR="${FEWSHOT_LR:-5e-5}"
WARMUP_STEPS="${WARMUP_STEPS:-0}"
DEVICE="${DEVICE:-}"

echo "==> Converting Excel annotations to BIO data"
python convert_excel_to_bio.py \
  --input-file "$EXCEL_FILE" \
  --output-dir "$DATA_DIR"

DEVICE_ARGS=()
if [[ -n "$DEVICE" ]]; then
  DEVICE_ARGS=(--device "$DEVICE")
fi

echo "==> Training label-semantics model"
python train.py \
  --model-path "$MODEL_PATH" \
  --data-dir "$DATA_DIR" \
  --label-file "$LABEL_FILE" \
  --train-file train.txt \
  --dev-file dev.txt \
  --test-file test.txt \
  --output-dir "$PRETRAIN_OUTPUT_DIR" \
  --batch-size "$PRETRAIN_BATCH_SIZE" \
  --max-length "$MAX_LENGTH" \
  --epochs "$PRETRAIN_EPOCHS" \
  --learning-rate "$PRETRAIN_LR" \
  --warmup-steps "$WARMUP_STEPS" \
  "${DEVICE_ARGS[@]}"

BEST_CHECKPOINT="$(python - "$PRETRAIN_OUTPUT_DIR" <<'PY'
from pathlib import Path
import sys

output_dir = Path(sys.argv[1])
checkpoints = sorted(output_dir.glob("model_epoch*_f1*.pth"))
if not checkpoints:
    raise SystemExit(f"No checkpoints found in {output_dir}")

def score(path):
    stem = path.stem
    return float(stem.rsplit("f1", 1)[1])

print(max(checkpoints, key=score))
PY
)"

echo "==> Fine-tuning from best checkpoint: $BEST_CHECKPOINT"
python train.py \
  --model-path "$MODEL_PATH" \
  --data-dir "$DATA_DIR" \
  --label-file "$LABEL_FILE" \
  --train-file train.txt \
  --dev-file dev.txt \
  --test-file test.txt \
  --output-dir "$FEWSHOT_OUTPUT_DIR" \
  --checkpoint-path "$BEST_CHECKPOINT" \
  --batch-size "$FEWSHOT_BATCH_SIZE" \
  --max-length "$MAX_LENGTH" \
  --epochs "$FEWSHOT_EPOCHS" \
  --learning-rate "$FEWSHOT_LR" \
  --warmup-steps "$WARMUP_STEPS" \
  "${DEVICE_ARGS[@]}"

echo "==> Done"
