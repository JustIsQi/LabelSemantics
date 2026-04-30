SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

python -m label_semantics.train \
  --model-path /home/chinese-roberta-wwm-ext \
  --data-dir data/excel_ner_data_en_augmented \
  --label-file data/excel_ner_data_en_augmented/labels.json \
  --train-file train.txt \
  --dev-file dev.txt \
  --test-file test.txt \
  --output-dir outputs/english_contrastive \
  --batch-size 64 \
  --epochs 30 \
  --learning-rate 1e-5 \
  --contrastive-weight 0.1 \
  --contrastive-temperature 0.1
  # --augment-role-prob 0.4