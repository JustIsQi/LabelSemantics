#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Override any of these via env before launching, e.g.:
#   LS_WORKERS=4 LS_DEVICE=cuda:0 bash scripts/serve.sh
export LS_MODEL_PATH="${LS_MODEL_PATH:-outputs/best_model}"
export LS_LABEL_FILE="${LS_LABEL_FILE:-data/excel_ner_data/labels.json}"
export LS_MAX_LENGTH="${LS_MAX_LENGTH:-128}"
export LS_HOST="${LS_HOST:-0.0.0.0}"
export LS_PORT="${LS_PORT:-8000}"
export LS_WORKERS="${LS_WORKERS:-4}"
export LS_LOG_LEVEL="${LS_LOG_LEVEL:-info}"

python -m service \
  --host "$LS_HOST" \
  --port "$LS_PORT" \
  --workers "$LS_WORKERS" \
  --log-level "$LS_LOG_LEVEL"
