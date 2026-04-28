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
export LS_KILL_OLD="${LS_KILL_OLD:-1}"
export LS_BACKGROUND="${LS_BACKGROUND:-1}"
export LS_PID_FILE="${LS_PID_FILE:-logs/service.pid}"
export LS_LOG_FILE="${LS_LOG_FILE:-logs/service.log}"

mkdir -p "$(dirname "$LS_PID_FILE")" "$(dirname "$LS_LOG_FILE")"

if [[ "$LS_KILL_OLD" == "1" ]]; then
  if [[ -f "$LS_PID_FILE" ]]; then
    OLD_PID="$(cat "$LS_PID_FILE")"
    if [[ -n "$OLD_PID" ]] && kill -0 "$OLD_PID" 2>/dev/null; then
      echo "Stopping old service process: $OLD_PID"
      kill "$OLD_PID"
      for _ in {1..20}; do
        if ! kill -0 "$OLD_PID" 2>/dev/null; then
          break
        fi
        sleep 0.2
      done
      if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "Force stopping old service process: $OLD_PID"
        kill -9 "$OLD_PID"
      fi
    fi
    rm -f "$LS_PID_FILE"
  fi

  mapfile -t OLD_PIDS < <(pgrep -f "python -m service .*--port $LS_PORT" || true)
  if (( ${#OLD_PIDS[@]} > 0 )); then
    echo "Stopping old service processes on port $LS_PORT: ${OLD_PIDS[*]}"
    kill "${OLD_PIDS[@]}" 2>/dev/null || true
  fi
fi

CMD=(
  python -m service
  --host "$LS_HOST"
  --port "$LS_PORT"
  --workers "$LS_WORKERS"
  --log-level "$LS_LOG_LEVEL"
)

if [[ "$LS_BACKGROUND" == "1" ]]; then
  nohup "${CMD[@]}" >"$LS_LOG_FILE" 2>&1 &
  echo "$!" > "$LS_PID_FILE"
  echo "Service started in background, pid=$(cat "$LS_PID_FILE"), log=$LS_LOG_FILE"
else
  exec "${CMD[@]}"
fi
