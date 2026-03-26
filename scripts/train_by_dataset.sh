#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/train_by_dataset.sh <dataset_name> [--device DEVICE] [--foreground] [extra train.py args...]

Supported dataset names:
  CUB_200_2011 | cub
  Image10      | image10
  Image50      | image50
  Image100     | image100
  Image200     | image200

Examples:
  bash scripts/train_by_dataset.sh Image200
  bash scripts/train_by_dataset.sh image10 --device cuda:0
  bash scripts/train_by_dataset.sh cub --foreground --device cuda:1
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

DATASET_INPUT="$1"
shift

DEVICE=""
FOREGROUND=0
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --device)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --device" >&2
        exit 1
      fi
      DEVICE="$2"
      shift 2
      ;;
    --foreground)
      FOREGROUND=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

DATASET_KEY="$(printf '%s' "$DATASET_INPUT" | tr '[:upper:]' '[:lower:]')"

case "$DATASET_KEY" in
  cub|cub_200_2011)
    DATASET_NAME="CUB_200_2011"
    CONFIG_REL="config/train/fine_grained/cub_vitl14_hypersphere_dpp_config.yaml"
    ;;
  image10)
    DATASET_NAME="Image10"
    CONFIG_REL="config/train/coarse_grained/image10_vitl14_hypersphere_dpp_config.yaml"
    ;;
  image50)
    DATASET_NAME="Image50"
    CONFIG_REL="config/train/medium_grained/image50_vitl14_hypersphere_dpp_config.yaml"
    ;;
  image100)
    DATASET_NAME="Image100"
    CONFIG_REL="config/train/medium_grained/image100_vitl14_hypersphere_dpp_config.yaml"
    ;;
  image200)
    DATASET_NAME="Image200"
    CONFIG_REL="config/train/medium_grained/image200_vitl14_hypersphere_dpp_config.yaml"
    ;;
  *)
    echo "Unsupported dataset name: $DATASET_INPUT" >&2
    usage
    exit 1
    ;;
esac

CONFIG_PATH="$ROOT_DIR/$CONFIG_REL"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config file not found: $CONFIG_PATH" >&2
  exit 1
fi

mapfile -t CONFIG_INFO < <(
  "$PYTHON_BIN" - "$CONFIG_PATH" <<'PY'
from pathlib import Path
import sys
import yaml

config_path = Path(sys.argv[1])
config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
print(config["dataset_dir"])
print(config["model_path"])
PY
)

DATASET_DIR="${CONFIG_INFO[0]}"
MODEL_PATH_REL="${CONFIG_INFO[1]}"
OUTPUT_DIR="$ROOT_DIR/$MODEL_PATH_REL"
PID_FILE="$OUTPUT_DIR/train.pid"

if [[ ! -d "$DATASET_DIR" ]]; then
  echo "Dataset directory does not exist for $DATASET_NAME: $DATASET_DIR" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

if [[ -f "$PID_FILE" ]]; then
  EXISTING_PID="$(tr -d '[:space:]' < "$PID_FILE")"
  if [[ -n "$EXISTING_PID" ]] && ps -p "$EXISTING_PID" > /dev/null 2>&1; then
    echo "Training for $DATASET_NAME is already running with PID $EXISTING_PID" >&2
    echo "PID file: $PID_FILE" >&2
    exit 1
  fi
fi

CMD=("$PYTHON_BIN" "train.py" "--config" "$CONFIG_REL")
if [[ -n "$DEVICE" ]]; then
  CMD+=("--device" "$DEVICE")
fi
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

cd "$ROOT_DIR"

echo "dataset: $DATASET_NAME"
echo "config: $CONFIG_REL"
echo "dataset_dir: $DATASET_DIR"
echo "output_dir: $MODEL_PATH_REL"
printf 'command:'
printf ' %q' "${CMD[@]}"
printf '\n'

if [[ "$FOREGROUND" -eq 1 ]]; then
  exec "${CMD[@]}"
fi

printf -v CMD_ESCAPED '%q ' "${CMD[@]}"
setsid nohup bash -lc "exec ${CMD_ESCAPED}" > /dev/null 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$PID_FILE"

sleep 1
if ! ps -p "$PID" > /dev/null 2>&1; then
  rm -f "$PID_FILE"
  echo "background training exited immediately" >&2
  echo "inspect log: $OUTPUT_DIR/train.log" >&2
  exit 1
fi

echo "started background training"
echo "pid: $PID"
echo "train_log: $OUTPUT_DIR/train.log"
echo "pid_file: $PID_FILE"
