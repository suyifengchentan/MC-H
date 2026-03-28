#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN_INPUT="${PYTHON_BIN:-python}"

if ! PYTHON_BIN="$(command -v "$PYTHON_BIN_INPUT" 2>/dev/null)"; then
  echo "Python executable not found: $PYTHON_BIN_INPUT" >&2
  exit 1
fi

usage() {
  cat <<'EOF'
Usage:
  bash scripts/train_by_dataset.sh <dataset_name> [--device DEVICE] [--foreground] [extra train.py args...]

Supported dataset names:
  CIFAR10      | cifar10
  CIFAR100     | cifar100
  CUB_200_2011 | cub
  FGVC_Aircraft | aircraft
  Image10      | image10
  Image50      | image50
  Image100     | image100
  Image200     | image200
  Oxford_Flowers | flowers
  Oxford_Pet   | pet
  STL10        | stl10
  Stanford_Dogs | dogs

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

if [[ "$1" == "-h" || "$1" == "--help" ]]; then
  usage
  exit 0
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
  cifar10)
    DATASET_NAME="CIFAR10"
    CONFIG_REL="config/train/coarse_grained/cifar10_vitb16_hypersphere_dpp_config.yaml"
    ;;
  cifar100)
    DATASET_NAME="CIFAR100"
    CONFIG_REL="config/train/coarse_grained/cifar100_vitb16_hypersphere_dpp_config.yaml"
    ;;
  cub|cub_200_2011)
    DATASET_NAME="CUB_200_2011"
    CONFIG_REL="config/train/fine_grained/cub_vitb16_hypersphere_dpp_config.yaml"
    ;;
  aircraft|fgvc_aircraft)
    DATASET_NAME="FGVC_Aircraft"
    CONFIG_REL="config/train/fine_grained/fgvc_aircraft_vitb16_hypersphere_dpp_config.yaml"
    ;;
  image10)
    DATASET_NAME="Image10"
    CONFIG_REL="config/train/coarse_grained/image10_vitb16_hypersphere_dpp_config.yaml"
    ;;
  image50)
    DATASET_NAME="Image50"
    CONFIG_REL="config/train/medium_grained/image50_vitb16_hypersphere_dpp_config.yaml"
    ;;
  image100)
    DATASET_NAME="Image100"
    CONFIG_REL="config/train/medium_grained/image100_vitb16_hypersphere_dpp_config.yaml"
    ;;
  image200)
    DATASET_NAME="Image200"
    CONFIG_REL="config/train/medium_grained/image200_vitb16_hypersphere_dpp_config.yaml"
    ;;
  flowers|oxford_flowers)
    DATASET_NAME="Oxford_Flowers"
    CONFIG_REL="config/train/fine_grained/oxford_flowers_vitb16_hypersphere_dpp_config.yaml"
    ;;
  pet|oxford_pet)
    DATASET_NAME="Oxford_Pet"
    CONFIG_REL="config/train/fine_grained/oxford_pet_vitb16_hypersphere_dpp_config.yaml"
    ;;
  stl10)
    DATASET_NAME="STL10"
    CONFIG_REL="config/train/coarse_grained/stl10_vitb16_hypersphere_dpp_config.yaml"
    ;;
  dogs|stanford_dogs)
    DATASET_NAME="Stanford_Dogs"
    CONFIG_REL="config/train/fine_grained/stanford_dogs_vitb16_hypersphere_dpp_config.yaml"
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
print("1" if config.get("reload", False) else "0")
PY
)

DATASET_DIR="${CONFIG_INFO[0]}"
MODEL_PATH_REL="${CONFIG_INFO[1]}"
RELOAD_ENABLED="${CONFIG_INFO[2]}"
BASE_OUTPUT_DIR="$ROOT_DIR/$MODEL_PATH_REL"

if [[ "$RELOAD_ENABLED" == "1" ]]; then
  OUTPUT_DIR="$BASE_OUTPUT_DIR"
else
  RUN_STAMP="$(date '+%Y%m%d_%H%M%S')"
  OUTPUT_DIR="${BASE_OUTPUT_DIR}_${RUN_STAMP}"
  if [[ -e "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="${OUTPUT_DIR}_$$"
  fi
fi

PID_FILE="$OUTPUT_DIR/train.pid"
LAUNCH_LOG="$OUTPUT_DIR/launcher.log"
TRAIN_LOG="$OUTPUT_DIR/train.log"
METRICS_LOG="$OUTPUT_DIR/metrics.jsonl"
RUN_CONFIG_PATH="$OUTPUT_DIR/train_config.resolved.yaml"

if [[ ! -d "$DATASET_DIR" ]]; then
  echo "Dataset directory does not exist for $DATASET_NAME: $DATASET_DIR" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

RUN_OUTPUT_DIR="$OUTPUT_DIR" "$PYTHON_BIN" - "$CONFIG_PATH" "$RUN_CONFIG_PATH" <<'PY'
from pathlib import Path
import os
import sys
import yaml

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
config = yaml.safe_load(src.read_text(encoding="utf-8"))
config["model_path"] = os.environ["RUN_OUTPUT_DIR"]
dst.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=False), encoding="utf-8")
PY

CMD=("$PYTHON_BIN" "train.py" "--config" "$RUN_CONFIG_PATH")
if [[ -n "$DEVICE" ]]; then
  CMD+=("--device" "$DEVICE")
fi
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

cd "$ROOT_DIR"

echo "dataset: $DATASET_NAME"
echo "config: $CONFIG_REL"
echo "resolved_config: $RUN_CONFIG_PATH"
echo "dataset_dir: $DATASET_DIR"
echo "base_output_dir: $MODEL_PATH_REL"
echo "reload: $RELOAD_ENABLED"
echo "run_output_dir: $OUTPUT_DIR"
printf 'command:'
printf ' %q' "${CMD[@]}"
printf '\n'

if [[ "$FOREGROUND" -eq 1 ]]; then
  : > "$TRAIN_LOG"
  : > "$METRICS_LOG"
  exec "${CMD[@]}"
fi

: > "$TRAIN_LOG"
: > "$METRICS_LOG"
: > "$LAUNCH_LOG"
nohup "${CMD[@]}" > "$LAUNCH_LOG" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$PID_FILE"

sleep 1
if ! kill -0 "$PID" > /dev/null 2>&1; then
  rm -f "$PID_FILE"
  echo "background training exited immediately" >&2
  echo "inspect log: $OUTPUT_DIR/train.log" >&2
  echo "launcher log: $LAUNCH_LOG" >&2
  exit 1
fi

echo "started background training"
echo "pid: $PID"
echo "train_log: $OUTPUT_DIR/train.log"
echo "launcher_log: $LAUNCH_LOG"
echo "pid_file: $PID_FILE"
