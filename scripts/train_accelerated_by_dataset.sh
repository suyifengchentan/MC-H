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
  bash scripts/train_accelerated_by_dataset.sh <dataset_name> [<dataset_name> ...] [--device DEVICE] [--foreground] [-- extra train args...]
  bash scripts/train_accelerated_by_dataset.sh --datasets <dataset_name> [<dataset_name> ...] [--device DEVICE] [--foreground] [-- extra train args...]

Behavior:
  - Uses torchrun + train_parallel_accel.py
  - Single dataset defaults to background launch with nohup unless --foreground is set
  - Multiple datasets run sequentially, waiting for each dataset to finish
  - --device can be a comma-separated GPU list such as 0,1 or a CUDA spec like cuda:0
  - Creates a timestamped output directory when reload=false, same as scripts/train_by_dataset.sh

Supported dataset names:
  CIFAR10      | cifar10
  CIFAR100     | cifar100
  CUB_200_2011 | cub
  FGVC_Aircraft | aircraft
  Image10      | image10
  ImageNet1K   | imagenet1k | imagenet-1k
  Image50      | image50
  Image100     | image100
  Image200     | image200
  TinyImageNet200 | tinyimagenet200 | tiny-imagenet-200 | tiny200
  Oxford_Flowers | flowers
  Oxford_Pet   | pet
  STL10        | stl10
  Stanford_Dogs | dogs

Examples:
  bash scripts/train_accelerated_by_dataset.sh ImageNet1K
  bash scripts/train_accelerated_by_dataset.sh ImageNet1K --device 0,1
  bash scripts/train_accelerated_by_dataset.sh ImageNet1K --device 0,1 -- --disable-cudnn-benchmark
  bash scripts/train_accelerated_by_dataset.sh --datasets image50 image100 --device 0,1
EOF
}

DEVICE=""
FOREGROUND=0
EXTRA_ARGS=()
DATASET_INPUTS=()
PARSE_EXTRA_ARGS=0

while [[ $# -gt 0 ]]; do
  if [[ "$PARSE_EXTRA_ARGS" -eq 1 ]]; then
    EXTRA_ARGS+=("$1")
    shift
    continue
  fi

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
    --datasets)
      shift
      if [[ $# -lt 1 || "$1" == "--" || "$1" == --* ]]; then
        echo "Missing dataset names after --datasets" >&2
        exit 1
      fi
      while [[ $# -gt 0 && "$1" != "--" && "$1" != --* ]]; do
        DATASET_INPUTS+=("$1")
        shift
      done
      ;;
    --)
      PARSE_EXTRA_ARGS=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -*)
      EXTRA_ARGS+=("$1")
      PARSE_EXTRA_ARGS=1
      shift
      ;;
    *)
      DATASET_INPUTS+=("$1")
      shift
      ;;
  esac
done

if [[ ${#DATASET_INPUTS[@]} -eq 0 ]]; then
  usage
  exit 1
fi

resolve_dataset() {
  local dataset_input="$1"
  local dataset_key
  dataset_key="$(printf '%s' "$dataset_input" | tr '[:upper:]' '[:lower:]')"

  case "$dataset_key" in
    cifar10)
      printf '%s\n%s\n' "CIFAR10" "config/train/coarse_grained/cifar10_vitb16_hypersphere_dpp_config.yaml"
      ;;
    cifar100)
      printf '%s\n%s\n' "CIFAR100" "config/train/coarse_grained/cifar100_vitb16_hypersphere_dpp_config.yaml"
      ;;
    cub|cub_200_2011)
      printf '%s\n%s\n' "CUB_200_2011" "config/train/fine_grained/cub_vitb16_hypersphere_dpp_config.yaml"
      ;;
    aircraft|fgvc_aircraft)
      printf '%s\n%s\n' "FGVC_Aircraft" "config/train/fine_grained/fgvc_aircraft_vitb16_hypersphere_dpp_config.yaml"
      ;;
    image10)
      printf '%s\n%s\n' "Image10" "config/train/coarse_grained/image10_vitb16_hypersphere_dpp_config.yaml"
      ;;
    imagenet1k|imagenet-1k|image1k)
      printf '%s\n%s\n' "ImageNet1K" "config/train/coarse_grained/imagenet1k_vitb16_hypersphere_dpp_config.yaml"
      ;;
    image50)
      printf '%s\n%s\n' "Image50" "config/train/medium_grained/image50_vitb16_hypersphere_dpp_config.yaml"
      ;;
    image100)
      printf '%s\n%s\n' "Image100" "config/train/medium_grained/image100_vitb16_hypersphere_dpp_config.yaml"
      ;;
    image200)
      printf '%s\n%s\n' "Image200" "config/train/medium_grained/image200_vitb16_hypersphere_dpp_config.yaml"
      ;;
    tinyimagenet200|tiny-imagenet-200|tiny_imagenet200|tiny200)
      printf '%s\n%s\n' "TinyImageNet200" "config/train/medium_grained/tiny_imagenet200_vitb16_hypersphere_dpp_config.yaml"
      ;;
    flowers|oxford_flowers)
      printf '%s\n%s\n' "Oxford_Flowers" "config/train/fine_grained/oxford_flowers_vitb16_hypersphere_dpp_config.yaml"
      ;;
    pet|oxford_pet)
      printf '%s\n%s\n' "Oxford_Pet" "config/train/fine_grained/oxford_pet_vitb16_hypersphere_dpp_config.yaml"
      ;;
    stl10)
      printf '%s\n%s\n' "STL10" "config/train/coarse_grained/stl10_vitb16_hypersphere_dpp_config.yaml"
      ;;
    dogs|stanford_dogs)
      printf '%s\n%s\n' "Stanford_Dogs" "config/train/fine_grained/stanford_dogs_vitb16_hypersphere_dpp_config.yaml"
      ;;
    *)
      echo "Unsupported dataset name: $dataset_input" >&2
      usage
      exit 1
      ;;
  esac
}

read_config_info() {
  local config_path="$1"
  "$PYTHON_BIN" - "$config_path" <<'PY'
from pathlib import Path
import sys
import yaml

config_path = Path(sys.argv[1])
config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
print(config["dataset_dir"])
print(config["model_path"])
print("1" if config.get("reload", False) else "0")
PY
}

resolve_parallel_launch() {
  local device_input="${1:-}"
  "$PYTHON_BIN" - "$device_input" <<'PY'
import os
import sys

import torch


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def normalize_visible_devices(spec: str) -> list[str] | None:
    normalized = spec.strip()
    if not normalized:
        return None

    lower = normalized.lower()
    if lower in {"cuda", "cuda:all", "all", "auto"}:
        return None

    if lower.startswith("cuda:"):
        suffix = normalized[5:]
        if "," in suffix:
            parts = split_csv(suffix)
            return parts if all(part.isdigit() for part in parts) else None
        return [suffix] if suffix.isdigit() else None

    if "," in normalized:
        parts = split_csv(normalized)
        return parts if all(part.isdigit() for part in parts) else None

    return [normalized] if normalized.isdigit() else None


requested_spec = sys.argv[1]
requested_devices = normalize_visible_devices(requested_spec)
cuda_visible_devices = ""

if requested_devices is not None:
    cuda_visible_devices = ",".join(requested_devices)
    nproc_per_node = len(requested_devices)
else:
    current_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if current_visible:
        nproc_per_node = len(split_csv(current_visible))
    else:
        nproc_per_node = torch.cuda.device_count()

if nproc_per_node < 1:
    raise SystemExit("Accelerated parallel training requires at least one visible CUDA device.")

print(cuda_visible_devices)
print(nproc_per_node)
PY
}

build_output_dir() {
  local base_output_dir="$1"
  local reload_enabled="$2"

  if [[ "$reload_enabled" == "1" ]]; then
    printf '%s\n' "$base_output_dir"
    return
  fi

  local run_stamp candidate suffix
  run_stamp="$(date '+%Y%m%d_%H%M%S')"
  candidate="${base_output_dir}_${run_stamp}"
  suffix=0
  while [[ -e "$candidate" ]]; do
    suffix=$((suffix + 1))
    candidate="${base_output_dir}_${run_stamp}_${suffix}"
  done
  printf '%s\n' "$candidate"
}

write_resolved_config() {
  local config_path="$1"
  local run_config_path="$2"
  local output_dir="$3"

  RUN_OUTPUT_DIR="$output_dir" "$PYTHON_BIN" - "$config_path" "$run_config_path" <<'PY'
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
}

launch_dataset() {
  local dataset_input="$1"
  local dataset_name config_rel config_path
  local -a dataset_info config_info cmd parallel_info

  mapfile -t dataset_info < <(resolve_dataset "$dataset_input")
  dataset_name="${dataset_info[0]}"
  config_rel="${dataset_info[1]}"
  config_path="$ROOT_DIR/$config_rel"

  if [[ ! -f "$config_path" ]]; then
    echo "Config file not found: $config_path" >&2
    exit 1
  fi

  mapfile -t config_info < <(read_config_info "$config_path")

  local dataset_dir model_path_rel reload_enabled base_output_dir output_dir
  local pid_file launch_log train_log metrics_log run_config_path
  local parallel_visible_devices parallel_nproc

  dataset_dir="${config_info[0]}"
  model_path_rel="${config_info[1]}"
  reload_enabled="${config_info[2]}"
  base_output_dir="$ROOT_DIR/$model_path_rel"
  output_dir="$(build_output_dir "$base_output_dir" "$reload_enabled")"

  pid_file="$output_dir/train.pid"
  launch_log="$output_dir/launcher.log"
  train_log="$output_dir/train.log"
  metrics_log="$output_dir/metrics.jsonl"
  run_config_path="$output_dir/train_config.resolved.yaml"

  if [[ ! -d "$dataset_dir" ]]; then
    echo "Dataset directory does not exist for $dataset_name: $dataset_dir" >&2
    exit 1
  fi

  mkdir -p "$output_dir"
  write_resolved_config "$config_path" "$run_config_path" "$output_dir"

  mapfile -t parallel_info < <(resolve_parallel_launch "$DEVICE")
  parallel_visible_devices="${parallel_info[0]}"
  parallel_nproc="${parallel_info[1]}"

  cmd=()
  if [[ -n "$parallel_visible_devices" ]]; then
    cmd+=("env" "CUDA_VISIBLE_DEVICES=$parallel_visible_devices")
  fi
  cmd+=(
    "$PYTHON_BIN"
    "-m"
    "torch.distributed.run"
    "--standalone"
    "--nproc_per_node"
    "$parallel_nproc"
    "train_parallel_accel.py"
    "--config"
    "$run_config_path"
  )
  if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    cmd+=("${EXTRA_ARGS[@]}")
  fi

  cd "$ROOT_DIR"

  echo "dataset: $dataset_name"
  echo "config: $config_rel"
  echo "resolved_config: $run_config_path"
  echo "dataset_dir: $dataset_dir"
  echo "base_output_dir: $model_path_rel"
  echo "reload: $reload_enabled"
  echo "run_output_dir: $output_dir"
  if [[ -n "$parallel_visible_devices" ]]; then
    echo "cuda_visible_devices: $parallel_visible_devices"
  else
    echo "cuda_visible_devices: <inherit>"
  fi
  echo "nproc_per_node: $parallel_nproc"
  printf 'command:'
  printf ' %q' "${cmd[@]}"
  printf '\n'

  if [[ "$FOREGROUND" -eq 1 || ${#DATASET_INPUTS[@]} -gt 1 ]]; then
    if [[ ${#DATASET_INPUTS[@]} -gt 1 ]]; then
      echo "execution_mode: sequential"
      echo "launcher_log: $launch_log"
    else
      echo "execution_mode: foreground"
    fi
    : > "$train_log"
    : > "$metrics_log"
    : > "$launch_log"
    "${cmd[@]}" > "$launch_log" 2>&1
    return
  fi

  : > "$train_log"
  : > "$metrics_log"
  : > "$launch_log"
  echo "execution_mode: background"
  nohup "${cmd[@]}" > "$launch_log" 2>&1 < /dev/null &
  local pid=$!
  echo "$pid" > "$pid_file"

  sleep 1
  if ! kill -0 "$pid" > /dev/null 2>&1; then
    rm -f "$pid_file"
    echo "background training exited immediately" >&2
    echo "inspect log: $output_dir/train.log" >&2
    echo "launcher log: $launch_log" >&2
    exit 1
  fi

  echo "started background training"
  echo "pid: $pid"
  echo "train_log: $output_dir/train.log"
  echo "launcher_log: $launch_log"
  echo "pid_file: $pid_file"
}

for dataset_input in "${DATASET_INPUTS[@]}"; do
  launch_dataset "$dataset_input"
done
