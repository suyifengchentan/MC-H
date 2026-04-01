#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
STATE_DIR_DEFAULT="$REPO_ROOT/save/run_exp"

PYTHON_BIN_INPUT="${PYTHON_BIN:-python}"
if ! PYTHON_BIN="$(command -v "$PYTHON_BIN_INPUT" 2>/dev/null)"; then
  echo "Python executable not found: $PYTHON_BIN_INPUT" >&2
  exit 1
fi

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_exp.sh <action> [options]

Actions:
  start      Start a managed experiments/run_suite.py job in the background
  resume     Resume from last.pt with reload=true and --force-retrain
  stop       Stop one managed job by killing its process group
  stop-all   Stop every managed job recorded by this script
  status     Show one managed job or all managed jobs
  list       Alias of status
  tail       Tail the launcher log of one managed job

Common options:
  --name NAME          Run name. If omitted for start/resume, derived from suite/case/seed
  --state-dir DIR      Metadata/log directory. Default: save/run_exp
  --python-bin BIN     Python executable. Default: $PYTHON_BIN or python on PATH
  --device DEVICE      Device passed to run_suite.py, e.g. cuda:0

Start/resume options:
  --suite PATH         Suite YAML path, e.g. experiments/suites/freek_validation.yaml
  --case ID            Case id filter. Can be repeated
  --seed N             Seed filter. Can be repeated
  --override K=V       Train override. Can be repeated
  --eval-override K=V  Eval override. Can be repeated
  --eval-split SPLIT   One of val, test, both
  --generated-root DIR Forwarded to run_suite.py
  --output-root DIR    Forwarded to run_suite.py
  --skip-train         Forwarded to run_suite.py
  --skip-eval          Forwarded to run_suite.py
  --force-retrain      Forwarded to run_suite.py
  --fail-fast          Forwarded to run_suite.py
  --plan-only          Do not add --execute

Stop/tail options:
  --force              For stop/stop-all: use SIGKILL instead of SIGTERM
  --lines N            For tail: number of lines before follow. Default: 50

Examples:
  bash scripts/run_exp.sh start \
    --suite experiments/suites/validation_methodology.yaml \
    --case medium_image50_full \
    --seed 42 \
    --device cuda:0

  bash scripts/run_exp.sh resume \
    --suite experiments/suites/validation_methodology.yaml \
    --case medium_image50_full \
    --seed 42 \
    --device cuda:0

  bash scripts/run_exp.sh status
  bash scripts/run_exp.sh stop --name validation_methodology.medium_image50_full.seed42
  bash scripts/run_exp.sh tail --name validation_methodology.medium_image50_full.seed42
EOF
}

quote_shell() {
  printf '%q' "$1"
}

sanitize_name() {
  local raw="$1"
  raw="${raw//\//_}"
  raw="${raw// /_}"
  raw="${raw//:/_}"
  raw="${raw//=/__}"
  raw="${raw//,/__}"
  printf '%s\n' "$raw"
}

ensure_state_dir() {
  mkdir -p "$STATE_DIR"
}

meta_path_for_name() {
  local name="$1"
  printf '%s/%s.meta\n' "$STATE_DIR" "$name"
}

load_meta() {
  local name="$1"
  local meta_path
  meta_path="$(meta_path_for_name "$name")"
  if [[ ! -f "$meta_path" ]]; then
    echo "Run metadata not found: $meta_path" >&2
    exit 1
  fi
  # shellcheck disable=SC1090
  source "$meta_path"
}

is_pid_running() {
  local pid="$1"
  [[ -n "$pid" ]] && kill -0 "$pid" >/dev/null 2>&1
}

write_meta() {
  local meta_path="$1"
  local name="$2"
  local pid="$3"
  local pgid="$4"
  local log_path="$5"
  local suite="$6"
  local started_at="$7"
  local command_text="$8"

  {
    printf 'RUN_NAME=%q\n' "$name"
    printf 'RUN_PID=%q\n' "$pid"
    printf 'RUN_PGID=%q\n' "$pgid"
    printf 'RUN_LOG=%q\n' "$log_path"
    printf 'RUN_SUITE=%q\n' "$suite"
    printf 'RUN_STARTED_AT=%q\n' "$started_at"
    printf 'RUN_COMMAND=%q\n' "$command_text"
  } > "$meta_path"
}

build_default_name() {
  local suite="$1"
  local suite_stem case_part seed_part raw

  suite_stem="$(basename "$suite")"
  suite_stem="${suite_stem%.yaml}"

  if [[ ${#CASES[@]} -eq 1 ]]; then
    case_part="${CASES[0]}"
  elif [[ ${#CASES[@]} -gt 1 ]]; then
    case_part="multi_case"
  else
    case_part="all"
  fi

  if [[ ${#SEEDS[@]} -eq 1 ]]; then
    seed_part="seed${SEEDS[0]}"
  elif [[ ${#SEEDS[@]} -gt 1 ]]; then
    seed_part="multi_seed"
  else
    seed_part="default_seed"
  fi

  raw="${suite_stem}.${case_part}.${seed_part}"
  sanitize_name "$raw"
}

print_one_status() {
  local name="$1"
  load_meta "$name"

  local state="stopped"
  if is_pid_running "$RUN_PID"; then
    state="running"
  fi

  printf 'name: %s\n' "$RUN_NAME"
  printf 'state: %s\n' "$state"
  printf 'pid: %s\n' "$RUN_PID"
  printf 'pgid: %s\n' "$RUN_PGID"
  printf 'suite: %s\n' "$RUN_SUITE"
  printf 'started_at: %s\n' "$RUN_STARTED_AT"
  printf 'log: %s\n' "$RUN_LOG"
  printf 'command: %s\n' "$RUN_COMMAND"
}

list_all_statuses() {
  ensure_state_dir
  shopt -s nullglob
  local meta_files=("$STATE_DIR"/*.meta)
  shopt -u nullglob

  if [[ ${#meta_files[@]} -eq 0 ]]; then
    echo "No managed runs recorded in $STATE_DIR"
    return
  fi

  local meta_file name
  for meta_file in "${meta_files[@]}"; do
    name="$(basename "$meta_file" .meta)"
    load_meta "$name"
    if is_pid_running "$RUN_PID"; then
      printf '%-45s running pid=%s pgid=%s\n' "$RUN_NAME" "$RUN_PID" "$RUN_PGID"
    else
      printf '%-45s stopped pid=%s pgid=%s\n' "$RUN_NAME" "$RUN_PID" "$RUN_PGID"
    fi
  done
}

stop_one_run() {
  local name="$1"
  load_meta "$name"

  local target="${RUN_PGID:-$RUN_PID}"
  if [[ -z "$target" ]]; then
    echo "Missing pid/pgid in metadata for $name" >&2
    exit 1
  fi

  if ! is_pid_running "$RUN_PID"; then
    echo "Run is already stopped: $RUN_NAME"
    return
  fi

  if [[ "$FORCE_STOP" -eq 1 ]]; then
    kill -KILL "-$target"
    echo "Sent SIGKILL to process group $target for $RUN_NAME"
  else
    kill -TERM "-$target"
    echo "Sent SIGTERM to process group $target for $RUN_NAME"
  fi
}

start_or_resume_run() {
  local mode="$1"
  ensure_state_dir

  if [[ -z "$SUITE" ]]; then
    echo "--suite is required for $mode" >&2
    usage >&2
    exit 1
  fi

  if [[ -z "$NAME" ]]; then
    NAME="$(build_default_name "$SUITE")"
  else
    NAME="$(sanitize_name "$NAME")"
  fi

  local meta_path log_path
  meta_path="$(meta_path_for_name "$NAME")"
  log_path="$STATE_DIR/$NAME.log"

  if [[ -f "$meta_path" ]]; then
    load_meta "$NAME"
    if is_pid_running "$RUN_PID"; then
      echo "Run is already active: $NAME (pid=$RUN_PID pgid=$RUN_PGID)" >&2
      echo "Stop it first or use another --name." >&2
      exit 1
    fi
  fi

  local -a cmd
  cmd=("$PYTHON_BIN" "experiments/run_suite.py" "--suite" "$SUITE")
  if [[ "$PLAN_ONLY" -eq 0 ]]; then
    cmd+=("--execute")
  fi
  if [[ -n "$DEVICE" ]]; then
    cmd+=("--device" "$DEVICE")
  fi
  if [[ -n "$GENERATED_ROOT" ]]; then
    cmd+=("--generated-root" "$GENERATED_ROOT")
  fi
  if [[ -n "$OUTPUT_ROOT" ]]; then
    cmd+=("--output-root" "$OUTPUT_ROOT")
  fi
  if [[ -n "$EVAL_SPLIT" ]]; then
    cmd+=("--eval-split" "$EVAL_SPLIT")
  fi
  if [[ "$SKIP_TRAIN" -eq 1 ]]; then
    cmd+=("--skip-train")
  fi
  if [[ "$SKIP_EVAL" -eq 1 ]]; then
    cmd+=("--skip-eval")
  fi
  if [[ "$FAIL_FAST" -eq 1 ]]; then
    cmd+=("--fail-fast")
  fi
  if [[ "$FORCE_RETRAIN" -eq 1 ]]; then
    cmd+=("--force-retrain")
  fi
  if [[ "$mode" == "resume" ]]; then
    cmd+=("--override" "reload=true" "--force-retrain")
  fi

  local case_id seed override eval_override
  for case_id in "${CASES[@]}"; do
    cmd+=("--case" "$case_id")
  done
  for seed in "${SEEDS[@]}"; do
    cmd+=("--seed" "$seed")
  done
  for override in "${TRAIN_OVERRIDES[@]}"; do
    cmd+=("--override" "$override")
  done
  for eval_override in "${EVAL_OVERRIDES[@]}"; do
    cmd+=("--eval-override" "$eval_override")
  done
  if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    cmd+=("${EXTRA_ARGS[@]}")
  fi

  local command_text started_at pid pgid
  command_text="$(printf '%q ' "${cmd[@]}")"
  command_text="${command_text% }"
  started_at="$(date -Iseconds)"

  : > "$log_path"
  (
    cd "$REPO_ROOT"
    setsid nohup "${cmd[@]}" > "$log_path" 2>&1 < /dev/null &
    echo "$!"
  ) | {
    read -r pid
    if [[ -z "$pid" ]]; then
      echo "Failed to capture launcher pid" >&2
      exit 1
    fi

    sleep 1
    if ! is_pid_running "$pid"; then
      echo "Managed run exited immediately. Inspect: $log_path" >&2
      exit 1
    fi

    pgid="$(ps -o pgid= -p "$pid" | tr -d ' ')"
    if [[ -z "$pgid" ]]; then
      pgid="$pid"
    fi

    write_meta "$meta_path" "$NAME" "$pid" "$pgid" "$log_path" "$SUITE" "$started_at" "$command_text"

    echo "started: $NAME"
    echo "pid: $pid"
    echo "pgid: $pgid"
    echo "log: $log_path"
    echo "meta: $meta_path"
  }
}

ACTION="${1:-}"
if [[ -z "$ACTION" ]]; then
  usage >&2
  exit 1
fi
if [[ "$ACTION" == "-h" || "$ACTION" == "--help" ]]; then
  usage
  exit 0
fi
shift || true

NAME=""
SUITE=""
DEVICE=""
STATE_DIR="$STATE_DIR_DEFAULT"
GENERATED_ROOT=""
OUTPUT_ROOT=""
EVAL_SPLIT=""
PLAN_ONLY=0
SKIP_TRAIN=0
SKIP_EVAL=0
FAIL_FAST=0
FORCE_RETRAIN=0
FORCE_STOP=0
TAIL_LINES=50
CASES=()
SEEDS=()
TRAIN_OVERRIDES=()
EVAL_OVERRIDES=()
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name)
      NAME="${2:-}"
      shift 2
      ;;
    --suite)
      SUITE="${2:-}"
      shift 2
      ;;
    --case)
      CASES+=("${2:-}")
      shift 2
      ;;
    --seed)
      SEEDS+=("${2:-}")
      shift 2
      ;;
    --device)
      DEVICE="${2:-}"
      shift 2
      ;;
    --state-dir)
      STATE_DIR="${2:-}"
      shift 2
      ;;
    --python-bin)
      PYTHON_BIN_INPUT="${2:-}"
      if ! PYTHON_BIN="$(command -v "$PYTHON_BIN_INPUT" 2>/dev/null)"; then
        echo "Python executable not found: $PYTHON_BIN_INPUT" >&2
        exit 1
      fi
      shift 2
      ;;
    --override)
      TRAIN_OVERRIDES+=("${2:-}")
      shift 2
      ;;
    --eval-override)
      EVAL_OVERRIDES+=("${2:-}")
      shift 2
      ;;
    --eval-split)
      EVAL_SPLIT="${2:-}"
      shift 2
      ;;
    --generated-root)
      GENERATED_ROOT="${2:-}"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="${2:-}"
      shift 2
      ;;
    --skip-train)
      SKIP_TRAIN=1
      shift
      ;;
    --skip-eval)
      SKIP_EVAL=1
      shift
      ;;
    --fail-fast)
      FAIL_FAST=1
      shift
      ;;
    --force-retrain)
      FORCE_RETRAIN=1
      shift
      ;;
    --plan-only)
      PLAN_ONLY=1
      shift
      ;;
    --force)
      FORCE_STOP=1
      shift
      ;;
    --lines)
      TAIL_LINES="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

case "$ACTION" in
  start)
    start_or_resume_run "start"
    ;;
  resume)
    start_or_resume_run "resume"
    ;;
  stop)
    if [[ -z "$NAME" ]]; then
      echo "--name is required for stop" >&2
      exit 1
    fi
    NAME="$(sanitize_name "$NAME")"
    stop_one_run "$NAME"
    ;;
  stop-all)
    ensure_state_dir
    shopt -s nullglob
    meta_files=("$STATE_DIR"/*.meta)
    shopt -u nullglob
    if [[ ${#meta_files[@]} -eq 0 ]]; then
      echo "No managed runs recorded in $STATE_DIR"
      exit 0
    fi
    for meta_file in "${meta_files[@]}"; do
      stop_one_run "$(basename "$meta_file" .meta)"
    done
    ;;
  status|list)
    if [[ -n "$NAME" ]]; then
      NAME="$(sanitize_name "$NAME")"
      print_one_status "$NAME"
    else
      list_all_statuses
    fi
    ;;
  tail)
    if [[ -z "$NAME" ]]; then
      echo "--name is required for tail" >&2
      exit 1
    fi
    NAME="$(sanitize_name "$NAME")"
    load_meta "$NAME"
    echo "tailing log: $RUN_LOG"
    exec tail -n "$TAIL_LINES" -f "$RUN_LOG"
    ;;
  *)
    echo "Unknown action: $ACTION" >&2
    usage >&2
    exit 1
    ;;
esac
