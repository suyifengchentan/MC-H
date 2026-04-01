#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE=""
SEED=42
PLAN_ONLY=0
SKIP_COLLECT=0
SKIP_POST=0

usage() {
  cat <<'EOF'
Usage:
  bash experiments/run_all_representative.sh [--device DEVICE] [--seed SEED] [--plan-only] [--skip-collect] [--skip-post]

Options:
  --device DEVICE   Optional device override passed to train.py/eval.py, e.g. cuda:0
  --seed SEED       Single seed used for all suites. Default: 42
  --plan-only       Only generate suite plans/configs, do not execute train/eval
  --skip-collect    Skip per-suite result collection
  --skip-post       Skip final table / figure generation
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --device)
      DEVICE="${2:-}"
      shift 2
      ;;
    --seed)
      SEED="${2:-}"
      shift 2
      ;;
    --plan-only)
      PLAN_ONLY=1
      shift
      ;;
    --skip-collect)
      SKIP_COLLECT=1
      shift
      ;;
    --skip-post)
      SKIP_POST=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

run_suite() {
  local suite_rel="$1"
  local -a cmd
  cmd=("$PYTHON_BIN" "experiments/run_suite.py" "--suite" "$suite_rel" "--seed" "$SEED")
  if [[ "$PLAN_ONLY" -eq 0 ]]; then
    cmd+=("--execute")
    if [[ -n "$DEVICE" ]]; then
      cmd+=("--device" "$DEVICE")
    fi
  fi
  (
    cd "$REPO_ROOT"
    "${cmd[@]}"
  )
}

collect_suite() {
  local suite_rel="$1"
  if [[ "$SKIP_COLLECT" -eq 1 ]]; then
    return
  fi
  (
    cd "$REPO_ROOT"
    "$PYTHON_BIN" "experiments/collect_results.py" "--suite" "$suite_rel"
  )
}

post_process() {
  if [[ "$SKIP_POST" -eq 1 || "$PLAN_ONLY" -eq 1 ]]; then
    return
  fi

  (
    cd "$REPO_ROOT"
    "$PYTHON_BIN" "experiments/generate_core_ablation_table.py"
    "$PYTHON_BIN" "experiments/generate_k_estimation_table.py"
  )

  if [[ -f "$REPO_ROOT/experiments/generated/methodology_validation/summary_grouped.csv" ]]; then
    (
      cd "$REPO_ROOT"
      "$PYTHON_BIN" "experiments/figures/generate_fig3_empirical.py" \
        --summary "experiments/generated/methodology_validation/summary_grouped.csv"
    )
  fi

  local metrics_path="$REPO_ROOT/save/experiments/core_ablation_matrix/fine_cub_full/seed_${SEED}/metrics.jsonl"
  if [[ -f "$metrics_path" ]]; then
    (
      cd "$REPO_ROOT"
      "$PYTHON_BIN" "experiments/figures/generate_fig6_empirical.py" \
        --metrics "save/experiments/core_ablation_matrix/fine_cub_full/seed_${SEED}/metrics.jsonl"
    )
  fi
}

SUITES=(
  "experiments/suites/validation_methodology.yaml"
  "experiments/suites/ablation_methodology.yaml"
  "experiments/suites/oracle_k_fairness.yaml"
  "experiments/suites/core_ablation_matrix.yaml"
  "experiments/suites/k_estimation_benchmark.yaml"
  "experiments/suites/freek_validation.yaml"
  "experiments/suites/freek_ablation.yaml"
)

for suite_rel in "${SUITES[@]}"; do
  echo "[suite] $suite_rel"
  run_suite "$suite_rel"
  collect_suite "$suite_rel"
done

post_process

echo "[done] representative experiment pipeline finished"
