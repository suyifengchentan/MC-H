#!/usr/bin/env bash
set -euo pipefail

python experiments/run_suite.py \
  --suite experiments/suites/core_ablation_matrix.yaml \
  "$@"
