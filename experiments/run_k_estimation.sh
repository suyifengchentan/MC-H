#!/usr/bin/env bash
set -euo pipefail

python experiments/run_suite.py \
  --suite experiments/suites/k_estimation_benchmark.yaml \
  "$@"
