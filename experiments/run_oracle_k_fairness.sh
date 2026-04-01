#!/usr/bin/env bash
set -euo pipefail

python experiments/run_suite.py \
  --suite experiments/suites/oracle_k_fairness.yaml \
  "$@"
